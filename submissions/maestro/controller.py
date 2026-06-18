#!/usr/bin/env python3
"""Closed-loop melody player for the dexterous-hand / 7-key piano scene (v2).

The melody comes entirely from config/song.json (via src/score_parser);
sensor reads come from src/feedback. The controller is a closed-loop state
machine. For each beat it:

  1. SLIDEs the whole hand sideways on a prismatic wrist joint so the right
     fingers line up over the beat's key(s) -- the hand reaches across all
     7 keys.
  2. PRESSes the assigned finger(s) -- one for a melody note, two or three
     (different fingers) for a chord -- by smoothly interpolating them down,
     the press SPEED scaled by the note's velocity (louder = faster/harder).
  3. Advances only once EVERY assigned key's touch sensor CONFIRMS the press
     (force > threshold); a timeout logs a warning and continues if a press
     is somehow missed.
  4. HOLDs for the note's tempo-scaled duration, then RELEASEs.

Each confirmed press also emits an audio note event; in --record mode those
events are synthesized into the actual tune and muxed onto the MP4 (src/audio).

Usage:
    python controller.py            # play live in the interactive viewer
    python controller.py --record   # render headless to media/ode_to_joy.mp4 (with audio)
    python controller.py --record out.gif   # GIF instead (no audio track)
    python controller.py --loops 2  # repeat the whole piece N times
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

# make src/ importable regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import score_parser                 # noqa: E402
import audio                        # noqa: E402
from feedback import TouchFeedback  # noqa: E402

# ----------------------------------------------------------------------
# TUNABLES
# ----------------------------------------------------------------------
SEED = 0                       # fixed seed -> deterministic run

PRESS_THRESHOLD = 1.0          # N; sensor force that CONFIRMS a press
RELEASE_THRESHOLD = 0.1        # N; below this the key counts as released
PRESS_TIMEOUT = 1.5            # s; if no confirm by now -> warn & continue
RELEASE_RAMP = 0.28            # s; time to interpolate press -> rest
RELEASE_GAP = 0.06             # s; extra settle after release before next beat
SETTLE_SECONDS = 0.5           # let things settle before the first beat

# Velocity-sensitive press: a louder note (velocity -> 1) is struck with a
# SHORTER ramp (faster, harder); a soft note uses a longer, gentler ramp.
PRESS_RAMP_SLOW = 0.46         # s; ramp for the softest note (velocity -> 0)
PRESS_RAMP_FAST = 0.16         # s; ramp for the loudest note (velocity -> 1)

# Finger poses (radians per joint: proximal, middle, distal). Proximal flexes
# most so the fingertip reaches DOWN onto the key. PRESS_DEPTH was tuned to the
# peak-force sweet spot (deeper over-curls the tip and lifts it off the key).
REST_POSE = np.array([0.0, 0.0, 0.0])
PRESS_DEPTH = -1.15
PRESS_POSE = np.array([PRESS_DEPTH, -0.40, -0.20])

# Wrist slide motion: ramp duration scales with distance for ~constant speed.
SLIDE_SPEED = 0.28             # m/s nominal slide speed
SLIDE_RAMP_MIN = 0.12          # s
SLIDE_RAMP_MAX = 1.10          # s
SLIDE_SETTLE = 0.14            # s; settle after the slide ramp before pressing
SLIDE_TOL = 0.006              # m; "close enough" to the slide target

REALTIME = True                # pace the sim to wall-clock so it's watchable

# Recording (headless --record mode). 1280x720 is clean HD and divisible by 16.
RECORD_FPS = 30
RECORD_W = 1280
RECORD_H = 720
GIF_MAX_W = 640                # GIF output is downscaled to this width
DEFAULT_RECORD_PATH = os.path.join(HERE, "media", "ode_to_joy.mp4")
RECORD_SAFETY_SECONDS = 120.0  # hard stop so a stuck run can't render forever

# Note name -> key index (0..6), low -> high. Matches the keys in scene.xml
# (key0=C..key6=B) and the pitches in src/audio.NOTE_FREQ.
NOTE_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
NUM_FINGERS = 5                # fingers f1..f5, spaced one-per-key

SONG_PATH = os.path.join(HERE, "config", "song.json")
# ----------------------------------------------------------------------


def smoothstep(x):
    """Eased 0->1 ramp (zero velocity at both ends)."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class Ramp:
    """Smoothly interpolates one or more actuator targets over time."""

    def __init__(self, act_ids, pose):
        self.act_ids = act_ids
        self.start = np.array(pose, dtype=float)
        self.target = np.array(pose, dtype=float)
        self.t0 = 0.0
        self.duration = 1e-9

    def move_to(self, pose, now, duration):
        # start the new ramp from wherever we currently are (continuous motion)
        self.start = self.current(now)
        self.target = np.array(pose, dtype=float)
        self.t0 = now
        self.duration = max(duration, 1e-9)

    def current(self, now):
        alpha = smoothstep((now - self.t0) / self.duration)
        return self.start + (self.target - self.start) * alpha

    def apply(self, data, now):
        for aid, val in zip(self.act_ids, self.current(now)):
            data.ctrl[aid] = val


class MelodyPlayer:
    """Closed-loop state machine that plays one song on the 7-key keyboard.

    `tick()` advances the controller by exactly one physics step and returns
    False once the melody is finished. The interactive viewer and the headless
    recorder drive it identically, so the control logic is shared.

    States per beat: SETTLE -> SLIDE -> PRESS -> HOLD -> RELEASE.
    """

    def __init__(self, model, data, song, loops=1):
        self.model = model
        self.data = data
        self.song = song
        self.loops = max(1, loops)
        self.loop_idx = 1
        self.feedback = TouchFeedback(model)
        self.audio_events = []        # filled on each confirmed press

        def act_id(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # one smooth ramp per finger (f1..f5), plus the wrist-slide ramp
        self.finger_ramps = {
            f: Ramp([act_id(f"f{f}_j{j}_act") for j in (1, 2, 3)], REST_POSE)
            for f in range(1, NUM_FINGERS + 1)
        }
        self.wrist_ramp = Ramp([act_id("wrist_act")], [0.0])
        self.wrist_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                           "wrist_slide")
        self.wrist_qadr = int(model.jnt_qposadr[self.wrist_jnt])

        # Self-calibrate geometry: with the slide at 0, read each fingertip's
        # world x and each key's touch-site world x. Sliding by delta shifts a
        # fingertip's x by exactly delta, so the slide that puts finger f over
        # key k is (key_x[k] - tip_x0[f]).
        self.tip_x0 = {f: self._site_x(f"f{f}_tip") for f in range(1, NUM_FINGERS + 1)}
        self.key_x = {k: self._site_x(f"key{k}_touch") for k in range(7)}
        lo, hi = model.jnt_range[self.wrist_jnt]
        self.slide_lo, self.slide_hi = float(lo), float(hi)

        self.cur_slide = 0.0
        self.state = "SETTLE"
        self.state_t0 = 0.0
        self.beat_idx = 0
        self.hold_until = 0.0
        self.assignment = {}          # key_index -> finger number (this beat)
        self.slide_target = 0.0
        self.slide_ramp_dur = SLIDE_RAMP_MIN
        self.done = False

    # ---- geometry helpers ------------------------------------------------
    def _site_x(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return float(self.data.site_xpos[sid][0])

    def _beat(self):
        return self.song.notes[self.beat_idx]

    def _key_indices(self, beat):
        return sorted(NOTE_INDEX[name] for name in beat.keys)

    def _plan(self, key_indices, from_slide):
        """Choose a finger assignment + slide target for this beat.

        Fingers are spaced one-per-key, so a chord whose keys are sorted
        ascending maps to fingers preserving the key gaps: finger(k_i) =
        base + (k_i - k0). We pick the base (and thus the slide) that is
        VALID (slide in range, fingers 1..NUM_FINGERS) and needs the LEAST
        wrist travel from the current position -> smooth, minimal motion.
        Returns (assignment dict, slide_target) or (None, None) if infeasible.
        """
        k0 = key_indices[0]
        span = key_indices[-1] - k0
        best = None
        for base in range(1, NUM_FINGERS + 1 - span):
            # slide so finger `base` sits over key k0 (consistent for all keys)
            slide = self.key_x[k0] - self.tip_x0[base]
            if not (self.slide_lo - 1e-6 <= slide <= self.slide_hi + 1e-6):
                continue
            cost = abs(slide - from_slide)
            if best is None or cost < best[0]:
                assignment = {k: base + (k - k0) for k in key_indices}
                best = (cost, slide, assignment)
        if best is None:
            return None, None
        return best[2], float(best[1])

    # ---- state transitions ----------------------------------------------
    def _begin_beat(self, now):
        beat = self._beat()
        keys = self._key_indices(beat)
        assignment, slide = self._plan(keys, self.cur_slide)
        if assignment is None:
            # unreachable voicing (shouldn't happen for valid songs) -> skip
            print(f"[{now:5.2f}s] WARNING: cannot voice beat "
                  f"{self.beat_idx + 1} {beat.keys}; skipping", flush=True)
            self._advance_or_finish(now, after_skip=True)
            return
        self.assignment = assignment
        self.slide_target = slide
        dist = abs(slide - self.cur_slide)
        self.slide_ramp_dur = float(np.clip(dist / SLIDE_SPEED,
                                             SLIDE_RAMP_MIN, SLIDE_RAMP_MAX))
        # all fingers up while sliding; wrist ramps to the new target
        for f in self.finger_ramps.values():
            f.move_to(REST_POSE, now, RELEASE_RAMP)
        self.wrist_ramp.move_to([slide], now, self.slide_ramp_dur)
        kind = "chord" if beat.is_chord else "note "
        names = "+".join(beat.keys)
        fingers = ",".join(f"f{assignment[k]}" for k in keys)
        print(f"[{now:5.2f}s] beat {self.beat_idx + 1}/{len(self.song.notes)}: "
              f"{kind} {names:7} -> slide to {slide:+.3f} m, fingers {fingers}",
              flush=True)
        self.state, self.state_t0 = "SLIDE", now

    def _begin_press(self, now):
        beat = self._beat()
        ramp_dur = PRESS_RAMP_SLOW + (PRESS_RAMP_FAST - PRESS_RAMP_SLOW) * beat.velocity
        for f in self.assignment.values():
            self.finger_ramps[f].move_to(PRESS_POSE, now, ramp_dur)
        self.state, self.state_t0 = "PRESS", now

    def _sensors(self):
        return [f"key{k}_sensor" for k in self.assignment]

    def _all_pressed(self):
        return all(self.feedback.is_pressed(self.data, s, PRESS_THRESHOLD)
                   for s in self._sensors())

    def _all_released(self):
        return all(self.feedback.is_released(self.data, s, RELEASE_THRESHOLD)
                   for s in self._sensors())

    def _record_audio(self, now):
        beat = self._beat()
        hold_s = beat.seconds(self.song.tempo_bpm)
        for name in beat.keys:
            self.audio_events.append(
                {"t": now, "note": name, "dur": hold_s + RELEASE_RAMP,
                 "vel": beat.velocity})

    def _advance_or_finish(self, now, after_skip=False):
        self.beat_idx += 1
        if self.beat_idx < len(self.song.notes):
            self._begin_beat(now)
            return
        if self.loop_idx >= self.loops:
            print(f'\n[{now:5.2f}s] performance complete: "{self.song.title}" '
                  f'played {self.loops}x ({len(self.audio_events)} notes).',
                  flush=True)
            self.done = True
        else:
            self.loop_idx += 1
            self.beat_idx = 0
            print(f"\n[{now:5.2f}s] --- loop {self.loop_idx}/{self.loops} ---",
                  flush=True)
            self._begin_beat(now)

    # ---- main step -------------------------------------------------------
    def tick(self):
        """Run one control step + one physics step. Returns False when done."""
        data = self.data
        now = data.time
        elapsed = now - self.state_t0
        self.cur_slide = float(data.qpos[self.wrist_qadr])

        if self.state == "SETTLE":
            if elapsed >= SETTLE_SECONDS:
                self._begin_beat(now)

        elif self.state == "SLIDE":
            at_target = abs(self.cur_slide - self.slide_target) <= SLIDE_TOL
            if elapsed >= self.slide_ramp_dur + SLIDE_SETTLE and at_target:
                self._begin_press(now)

        elif self.state == "PRESS":
            if self._all_pressed():
                beat = self._beat()
                hold_s = beat.seconds(self.song.tempo_bpm)
                forces = ", ".join(
                    f"{self.feedback.force(data, s):.1f}N" for s in self._sensors())
                print(f"[{now:5.2f}s]   CONFIRMED [{forces}] "
                      f"vel={beat.velocity:.2f} hold {hold_s:.2f}s", flush=True)
                self._record_audio(now)
                self.hold_until = now + hold_s
                self.state, self.state_t0 = "HOLD", now
            elif elapsed >= PRESS_TIMEOUT:
                print(f"[{now:5.2f}s]   WARNING: press not confirmed within "
                      f"{PRESS_TIMEOUT:.1f}s; skipping beat", flush=True)
                self.hold_until = now
                self.state, self.state_t0 = "HOLD", now

        elif self.state == "HOLD":
            if now >= self.hold_until:
                for f in self.assignment.values():
                    self.finger_ramps[f].move_to(REST_POSE, now, RELEASE_RAMP)
                self.state, self.state_t0 = "RELEASE", now

        elif self.state == "RELEASE":
            if elapsed >= RELEASE_RAMP + RELEASE_GAP and self._all_released():
                self._advance_or_finish(now)

        # advance every smooth interpolation, then step physics
        for f in self.finger_ramps.values():
            f.apply(data, now)
        self.wrist_ramp.apply(data, now)
        mujoco.mj_step(self.model, data)
        return not self.done


def _build():
    """Load song + model + data; reset to a deterministic initial state."""
    np.random.seed(SEED)
    song = score_parser.load_song(SONG_PATH)
    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)   # compute kinematics so the first frame renders
    return song, model, data


def _overview_cam(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")


def run_interactive(loops=1):
    song, model, data = _build()
    player = MelodyPlayer(model, data, song, loops=loops)
    dt = model.opt.timestep

    print(f'Closed-loop play of "{song.title}"  @ {song.tempo_bpm} bpm  '
          f'({len(song.notes)} beats x{loops})', flush=True)
    print(f"Hand slides on a prismatic wrist to reach 7 keys; advancing only "
          f"on sensor confirm (> {PRESS_THRESHOLD} N).\n", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = _overview_cam(model)
        while viewer.is_running():
            step_start = time.perf_counter()
            cont = player.tick()
            viewer.sync()
            if not cont:
                break
            if REALTIME:
                lag = dt - (time.perf_counter() - step_start)
                if lag > 0:
                    time.sleep(lag)


def _write_gif(frames, path, fps):
    """Write buffered frames to an animated GIF via PIL, downscaled to GIF_MAX_W."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    if imgs and imgs[0].width > GIF_MAX_W:
        w = GIF_MAX_W
        h = round(imgs[0].height * w / imgs[0].width)
        imgs = [im.resize((w, h), Image.LANCZOS) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000.0 / fps)), loop=0, optimize=True)
    return path


def run_record(path, loops=1):
    song, model, data = _build()
    player = MelodyPlayer(model, data, song, loops=loops)
    cam = _overview_cam(model)
    ext = os.path.splitext(path)[1].lower()
    want_mp4 = ext == ".mp4"

    print(f'Recording "{song.title}" x{loops} headless @ {RECORD_FPS} fps '
          f'({RECORD_W}x{RECORD_H})...', flush=True)

    # MP4 streams each frame straight to a (silent) temp video via ffmpeg
    # (constant memory); GIF buffers frames and writes once via PIL.
    writer = None
    frames = None
    silent_path = None
    if want_mp4:
        try:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            silent_path = os.path.splitext(path)[0] + ".silent.mp4"
            writer = imageio.get_writer(silent_path, fps=RECORD_FPS,
                                        macro_block_size=16)
        except Exception as e:
            path = os.path.splitext(path)[0] + ".gif"
            want_mp4 = False
            print(f"  (mp4 encoder unavailable: {e}; falling back to {path})",
                  flush=True)
    if writer is None:
        frames = []

    renderer = mujoco.Renderer(model, height=RECORD_H, width=RECORD_W)
    frame_interval = 1.0 / RECORD_FPS
    next_capture = 0.0
    n_frames = 0
    safety = max(RECORD_SAFETY_SECONDS, loops * 90.0 + 30.0)
    try:
        while True:
            if data.time >= next_capture:
                renderer.update_scene(data, camera=cam)
                img = renderer.render()
                if writer is not None:
                    writer.append_data(img)
                else:
                    frames.append(img.copy())
                n_frames += 1
                next_capture += frame_interval
            if not player.tick():
                break
            if data.time > safety:
                print("  (safety stop hit)", flush=True)
                break
    finally:
        renderer.close()
        if writer is not None:
            writer.close()

    duration_s = n_frames / RECORD_FPS

    if want_mp4:
        # synthesize the tune from the confirmed-press events and mux it on
        wav_path = os.path.splitext(path)[0] + ".wav"
        track = audio.synthesize(player.audio_events, duration_s)
        audio.write_wav(track, wav_path)
        try:
            audio.mux(silent_path, wav_path, path)
            os.remove(silent_path)
            os.remove(wav_path)
            out = path
            audio_note = " (with synthesized audio)"
        except Exception as e:
            os.replace(silent_path, path)   # keep the silent video at least
            out = path
            audio_note = f" (audio mux failed: {e}; video is silent)"
    else:
        out = _write_gif(frames, path, RECORD_FPS)
        audio_note = " (GIF: no audio track)"

    size_kb = os.path.getsize(out) / 1024.0
    print(f"Saved {n_frames} frames ({duration_s:.1f}s) -> {out} "
          f"({size_kb:.0f} KB){audio_note}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Play the piano melody (v2).")
    parser.add_argument(
        "--record", nargs="?", const=DEFAULT_RECORD_PATH, default=None,
        metavar="PATH",
        help="render headless to a video file instead of opening the viewer "
             f"(default: {os.path.relpath(DEFAULT_RECORD_PATH, HERE)}; "
             ".mp4 gets synthesized audio, .gif is silent)")
    parser.add_argument(
        "--loops", type=int, default=1, metavar="N",
        help="repeat the whole piece N times (default 1)")
    args = parser.parse_args()
    loops = max(1, args.loops)

    if args.record is not None:
        run_record(args.record, loops)
    else:
        run_interactive(loops)


if __name__ == "__main__":
    main()

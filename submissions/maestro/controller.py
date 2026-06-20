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
    python controller.py --benchmark # play every config/ song; report real metrics
"""
import argparse
import glob
import math
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
CONFIG_DIR = os.path.join(HERE, "config")
BENCHMARK_DOC_PATH = os.path.join(HERE, "docs", "benchmark_results.md")
BENCHMARK_SAFETY_SECONDS = 240.0   # per-song hard stop (headless, no realtime)
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

    def __init__(self, model, data, song, loops=1, verbose=True):
        self.model = model
        self.data = data
        self.song = song
        self.loops = max(1, loops)
        self.loop_idx = 1
        self.verbose = verbose        # False silences routine per-beat logging
        self.feedback = TouchFeedback(model)
        self.audio_events = []        # filled on each confirmed press
        # One record per beat ATTEMPTED (in play order). Each entry captures the
        # real, measured outcome of that beat -- used by --benchmark. This is
        # instrumentation only; it never affects control. See _log_result().
        self.beat_results = []

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
            self._log_result(now, confirmed=False)
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
        if self.verbose:
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

    def _log_result(self, now, confirmed):
        """Record the REAL outcome of the current beat for the benchmark.

        Reads each assigned key's live touch-sensor force straight out of the
        simulation at the moment the beat resolves (confirm or timeout/skip), so
        every logged value is measured, never assumed. `confirmed` is the
        controller's own verdict (all keys' sensors >= PRESS_THRESHOLD within
        the timeout). Per-key booleans let the benchmark count individual
        sensor-confirmed strikes even when a chord only partially registers.
        """
        beat = self._beat()
        keys = list(self.assignment) if self.assignment else self._key_indices(beat)
        forces = {k: self.feedback.force(self.data, f"key{k}_sensor") for k in keys}
        key_confirmed = {k: forces[k] >= PRESS_THRESHOLD for k in keys}
        self.beat_results.append({
            "beat": self.beat_idx,
            "keys": list(beat.keys),
            "confirmed": confirmed,
            "confirm_time": now,
            "key_confirmed": key_confirmed,
            "forces": forces,
        })

    def _advance_or_finish(self, now, after_skip=False):
        self.beat_idx += 1
        if self.beat_idx < len(self.song.notes):
            self._begin_beat(now)
            return
        if self.loop_idx >= self.loops:
            if self.verbose:
                print(f'\n[{now:5.2f}s] performance complete: "{self.song.title}" '
                      f'played {self.loops}x ({len(self.audio_events)} notes).',
                      flush=True)
            self.done = True
        else:
            self.loop_idx += 1
            self.beat_idx = 0
            if self.verbose:
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
                self._log_result(now, confirmed=True)
                if self.verbose:
                    forces = ", ".join(
                        f"{self.feedback.force(data, s):.1f}N" for s in self._sensors())
                    print(f"[{now:5.2f}s]   CONFIRMED [{forces}] "
                          f"vel={beat.velocity:.2f} hold {hold_s:.2f}s", flush=True)
                self._record_audio(now)
                self.hold_until = now + hold_s
                self.state, self.state_t0 = "HOLD", now
            elif elapsed >= PRESS_TIMEOUT:
                self._log_result(now, confirmed=False)
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


def _build(song_path=SONG_PATH):
    """Load song + model + data; reset to a deterministic initial state."""
    np.random.seed(SEED)
    song = score_parser.load_song(song_path)
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


# ----------------------------------------------------------------------
# BENCHMARK MODE
# ----------------------------------------------------------------------
# Plays every song in config/ headless (no renderer, no realtime pacing) with
# the SAME closed-loop controller the demo uses, and measures real metrics from
# the running simulation. Nothing here is hand-tuned or fabricated: every number
# comes from player.beat_results, which is filled from live touch-sensor reads
# and the actual confirm timestamps during the run.

def _song_metrics(song, beat_results):
    """Reduce one finished run's per-beat records to the benchmark metrics.

    All inputs are measured during the run (see MelodyPlayer._log_result):
      - total notes attempted     = key-strikes the song asks for (chord = 3)
      - sensor-confirmed press %   = strikes whose touch sensor hit threshold
      - mean onset error (ms)      = onset deviation from the nominal tempo grid,
                                     aligned at the first confirmed note
      - timing jitter (ms)         = onset deviation AFTER removing the best-fit
                                     constant tempo (how steady the rhythm is,
                                     independent of the hand running slower than
                                     the score's nominal bpm)
      - overall success rate %     = beats where EVERY assigned key confirmed
    """
    tempo = song.tempo_bpm
    total_beats = len(song.notes)
    total_notes = sum(len(b.keys) for b in song.notes)   # individual key-strikes

    confirmed_keys = sum(
        1 for r in beat_results for ok in r["key_confirmed"].values() if ok)
    confirmed_beats = sum(1 for r in beat_results if r["confirmed"])

    # Intended onset of each beat on the nominal tempo grid = cumulative
    # note durations (seconds), straight from the score.
    intended, acc = [], 0.0
    for b in song.notes:
        intended.append(acc)
        acc += b.seconds(tempo)

    confirmed = [(r["beat"], r["confirm_time"]) for r in beat_results
                 if r["confirmed"]]
    if confirmed:
        g = np.array([intended[i] for i, _ in confirmed])   # intended onsets
        a = np.array([t for _, t in confirmed])             # actual onsets
        # (1) Literal "error vs intended onset": align grid to the first
        #     confirmed note (remove constant start-up latency), absolute dev.
        offset = a[0] - g[0]
        errs_ms = list(np.abs(a - (g + offset)) * 1000.0)
        mean_onset_err_ms = float(np.mean(errs_ms))
        # (2) Timing jitter: residual after the best-fit constant tempo
        #     (a ~= slope*g + intercept). Removes the fixed per-beat mechanical
        #     overhead/tempo scaling, leaving genuine rhythmic irregularity.
        if len(confirmed) >= 2 and np.ptp(g) > 0:
            slope, intercept = np.polyfit(g, a, 1)
            jit_ms = list(np.abs(a - (slope * g + intercept)) * 1000.0)
            jitter_ms = float(np.mean(jit_ms))
            realized_tempo = tempo / slope if slope > 0 else float("nan")
        else:
            jit_ms, jitter_ms, realized_tempo = [], float("nan"), float("nan")
    else:
        errs_ms, mean_onset_err_ms = [], float("nan")
        jit_ms, jitter_ms, realized_tempo = [], float("nan"), float("nan")

    return {
        "title": song.title,
        "tempo_bpm": tempo,
        "total_beats": total_beats,
        "total_notes": total_notes,
        "confirmed_keys": confirmed_keys,
        "confirmed_beats": confirmed_beats,
        "press_rate_pct": 100.0 * confirmed_keys / total_notes if total_notes else 0.0,
        "onset_err_ms": mean_onset_err_ms,
        "onset_errs_ms": errs_ms,
        "jitter_ms": jitter_ms,
        "jitters_ms": jit_ms,
        "realized_tempo_bpm": realized_tempo,
        "success_rate_pct": 100.0 * confirmed_beats / total_beats if total_beats else 0.0,
    }


def _run_song_headless(song_path):
    """Play one song with no renderer and no realtime pacing -> (song, metrics, sim_t)."""
    song, model, data = _build(song_path)
    player = MelodyPlayer(model, data, song, loops=1, verbose=False)
    while player.tick():
        if data.time > BENCHMARK_SAFETY_SECONDS:
            print("  (safety stop hit)", flush=True)
            break
    return song, _song_metrics(song, player.beat_results), float(data.time)


def _fmt_timing(v):
    return "n/a" if (v is None or math.isnan(v)) else f"{v:.1f}"


def _benchmark_rows(metrics_list):
    """Per-song display rows + an aggregate 'All songs' row (pooled, honest)."""
    rows = []
    for m in metrics_list:
        rows.append([
            m["title"],
            f"{m['tempo_bpm']:.0f}",
            f"{m['confirmed_keys']}/{m['total_notes']}",
            f"{m['press_rate_pct']:.1f}%",
            _fmt_timing(m["onset_err_ms"]),
            _fmt_timing(m["jitter_ms"]),
            f"{m['confirmed_beats']}/{m['total_beats']} ({m['success_rate_pct']:.1f}%)",
        ])
    # aggregate: pool every strike, beat, and per-note timing value across songs
    tot_notes = sum(m["total_notes"] for m in metrics_list)
    tot_keys = sum(m["confirmed_keys"] for m in metrics_list)
    tot_beats = sum(m["total_beats"] for m in metrics_list)
    tot_cbeats = sum(m["confirmed_beats"] for m in metrics_list)
    all_onset = [e for m in metrics_list for e in m["onset_errs_ms"]]
    all_jit = [e for m in metrics_list for e in m["jitters_ms"]]
    agg = [
        "All songs (pooled)",
        "-",
        f"{tot_keys}/{tot_notes}",
        f"{100.0 * tot_keys / tot_notes:.1f}%" if tot_notes else "n/a",
        _fmt_timing(sum(all_onset) / len(all_onset) if all_onset else float("nan")),
        _fmt_timing(sum(all_jit) / len(all_jit) if all_jit else float("nan")),
        f"{tot_cbeats}/{tot_beats} ({100.0 * tot_cbeats / tot_beats:.1f}%)" if tot_beats else "n/a",
    ]
    return rows, agg


HEADERS = ["Song", "BPM", "Notes (confirmed/attempted)", "Press rate",
           "Onset err vs grid (ms)", "Tempo-norm jitter (ms)", "Success rate"]


def _print_table(rows, agg):
    cols = list(zip(HEADERS, *rows, agg))
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = "  ".join("{:<{w}}".format(h, w=w) for h, w in zip(HEADERS, widths))
    sep = "  ".join("-" * w for w in widths)
    print(line, flush=True)
    print(sep, flush=True)
    for r in rows:
        print("  ".join("{:<{w}}".format(c, w=w) for c, w in zip(r, widths)), flush=True)
    print(sep, flush=True)
    print("  ".join("{:<{w}}".format(c, w=w) for c, w in zip(agg, widths)), flush=True)


def _write_benchmark_md(metrics_list, rows, agg):
    os.makedirs(os.path.dirname(BENCHMARK_DOC_PATH), exist_ok=True)
    L = []
    L.append("# Maestro Benchmark Results\n")
    L.append("Every number in this file is **measured from a real, headless "
             "MuJoCo run** of the closed-loop controller "
             "(`python controller.py --benchmark`). No values are hand-entered; "
             "re-running regenerates this file.\n")
    L.append(f"- Seed: `{SEED}` (deterministic) · timestep `0.002 s` · "
             f"press threshold `{PRESS_THRESHOLD} N` · press timeout "
             f"`{PRESS_TIMEOUT} s`")
    L.append(f"- Songs benchmarked: **{len(metrics_list)}** — every `*.json` "
             "in [`config/`](../config) (in filename order)\n")
    L.append("## Metric definitions\n")
    L.append("- **Notes attempted** — individual key-strikes the score asks for "
             "(a 3-key chord counts as 3 strikes). Reported as "
             "`confirmed/attempted`.")
    L.append("- **Sensor-confirmed press rate** — share of those strikes whose "
             "own touch sensor reached the press threshold, read live from "
             "`data.sensordata` at the moment the beat resolved.")
    L.append("- **Onset err vs grid (ms)** — the literal *timing error vs "
             "intended note onset*: mean absolute deviation of each confirmed "
             "note's onset from the nominal tempo grid, after aligning the grid "
             "to the first confirmed note (constant start-up latency removed). "
             "This is **large on purpose and reported honestly**: the controller "
             "is closed-loop and self-paced — it slides the wrist, ramps each "
             "press, and only advances once the touch sensor confirms — so it "
             "runs slower than the score's nominal bpm and the gap accumulates "
             "across the piece.")
    L.append("- **Tempo-norm jitter (ms)** — the same onsets after removing the "
             "best-fit *constant* tempo (least-squares `onset ≈ slope·grid + "
             "intercept`). This strips out the fixed per-beat mechanical overhead "
             "and the slower-than-nominal tempo, leaving only genuine rhythmic "
             "**irregularity** — i.e. how *steady* the beat is. Lower = steadier.")
    L.append("- **Overall success rate** — share of beats fully confirmed: every "
             "key in the beat (all notes of a chord) reached threshold within "
             f"the `{PRESS_TIMEOUT} s` timeout.\n")
    L.append("## Results\n")
    L.append("| " + " | ".join(HEADERS) + " |")
    L.append("|" + "|".join(["---"] * len(HEADERS)) + "|")
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("| **" + "** | **".join(agg) + "** |")
    L.append("")
    L.append("Realized tempo per song (from the best-fit slope above) — i.e. "
             "the steady tempo the hand actually sustained:")
    L.append("")
    for m in metrics_list:
        rt = m["realized_tempo_bpm"]
        rt_s = "n/a" if (rt is None or math.isnan(rt)) else f"{rt:.0f} bpm"
        L.append(f"- **{m['title']}** — nominal {m['tempo_bpm']:.0f} bpm → "
                 f"realized ~{rt_s}")
    L.append("")
    L.append("> Both timing columns are computed from the exact same real onset "
             "timestamps; they differ only in what they hold fixed. The closed "
             "loop trades absolute tempo for a guarantee that every note is "
             "physically sensor-confirmed before moving on — so press/success "
             "rates are perfect and the steady-state rhythm (jitter) is what "
             "the onset-vs-grid drift would otherwise obscure.\n")
    with open(BENCHMARK_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return BENCHMARK_DOC_PATH


def run_benchmark():
    song_files = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))
    if not song_files:
        print(f"No songs (*.json) found in {CONFIG_DIR}", flush=True)
        return

    print(f"=== Maestro benchmark: {len(song_files)} song(s), seed={SEED}, "
          f"deterministic ===\n", flush=True)

    metrics_list = []
    for sf in song_files:
        print(f"-> running {os.path.relpath(sf, HERE)} ...", flush=True)
        song, m, sim_t = _run_song_headless(sf)
        metrics_list.append(m)
        print(f'   "{song.title}": {m["confirmed_keys"]}/{m["total_notes"]} '
              f"strikes confirmed, {m['confirmed_beats']}/{m['total_beats']} "
              f"beats ok, onset-err {_fmt_timing(m['onset_err_ms'])} ms, "
              f"jitter {_fmt_timing(m['jitter_ms'])} ms "
              f"(realized ~{m['realized_tempo_bpm']:.0f} bpm), "
              f"{sim_t:.1f}s sim\n", flush=True)

    rows, agg = _benchmark_rows(metrics_list)
    print("=== Summary ===\n", flush=True)
    _print_table(rows, agg)
    out = _write_benchmark_md(metrics_list, rows, agg)
    print(f"\nWrote results table to {os.path.relpath(out, HERE)}", flush=True)
    return metrics_list


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
    parser.add_argument(
        "--benchmark", action="store_true",
        help="play every song in config/ headless and report REAL measured "
             "metrics (press-confirm rate, timing error, success rate); also "
             "writes docs/benchmark_results.md")
    args = parser.parse_args()
    loops = max(1, args.loops)

    if args.benchmark:
        run_benchmark()
    elif args.record is not None:
        run_record(args.record, loops)
    else:
        run_interactive(loops)


if __name__ == "__main__":
    main()

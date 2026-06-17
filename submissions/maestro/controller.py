#!/usr/bin/env python3
"""Closed-loop melody player for the dexterous-hand / piano scene.

The melody comes entirely from config/song.json (via src/score_parser);
sensor reads come from src/feedback. The controller is a closed-loop
state machine: it presses a key by smoothly interpolating the finger
from its rest pose to its press pose, and it only advances to the next
note once that key's touch sensor CONFIRMS the press (force > threshold).
If a press is somehow missed, a timeout logs a warning and the melody
continues without crashing.

Usage:
    python controller.py            # play live in the interactive viewer
    python controller.py --record   # render headless to media/hot_cross_buns.gif
    python controller.py --record out.mp4   # custom path (.mp4 needs imageio-ffmpeg)
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
import score_parser            # noqa: E402
from feedback import TouchFeedback   # noqa: E402

# ----------------------------------------------------------------------
# TUNABLES
# ----------------------------------------------------------------------
SEED = 0                       # fixed seed -> deterministic run

PRESS_THRESHOLD = 2.0          # N; sensor force that CONFIRMS a press
RELEASE_THRESHOLD = 0.1        # N; below this the key counts as released
PRESS_TIMEOUT = 1.5            # s; if no confirm by now -> warn & continue
PRESS_RAMP = 0.35              # s; time to interpolate rest -> press (smooth, graceful)
RELEASE_RAMP = 0.30            # s; time to interpolate press -> rest
RELEASE_GAP = 0.10             # s; extra settle after release before next note
SETTLE_SECONDS = 0.5           # let things settle before the first note

# Finger press pose (radians). Proximal joint flexes most so the fingertip
# reaches DOWN onto the key instead of curling into a fist.
REST_POSE = np.array([0.0, 0.0, 0.0])
PRESS_DEPTH = -1.20            # proximal-joint target; more negative = harder press
PRESS_POSE = np.array([PRESS_DEPTH, -0.40, -0.20])

# The pinky (f4) plays no note in this 3-key melody; hold it in a slightly
# curled, relaxed pose so it looks like a naturally resting finger.
PINKY_FINGER = "f4"
PINKY_REST_POSE = np.array([-0.35, -0.60, -0.60])

REALTIME = True                # pace the sim to wall-clock so it's watchable

# Recording (headless --record mode). 1280x720 is clean HD and divisible by 16
# (so the mp4 encoder needs no padding); GIFs are downscaled on write to stay small.
RECORD_FPS = 30
RECORD_W = 1280
RECORD_H = 720
GIF_MAX_W = 640                # GIF output is downscaled to this width to limit size
DEFAULT_RECORD_PATH = os.path.join(HERE, "media", "hot_cross_buns.gif")
RECORD_SAFETY_SECONDS = 60.0   # hard stop so a stuck run can't render forever

# Map each color in the song to the finger + touch sensor in the model.
#   red = left (key_C) -> f1 | green = center (key_D) -> f2 | blue = right (key_E) -> f3
KEY_MAP = {
    "red":   {"finger": "f1", "sensor": "key_C_sensor"},
    "green": {"finger": "f2", "sensor": "key_D_sensor"},
    "blue":  {"finger": "f3", "sensor": "key_E_sensor"},
}
SONG_PATH = os.path.join(HERE, "config", "song.json")
# ----------------------------------------------------------------------


def smoothstep(x):
    """Eased 0->1 ramp (zero velocity at both ends)."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class FingerRamp:
    """Smoothly interpolates one finger's actuator targets over time."""

    def __init__(self, act_ids, pose):
        self.act_ids = act_ids
        self.start = np.array(pose, dtype=float)
        self.target = np.array(pose, dtype=float)
        self.t0 = 0.0
        self.duration = 1e-9

    def move_to(self, pose, now, duration):
        # start the new ramp from wherever we currently are
        self.start = self.current(now)
        self.target = np.array(pose, dtype=float)
        self.t0 = now
        self.duration = max(duration, 1e-9)

    def current(self, now):
        alpha = smoothstep((now - self.t0) / self.duration)
        return self.start + (self.target - self.start) * alpha

    def apply(self, data, now):
        pose = self.current(now)
        for aid, val in zip(self.act_ids, pose):
            data.ctrl[aid] = val


class MelodyPlayer:
    """Closed-loop state machine that plays one song on the keyboard.

    Owns the per-note state; `tick()` advances the controller by exactly one
    physics step and returns False once the melody is finished. Both the
    interactive viewer and the headless recorder drive it the same way, so the
    control logic is identical in either mode.
    """

    def __init__(self, model, data, song, loops=1):
        self.model = model
        self.data = data
        self.song = song
        self.loops = max(1, loops)
        self.loop_idx = 1
        self.feedback = TouchFeedback(model)

        def actuator_ids(finger):
            return [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{finger}_j{j}_act")
                for j in (1, 2, 3)
            ]

        # one smooth ramp per active finger; all start at rest
        self.ramps = {color: FingerRamp(actuator_ids(m["finger"]), REST_POSE)
                      for color, m in KEY_MAP.items()}
        self.sensors = {color: m["sensor"] for color, m in KEY_MAP.items()}

        # hold the unused pinky in a relaxed curl (never driven by the melody)
        for aid, target in zip(actuator_ids(PINKY_FINGER), PINKY_REST_POSE):
            data.ctrl[aid] = target

        self.state = "SETTLE"
        self.state_t0 = 0.0
        self.note_idx = 0
        self.hold_until = 0.0
        self.done = False

    def _note(self):
        return self.song.notes[self.note_idx]

    def _begin_press(self, now):
        n = self._note()
        self.ramps[n.key].move_to(PRESS_POSE, now, PRESS_RAMP)
        print(f"[{now:5.2f}s] note {self.note_idx + 1}/{len(self.song.notes)}: "
              f"{n.key} -> pressing...", flush=True)

    def tick(self):
        """Run one control step + one physics step. Returns False when done."""
        data = self.data
        now = data.time
        elapsed = now - self.state_t0

        if self.state == "SETTLE":
            if elapsed >= SETTLE_SECONDS:
                self._begin_press(now)
                self.state, self.state_t0 = "PRESS", now

        elif self.state == "PRESS":
            n = self._note()
            f = self.feedback.force(data, self.sensors[n.key])
            if self.feedback.is_pressed(data, self.sensors[n.key], PRESS_THRESHOLD):
                # closed-loop confirmation: only now do we commit to advance
                hold_s = n.seconds(self.song.tempo_bpm)
                print(f"[{now:5.2f}s] KEY PRESSED  "
                      f"({n.key}, force={f:.2f} N, hold {hold_s:.2f}s)", flush=True)
                self.hold_until = now + hold_s
                self.state, self.state_t0 = "HOLD", now
            elif elapsed >= PRESS_TIMEOUT:
                print(f"[{now:5.2f}s] WARNING: {n.key} press not confirmed "
                      f"within {PRESS_TIMEOUT:.1f}s (force={f:.2f} N); "
                      f"skipping note and continuing", flush=True)
                self.hold_until = now   # no hold; release immediately
                self.state, self.state_t0 = "HOLD", now

        elif self.state == "HOLD":
            if now >= self.hold_until:
                self.ramps[self._note().key].move_to(REST_POSE, now, RELEASE_RAMP)
                self.state, self.state_t0 = "RELEASE", now

        elif self.state == "RELEASE":
            released = self.feedback.is_released(
                data, self.sensors[self._note().key], RELEASE_THRESHOLD)
            if elapsed >= RELEASE_RAMP + RELEASE_GAP and released:
                self.note_idx += 1
                if self.note_idx >= len(self.song.notes):
                    if self.loop_idx >= self.loops:
                        print(f'\n[{now:5.2f}s] melody complete: '
                              f'"{self.song.title}" played '
                              f'{self.loops}x.', flush=True)
                        self.done = True
                    else:
                        self.loop_idx += 1
                        self.note_idx = 0
                        print(f'\n[{now:5.2f}s] --- loop {self.loop_idx}/'
                              f'{self.loops} ---', flush=True)
                        self._begin_press(now)
                        self.state, self.state_t0 = "PRESS", now
                else:
                    self._begin_press(now)
                    self.state, self.state_t0 = "PRESS", now

        # advance every finger's smooth interpolation, then step physics
        for ramp in self.ramps.values():
            ramp.apply(data, now)
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
          f'({len(song.notes)} notes x{loops})', flush=True)
    print(f"Advancing only on sensor confirm (> {PRESS_THRESHOLD} N). "
          f"Watch the viewer.\n", flush=True)

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
    """Write buffered frames to an animated GIF via PIL, downscaled to GIF_MAX_W
    so the file stays small. (MP4 is streamed separately and keeps full res.)"""
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

    print(f'Recording "{song.title}" x{loops} headless @ {RECORD_FPS} fps '
          f'({RECORD_W}x{RECORD_H})...', flush=True)

    # MP4 streams each frame straight to ffmpeg (constant memory -> safe for long
    # loops); GIF buffers frames in a list and writes once at the end via PIL.
    writer = None
    frames = None
    if ext == ".mp4":
        try:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            writer = imageio.get_writer(path, fps=RECORD_FPS)
        except Exception as e:
            path = os.path.splitext(path)[0] + ".gif"
            print(f"  (mp4 encoder unavailable: {e}; falling back to {path})",
                  flush=True)
    if writer is None:
        frames = []

    renderer = mujoco.Renderer(model, height=RECORD_H, width=RECORD_W)
    frame_interval = 1.0 / RECORD_FPS
    next_capture = 0.0
    n_frames = 0
    safety = max(RECORD_SAFETY_SECONDS, loops * 15.0 + 30.0)
    try:
        while True:
            if data.time >= next_capture:
                renderer.update_scene(data, camera=cam)
                img = renderer.render()
                if writer is not None:
                    writer.append_data(img)          # streamed; no copy needed
                else:
                    frames.append(img.copy())        # buffered for GIF
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

    out = path if writer is not None else _write_gif(frames, path, RECORD_FPS)
    size_kb = os.path.getsize(out) / 1024.0
    print(f"Saved {n_frames} frames ({n_frames / RECORD_FPS:.1f}s) -> {out} "
          f"({size_kb:.0f} KB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Play the piano melody.")
    parser.add_argument(
        "--record", nargs="?", const=DEFAULT_RECORD_PATH, default=None,
        metavar="PATH",
        help="render headless to a video file instead of opening the viewer "
             f"(default: {os.path.relpath(DEFAULT_RECORD_PATH, HERE)})")
    parser.add_argument(
        "--loops", type=int, default=1, metavar="N",
        help="repeat the whole melody N times (default 1)")
    args = parser.parse_args()
    loops = max(1, args.loops)

    if args.record is not None:
        run_record(args.record, loops)
    else:
        run_interactive(loops)


if __name__ == "__main__":
    main()

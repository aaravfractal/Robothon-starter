# Rubric Mapping — Detailed (v2)

This document maps each of the 8 rubric criteria to the concrete code that
satisfies it, with exact file and line references. It is written to be
**honest**: where a feature is simpler than its name might suggest, the
limitation is stated plainly.

v2 upgrades the v1 (3-key, welded-hand) submission to a **5-finger hand on a
sliding (prismatic) wrist** over a **7-key keyboard**, playing **Ode to Joy**
with **chords** and **velocity dynamics**, and **synthesizing the audio** into
the recorded video. Line numbers refer to the files as committed here.

---

## 1. Runnability

**Claim:** The project runs from a clean checkout with one command and no
absolute paths.

- **Single entry point.** `python controller.py` plays the full piece in the
  viewer; an `argparse` front end (`controller.py:473-490`) also exposes
  `--record` (headless MP4/GIF) and `--loops`.
- **Relative / cwd-independent paths.** Every path is derived from the script
  location: `HERE = os.path.dirname(os.path.abspath(__file__))`
  (`controller.py:38`), `sys.path.insert(...,"src")` (`:39`),
  `SONG_PATH` (`:90`), `scene.xml` via `os.path.join(HERE, "scene.xml")`
  (`:341`). `play.py`, `score_parser.py`, and `audio.py` follow the same pattern.
- **Pinned dependencies.** `requirements.txt` pins `mujoco==3.9.0`,
  `numpy==2.4.6`, `Pillow==12.2.0`, and `imageio`/`imageio-ffmpeg`. The MP4
  audio mux uses the ffmpeg binary **bundled** with `imageio-ffmpeg`
  (`src/audio.py:96-104`), so **no system ffmpeg install is required**.
- **Determinism:** `SEED = 0` (`controller.py:47`), `np.random.seed(SEED)`
  (`:339`), `mj_resetData` + `mj_forward` (`:343-344`), fixed `0.002 s`
  timestep (`scene.xml:4`). The geometry is also self-calibrated from the model
  at startup (`controller.py:162-163`), so slide targets are exact regardless of
  small XML edits.

*Verified:* a full headless render played all 40 beats with **zero** press
timeouts and produced a 1280×720 MP4 with an AAC audio track (see §7).

---

## 2. Depth of MuJoCo Use

**Claim:** The scene uses a broad, idiomatic slice of MJCF, and the code reads
sensor data back from the simulation.

In `scene.xml`:
- **Options** — gravity, `0.002` timestep, `implicitfast` integrator (`:4`).
- **Visual / rendering** — headlight tuning + offscreen framebuffer
  (`offwidth/offheight`) for HD capture (`:6-9`).
- **`default` classes** — reusable `phalanx`, `finger_act`, `key` classes
  (`:11-29`).
- **Assets** — procedural checker texture + materials (`:31-39`).
- **Joints (two kinds):**
  - **Hinge** — finger flexion (`phalanx` class, `:16-17`) and the spring-return
    keys (`key` class, `:24-26`).
  - **Prismatic (slide)** — the wrist carriage that translates the hand sideways
    (`:117-122`). This is the new DOF that makes "reach" possible.
- **Spring-dampers** — keys use `stiffness=40, damping=1.2` to rest up and
  return without bounce (`:25-26`); finger joints use `stiffness=0.3,
  damping=0.05` (`:16-17`); the slide is damped (`:118-119`).
- **Tendons** — five **fixed tendons** couple each finger's middle + distal
  flexion joints into one "flexor cable" length (`:203-208`).
- **Actuators** — 15 `position` actuators (one per finger joint) + 1 wrist-slide
  `position` actuator = **16 actuated DOF** (`:213-236`).
- **Sensors (three kinds, 18 total):** 7 `touch` (one per key top, `:243-249`),
  1 `jointpos` on the wrist slide + 5 `jointpos` on the finger proximals
  (`:251-257`), and 5 `tendonpos` on the flexor tendons (`:259-263`).

In code:
- Touch values are read back from `data.sensordata` by sensor address
  (`feedback.py:20-33`) and used to confirm presses every step.
- The wrist slide position is read from `data.qpos` to know when the hand has
  arrived before pressing (`controller.py:159-160,289,298`).
- An offscreen `mujoco.Renderer` renders frames from the fixed camera for
  `--record` (`controller.py:417-432`).

*Limitation, stated honestly:* the fixed tendons are **read back** (via
`tendonpos` sensors, logged) but are **not** the actuation path — fingers are
driven by per-joint position actuators, a deliberate trade-off for clean,
independent, testable control. The tendons model the kinematic coupling and
provide an extra sensed channel; they are not load-bearing flexor cables.

---

## 3. Task Design

**Claim:** A clear, recognizable task with well-shaped physics, now with
chords and a wider range.

- **Seven color-coded keys** — a rainbow C→B, each a hinge that pivots down
  about its back edge and **springs back up** (`scene.xml:65-114`); spacing
  (0.045 m) equals the finger spacing so the 5 fingers line up one-per-key at
  any slide offset.
- **A real, longer tune** — *Ode to Joy*: an ascending run across all 7 keys,
  the full two-period theme (keys C–G), and a **C major → E minor → C major**
  cadence (`config/song.json`). The E-minor chord uses key B, so the score
  exercises the **entire** keyboard.
- **Chords** — a beat may list 2–3 keys struck **simultaneously by different
  fingers** (`config/song.json`; parsed at `score_parser.py:66-83`).
- **Data-driven** — title, tempo, beats, chords, and per-note velocity all live
  in JSON; `score_parser.py` validates and turns them into `Note`s. Editing the
  JSON changes the performance (and the synthesized audio) with no code change.

---

## 4. Control

**Claim:** A genuine closed-loop controller that slides, presses chords, and
follows tempo and dynamics.

- **State machine** — `SETTLE → SLIDE → PRESS → HOLD → RELEASE` per beat,
  looping the score, in `MelodyPlayer.tick()` (`controller.py:286-334`).
- **Per-beat planning** — `_plan()` (`controller.py:188-213`) chooses a finger
  assignment **and** a wrist-slide target: a chord's sorted keys map to fingers
  that preserve the key gaps, and among the valid bases it picks the one needing
  the **least wrist travel** (smooth, minimal motion). Geometry is
  self-calibrated from fingertip/key site positions (`:162-163,178-180`).
- **Smooth motion** — `smoothstep` easing (`:94-97`) drives a `Ramp` that
  interpolates both finger targets and the wrist target from their current pose
  to the new one (`:100-124`); the slide ramp duration scales with distance for
  ~constant speed (`:228-229`).
- **Velocity-sensitive press** — the press ramp duration is interpolated between
  `PRESS_RAMP_SLOW` and `PRESS_RAMP_FAST` by the note's velocity, so louder
  notes are struck faster/harder (`controller.py:242-247`). Measured peak
  contact force tracks velocity (e.g. ≈5.4 N at vel 0.86 vs ≈3.4 N at vel 0.55).
- **Closed-loop advancement** — `PRESS` polls **every** struck key's sensor and
  only commits to `HOLD` once `_all_pressed()` is true (`:252-255,300-311`);
  `RELEASE` waits for `_all_released()` (`:256-258,318-326`). For a chord this
  means all 2–3 sensors must confirm before the beat advances.
- **Timeout safety** — if a press is not confirmed within `PRESS_TIMEOUT`
  (1.5 s) it logs a warning and continues (`:313-316`).
- **Tempo** — hold time per beat is `duration * 60 / tempo_bpm`
  (`score_parser.py:54-56`, used at `controller.py:305`).

---

## 5. Dexterous Manipulation

**Claim:** A multi-finger hand with a reaching wrist, used for independent,
coordinated, simultaneous finger control.

- **Hand structure** — a carriage on a **prismatic wrist** (`scene.xml:117-122`)
  carrying **5 fingers**, each a 3-link capsule chain on **3 hinge flexion
  joints** = **15 finger DOF + 1 slide DOF = 16 actuated DOF**
  (`scene.xml:123-200`).
- **Reach** — because the 5 fingers span only 5 of the 7 keys at once, the
  controller slides the wrist to bring the right finger over the target key, and
  the *same* finger plays different keys at different slide positions (and vice
  versa). The ascending intro run visibly translates the hand across the whole
  keyboard.
- **Simultaneous multi-finger chords** — a 3-note chord is struck by **three
  different fingers at once** (e.g. C-E-G → fingers f1, f3, f5); all three touch
  sensors confirm together (`controller.py:188-213,242-247,252-255`). The
  in-between keys are **not** pressed — verified by the other sensors reading
  0 N (no cross-talk).

*Limitation, stated honestly:* this is finger **flexion to depress keys** plus a
1-DOF sideways wrist, **not** object grasping or in-hand manipulation. Each
finger joint has a single flexion axis (no abduction/adduction). Within that
scope, the hand demonstrates independent, coordinated, simultaneous multi-finger
control with whole-hand repositioning.

---

## 6. Engineering Quality

**Claim:** Modular, validated, readable code.

- **Separation of concerns** — score parsing (`src/score_parser.py`), sensor
  feedback (`src/feedback.py`), **audio synthesis/mux** (`src/audio.py`), and
  control (`controller.py`) are independent modules; `play.py` is a separate
  manual-test harness.
- **Typed data model** — `Note`/`Song` are frozen dataclasses with an
  `is_chord` property and `seconds()` tempo math (`score_parser.py:44-66`).
- **Input validation** — tempo > 0, duration > 0, velocity ∈ (0,1], chord size
  1–3, no duplicate keys, known note names — each raising a clear error
  (`score_parser.py:66-105`).
- **Reusable control core** — one `Ramp` class drives both fingers and the wrist
  (`controller.py:100-124`); one `MelodyPlayer.tick()` is shared by the viewer
  and the recorder (`:352-375,391-470`).
- **Self-calibrating geometry** — slide targets are computed from the model's
  actual site positions, not hardcoded (`controller.py:162-163,178-180,203`), so
  the control stays correct if the scene's spacing changes.
- **Named tunables** — all thresholds, ramps, poses, slide params grouped at the
  top of `controller.py` (`:45-90`); no magic numbers in the loop.
- **Docstrings** — every module, class, and non-trivial method is documented.

---

## 7. Presentation

**Claim:** The simulation is set up to be watched, heard, and understood.

- **Purpose-built camera** — fixed `overview` camera framed for the wider 7-key
  keyboard + the sliding hand (`scene.xml:49-51`); the controller selects it
  automatically (`controller.py:363-364`).
- **HD capture** — offscreen buffer at 1280×720 (`scene.xml:8`); the recorder
  renders at that resolution (`controller.py:60-62,417`).
- **Synthesized audio in the video** — each key maps to a pitch
  (`src/audio.py:24-27`); on every confirmed press the controller records a note
  event (`controller.py:260-266`); after the run the events are synthesized into
  a struck-tone track (harmonic stack + attack/decay envelope,
  `src/audio.py:34-81`), written to WAV, and **muxed onto the MP4** with the
  bundled ffmpeg (`controller.py:449-463`, `src/audio.py:106-118`). Audio time =
  sim time = video time, so each note lands on its press. **The demo plays the
  tune.**
- **Live narration** — the controller prints each beat (keys, chord/note,
  slide target, fingers), the confirming forces, the velocity, and a completion
  line (`controller.py:237-240,304-308,328`).
- **Discoverability** — `play.py` prints every actuator and sensor name on
  startup and lets you drag the fingers and the wrist slide by hand.

*Output:* `media/ode_to_joy.mp4` (~63 s, 1280×720, H.264 + AAC audio) plus a
small silent `media/ode_to_joy_preview.gif` for inline preview.

---

## 8. Innovation

**Claim:** Several ideas beyond a minimal scripted demo.

- **Data-driven chord/velocity score, decoupled from control.** The performance
  is a JSON document (`config/song.json`) of beats — each a list of keys plus a
  dynamic — parsed into a validated note list (`score_parser.py`). The
  controller knows nothing about *which* song it plays.
- **Per-beat motion planner.** `_plan()` (`controller.py:188-213`) automatically
  derives a finger assignment **and** a wrist-slide position for any beat
  (single note or chord) by minimizing wrist travel — the hand "figures out how
  to reach" each beat rather than following hardcoded poses.
- **Sensor-confirmed playback that drives procedurally-synthesized audio.** The
  controller closes the loop on the touch sensors (advancing only on physical
  confirmation of **all** struck keys, `controller.py:252-255,300-311`) and uses
  exactly those confirmation events to **synthesize and mux the soundtrack**
  (`src/audio.py`). The sim doesn't just *look* like it plays — its sensed
  contacts *generate* the music — and it degrades gracefully with a logged
  warning if a press is ever missed (`controller.py:313-316`).

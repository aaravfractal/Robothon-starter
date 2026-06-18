#!/usr/bin/env python3
"""Procedural audio for the piano demo.

Each of the 7 color-coded keys maps to a pitch (one diatonic C-major octave,
C4..B4). When the controller confirms a press it records a note event
(start time, note name, hold duration, velocity); after the run this module
synthesizes a struck-string-like tone for every event, sums overlapping
events (so chords sound together), writes a WAV, and muxes it onto the
recorded MP4 with the bundled ffmpeg (imageio-ffmpeg) so the demo *plays the
tune*. No external audio assets or system ffmpeg are required.

Audio time == simulation time == video time: in `--record` mode the recorder
captures one frame per 1/fps of *sim* time and the MP4 plays back at that fps,
so a note confirmed at sim-time t lands at second t of the video.
"""
import os
import struct
import subprocess
import wave

import numpy as np

# Note name -> fundamental frequency (Hz), one diatonic octave, C4..B4.
NOTE_FREQ = {
    "C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23,
    "G": 392.00, "A": 440.00, "B": 493.88,
}

SR = 44100                     # audio sample rate
# Relative weights of the first few harmonics -> a warm, slightly bright tone.
HARMONICS = (1.0, 0.45, 0.22, 0.11)


def _tone(freq, dur, velocity, sr=SR):
    """One struck-string-like note: harmonic stack under a fast-attack,
    exponentially-decaying envelope. `velocity` (0..1) scales loudness."""
    n = max(1, int(round(dur * sr)))
    t = np.arange(n) / sr
    wave_sum = np.zeros(n, dtype=np.float64)
    for h, weight in enumerate(HARMONICS, start=1):
        wave_sum += weight * np.sin(2.0 * np.pi * freq * h * t)
    wave_sum /= sum(HARMONICS)

    # envelope: ~6 ms attack, exponential decay, short release tail
    env = np.ones(n, dtype=np.float64)
    atk = min(n, int(0.006 * sr))
    if atk > 0:
        env[:atk] = np.linspace(0.0, 1.0, atk)
    tau = max(0.08, dur * 0.6)
    env *= np.exp(-t / tau)
    rel = min(n, int(0.03 * sr))
    if rel > 0:
        env[-rel:] *= np.linspace(1.0, 0.0, rel)

    amp = 0.18 + 0.82 * float(np.clip(velocity, 0.0, 1.0))
    return (wave_sum * env * amp).astype(np.float64)


def synthesize(events, total_seconds, sr=SR):
    """Sum every note event into one mono float32 track of `total_seconds`.

    events: iterable of dicts {"t": start_s, "note": name, "dur": s, "vel": v}.
    Overlapping events (chords, or a note ringing into the next) simply add.
    """
    total = max(1, int(round(total_seconds * sr)) + sr // 2)  # small tail
    track = np.zeros(total, dtype=np.float64)
    for e in events:
        freq = NOTE_FREQ.get(e["note"])
        if freq is None:
            continue
        seg = _tone(freq, e["dur"], e["vel"], sr)
        start = int(round(e["t"] * sr))
        end = min(total, start + len(seg))
        if start < total and end > start:
            track[start:end] += seg[: end - start]

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1e-9:
        track *= 0.89 / peak          # normalize, leave a little headroom
    return track


def write_wav(track, path, sr=SR):
    """Write a mono float track to a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = np.clip(track, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _ffmpeg_exe():
    """Path to an ffmpeg binary: prefer the one bundled with imageio-ffmpeg
    (no system install needed); fall back to a system `ffmpeg` on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def mux(video_path, wav_path, out_path):
    """Mux a WAV onto a (silent) MP4, writing `out_path`. Video is stream-copied
    (no re-encode); audio is encoded to AAC. Returns out_path on success."""
    exe = _ffmpeg_exe()
    cmd = [
        exe, "-y",
        "-i", video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

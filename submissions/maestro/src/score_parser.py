#!/usr/bin/env python3
"""Parse a song file (config/song.json) into an ordered list of beats.

The v2 JSON schema supports chords (several keys struck at once) and a
per-beat dynamic (velocity):

    {
      "title":     str,
      "tempo_bpm": number,
      "notes": [
        { "keys": ["E"],          "duration": 1, "velocity": 0.6 },
        { "keys": ["C","E","G"],  "duration": 2, "velocity": 0.9 },
        ...
      ]
    }

- `keys` is a list of note names (a single-element list is a melody note; two
  or three names are a chord struck simultaneously by different fingers).
  `"key": "E"` (singular, scalar) is still accepted as shorthand for
  `"keys": ["E"]` so older single-note scores keep working.
- `duration` is in *beats*; convert to seconds with the song tempo.
- `velocity` is the dynamic in (0, 1]: higher = struck faster/harder (louder).
  Optional; defaults to `DEFAULT_VELOCITY`.

JSON preserves array order, so the returned beat list is the play order.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

# Note names map to the 7 color-coded keys, low -> high (one diatonic octave).
VALID_KEYS = ("C", "D", "E", "F", "G", "A", "B")
DEFAULT_VELOCITY = 0.7
MAX_CHORD = 3   # at most 3 keys struck at once (the hand has 5 fingers)

# default path, relative to this file -> works regardless of cwd
_DEFAULT_SONG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "song.json",
)


@dataclass(frozen=True)
class Note:
    keys: Tuple[str, ...]   # one note name, or several for a chord
    duration: float         # in beats
    velocity: float         # dynamic in (0, 1]; louder = struck faster

    @property
    def is_chord(self) -> bool:
        return len(self.keys) > 1

    def seconds(self, tempo_bpm: float) -> float:
        """This beat's hold time in seconds at the given tempo."""
        return self.duration * (60.0 / tempo_bpm)


@dataclass(frozen=True)
class Song:
    title: str
    tempo_bpm: float
    notes: List[Note]


def _beat_keys(raw: dict, i: int) -> Tuple[str, ...]:
    """Pull and validate the key list from one raw beat (accepts key|keys)."""
    if "keys" in raw:
        keys = list(raw["keys"])
    elif "key" in raw:
        keys = [raw["key"]]
    else:
        raise ValueError(f"beat {i}: must have 'keys' (or 'key')")
    if not 1 <= len(keys) <= MAX_CHORD:
        raise ValueError(
            f"beat {i}: {len(keys)} keys; must be 1..{MAX_CHORD}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"beat {i}: duplicate keys in chord {keys}")
    for k in keys:
        if k not in VALID_KEYS:
            raise ValueError(
                f"beat {i}: key {k!r} not one of {VALID_KEYS}")
    return tuple(keys)


def load_song(path: str = _DEFAULT_SONG) -> Song:
    """Read the song file and return a Song with an ordered beat list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data["title"]
    tempo = float(data["tempo_bpm"])
    if tempo <= 0:
        raise ValueError(f"tempo_bpm must be positive, got {tempo}")

    notes: List[Note] = []
    for i, n in enumerate(data["notes"]):
        keys = _beat_keys(n, i)
        duration = float(n["duration"])
        if duration <= 0:
            raise ValueError(f"beat {i}: duration must be positive, got {duration}")
        velocity = float(n.get("velocity", DEFAULT_VELOCITY))
        if not 0.0 < velocity <= 1.0:
            raise ValueError(
                f"beat {i}: velocity must be in (0, 1], got {velocity}")
        notes.append(Note(keys=keys, duration=duration, velocity=velocity))

    return Song(title=title, tempo_bpm=tempo, notes=notes)


def get_notes(path: str = _DEFAULT_SONG) -> List[Note]:
    """Convenience: return just the ordered list of beats."""
    return load_song(path).notes


if __name__ == "__main__":
    song = load_song()
    print(f"{song.title}  @ {song.tempo_bpm} bpm  ({len(song.notes)} beats)")
    for i, note in enumerate(song.notes):
        kind = "chord" if note.is_chord else "note "
        print(f"  {i + 1:2}. {kind} {'+'.join(note.keys):10} "
              f"{note.duration} beats = {note.seconds(song.tempo_bpm):.2f}s "
              f"vel={note.velocity:.2f}")

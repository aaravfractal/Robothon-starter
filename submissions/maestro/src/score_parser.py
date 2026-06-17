#!/usr/bin/env python3
"""Parse a song file (config/song.json) into an ordered list of notes.

The JSON schema:
    {
      "title":     str,
      "tempo_bpm": number,
      "notes": [ { "key": "red"|"green"|"blue", "duration": number }, ... ]
    }

`duration` is in *beats*; convert to seconds with the song tempo.
JSON preserves array order, so the returned note list is the play order.
"""
import json
import os
from dataclasses import dataclass
from typing import List

VALID_KEYS = ("red", "green", "blue")

# default path, relative to this file -> works regardless of cwd
_DEFAULT_SONG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "song.json",
)


@dataclass(frozen=True)
class Note:
    key: str          # color: "red" | "green" | "blue"
    duration: float   # in beats

    def seconds(self, tempo_bpm: float) -> float:
        """This note's hold time in seconds at the given tempo."""
        return self.duration * (60.0 / tempo_bpm)


@dataclass(frozen=True)
class Song:
    title: str
    tempo_bpm: float
    notes: List[Note]


def load_song(path: str = _DEFAULT_SONG) -> Song:
    """Read the song file and return a Song with an ordered note list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data["title"]
    tempo = float(data["tempo_bpm"])
    if tempo <= 0:
        raise ValueError(f"tempo_bpm must be positive, got {tempo}")

    notes: List[Note] = []
    for i, n in enumerate(data["notes"]):
        key = n["key"]
        if key not in VALID_KEYS:
            raise ValueError(
                f"note {i}: key {key!r} not one of {VALID_KEYS}")
        notes.append(Note(key=key, duration=float(n["duration"])))

    return Song(title=title, tempo_bpm=tempo, notes=notes)


def get_notes(path: str = _DEFAULT_SONG) -> List[Note]:
    """Convenience: return just the ordered list of notes."""
    return load_song(path).notes


if __name__ == "__main__":
    song = load_song()
    print(f"{song.title}  @ {song.tempo_bpm} bpm  ({len(song.notes)} notes)")
    for i, note in enumerate(song.notes):
        print(f"  {i + 1}. {note.key:5}  {note.duration} beats "
              f"= {note.seconds(song.tempo_bpm):.2f}s")

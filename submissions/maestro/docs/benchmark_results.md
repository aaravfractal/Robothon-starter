# Maestro Benchmark Results

Every number in this file is **measured from a real, headless MuJoCo run** of the closed-loop controller (`python controller.py --benchmark`). No values are hand-entered; re-running regenerates this file.

- Seed: `0` (deterministic) · timestep `0.002 s` · press threshold `1.0 N` · press timeout `1.5 s`
- Songs benchmarked: **3** — every `*.json` in [`config/`](../config) (in filename order)

## Metric definitions

- **Notes attempted** — individual key-strikes the score asks for (a 3-key chord counts as 3 strikes). Reported as `confirmed/attempted`.
- **Sensor-confirmed press rate** — share of those strikes whose own touch sensor reached the press threshold, read live from `data.sensordata` at the moment the beat resolved.
- **Onset err vs grid (ms)** — the literal *timing error vs intended note onset*: mean absolute deviation of each confirmed note's onset from the nominal tempo grid, after aligning the grid to the first confirmed note (constant start-up latency removed). This is **large on purpose and reported honestly**: the controller is closed-loop and self-paced — it slides the wrist, ramps each press, and only advances once the touch sensor confirms — so it runs slower than the score's nominal bpm and the gap accumulates across the piece.
- **Tempo-norm jitter (ms)** — the same onsets after removing the best-fit *constant* tempo (least-squares `onset ≈ slope·grid + intercept`). This strips out the fixed per-beat mechanical overhead and the slower-than-nominal tempo, leaving only genuine rhythmic **irregularity** — i.e. how *steady* the beat is. Lower = steadier.
- **Overall success rate** — share of beats fully confirmed: every key in the beat (all notes of a chord) reached threshold within the `1.5 s` timeout.

## Results

| Song | BPM | Notes (confirmed/attempted) | Press rate | Onset err vs grid (ms) | Tempo-norm jitter (ms) | Success rate |
|---|---|---|---|---|---|---|
| C Major Scale (up and down) | 110 | 13/13 | 100.0% | 4921.0 | 55.0 | 13/13 (100.0%) |
| Ode to Joy | 90 | 46/46 | 100.0% | 15795.8 | 398.3 | 40/40 (100.0%) |
| Twinkle Twinkle Little Star | 100 | 14/14 | 100.0% | 5181.6 | 170.1 | 14/14 (100.0%) |
| **All songs (pooled)** | **-** | **73/73** | **100.0%** | **11467.9** | **284.0** | **67/67 (100.0%)** |

Realized tempo per song (from the best-fit slope above) — i.e. the steady tempo the hand actually sustained:

- **C Major Scale (up and down)** — nominal 110 bpm → realized ~44 bpm
- **Ode to Joy** — nominal 90 bpm → realized ~42 bpm
- **Twinkle Twinkle Little Star** — nominal 100 bpm → realized ~45 bpm

> Both timing columns are computed from the exact same real onset timestamps; they differ only in what they hold fixed. The closed loop trades absolute tempo for a guarantee that every note is physically sensor-confirmed before moving on — so press/success rates are perfect and the steady-state rhythm (jitter) is what the onset-vs-grid drift would otherwise obscure.

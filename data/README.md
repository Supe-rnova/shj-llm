# Data

One row per trial in every file. Raw model output is kept verbatim
(`response_raw`, plus `rejected_raw` for any reply that was asked again).

```
data/
  pilot/                                   design decisions were made from these
    smoke_gemini-3.1-flash-lite_2026-09-09.csv   1 run, setup check
    pilot_gemini-3.1-flash-lite.csv              Types I + VI, 10 full runs each
    smoke_qwen3.8-27b_2026-10-12.csv             1 run, setup check for Qwen
    fake_test.csv                                fake answers, code test only
  main/                                    the data the paper analyses
    main_gemini-3.1-flash-lite.csv
    main_qwen3.8-27b.csv
```

Pilot and main data are never mixed. The pilot shaped the design (and
produced the below-chance Type VI observation), so the main study tests
everything on fresh runs with different seeds.

## Main study design (written by `src/main_study.py`)

- 40 runs per type per model, all six types.
- Short run: block 1 only (8 trials). Ceiling run: blocks 1-3 (24 trials).
- Block 1 of every run is the first pass, the only part that measures
  learning, so each type has 40 first passes per model.
- Ceiling runs (runs 1-3 of a type): Qwen, every type; Gemini, Types II-V.
  Gemini Types I and VI have none because the pilot already has 10 full
  64-trial runs of each.
- Seeds: `100000 + 1000 * type + run_number`, the same for both models, so
  run N of type T shows both models the same mapping and block-1 order.
- Column `run_kind` = `short` / `ceiling` (pilot rows: `full` or empty).

| Model | Short runs | Ceiling runs | Trials |
|---|---|---|---|
| Qwen 3.8 27B (Groq) | 222 | 18 | 2,208 |
| Gemini 3.1 Flash-Lite | 228 | 12 | 2,112 |

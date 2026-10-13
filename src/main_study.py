"""
main_study.py

Main data collection. Kept separate from run_pilot.py so the pilot code
stays exactly as it was run, and writes to data/main/, never data/pilot/.

    python src/main_study.py status <model>   # what is done, what is left
    python src/main_study.py run <model>      # collect; rerun daily

    <model> = gemini-3.1-flash-lite   or   qwen/qwen3.8-27b

DESIGN (decided 12 Oct 2026; summary in data/README.md)
  40 runs per type per model. Every run starts with block 1, the first
  pass through all 8 samples - the only part that measures learning.
    short run    1 block  = 8 trials
    ceiling run  3 blocks = 24 trials (block 1, then blocks 2-3 to show
                 the model is perfect once every answer is in its prompt)
  Block 1 of a ceiling run is identical to a short run, so every run
  gives one first pass: 40 first passes per type per model.

  Ceiling runs per type
    Qwen:   3 for every type
    Gemini: 3 for Types II-V; 0 for Types I and VI (the pilot already
            has 10 full 64-trial runs of each)
  Runs 1-3 of a type are its ceiling runs (when it has any); the rest
  are short.

SEEDS: 100000 + 1000 * type + run_number. Separate from the pilot seeds
(1000 * type + run_number), so no main-study run repeats a pilot run.
The SAME seeds are used for both models: run 7 of Type III shows Qwen and
Gemini the same feature mapping and the same block-1 order.

ORDER: runs are done round-robin (run 1 of Types I-VI, then run 2 of
Types I-VI, ...), so if collection stops early every type has a similar
number of runs.

RESUMING: rerun the same command each day. Finished runs are skipped; a
run cut off part-way is deleted and redone from its start. When a daily
limit is hit the script stops cleanly and tells you to rerun tomorrow.

OUTPUT: data/main/main_<model>.csv, one row per trial, both run kinds in
one file; the run_kind column says which.
"""

import sys
from pathlib import Path

import pandas as pd

from runner import run_one

TEMPERATURE = 1.0
INSTRUCTION_STYLE = "neutral"
SURFACE_SCHEME = "minerals"

RUNS_PER_TYPE = 40
CEILING_BLOCKS = 3
TYPES = (1, 2, 3, 4, 5, 6)
CEILING_RUNS = {
    "groq":   {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3},
    "google": {1: 0, 2: 3, 3: 3, 4: 3, 5: 3, 6: 0},
}
TRIALS = {"short": 8, "ceiling": 8 * CEILING_BLOCKS}

# Rough cost per run, measured on the Qwen smoke run (12 Oct 2026).
TOKENS_PER_RUN = {"short": 2_100, "ceiling": 11_500}

OUT_DIR = Path("data/main")


def seed_for(shj_type, run_number):
    return 100_000 + 1000 * shj_type + run_number


def provider_for(model_name):
    return "groq" if model_name.startswith("qwen/") else "google"


def plan(provider):
    """Every run in the design, in the order it will be collected."""
    jobs = []
    for run_number in range(1, RUNS_PER_TYPE + 1):
        for shj_type in TYPES:
            kind = ("ceiling" if run_number <= CEILING_RUNS[provider][shj_type]
                    else "short")
            jobs.append({"shj_type": shj_type, "run_number": run_number,
                         "kind": kind, "seed": seed_for(shj_type, run_number)})
    return jobs


def load(out_path):
    if not out_path.exists():
        return pd.DataFrame()
    return pd.read_csv(out_path)


def drop_partial_runs(out_path):
    """Delete rows of any run that stopped before its last trial."""
    df = load(out_path)
    if df.empty:
        return 0
    expected = df.run_kind.map(TRIALS)
    sizes = df.groupby("run_id").run_id.transform("size")
    kept = df[sizes == expected]
    dropped = len(df) - len(kept)
    if dropped:
        kept.to_csv(out_path, index=False)
    return dropped


def completed_seeds(out_path):
    df = load(out_path)
    if df.empty:
        return set()
    done = set()
    for (seed, kind), run in df.groupby(["seed", "run_kind"]):
        for _, rows in run.groupby("run_id"):
            if len(rows) == TRIALS[kind]:
                done.add(int(seed))
    return done


def make_responder(model_name, provider):
    if provider == "groq":
        from qwen_responder import make_qwen_responder
        return make_qwen_responder(model_name, temperature=TEMPERATURE,
                                   verbose=True)
    from gemini_responder import make_gemini_responder
    return make_gemini_responder(model_name, temperature=TEMPERATURE,
                                 verbose=True)


def status(model_name, provider, out_path, remaining):
    total = plan(provider)
    done = len(total) - len(remaining)
    print(f"{model_name} -> {out_path}")
    print(f"runs done {done}/{len(total)}\n")
    print("type   short done/planned   ceiling done/planned")
    finished = {(j["shj_type"], j["run_number"]) for j in total} - {
        (j["shj_type"], j["run_number"]) for j in remaining}
    for t in TYPES:
        row = []
        for kind in ("short", "ceiling"):
            planned = [j for j in total if j["shj_type"] == t and j["kind"] == kind]
            got = [j for j in planned if (t, j["run_number"]) in finished]
            row.append(f"{len(got):>4}/{len(planned):<4}")
        print(f"{t:>4}   {row[0]:>15}   {row[1]:>18}")
    requests = sum(TRIALS[j["kind"]] for j in remaining)
    tokens = sum(TOKENS_PER_RUN[j["kind"]] for j in remaining)
    print(f"\nleft: {len(remaining)} runs, about {requests} requests", end="")
    if provider == "groq":
        print(f", about {tokens / 1000:.0f}k tokens "
              f"(~{tokens / 190_000:.1f} days at the 200k/day limit)")
    else:
        print(f" (~{requests / 480:.1f} days at the 500/day limit)")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("status", "run"):
        raise SystemExit(__doc__)
    mode, model_name = sys.argv[1], sys.argv[2]
    provider = provider_for(model_name)
    short = model_name.split("/")[-1]
    out_path = OUT_DIR / f"main_{short}.csv"

    if mode == "status":
        # Read-only: never edits the file, so it is safe to run while a
        # collection for the same model is still writing to it.
        done = completed_seeds(out_path)
        status(model_name, provider, out_path,
               [j for j in plan(provider) if j["seed"] not in done])
        return

    dropped = drop_partial_runs(out_path)
    if dropped:
        print(f"Removed {dropped} rows from a run that was cut off.\n")

    done = completed_seeds(out_path)
    remaining = [j for j in plan(provider) if j["seed"] not in done]
    if not remaining:
        print("All runs complete. Nothing left to collect.")
        return

    responder = make_responder(model_name, provider)
    print(f"{len(remaining)} runs left -> {out_path}\n")
    for i, job in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] Type {job['shj_type']}, run "
              f"{job['run_number']} ({job['kind']}), seed {job['seed']}")
        try:
            run_one(responder, shj_type=job["shj_type"], seed=job["seed"],
                    out_path=out_path, model=short, provider=provider,
                    temperature=TEMPERATURE,
                    instruction_style=INSTRUCTION_STYLE,
                    surface_scheme=SURFACE_SCHEME,
                    n_blocks=1 if job["kind"] == "short" else CEILING_BLOCKS,
                    run_kind=job["kind"], run_number=job["run_number"],
                    verbose=job["kind"] == "ceiling")
        except RuntimeError as e:
            print(f"\nStopped: {e}")
            print("This is most likely the daily limit. Nothing is lost: "
                  "the unfinished run will be redone next time.")
            print(f"Rerun tomorrow:  python src/main_study.py run {model_name}")
            break
    else:
        print("\nAll runs complete.")

    print()
    status(model_name, provider,
           out_path, [j for j in plan(provider)
                      if j["seed"] not in completed_seeds(out_path)])


if __name__ == "__main__":
    main()

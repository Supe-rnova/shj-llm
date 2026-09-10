"""
run_pilot.py

Smoke test and pilot runner.

    python src/run_pilot.py smoke <model>
    python src/run_pilot.py pilot <model>

Smoke: one Type I run. Pilot: Types I and VI, 10 runs each.
Seeds are fixed and derived from type and run number, so any run is
reproducible and re-running appends without colliding.
"""

import sys
from datetime import date
from pathlib import Path
import pandas as pd

from gemini_responder import make_gemini_responder
from runner import run_one

TEMPERATURE = 1.0
INSTRUCTION_STYLE = "neutral"
SURFACE_SCHEME = "minerals"
RUNS_PER_TYPE = 10


def seed_for(shj_type, run_number):
    """Fixed, documented, collision-free."""
    return 1000 * shj_type + run_number

def drop_partial_runs(out_path):
    """Delete rows from runs that didn't reach 64 trials."""
    if not out_path.exists():
        return 0
    df = pd.read_csv(out_path)
    sizes = df.groupby("run_id").size()
    complete = sizes[sizes == 64].index
    kept = df[df.run_id.isin(complete)]
    dropped = len(df) - len(kept)
    if dropped:
        kept.to_csv(out_path, index=False)
    return dropped


def completed_seeds(out_path):
    """Seeds that already have a full 64-trial run in this file."""
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path)
    sizes = df.groupby(["seed", "run_id"]).size()
    return {seed for (seed, _), n in sizes.items() if n == 64}


def main():
    mode = sys.argv[1]
    model_name = sys.argv[2]
    short = model_name.split("/")[-1]

    responder = make_gemini_responder(
        model_name, temperature=TEMPERATURE, verbose=True)

    if mode == "smoke":
        jobs = [(1, 0)]
        out = Path(f"data/pilot/smoke_{short}_{date.today()}.csv")
    elif mode == "pilot":
        jobs = [(t, n) for t in (1, 6) for n in range(1, RUNS_PER_TYPE + 1)]
        out = Path(f"data/pilot/pilot_{short}.csv")
    else:
        raise SystemExit("mode must be 'smoke' or 'pilot'")


    dropped = drop_partial_runs(out)
    if dropped:
        print(f"Removed {dropped} rows from an incomplete run.")

    done = completed_seeds(out)
    jobs = [(t, n) for (t, n) in jobs if seed_for(t, n) not in done]
    if done:
        print(f"Resuming: {len(done)} run(s) already complete, skipping those.")

    if not jobs:
        print("Nothing left to run.")
        return


    print(f"{len(jobs)} run(s), {len(jobs) * 64} requests -> {out}")
    print(f"estimated wall clock: {len(jobs) * 64 * 6 / 60:.0f} minutes\n")

    for i, (shj_type, run_number) in enumerate(jobs, start=1):
        seed = seed_for(shj_type, run_number)
        print(f"[{i}/{len(jobs)}] Type {shj_type}, run {run_number}, seed {seed}")
        run_one(
            responder, shj_type=shj_type, seed=seed, out_path=out,
            model=short, provider="google",
            temperature=TEMPERATURE,
            instruction_style=INSTRUCTION_STYLE,
            surface_scheme=SURFACE_SCHEME,
            verbose=True,
        )

    print(f"\nDone. {out}")


if __name__ == "__main__":
    main()
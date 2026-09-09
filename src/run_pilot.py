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

from gemini_responder import make_gemini_responder
from runner import run_one

TEMPERATURE = 1.0
INSTRUCTION_STYLE = "neutral"
SURFACE_SCHEME = "minerals"
RUNS_PER_TYPE = 10


def seed_for(shj_type, run_number):
    """Fixed, documented, collision-free."""
    return 1000 * shj_type + run_number


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
        out = Path(f"data/pilot/pilot_{short}_{date.today()}.csv")
    else:
        raise SystemExit("mode must be 'smoke' or 'pilot'")

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
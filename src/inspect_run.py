"""Checks on a completed run's CSV. Pipeline section 1.6."""

import json
import sys
import pandas as pd
from structures import ITEMS, STRUCTURES

path = sys.argv[1] if len(sys.argv) > 1 else "data/pilot/fake_test.csv"
df = pd.read_csv(path)

for run_id, run in df.groupby("run_id"):
    print("=" * 60)
    print(f"run {run_id[:8]}  model={run.model.iloc[0]}  "
          f"type={run.shj_type.iloc[0]}  seed={run.seed.iloc[0]}")

    print(f"  rows                 : {len(run)}  (expect 64)")
    print(f"  blocks               : {sorted(run.block.unique())}")
    print(f"  trial_index 1..64    : {list(run.trial_index) == list(range(1, 65))}")

    per_block = run.groupby("block").item_bits.nunique()
    print(f"  8 distinct items/block: {(per_block == 8).all()}")

    # Verify category_true against the structure definition, every row.
    shj_type = run.shj_type.iloc[0]
    label_to_group = json.loads(run.label_to_group.iloc[0])
    mismatches = 0
    for _, r in run.iterrows():
        item = tuple(int(b) for b in f"{r.item_bits:03d}")
        expected = label_to_group[str(STRUCTURES[shj_type][ITEMS.index(item)])]
        if expected != r.category_true:
            mismatches += 1
    print(f"  category_true correct : {mismatches == 0}  ({mismatches} bad rows)")

    parsed = run[run.parse_ok == True]
    print(f"  parse_ok rate         : {len(parsed)}/{len(run)}")
    if len(parsed):
        print(f"  accuracy (parsed only): {parsed.correct.mean():.3f}")

    print(f"  prompt_tokens 1 -> 64 : {run.prompt_tokens.iloc[0]} -> "
          f"{run.prompt_tokens.iloc[-1]}")
    print()
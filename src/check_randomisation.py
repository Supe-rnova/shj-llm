"""Quick check that the per-run randomisation is actually uniform."""

import random
from collections import Counter
from stimuli import make_stimulus

perms = Counter()
groups = Counter()
values = Counter()

for seed in range(600):
    s = make_stimulus(random.Random(seed))
    perms[tuple(s.dim_to_feature[d] for d in (1, 2, 3))] += 1
    groups[s.label_to_group[0]] += 1
    values[s.value_map["lustre"]["0"]] += 1

print("Dimension mappings (expect 6 kinds, ~100 each):")
for k, v in sorted(perms.items()):
    print(f"  {k}  {v}")

print("\nLabel 0 -> group (expect ~300/300):")
for k, v in groups.items():
    print(f"  {k}  {v}")

print("\nLustre bit 0 (expect ~300/300):")
for k, v in values.items():
    print(f"  {k}  {v}")
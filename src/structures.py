"""
structures.py

The six Shepard, Hovland & Jenkins (1961) category structures.

STRUCTURE LAYER ONLY. Nothing in this file knows about lustre, banding,
grain, or any other physical feature. Items are 3-bit tuples and the six
category types are defined purely over those bits. The mapping onto
physical features lives in stimuli.py and is randomised per run.

Labels: 1 = Category A, 0 = Category B. Which one becomes "Group Alpha"
in the actual prompt is decided per run, in stimuli.py, and logged.
"""

from itertools import product

# The eight items, in a fixed canonical order.
# Position i in this tuple means the same item everywhere in the project.
ITEMS = tuple(product((0, 1), repeat=3))
# ((0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1))


def _labels_from_set(category_a_items):
    """Turn a set of items into a tuple of 8 labels in ITEMS order."""
    return tuple(1 if item in category_a_items else 0 for item in ITEMS)


# --- Types with clean computable definitions -------------------------

# Type I: one dimension decides. Here, dimension 1 (the first bit).
TYPE_I = tuple(item[0] for item in ITEMS)

# Type II: exclusive-or on dimensions 1 and 2. Dimension 3 irrelevant.
TYPE_II = tuple(item[0] ^ item[1] for item in ITEMS)

# Type IV: majority rule. Label 1 when at least two bits are 1.
TYPE_IV = tuple(1 if sum(item) >= 2 else 0 for item in ITEMS)

# Type VI: parity. Label 1 when an odd number of bits are 1.
TYPE_VI = tuple(sum(item) % 2 for item in ITEMS)


# --- Types III and V: no one-line definition -------------------------
# Both are a single-dimension rule with two exceptions. They differ only
# in how far apart those two exception items sit on the cube:
#
#   Type IV : exceptions differ on all three dimensions  (antipodal)
#   Type V  : exceptions differ on ONE dimension - the rule dimension
#   Type III: exceptions differ on TWO dimensions
#
# Below, both use the rule "first bit is 0 -> Category A", which alone
# gives {000, 001, 010, 011}, then swap one item from each side.
#
# THE III/V ASSIGNMENT BELOW IS NOT YET VERIFIED. The test proves these
# are the two distinct size-24 classes, but not which is which.
# Check against SHJ (1961) Figure 3, page 4. See note at end of file.

# Swap 000 <-> 101: the pair differs on dimensions 1 and 3.
TYPE_III = _labels_from_set({(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1)})

# Swap 000 <-> 100: the pair differs on dimension 1 only.
TYPE_V = _labels_from_set({(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)})


STRUCTURES = {
    1: TYPE_I,
    2: TYPE_II,
    3: TYPE_III,
    4: TYPE_IV,
    5: TYPE_V,
    6: TYPE_VI,
}


def category_of(shj_type, item):
    """Return the label (0 or 1) for one item under one SHJ type."""
    return STRUCTURES[shj_type][ITEMS.index(item)]


def show(shj_type):
    """Print a structure as eight lines, for eyeballing against Figure 3."""
    labels = STRUCTURES[shj_type]
    print(f"Type {shj_type}")
    for item, label in zip(ITEMS, labels):
        bits = "".join(str(b) for b in item)
        print(f"  {bits}  ->  {'A' if label else 'B'}")


if __name__ == "__main__":
    for t in range(1, 7):
        show(t)
        print()


# --- Note on Types III and V -----------------------------------------
# Both sit in equivalence classes of size 24, so the class-size check
# cannot separate them, and neither is linearly separable, so that
# cannot either. The test distinguishes them by exception geometry
# (distance 1 vs distance 2), which proves they are the two different
# classes but not which label belongs to which.
#
# My assignment above rests on Type V being the canonical
# "rule plus exception" structure in the later modelling literature
# (Nosofsky's RULEX): its two exceptions differ only on the rule
# dimension, so the exception is itself statable as a rule.
#
# I am not fully confident in this. Confirm against SHJ Figure 3.
# The pilot uses only Types I and VI, so this does not block you.
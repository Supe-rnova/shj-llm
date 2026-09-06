"""
test_structures.py

Proves that the six structures in structures.py really are the six SHJ
types. Run from the project root with:

    python src/test_structures.py

The core argument: there are 70 ways to split eight items 4/4. Renaming
dimensions, flipping values, or swapping the group labels does not change
how hard a problem is to learn. Under those symmetries the 70 collapse to
exactly six classes, with sizes 6, 6, 24, 8, 24, 2 (SHJ p. 34).
"""

from itertools import combinations, permutations, product
from structures import ITEMS, STRUCTURES

EXPECTED_CLASS_SIZES = {1: 6, 2: 6, 3: 24, 4: 8, 5: 24, 6: 2}
EXPECTED_RELEVANT_DIMS = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}


def transform(labels, perm, flips, swap):
    """Apply one symmetry operation to a labelling.

    perm  : a reordering of the three dimensions
    flips : which dimensions have their 0 and 1 exchanged
    swap  : whether Category A and Category B trade names
    """
    new = [None] * 8
    for idx, item in enumerate(ITEMS):
        permuted = tuple(item[p] for p in perm)
        flipped = tuple(b ^ f for b, f in zip(permuted, flips))
        new[ITEMS.index(flipped)] = labels[idx]
    if swap:
        new = [1 - v for v in new]
    return tuple(new)


def orbit(labels):
    """Every labelling reachable from this one. This is its class."""
    reachable = set()
    for perm in permutations(range(3)):
        for flips in product((0, 1), repeat=3):
            for swap in (False, True):
                reachable.add(transform(labels, perm, flips, swap))
    return frozenset(reachable)


def all_44_splits():
    """All 70 ways of splitting the eight items four and four."""
    return [
        tuple(1 if i in subset else 0 for i in range(8))
        for subset in combinations(range(8), 4)
    ]


def relevant_dimensions(labels):
    """Which dimensions actually change the label when flipped."""
    relevant = []
    for d in range(3):
        for idx, item in enumerate(ITEMS):
            neighbour = list(item)
            neighbour[d] ^= 1
            if labels[idx] != labels[ITEMS.index(tuple(neighbour))]:
                relevant.append(d + 1)
                break
    return relevant


def is_linearly_separable(labels):
    """Can one straight cut through the cube separate A from B?"""
    weights = range(-4, 5)
    for w in product(weights, repeat=3):
        score = [w[0] * i[0] + w[1] * i[1] + w[2] * i[2] for i in ITEMS]
        a = [s for s, l in zip(score, labels) if l == 1]
        b = [s for s, l in zip(score, labels) if l == 0]
        if min(a) > max(b):
            return True
    return False


def best_single_dimension_rule(labels):
    """Find the single-dimension rule that gets the most items right.

    Returns the list of items it gets wrong. Types III, IV and V are all
    'one dimension plus two exceptions'; this returns those exceptions.
    """
    best = None
    for d in range(3):
        for polarity in (0, 1):
            predicted = tuple(1 if item[d] == polarity else 0 for item in ITEMS)
            wrong = [ITEMS[i] for i in range(8) if predicted[i] != labels[i]]
            if best is None or len(wrong) < len(best):
                best = wrong
    return best


def hamming(a, b):
    """How many dimensions two items differ on."""
    return sum(x != y for x, y in zip(a, b))



def main():
    failures = []

    def check(condition, message):
        print(f"  {'PASS' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    # 1. Every structure splits the items 4/4.
    print("\n[1] Four-four splits")
    for t, labels in STRUCTURES.items():
        check(sum(labels) == 4, f"Type {t} has four items in each category")

    # 2. Each type depends on the right number of dimensions.
    print("\n[2] Relevant dimensions")
    for t, labels in STRUCTURES.items():
        n = len(relevant_dimensions(labels))
        check(n == EXPECTED_RELEVANT_DIMS[t],
              f"Type {t} depends on {n} dimension(s), expected {EXPECTED_RELEVANT_DIMS[t]}")

    # 3. The 70 splits collapse to exactly six classes.
    print("\n[3] The 70 to 6 collapse")
    splits = all_44_splits()
    check(len(splits) == 70, f"generated {len(splits)} four-four splits, expected 70")

    classes = {orbit(s) for s in splits}
    check(len(classes) == 6, f"found {len(classes)} equivalence classes, expected 6")

    sizes = sorted(len(c) for c in classes)
    check(sizes == [2, 6, 6, 8, 24, 24], f"class sizes {sizes}, expected [2, 6, 6, 8, 24, 24]")
    check(sum(sizes) == 70, f"class sizes sum to {sum(sizes)}, expected 70")

    # 4. Our six structures land in six different classes, of the right sizes.
    print("\n[4] Our structures against the classes")
    orbits = {t: orbit(labels) for t, labels in STRUCTURES.items()}
    check(len({frozenset(o) for o in orbits.values()}) == 6,
          "the six structures fall in six distinct classes")
    for t, o in orbits.items():
        check(len(o) == EXPECTED_CLASS_SIZES[t],
              f"Type {t} sits in a class of size {len(o)}, expected {EXPECTED_CLASS_SIZES[t]}")

    # 5. Linear separability. Only Types I and IV are separable.
    print("\n[5] Linear separability")
    expected_separable = {1: True, 2: False, 3: False, 4: True, 5: False, 6: False}
    for t, labels in STRUCTURES.items():
        sep = is_linearly_separable(labels)
        check(sep == expected_separable[t],
              f"Type {t} linearly separable = {sep}, expected {expected_separable[t]}")


    # 6. Exception geometry. This is what separates III, IV and V.
    print("\n[6] Exception geometry (III vs IV vs V)")
    expected_n_exceptions = {1: 0, 2: 4, 3: 2, 4: 2, 5: 2, 6: 4}
    expected_distance = {3: 2, 4: 3, 5: 1}
    for t, labels in STRUCTURES.items():
        wrong = best_single_dimension_rule(labels)
        check(len(wrong) == expected_n_exceptions[t],
              f"Type {t}: best single-dimension rule misses {len(wrong)} items, "
              f"expected {expected_n_exceptions[t]}")
        if t in expected_distance:
            d = hamming(wrong[0], wrong[1])
            check(d == expected_distance[t],
                  f"Type {t}: its two exceptions differ on {d} dimension(s), "
                  f"expected {expected_distance[t]}")


    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED. Do not proceed.")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")
    print("Still confirm Types III and V by eye against SHJ Figure 3.")


if __name__ == "__main__":
    main()
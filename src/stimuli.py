"""
stimuli.py

SURFACE LAYER. Turns 3-bit structural items into readable text.

structures.py knows only about bits. This file owns the mapping onto
physical features, and randomises that mapping per run so that no
single physical feature is permanently tied to a structural dimension.

Every randomised choice is returned in the Stimulus object so it can be
written into the trial log. Kurtz (2013) found subtype effects - which
physical feature happened to be irrelevant - larger than the Type II /
Type IV effect he was studying. Randomising without logging discards
that variance.
"""

import json
import random
from dataclasses import dataclass

# The default surface scheme: mineral samples.
# Three properties, two values each, in the order they appear in item text.
SURFACE_SCHEMES = {
    "minerals": {
        "domain_noun": "mineral sample",
        "domain_plural": "mineral samples",
        "features": [
            ("lustre", ("dull", "glassy")),
            ("banding", ("banded", "unbanded")),
            ("grain", ("coarse", "fine")),
        ],
    }
}

GROUP_NAMES = ("Group Alpha", "Group Beta")


@dataclass
class Stimulus:
    """Everything the surface layer decided for one run.

    Created once per run. Held constant for all 64 trials of that run.
    Every field here goes into the log.
    """
    scheme_name: str
    domain_noun: str
    domain_plural: str
    dim_to_feature: dict      # structural dim (1,2,3) -> feature name
    value_map: dict           # feature name -> {"0": word, "1": word}
    label_to_group: dict      # structural label (0,1) -> group name
    feature_order: list       # order features appear in item text

    def item_text(self, item):
        """Render a 3-bit item as an adjective string.

        (0, 1, 0) -> "a dull, unbanded, coarse sample"
        """
        adjectives = []
        for feature in self.feature_order:
            dim = next(d for d, f in self.dim_to_feature.items() if f == feature)
            bit = item[dim - 1]
            adjectives.append(self.value_map[feature][str(bit)])
        article = "an" if adjectives[0][0] in "aeiou" else "a"
        return f"{article} {', '.join(adjectives)} {self.domain_noun}"

    def group_of(self, label):
        """Structural label 0 or 1 -> the group name shown to the model."""
        return self.label_to_group[label]

    def log_fields(self):
        """The columns this object contributes to every trial row."""
        return {
            "surface_scheme": self.scheme_name,
            "dim_to_feature": json.dumps(self.dim_to_feature),
            "value_map": json.dumps(self.value_map),
            "label_to_group": json.dumps(self.label_to_group),
        }


def make_stimulus(rng, scheme_name="minerals"):
    """Build one run's surface mapping. Three independent randomisations.

    rng: a random.Random instance, seeded per run, so the whole mapping
         is reproducible from the seed alone.
    """
    scheme = SURFACE_SCHEMES[scheme_name]
    feature_names = [name for name, _ in scheme["features"]]

    # 1. Which structural dimension becomes which physical feature.
    shuffled = feature_names[:]
    rng.shuffle(shuffled)
    dim_to_feature = {1: shuffled[0], 2: shuffled[1], 3: shuffled[2]}

    # 2. Which value of each feature counts as bit 0.
    value_map = {}
    for name, values in scheme["features"]:
        pair = list(values)
        if rng.random() < 0.5:
            pair.reverse()
        value_map[name] = {"0": pair[0], "1": pair[1]}

    # 3. Which structural label becomes Group Alpha.
    groups = list(GROUP_NAMES)
    if rng.random() < 0.5:
        groups.reverse()
    label_to_group = {0: groups[0], 1: groups[1]}

    return Stimulus(
        scheme_name=scheme_name,
        domain_noun=scheme["domain_noun"],
        domain_plural=scheme["domain_plural"],
        dim_to_feature=dim_to_feature,
        value_map=value_map,
        label_to_group=label_to_group,
        feature_order=feature_names,
    )


def block_order(rng, items):
    """One block: all eight items, freshly shuffled. Returns a new list."""
    order = list(items)
    rng.shuffle(order)
    return order


if __name__ == "__main__":
    from structures import ITEMS, STRUCTURES

    for demo_seed in (1, 2):
        rng = random.Random(demo_seed)
        stim = make_stimulus(rng)
        print(f"--- seed {demo_seed} ---")
        print("dim -> feature :", stim.dim_to_feature)
        print("value map      :", stim.value_map)
        print("label -> group :", stim.label_to_group)
        print("Type I items:")
        for item in ITEMS:
            label = STRUCTURES[1][ITEMS.index(item)]
            bits = "".join(str(b) for b in item)
            print(f"  {bits}  {stim.item_text(item):<45} {stim.group_of(label)}")
        print()
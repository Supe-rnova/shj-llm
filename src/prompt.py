"""
prompt.py

Builds the exact text sent to the model on each trial.

Two instruction styles, differing by four words. That difference is the
whole of Kurtz et al. (2013)'s manipulation, and it is the choice
Jagadish et al. (2024) made without discussing it - they used rule
language throughout their Appendix G.4 SHJ runs.

The pilot uses NEUTRAL only. RULE is here so the wording is fixed and
version-controlled now, not improvised in September.

No conversation object is ever reused. The full prompt is rebuilt from
scratch on every call, from a trial history held in the caller's own
data structure. See build_prompt().
"""

INSTRUCTIONS = {
    "neutral": (
        "In this experiment, you will be shown examples of {domain_plural}. "
        "Each {domain_noun} has three different properties. "
        "Your job is to learn to tell whether each example belongs to "
        "{group_a} or {group_b}. "
        "As you are shown each example, you will be asked to make a category "
        "judgment and then you will receive feedback. At first you will have "
        "to guess, but you will gain experience as you go along. "
        "Try your best to gain mastery of the {group_a} and {group_b} categories."
    ),
    "rule": (
        "In this experiment, you will be shown examples of {domain_plural}. "
        "Each {domain_noun} has three different properties. "
        "Your job is to learn a rule based on the {domain_noun} properties "
        "that allows you to tell whether each example belongs to "
        "{group_a} or {group_b}. "
        "As you are shown each example, you will be asked to make a category "
        "judgment and then you will receive feedback. At first you will have "
        "to guess, but you will gain experience as you go along. "
        "Try your best to gain mastery of the {group_a} and {group_b} categories."
    ),
}


def build_instructions(stimulus, instruction_style):
    """Fill the instruction template with this run's surface details."""
    if instruction_style not in INSTRUCTIONS:
        raise ValueError(
            f"unknown instruction_style {instruction_style!r}; "
            f"expected one of {sorted(INSTRUCTIONS)}"
        )
    return INSTRUCTIONS[instruction_style].format(
        domain_noun=stimulus.domain_noun,
        domain_plural=stimulus.domain_plural,
        group_a=stimulus.group_of(0),
        group_b=stimulus.group_of(1),
    )


def history_line(trial_number, item_text, chosen_group, correct_group):
    """One completed trial, as the model will read it back.

    Mirrors the Jagadish et al. format so results stay comparable.
    """
    return (
        f"- In trial {trial_number}, you picked {chosen_group} for "
        f"{item_text} and {correct_group} was correct."
    )


def build_prompt(stimulus, instruction_style, history, current_item):
    """Assemble the complete prompt for one trial.

    history: a list of dicts, one per COMPLETED trial, each with keys
             trial_number, item_text, chosen_group, correct_group.
             Held by the caller. Never a conversation object.
    """
    parts = [build_instructions(stimulus, instruction_style), ""]

    if history:
        for h in history:
            parts.append(
                history_line(
                    h["trial_number"],
                    h["item_text"],
                    h["chosen_group"],
                    h["correct_group"],
                )
            )
        parts.append("")

    parts.append(
        f"What group would {stimulus.item_text(current_item)} belong to?"
    )
    parts.append('(Give the answer in the form "Group <your answer>".)')
    parts.append("")
    parts.append("Group")

    return "\n".join(parts)


def parse_response(raw, stimulus):
    """Extract a group name from the model's output.

    Returns (parsed_group_or_None, parse_ok). Deliberately strict: a
    response mentioning both groups, or neither, is a parse failure, not
    a guess. Silent mis-parsing would corrupt the accuracy measure.
    """
    if raw is None:
        return None, False

    text = raw.strip().lower()
    alpha_seen = "alpha" in text
    beta_seen = "beta" in text

    if alpha_seen and not beta_seen:
        matched = "Group Alpha"
    elif beta_seen and not alpha_seen:
        matched = "Group Beta"
    else:
        return None, False

    if matched not in (stimulus.group_of(0), stimulus.group_of(1)):
        return None, False
    return matched, True


if __name__ == "__main__":
    import random
    from structures import ITEMS, STRUCTURES
    from stimuli import make_stimulus

    rng = random.Random(1)
    stim = make_stimulus(rng)
    shj_type = 1

    def group_for(item):
        return stim.group_of(STRUCTURES[shj_type][ITEMS.index(item)])

    print("=" * 68)
    print("TRIAL 1 - no history yet")
    print("=" * 68)
    print(build_prompt(stim, "neutral", [], ITEMS[0]))

    history = []
    for n, item in enumerate(ITEMS[:3], start=1):
        history.append({
            "trial_number": n,
            "item_text": stim.item_text(item),
            "chosen_group": group_for(item),
            "correct_group": group_for(item),
        })

    print()
    print("=" * 68)
    print("TRIAL 4 - three completed trials in history")
    print("=" * 68)
    print(build_prompt(stim, "neutral", history, ITEMS[3]))

    print()
    print("=" * 68)
    print("THE FOUR-WORD DIFFERENCE")
    print("=" * 68)
    print("NEUTRAL:", build_instructions(stim, "neutral")[:230])
    print()
    print("RULE   :", build_instructions(stim, "rule")[:230])

    print()
    print("=" * 68)
    print("PARSER")
    print("=" * 68)
    for raw in [" Alpha", "Group Beta", "Alpha.", "I think Group Alpha",
                "Alpha or Beta", "unclear", "", None]:
        print(f"  {raw!r:25} -> {parse_response(raw, stim)}")
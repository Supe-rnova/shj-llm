"""
runner.py

The trial loop. Runs one SHJ problem end to end and writes one CSV row
per trial, incrementally.

Two non-negotiables live here:

  Fresh context every run. The prompt is rebuilt from scratch on every
  call, from a history list held in this file. No conversation object
  is ever reused, across trials or across runs.

  Incremental logging. Each row is written and flushed as soon as the
  trial completes. If a run dies at trial 50 you keep 49 rows.

The model is passed in as a function, so the loop can be tested against
a fake responder with no API and no cost.
"""

import csv
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from structures import ITEMS, STRUCTURES
from stimuli import make_stimulus, block_order
from prompt import build_prompt, parse_response

BLOCKS = 8
ITEMS_PER_BLOCK = 8

FIELDNAMES = [
    "run_id", "timestamp", "model", "provider", "temperature", "seed",
    "surface_scheme", "instruction_style", "shj_type",
    "block", "trial_in_block", "trial_index",
    "item_bits", "dim_to_feature", "value_map", "label_to_group",
    "item_text", "category_true",
    "response_raw", "response_parsed", "parse_ok", "correct",
    "prompt_tokens", "completion_tokens", "latency_ms",
    "attempts", "rejected_raw",
    "run_kind", "run_number",
]
# run_kind / run_number: which kind of run this row belongs to ("full" =
# the 8-block pilot design; "short" = 1 block; "ceiling" = 3 blocks) and
# its number within its type. Set by main_study.py.
# attempts / rejected_raw: how many times the question was asked, and the
# verbatim replies that were rejected as unreadable (qwen_responder.py).
# Responders that never retry (Gemini, fakes) log attempts=1.


def run_one(responder, shj_type, seed, out_path,
            model="fake", provider="fake", temperature=1.0,
            instruction_style="neutral", surface_scheme="minerals",
            verbose=False, n_blocks=BLOCKS, run_kind="full",
            run_number=None):
    """Run one SHJ problem: n_blocks x 8 items (default 8 x 8 = 64 trials).

    n_blocks < 8 stops the run early. Block 1 is identical whatever
    n_blocks is: the same seed gives the same mapping and the same
    block-1 order, because later blocks are drawn only after it.

    responder: a function taking a prompt string and returning
               (raw_text, prompt_tokens, completion_tokens), optionally
               with a fourth element: a dict of extra log columns
               (attempts, rejected_raw).
    Returns the run_id.
    """
    run_id = str(uuid.uuid4())
    rng = random.Random(seed)
    stimulus = make_stimulus(rng, surface_scheme)
    labels = STRUCTURES[shj_type]
    surface_fields = stimulus.log_fields()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()

    history = []
    trial_index = 0

    # Appending to a file written before the attempts / rejected_raw
    # columns existed: keep that file's own header so columns stay aligned.
    fieldnames = FIELDNAMES
    if not new_file:
        with open(out_path, newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), None)
        if header:
            fieldnames = header

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()

        for block in range(1, n_blocks + 1):
            for trial_in_block, item in enumerate(block_order(rng, ITEMS), start=1):
                trial_index += 1

                true_label = labels[ITEMS.index(item)]
                true_group = stimulus.group_of(true_label)
                item_text = stimulus.item_text(item)

                prompt = build_prompt(stimulus, instruction_style, history, item)

                t0 = time.perf_counter()
                result = responder(prompt)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                raw, p_tok, c_tok = result[:3]
                extra = (result[3] if len(result) > 3
                         else {"attempts": 1, "rejected_raw": ""})

                parsed, parse_ok = parse_response(raw, stimulus)
                correct = (parsed == true_group) if parse_ok else None

                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "provider": provider,
                    "temperature": temperature,
                    "seed": seed,
                    "instruction_style": instruction_style,
                    "shj_type": shj_type,
                    "block": block,
                    "trial_in_block": trial_in_block,
                    "trial_index": trial_index,
                    "item_bits": "".join(str(b) for b in item),
                    "item_text": item_text,
                    "category_true": true_group,
                    "response_raw": raw,
                    "response_parsed": parsed,
                    "parse_ok": parse_ok,
                    "correct": correct,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "latency_ms": latency_ms,
                }
                row.update(surface_fields)
                row.update(extra)
                row["run_kind"] = run_kind
                row["run_number"] = run_number

                writer.writerow(row)
                fh.flush()

                # Feedback goes into history only AFTER the row is written.
                history.append({
                    "trial_number": trial_index,
                    "item_text": item_text,
                    "chosen_group": parsed if parse_ok else "no clear answer",
                    "correct_group": true_group,
                })

                if verbose and trial_index % 8 == 0:
                    done = sum(1 for h in history if h["chosen_group"] != "no clear answer")
                    print(f"  block {block} complete, {done}/{trial_index} parsed")

    return run_id


# --- Fake responders, for testing the loop with no API ----------------

def make_perfect_responder(shj_type, stimulus):
    """Always correct. Tests that the loop scores accuracy properly."""
    labels = STRUCTURES[shj_type]

    def responder(prompt):
        for item in ITEMS:
            if stimulus.item_text(item) in prompt.split("What group would")[-1]:
                label = labels[ITEMS.index(item)]
                return stimulus.group_of(label), len(prompt) // 4, 2
        raise RuntimeError("perfect responder could not identify the current item")

    return responder


def make_random_responder(seed=0):
    """Coin flip. Should score near 50%."""
    rng = random.Random(seed)

    def responder(prompt):
        return rng.choice(["Alpha", "Beta"]), len(prompt) // 4, 2

    return responder


def make_messy_responder(seed=0):
    """Sometimes unparseable. Tests that parse failures are handled."""
    rng = random.Random(seed)
    options = ["Alpha", "Beta", "Alpha or Beta", "I'm not sure", ""]

    def responder(prompt):
        return rng.choice(options), len(prompt) // 4, 3

    return responder


if __name__ == "__main__":
    out = Path("data/pilot/fake_test.csv")
    if out.exists():
        out.unlink()

    print("Random responder, Type I, seed 101")
    run_one(make_random_responder(seed=1), shj_type=1, seed=101,
            out_path=out, model="fake-random", verbose=True)

    print("\nMessy responder, Type VI, seed 102")
    run_one(make_messy_responder(seed=2), shj_type=6, seed=102,
            out_path=out, model="fake-messy", verbose=True)

    print(f"\nWrote {out}")
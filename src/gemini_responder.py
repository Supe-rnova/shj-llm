"""
gemini_responder.py

The only component that talks to an API.

Uses PREFILL: the prompt's trailing "Group" is moved into the model's
own turn, so the model continues its own sentence rather than starting
a reply. This reproduces the mechanism in Jagadish et al. App. G.4,
whose prompt ends "Assistant: Category".

Verified 9 Sep 2026 on gemini-3.5-flash and gemini-3.1-flash-lite:
one output token, finish_reason STOP, no preamble, no refusal.
Without prefill both models produced preamble; flash-lite refused to
guess on trial 1 entirely.

thinking_budget=0 is required. Gemini 3.x models otherwise spend
hundreds of tokens deliberating before answering (3.6-flash: 292
thinking tokens for a one-token answer), which is a different task
from the one Claude-v2 performed.
"""

import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MIN_SECONDS_BETWEEN_CALLS = 6.0
MAX_RETRIES = 5
PREFILL = "Group"


def make_gemini_responder(model_name, temperature=1.0, max_output_tokens=16,
                          min_gap=MIN_SECONDS_BETWEEN_CALLS,
                          thinking_budget=0, verbose=False):
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    state = {"last_call": 0.0}

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )

    def responder(prompt):
        # Split the trailing "Group" off and make it the model's own turn.
        body = prompt.rsplit("\n" + PREFILL, 1)[0]
        contents = [
            types.Content(role="user", parts=[types.Part(text=body)]),
            types.Content(role="model", parts=[types.Part(text=PREFILL)]),
        ]

        elapsed = time.time() - state["last_call"]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                state["last_call"] = time.time()
                resp = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                u = resp.usage_metadata
                return (
                    resp.text,
                    getattr(u, "prompt_token_count", None),
                    getattr(u, "candidates_token_count", None),
                )
            except Exception as e:
                last_error = e
                text = str(e)
                permanent = (any(c in text for c in ("400", "403", "404"))
                             and "429" not in text)
                if permanent:
                    raise RuntimeError(f"permanent error, not retrying: {e}")
                wait = (2 ** attempt) + random.random()
                if verbose:
                    print(f"    attempt {attempt + 1} failed "
                          f"({type(e).__name__}), waiting {wait:.1f}s")
                time.sleep(wait)

        raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_error}")

    return responder


if __name__ == "__main__":
    import sys
    from structures import ITEMS
    from stimuli import make_stimulus
    from prompt import build_prompt, parse_response

    model_name = sys.argv[1]
    stim = make_stimulus(random.Random(1))
    responder = make_gemini_responder(model_name, verbose=True)

    for label, hist in [("trial 1, no history", []),
                        ("trial 4, three history lines", [
                            {"trial_number": n,
                             "item_text": stim.item_text(ITEMS[n - 1]),
                             "chosen_group": "Group Alpha",
                             "correct_group": "Group Alpha"} for n in (1, 2, 3)])]:
        item = ITEMS[len(hist)]
        raw, p, c = responder(build_prompt(stim, "neutral", hist, item))
        print(f"{label:32} raw={raw!r:12} parsed={parse_response(raw, stim)} "
              f"tokens={p}/{c}")
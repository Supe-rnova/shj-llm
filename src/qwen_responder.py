"""
qwen_responder.py

The API-facing component for Qwen 3.8 27B on Groq. Returns the same
(raw_text, prompt_tokens, completion_tokens) as gemini_responder.py, plus
a fourth element with retry information that runner.py logs.

NO PREFILL. Groq does not continue a prefilled assistant message: it
treats it as a finished turn and starts a new reply (tested 12 Oct 2026,
diag_qwen_prefill.py: 0/10 replies continued an "Answer: Group" prefill,
and prefill raised the failure rate from 1/10 to 5/10). So the user
message is the prompt WITHOUT its final "Group" line - exactly the text
Gemini received - and Qwen writes its own reply, guided by the existing
line (Give the answer in the form "Group <your answer>".).
Methods note: same prompt text for both models; Gemini's answer was a
continuation of "Group", Qwen's is a fresh reply.

ANSWER-ONLY INSTRUCTION. Without prefill, Qwen often wrote out its
working ("To determine the group for ... let's analyze") instead of
answering: 8 unreadable replies in the first trials of the 12 Oct smoke
run. That is visible step-by-step reasoning, which the thinking-off
decision excludes. So one sentence, FORMAT_INSTRUCTION, is appended to
the end of Qwen's message only. prompt.py is unchanged, so Gemini's
prompt is exactly as in the pilot. Methods: answer format enforced by
prefill for Gemini, by this sentence for Qwen (Groq cannot prefill).
The sentence names no group, so it cannot favour Alpha or Beta.

ASK AGAIN ON UNREADABLE REPLIES. About 1 in 10 replies is an explanation
instead of an answer. If a reply does not name exactly one group, the
same request is sent again, up to MAX_ATTEMPTS times. Every rejected
reply is kept verbatim and logged (columns attempts, rejected_raw).
After MAX_ATTEMPTS the last reply is returned and logged as a parse
failure, as before.

THINKING OFF: reasoning_effort="none" ("Disable reasoning. The model will
not use any reasoning tokens." - console.groq.com/docs/reasoning). Any
response carrying reasoning raises instead of being logged.

PREVIEW MODEL: Groq lists qwen/qwen3.8-27b as preview, "may be
discontinued at short notice". Record the exact ID; collect promptly.

Key: GROQ_API_KEY in .env (never in code).
"""

import json
import os
import random
import time

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.groq.com/openai/v1/chat/completions"
MIN_SECONDS_BETWEEN_CALLS = 2.5   # stays under 30 requests per minute
MAX_RETRIES = 5                   # for network / rate-limit errors
MAX_ATTEMPTS = 3                  # for unreadable replies
MAX_WAIT_SECONDS = 120            # longer = daily limit; stop instead


class DailyLimitReached(RuntimeError):
    """Raised when Groq's daily request or token limit is used up."""
TRAILING_LINE = "Group"
FORMAT_INSTRUCTION = ("Reply with only the name of the group and nothing "
                      "else. Do not explain.")


def names_one_group(text):
    """Same rule as prompt.parse_response: exactly one of alpha / beta."""
    low = (text or "").lower()
    return ("alpha" in low) != ("beta" in low)


def make_qwen_responder(model_name="qwen/qwen3.8-27b", temperature=1.0,
                        max_output_tokens=16,
                        min_gap=MIN_SECONDS_BETWEEN_CALLS,
                        max_attempts=MAX_ATTEMPTS, verbose=False):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    state = {"last_call": 0.0}

    def one_call(payload):
        """One successful API response, with network/rate-limit retries."""
        elapsed = time.time() - state["last_call"]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        last_error = None
        for attempt in range(MAX_RETRIES):
            state["last_call"] = time.time()
            try:
                resp = requests.post(URL, headers=headers, json=payload,
                                     timeout=60)
            except requests.RequestException as e:
                last_error = e
                wait = (2 ** attempt) + random.random()
                if verbose:
                    print(f"    network error ({type(e).__name__}), "
                          f"waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                data = resp.json()
                message = data["choices"][0]["message"]
                usage = data.get("usage", {})
                details = usage.get("completion_tokens_details") or {}
                text = message.get("content") or ""
                if (message.get("reasoning") or details.get("reasoning_tokens")
                        or "<think>" in text):
                    raise RuntimeError(
                        "model returned reasoning although reasoning_effort="
                        f"'none' was set; stopping. Response: {message!r}")
                return text, usage.get("prompt_tokens"), usage.get("completion_tokens")

            last_error = f"{resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"permanent error, not retrying: {last_error}")
            retry_after = resp.headers.get("retry-after")
            try:
                wait = float(retry_after) + random.random()
            except (TypeError, ValueError):
                wait = (2 ** attempt) + random.random()
            if resp.status_code == 429 and wait > MAX_WAIT_SECONDS:
                # A long wait means the DAILY limit, not the per-minute one.
                raise DailyLimitReached(
                    f"Groq asked to wait {wait / 60:.0f} minutes: daily "
                    f"limit reached. {last_error}")
            if verbose:
                print(f"    got {resp.status_code}, waiting {wait:.1f}s")
            time.sleep(wait)

        raise RuntimeError(f"failed after {MAX_RETRIES} tries: {last_error}")

    def responder(prompt):
        # Same text Gemini received (the prompt minus its final "Group"
        # line), plus the one answer-only sentence at the end.
        body = (prompt.rsplit("\n" + TRAILING_LINE, 1)[0].rstrip()
                + "\n" + FORMAT_INSTRUCTION)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": body}],
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
            "reasoning_effort": "none",
        }

        rejected = []
        for attempt in range(1, max_attempts + 1):
            text, p_tok, c_tok = one_call(payload)
            if names_one_group(text) or attempt == max_attempts:
                extra = {"attempts": attempt,
                         "rejected_raw": json.dumps(rejected) if rejected else ""}
                return text, p_tok, c_tok, extra
            rejected.append(text)
            if verbose:
                print(f"    unreadable reply, asking again: {text[:60]!r}")

    return responder


if __name__ == "__main__":
    # Two-call check:  python src/qwen_responder.py qwen/qwen3.8-27b
    import sys
    from structures import ITEMS
    from stimuli import make_stimulus
    from prompt import build_prompt, parse_response

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.8-27b"
    stim = make_stimulus(random.Random(1))
    responder = make_qwen_responder(model_name, verbose=True)

    for label, hist in [("trial 1, no history", []),
                        ("trial 4, three history lines", [
                            {"trial_number": n,
                             "item_text": stim.item_text(ITEMS[n - 1]),
                             "chosen_group": "Group Alpha",
                             "correct_group": "Group Alpha"} for n in (1, 2, 3)])]:
        item = ITEMS[len(hist)]
        raw, p, c, extra = responder(build_prompt(stim, "neutral", hist, item))
        print(f"{label:32} raw={raw!r:14} parsed={parse_response(raw, stim)} "
              f"tokens={p}/{c} attempts={extra['attempts']}")

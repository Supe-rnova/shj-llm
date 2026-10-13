"""
diag_qwen_prefill.py

Diagnostic, not data collection. Answers two questions about prefill on
Groq before any Qwen data is collected:

  1. Does Groq really CONTINUE the prefilled assistant message, or does it
     start a fresh reply after it? (Matters for the methods section: the
     Gemini runs were true continuations.)
  2. On trial 1 (no history), does Qwen answer, or stop without choosing?

    python src/diag_qwen_prefill.py

30 requests: 3 conditions x 2 prompts x 5 repeats, about 2 minutes.

Conditions
  prefill     user = prompt without the final "Group"; assistant = "Group"
              (exactly what qwen_responder.py sends)
  marker      same, but assistant = "Answer: Group". If the returned text
              starts with "Answer:", Groq echoes the prefill; if it is just
              " Alpha"/" Beta", Groq returns only the continuation; if it is
              "Group Alpha", the prefill was treated as a finished message.
  no_prefill  user = prompt without the final "Group"; no assistant message
"""

import os
import random
import time

import requests
from dotenv import load_dotenv

from structures import ITEMS
from stimuli import make_stimulus
from prompt import build_prompt

load_dotenv()

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3.8-27b"
REPEATS = 5
GAP = 2.5

headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
           "Content-Type": "application/json"}

stim = make_stimulus(random.Random(1))
history = [{"trial_number": n, "item_text": stim.item_text(ITEMS[n - 1]),
            "chosen_group": "Group Alpha", "correct_group": "Group Alpha"}
           for n in (1, 2, 3)]
prompts = {
    "trial 1": build_prompt(stim, "neutral", [], ITEMS[0]),
    "trial 4": build_prompt(stim, "neutral", history, ITEMS[3]),
}
conditions = {"prefill": "Group", "marker": "Answer: Group", "no_prefill": None}


def call(body, prefill):
    messages = [{"role": "user", "content": body}]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    payload = {"model": MODEL, "messages": messages, "temperature": 1.0,
               "max_completion_tokens": 40, "reasoning_effort": "none"}
    resp = requests.post(URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        return f"HTTP {resp.status_code}: {resp.text[:200]}", "", "", []
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    extra = [k for k in msg if k not in ("role", "content")]
    return (msg.get("content"), choice.get("finish_reason"),
            data.get("usage", {}).get("completion_tokens"), extra)


for cond, prefill in conditions.items():
    print(f"\n=== {cond} (assistant message: {prefill!r}) ===")
    for label, prompt in prompts.items():
        body = prompt.rsplit("\nGroup", 1)[0]
        for i in range(1, REPEATS + 1):
            content, finish, ntok, extra = call(body, prefill)
            print(f"  {label} #{i}: content={content!r:40} "
                  f"finish={finish} tokens={ntok} extra_fields={extra}")
            time.sleep(GAP)

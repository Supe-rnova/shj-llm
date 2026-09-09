"""Try several ways of forcing a bare category answer."""

import os
import random
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from structures import ITEMS
from stimuli import make_stimulus
from prompt import build_prompt, parse_response

load_dotenv()
model_name = sys.argv[1]
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
stim = make_stimulus(random.Random(1))
base = build_prompt(stim, "neutral", [], ITEMS[0])

NO_THINK = types.ThinkingConfig(thinking_budget=0)


def show(label, resp):
    u = resp.usage_metadata
    text = resp.text
    print(f"\n[{label}]")
    print(f"  thoughts={u.thoughts_token_count} out={u.candidates_token_count} "
          f"finish={resp.candidates[0].finish_reason}")
    print(f"  text   : {text!r}")
    print(f"  parsed : {parse_response(text, stim)}")


# A: prefill - put "Group" in the model's own turn, as Jagadish did
try:
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(role="user", parts=[types.Part(
                text=base.rsplit("\nGroup", 1)[0])]),
            types.Content(role="model", parts=[types.Part(text="Group")]),
        ],
        config=types.GenerateContentConfig(
            temperature=1.0, max_output_tokens=100, thinking_config=NO_THINK),
    )
    show("A: prefill", resp)
except Exception as e:
    print(f"\n[A: prefill] REJECTED: {str(e)[:120]}")

# B: system instruction demanding one word
try:
    resp = client.models.generate_content(
        model=model_name, contents=base,
        config=types.GenerateContentConfig(
            temperature=1.0, max_output_tokens=100, thinking_config=NO_THINK,
            system_instruction=(
                "Reply with exactly one word: Alpha or Beta. "
                "No explanation, no preamble. If you are unsure, guess."),
        ),
    )
    show("B: system instruction", resp)
except Exception as e:
    print(f"\n[B: system] REJECTED: {str(e)[:120]}")

# C: explicit guess instruction in the prompt body
guess_prompt = base.replace(
    '(Give the answer in the form "Group <your answer>".)',
    '(Answer with exactly one word: Alpha or Beta. You must choose one '
    'even if you are unsure.)')
try:
    resp = client.models.generate_content(
        model=model_name, contents=guess_prompt,
        config=types.GenerateContentConfig(
            temperature=1.0, max_output_tokens=100, thinking_config=NO_THINK),
    )
    show("C: explicit guess instruction", resp)
except Exception as e:
    print(f"\n[C: explicit] REJECTED: {str(e)[:120]}")
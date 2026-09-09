"""Dump the full response object so we can see what actually came back."""

import os
import random
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from structures import ITEMS
from stimuli import make_stimulus
from prompt import build_prompt

load_dotenv()
model_name = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
stim = make_stimulus(random.Random(1))
prompt = build_prompt(stim, "neutral", [], ITEMS[0])

resp = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=max_tokens,
    ),
)

print("USAGE")
print(resp.usage_metadata)
print("\nCANDIDATES")
for c in resp.candidates or []:
    print("  finish_reason :", c.finish_reason)
    print("  content       :", c.content)
print("\nresp.text :", repr(resp.text))
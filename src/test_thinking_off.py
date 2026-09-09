"""Can thinking be disabled on this model?"""

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

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
stim = make_stimulus(random.Random(1))
prompt = build_prompt(stim, "neutral", [], ITEMS[0])

try:
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=100,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    resp = client.models.generate_content(
        model=model_name, contents=prompt, config=config
    )
    print("thinking_budget=0 accepted")
    print("  thoughts   :", resp.usage_metadata.thoughts_token_count)
    print("  candidates :", resp.usage_metadata.candidates_token_count)
    print("  finish     :", resp.candidates[0].finish_reason)
    print("  text       :", repr(resp.text))
except Exception as e:
    print("thinking_budget=0 rejected:", e)
"""Ask the API which models this key can use. Run once."""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"{m.name:55} {getattr(m, 'display_name', '')}")
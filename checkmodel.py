import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Add it to .env and run from the project root.")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
if response.status_code != 200:
    raise SystemExit(f"Error {response.status_code}: {response.text}")

for model in sorted(response.json()["data"], key=lambda m: m["id"]):
    print(model["id"])
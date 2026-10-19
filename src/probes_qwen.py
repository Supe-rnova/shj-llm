"""
probes_qwen.py

Contamination probes A, B and C for Qwen 3.8 27B on Groq. Same three
probes, same wording and same flag list as probes.py (Gemini), so the two
models can be compared directly.

    python src/probes_qwen.py

A - post-run strategy report. Replays a completed run's history, then
    asks the model to state the rule it used.
    Difference from the Gemini version: Qwen has no 64-trial pilot run,
    so the replayed run is a 24-trial CEILING run from the main study
    (first complete one for Type I and for Type VI).
B - paradigm recognition, fresh context. Abstract 3-bit structure only.
C - direct knowledge check, fresh context.

Settings match the trials: temperature 1.0, reasoning_effort "none".
No answer-only sentence is added here: these probes ask for prose.

About 6 requests and 6k tokens. Output: notes/probe_results_qwen3.8-27b_<date>.md
"""

import os
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from structures import ITEMS, STRUCTURES
from stimuli import make_stimulus
from prompt import build_prompt
from probes import flags_in, rebuild_history


# Probe B and C wording: copied verbatim from probes.py (Gemini) so both
# models get the same questions. probes.py builds them inside its
# functions, so they are repeated here rather than imported.
def probe_b_prompt(shj_type):
    labels = STRUCTURES[shj_type]
    rows = "\n".join(
        f"  {''.join(str(b) for b in item)} -> {'A' if lab else 'B'}"
        for item, lab in zip(ITEMS, labels))
    return (
        "Here is a set of eight items. Each item is described by three "
        "binary features, and each is assigned to category A or B.\n\n"
        f"{rows}\n\n"
        "What task or experimental paradigm is this? Be specific if you "
        "recognise it.")


PROBE_C_PROMPT = ("What is the Shepard, Hovland and Jenkins (1961) category "
                  "learning task, and what is the difficulty ordering of its "
                  "six types?")

load_dotenv()

MODEL = "qwen/qwen3.8-27b"
SHORT = MODEL.split("/")[-1]
URL = "https://api.groq.com/openai/v1/chat/completions"
MAIN_CSV = "data/main/main_qwen3.8-27b.csv"
GAP_SECONDS = 15   # long prompts; stays well under 8k tokens per minute


def ask(prompt, max_tokens=800):
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
               "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 1.0, "max_completion_tokens": max_tokens,
               "reasoning_effort": "none"}
    for attempt in range(5):
        resp = requests.post(URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            time.sleep(GAP_SECONDS)
            message = resp.json()["choices"][0]["message"]
            if message.get("reasoning"):
                raise RuntimeError("reasoning returned with reasoning_effort='none'")
            return message.get("content") or ""
        if resp.status_code in (400, 401, 403, 404):
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        wait = float(resp.headers.get("retry-after") or 2 ** attempt) + 1
        if wait > 300:
            raise RuntimeError(f"daily limit reached: {resp.text[:300]}")
        print(f"    got {resp.status_code}, waiting {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError("failed after 5 attempts")


def probe_a(shj_type):
    df = pd.read_csv(MAIN_CSV)
    runs = df[(df.shj_type == shj_type) & (df.run_kind == "ceiling")]
    runs = runs.groupby("run_id").filter(lambda g: len(g) == 24)
    run_id = runs.sort_values("run_number").run_id.iloc[0]
    run = runs[runs.run_id == run_id].sort_values("trial_index")

    seed = int(run.seed.iloc[0])
    stim = make_stimulus(random.Random(seed))
    body = build_prompt(stim, "neutral", [], ITEMS[0]).split(
        "\n\nWhat group would")[0]
    body = body + "\n" + "\n".join(rebuild_history(run))
    question = ("\n\nThe experiment is now over. Describe the rule or "
                "strategy you used to decide which group each sample "
                "belonged to.")
    text = ask(body + question)
    return {"run_id": run_id, "seed": seed, "trials": len(run),
            "dim_to_feature": run.dim_to_feature.iloc[0],
            "accuracy": round(float(run.correct.mean()), 3),
            "response": text, "flags": flags_in(text)}


def main():
    out = Path(f"notes/probe_results_{SHORT}_{date.today()}.md")
    lines = [f"# Contamination probes — {SHORT}", f"**Date:** {date.today()}",
             "**Settings:** temperature 1.0, reasoning_effort=none (Groq)",
             "**Probe A source:** 24-trial ceiling runs from the main study "
             "(Gemini used 64-trial pilot runs)", ""]

    print("Probe A — post-run strategy report\n")
    lines.append("## Probe A — post-run strategy report\n")
    for t in (1, 6):
        r = probe_a(t)
        print(f"  Type {t} (run accuracy {r['accuracy']}, flags: {r['flags'] or 'none'})")
        print(f"    {r['response']!r}\n")
        lines += [f"### Type {t}",
                  f"- run_id: `{r['run_id']}`, seed {r['seed']}, "
                  f"{r['trials']} trials, accuracy {r['accuracy']}",
                  f"- mapping: `{r['dim_to_feature']}`",
                  f"- flags: {r['flags'] or 'none'}", "",
                  "```", r["response"].strip(), "```", ""]

    print("Probe B — paradigm recognition, fresh context\n")
    lines.append("## Probe B — paradigm recognition (fresh context)\n")
    for t in (1, 2, 6):
        text = ask(probe_b_prompt(t))
        flags = flags_in(text)
        print(f"  Type {t} (flags: {flags or 'none'})\n    {text!r}\n")
        lines += [f"### Type {t}", f"- flags: {flags or 'none'}", "",
                  "```", text.strip(), "```", ""]

    print("Probe C — direct knowledge check, fresh context\n")
    lines.append("## Probe C — direct knowledge check (fresh context)\n")
    text = ask(PROBE_C_PROMPT, max_tokens=1200)
    flags = flags_in(text)
    print(f"  flags: {flags or 'none'}\n    {text!r}\n")
    lines += [f"- flags: {flags or 'none'}", "", "```", text.strip(), "```", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()

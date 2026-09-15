"""
probes.py

Contamination probes, Pipeline section 2.2.

    python src/probes.py <model>

Run AFTER the main pilot so they cannot contaminate it.

A - post-run strategy report. Replays a completed run's history, then
    asks the model to state the rule it used. Same conversation.
B - paradigm recognition, fresh context. Abstract structure only, no
    cover story. Does the model recognise the task from the structure?
C - direct knowledge check, fresh context. Does the model know SHJ at
    all? Establishes that knowledge exists, separately from whether it
    was deployed during a run.

The informative outcome is A and B clean while C comes back informed.

Output is written to notes/probe_results_<model>_<date>.md verbatim.
Raw text is the data here - do not summarise it away.
"""

import json
import os
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from structures import ITEMS, STRUCTURES
from stimuli import make_stimulus
from prompt import build_prompt, history_line

load_dotenv()

FLAGS = ["shepard", "hovland", "jenkins", "type ii", "type vi", "xor",
         "exclusive or", "parity", "category learning", "categorization",
         "categorisation", "nosofsky", "kurtz", "alcove", "gcm",
         "exemplar", "prototype", "cognitive psychology", "psychology",
         "experiment", "1961"]

PILOT_CSV = "data/pilot/pilot_gemini-3.1-flash-lite.csv"


def ask(client, model_name, contents, max_tokens=800):
    """One call, thinking off, generous output budget for prose."""
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


def flags_in(text):
    low = (text or "").lower()
    return [f for f in FLAGS if f in low]


def rebuild_history(run):
    """Reconstruct the trial history text from a logged run."""
    lines = []
    for _, r in run.iterrows():
        lines.append(history_line(
            int(r.trial_index), r.item_text,
            r.response_parsed if r.parse_ok else "no clear answer",
            r.category_true))
    return lines


def probe_a(client, model_name, csv_path, shj_type):
    """Replay a completed run, then ask for the strategy."""
    df = pd.read_csv(csv_path)
    run = df[df.shj_type == shj_type].groupby("run_id").filter(
        lambda g: len(g) == 64)
    run_id = run.run_id.iloc[0]
    run = run[run.run_id == run_id].sort_values("trial_index")

    stim_seed = int(run.seed.iloc[0])
    stim = make_stimulus(random.Random(stim_seed))

    body = build_prompt(stim, "neutral", [], ITEMS[0]).split(
        "\n\nWhat group would")[0]
    body = body + "\n" + "\n".join(rebuild_history(run))

    question = ("\n\nThe experiment is now over. Describe the rule or "
                "strategy you used to decide which group each sample "
                "belonged to.")

    text = ask(client, model_name, body + question)
    return {
        "run_id": run_id, "seed": stim_seed, "shj_type": shj_type,
        "dim_to_feature": run.dim_to_feature.iloc[0],
        "accuracy": round(float(run.correct.mean()), 3),
        "response": text, "flags": flags_in(text),
    }


def probe_b(client, model_name, shj_type):
    """Abstract structure only. No cover story, no feature names."""
    labels = STRUCTURES[shj_type]
    rows = "\n".join(
        f"  {''.join(str(b) for b in item)} -> {'A' if lab else 'B'}"
        for item, lab in zip(ITEMS, labels))
    prompt = (
        "Here is a set of eight items. Each item is described by three "
        "binary features, and each is assigned to category A or B.\n\n"
        f"{rows}\n\n"
        "What task or experimental paradigm is this? Be specific if you "
        "recognise it.")
    text = ask(client, model_name, prompt)
    return {"shj_type": shj_type, "prompt": prompt,
            "response": text, "flags": flags_in(text)}


def probe_c(client, model_name):
    """Direct knowledge check."""
    prompt = ("What is the Shepard, Hovland and Jenkins (1961) category "
              "learning task, and what is the difficulty ordering of its "
              "six types?")
    text = ask(client, model_name, prompt, max_tokens=1200)
    return {"prompt": prompt, "response": text, "flags": flags_in(text)}


def main():
    model_name = sys.argv[1]
    short = model_name.split("/")[-1]
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    out = Path(f"notes/probe_results_{short}_{date.today()}.md")
    out.parent.mkdir(exist_ok=True)
    lines = [f"# Contamination probes — {short}", f"**Date:** {date.today()}",
             f"**Settings:** temperature 1.0, thinking_budget=0", ""]

    print("Probe A — post-run strategy report\n")
    lines.append("## Probe A — post-run strategy report\n")
    for shj_type in (1, 6):
        r = probe_a(client, model_name, PILOT_CSV, shj_type)
        print(f"  Type {shj_type} (run accuracy {r['accuracy']}, "
              f"flags: {r['flags'] or 'none'})")
        print(f"    {r['response']!r}\n")
        lines += [f"### Type {shj_type}",
                  f"- run_id: `{r['run_id']}`, seed {r['seed']}, "
                  f"accuracy {r['accuracy']}",
                  f"- mapping: `{r['dim_to_feature']}`",
                  f"- flags: {r['flags'] or 'none'}", "",
                  "```", (r["response"] or "").strip(), "```", ""]

    print("Probe B — paradigm recognition, fresh context\n")
    lines.append("## Probe B — paradigm recognition (fresh context)\n")
    for shj_type in (1, 2, 6):
        r = probe_b(client, model_name, shj_type)
        print(f"  Type {shj_type} (flags: {r['flags'] or 'none'})")
        print(f"    {r['response']!r}\n")
        lines += [f"### Type {shj_type}",
                  f"- flags: {r['flags'] or 'none'}", "",
                  "```", (r["response"] or "").strip(), "```", ""]

    print("Probe C — direct knowledge check, fresh context\n")
    lines.append("## Probe C — direct knowledge check (fresh context)\n")
    r = probe_c(client, model_name)
    print(f"  flags: {r['flags'] or 'none'}")
    print(f"    {r['response']!r}\n")
    lines += [f"- flags: {r['flags'] or 'none'}", "",
              "```", (r["response"] or "").strip(), "```", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
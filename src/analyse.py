"""
analyse.py

All behavioural analyses for the paper, from the raw trial logs.

    python src/analyse.py

Reads    data/pilot/pilot_gemini-3.1-flash-lite.csv   (pilot, Figure 1A only)
         data/main/main_gemini-3.1-flash-lite.csv
         data/main/main_qwen3.8-27b.csv
Writes   outputs/fig1_ceiling.png / .pdf
         outputs/fig2_first_pass_by_trial.png / .pdf
         outputs/fig3_type_order.png / .pdf
         outputs/fig4_nearest_neighbour.png / .pdf
         outputs/results_summary.md      every number quoted in the paper
         outputs/block1_run_means.csv    one row per run (for checking)

Rules used throughout
  * The RUN is the unit of analysis. Each run's block-1 accuracy is one
    number; statistics are done on those numbers, never on pooled trials
    (trials within a run are not independent).
  * Block-1 accuracy uses trials 2-8. Trial 1 is a pure guess: nothing has
    been seen yet.
  * Unreadable answers (parse_ok False; Qwen only, 29 trials) are left out
    of accuracy, not counted as wrong.
  * Pilot data is used only for Figure 1A. All other analyses use the main
    study, which was collected fresh with new seeds.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PILOT = Path("data/pilot/pilot_gemini-3.1-flash-lite.csv")
MAIN = {"Gemini 3.1 Flash-Lite": Path("data/main/main_gemini-3.1-flash-lite.csv"),
        "Qwen 3.8 27B": Path("data/main/main_qwen3.8-27b.csv")}
OUT = Path("outputs")

TYPES = [1, 2, 3, 4, 5, 6]
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# Humans: trials to criterion, Nosofsky et al. (1994), as recorded in
# project_pipeline_v2 section 0.2. Lower = easier. Block-level error
# proportions (Nosofsky Table 1) are not transcribed yet.
HUMAN_TTC = {1: 44.0, 2: 85.4, 3: 121.6, 4: 127.0, 5: 133.8, 6: 189.2}

# Figure style: one colour per model (fixed order), thin marks,
# recessive axes. Gray is reserved for chance and for humans.
COLOR = {"Gemini 3.1 Flash-Lite": "#2a78d6", "Qwen 3.8 27B": "#eb6834"}
MARKER = {"Gemini 3.1 Flash-Lite": "o", "Qwen 3.8 27B": "s"}
INK, INK2, CHANCE = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- loading

def load(path):
    d = pd.read_csv(path)
    d["item_bits"] = d["item_bits"].astype(str).str.zfill(3)
    d["readable"] = d["parse_ok"].astype(str).str.lower() == "true"
    d["correct"] = d["correct"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    d.loc[~d.readable, "correct"] = np.nan
    return d


def drop_trial1(d):
    """Remove trial 1 of block 1 (a pure guess); keep everything else."""
    return d[~((d.block == 1) & (d.trial_in_block == 1))]


def ci95(x):
    """Mean and t-based 95% confidence half-width over runs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1)
                                  * x.std(ddof=1) / np.sqrt(len(x)))


def holm(pvals):
    """Holm correction for a list of p values (same order back)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    return "< .001" if p < 0.001 else f"{p:.3f}".replace("0.", ".", 1)


# ------------------------------------------------- block-1 run-level table

def block1_run_means(d, model):
    b1 = d[(d.block == 1) & (d.trial_in_block >= 2)]
    g = b1.groupby(["shj_type", "run_id", "seed"])
    out = g.correct.mean().rename("acc").reset_index()
    out["n_readable"] = g.correct.count().values
    out["model"] = model
    return out


# ------------------------------------------------- nearest-neighbour check

def nearest_neighbour_rows(d, model):
    """One row per block-1 trial 2-8 with the nearest-neighbour prediction.

    For the current sample, find the most similar sample(s) already seen
    in this run (fewest differing features). If they all had the same
    correct answer, a 'copy the nearest one' learner would give that
    answer: nn_label. If they disagree, it is a tie (no prediction).
    own_label is the same idea using the model's OWN earlier answers
    instead of the correct ones (tests copying its own picks).
    """
    rows = []
    b1 = d[d.block == 1].sort_values(["run_id", "trial_in_block"])
    for run_id, run in b1.groupby("run_id"):
        seen = []
        for _, r in run.iterrows():
            if seen:
                dist = [sum(a != b for a, b in zip(r.item_bits, s.item_bits))
                        for s in seen]
                dmin = min(dist)
                near = [s for s, k in zip(seen, dist) if k == dmin]
                correct_labels = {s.category_true for s in near}
                own = {s.response_parsed for s in near if s.readable}
                rows.append({
                    "model": model, "shj_type": r.shj_type, "run_id": run_id,
                    "trial": r.trial_in_block, "dmin": dmin,
                    "readable": r.readable, "choice": r.response_parsed,
                    "truth": r.category_true,
                    "nn_label": (next(iter(correct_labels))
                                 if len(correct_labels) == 1 else None),
                    "own_label": next(iter(own)) if len(own) == 1 else None,
                })
            seen.append(r)
    return pd.DataFrame(rows)


# ================================================================== main

def main():
    OUT.mkdir(exist_ok=True)
    data = {m: load(p) for m, p in MAIN.items()}
    pilot = load(PILOT)
    lines = ["# Results summary", "",
             "Generated by `src/analyse.py`. The run is the unit of analysis; "
             "block-1 accuracy uses trials 2-8; unreadable answers are excluded.",
             ""]

    # ---------------------------------------------------- 0. data overview
    lines += ["## 0. Data", "",
              "| Model | Runs | Trials | Readable | Needed a retry | "
              "Trial-1 'Alpha' rate |", "|---|---|---|---|---|---|"]
    for m, d in data.items():
        t1 = d[(d.block == 1) & (d.trial_in_block == 1) & d.readable]
        alpha = (t1.response_parsed == "Group Alpha").mean()
        retry = int((d.get("attempts", pd.Series(1, index=d.index)) > 1).sum())
        lines.append(f"| {m} | {d.run_id.nunique()} | {len(d)} | "
                     f"{d.readable.mean():.1%} ({int((~d.readable).sum())} not) "
                     f"| {retry} | {alpha:.2f} |")
    lines.append("")

    # ---------------------------------------------------- 1. the ceiling
    lines += ["## 1. The ceiling (open-book)", "",
              "Block 1 = trials 2-8, as in every other section (trial 1 is a "
              "pure guess). Both columns are pooled proportions: correct / "
              "readable answers. The last column is the error rate after "
              "block 1 with a one-sided 95% upper bound (exact binomial; with "
              "0 errors in n trials this is about 3/n, the 'rule of three').",
              "",
              "| Data | Block 1 accuracy | Accuracy after block 1 | "
              "Error rate after block 1 (95% upper bound) |",
              "|---|---|---|---|"]

    def ceiling_row(label, rows):
        b1 = rows[(rows.block == 1) & (rows.trial_in_block >= 2)].correct
        after = rows[rows.block > 1].correct
        return (label, int(b1.sum()), int(b1.count()),
                int(after.sum()), int(after.count()))

    ceiling_rows = [ceiling_row(f"Pilot Gemini, Type {ROMAN[t]} (10 runs x 8 blocks)",
                                pilot[pilot.shj_type == t]) for t in (1, 6)]
    for m, d in data.items():
        c = d[d.run_kind == "ceiling"]
        ceiling_rows.append(ceiling_row(
            f"Main {m}, ceiling runs (types "
            f"{', '.join(ROMAN[t] for t in sorted(c.shj_type.unique()))})", c))
    gemini_all = [r for r in ceiling_rows if "Qwen" not in r[0]]
    ceiling_rows.append(("All Gemini data combined", sum(r[1] for r in gemini_all),
                         sum(r[2] for r in gemini_all), sum(r[3] for r in gemini_all),
                         sum(r[4] for r in gemini_all)))
    for label, b1_ok, b1_n, ok, n in ceiling_rows:
        errors = n - ok
        upper = stats.binomtest(errors, n).proportion_ci(
            confidence_level=0.90, method="exact").high   # one-sided 95%
        lines.append(f"| {label} | {b1_ok / b1_n:.3f} ({b1_ok}/{b1_n}) | "
                     f"{ok / n:.3f} ({ok}/{n}) | {errors / n:.2%} "
                     f"(< {upper:.2%}) |")
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for t, ls in ((1, "-"), (6, "--")):
        p = drop_trial1(pilot[pilot.shj_type == t])
        runs = p.groupby(["run_id", "block"]).correct.mean()
        by_block = runs.groupby(level="block")
        mean = by_block.mean()
        half = by_block.apply(lambda x: ci95(x)[1])
        ax.errorbar(mean.index, mean.values, yerr=half.values, ls=ls,
                    color=COLOR["Gemini 3.1 Flash-Lite"], marker="o", ms=4,
                    capsize=2, lw=1.6)
        ax.annotate(f"Type {ROMAN[t]}", (1, mean.loc[1]), xytext=(6, -3 if t == 1 else 0),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
    ax.set(xlabel="Block (8 trials each)", ylabel="Accuracy", ylim=(0, 1.05),
           xticks=range(1, 9), title="A. Pilot: Gemini, 64-trial runs")
    ax = axes[1]
    for i, (m, d) in enumerate(data.items()):
        c = d[d.run_kind == "ceiling"]
        runs = drop_trial1(c).groupby(["run_id", "block"]).correct.mean()
        by_block = runs.groupby(level="block")
        mean = by_block.mean()
        half = by_block.apply(lambda x: ci95(x)[1])
        ax.errorbar(mean.index + (i - 0.5) * 0.08, mean.values, yerr=half.values,
                    color=COLOR[m], marker=MARKER[m], ms=5, capsize=2, lw=1.6,
                    label=m.split()[0])
    ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
    ax.set(xlabel="Block", ylim=(0, 1.05), xticks=[1, 2, 3],
           title="B. Main study: ceiling runs")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_ceiling.{ext}")
    plt.close(fig)

    # ------------------------------------- 2. first pass, trial by trial
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, TYPES):
        for m, d in data.items():
            b1 = d[(d.block == 1) & (d.shj_type == t)]
            per_trial = b1.groupby("trial_in_block").correct
            mean = per_trial.mean()
            se = np.sqrt(mean * (1 - mean) / per_trial.count())
            ax.fill_between(mean.index, mean - 1.96 * se, mean + 1.96 * se,
                            color=COLOR[m], alpha=0.12, lw=0)
            ax.plot(mean.index, mean.values, color=COLOR[m], marker=MARKER[m],
                    ms=3.5, lw=1.6, label=m.split()[0])
        ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
        ax.set_title(f"Type {ROMAN[t]}", loc="left")
        ax.set(ylim=(0, 1), xticks=range(1, 9))
    for ax in axes[1]:
        ax.set_xlabel("Trial in block 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_first_pass_by_trial.{ext}")
    plt.close(fig)

    # ------------------------------------------ 3. block-1 accuracy by type
    rm = pd.concat([block1_run_means(d, m) for m, d in data.items()])
    rm.to_csv(OUT / "block1_run_means.csv", index=False)

    lines += ["## 2. Block-1 accuracy by type (trials 2-8, run means)", "",
              "One-sample t test against chance (0.5), two-sided; Wilcoxon "
              "signed-rank as a check. p values Holm-corrected across the 12 "
              "tests.", "",
              "| Model | Type | Runs | Mean | 95% CI | t | p (Holm) | "
              "Wilcoxon p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    tests = []
    for m in data:
        for t in TYPES:
            x = rm[(rm.model == m) & (rm.shj_type == t)].acc.dropna()
            tt = stats.ttest_1samp(x, 0.5)
            w = stats.wilcoxon(x - 0.5, zero_method="wilcox")
            tests.append((m, t, len(x), *ci95(x), tt.statistic, tt.pvalue, w.pvalue))
    p_t = holm([r[6] for r in tests])
    p_w = holm([r[7] for r in tests])
    for r, pt, pw in zip(tests, p_t, p_w):
        m, t, n, mean, half, tstat = r[:6]
        lines.append(f"| {m} | {ROMAN[t]} | {n} | {mean:.3f} | "
                     f"[{mean - half:.3f}, {mean + half:.3f}] | {tstat:.2f} | "
                     f"{fmt_p(pt)} | {fmt_p(pw)} |")
    lines.append("")

    lines += ["## 3. Do the types differ? (per model)", ""]
    for m in data:
        groups = [rm[(rm.model == m) & (rm.shj_type == t)].acc.dropna() for t in TYPES]
        f = stats.f_oneway(*groups)
        k = stats.kruskal(*groups)
        lines.append(f"- **{m}:** one-way ANOVA F(5, {sum(map(len, groups)) - 6}) = "
                     f"{f.statistic:.2f}, p {fmt_p(f.pvalue)}; Kruskal-Wallis "
                     f"H = {k.statistic:.2f}, p {fmt_p(k.pvalue)}.")
    lines += ["", "Pairwise Welch t tests between types, Holm-corrected within "
              "each model (15 pairs):", "",
              "| Model | Pair | Difference | t | p (Holm) |", "|---|---|---|---|---|"]
    for m in data:
        pairs, raw = [], []
        for i, a in enumerate(TYPES):
            for b in TYPES[i + 1:]:
                xa = rm[(rm.model == m) & (rm.shj_type == a)].acc.dropna()
                xb = rm[(rm.model == m) & (rm.shj_type == b)].acc.dropna()
                res = stats.ttest_ind(xa, xb, equal_var=False)
                pairs.append((a, b, xa.mean() - xb.mean(), res.statistic))
                raw.append(res.pvalue)
        for (a, b, diff, tstat), p in zip(pairs, holm(raw)):
            lines.append(f"| {m} | {ROMAN[a]} vs {ROMAN[b]} | {diff:+.3f} | "
                         f"{tstat:.2f} | {fmt_p(p)} |")
    lines.append("")

    lines += ["## 4. Gemini vs Qwen, same samples (paired by seed)", "",
              "| Type | Gemini | Qwen | Difference | Paired t | p (Holm) |",
              "|---|---|---|---|---|---|"]
    names = list(data)
    rows, raw = [], []
    for t in TYPES:
        a = rm[(rm.model == names[0]) & (rm.shj_type == t)].set_index("seed").acc
        b = rm[(rm.model == names[1]) & (rm.shj_type == t)].set_index("seed").acc
        both = pd.concat([a, b], axis=1, keys=["g", "q"]).dropna()
        res = stats.ttest_rel(both.g, both.q)
        rows.append((t, both.g.mean(), both.q.mean(), res.statistic))
        raw.append(res.pvalue)
    for (t, g, q, tstat), p in zip(rows, holm(raw)):
        lines.append(f"| {ROMAN[t]} | {g:.3f} | {q:.3f} | {g - q:+.3f} | "
                     f"{tstat:.2f} | {fmt_p(p)} |")
    lines.append("")

    lines += ["## 5. Order of the types vs humans", "",
              "Humans: trials to criterion, Nosofsky et al. (1994): "
              + ", ".join(f"{ROMAN[t]} {v:g}" for t, v in HUMAN_TTC.items())
              + ". Lower = easier, so agreement with humans shows as a "
              "NEGATIVE correlation with model accuracy. Six points only: "
              "descriptive.", ""]
    for m in data:
        means = [rm[(rm.model == m) & (rm.shj_type == t)].acc.mean() for t in TYPES]
        order = sorted(TYPES, key=lambda t: -means[t - 1])
        rho = stats.spearmanr(means, [HUMAN_TTC[t] for t in TYPES])
        tau = stats.kendalltau(means, [HUMAN_TTC[t] for t in TYPES])
        lines.append(f"- **{m}:** easiest to hardest "
                     f"{' > '.join(ROMAN[t] for t in order)}; Spearman rho = "
                     f"{rho.statistic:.2f} (p {fmt_p(rho.pvalue)}), Kendall tau = "
                     f"{tau.statistic:.2f}.")
    lines += ["- **Humans:** I > II > III > IV > V > VI.", ""]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for i, m in enumerate(data):
        mean_half = [ci95(rm[(rm.model == m) & (rm.shj_type == t)].acc) for t in TYPES]
        ax.errorbar(np.array(TYPES) + (i - 0.5) * 0.22, [x[0] for x in mean_half],
                    yerr=[x[1] for x in mean_half], fmt=MARKER[m], color=COLOR[m],
                    ms=5, capsize=2, lw=1.4, label=m)
    ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
    ax.text(6.45, 0.505, "chance", fontsize=7.5, color=CHANCE, va="bottom", ha="right")
    ax.set(xticks=TYPES, xticklabels=[ROMAN[t] for t in TYPES], ylim=(0, 1),
           xlabel="Type", ylabel="Block-1 accuracy (trials 2-8)",
           title="A. Models: first pass, mean of 40 runs, 95% CI")
    ax.legend(loc="upper right", fontsize=7.5)
    ax = axes[1]
    ax.bar(TYPES, [HUMAN_TTC[t] for t in TYPES], color=CHANCE, width=0.6)
    ax.set(xticks=TYPES, xticklabels=[ROMAN[t] for t in TYPES], xlabel="Type",
           ylabel="Trials to criterion", title="B. Humans (Nosofsky et al., 1994)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_type_order.{ext}")
    plt.close(fig)

    # ------------------------------------------ 6. nearest-neighbour check
    nn = pd.concat([nearest_neighbour_rows(d, m) for m, d in data.items()])
    nn = nn[nn.readable]
    lines += ["## 6. Nearest-neighbour check (block 1, trials 2-8)", "",
              "For each trial: the most similar sample(s) already seen in the "
              "run. 'Agreement' = the model gave the same answer as the "
              "nearest seen sample's correct answer (ties excluded). Run-level "
              "t test against 0.5.", "",
              "| Model | Type | Trials with a prediction | Ties | Agreement "
              "| t (runs) | p | Accuracy of the copy rule itself |",
              "|---|---|---|---|---|---|---|---|"]
    for m in data:
        for t in TYPES + ["all"]:
            s = nn[nn.model == m] if t == "all" else nn[(nn.model == m) & (nn.shj_type == t)]
            pred = s[s.nn_label.notna()]
            agree_run = (pred.choice == pred.nn_label).groupby(pred.run_id).mean()
            tt = stats.ttest_1samp(agree_run, 0.5)
            lines.append(f"| {m} | {ROMAN.get(t, 'all')} | {len(pred)} | "
                         f"{int(s.nn_label.isna().sum())} | "
                         f"{(pred.choice == pred.nn_label).mean():.3f} | "
                         f"{tt.statistic:.2f} | {fmt_p(tt.pvalue)} | "
                         f"{(pred.nn_label == pred.truth).mean():.3f} |")
    lines += ["", "Does the copy rule predict which types are hard? Expected "
              "block-1 accuracy of a pure 'copy the nearest seen sample' "
              "learner on the same trials (a tie counts as a 0.5 guess), next "
              "to the model's own accuracy:", "",
              "| Type | Copy rule (Gemini trials) | Gemini | Copy rule (Qwen "
              "trials) | Qwen |", "|---|---|---|---|---|"]
    rule_acc = {}
    for m in data:
        s = nn[nn.model == m].copy()
        s["rule_correct"] = np.where(s.nn_label.isna(), 0.5,
                                     (s.nn_label == s.truth).astype(float))
        rule_acc[m] = s.groupby("shj_type").rule_correct.mean()
    model_acc = {m: rm[rm.model == m].groupby("shj_type").acc.mean() for m in data}
    for t in TYPES:
        cells = []
        for m in data:
            cells += [f"{rule_acc[m][t]:.3f}", f"{model_acc[m][t]:.3f}"]
        lines.append(f"| {ROMAN[t]} | " + " | ".join(cells) + " |")
    lines.append("")
    for m in data:
        rho = stats.spearmanr(rule_acc[m].loc[TYPES], model_acc[m].loc[TYPES])
        r = stats.pearsonr(rule_acc[m].loc[TYPES], model_acc[m].loc[TYPES])
        rho_h = stats.spearmanr(model_acc[m].loc[TYPES], [-HUMAN_TTC[t] for t in TYPES])
        lines.append(f"- **{m}:** across the six types, model accuracy vs copy "
                     f"rule: Spearman rho = {rho.statistic:.2f}, Pearson r = "
                     f"{r.statistic:.2f}; vs human order: rho = "
                     f"{rho_h.statistic:.2f}. Six points: descriptive.")
    lines += ["", "Accuracy split by how far away the nearest seen sample was "
              "(1 = differs in one feature):", "",
              "| Model | Type | Nearest at distance 1 | Nearest at distance 2+ |",
              "|---|---|---|---|"]
    for m in data:
        for t in TYPES:
            s = nn[(nn.model == m) & (nn.shj_type == t)]
            acc = lambda q: (q.choice == q.truth).mean()
            d1, d2 = s[s.dmin == 1], s[s.dmin >= 2]
            lines.append(f"| {m} | {ROMAN[t]} | {acc(d1):.3f} (n={len(d1)}) | "
                         f"{acc(d2):.3f} (n={len(d2)}) |")
    lines.append("")

    lines += ["## 7. Copying its own earlier answers?", "",
              "Only trials where the nearest seen sample's correct answer and "
              "the model's own earlier answer to it DIFFER (the model got it "
              "wrong earlier). Which one does the model follow now?", "",
              "| Model | Trials | Follows own earlier answer | Follows the "
              "correct answer |", "|---|---|---|---|"]
    for m in data:
        s = nn[(nn.model == m) & nn.nn_label.notna() & nn.own_label.notna()
               & (nn.nn_label != nn.own_label)]
        lines.append(f"| {m} | {len(s)} | {(s.choice == s.own_label).mean():.3f} | "
                     f"{(s.choice == s.nn_label).mean():.3f} |")
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    for i, m in enumerate(data):
        vals = []
        for t in TYPES:
            pred = nn[(nn.model == m) & (nn.shj_type == t) & nn.nn_label.notna()]
            vals.append(ci95((pred.choice == pred.nn_label).groupby(pred.run_id).mean()))
        ax.errorbar(np.array(TYPES) + (i - 0.5) * 0.22, [v[0] for v in vals],
                    yerr=[v[1] for v in vals], fmt=MARKER[m], color=COLOR[m],
                    ms=5, capsize=2, lw=1.4, label=m.split()[0])
    ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
    ax.set(xticks=TYPES, xticklabels=[ROMAN[t] for t in TYPES], ylim=(0, 1),
           xlabel="Type", ylabel="Agreement with nearest seen sample",
           title="A. Copies the nearest sample's answer?")
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1]
    width = 0.36
    for i, m in enumerate(data):
        s = nn[(nn.model == m) & (nn.shj_type == 6)]
        groups = [s[s.dmin == 1], s[s.dmin >= 2]]
        vals = np.array([(g.choice == g.truth).mean() for g in groups])
        ns = np.array([len(g) for g in groups])
        err = 1.96 * np.sqrt(vals * (1 - vals) / ns)
        ax.bar(np.arange(2) + (i - 0.5) * width, vals, width=width - 0.03,
               yerr=err, capsize=2, error_kw={"lw": 1, "ecolor": INK2},
               color=COLOR[m], label=m.split()[0])
    ax.axhline(0.5, color=CHANCE, lw=1, ls=":")
    ax.set(xticks=[0, 1], xticklabels=["1 feature away", "2+ features away"],
           ylim=(0, 1), xlabel="Nearest sample already seen",
           ylabel="Accuracy", title="B. Type VI accuracy by nearest distance")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_nearest_neighbour.{ext}")
    plt.close(fig)

    # --------------------------------------- 8. Qwen retry sensitivity check
    lines += ["## 8. Sensitivity: Qwen without retried trials", "",
              "Block-1 accuracy (trials 2-8) using only trials answered on the "
              "first try.", "", "| Type | All readable trials | First-try only |",
              "|---|---|---|"]
    q = data["Qwen 3.8 27B"]
    b1 = q[(q.block == 1) & (q.trial_in_block >= 2)]
    for t in TYPES:
        s = b1[b1.shj_type == t]
        first = s[s.attempts == 1]
        lines.append(f"| {ROMAN[t]} | {s.groupby('run_id').correct.mean().mean():.3f} "
                     f"| {first.groupby('run_id').correct.mean().mean():.3f} |")
    lines.append("")

    (OUT / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFigures and summary written to {OUT}/")


if __name__ == "__main__":
    main()

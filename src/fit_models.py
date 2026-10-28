"""
fit_models.py

Fits three models of category learning to each LLM's block-1 choices
(trials 2-8), pooled across all six types (design option b, 28 Oct).

    python src/fit_models.py recover   # step 1: recovery check, fake data only
    python src/fit_models.py fit       # step 2: fit the real choices

THE THREE MODELS
Each gives, for every trial, the probability of answering "Group Alpha".
They see what the learner saw: the samples from earlier trials in the
same run, with their CORRECT groups (the feedback in the prompt).

  baseline   P(Alpha) = beta. No learning, just a leaning towards one
             label. This is the one-parameter model Jagadish et al. (2024)
             fitted to Claude-v2.

  exemplar   GCM (Nosofsky, 1986). Remembers every sample seen.
             similarity to a remembered sample = exp(-c * d), where d is
             the share of the 3 features that differ (0, 1/3, 2/3, 1).
             Evidence for a group = sum of similarities to its samples.

  prototype  Remembers one average per group (mean of its samples'
             features) and compares the new sample with each average,
             using the same similarity formula.

  Both learning models turn evidence into a choice the same way:
      P(Alpha) = beta * E_A / (beta * E_A + (1 - beta) * E_B)
  and then mix in a small share of pure guessing:
      P(Alpha) = (1 - eps) * P(Alpha) + eps * 0.5
  eps also handles trials where one group has not been seen yet.

  Dials: c (specificity: how fast similarity falls with distance; high c
  = only the closest samples count, i.e. "copy the nearest"), beta
  (leaning towards Alpha), eps (guessing). Attention is fixed EQUAL
  across the three features: with one fit across six types, per-feature
  attention would have no single meaning.

WHICH TRIALS ARE FITTED: the both-groups-seen rule (decided 28 Oct).
  Only block-1 trials where the learner has already seen at least one
  sample of EACH group. On a trial where only one group has been seen,
  both learning models must answer "that group" with certainty (there is
  nothing in the other group to compare with), so they make IDENTICAL
  predictions and cannot be told apart there. The LLMs do something else
  on those trials (a sample that looks different is put in the unseen
  group), which made a first, all-trials fit put ~2/3 of answers down to
  guessing. Those trials (393 of 1,680 per model, every trial 2 included)
  are reported separately, descriptively, in the one-group table.

FITTING: maximum likelihood (the dial values that make the observed
choices most probable), 12 random starts, bounded search. Models are
compared with AIC and BIC (lower = better; both penalise extra dials).

RECOVERY CHECK (run before looking at real fits)
  Parameter recovery: make fake choices from a model with known dials,
  on the REAL trial sequences, fit it, and see if the dials come back.
  Model recovery: make fake choices from each model, fit all three, and
  count how often the true model wins on BIC.
Fake choices are exact simulations: in this task the history the model
sees (samples and correct groups) does not depend on the learner's own
earlier choices.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAIN = {"Gemini 3.1 Flash-Lite": Path("data/main/main_gemini-3.1-flash-lite.csv"),
        "Qwen 3.8 27B": Path("data/main/main_qwen3.8-27b.csv")}
OUT = Path("outputs")
TYPES = [1, 2, 3, 4, 5, 6]
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

BOUNDS = {"c": (0.01, 30.0), "beta": (0.01, 0.99), "eps": (0.001, 0.999)}
PARAMS = {"baseline": ["beta"], "prototype": ["c", "beta", "eps"],
          "exemplar": ["c", "beta", "eps"]}
N_STARTS = 12


# ------------------------------------------------------------ the trials

def build_trials(path):
    """One row per block-1 trial 2-8, with everything the models need.

    D[i, j]   share of features in which earlier sample j differs (0-1)
    LA[i, j]  1 if earlier sample j's correct group was Alpha
    M[i, j]   1 if slot j holds a real earlier sample (padding = 0)
    X[i]      current sample's 3 bits; PA/PB prototype averages so far
    """
    d = pd.read_csv(path)
    d["item_bits"] = d["item_bits"].astype(str).str.zfill(3)
    d["readable"] = d["parse_ok"].astype(str).str.lower() == "true"
    b1 = d[d.block == 1].sort_values(["run_id", "trial_in_block"])

    rows = []
    for run_id, run in b1.groupby("run_id"):
        bits = [np.array([int(ch) for ch in s]) for s in run.item_bits]
        is_a = (run.category_true == "Group Alpha").to_numpy().astype(float)
        for k in range(1, len(run)):
            r = run.iloc[k]
            seen = np.array(bits[:k])
            seen_a = is_a[:k]
            x = bits[k]
            pa = seen[seen_a == 1].mean(axis=0) if seen_a.any() else np.full(3, np.nan)
            pb = seen[seen_a == 0].mean(axis=0) if (seen_a == 0).any() else np.full(3, np.nan)
            rows.append({
                "run_id": run_id, "shj_type": int(r.shj_type),
                "trial": int(r.trial_in_block),
                "d": np.abs(seen - x).sum(axis=1) / 3.0, "la": seen_a,
                "x": x, "pa": pa, "pb": pb,
                "truth_a": float(is_a[k]),
                "choice_a": (1.0 if r.response_parsed == "Group Alpha" else 0.0)
                if r.readable else np.nan,
            })
    n = len(rows)
    D = np.zeros((n, 7)); LA = np.zeros((n, 7)); M = np.zeros((n, 7))
    for i, row in enumerate(rows):
        k = len(row["d"])
        D[i, :k], LA[i, :k], M[i, :k] = row["d"], row["la"], 1.0
    X = np.array([r["x"] for r in rows])
    PA = np.array([r["pa"] for r in rows]); PB = np.array([r["pb"] for r in rows])
    return {
        "D": D, "LA": LA, "M": M, "X": X, "PA": PA, "PB": PB,
        "type": np.array([r["shj_type"] for r in rows]),
        "truth_a": np.array([r["truth_a"] for r in rows]),
        "choice_a": np.array([r["choice_a"] for r in rows]),
        "run_id": np.array([r["run_id"] for r in rows]),
    }


def both_groups_seen(T):
    """True on trials where at least one sample of each group was seen."""
    seen_a = (T["LA"] * T["M"]).sum(axis=1) > 0
    seen_b = ((1 - T["LA"]) * T["M"]).sum(axis=1) > 0
    return seen_a & seen_b


def subset(T, idx):
    return {k: v[idx] for k, v in T.items()}


def fit_trials(path):
    """The trials the models are fitted to (both-groups-seen rule)."""
    T = build_trials(path)
    return subset(T, np.where(both_groups_seen(T))[0])


# ------------------------------------------------------------ the models

def p_alpha(model, theta, T):
    """Probability of answering Alpha on every trial."""
    if model == "baseline":
        return np.full(len(T["D"]), theta[0])
    c, beta, eps = theta
    if model == "exemplar":
        sim = np.exp(-c * T["D"]) * T["M"]
        ea = (sim * T["LA"]).sum(axis=1)
        eb = (sim * (1 - T["LA"])).sum(axis=1)
    else:  # prototype
        ea = np.where(np.isnan(T["PA"][:, 0]), 0.0,
                      np.exp(-c * np.nan_to_num(np.abs(T["X"] - T["PA"])).sum(axis=1) / 3))
        eb = np.where(np.isnan(T["PB"][:, 0]), 0.0,
                      np.exp(-c * np.nan_to_num(np.abs(T["X"] - T["PB"])).sum(axis=1) / 3))
    num, den = beta * ea, beta * ea + (1 - beta) * eb
    p = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.5)
    return (1 - eps) * p + eps * 0.5


def neg_log_lik(theta, model, T, choices):
    p = np.clip(p_alpha(model, theta, T), 1e-9, 1 - 1e-9)
    ok = ~np.isnan(choices)
    y = choices[ok]
    return -(y * np.log(p[ok]) + (1 - y) * np.log(1 - p[ok])).sum()


def fit(model, T, choices, rng):
    names = PARAMS[model]
    bounds = [BOUNDS[n] for n in names]
    best = None
    for _ in range(N_STARTS):
        start = [rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
                 for lo, hi in bounds]
        if "c" in names:  # most plausible c values are small: sample on a log scale
            start[0] = float(np.exp(rng.uniform(np.log(0.1), np.log(20))))
        res = minimize(neg_log_lik, start, args=(model, T, choices),
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    n = int((~np.isnan(choices)).sum())
    k = len(names)
    return {"model": model, "theta": dict(zip(names, best.x)), "nll": best.fun,
            "aic": 2 * k + 2 * best.fun, "bic": k * np.log(n) + 2 * best.fun,
            "n": n, "k": k}


def simulate(model, theta, T, rng):
    """Fake choices from a model, on the real trial sequences."""
    choices = (rng.random(len(T["D"])) < p_alpha(model, theta, T)).astype(float)
    choices[np.isnan(T["choice_a"])] = np.nan   # keep the same missing trials
    return choices


def random_theta(model, rng):
    if model == "baseline":
        return [rng.uniform(0.3, 0.7)]
    # plausible region: c 0.5-15 (log scale), beta 0.3-0.7, eps 0.02-0.5
    return [float(np.exp(rng.uniform(np.log(0.5), np.log(15)))),
            rng.uniform(0.3, 0.7), rng.uniform(0.02, 0.5)]


# ------------------------------------------------------------ step 1

def recover():
    rng = np.random.default_rng(2026_10_28)
    OUT.mkdir(exist_ok=True)
    lines = ["# Model recovery check (fake data only)", "",
             "Generated by `src/fit_models.py recover`. Fake choices are "
             "simulated on the REAL block-1 trials that are fitted (both "
             "groups already seen) for each LLM, so "
             "the amount and shape of data match the real fit exactly. No "
             "real choices are used here.", ""]

    for name, path in MAIN.items():
        T = fit_trials(path)
        n = int((~np.isnan(T["choice_a"])).sum())
        lines += [f"## {name} trial sequences ({n} choices, {T['run_id'].size and len(set(T['run_id']))} runs)", ""]

        # Parameter recovery: 30 fake data sets per learning model.
        lines += ["### Parameter recovery (30 fake data sets per model)", "",
                  "Correlation between true and recovered dial values "
                  "(1 = perfect), and the typical size of the error.", "",
                  "| Model | Dial | Correlation true vs recovered | Median "
                  "absolute error | True range used |", "|---|---|---|---|---|"]
        for model in ("exemplar", "prototype"):
            true, got = [], []
            for _ in range(30):
                theta = random_theta(model, rng)
                fake = simulate(model, theta, T, rng)
                res = fit(model, T, fake, rng)
                true.append(theta); got.append(list(res["theta"].values()))
            true, got = np.array(true), np.array(got)
            for j, dial in enumerate(PARAMS[model]):
                if dial == "c":   # c works on a multiplicative scale
                    r = np.corrcoef(np.log(true[:, j]), np.log(got[:, j]))[0, 1]
                else:
                    r = np.corrcoef(true[:, j], got[:, j])[0, 1]
                err = np.median(np.abs(true[:, j] - got[:, j]))
                lines.append(f"| {model} | {dial} | {r:.2f} | {err:.3f} | "
                             f"{true[:, j].min():.2f}-{true[:, j].max():.2f} |")
        lines.append("")

        # Model recovery: 30 fake data sets per generating model.
        lines += ["### Model recovery (30 fake data sets per true model)", "",
                  "Rows: the model that made the fake choices. Columns: how "
                  "often each model won on BIC. A good check has large "
                  "numbers on the diagonal.", "",
                  "| True model | baseline wins | prototype wins | exemplar wins |",
                  "|---|---|---|---|"]
        for true_model in ("baseline", "prototype", "exemplar"):
            wins = {"baseline": 0, "prototype": 0, "exemplar": 0}
            for _ in range(30):
                fake = simulate(true_model, random_theta(true_model, rng), T, rng)
                fits = [fit(m, T, fake, rng) for m in wins]
                wins[min(fits, key=lambda f: f["bic"])["model"]] += 1
            lines.append(f"| {true_model} | {wins['baseline']} | "
                         f"{wins['prototype']} | {wins['exemplar']} |")
        lines.append("")
        print(f"done: {name}")

    (OUT / "model_recovery.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {OUT / 'model_recovery.md'}")


# ------------------------------------------------------------ step 2

N_BOOT = 100   # bootstrap resamples of runs, for confidence intervals


def accuracy_by_type(model, theta, T):
    """Model's expected accuracy per type, on the real trials it is fitted to."""
    p = p_alpha(model, theta, T)
    p_correct = np.where(T["truth_a"] == 1, p, 1 - p)
    ok = ~np.isnan(T["choice_a"])
    return {t: float(p_correct[ok & (T["type"] == t)].mean()) for t in TYPES}


def fit_real():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    rng = np.random.default_rng(2026_10_29)
    OUT.mkdir(exist_ok=True)
    lines = ["# Model fits to the real block-1 choices", "",
             "Generated by `src/fit_models.py fit`. Block 1, trials 2-8, "
             "**only trials where both groups had already been seen** (the "
             "both-groups-seen rule; the other trials are in the last table). "
             "One fit per LLM across all six types (option b); attention "
             "equal across the three features. Lower AIC/BIC = better. "
             "Delta = difference from the best model (BIC differences above "
             "10 count as very strong evidence). 95% intervals: bootstrap over "
             f"runs ({N_BOOT} resamples).", ""]
    results, one_group = {}, {}

    for name, path in MAIN.items():
        T_all = build_trials(path)
        both = both_groups_seen(T_all)

        # Descriptive: trials where only ONE group had been seen. Does the
        # LLM put the new sample in the seen group, or in the unseen one?
        ok_all = ~np.isnan(T_all["choice_a"])
        seen_is_a = (T_all["LA"] * T_all["M"]).sum(axis=1) > 0
        nearest = (np.where(T_all["M"] == 1, T_all["D"], 9).min(axis=1) * 3).round()
        one_group[name] = {}
        for dist in (1, 2, 3):
            s = ok_all & ~both & (nearest == dist)
            same = (T_all["choice_a"][s] == seen_is_a[s].astype(float)).mean()
            one_group[name][dist] = (int(s.sum()), float(same))

        T = subset(T_all, np.where(both)[0])
        ok = ~np.isnan(T["choice_a"])
        fits = {m: fit(m, T, T["choice_a"], rng) for m in PARAMS}
        best_bic = min(f["bic"] for f in fits.values())
        best_aic = min(f["aic"] for f in fits.values())

        # Bootstrap: resample whole runs, refit the two learning models.
        runs = np.unique(T["run_id"])
        by_run = {r: np.where(T["run_id"] == r)[0] for r in runs}
        boot = {"exemplar": [], "prototype": []}
        exemplar_wins = 0
        for _ in range(N_BOOT):
            pick = rng.choice(runs, size=len(runs), replace=True)
            idx = np.concatenate([by_run[r] for r in pick])
            Tb = subset(T, idx)
            fe = fit("exemplar", Tb, Tb["choice_a"], rng)
            fp = fit("prototype", Tb, Tb["choice_a"], rng)
            boot["exemplar"].append(list(fe["theta"].values()))
            boot["prototype"].append(list(fp["theta"].values()))
            exemplar_wins += fe["bic"] < fp["bic"]
        boot = {m: np.array(v) for m, v in boot.items()}

        lines += [f"## {name} ({fits['baseline']['n']} choices, "
                  f"{len(runs)} runs)", "",
                  "| Model | Dials | Log-likelihood | AIC | Delta AIC | BIC | "
                  "Delta BIC |", "|---|---|---|---|---|---|---|"]
        for m in ("baseline", "prototype", "exemplar"):
            f = fits[m]
            lines.append(f"| {m} | {f['k']} | {-f['nll']:.1f} | {f['aic']:.1f} | "
                         f"{f['aic'] - best_aic:.1f} | {f['bic']:.1f} | "
                         f"{f['bic'] - best_bic:.1f} |")
        lines += ["", f"Bootstrap (runs resampled {N_BOOT} times, both models "
                  f"refitted): prototype had the lower BIC in **"
                  f"{N_BOOT - exemplar_wins} of {N_BOOT}**, exemplar in **"
                  f"{exemplar_wins} of {N_BOOT}**.", "",
                  "| Model | Dial | Estimate | 95% interval |", "|---|---|---|---|"]
        for m in ("prototype", "exemplar"):
            for j, dial in enumerate(PARAMS[m]):
                lo, hi = np.percentile(boot[m][:, j], [2.5, 97.5])
                lines.append(f"| {m} | {dial} | {fits[m]['theta'][dial]:.3f} | "
                             f"[{lo:.3f}, {hi:.3f}] |")
        lines.append(f"| baseline | beta | {fits['baseline']['theta']['beta']:.3f} | - |")

        # Which types does each model explain? Log-likelihood per type.
        lines += ["", "Where the difference comes from: log-likelihood per "
                  "type (higher = better) and accuracy per type.", "",
                  "| Type | Choices | LL prototype | LL exemplar | Exemplar "
                  "advantage | Observed accuracy | Prototype predicts | "
                  "Exemplar predicts |", "|---|---|---|---|---|---|---|---|"]
        pred = {m: accuracy_by_type(m, list(fits[m]["theta"].values()), T)
                for m in PARAMS}
        observed = {}
        for t in TYPES:
            s = ok & (T["type"] == t)
            Tt = subset(T, np.where(T["type"] == t)[0])
            ll = {m: -neg_log_lik(list(fits[m]["theta"].values()), m, Tt,
                                  Tt["choice_a"]) for m in ("prototype", "exemplar")}
            observed[t] = float((T["choice_a"][s] == T["truth_a"][s]).mean())
            lines.append(f"| {ROMAN[t]} | {int(s.sum())} | {ll['prototype']:.1f} | "
                         f"{ll['exemplar']:.1f} | {ll['exemplar'] - ll['prototype']:+.1f} | "
                         f"{observed[t]:.3f} | {pred['prototype'][t]:.3f} | "
                         f"{pred['exemplar'][t]:.3f} |")
        lines.append("")
        results[name] = {"T": T, "pred": pred, "observed": observed}
        print(f"done: {name}")

    # Figure 5: observed vs predicted accuracy per type.
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    style = {"exemplar": ("#1baf7a", "-", "o"), "prototype": ("#6250d6", "--", "s"),
             "baseline": ("#8a8984", ":", None)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        T = r["T"]
        ok = ~np.isnan(T["choice_a"])
        runs_acc = {t: [] for t in TYPES}
        for run in np.unique(T["run_id"]):
            s = ok & (T["run_id"] == run)
            t = int(T["type"][s][0]) if s.any() else None
            if t:
                runs_acc[t].append((T["choice_a"][s] == T["truth_a"][s]).mean())
        mean = [np.mean(runs_acc[t]) for t in TYPES]
        half = [stats.t.ppf(0.975, len(runs_acc[t]) - 1) * np.std(runs_acc[t], ddof=1)
                / np.sqrt(len(runs_acc[t])) for t in TYPES]
        ax.errorbar(TYPES, mean, yerr=half, fmt="o", color="#0b0b0b", ms=5,
                    capsize=2, lw=1.2, label="Observed (95% CI)", zorder=3)
        for m in ("exemplar", "prototype", "baseline"):
            color, ls, mk = style[m]
            ax.plot(TYPES, [r["pred"][m][t] for t in TYPES], color=color, ls=ls,
                    marker=mk, ms=4, lw=1.8, label=m.capitalize())
        ax.axhline(0.5, color="#b4b2a9", lw=0.8)
        ax.set(xticks=TYPES, xticklabels=[ROMAN[t] for t in TYPES], ylim=(0, 1),
               xlabel="Type", title=name)
    axes[0].set_ylabel("Block-1 accuracy\n(both groups already seen)")
    axes[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_model_fits.{ext}")
    plt.close(fig)

    names = list(one_group)
    lines += ["## Trials where only ONE group had been seen (not fitted)", "",
              "Every trial 2, plus later trials in runs where the second group "
              "appeared late. Both learning models must answer 'the group "
              "already seen' with certainty here. What the LLMs actually did, "
              "by how different the new sample was from the nearest sample "
              "seen so far:", "",
              f"| Nearest seen sample | {names[0]}: trials | picks the seen "
              f"group | {names[1]}: trials | picks the seen group |",
              "|---|---|---|---|---|"]
    for dist in (1, 2, 3):
        a, b = one_group[names[0]][dist], one_group[names[1]][dist]
        lines.append(f"| {dist} feature{'s' if dist > 1 else ''} away | {a[0]} | "
                     f"{a[1]:.2f} | {b[0]} | {b[1]:.2f} |")
    lines.append("")

    (OUT / "model_fits.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {OUT / 'model_fits.md'} and {OUT / 'fig5_model_fits.png'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "recover":
        recover()
    elif mode == "fit":
        fit_real()
    else:
        raise SystemExit(__doc__)

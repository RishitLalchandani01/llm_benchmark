"""
Cross-model comparison of a single question paper.

The runner already gives every model the *identical* item set in the identical
order (one seeded permutation, reused for each model), so the design is paired.
This module is what exploits that pairing: it joins the per-item rows back
together on `question_id` and answers the questions the aggregate tables cannot.

    which questions separated the 3B model from the 8B?
    is an 11-point gap real, or within paired sampling noise?
    do two models fail on the same items, or on different ones?
    what does a decade of parameters actually buy, with an interval on it?

Everything here reads `results/raw_responses.csv` and writes CSVs. No API calls
are made, so the analysis can be re-run and revised as often as you like against
data you have already paid for:

    python -m benchmark.compare results/raw_responses.csv

Statistics are implemented directly rather than pulled from scipy: Wilson
intervals, McNemar's exact test and Cohen's kappa are each a few lines of
closed-form arithmetic, and a dependency for three formulas is not worth it.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Z95 = 1.959963984540054  # two-sided normal quantile at 95%

# Rows are keyed by this pair; a duplicate means a resume cache replayed an item.
_KEY = ["model_name", "question_id"]


# --------------------------------------------------------------------------- #
# Loading and pairing
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw_responses.csv and normalise the columns this module needs."""
    df = pd.read_csv(path)
    missing = {"model_name", "question_id", "is_correct"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    # Columns added in later versions of the harness may be absent in old runs.
    for col, default in (
        ("question_type", "mcq"), ("domain", ""), ("graded", True),
        ("score", np.nan), ("max_score", np.nan), ("params_b", np.nan),
        ("confidence", np.nan), ("predicted_letter", None), ("correct_letter", ""),
        ("latency_s", np.nan), ("status", "ok"), ("parse_ok", True),
        ("answer_text", None), ("judge_reasoning", None),
    ):
        if col not in df.columns:
            df[col] = default

    df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(("true", "1", "1.0"))
    df["graded"] = df["graded"].astype(str).str.lower().isin(("true", "1", "1.0"))
    df["question_type"] = df["question_type"].fillna("mcq").astype(str).str.lower()

    # Keep the last record for a repeated (model, item): a resumed run appends.
    before = len(df)
    df = df.drop_duplicates(subset=_KEY, keep="last").reset_index(drop=True)
    if len(df) < before:
        print(f"  [compare] dropped {before - len(df)} duplicate (model, item) row(s)")
    return df


def item_credit(df: pd.DataFrame) -> pd.Series:
    """
    Credit in [0, 1] per row: exact-match for multiple choice, rubric fraction
    for free response, NaN where nothing could score the item.
    """
    frq = df["question_type"].eq("frq")
    score = pd.to_numeric(df["score"], errors="coerce")
    max_score = pd.to_numeric(df["max_score"], errors="coerce").replace(0, np.nan)
    credit = pd.Series(np.where(frq, score / max_score, df["is_correct"].astype(float)),
                       index=df.index)
    return credit.where(df["graded"], np.nan).clip(0, 1)


def paired_subset(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Restrict to items every model actually answered, and say so when that bites.

    Pairing is an assumption, not a guarantee: changing --limit between runs and
    then using --resume leaves models with different item sets, and every paired
    test below would be silently invalid. So we intersect and report.
    """
    models = sorted(df["model_name"].unique())
    per_model = {m: set(g["question_id"]) for m, g in df.groupby("model_name")}
    common = set.intersection(*per_model.values()) if per_model else set()
    union = set.union(*per_model.values()) if per_model else set()

    if verbose:
        if len(common) == len(union):
            print(f"  [compare] {len(models)} models, {len(common)} items, fully paired")
        else:
            print(f"  [compare] WARNING: item sets differ across models. "
                  f"{len(union)} distinct items, only {len(common)} answered by all "
                  f"{len(models)} models; paired statistics use the {len(common)} common items.")
            for m in models:
                gap = len(union) - len(per_model[m])
                if gap:
                    print(f"             {m}: missing {gap} item(s)")
        if not common:
            print("  [compare] no item is shared by every model -- paired output will be empty")

    return df[df["question_id"].isin(common)].copy(), models


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_interval(k: float, n: float, z: float = Z95) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion, as percentages.
    Preferred to the normal approximation because it stays inside [0, 1] and
    behaves at the tails, where a 1B model on a hard domain actually sits.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    k = max(0.0, min(float(k), float(n)))
    denom = n + z * z
    centre = (k + z * z / 2) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z * z / 4)
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value for paired binary outcomes.

    `b` = items the first model got right and the second wrong, `c` the reverse.
    Concordant pairs carry no information about a difference and are excluded by
    construction: under the null, b ~ Binomial(b + c, 1/2).
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    m = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """
    Cohen's kappa over the *chosen option letters*, not over correctness.

    Two models can agree on accuracy while picking different wrong answers;
    agreement on the letter is the stronger signal that they share a failure
    mode (the same attractive distractor), which correctness hides.
    """
    pairs = [(x, y) for x, y in zip(a, b)]
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in pairs if x == y) / n
    cats = {x for x, _ in pairs} | {y for _, y in pairs}
    expected = sum(
        (sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n)
        for c in cats
    )
    if abs(1 - expected) < 1e-12:
        return float("nan")
    return (observed - expected) / (1 - expected)


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values, preserving input order.
    Controls family-wise error across all pairwise comparisons without the
    conservatism of plain Bonferroni.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------- #
# 1. Question-by-question matrix
# --------------------------------------------------------------------------- #
def item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per question, one column per model: the literal side-by-side view of
    the same paper answered by different models.

    Multiple-choice columns hold the chosen letter; free-response columns hold
    the awarded fraction, since there is no letter to show.
    """
    df = df.copy()
    df["_credit"] = item_credit(df)
    frq = df["question_type"].eq("frq")
    df["_cell"] = np.where(
        frq,
        df["_credit"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
        df["predicted_letter"].fillna("?").astype(str),
    )

    answers = df.pivot(index="question_id", columns="model_name", values="_cell")
    credit = df.pivot(index="question_id", columns="model_name", values="_credit")

    meta = (
        df.groupby("question_id")
        .agg(domain=("domain", "first"),
             question_type=("question_type", "first"),
             correct=("correct_letter", "first"))
    )
    out = meta.join(answers)
    out["n_models_scored"] = credit.notna().sum(axis=1)
    out["n_correct"] = (credit >= 1.0).sum(axis=1)
    out["mean_credit"] = credit.mean(axis=1, skipna=True).round(3)
    # Fraction of the suite that solved it: 1.0 = trivial, 0.0 = nobody.
    out["item_difficulty"] = (
        out["n_correct"] / out["n_models_scored"].replace(0, np.nan)
    ).round(3)
    return out.sort_values(["domain", "question_id"])


# --------------------------------------------------------------------------- #
# 2. Pairwise model comparison
# --------------------------------------------------------------------------- #
def pairwise_models(df: pd.DataFrame, by_domain: bool = False) -> pd.DataFrame:
    """
    Every model pair, with the 2x2 discordance table, agreement, letter-level
    kappa, and McNemar's exact test with Holm-Bonferroni adjustment.

    Only multiple-choice items enter the significance test: McNemar is defined
    for paired *binary* outcomes, and rubric-scored free response is not binary.
    """
    mcq = df[df["question_type"].ne("frq") & df["graded"]].copy()
    if mcq.empty:
        return pd.DataFrame()

    groups: List[Tuple[str, pd.DataFrame]] = (
        list(mcq.groupby("domain")) if by_domain else [("(all domains)", mcq)]
    )

    rows: List[Dict[str, object]] = []
    for scope, chunk in groups:
        correct = chunk.pivot(index="question_id", columns="model_name", values="is_correct")
        letters = chunk.pivot(index="question_id", columns="model_name",
                              values="predicted_letter").fillna("∅")
        models = sorted(correct.columns)

        for a, b in combinations(models, 2):
            pair = correct[[a, b]].dropna()
            if pair.empty:
                continue
            ca = pair[a].astype(bool)
            cb = pair[b].astype(bool)
            both = int((ca & cb).sum())
            neither = int((~ca & ~cb).sum())
            only_a = int((ca & ~cb).sum())
            only_b = int((~ca & cb).sum())
            n = both + neither + only_a + only_b

            lp = letters.loc[pair.index, [a, b]]
            rows.append({
                "scope": scope,
                "model_a": a,
                "model_b": b,
                "n_paired_items": n,
                "both_correct": both,
                "both_wrong": neither,
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "acc_a_pct": round(100 * (both + only_a) / n, 2) if n else np.nan,
                "acc_b_pct": round(100 * (both + only_b) / n, 2) if n else np.nan,
                "diff_pts": round(100 * (only_a - only_b) / n, 2) if n else np.nan,
                "correctness_agreement_pct": round(100 * (both + neither) / n, 2) if n else np.nan,
                "letter_agreement_pct": round(100 * (lp[a] == lp[b]).mean(), 2),
                "letter_kappa": round(cohens_kappa(lp[a].tolist(), lp[b].tolist()), 4),
                "mcnemar_p": mcnemar_exact(only_a, only_b),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Adjust within each scope: the comparison family is the set of pairs.
    out["mcnemar_p_holm"] = np.nan
    for scope, idx in out.groupby("scope").groups.items():
        out.loc[idx, "mcnemar_p_holm"] = holm_bonferroni(out.loc[idx, "mcnemar_p"].tolist())
    out["significant_at_05"] = out["mcnemar_p_holm"] < 0.05
    out["mcnemar_p"] = out["mcnemar_p"].round(6)
    out["mcnemar_p_holm"] = out["mcnemar_p_holm"].round(6)
    return out.sort_values(["scope", "diff_pts"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Confidence intervals on per-domain accuracy
# --------------------------------------------------------------------------- #
def accuracy_with_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (model, domain) accuracy with a Wilson 95% interval.

    The interval is computed on multiple-choice items only, where accuracy is a
    genuine binomial proportion. Free-response points are reported alongside but
    get a bootstrap interval instead (see marginal_utility), because a partial
    credit mean is not a proportion and Wilson does not apply to it.
    """
    rows: List[Dict[str, object]] = []
    for (model, domain), g in df.groupby(["model_name", "domain"], dropna=False):
        mcq = g[g["question_type"].ne("frq") & g["graded"]]
        frq = g[g["question_type"].eq("frq") & g["graded"]]
        k = int(mcq["is_correct"].sum())
        n = int(len(mcq))
        lo, hi = wilson_interval(k, n)
        pts = pd.to_numeric(frq["score"], errors="coerce").sum()
        avail = pd.to_numeric(frq["max_score"], errors="coerce").sum()
        rows.append({
            "model_name": model,
            "params_b": g["params_b"].dropna().iloc[0] if g["params_b"].notna().any() else np.nan,
            "domain": domain,
            "n_mcq_graded": n,
            "n_correct": k,
            "accuracy_pct": round(100 * k / n, 2) if n else np.nan,
            "wilson_lo_pct": round(lo, 2),
            "wilson_hi_pct": round(hi, 2),
            "wilson_half_width_pts": round((hi - lo) / 2, 2) if n else np.nan,
            "n_frq_graded": int(len(frq)),
            "frq_points": round(float(pts), 2),
            "frq_points_available": round(float(avail), 2),
            "frq_score_pct": round(100 * pts / avail, 2) if avail else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["domain", "accuracy_pct"],
                                          ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Marginal utility per decade of parameters
# --------------------------------------------------------------------------- #
def _macro_utility_matrix(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    {domain: array of shape (n_items, n_models)} of per-item credit, plus the
    model order. Used both for the point estimate and the bootstrap, so the two
    can never disagree about the estimator.
    """
    d = df.copy()
    d["_credit"] = item_credit(d)
    piv = d.pivot_table(index=["domain", "question_id"], columns="model_name",
                        values="_credit", aggfunc="first")
    models = list(piv.columns)
    arrays = {dom: piv.loc[dom].to_numpy(dtype=float) for dom in piv.index.get_level_values(0).unique()}
    return arrays, models


def _macro_utility(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Unweighted macro-average of per-domain mean credit, matching summarize()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column for a model
        per_domain = [np.nanmean(a, axis=0) for a in arrays.values()]
    return np.nanmean(np.vstack(per_domain), axis=0) * 100


def _bootstrap_utility(
    arrays: Dict[str, np.ndarray], n_boot: int, seed: int, chunk: int = 250
) -> np.ndarray:
    """
    Stratified item bootstrap: resample items *within* each domain, so the
    resampling distribution matches the macro-average estimator rather than a
    pooled one. Returns an (n_boot, n_models) array of utilities.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, next(iter(arrays.values())).shape[1]), dtype=float)
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while done < n_boot:
            size = min(chunk, n_boot - done)
            acc = np.zeros((size, out.shape[1]), dtype=float)
            for a in arrays.values():
                idx = rng.integers(0, a.shape[0], size=(size, a.shape[0]))
                acc += np.nanmean(a[idx], axis=1)
            out[done:done + size] = acc / len(arrays) * 100
            done += size
    return out


def marginal_utility(
    df: pd.DataFrame, n_boot: int = 10_000, seed: int = 1234, verbose: bool = True
) -> pd.DataFrame:
    """
    Marginal utility over adjacent scale intervals: accuracy points gained per
    decade of parameters, with a bootstrap interval.

    A logarithmic denominator is not cosmetic. The scaling literature reports
    power-law dependence, so a linear denominator would make the 1B->3B and
    8B->70B intervals incomparable by construction.

    Models with no disclosed parameter count (closed frontier systems) are
    excluded: an interval needs two abscissae, and imputing one would put a
    guess inside the paper's central quantity.
    """
    known = df[df["params_b"].notna()].copy()
    if known.empty:
        if verbose:
            print("  [compare] no model has a disclosed params_b -- marginal utility skipped")
        return pd.DataFrame()

    arrays, models = _macro_utility_matrix(known)
    if not arrays:
        return pd.DataFrame()

    point = _macro_utility(arrays)
    params = {
        m: float(known.loc[known["model_name"].eq(m), "params_b"].dropna().iloc[0])
        for m in models
    }
    ordered = sorted(models, key=lambda m: params[m])
    if len(ordered) < 2:
        if verbose:
            print("  [compare] need >= 2 models with disclosed params_b for an interval")
        return pd.DataFrame()

    boot = _bootstrap_utility(arrays, n_boot=n_boot, seed=seed)
    col = {m: i for i, m in enumerate(models)}

    rows: List[Dict[str, object]] = []
    for lo_m, hi_m in zip(ordered, ordered[1:]):
        d_log = math.log10(params[hi_m]) - math.log10(params[lo_m])
        if d_log <= 0:
            continue
        d_u = point[col[hi_m]] - point[col[lo_m]]
        mu_boot = (boot[:, col[hi_m]] - boot[:, col[lo_m]]) / d_log
        rows.append({
            "interval": f"{lo_m} → {hi_m}",
            "params_lo_b": params[lo_m],
            "params_hi_b": params[hi_m],
            "fold_increase": round(params[hi_m] / params[lo_m], 2),
            "delta_log10_params": round(d_log, 3),
            "utility_lo_pct": round(point[col[lo_m]], 2),
            "utility_hi_pct": round(point[col[hi_m]], 2),
            "delta_utility_pts": round(d_u, 2),
            "mu_pts_per_decade": round(d_u / d_log, 2),
            "mu_ci_lo": round(float(np.nanpercentile(mu_boot, 2.5)), 2),
            "mu_ci_hi": round(float(np.nanpercentile(mu_boot, 97.5)), 2),
            "n_bootstrap": n_boot,
        })
    return pd.DataFrame(rows)


def marginal_utility_by_format(
    df: pd.DataFrame, n_boot: int = 2_000, seed: int = 1234
) -> pd.DataFrame:
    """
    The same intervals computed separately over multiple-choice and free-response
    items -- the paper's central test (H3), that saturation is format-conditional
    rather than a property of the capability.
    """
    frames = []
    for label, subset in (
        ("mcq", df[df["question_type"].ne("frq")]),
        ("frq", df[df["question_type"].eq("frq")]),
    ):
        if subset.empty or not subset["graded"].any():
            continue
        part = marginal_utility(subset, n_boot=n_boot, seed=seed, verbose=False)
        if part.empty:
            continue
        part.insert(0, "format", label)
        frames.append(part)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Ratio of free-response to multiple-choice marginal utility per interval:
    # > 1 means scale is still buying capability the MCQ metric cannot see.
    wide = out.pivot_table(index="interval", columns="format",
                           values="mu_pts_per_decade", aggfunc="first")
    if {"mcq", "frq"} <= set(wide.columns):
        wide["frq_over_mcq_ratio"] = (
            wide["frq"] / wide["mcq"].replace(0, np.nan)
        ).round(3)
        out = out.merge(wide[["frq_over_mcq_ratio"]], on="interval", how="left")
    return out


# --------------------------------------------------------------------------- #
# 5. Discriminating items
# --------------------------------------------------------------------------- #
def discriminating_items(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Items that split the suite, ranked.

    Two things are worth reading here. An evenly split item discriminates
    between models and is where a scaling claim is actually earned. An
    *inversion* -- the largest model wrong where a smaller one is right -- is
    evidence against a monotone reading of scale, and is the qualitative
    material behind any claim about which items the frontier still misses.
    """
    d = df[df["graded"]].copy()
    if d.empty:
        return pd.DataFrame()
    d["_credit"] = item_credit(d)

    sized = d[d["params_b"].notna()]
    largest = (sized.loc[sized["params_b"].idxmax(), "model_name"]
               if not sized.empty else None)

    rows: List[Dict[str, object]] = []
    for qid, g in d.groupby("question_id"):
        credit = g.set_index("model_name")["_credit"].dropna()
        if credit.empty:
            continue
        n = len(credit)
        n_right = int((credit >= 1.0).sum())
        solved = set(credit[credit >= 1.0].index)
        inversion = False
        if largest in credit.index and largest not in solved:
            smaller = sized.loc[sized["model_name"].isin(solved)
                                & sized["params_b"].lt(
                                    sized.loc[sized["model_name"].eq(largest),
                                              "params_b"].iloc[0]), "model_name"]
            inversion = not smaller.empty

        rows.append({
            "question_id": qid,
            "domain": g["domain"].iloc[0],
            "question_type": g["question_type"].iloc[0],
            "n_models_scored": n,
            "n_correct": n_right,
            "item_difficulty": round(n_right / n, 3),
            "split_score": min(n_right, n - n_right),   # higher = more evenly split
            "scale_inversion": inversion,
            "solved_by": ", ".join(sorted(solved)) or "(none)",
            "missed_by": ", ".join(sorted(set(credit.index) - solved)) or "(none)",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ranked = out.sort_values(["split_score", "scale_inversion", "item_difficulty"],
                             ascending=[False, False, True]).reset_index(drop=True)
    # Rates over EVERY scored item, carried alongside the truncated listing: the
    # ranking puts inversions first, so counting them within the head would
    # report the sort order rather than the data.
    head = ranked.head(top_n).copy()
    head.attrs["n_items_total"] = int(len(out))
    head.attrs["n_inversions_total"] = int(out["scale_inversion"].sum())
    head.attrs["n_unanimous_correct"] = int((out["item_difficulty"] >= 1.0).sum())
    head.attrs["n_unanimous_wrong"] = int((out["item_difficulty"] <= 0.0).sum())
    return head


# --------------------------------------------------------------------------- #
# 6. Free-response side-by-side
# --------------------------------------------------------------------------- #
def frq_report(df: pd.DataFrame) -> str:
    """
    Markdown laying every model's free-response answer for one question next to
    the others, with the judge's score and reasoning.

    This is the document to read before trusting Table VIII of the paper: a
    judged score is one model's opinion, and the only way to bound that is to
    look at the answers and the reasoning by hand.
    """
    frq = df[df["question_type"].eq("frq")].copy()
    if frq.empty:
        return "# Free-response comparison\n\nNo free-response items in this run.\n"

    lines = ["# Free-response comparison", ""]
    lines.append("Every model's answer to the same question, with the judge's award. "
                 "Scores are produced by a judge model, not ground truth — "
                 "spot-check a sample and report the agreement.")
    lines.append("")

    for qid, g in frq.groupby("question_id"):
        g = g.copy()
        g["_frac"] = item_credit(g)
        lines.append(f"## {qid}  ·  {g['domain'].iloc[0]}")
        lines.append("")
        lines.append("| Model | Award | Fraction |")
        lines.append("|---|---|---|")
        for _, r in g.sort_values("_frac", ascending=False).iterrows():
            award = ("—" if pd.isna(r["score"])
                     else f"{r['score']:.1f} / {r['max_score']:.0f}")
            frac = "—" if pd.isna(r["_frac"]) else f"{r['_frac']:.0%}"
            lines.append(f"| {r['model_name']} | {award} | {frac} |")
        lines.append("")
        for _, r in g.sort_values("_frac", ascending=False).iterrows():
            lines.append(f"### {r['model_name']}")
            if isinstance(r.get("judge_reasoning"), str) and r["judge_reasoning"].strip():
                lines.append(f"*Judge:* {r['judge_reasoning'].strip()}")
                lines.append("")
            answer = r.get("answer_text")
            lines.append("```text")
            lines.append(str(answer).strip() if isinstance(answer, str) and answer.strip()
                         else "(no answer recorded)")
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 7. Calibration (enabled by persisting confidence)
# --------------------------------------------------------------------------- #
def calibration(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """
    Accuracy within bands of self-reported confidence, per model.

    This is the measurement that decides whether a compact model can be trusted
    to answer alone: a cascade that escalates on low confidence only works if
    confidence tracks correctness. Reports expected calibration error per model.
    """
    d = df[df["question_type"].ne("frq") & df["graded"]].copy()
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
    d = d[d["confidence"].notna()]
    if d.empty:
        print("  [compare] no confidence values recorded -- calibration skipped "
              "(runs made before confidence was persisted)")
        return pd.DataFrame()

    edges = np.linspace(0, 1, bins + 1)
    d["_bin"] = pd.cut(d["confidence"].clip(0, 1), bins=edges,
                       include_lowest=True, right=True)

    rows: List[Dict[str, object]] = []
    for model, g in d.groupby("model_name"):
        ece = 0.0
        for interval, gb in g.groupby("_bin", observed=True):
            n = len(gb)
            acc = gb["is_correct"].mean()
            conf = gb["confidence"].mean()
            ece += (n / len(g)) * abs(acc - conf)
            rows.append({
                "model_name": model,
                "confidence_bin": str(interval),
                "n_items": n,
                "mean_confidence_pct": round(100 * conf, 2),
                "accuracy_pct": round(100 * acc, 2),
                "gap_pts": round(100 * (conf - acc), 2),
            })
        rows.append({
            "model_name": model, "confidence_bin": "ALL (expected calib. error)",
            "n_items": len(g),
            "mean_confidence_pct": round(100 * g["confidence"].mean(), 2),
            "accuracy_pct": round(100 * g["is_correct"].mean(), 2),
            "gap_pts": round(100 * ece, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compare_all(
    raw: pd.DataFrame,
    out_dir: str | Path,
    *,
    n_boot: int = 10_000,
    seed: int = 1234,
    top_items: int = 25,
    by_domain: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run every comparison and write the outputs. Returns the frames."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired, models = paired_subset(raw, verbose=verbose)
    if paired.empty:
        print("  [compare] nothing to compare")
        return {}

    frames: Dict[str, pd.DataFrame] = {
        "comparison_by_question": item_matrix(paired),
        "pairwise_models": pairwise_models(paired, by_domain=False),
        "accuracy_with_intervals": accuracy_with_intervals(paired),
        "marginal_utility": marginal_utility(paired, n_boot=n_boot, seed=seed),
        "marginal_utility_by_format": marginal_utility_by_format(
            paired, n_boot=max(1000, n_boot // 5), seed=seed),
        "discriminating_items": discriminating_items(paired, top_n=top_items),
        "calibration": calibration(paired),
    }
    if by_domain:
        frames["pairwise_models_by_domain"] = pairwise_models(paired, by_domain=True)

    written: List[Path] = []
    for name, frame in frames.items():
        if frame is None or frame.empty:
            continue
        path = out / f"{name}.csv"
        frame.to_csv(path, index=(name == "comparison_by_question"))
        written.append(path)

    md = out / "frq_comparison.md"
    md.write_text(frq_report(paired), encoding="utf-8")
    written.append(md)

    if verbose:
        _print_highlights(frames, models)
        print("\nwrote:")
        for path in written:
            print(f"  {path}")
    return frames


def _print_highlights(frames: Dict[str, pd.DataFrame], models: Sequence[str]) -> None:
    """Console summary: the two tables a reader actually wants to see."""
    pd.set_option("display.width", 200, "display.max_columns", 40)

    mu = frames.get("marginal_utility")
    if mu is not None and not mu.empty:
        print("\n" + "=" * 72)
        print("MARGINAL UTILITY PER DECADE OF PARAMETERS")
        print("=" * 72)
        print(mu[["interval", "fold_increase", "delta_utility_pts",
                  "mu_pts_per_decade", "mu_ci_lo", "mu_ci_hi"]].to_string(index=False))

    fmt = frames.get("marginal_utility_by_format")
    if fmt is not None and not fmt.empty and "frq_over_mcq_ratio" in fmt.columns:
        print("\nMARGINAL UTILITY BY TASK FORMAT")
        print(fmt.pivot_table(index="interval", columns="format",
                              values="mu_pts_per_decade", aggfunc="first").to_string())

    pw = frames.get("pairwise_models")
    if pw is not None and not pw.empty:
        print("\n" + "=" * 72)
        print("PAIRWISE COMPARISON (paired items, McNemar exact + Holm)")
        print("=" * 72)
        print(pw[["model_a", "model_b", "n_paired_items", "only_a_correct",
                  "only_b_correct", "diff_pts", "letter_kappa", "mcnemar_p_holm",
                  "significant_at_05"]].to_string(index=False))

    di = frames.get("discriminating_items")
    if di is not None and not di.empty:
        total = di.attrs.get("n_items_total", len(di))
        inv = di.attrs.get("n_inversions_total", 0)
        easy = di.attrs.get("n_unanimous_correct", 0)
        hard = di.attrs.get("n_unanimous_wrong", 0)
        print("\n" + "=" * 72)
        print("ITEM-LEVEL STRUCTURE")
        print("=" * 72)
        print(f"  {total} scored items: {easy} solved by every model, "
              f"{hard} solved by none, {total - easy - hard} discriminate between models")
        print(f"  {inv} item(s) ({100 * inv / total:.1f}%) show a scale inversion -- the "
              f"largest model wrong where a smaller one is right")
        print(f"  the {len(di)} most evenly split are listed in discriminating_items.csv")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark.compare",
        description="Compare how different models answered the same question paper. "
                    "Reads an existing raw_responses.csv; makes no API calls.",
    )
    p.add_argument("raw_csv", nargs="?", default="results/raw_responses.csv",
                   help="path to results/raw_responses.csv (default: %(default)s)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the comparison files "
                        "(default: <raw_csv dir>/comparison)")
    p.add_argument("--bootstrap", type=int, default=10_000,
                   help="bootstrap resamples for marginal-utility intervals "
                        "(default: %(default)s)")
    p.add_argument("--seed", type=int, default=1234, help="bootstrap seed")
    p.add_argument("--top-items", type=int, default=25,
                   help="how many discriminating items to list")
    p.add_argument("--no-by-domain", action="store_true",
                   help="skip the per-domain pairwise breakdown")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        print(f"error: {raw_path} not found. Run a benchmark first, or pass the path.")
        return 1

    print(f"reading {raw_path}")
    raw = load_raw(raw_path)
    out_dir = Path(args.out_dir) if args.out_dir else raw_path.parent / "comparison"
    compare_all(raw, out_dir, n_boot=args.bootstrap, seed=args.seed,
                top_items=args.top_items, by_domain=not args.no_by_domain)
    return 0


if __name__ == "__main__":
    sys.exit(main())

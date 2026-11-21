"""Step 6: metric tables + charts + a markdown report.

    python -m evaluation.analyze

Reads  evaluation/results/scored.jsonl   (from score_runs)
       evaluation/results/retrieval_eval.jsonl   (from run_retrieval_eval, optional)
Writes evaluation/results/report.md and PNG charts next to it.

Cost: fill evaluation/prices.json with $/1M tokens for your models; otherwise cost is blank.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.common import EVAL_DIR, RESULTS_DIR  # noqa: E402
from src.utils.run_logger import read_jsonl  # noqa: E402

# Validated default categorical palette (dataviz skill), light surface. Slots 1-3 are
# validated for all-pairs use, so keep <= 3 series per chart. Low-contrast slots carry
# visible value labels (the skill's relief rule).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

MODE_ORDER = ["direct", "vector", "bm25", "hybrid", "hybrid_fy", "oracle", "agentic", "agentic_verified"]


def md_table(df, floatfmt="{:.3f}"):
    if df.empty:
        return "_no data_\n"
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def order_modes(index):
    known = [m for m in MODE_ORDER if m in index]
    return known + [m for m in index if m not in known]


def style_axes(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def grouped_bars(df, title, ylabel, path, ylim=(0, 1.16), fmt="{:.2f}"):
    """df: rows = x categories, columns = series (max 3)."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n = len(df.columns)
    width = 0.8 / max(n, 1)
    x = np.arange(len(df.index))
    for i, col in enumerate(df.columns):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (n - 1) / 2) * width, np.nan_to_num(vals), width * 0.92,
                      color=SERIES[i % len(SERIES)], label=str(col), edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, fmt.format(v),
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, color=INK2)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, 1.01, 0.2))          # keep the legend band above the 1.0 line
    style_axes(ax, title, ylabel)
    if n >= 2:
        ax.legend(frameon=False, labelcolor=INK2, loc="upper left", ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def latency_chart(lat, path):
    """Horizontal stacked bars: retrieval vs LLM seconds per mode."""
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(lat) + 2.2))
    y = np.arange(len(lat))
    retr = lat["retrieval_s"].to_numpy()
    llm = lat["llm_s"].to_numpy()
    ax.barh(y, retr, 0.55, color=SERIES[0], label="Retrieval", edgecolor=SURFACE, linewidth=2)
    ax.barh(y, llm, 0.55, left=retr, color=SERIES[1], label="LLM generation", edgecolor=SURFACE, linewidth=2)
    for yi, (r, l) in enumerate(zip(retr, llm)):
        ax.text(r + l + 0.03 * max((retr + llm).max(), 1e-9), yi,
                f"retrieval {r * 1000:,.0f} ms  |  LLM {l:.2f} s", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(lat.index, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, (retr + llm).max() * 1.55 if (retr + llm).max() > 0 else 1)
    style_axes(ax, "Mean latency per question", None)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("seconds", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_prices():
    p = EVAL_DIR / "prices.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(read_jsonl(RESULTS_DIR / "scored.jsonl"))
    retrieval = pd.DataFrame(read_jsonl(RESULTS_DIR / "retrieval_eval.jsonl"))
    report = ["# Results\n"]

    if not runs.empty:
        runs["model"] = runs["model"].fillna("?")
        scorable = runs[runs["correct"].notna()].copy()
        scorable["correct"] = scorable["correct"].astype(float)

        # 1. headline: correctness by mode x model
        acc = scorable.pivot_table(index="mode", columns="model", values="correct", aggfunc="mean")
        acc = acc.reindex(order_modes(list(acc.index)))
        report += ["## Correctness by architecture and model\n",
                   f"{len(scorable)} scored runs ({len(runs) - len(scorable)} excluded: errors or no gold label).\n",
                   md_table(acc)]
        if 1 <= acc.shape[1] <= 3:
            grouped_bars(acc, "Answer correctness by architecture", "share correct",
                         RESULTS_DIR / "chart_accuracy.png")
            report.append("![accuracy](chart_accuracy.png)\n")

        # 2. by category
        by_cat = scorable.pivot_table(index="category", columns="mode", values="correct", aggfunc="mean")
        by_cat = by_cat[order_modes(list(by_cat.columns))]
        report += ["## Correctness by question category\n", md_table(by_cat)]

        # 3. abstention behaviour
        ab = []
        for (mode, model), g in runs[runs["error"].isna()].groupby(["mode", "model"]):
            should, did = g["should_abstain"].astype(bool), g["abstained"].astype(bool)
            tp, fp, fn = (should & did).sum(), (~should & did).sum(), (should & ~did).sum()
            ab.append({"mode": mode, "model": model,
                       "abstain_recall": tp / (tp + fn) if tp + fn else np.nan,
                       "abstain_precision": tp / (tp + fp) if tp + fp else np.nan,
                       "false_abstain_rate": fp / (~should).sum() if (~should).sum() else np.nan})
        if ab:
            ab = pd.DataFrame(ab).set_index(["mode", "model"])
            report += ["## Insufficient-evidence behaviour\n",
                       "recall = unanswerable questions correctly declined; false-abstain = answerable questions wrongly declined.\n",
                       md_table(ab)]

        # 4. latency
        runs["retrieval_s"] = runs["retrieval_ms"].fillna(0) / 1000
        runs["llm_s"] = runs["llm_ms"].fillna(0) / 1000
        runs["total_s"] = runs["total_ms"].fillna(0) / 1000
        lat = runs[runs["error"].isna()].groupby("mode")[["retrieval_s", "llm_s", "total_s"]].mean()
        lat = lat.reindex(order_modes(list(lat.index)))
        p95 = runs[runs["error"].isna()].groupby("mode")["total_s"].quantile(0.95).rename("total_p95_s")
        report += ["## Latency (seconds)\n", md_table(lat.join(p95))]
        if len(lat):
            latency_chart(lat, RESULTS_DIR / "chart_latency.png")
            report.append("![latency](chart_latency.png)\n")

        # 5. tokens and cost
        tok = runs[runs["error"].isna()].groupby(["mode", "model"])[["prompt_tokens", "completion_tokens"]].mean()
        prices = load_prices()

        def cost(row):
            p = prices.get(row.name[1]) or prices.get(row.name[1].split(":", 1)[-1])
            if not p:
                return np.nan
            return (row["prompt_tokens"] * p["input_per_1m"] + row["completion_tokens"] * p["output_per_1m"]) / 1e6

        tok["avg_cost_usd"] = tok.apply(cost, axis=1)
        report += ["## Tokens and cost per question\n",
                   "_Fill evaluation/prices.json to populate cost._\n" if not prices else "",
                   md_table(tok, "{:.5f}")]

        # 6. citations
        cit = runs[runs["mode"].isin(["vector", "bm25", "hybrid", "hybrid_fy", "oracle"])]
        cit = cit.groupby("mode")[["has_citation", "citation_valid_rate", "citation_relevant_rate"]].mean()
        report += ["## Citation behaviour (attribution proxy)\n",
                   "citation_relevant_rate = share of cited chunks that contain gold evidence. This does not check that "
                   "the claim is entailed by the passage - that is the claim-level verification step.\n",
                   md_table(cit)]

        # 7. failure triage
        fails = scorable[scorable["correct"] == 0.0]
        if len(fails):
            report += ["## Failure triage (wrong answers)\n"]
            tri = []
            for _, r in fails.iterrows():
                mode = r["mode"]
                if mode == "direct":
                    cause = "no retrieval (model memory)"
                elif r.get("false_abstain"):
                    cause = "declined an answerable question"
                elif r.get("missed_abstain"):
                    cause = "answered an unanswerable question"
                elif mode == "oracle":
                    cause = "generation/reasoning (gold evidence was supplied)"
                else:
                    cause = "retrieval or generation (compare with oracle)"
                tri.append(cause)
            report.append(md_table(pd.Series(tri).value_counts().rename("runs").to_frame("runs")))

    if not retrieval.empty:
        cols = [c for c in ["mrr", "hit@1", "hit@3", "hit@5", "hit@10", "full@5", "retrieval_ms"] if c in retrieval]
        rt = retrieval.groupby("mode")[cols].mean()
        rt = rt.reindex(order_modes(list(rt.index)))
        report += ["## Retrieval quality (no LLM)\n", md_table(rt)]
        chart_cols = [c for c in ["hit@5", "full@5", "mrr"] if c in rt]
        if chart_cols:
            grouped_bars(rt[chart_cols], "Retrieval quality by method", "score",
                         RESULTS_DIR / "chart_retrieval.png")
            report.append("![retrieval](chart_retrieval.png)\n")

    out = RESULTS_DIR / "report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

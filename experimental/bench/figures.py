"""Plot helpers for the Trust Benchmark.

Loads results/raw.csv and renders the deck's headline figures. Each function
is small + standalone so they can be re-run / tweaked from a notebook.

Headline figures (when all pillars have run):
  - recall_lift_bar.png         the +Xpt rare-class recall lift, per synthesizer
  - privacy_utility_frontier.png  TR+STR recall lift on X, MIA AUC on Y; each point a synth
  - rule_compliance_bar.png     violations before vs after repair, per synthesizer
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise RuntimeError(f"{results_csv} is empty")
    return df


def recall_lift_bar(results_csv: Path, out: Path) -> Path:
    """Headline figure: the deck's '+17pt recall lift' bar chart, per synthesizer."""
    df = _load(results_csv)
    if df["recall_lift_pct"].isna().all():
        raise RuntimeError("recall_lift_pct is empty — utility pillar didn't run")

    agg = df.dropna(subset=["recall_lift_pct"]).groupby("synthesizer")["recall_lift_pct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(agg.index, agg.values, color=["#2A6EBB", "#E07A5F", "#7CA982"][:len(agg)])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Recall lift (percentage points)")
    ax.set_title("Rare-class recall lift from synthetic augmentation")
    for b, v in zip(bars, agg.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def privacy_utility_frontier(results_csv: Path, out: Path) -> Path:
    """Scatter: recall_lift_pct (x) vs MIA AUC (y). One marker per (synth, dataset) run.

    Top-left quadrant = high utility + low privacy risk (the goal).
    """
    df = _load(results_csv)
    needed = ["recall_lift_pct", "privacy_mia_auc"]
    if any(df[c].isna().all() for c in needed):
        raise RuntimeError("frontier needs both utility and privacy MIA; one is missing")

    fig, ax = plt.subplots(figsize=(6, 5))
    for synth, sub in df.dropna(subset=needed).groupby("synthesizer"):
        ax.scatter(sub["recall_lift_pct"], sub["privacy_mia_auc"], s=60, label=synth, alpha=0.75)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, label="MIA chance (private)")
    ax.set_xlabel("Recall lift (pp) — more is better")
    ax.set_ylabel("Membership inference AUC — closer to 0.5 is more private")
    ax.set_title("Privacy / utility frontier")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def rule_compliance_bar(results_csv: Path, out: Path) -> Path:
    """Stacked bar of violations_before vs violations_after, per synthesizer."""
    df = _load(results_csv)
    if df["violations_before"].isna().all():
        raise RuntimeError("rule_pack pillar didn't run — no violation counts")

    agg = df.dropna(subset=["violations_before"]).groupby("synthesizer")[
        ["violations_before", "violations_after"]
    ].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(agg))
    w = 0.35
    ax.bar([i - w / 2 for i in x], agg["violations_before"], w, label="before repair", color="#E07A5F")
    ax.bar([i + w / 2 for i in x], agg["violations_after"], w, label="after repair", color="#7CA982")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Total rule violations (mean across runs)")
    ax.set_title("Business-rule compliance: before vs after repair")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_all(results_csv: Path, out_dir: Path) -> list[Path]:
    """Render every figure that the current results CSV has data for."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fn in [
        ("recall_lift_bar.png", recall_lift_bar),
        ("privacy_utility_frontier.png", privacy_utility_frontier),
        ("rule_compliance_bar.png", rule_compliance_bar),
    ]:
        try:
            written.append(fn(results_csv, out_dir / name))
            print(f"  wrote {out_dir / name}")
        except RuntimeError as e:
            print(f"  skipped {name}: {e}")
    return written


if __name__ == "__main__":
    import sys
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "raw.csv"
    out_dir = csv_path.parent / "figures"
    make_all(csv_path, out_dir)

"""Publication-quality figures for the Aperture promo video.

Produces six figures in notes/figures/ summarizing the multi-generator
evaluation on fraud_oracle:

  1. radar.png            — 5-axis radar per generator (utility / fidelity /
                            privacy / detection-resistance / rule-compliance)
  2. tstr_vs_real.png     — TSTR + TR+STR vs real-data ceiling, with recall-lift
  3. privacy_ensemble.png — distance-MIA + DOMIAS + 3 Anonymeter risks per gen
  4. fidelity_triple.png  — alpha-precision / beta-recall / authenticity per gen
  5. class_prior.png      — synthetic positive-class rate vs real (the
                            diagnosis behind TabDDPM/TVAE failures)
  6. pipeline.png         — integration diagram showing the six-pillar pipeline

Reads three CSVs under experimental/showcase/results/:
  - showcase_multiseed.csv         (original TSTR / detection / MIA / etc.)
  - rule_violations_per_generator.csv
  - new_pillars_per_seed.csv       (from compute_new_pillars.py)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

GEN_COLORS = {
    "GaussianCopula": "#9aa3aa",
    "TVAE":           "#e87a3d",
    "TabDDPM":        "#7d63a8",
    "TabSyn":         "#3e8e6f",
    "TabDiff":        "#2db380",
    "Real (TRTR)":    "#222222",
}

RESULTS = Path("experimental/showcase/results")
FIG_DIR = Path("notes/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_data():
    multiseed = pd.read_csv(RESULTS / "showcase_multiseed.csv")
    rule_viol = pd.read_csv(RESULTS / "rule_violations_per_generator.csv")
    new_path = RESULTS / "new_pillars_per_seed.csv"
    new = pd.read_csv(new_path) if new_path.exists() else None
    return multiseed, rule_viol, new


def _agg_new_pillars(new_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-seed new-pillar metrics by generator (mean / std)."""
    if new_seed is None:
        return pd.DataFrame()
    g = new_seed.groupby("generator")
    summary = pd.concat({
        "mean": g.mean(numeric_only=True),
        "std": g.std(numeric_only=True).fillna(0.0),
    }, axis=1)
    summary.columns = [f"{stat}_{col}" for stat, col in summary.columns]
    return summary.reset_index()


# ── Figure 1 — Five-axis radar ───────────────────────────────────────────────

def fig_radar(multiseed: pd.DataFrame, rule_viol: pd.DataFrame, new_agg: pd.DataFrame):
    """One polygon per generator across five rescaled-higher-is-better axes."""
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="polar")

    axes_labels = ["TSTR\nutility", "α-precision\n(in-support)", "1 − detect AUC\n(realism)",
                   "1 − distance-MIA\n(privacy)", "rule\ncompliance"]
    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    seen = set()
    for _, row in multiseed.iterrows():
        gen = row.get("model", row.get("generator"))
        if gen not in GEN_COLORS or gen in seen:
            continue
        seen.add(gen)
        tstr = float(row.get("tstr_auc_mean", np.nan))
        det_auc = float(row.get("detection_auc_mean", row.get("xgboost_auc_mean", np.nan)))
        realism = max(0.0, 1.0 - det_auc) if not np.isnan(det_auc) else np.nan
        mia = float(row.get("mia_auc_mean", row.get("membership_inference_auc_mean", 0.5)))
        privacy_safe = 1.0 - abs(mia - 0.5) * 2  # peak at 0.5 (no leakage)

        rule_match = rule_viol[rule_viol["generator"] == gen]
        if len(rule_match):
            compliance = float(rule_match.iloc[0].get("rule_compliance_pct_mean", 100.0)) / 100.0
        else:
            compliance = np.nan

        alpha_prec = np.nan
        if not new_agg.empty:
            r = new_agg[new_agg["generator"] == gen]
            if len(r):
                alpha_prec = float(r.iloc[0].get("mean_alpha_precision", np.nan))

        values = [tstr, alpha_prec, realism, privacy_safe, compliance]
        if any(np.isnan(v) for v in values):
            # TVAE-style mode collapse: TSTR is undefined.
            values = [0.0 if np.isnan(v) else v for v in values]
        values = [max(0.0, min(1.0, v)) for v in values] + [values[0]]
        ax.plot(angles, values, color=GEN_COLORS[gen], linewidth=2.0, label=gen, marker="o", markersize=4)
        ax.fill(angles, values, color=GEN_COLORS[gen], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="#666")
    ax.set_ylim(0, 1.0)
    ax.set_title("Aperture five-pillar comparison on fraud_oracle\n(higher = better on every axis)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=False)

    fig.savefig(FIG_DIR / "radar.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "radar.png")


# ── Figure 2 — TSTR vs real-data ceiling + recall lift ───────────────────────

def fig_tstr_vs_real(multiseed: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.0, 1.0]})
    trtr_col = multiseed["trtr_auc_mean"].dropna()
    real_ceiling = float(trtr_col.iloc[0]) if len(trtr_col) else 0.836

    gens = [g for g in ["GaussianCopula", "TVAE", "TabDDPM", "TabSyn"]
            if g in multiseed["model"].values]
    means = []
    stds = []
    for g in gens:
        r = multiseed[multiseed["model"] == g].iloc[0]
        v = float(r.get("tstr_auc_mean", 0.0))
        means.append(v if v > 0 else np.nan)
        stds.append(float(r.get("tstr_auc_std", 0.0) or 0.0))
    means = np.array(means)
    stds = np.array(stds)

    bars = ax1.bar(gens, np.nan_to_num(means, nan=0.0),
                   yerr=np.nan_to_num(stds, nan=0.0),
                   capsize=4,
                   color=[GEN_COLORS[g] for g in gens])
    ax1.axhline(real_ceiling, color="#222", linestyle="--", linewidth=1.3, label=f"Real data ceiling ({real_ceiling:.3f})")
    for i, (m, s) in enumerate(zip(means, stds)):
        if np.isnan(m):
            ax1.annotate("n/a\n(mode\ncollapse)", (i, 0.05), ha="center", va="bottom", fontsize=9, color="#a44")
        else:
            ax1.annotate(f"{m:.3f}", (i, m + s + 0.01), ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("TSTR ROC-AUC")
    ax1.set_title("Train on Synthetic, Test on Real\n(higher → synth alone trains useful models)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)

    # right panel — recall lift
    recall_lift = []
    recall_std = []
    for g in gens:
        r = multiseed[multiseed["model"] == g].iloc[0]
        recall_lift.append(float(r.get("recall_lift_pct_mean", 0.0)))
        recall_std.append(float(r.get("recall_lift_std", 0.0) or 0.0))
    recall_lift = np.array(recall_lift)
    recall_std = np.array(recall_std)
    bars2 = ax2.bar(gens, recall_lift, yerr=recall_std, capsize=4,
                    color=[GEN_COLORS[g] for g in gens])
    ax2.axhline(0, color="#222", linewidth=0.8)
    for i, (m, s) in enumerate(zip(recall_lift, recall_std)):
        offset = s + 0.5 if m >= 0 else -(s + 1.5)
        ax2.annotate(f"{m:+.1f}", (i, m + offset), ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
    ax2.set_ylabel("Δ minority-class recall (pp)")
    ax2.set_title("Augmentation: ΔRecall vs Real-Only Training\n(positive → synthetic data helps fraud detection)")

    for ax in (ax1, ax2):
        for label in ax.get_xticklabels():
            label.set_rotation(15)

    fig.suptitle("fraud_oracle: utility of synthetic data trained on 5 seeds", y=1.02)
    fig.savefig(FIG_DIR / "tstr_vs_real.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "tstr_vs_real.png")


# ── Figure 3 — Privacy attack ensemble ───────────────────────────────────────

def fig_privacy_ensemble(multiseed: pd.DataFrame, new_agg: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    gens = [g for g in ["GaussianCopula", "TVAE", "TabDDPM", "TabSyn"]
            if g in multiseed["model"].values]
    attack_cols = [
        ("Distance MIA",           "mia_auc_mean", "membership_inference_auc_mean"),
        ("DOMIAS density MIA",     "mean_domias_auc", None),
        ("Anonymeter singling-out", "mean_anonymeter_singling_out_risk", None),
        ("Anonymeter linkability", "mean_anonymeter_linkability_risk", None),
        ("Anonymeter inference",   "mean_anonymeter_inference_risk", None),
    ]
    attack_labels = [a[0] for a in attack_cols]

    n_gens = len(gens)
    n_attacks = len(attack_cols)
    width = 0.15
    x = np.arange(n_attacks)
    for i, g in enumerate(gens):
        row_multi = multiseed[multiseed["model"] == g]
        row_new = new_agg[new_agg["generator"] == g] if not new_agg.empty else pd.DataFrame()
        vals = []
        for _, col_ms, col_alt in attack_cols:
            v = np.nan
            if not row_new.empty and col_ms in row_new.columns:
                v = float(row_new.iloc[0][col_ms])
            if np.isnan(v) and len(row_multi):
                if col_ms in row_multi.columns:
                    v = float(row_multi.iloc[0][col_ms])
                elif col_alt and col_alt in row_multi.columns:
                    v = float(row_multi.iloc[0][col_alt])
            vals.append(0.0 if np.isnan(v) else v)
        ax.bar(x + (i - n_gens / 2) * width + width / 2, vals, width=width,
               label=g, color=GEN_COLORS.get(g, "#888"))

    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, rotation=15, ha="right")
    ax.set_ylabel("Attack success (higher = worse privacy)")
    ax.set_title("Privacy attack ensemble — five attacks per generator\n"
                 "distance-MIA ≈ 0.5 = no signal; Anonymeter / DOMIAS in [0, 1]")
    ax.axhline(0.5, color="#999", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=2)
    ax.text(0.01, 0.97, "[Lower bars = safer synthetic data]",
            transform=ax.transAxes, fontsize=9, color="#666", va="top")

    fig.savefig(FIG_DIR / "privacy_ensemble.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "privacy_ensemble.png")


# ── Figure 4 — Fidelity triple ───────────────────────────────────────────────

def fig_fidelity_triple(new_agg: pd.DataFrame):
    if new_agg.empty:
        print("skipping fidelity_triple: no new_agg data")
        return
    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    gens = [g for g in ["GaussianCopula", "TVAE", "TabDDPM", "TabSyn"]
            if g in new_agg["generator"].values]
    metrics = [
        ("α-precision",  "mean_alpha_precision", "in-support quality"),
        ("β-recall",     "mean_beta_recall",     "mode coverage"),
        ("Authenticity", "mean_authenticity",    "not memorized"),
    ]
    x = np.arange(len(metrics))
    width = 0.18
    n_gens = len(gens)
    for i, g in enumerate(gens):
        r = new_agg[new_agg["generator"] == g]
        vals = [float(r.iloc[0].get(col, np.nan)) for _, col, _ in metrics]
        offset = (i - n_gens / 2) * width + width / 2
        ax.bar(x + offset, [0.0 if np.isnan(v) else v for v in vals],
               width=width, label=g, color=GEN_COLORS.get(g, "#888"))
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (x[j] + offset, v + 0.01),
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[0]}\n({m[2]})" for m in metrics], fontsize=10)
    ax.set_ylabel("Score (higher = better)")
    ax.set_title("Fidelity diagnostic (Alaa et al., ICML 2022)\n"
                 "Decomposes 'is the synthetic data good' into three independent dimensions")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.savefig(FIG_DIR / "fidelity_triple.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "fidelity_triple.png")


# ── Figure 5 — Class prior preservation ──────────────────────────────────────

def fig_class_prior():
    """Hard-coded from the showcase results table (Table 3 in notes/results_table.md)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    gens = ["Real", "GaussianCopula", "TVAE", "TabDDPM", "TabSyn"]
    rates = [5.99, 5.6, 0.0, 50.1, 6.0]
    colors = ["#222", GEN_COLORS["GaussianCopula"], GEN_COLORS["TVAE"],
              GEN_COLORS["TabDDPM"], GEN_COLORS["TabSyn"]]
    bars = ax.bar(gens, rates, color=colors)
    ax.axhline(5.99, color="#222", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_ylabel("Synthetic positive (fraud) rate (%)")
    ax.set_title("Class prior preservation — root-cause diagnosis of generator failures\n"
                 "Real fraud rate = 5.99%")
    for i, r in enumerate(rates):
        ax.annotate(f"{r}%", (i, r + 1.5), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 60)

    annotations = {
        2: "mode\ncollapse",
        3: "ignores\nclass prior",
    }
    for idx, txt in annotations.items():
        ax.annotate(txt, (idx, rates[idx] + 6.0), ha="center",
                    fontsize=8, color="#a44", style="italic")

    fig.savefig(FIG_DIR / "class_prior.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "class_prior.png")


# ── Figure 6 — Pipeline integration diagram ──────────────────────────────────

def fig_pipeline():
    """Static diagram showing where the new pillars plug into /api/generate."""
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Pipeline boxes (left-to-right top row, then bottom row)
    boxes_top = [
        (0.5, 7.5, 1.8, 1.1, "User\nupload\n(CSV)", "#e9ecef"),
        (3.0, 7.5, 1.8, 1.1, "synthesize()\nGaussianCopula\n/CTGAN/TVAE\n/TabDDPM/TabSyn\n/TabDiff", "#a8d5ba"),
        (5.5, 7.5, 1.8, 1.1, "rule_packs\ndetect→check→\nrepair→recheck", "#ffe4b5"),
        (8.0, 7.5, 1.8, 1.1, "synthetic.csv\nstored in\nsession", "#e9ecef"),
    ]
    for (x, y, w, h, label, color) in boxes_top:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#444", linewidth=1.2))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)

    # Arrows top row
    for x_start in [2.3, 4.8, 7.3]:
        ax.annotate("", xy=(x_start + 0.7, 8.05), xytext=(x_start, 8.05),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    # Down arrow to pillars
    ax.annotate("", xy=(4.9, 5.4), xytext=(4.9, 7.45),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    # Pillar boxes (the six pillars, with the four NEW ones highlighted)
    pillars = [
        (0.2, 3.5, 1.5, 1.4, "utility\n(TSTR /\nTR+STR /\nrecall lift)", "#cfe2f3", "team"),
        (1.8, 3.5, 1.5, 1.4, "rule packs\n(insurance /\nclinical /\nfraud_oracle)", "#cfe2f3", "team+mine"),
        (3.4, 3.5, 1.5, 1.4, "LLM\nauditor\n(plausibility)", "#cfe2f3", "team"),
        (5.0, 3.5, 1.5, 1.4, "detection\n(XGB +\nLogReg +\nECE)", "#a8d5ba", "MINE"),
        (6.6, 3.5, 1.5, 1.4, "privacy\n(DCR / NNDR /\ndistance-MIA)", "#a8d5ba", "MINE"),
        (8.2, 3.5, 1.5, 1.4, "fidelity\n(α-precision /\nβ-recall /\nauthenticity)", "#a8d5ba", "MINE"),
    ]
    for (x, y, w, h, label, color, owner) in pillars:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#444", linewidth=1.2))
        ax.text(x + w/2, y + h/2 + 0.05, label, ha="center", va="center", fontsize=8)
        if owner == "MINE":
            ax.text(x + w/2, y + h - 0.15, "★ added", ha="center", fontsize=7, color="#06402b", fontweight="bold")
        elif owner == "team+mine":
            ax.text(x + w/2, y + h - 0.15, "★ extended", ha="center", fontsize=7, color="#06402b", fontweight="bold")

    # Privacy attack super-pillar (the most novel addition)
    ax.add_patch(plt.Rectangle((0.2, 0.7), 9.5, 1.7, facecolor="#7fbf9b", edgecolor="#06402b", linewidth=1.8))
    ax.text(5.0, 1.55, "privacy_attacks (NEW — GDPR-aligned attack ensemble)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#06402b")
    ax.text(5.0, 1.1, "Anonymeter singling-out + linkability + attribute-inference  ·  DOMIAS density-MIA  ·  compose_privacy_ensemble verdict",
            ha="center", va="center", fontsize=8.5, color="#06402b")

    ax.annotate("", xy=(5.0, 2.4), xytext=(5.0, 3.4),
                arrowprops=dict(arrowstyle="<-", color="#06402b", lw=1.5))

    # Trust report at right
    ax.add_patch(plt.Rectangle((8.0, 0.7), 1.7, 1.7, facecolor="#fce5cd", edgecolor="#444", linewidth=1.2))
    ax.text(8.85, 1.55, "trust_report\nHTML\n(6 pillar\nsections +\nverdict tier)",
            ha="center", va="center", fontsize=8.5)

    ax.set_title("Aperture pipeline: where my contributions plug into /api/generate", pad=10)

    # Legend — placed outside the pillar rows to avoid overlap.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#cfe2f3", edgecolor="#444", label="team pillar"),
        Patch(facecolor="#a8d5ba", edgecolor="#444", label="★ my contribution"),
        Patch(facecolor="#7fbf9b", edgecolor="#06402b", label="★ new attack-ensemble module"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 6.7 / 10.0),
              frameon=False, fontsize=9, ncol=3)

    fig.savefig(FIG_DIR / "pipeline.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("wrote", FIG_DIR / "pipeline.png")


def main():
    multiseed, rule_viol, new_seed = _load_data()
    new_agg = _agg_new_pillars(new_seed)
    print(f"loaded {len(multiseed)} generators × showcase, {len(rule_viol)} × rule_viol, "
          f"{len(new_seed) if new_seed is not None else 0} × new-seed pillar rows")

    fig_radar(multiseed, rule_viol, new_agg)
    fig_tstr_vs_real(multiseed)
    fig_privacy_ensemble(multiseed, new_agg)
    fig_fidelity_triple(new_agg)
    fig_class_prior()
    fig_pipeline()

    print(f"\nall figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()

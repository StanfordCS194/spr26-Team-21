"""Run the Trust Benchmark.

Loops over (dataset × synthesizer × n_synth) and runs every Aperture pillar
that is currently importable. Writes one row per run to results/raw.csv.

Usage:
    python experimental/bench/run_trust_benchmark.py \
        --datasets pima_diabetes fraud_oracle \
        --synthesizers GaussianCopula \
        --n-synth 500 2000 \
        --out experimental/bench/results

Pillars are imported defensively — running on a branch that's missing
detection.py or privacy.py still produces a partial result. The orchestrator
fills missing columns with None so the schema is stable across runs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Path setup: this file's directory (for sibling imports) + backend/ (for services.*).
_HERE = Path(__file__).resolve().parent
BACKEND = _HERE.parent.parent / "backend"
for p in (_HERE, BACKEND):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import datasets         # type: ignore  # noqa: E402
import synthesizers     # type: ignore  # noqa: E402

# ── Defensive pillar imports ──────────────────────────────────────────────────
# Different branches have different pillars merged. We try each; missing ones
# just don't show up in the results row.

try:
    from services.utility import compute_utility
except Exception:
    compute_utility = None
try:
    from services.rule_packs import apply_pack
except Exception:
    apply_pack = None
try:
    from services.llm_auditor import audit_sample
except Exception:
    audit_sample = None
try:
    from services.privacy import compute_privacy
except Exception:
    compute_privacy = None
try:
    from services.detection import compute_detection
except Exception:
    compute_detection = None


def _flatten_metric(value):
    """Round floats to 4 dp, leave None / strings as-is."""
    if isinstance(value, float):
        return round(value, 4)
    return value


def _row(dataset_name: str, synth_name: str, n_synth: int) -> dict:
    """One result row — schema-stable across runs."""
    return {
        "dataset": dataset_name,
        "synthesizer": synth_name,
        "n_synth": n_synth,
        "fit_seconds": None,
        "sample_seconds": None,
        # utility
        "trtr_auc": None, "tstr_auc": None, "augmented_auc": None,
        "trtr_recall": None, "tstr_recall": None, "augmented_recall": None,
        "recall_lift_pct": None, "utility_verdict": None,
        # rule_pack
        "rule_pack": None, "violations_before": None, "violations_after": None,
        # audit
        "audit_mean_plausibility": None, "audit_n_flagged": None,
        # privacy (defensive)
        "privacy_n_exact_matches": None, "privacy_dcr_median": None,
        "privacy_baseline_score": None, "privacy_mia_auc": None,
        # detection (defensive)
        "detection_xgb_auc": None, "detection_lr_auc": None, "detection_agreement": None,
    }


def _run_one(ds: datasets.Dataset, synth_cls, n_synth: int) -> dict:
    row = _row(ds.name, synth_cls.name, n_synth)
    print(f"  [{ds.name} × {synth_cls.name} × n={n_synth}]")

    t0 = time.time()
    synth = synth_cls()
    synth.fit(ds.train_df)
    row["fit_seconds"] = round(time.time() - t0, 1)

    t1 = time.time()
    synth_df = synth.sample(n_synth)
    row["sample_seconds"] = round(time.time() - t1, 1)

    # ── Rule pack ──
    if apply_pack is not None:
        rep = apply_pack(synth_df)
        if rep is not None:
            row["rule_pack"] = rep["pack"]
            row["violations_before"] = rep["violations_before"]
            row["violations_after"] = rep["violations_after"]
            # Swap repaired_df in so downstream pillars see the repaired data
            # (this mirrors what /api/generate does).
            synth_df = rep.pop("repaired_df")

    # ── Utility ──
    if compute_utility is not None:
        util = compute_utility(ds.train_df, synth_df, label_col=ds.label_col)
        if util is not None and util.get("available"):
            row["trtr_auc"] = _flatten_metric(util["trtr"].get("auc"))
            row["tstr_auc"] = _flatten_metric(util["tstr"].get("auc"))
            row["augmented_auc"] = _flatten_metric(util["augmented"].get("auc"))
            row["trtr_recall"] = _flatten_metric(util["trtr"].get("recall"))
            row["tstr_recall"] = _flatten_metric(util["tstr"].get("recall"))
            row["augmented_recall"] = _flatten_metric(util["augmented"].get("recall"))
            row["recall_lift_pct"] = util.get("recall_lift_pct")
            row["utility_verdict"] = util.get("verdict")

    # ── Audit ──
    if audit_sample is not None:
        aud = audit_sample(synth_df, use_llm=False)
        if aud.get("available"):
            row["audit_mean_plausibility"] = _flatten_metric(aud.get("mean_plausibility"))
            row["audit_n_flagged"] = aud.get("n_flagged")

    # ── Privacy ──
    if compute_privacy is not None:
        priv = compute_privacy(ds.train_df, synth_df, holdout_df=ds.holdout_df)
        if priv and priv.get("available"):
            row["privacy_n_exact_matches"] = priv["dcr"].get("n_exact_matches")
            row["privacy_dcr_median"] = _flatten_metric(priv["dcr"].get("median"))
            row["privacy_baseline_score"] = _flatten_metric(priv["baseline_protection"].get("score"))
            mia = priv.get("membership_inference") or {}
            row["privacy_mia_auc"] = _flatten_metric(mia.get("roc_auc"))

    # ── Detection ──
    if compute_detection is not None:
        det = compute_detection(ds.train_df, synth_df)
        if det and det.get("available"):
            xgb = det.get("xgboost") or {}
            lr = det.get("logreg") or {}
            row["detection_xgb_auc"] = _flatten_metric(xgb.get("auc"))
            row["detection_lr_auc"] = _flatten_metric(lr.get("auc"))
            row["detection_agreement"] = det.get("agreement")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pima_diabetes"],
                        choices=list(datasets.LOADERS),
                        help="Which datasets to benchmark.")
    parser.add_argument("--synthesizers", nargs="+", default=["GaussianCopula"],
                        help="Which synthesizers (by name) — see synthesizers.REGISTRY.")
    parser.add_argument("--n-synth", nargs="+", type=int, default=[500],
                        help="Number(s) of synthetic rows to generate per run.")
    parser.add_argument("--out", type=Path, default=_HERE / "results",
                        help="Where to write raw.csv.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "raw.csv"

    rows = []
    for ds_name in args.datasets:
        ds = datasets.load(ds_name)
        print(f"[{ds_name}] train={len(ds.train_df)} holdout={len(ds.holdout_df)} label={ds.label_col}")
        for synth_name in args.synthesizers:
            if synth_name not in synthesizers.REGISTRY:
                print(f"  ! unknown synthesizer {synth_name!r}, skipping")
                continue
            synth_cls = synthesizers.REGISTRY[synth_name]
            for n in args.n_synth:
                try:
                    rows.append(_run_one(ds, synth_cls, n))
                except Exception as e:
                    print(f"  ! {synth_name} on {ds_name} n={n} failed: {e}")

    if not rows:
        print("No rows produced.")
        return

    # Write CSV with stable column order (taken from the first row's keys).
    fieldnames = list(rows[0].keys())
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

"""Prepare fraud_oracle for TabSyn (amazon-science/tabsyn) training.

TabSyn ingests datasets via a two-step layout:

  1) data/<name>/<name>.csv             — raw CSV (header row, no index)
  2) data/Info/<name>.json              — column metadata used by process_dataset.py
        to split into train/test and write the .npy + transformer-friendly
        layout the VAE+latent-diffusion pipeline consumes.

This script writes BOTH files under a portable `--out_dir` so the user can rsync
the result up to Sherlock and then run TabSyn's own `python process_dataset.py
--dataname fraud_oracle` once to generate the internal binary tensors.

Dataset facts (verified 2026-05-30 against
experimental/data/fraud_oracle.csv):
  rows           = 15,420
  cols           = 33
  target         = FraudFound_P (binary, positive rate 5.99% — class imbalanced)
  PolicyNumber   = a row-unique identifier; DROPPED (would leak through any
                   learner and is useless for a generative prior anyway).
  truly numeric  = Age, Year                  (2 columns)
  categorical    = 30 remaining feature columns
  target column  = FraudFound_P               (1 column, binary)

TabSyn's process_dataset.py expects `num_col_idx`, `cat_col_idx`,
`target_col_idx` as POSITIONAL indices into the saved CSV header. We compute
those from the dropped+reordered dataframe to keep the JSON self-consistent.

Stratified split: 80/10/10 (train/val/test). TabSyn's process_dataset.py uses
the test slice for the held-out eval shipped in the repo, and re-splits train
into its own train/val internally. The val slice we emit is for our downstream
validation pillars (TSTR / utility / privacy) and is NOT consumed by TabSyn.

Usage
-----
    python experimental/training/tabsyn/prep_fraud_oracle.py \\
        --csv  experimental/data/fraud_oracle.csv \\
        --out  experimental/data/tabsyn_fraud_oracle/

Outputs under <out>/:
    fraud_oracle.csv          # all rows, columns reordered: [num..., cat..., target]
    fraud_oracle_train.csv    # 80% stratified on FraudFound_P
    fraud_oracle_val.csv      # 10% (for our pillars, not TabSyn)
    fraud_oracle_test.csv     # 10% (consumed by TabSyn process_dataset.py)
    Info/fraud_oracle.json    # TabSyn's column metadata JSON
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Column policy — verified against the raw CSV header.
# Genuinely continuous numeric columns (binnable, ordered, large support):
NUM_COLS: list[str] = ["Age", "Year"]

# Drop list: row-unique identifiers and known leakers.
DROP_COLS: list[str] = ["PolicyNumber"]

TARGET_COL: str = "FraudFound_P"
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--csv",
        default="experimental/data/fraud_oracle.csv",
        help="path to the raw fraud_oracle.csv",
    )
    p.add_argument(
        "--out",
        default="experimental/data/tabsyn_fraud_oracle",
        help="output directory (will be created)",
    )
    p.add_argument("--seed", type=int, default=42, help="train_test_split random_state")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Info").mkdir(exist_ok=True)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    # The raw file has a BOM on the first column header on some downloads.
    df.columns = df.columns.str.replace("﻿", "", regex=False)
    print(f"  loaded {df.shape[0]} rows x {df.shape[1]} cols")

    # ----- 1. drop leakers / id columns -----
    missing_drop = [c for c in DROP_COLS if c not in df.columns]
    if missing_drop:
        raise KeyError(f"expected drop columns missing from CSV: {missing_drop}")
    df = df.drop(columns=DROP_COLS)
    print(f"  dropped {DROP_COLS} -> {df.shape[1]} cols remain")

    # ----- 2. reorder: numeric columns first, then categorical, then target -----
    if TARGET_COL not in df.columns:
        raise KeyError(f"target column {TARGET_COL!r} not in CSV")
    missing_num = [c for c in NUM_COLS if c not in df.columns]
    if missing_num:
        raise KeyError(f"declared numeric columns missing from CSV: {missing_num}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    cat_cols = [c for c in feature_cols if c not in NUM_COLS]
    reordered = NUM_COLS + cat_cols + [TARGET_COL]
    df = df[reordered]
    print(f"  reordered: {len(NUM_COLS)} num + {len(cat_cols)} cat + 1 target")

    # ----- 3. coerce dtypes -----
    # TabSyn happily reads strings for categoricals; we just need numeric cols to
    # be numeric. The target stays as int (binclass).
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.float64)
    df[TARGET_COL] = df[TARGET_COL].astype(np.int64)
    # Cast categoricals to string so np.save/csv doesn't mix int-coded levels
    # with string levels for columns like "Year" vs "Make".
    for c in cat_cols:
        df[c] = df[c].astype(str)

    pos_rate = float(df[TARGET_COL].mean())
    print(f"  target positive rate: {pos_rate:.4f} ({int(df[TARGET_COL].sum())}/{len(df)})")

    # ----- 4. stratified split 80/10/10 -----
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, stratify=df[TARGET_COL], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, stratify=tmp_df[TARGET_COL], random_state=args.seed
    )
    print(
        f"  split sizes: train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)} "
        f"(pos rates {train_df[TARGET_COL].mean():.4f}/"
        f"{val_df[TARGET_COL].mean():.4f}/{test_df[TARGET_COL].mean():.4f})"
    )

    # ----- 5. emit CSVs -----
    full_csv = out_dir / "fraud_oracle.csv"
    train_csv = out_dir / "fraud_oracle_train.csv"
    val_csv = out_dir / "fraud_oracle_val.csv"
    test_csv = out_dir / "fraud_oracle_test.csv"
    df.to_csv(full_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"  wrote {full_csv.name}, *_train.csv, *_val.csv, *_test.csv")

    # ----- 6. compute positional column indices (into reordered header) -----
    header = reordered
    num_col_idx = [header.index(c) for c in NUM_COLS]
    cat_col_idx = [header.index(c) for c in cat_cols]
    target_col_idx = [header.index(TARGET_COL)]

    info = {
        "name": "fraud_oracle",
        "task_type": "binclass",
        "header": "infer",
        "column_names": header,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        # Paths are written relative so the JSON is portable between Mac and
        # Sherlock. TabSyn resolves these against its data/ root at runtime.
        "data_path": "data/fraud_oracle/fraud_oracle.csv",
        "test_path": "data/fraud_oracle/fraud_oracle_test.csv",
        "n_classes": int(df[TARGET_COL].nunique()),
        "n_num_features": len(NUM_COLS),
        "n_cat_features": len(cat_cols),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "pos_rate": pos_rate,
    }
    info_path = out_dir / "Info" / "fraud_oracle.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"  wrote {info_path.relative_to(out_dir)}")

    print("\nDone. Next step: rsync this directory up to Sherlock, e.g.")
    print(
        f"  rsync -avR {out_dir.as_posix()}/ "
        "sherlock:/scratch/groups/rbaltman/mstojkov/tabsyn/data/"
    )
    print("Then on Sherlock submit experimental/training/tabsyn/train.sbatch.")


if __name__ == "__main__":
    main()

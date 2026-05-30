"""Upload a trained TabDDPM checkpoint to HuggingFace Hub.

Defensive by default: --dry-run prints what would be pushed without touching HF.
Pass --execute to actually upload.

Usage
-----
    # Dry run
    python experimental/tabddpm/upload_hf.py \\
        --checkpoint-dir experimental/data/tabddpm_pima \\
        --repo mstojkov2024/aperture-tabddpm-pima

    # Real upload (needs HUGGINGFACE_TOKEN in env)
    export HUGGINGFACE_TOKEN=hf_...
    python experimental/tabddpm/upload_hf.py \\
        --checkpoint-dir experimental/data/tabddpm_pima \\
        --repo mstojkov2024/aperture-tabddpm-pima \\
        --execute

What gets uploaded
------------------
  model.pt              the trained weights
  model_ema.pt          EMA-averaged weights (usually slightly better samples)
  config.toml           training hyperparameters + model architecture
  info.json             dataset shape and task metadata
  loss.csv              full training-loss curve
  eval_catboost.json    paper-style 50-seed CatBoost TSTR vs real-data eval
  README.md             auto-generated model card (overwrites any existing)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Files we expect from a TabDDPM training run (per the upstream pipeline.py).
UPLOAD_PATTERNS = [
    "model.pt",
    "model_ema.pt",
    "config.toml",
    "info.json",
    "loss.csv",
    "eval_catboost.json",
]


def _readme(checkpoint_dir: Path, dataset_hint: str | None) -> str:
    """Generate a model card for the HF repo."""
    info = {}
    info_path = checkpoint_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())

    eval_json = checkpoint_dir / "eval_catboost.json"
    eval_block = ""
    if eval_json.exists():
        e = json.loads(eval_json.read_text())
        synth = (e.get("synthetic") or {}).get("test", {})
        real = (e.get("real") or {}).get("test", {})
        if synth or real:
            eval_block = (
                "\n## Evaluation (TabDDPM repo's CatBoost suite)\n\n"
                "| run | ROC-AUC | F1 | Accuracy |\n"
                "|---|---|---|---|\n"
                f"| synth → real test | {synth.get('roc_auc-mean', '?'):.3f} ± {synth.get('roc_auc-std', 0):.3f} | "
                f"{synth.get('f1-mean', '?'):.3f} | {synth.get('acc-mean', '?'):.3f} |\n"
                f"| real → real test (ceiling) | {real.get('roc_auc-mean', '?'):.3f} ± {real.get('roc_auc-std', 0):.3f} | "
                f"{real.get('f1-mean', '?'):.3f} | {real.get('acc-mean', '?'):.3f} |\n"
            )

    dataset = info.get("name") or dataset_hint or checkpoint_dir.name

    return (
        f"# Aperture TabDDPM — {dataset}\n\n"
        f"Trained checkpoint for a TabDDPM tabular-diffusion model on **{dataset}**, produced\n"
        "as part of the Aperture synthetic-data product (Stanford CS194W spr26-Team-21).\n\n"
        "## What this is\n\n"
        "A reproduction of [TabDDPM](https://arxiv.org/abs/2209.15421) (Kotelnikov et al., ICML 2023)\n"
        "trained on insurance-domain tabular data. The weights pair with Aperture's evaluation\n"
        "suite (utility, rule_packs, audit, privacy, detection) so downstream users get\n"
        "synthetic insurance data plus a regulator-style trust report on the same workflow.\n\n"
        "**Novelty by application.** The accompanying literature survey\n"
        "(notes/insurance_genmodels_lit_review.md in the project repo) finds no published\n"
        "work applying TabDDPM specifically to insurance-fraud or claims data. The closest\n"
        "prior work is FinDiff (ICAIF 2023) on credit-default and EmDT (arXiv 2603.13566) on\n"
        "credit-card fraud. This checkpoint fills that gap for insurance applications.\n\n"
        "## How to use\n\n"
        "```python\n"
        "from experimental.tabddpm.wrapper import load_from_huggingface\n"
        f'synth = load_from_huggingface("<this-repo-id>")\n'
        "df = synth.sample(500)   # 500 synthetic rows in the original schema\n"
        "```\n"
        f"{eval_block}\n"
        "## Training details\n\n"
        f"  Dataset:           {dataset}\n"
        f"  n_num_features:    {info.get('n_num_features', '?')}\n"
        f"  n_cat_features:    {info.get('n_cat_features', '?')}\n"
        f"  Train size:        {info.get('train_size', '?')}\n"
        f"  Task type:         {info.get('task_type', '?')}\n\n"
        "Hardware: NVIDIA A40 on Stanford Sherlock (rbaltman partition).\n"
        "Full training config is in `config.toml`; loss curve in `loss.csv`.\n\n"
        "## License\n\n"
        "MIT, following the upstream Yandex TabDDPM repo.\n\n"
        "## Citation\n\n"
        "```bibtex\n"
        "@inproceedings{kotelnikov2023tabddpm,\n"
        "  title     = {TabDDPM: Modelling Tabular Data with Diffusion Models},\n"
        "  author    = {Kotelnikov, Akim and Baranchuk, Dmitry and Rubachev, Ivan and Babenko, Artem},\n"
        "  booktitle = {ICML},\n"
        "  year      = {2023}\n"
        "}\n"
        "```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a TabDDPM checkpoint to HuggingFace Hub.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Local directory with model.pt + config.toml + ...")
    parser.add_argument("--repo", type=str, required=True,
                        help="HF repo id, e.g. mstojkov2024/aperture-tabddpm-pima")
    parser.add_argument("--dataset-hint", type=str, default=None,
                        help="Friendly dataset name for the model card (optional)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually push to HF. Without this flag, dry-run only.")
    parser.add_argument("--private", action="store_true",
                        help="Create the HF repo as private.")
    args = parser.parse_args()

    cd = args.checkpoint_dir
    if not cd.exists():
        sys.exit(f"checkpoint dir not found: {cd}")

    print(f"== checkpoint dir: {cd} ==")
    matched = []
    for pattern in UPLOAD_PATTERNS:
        p = cd / pattern
        if p.exists():
            matched.append((pattern, p.stat().st_size))
            print(f"  ✓ {pattern}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"  · {pattern}  (missing — will skip)")

    print(f"\n== model card ==\n{_readme(cd, args.dataset_hint)[:600]}\n   …\n")

    if not args.execute:
        print(f"\nDRY RUN — would push {len(matched)} files to {args.repo}.")
        print("Re-run with --execute to actually upload (needs HUGGINGFACE_TOKEN env var).")
        return 0

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HUGGINGFACE_TOKEN not in env — `export HUGGINGFACE_TOKEN=hf_...` first")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("pip install huggingface_hub")

    api = HfApi(token=token)
    create_repo(args.repo, token=token, exist_ok=True, private=args.private)

    # Stage the auto-generated README.
    readme_path = cd / "README.md"
    readme_path.write_text(_readme(cd, args.dataset_hint))

    for pattern, _ in matched + [("README.md", readme_path.stat().st_size)]:
        path = cd / pattern
        print(f"  uploading {pattern} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=pattern,
            repo_id=args.repo,
            token=token,
        )

    print(f"\n✓ pushed to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

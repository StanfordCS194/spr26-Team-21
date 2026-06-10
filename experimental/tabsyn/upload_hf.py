"""Upload a trained TabSyn checkpoint to HuggingFace Hub.

TabSyn trains two networks: a tabular VAE that produces a continuous latent,
and a latent-space diffusion model that operates on those latents. After a
full training run on Sherlock both checkpoints live under
`/scratch/.../tabsyn/tabsyn/ckpt/<dataname>/` as:

    vae/model.pt           VAE encoder/decoder weights
    model.pt               diffusion model weights (final)
    model_<epoch>.pt       optional intermediate diffusion checkpoints

This script bundles those checkpoints + a small set of metadata files into a
HuggingFace repo so the wrapper can pull them on demand.

Defensive by default: `--dry-run` prints the file plan without touching HF.
Pass `--execute` to perform the actual upload.

Usage
-----
    # Dry run, see what would be pushed
    python experimental/tabsyn/upload_hf.py \\
        --checkpoint-dir /scratch/.../tabsyn/ckpt/fraud_oracle_seed0 \\
        --info-json /scratch/.../tabsyn/data/Info/fraud_oracle_seed0.json \\
        --repo mstojkov2024/aperture-tabsyn-fraud-oracle

    # Real upload (requires HUGGINGFACE_TOKEN in env)
    export HUGGINGFACE_TOKEN=hf_...
    python experimental/tabsyn/upload_hf.py \\
        --checkpoint-dir /scratch/.../tabsyn/ckpt/fraud_oracle_seed0 \\
        --info-json /scratch/.../tabsyn/data/Info/fraud_oracle_seed0.json \\
        --repo mstojkov2024/aperture-tabsyn-fraud-oracle \\
        --execute

What gets uploaded
------------------
  vae/model.pt           VAE encoder/decoder
  model.pt               latent diffusion model (final epoch)
  config.json            VAE + diffusion architecture, layer dims, hyperparameters
  info.json              dataset shape, task type, column type indices
  README.md              auto-generated model card (overwritten on each push)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


_README_TEMPLATE = """---
license: cc-by-4.0
tags:
- tabular
- synthetic-data
- diffusion
- tabsyn
- insurance
---

# TabSyn — {dataset}

A TabSyn (Zhang et al., ICLR 2024) checkpoint trained on the **{dataset}** dataset
as part of the Aperture project (Stanford CS 194W, Spring 2026). Two-stage
training: a tabular VAE produces a continuous latent space, then a
score-based diffusion model is trained on those latents.

## Use this checkpoint

```python
from huggingface_hub import snapshot_download
from experimental.tabsyn.wrapper import TabSynSynthesizer

ckpt = snapshot_download(repo_id="{repo}")
synth = TabSynSynthesizer(checkpoint_dir=ckpt)
df = synth.sample(1000)
```

## Training provenance

- Source data: `{dataset}` ({n_rows} rows, {n_features} features, {n_classes}-class task)
- Hardware: NVIDIA A40 (46 GB) on Stanford Sherlock, `rbaltman` partition
- Wall time: ~30 minutes per seed
- Seed: {seed}
- Upstream code: <https://github.com/amazon-science/tabsyn>
- Trained from `experimental/training/tabsyn/train.sbatch` in the Aperture repo

## Files

| File | What it is |
|---|---|
| `vae/model.pt` | tabular VAE weights |
| `model.pt` | latent-space diffusion model weights |
| `config.json` | architecture + hyperparameters |
| `info.json` | column-type indices + task metadata |

## Citation

> Zhang, H., Zhang, J., Srinivasan, B., Shen, Z., Qin, X., Faloutsos, C.,
> Rangwala, H., & Karypis, G. (2024). Mixed-Type Tabular Data Synthesis
> with Score-Based Diffusion in Latent Space. *ICLR 2024*.
"""


def _enumerate_upload_files(checkpoint_dir: Path, info_json: Path | None) -> list[tuple[Path, str]]:
    """Return (local_path, relative_path_in_repo) pairs to upload."""
    pairs: list[tuple[Path, str]] = []
    vae = checkpoint_dir / "vae" / "model.pt"
    if vae.exists():
        pairs.append((vae, "vae/model.pt"))
    diffusion = checkpoint_dir / "model.pt"
    if diffusion.exists():
        pairs.append((diffusion, "model.pt"))
    cfg = checkpoint_dir / "config.json"
    if cfg.exists():
        pairs.append((cfg, "config.json"))
    if info_json and info_json.exists():
        pairs.append((info_json, "info.json"))
    return pairs


def _build_readme(repo: str, info_json: Path | None, seed: int) -> str:
    info = json.loads(info_json.read_text()) if info_json and info_json.exists() else {}
    return _README_TEMPLATE.format(
        repo=repo,
        dataset=info.get("name", "<dataset>"),
        n_rows=info.get("train_size", info.get("n_rows", "?")),
        n_features=len(info.get("column_names", [])) or "?",
        n_classes=info.get("n_classes", "?"),
        seed=seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload a TabSyn checkpoint to HuggingFace Hub.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing vae/model.pt + model.pt + config.json")
    parser.add_argument("--info-json", type=Path, default=None,
                        help="Path to the TabSyn Info JSON for this dataset (optional)")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo id, e.g. user/aperture-tabsyn-fraud-oracle")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (recorded in the README only)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the upload. Without this, prints the plan and exits.")
    args = parser.parse_args(argv)

    files = _enumerate_upload_files(args.checkpoint_dir, args.info_json)
    if not files:
        print(f"ERROR: no checkpoint files found under {args.checkpoint_dir}", file=sys.stderr)
        return 2

    print(f"plan: upload {len(files)} files to HF repo {args.repo!r}")
    for src, rel in files:
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  {src}  ->  {rel}  ({size_mb:.1f} MB)")

    if not args.execute:
        print("\n[dry-run] not uploading. Add --execute to push.")
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed. `pip install huggingface_hub` first.", file=sys.stderr)
        return 3

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HUGGINGFACE_TOKEN (or HF_TOKEN) in env before --execute.", file=sys.stderr)
        return 4

    api = HfApi(token=token)
    create_repo(args.repo, token=token, exist_ok=True, repo_type="model")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for src, rel in files:
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        (tmp_path / "README.md").write_text(_build_readme(args.repo, args.info_json, args.seed))

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"upload TabSyn checkpoint (seed {args.seed})",
        )

    print(f"\nuploaded to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

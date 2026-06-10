"""Route /api/generate to the right synthesizer backend.

The team's `services.synthesis.synthesize()` handles the SDV cache (Gaussian
Copula, CTGAN, TVAE) and a statistical fallback. This router adds the offline
diffusion backends (TabSyn, TabDDPM) that were trained on Sherlock and whose
outputs are served from disk.

Public entry point: `route_synthesizer(model_id, n, schema_columns, source_stats)`.

The router is only consulted when `model_id` is a known offline-backend
sentinel ("tabsyn", "tabddpm"). For any other model_id, the caller should
fall through to the SDV path in `synthesis.synthesize`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/services/.. = backend/.. = repo
TABSYN_DEFAULT_SAMPLES = _REPO_ROOT / "experimental/training/tabsyn/samples"
TABDDPM_DEFAULT_CHECKPOINTS = _REPO_ROOT / "experimental/training/tabddpm/checkpoints"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def is_offline_synthesizer(model_id: str | None) -> bool:
    """Whether `model_id` names an offline diffusion backend handled here."""
    return model_id in {"tabsyn", "tabddpm"}


def route_synthesizer(
    model_id: str,
    n: int,
    schema_columns: list[dict],
    source_stats: dict | None = None,
) -> pd.DataFrame:
    """Dispatch to TabSyn or TabDDPM. Raises RuntimeError on missing artefacts."""
    if model_id == "tabsyn":
        return _route_tabsyn(n, schema_columns)
    if model_id == "tabddpm":
        return _route_tabddpm(n, schema_columns)
    raise ValueError(f"unsupported offline synthesizer: {model_id!r}")


def _route_tabsyn(n: int, schema_columns: list[dict]) -> pd.DataFrame:
    from experimental.tabsyn.wrapper import TabSynNotAvailable, TabSynSynthesizer

    samples_dir = Path(os.environ.get("TABSYN_SAMPLES_DIR", TABSYN_DEFAULT_SAMPLES))
    try:
        synth = TabSynSynthesizer(samples_dir=samples_dir)
        df = synth.sample(n)
    except TabSynNotAvailable as e:
        raise RuntimeError(
            f"TabSyn backend not available: {e}. "
            "Run the Sherlock sweep (experimental/training/tabsyn/train_sweep.sbatch) "
            "or set TABSYN_SAMPLES_DIR to a directory with seed_<N>/synth.csv files."
        ) from e
    return _align_columns(df, schema_columns)


def _route_tabddpm(n: int, schema_columns: list[dict]) -> pd.DataFrame:
    from experimental.tabddpm.wrapper import TabDDPMSynthesizer

    ckpt = Path(os.environ.get("TABDDPM_CHECKPOINT_DIR", TABDDPM_DEFAULT_CHECKPOINTS / "seed_0"))
    if not ckpt.exists():
        raise RuntimeError(
            f"TabDDPM checkpoint not found at {ckpt}. "
            "Train via experimental/training/tabddpm/train_sweep.sbatch on Sherlock "
            "and rsync the checkpoints back, or set TABDDPM_CHECKPOINT_DIR."
        )
    synth = TabDDPMSynthesizer(checkpoint_dir=ckpt)
    df = synth.sample(n)
    return _align_columns(df, schema_columns)


def _align_columns(df: pd.DataFrame, schema_columns: list[dict]) -> pd.DataFrame:
    """Order df's columns to match the request schema; drop unknowns, leave gaps as NaN."""
    requested = [c["column"] for c in schema_columns]
    present = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]
    out = df[present].copy() if present else pd.DataFrame(index=df.index)
    for col in missing:
        out[col] = pd.NA
    return out[requested]

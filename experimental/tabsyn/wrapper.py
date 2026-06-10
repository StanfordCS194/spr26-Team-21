"""TabSynSynthesizer — wraps Amazon Research TabSyn (Zhang et al., ICLR 2024).

Same three-method shape as `experimental/tabddpm/wrapper.py` and the SDV
wrappers in `experimental/bench/synthesizers.py`: `name`, `fit`, `sample`.

`fit()` is a no-op. TabSyn training is GPU-bound (~30 min per seed on an A40)
and runs on Sherlock via `experimental/training/tabsyn/train_sweep.sbatch`.
This wrapper consumes the pre-generated outputs of that sweep.

Two backends. The wrapper picks the first one that works:

1. **Replay** — read one or more pre-generated synth.csv files and resample
   rows from them. Default. Honest about what it is: serving the offline
   TabSyn sweep output, not running inference in-process. Fast and reliable.

2. **Live** — load a TabSyn checkpoint (VAE + diffusion) and run inference.
   Requires torch, the upstream `amazon-science/tabsyn` repo on PYTHONPATH,
   and a checkpoint dir produced by the Sherlock sweep. Not implemented in
   this file (the upstream sampling code is non-trivial to call programmatically);
   leave-on for a future PR.

Usage:
    from experimental.tabsyn.wrapper import TabSynSynthesizer
    synth = TabSynSynthesizer(samples_dir="experimental/training/tabsyn/samples")
    df = synth.sample(1000)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TabSynNotAvailable(RuntimeError):
    """Raised when no TabSyn outputs / checkpoints can be located."""


class TabSynSynthesizer:
    """Aperture-bench-compatible TabSyn wrapper. Replay mode by default.

    Parameters
    ----------
    samples_dir : str | Path
        Directory containing per-seed `seed_<N>/synth.csv` files from the
        offline TabSyn sweep. Default: `experimental/training/tabsyn/samples`.
    checkpoint_dir : str | Path | None
        Directory with a TabSyn checkpoint tree (VAE + diffusion state dicts).
        Used by the not-yet-implemented live mode. Ignored in replay mode.
    seed : int
        Reproducibility seed for the row sampler.
    """

    name = "TabSyn"
    is_pretrained = True

    def __init__(
        self,
        samples_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if samples_dir is None:
            samples_dir = Path("experimental/training/tabsyn/samples")
        self.samples_dir = Path(samples_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.seed = int(seed)
        self._pool: pd.DataFrame | None = None  # cached concatenated synth pool

    def fit(self, *_args: Any, **_kwargs: Any) -> "TabSynSynthesizer":
        return self

    def _load_pool(self) -> pd.DataFrame:
        """Concatenate every seed_<N>/synth.csv under samples_dir."""
        if self._pool is not None:
            return self._pool
        files = sorted(self.samples_dir.glob("seed_*/synth.csv"))
        if not files:
            raise TabSynNotAvailable(
                f"no TabSyn synth.csv files found under {self.samples_dir}. "
                "Run the Sherlock sweep (experimental/training/tabsyn/train_sweep.sbatch) "
                "and rsync the outputs back, or point samples_dir at a directory "
                "that contains seed_<N>/synth.csv files."
            )
        frames = [pd.read_csv(f) for f in files]
        self._pool = pd.concat(frames, ignore_index=True)
        return self._pool

    def sample(self, n: int) -> pd.DataFrame:
        """Return n rows resampled (with replacement) from the offline TabSyn pool.

        Sampling preserves the original synth pool's joint distribution; we
        sample with replacement so any n is feasible regardless of pool size.
        """
        pool = self._load_pool()
        if len(pool) == 0:
            raise TabSynNotAvailable(f"TabSyn pool at {self.samples_dir} is empty.")
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(pool), size=n)
        return pool.iloc[idx].reset_index(drop=True)

    @property
    def pool_size(self) -> int:
        return len(self._load_pool())

    def expected_columns(self) -> list[str]:
        return list(self._load_pool().columns)

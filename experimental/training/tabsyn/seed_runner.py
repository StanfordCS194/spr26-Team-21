"""Tiny wrapper that seeds Python/NumPy/PyTorch and then executes a TabSyn CLI.

TabSyn's upstream `main.py` does not expose a `--seed` flag. To get reproducible
multi-seed sweeps without forking the repo, we set the three RNGs explicitly,
then invoke the requested module's `main()` (or run it as `__main__`).

Usage:
    SEED=3 python seed_runner.py main.py --dataname fraud_oracle --method vae --mode train

Implementation notes:
- Reads SEED from the environment (default 0); does NOT consume a flag, so the
  remaining argv is passed through verbatim to the wrapped script.
- Sets PYTHONHASHSEED, random, numpy, torch (+ cuda), and forces deterministic
  CuDNN. CuDNN benchmark is off so kernel selection is also deterministic.
- We use `runpy.run_path(..., run_name="__main__")` so the wrapped script sees
  the same `__main__` semantics it would under `python <script>`.
"""
from __future__ import annotations

import os
import random
import runpy
import sys


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS workspace config: required for deterministic matmul on cu11+
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np  # noqa: WPS433
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic kernel selection
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: SEED=<int> python seed_runner.py <script.py> [args...]", file=sys.stderr)
        sys.exit(2)

    seed = int(os.environ.get("SEED", "0"))
    _seed_everything(seed)
    print(f"[seed_runner] SEED={seed} -> python {' '.join(sys.argv[1:])}", flush=True)

    script = sys.argv[1]
    # Re-shape argv so the wrapped script sees a normal invocation.
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

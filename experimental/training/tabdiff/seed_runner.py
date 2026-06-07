"""Wrapper that fixes global RNG seeds before delegating to TabDiff's main.py.

TabDiff has --deterministic but no --seed CLI flag. This wrapper reads SEED
from the env, sets python / numpy / torch seeds, then execs the named TabDiff
script with the remaining argv. Used by train.sbatch in this directory.
"""
from __future__ import annotations

import os
import random
import runpy
import sys

import numpy as np
import torch

SEED = int(os.environ.get("SEED", "0"))
print(f"[seed_runner] SEED = {SEED}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if len(sys.argv) < 2:
    print("usage: python seed_runner.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)

script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")

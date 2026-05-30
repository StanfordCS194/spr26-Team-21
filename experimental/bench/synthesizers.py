"""Uniform synthesizer wrappers for the Trust Benchmark.

Every synthesizer exposes the same three-method interface:

    synth = SomeSynthesizer()
    synth.fit(train_df)
    sample_df = synth.sample(n)

This is so run_trust_benchmark.py can loop over synthesizers without caring
which one is which.

Currently shipped:
  - GaussianCopulaWrapper (SDV)  — fast, statistical
  - CTGANWrapper          (SDV)  — neural, slower, often higher fidelity
  - TVAEWrapper           (SDV)  — neural autoencoder, fastest neural

TabDDPMWrapper lives in PR #6 once we have a trained checkpoint.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )
    _SDV_AVAILABLE = True
except ImportError:
    _SDV_AVAILABLE = False


class Synthesizer(ABC):
    """Tiny abstract base — name + fit + sample. That's all the benchmark needs."""
    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...
    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...


def _build_metadata(df: pd.DataFrame) -> "SingleTableMetadata":
    """SDV's auto-detection works fine for our datasets but logs warnings on
    every numeric. Detect once, reuse across runs."""
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta


class GaussianCopulaWrapper(Synthesizer):
    """SDV's GaussianCopulaSynthesizer — the project's current baseline."""
    name = "GaussianCopula"

    def __init__(self):
        self._syn = None

    def fit(self, df: pd.DataFrame) -> None:
        self._syn = GaussianCopulaSynthesizer(_build_metadata(df))
        self._syn.fit(df)

    def sample(self, n: int) -> pd.DataFrame:
        return self._syn.sample(num_rows=n)


class CTGANWrapper(Synthesizer):
    """SDV's CTGANSynthesizer — conditional GAN with mode-specific normalization."""
    name = "CTGAN"

    def __init__(self, epochs: int = 100):
        self._syn = None
        self._epochs = epochs

    def fit(self, df: pd.DataFrame) -> None:
        self._syn = CTGANSynthesizer(_build_metadata(df), epochs=self._epochs, verbose=False)
        self._syn.fit(df)

    def sample(self, n: int) -> pd.DataFrame:
        return self._syn.sample(num_rows=n)


class TVAEWrapper(Synthesizer):
    """SDV's TVAESynthesizer — variational autoencoder, often beats CTGAN on fidelity."""
    name = "TVAE"

    def __init__(self, epochs: int = 100):
        self._syn = None
        self._epochs = epochs

    def fit(self, df: pd.DataFrame) -> None:
        self._syn = TVAESynthesizer(_build_metadata(df), epochs=self._epochs)
        self._syn.fit(df)

    def sample(self, n: int) -> pd.DataFrame:
        return self._syn.sample(num_rows=n)


def available() -> list[type[Synthesizer]]:
    """List the synthesizers usable in the current environment."""
    if not _SDV_AVAILABLE:
        return []
    return [GaussianCopulaWrapper, CTGANWrapper, TVAEWrapper]


REGISTRY = {cls.name: cls for cls in [GaussianCopulaWrapper, CTGANWrapper, TVAEWrapper]}

"""TabDDPMSynthesizer — load a trained TabDDPM checkpoint and sample synthetic rows.

The wrapper implements the same three-method shape as the SDV-based wrappers in
`experimental/bench/synthesizers.py` (name + fit + sample), with one important
difference: `fit()` is a no-op. TabDDPM training is heavy (GPU-bound, minutes to
hours) and lives on Sherlock; the wrapper consumes a pre-trained checkpoint
produced there.

Usage
-----
    from experimental.tabddpm.wrapper import TabDDPMSynthesizer
    synth = TabDDPMSynthesizer(checkpoint_dir="<path to dir with model.pt + config.toml>")
    synth_df = synth.sample(500)

Dependencies are loaded lazily inside __init__ so importing this module is cheap
and works on a machine that doesn't have the Yandex tab-ddpm repo on PYTHONPATH.
The actual sampling requires:
  - torch >= 2.0
  - The Yandex `tab_ddpm` python package importable
    (clone https://github.com/yandex-research/tab-ddpm and add it to PYTHONPATH)
  - rtdl, libzero (per the upstream requirements)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TabDDPMSynthesizer:
    """Wraps a Yandex TabDDPM checkpoint as an Aperture-bench-compatible synthesizer.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing model.pt, model_ema.pt, config.toml, info.json — the
        layout the Yandex pipeline writes after training.
    device : str
        "cuda" or "cpu". "cuda" requires a GPU; "cpu" is supported but slow.
    use_ema : bool
        If True (default) use the EMA-averaged weights — usually a touch better.
    """

    name = "TabDDPM"
    is_pretrained = True  # tells the bench to skip fit() in its run loop

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda",
        use_ema: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.use_ema = use_ema
        self._model = None
        self._diffusion = None
        self._config: dict[str, Any] | None = None
        self._info: dict[str, Any] | None = None

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint_dir {self.checkpoint_dir} not found")
        if not (self.checkpoint_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"{self.checkpoint_dir / 'model.pt'} missing — has the model been trained?"
            )

    def fit(self, df: pd.DataFrame) -> None:
        """NO-OP. TabDDPM training is GPU-bound and lives on a training box.

        To train on a new dataset, run the pipeline on Sherlock:

            ssh sherlock 'srun --partition=rbaltman --gres=gpu:1 --time=02:00:00 \\
                python /scratch/.../tab-ddpm/scripts/pipeline.py \\
                --config /scratch/.../tab-ddpm/exp/<dataset>/ddpm_cb_best/config.toml \\
                --train --sample --change_val'

        Then point TabDDPMSynthesizer at the resulting checkpoint directory.
        See ./README.md for the full training-from-scratch recipe.
        """
        raise NotImplementedError(
            "TabDDPM checkpoints are trained offline on GPU hardware. "
            "See experimental/tabddpm/README.md for the training recipe."
        )

    def _lazy_load(self) -> None:
        """Import tab_ddpm + load the model the first time sample() is called."""
        if self._model is not None:
            return

        try:
            import tomli
            import torch
            from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
            from tab_ddpm.modules import MLPDiffusion
        except ImportError as e:
            raise ImportError(
                "TabDDPMSynthesizer needs the Yandex tab-ddpm package on PYTHONPATH. "
                "Clone https://github.com/yandex-research/tab-ddpm and add its root to "
                "PYTHONPATH, plus `pip install torch rtdl libzero tomli`. "
                f"Underlying error: {e}"
            ) from e

        with open(self.checkpoint_dir / "config.toml", "rb") as f:
            self._config = tomli.load(f)

        info_path = self.checkpoint_dir / "info.json"
        if info_path.exists():
            import json
            self._info = json.loads(info_path.read_text())

        model_params = self._config["model_params"]
        diff_params = self._config["diffusion_params"]
        num_numerical_features = self._config["num_numerical_features"]

        # Reconstruct the MLPDiffusion model with the same architecture as training.
        rtdl_params = model_params.get("rtdl_params", {})
        self._model = MLPDiffusion(
            d_in=model_params["d_in"],
            num_classes=model_params["num_classes"],
            is_y_cond=model_params["is_y_cond"],
            rtdl_params=rtdl_params,
        ).to(self.device)

        weights_path = self.checkpoint_dir / ("model_ema.pt" if self.use_ema else "model.pt")
        state = torch.load(weights_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.eval()

        # Reconstruct the diffusion wrapper.
        # num_classes_expanded helps multinomial diffusion know per-cat-feature sizes;
        # the upstream code reads it from the dataset, so for sampling-only we pass [] and
        # rely on num_numerical_features for the slicing.
        self._diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([]),
            num_numerical_features=num_numerical_features,
            denoise_fn=self._model,
            num_timesteps=diff_params["num_timesteps"],
            gaussian_loss_type=diff_params.get("gaussian_loss_type", "mse"),
            scheduler=diff_params.get("scheduler", "cosine"),
            device=torch.device(self.device),
        ).to(self.device)
        self._diffusion.eval()

    def sample(self, n: int) -> pd.DataFrame:
        """Sample n synthetic rows. Returns a DataFrame with the original schema.

        Note: column dtypes follow what TabDDPM emits — numerics come back as float
        (even columns that were conceptually int, like Age), and integer-coded
        categoricals come back as ints. Re-map to the original string vocabularies
        upstream if the consuming code needs them.
        """
        self._lazy_load()

        import torch
        with torch.no_grad():
            samples, _ = self._diffusion.sample_all(n, batch_size=min(n, 4096), ddim=False)
        samples = samples.cpu().numpy()

        n_num = self._config["num_numerical_features"]
        cols_num = [f"num_{i}" for i in range(n_num)]
        cols_cat = [f"cat_{i}" for i in range(samples.shape[1] - n_num)]
        df = pd.DataFrame(samples, columns=cols_num + cols_cat)

        return df


def load_from_huggingface(repo_id: str, cache_dir: str | Path | None = None) -> TabDDPMSynthesizer:
    """Convenience: download a published checkpoint from HuggingFace and instantiate.

    Example:
        >>> synth = load_from_huggingface("mstojkov2024/aperture-tabddpm-pima")
        >>> df = synth.sample(500)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub to use load_from_huggingface") from e
    local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    return TabDDPMSynthesizer(local_dir)

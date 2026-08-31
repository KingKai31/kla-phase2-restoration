"""
Synthetic NoisyLR generator for Phase 2 (SEM/NFFA-EUROPE data), calibrated
on reports/phase2_deep_dive.md's compound noise model - NOT Phase 1's
pure-multiplicative model, which was rejected on this data (Gamma-only
R2=0.9926 vs compound R2=0.9997, full-dataset comparison).

Model: NoisyLR = box_downsample(GT) * M + sqrt(box_downsample(GT)/K) * Z + A + bias(x)

  M ~ Gamma(L_gain, 1/L_gain), mean 1   multiplicative detector-gain noise
  Z ~ N(0, 1)                           Poisson shot-noise (Gaussian-approximated),
                                         scaled by sqrt(GT/K) so its variance is GT/K
  A ~ N(0, sigma_A)                     constant read-noise floor
  bias(x)                               cubic empirical correction for the measured
                                         brightness-dependent mean-residual bias
                                         (reports/residual_bias_investigation.json) -
                                         NOT physically derived, an empirical fit,
                                         documented as a known minor approximation
                                         the same way Phase 1 handled an analogous
                                         unexplained effect

giving Var(NoisyLR | GT) = GT^2/L_gain + GT/K + sigma_A^2, matching the
measured compound model exactly.

(L_gain, K_poisson, sigma_A) are drawn per synthesized image from the
per-cluster fitted triples (reports/compound_model_per_cluster_fits.csv) -
this pool represents REAL measured population diversity (17 real fitted
triples across visually-distinct unsupervised clusters), the same
"bootstrap/sample from real per-source fits, don't use one fixed value"
principle Phase 1 used for its per-source Gamma L pool. Excludes cluster
18 (flagged as a likely fitting-stability artifact from its unusually
dark/low-brightness image content, not a confirmed distinct physical
regime - reports/phase2_deep_dive.md) and clusters 9/16 (too small to fit
a stable curve, <50 images after scale-bar exclusion).

No blur kernel is modeled, matching Phase 1's approach and for the same
reason it wasn't revisited here: this data-understanding pass focused on
the noise MODEL FAMILY (compound vs. pure), not spatial blur - flagged as
open for a future insurance-check pass if the FFT comparison below shows
a spectral mismatch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


class CompoundNoiseDegrader:
    def __init__(self, reports_dir: Path, seed: int = 0,
                 exclude_clusters: tuple = (18, 9, 16), min_cluster_n: int = 50):
        reports_dir = Path(reports_dir)
        fits = pd.read_csv(reports_dir / "compound_model_per_cluster_fits.csv")
        fits = fits.dropna(subset=["a", "c", "e"])
        fits = fits[~fits["cluster"].isin(exclude_clusters)]
        fits = fits[fits["n"] >= min_cluster_n]
        if len(fits) == 0:
            raise ValueError("No valid per-cluster fits available to sample from")

        self.L_gain_pool = fits["L_gain"].to_numpy()
        self.K_poisson_pool = fits["K_poisson"].to_numpy()
        self.sigma_A_pool = fits["sigma_A"].to_numpy()
        self.n_pool = len(fits)

        with open(reports_dir / "residual_bias_investigation.json") as f:
            bias_info = json.load(f)
        self.bias_coeffs = np.array(bias_info["cubic_fit"]["coeffs_highest_first"])

        self.rng = np.random.default_rng(seed)

    def _sample_params(self):
        idx = self.rng.integers(0, self.n_pool)
        return self.L_gain_pool[idx], self.K_poisson_pool[idx], self.sigma_A_pool[idx]

    def _bias(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.bias_coeffs, x)

    def degrade(self, gt: np.ndarray, factor: int = 2) -> np.ndarray:
        """Real KLA/Phase-2-style 256x256 GT - exact factor-2 box downsample,
        matching the confirmed 256<->128 pair (see reports/phase2_data_inventory.md)."""
        gt_down = box_downsample(gt.astype(np.float64), factor)
        return self._apply_noise(gt_down)

    def _apply_noise(self, gt_down: np.ndarray) -> np.ndarray:
        L_gain, K_poisson, sigma_A = self._sample_params()

        M = self.rng.gamma(shape=L_gain, scale=1.0 / L_gain, size=gt_down.shape)
        Z = self.rng.normal(0.0, 1.0, size=gt_down.shape)
        A = self.rng.normal(0.0, sigma_A, size=gt_down.shape)
        shot_term = np.sqrt(np.clip(gt_down, 0.0, None) / K_poisson) * Z
        bias_term = self._bias(np.clip(gt_down, 0.0, 1.0))

        noisy = gt_down * M + shot_term + A + bias_term
        return noisy.astype(np.float32)

    def degrade_external(self, clean_img: np.ndarray, tile_size: int = 256, factor: int = 2) -> tuple:
        """Arbitrary-size external clean image support, mirroring Phase 1's
        pattern for future external-data augmentation if Phase 2 adopts one -
        not currently used (no external data mix decided for Phase 2 yet)."""
        arr = clean_img.astype(np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        h, w = arr.shape
        if h < tile_size or w < tile_size:
            scale = tile_size / min(h, w)
            new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
            y_idx = np.clip((np.arange(tile_size) / scale).astype(int), 0, h - 1)
            x_idx = np.clip((np.arange(tile_size) / scale).astype(int), 0, w - 1)
            arr = arr[y_idx][:, x_idx]
            h, w = arr.shape
        top = self.rng.integers(0, h - tile_size + 1)
        left = self.rng.integers(0, w - tile_size + 1)
        gt_tile = arr[top: top + tile_size, left: left + tile_size]

        gt_down = box_downsample(gt_tile.astype(np.float64), factor)
        noisy_tile = self._apply_noise(gt_down)
        return gt_tile.astype(np.float32), noisy_tile

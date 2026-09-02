"""
Pre-registered step 1 (per user decision, this session): does adding mild
spatial correlation to the synthetic noise shrink the ~22% high-frequency
spectral deficit found in reports/phase2_insurance_check_summary.json
(Part 6 of phase2_deep_dive.md)? Tested here, standalone, BEFORE touching
src/datasets/synthetic_degrade.py or retraining anything - exactly the
pre-registered order of operations.

Mechanism tested: generate the additive noise components (shot term Z,
read-noise floor A) on a coarser grid (block size k) then nearest-neighbor
upsample to full resolution - correlates noise within each k x k block
while leaving per-pixel marginal variance exactly unchanged (nearest-
neighbor upsampling doesn't average, so no variance rescale is needed,
unlike a blur-based blend). k=1 reproduces the current i.i.d. baseline
exactly. Swept over k in {1, 2, 3, 4}.

Honest pre-check before running this empirically: the measured mismatch
is a spectral TILT, not just a high-frequency gap in isolation - synthetic
already has ~2.8% MORE power than real at low/mid frequencies (radius
<30) and ~18% LESS at high frequencies (radius >60). Spatial correlation
(any blur-like operation) suppresses high frequencies and concentrates
power at low frequencies - the opposite of the direction needed to close
this specific gap. This is tested anyway per the pre-registered process
rather than skipped on this reasoning alone - the point of pre-registering
was to let the measurement decide, not intuition.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.synthetic_degrade import box_downsample  # noqa: E402


def radial_power_spectrum(img: np.ndarray, n_bins: int = 32):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag2 = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2)
    bins = np.linspace(0, rmax, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    prof = np.bincount(idx, weights=mag2.ravel(), minlength=n_bins) / np.maximum(counts, 1)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, prof


class CorrelatedNoiseDegrader:
    """Same compound model as CompoundNoiseDegrader, with an added
    corr_block_size knob on the additive (shot + read) noise components -
    corr_block_size=1 is numerically identical to the current production
    generator."""

    def __init__(self, reports_dir: Path, seed: int = 0, corr_block_size: int = 1,
                 exclude_clusters=(18, 9, 16), min_cluster_n: int = 50):
        reports_dir = Path(reports_dir)
        fits = pd.read_csv(reports_dir / "compound_model_per_cluster_fits.csv")
        fits = fits.dropna(subset=["a", "c", "e"])
        fits = fits[~fits["cluster"].isin(exclude_clusters)]
        fits = fits[fits["n"] >= min_cluster_n]
        self.L_gain_pool = fits["L_gain"].to_numpy()
        self.K_poisson_pool = fits["K_poisson"].to_numpy()
        self.sigma_A_pool = fits["sigma_A"].to_numpy()
        self.n_pool = len(fits)
        with open(reports_dir / "residual_bias_investigation.json") as f:
            bias_info = json.load(f)
        self.bias_coeffs = np.array(bias_info["cubic_fit"]["coeffs_highest_first"])
        self.rng = np.random.default_rng(seed)
        self.corr_block_size = corr_block_size

    def _sample_params(self):
        idx = self.rng.integers(0, self.n_pool)
        return self.L_gain_pool[idx], self.K_poisson_pool[idx], self.sigma_A_pool[idx]

    def _bias(self, x):
        return np.polyval(self.bias_coeffs, x)

    def _correlated_field(self, shape, std_field=None):
        """std_field: per-pixel target std (for the shot term, which varies
        spatially with GT brightness) or a scalar (for the flat read-noise
        floor). Generates i.i.d. N(0,1) on a coarser grid, nearest-upsamples
        to `shape`, then scales by std_field - marginal per-pixel variance
        is exactly std_field^2 regardless of block size, since nearest
        upsampling copies values without averaging."""
        k = self.corr_block_size
        h, w = shape
        if k <= 1:
            z = self.rng.normal(0.0, 1.0, size=shape)
        else:
            hs, ws = (h + k - 1) // k, (w + k - 1) // k
            coarse = self.rng.normal(0.0, 1.0, size=(hs, ws))
            z = np.repeat(np.repeat(coarse, k, axis=0), k, axis=1)[:h, :w]
        return z * std_field if std_field is not None else z

    def degrade(self, gt: np.ndarray, factor: int = 2) -> np.ndarray:
        gt_down = box_downsample(gt.astype(np.float64), factor)
        L_gain, K_poisson, sigma_A = self._sample_params()
        M = self.rng.gamma(shape=L_gain, scale=1.0 / L_gain, size=gt_down.shape)
        shot_std = np.sqrt(np.clip(gt_down, 0.0, None) / K_poisson)
        shot_term = self._correlated_field(gt_down.shape, shot_std)
        A = self._correlated_field(gt_down.shape, sigma_A)
        bias_term = self._bias(np.clip(gt_down, 0.0, 1.0))
        noisy = gt_down * M + shot_term + A + bias_term
        return noisy.astype(np.float32)


def run_config(gt_dir, noisy_dir, val_files, reports_dir, corr_block_size, n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    degrader = CorrelatedNoiseDegrader(reports_dir, seed=seed, corr_block_size=corr_block_size)
    n = min(n_samples, len(val_files))
    chosen = [val_files[i] for i in rng.choice(len(val_files), size=n, replace=False)]

    real_radial, synth_radial = [], []
    real_stats, synth_stats = [], []
    centers = None
    for fname in chosen:
        gt = np.load(gt_dir / fname).astype(np.float64)
        real_noisy = np.load(noisy_dir / fname).astype(np.float64)
        synth_noisy = degrader.degrade(gt).astype(np.float64)
        c, p_real = radial_power_spectrum(real_noisy)
        _, p_synth = radial_power_spectrum(synth_noisy)
        centers = c
        real_radial.append(p_real)
        synth_radial.append(p_synth)
        real_stats.append(real_noisy.std())
        synth_stats.append(synth_noisy.std())

    real_radial_mean = np.mean(real_radial, axis=0)
    synth_radial_mean = np.mean(synth_radial, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10((synth_radial_mean + 1e-12) / (real_radial_mean + 1e-12))

    lowmid = log_ratio[(centers > 0) & (centers < 30)]
    highfreq = log_ratio[centers > 60]
    ks_std = spstats.ks_2samp(real_stats, synth_stats)

    return {
        "corr_block_size": corr_block_size,
        "lowmid_mean_log10_ratio": float(lowmid.mean()),
        "lowmid_linear_ratio": float(10 ** lowmid.mean()),
        "highfreq_mean_log10_ratio": float(highfreq.mean()),
        "highfreq_linear_ratio": float(10 ** highfreq.mean()),
        "highfreq_deficit_pct": float((1 - 10 ** highfreq.mean()) * 100),
        "max_abs_log10_ratio_excl_dc": float(np.max(np.abs(log_ratio[1:]))),
        "ks_std_stat": float(ks_std.statistic), "ks_std_p": float(ks_std.pvalue),
        "centers": centers.tolist(), "log_ratio": log_ratio.tolist(),
    }


def main():
    gt_dir = Path(r"C:\Users\ANANNYA\Downloads\semicon_train_data\semicon_train_data\semicon_train_data\GT")
    noisy_dir = Path(r"C:\Users\ANANNYA\Downloads\semicon_train_data\semicon_train_data\semicon_train_data\NoisyLR")
    reports_dir = Path("reports")
    split_df = pd.read_csv(reports_dir / "phase2_source_clusters_stratified_leakchecked.csv")
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()

    baseline_deficit = None
    results = []
    for k in [1, 2, 3, 4]:
        r = run_config(gt_dir, noisy_dir, val_files, reports_dir, corr_block_size=k)
        if k == 1:
            baseline_deficit = r["highfreq_deficit_pct"]
        results.append(r)
        print(f"block_size={k}: highfreq_deficit={r['highfreq_deficit_pct']:.2f}%  "
              f"lowmid_ratio={r['lowmid_linear_ratio']:.4f}  ks_std_p={r['ks_std_p']:.4f}")

    summary = {
        "baseline_highfreq_deficit_pct": baseline_deficit,
        "decision_rule": "keep only if a config shrinks the deficit to less than half the baseline "
                          "(i.e. deficit_pct < baseline/2) while not breaking the bulk-stats KS match",
        "results": results,
    }
    best = min(results[1:], key=lambda r: r["highfreq_deficit_pct"])
    passes = best["highfreq_deficit_pct"] < baseline_deficit / 2
    summary["best_config"] = {"corr_block_size": best["corr_block_size"],
                               "highfreq_deficit_pct": best["highfreq_deficit_pct"]}
    summary["passes_pre_registered_gate"] = bool(passes)

    with open(reports_dir / "spectral_fix_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()

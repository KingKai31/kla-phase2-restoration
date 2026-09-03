"""
Item 5 (improvement pass): the audit found that tests/test_run_py_robustness.py
imports ZERO src modules - the model, every loss term, and the synthetic
degrader (732 lines) were untested except indirectly through run.py's
separate inlined copy. This file covers src/ directly.

Deliberate design decision on the "model is defined 3x" finding: run.py
MUST stay self-contained (inlined model, no src/ import) - that is a hard
submission requirement verified by the compliance chain, so refactoring
run.py to import from src/ would trade a real submission gate for code
tidiness. Instead, TestInlinedModelMatchesSource below asserts the two
copies are AST-identical, converting a silent-divergence risk into a
loud, re-runnable test failure. That is strictly safer than the refactor.

Run: pytest tests/test_src_modules.py -v
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.nafnet import NAFNetSR  # noqa: E402
from src.losses.stageB_composite import (  # noqa: E402
    CharbonnierLoss, SobelEdgeLoss, RangeConsistencyPenalty,
)
from src.datasets.synthetic_degrade import (  # noqa: E402
    box_downsample, to_unit_grayscale, CompoundNoiseDegrader,
)


# ---------------------------------------------------------------------------
class TestInlinedModelMatchesSource:
    """run.py inlines its model classes instead of importing from src/ (a
    hard self-containment requirement). These must never silently diverge.

    run.py now inlines TWO class groups, each guarded against its own
    source module:
      - the NAFNet building blocks + NAFNetSR  -> src/models/nafnet.py
      - NAFNetSRDecoderCapacity (the SHIPPED   -> src/models/nafnet_decoder_capacity.py
        architecture, adopted after clearing its pre-registered gate)

    Docstrings are stripped before comparison. The guard exists to catch
    BEHAVIORAL divergence, and a docstring cannot cause any; run.py's
    inlined copies legitimately carry extra explanation for a reader of
    the self-contained submission file, where the source modules put that
    context at module level instead. Comments and formatting were already
    normalized away by the unparse round-trip - this extends the same
    principle to docstrings. All executable logic is still compared
    exactly."""

    SOURCE_MAP = {
        "src/models/nafnet.py": ["LayerNorm2d", "SimpleGate", "SimplifiedChannelAttention",
                                  "NAFBlock", "NAFNetSR"],
        "src/models/nafnet_decoder_capacity.py": ["NAFNetSRDecoderCapacity"],
    }

    @staticmethod
    def _strip_docstring(node):
        node = ast.parse(ast.unparse(node)).body[0]
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    @classmethod
    def _classes(cls, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # round-trip through unparse so formatting/comment differences don't
        # matter; strip docstrings so only executable logic is compared
        return {n.name: ast.dump(ast.parse(ast.unparse(cls._strip_docstring(n))))
                for n in tree.body if isinstance(n, ast.ClassDef)}

    def test_run_py_defines_every_expected_class(self):
        run = self._classes(REPO_ROOT / "run.py")
        expected = {c for names in self.SOURCE_MAP.values() for c in names}
        missing = expected - set(run)
        assert not missing, f"run.py is missing model classes: {missing}"

    def test_shipped_architecture_is_the_decoder_capacity_variant(self):
        """The adopted architecture must actually be the one run.py builds -
        guards against run.py silently reverting to the plain backbone."""
        src = (REPO_ROOT / "run.py").read_text(encoding="utf-8")
        assert "model = NAFNetSRDecoderCapacity(" in src, (
            "run.py's load_model() no longer constructs NAFNetSRDecoderCapacity - "
            "the shipped checkpoint would fail to load."
        )

    @pytest.mark.parametrize("source_rel", list(SOURCE_MAP))
    def test_every_inlined_class_is_ast_identical_to_its_source(self, source_rel):
        src = self._classes(REPO_ROOT / source_rel)
        run = self._classes(REPO_ROOT / "run.py")
        expected = self.SOURCE_MAP[source_rel]
        mismatched = [c for c in expected if c in src and c in run and src[c] != run[c]]
        assert not mismatched, (
            f"run.py's inlined model has DIVERGED from {source_rel} for: {mismatched}. "
            "The shipped inference code no longer matches the trained architecture."
        )


# ---------------------------------------------------------------------------
class TestNAFNetSR:
    def test_upscales_by_two_and_preserves_batch(self):
        m = NAFNetSR(img_channel=1, width=8, upscale=2).eval()
        with torch.no_grad():
            out = m(torch.rand(2, 1, 32, 32))
        assert out.shape == (2, 1, 64, 64)
        assert torch.isfinite(out).all()

    def test_non_multiple_of_padder_size_still_works(self):
        """Reflect-pads to a multiple of 16 internally, then crops back."""
        m = NAFNetSR(img_channel=1, width=8, upscale=2).eval()
        with torch.no_grad():
            out = m(torch.rand(1, 1, 40, 56))
        assert out.shape == (1, 1, 80, 112)

    def test_tiny_input_raises_the_documented_reflect_pad_limit(self):
        """<=8px per side is a real, documented architectural limit that
        run.py catches with its per-image try/except and classical fallback."""
        m = NAFNetSR(img_channel=1, width=8, upscale=2).eval()
        with pytest.raises(RuntimeError, match="Padding size should be less"):
            with torch.no_grad():
                m(torch.rand(1, 1, 8, 8))

    def test_deterministic_for_fixed_weights(self):
        torch.manual_seed(0)
        m = NAFNetSR(img_channel=1, width=8, upscale=2).eval()
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            assert torch.equal(m(x), m(x))


# ---------------------------------------------------------------------------
class TestLossTerms:
    def test_charbonnier_is_zero_at_perfect_match_up_to_eps(self):
        loss = CharbonnierLoss(eps=1e-3)
        x = torch.rand(2, 1, 16, 16)
        assert loss(x, x).item() == pytest.approx(1e-3, abs=1e-6)

    def test_charbonnier_increases_with_error(self):
        loss = CharbonnierLoss()
        t = torch.zeros(1, 1, 8, 8)
        assert loss(torch.full_like(t, 0.5), t).item() > loss(torch.full_like(t, 0.1), t).item()

    def test_sobel_detects_an_edge_and_ignores_flat(self):
        sobel = SobelEdgeLoss()
        flat = torch.full((1, 1, 16, 16), 0.5)
        edge = flat.clone()
        edge[:, :, :, 8:] = 1.0
        # a flat prediction against a flat target: no edge disagreement
        flat_loss = sobel(flat, flat).item()
        # a flat prediction against an edged target: real disagreement
        edge_loss = sobel(flat, edge).item()
        assert edge_loss > flat_loss * 2, "Sobel term does not respond to a real edge"

    def test_sobel_is_polarity_blind_in_the_interior(self):
        """Gradient MAGNITUDE is used, so a polarity-flipped edge is
        indistinguishable - real behaviour, and the reason the term cannot
        tell a dark->bright boundary from a bright->dark one."""
        sobel = SobelEdgeLoss()
        a = torch.full((1, 1, 16, 16), 0.2); a[:, :, :, 8:] = 0.8
        b = torch.full((1, 1, 16, 16), 0.8); b[:, :, :, 8:] = 0.2
        ga, gb = sobel._gradient_magnitude(a), sobel._gradient_magnitude(b)
        # interior only - borders are contaminated by zero-padding, see below
        assert torch.allclose(ga[:, :, 1:-1, 1:-1], gb[:, :, 1:-1, 1:-1], atol=1e-5)

    def test_sobel_border_is_contaminated_by_zero_padding(self):
        """CHARACTERIZATION TEST for a real defect found while writing this
        suite (not caught by the original audit): SobelEdgeLoss uses
        F.conv2d(..., padding=1), which ZERO-pads. Every image border
        therefore produces a large spurious gradient proportional to border
        brightness, unrelated to any real structure.

        Measured on a real 256x256 official-test GT: border pixels are 1.56%
        of the image but carry ~3.7x the interior mean gradient magnitude.
        This COMPOUNDS the audit's main finding (the term is averaged over
        all 65k pixels, so true boundaries at <5% of pixels are diluted) -
        part of the little signal it does carry is padding artifact.
        A boundary-masked edge loss would exclude borders naturally.

        This test documents the behaviour so a future fix (replicate/reflect
        padding, or a masked term) shows up as an intentional change."""
        sobel = SobelEdgeLoss()
        border_by_brightness = {}
        for v in (0.2, 0.5, 0.8):
            img = torch.full((1, 1, 32, 32), v)  # uniform: ANY gradient is artifact
            g = sobel._gradient_magnitude(img)
            border = torch.zeros_like(g, dtype=torch.bool)
            border[:, :, 0, :] = border[:, :, -1, :] = True
            border[:, :, :, 0] = border[:, :, :, -1] = True
            # interior floor is sqrt(1e-6)=1e-3, the epsilon inside _gradient_magnitude
            assert g[~border].max().item() <= 1.001e-3, "uniform image must have no real interior gradient"
            border_by_brightness[v] = g[border].mean().item()

        # the artifact is large and scales LINEARLY with border brightness -
        # the signature of convolving against a zero pad
        assert border_by_brightness[0.8] > 3.0
        assert border_by_brightness[0.8] / border_by_brightness[0.2] == pytest.approx(4.0, rel=0.05)

    def test_range_penalty_zero_inside_and_positive_outside(self):
        pen = RangeConsistencyPenalty(0.0, 1.0)
        assert pen(torch.rand(1, 1, 8, 8)).item() == pytest.approx(0.0, abs=1e-9)
        assert pen(torch.full((1, 1, 8, 8), 1.5)).item() > 0
        assert pen(torch.full((1, 1, 8, 8), -0.5)).item() > 0

    def test_range_penalty_is_quadratic_in_overshoot(self):
        pen = RangeConsistencyPenalty(0.0, 1.0)
        one = pen(torch.full((1, 1, 4, 4), 2.0)).item()   # overshoot 1.0
        two = pen(torch.full((1, 1, 4, 4), 3.0)).item()   # overshoot 2.0
        assert two == pytest.approx(4 * one, rel=1e-6)


# ---------------------------------------------------------------------------
class TestSyntheticDegrade:
    def test_box_downsample_shape_and_mean_preserving(self):
        arr = np.arange(64, dtype=np.float64).reshape(8, 8)
        out = box_downsample(arr, 2)
        assert out.shape == (4, 4)
        assert out.mean() == pytest.approx(arr.mean())

    def test_box_downsample_factor_one_is_identity(self):
        arr = np.random.default_rng(0).random((8, 8))
        assert np.array_equal(box_downsample(arr, 1), arr)

    def test_to_unit_grayscale_handles_rgb_uint8_and_float(self):
        rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert to_unit_grayscale(rgb).shape == (4, 4)
        assert to_unit_grayscale(rgb).max() == pytest.approx(1.0)
        already = np.full((4, 4), 0.5)
        assert to_unit_grayscale(already).max() == pytest.approx(0.5)

    @pytest.fixture(scope="class")
    def degrader(self):
        reports = REPO_ROOT / "reports"
        if not (reports / "compound_model_per_cluster_fits.csv").exists():
            pytest.skip("per-cluster fits not available")
        return CompoundNoiseDegrader(reports, seed=0)

    def test_degrade_halves_resolution(self, degrader):
        out = degrader.degrade(np.full((256, 256), 0.5, dtype=np.float32))
        assert out.shape == (128, 128)
        assert np.all(np.isfinite(out))

    def test_noise_variance_grows_with_brightness(self, degrader):
        """The whole point of the compound model: variance is signal-dependent
        (a*x^2 + c*x + e), so a bright field must be noisier than a dark one."""
        dark = degrader.degrade(np.full((256, 256), 0.1, dtype=np.float32))
        bright = degrader.degrade(np.full((256, 256), 0.9, dtype=np.float32))
        assert bright.std() > dark.std() * 2

    def test_degrade_is_seed_deterministic(self):
        reports = REPO_ROOT / "reports"
        if not (reports / "compound_model_per_cluster_fits.csv").exists():
            pytest.skip("per-cluster fits not available")
        gt = np.full((256, 256), 0.5, dtype=np.float32)
        a = CompoundNoiseDegrader(reports, seed=7).degrade(gt)
        b = CompoundNoiseDegrader(reports, seed=7).degrade(gt)
        assert np.array_equal(a, b), "degrader is not reproducible for a fixed seed"

    def test_different_seeds_give_different_noise(self):
        reports = REPO_ROOT / "reports"
        if not (reports / "compound_model_per_cluster_fits.csv").exists():
            pytest.skip("per-cluster fits not available")
        gt = np.full((256, 256), 0.5, dtype=np.float32)
        a = CompoundNoiseDegrader(reports, seed=1).degrade(gt)
        b = CompoundNoiseDegrader(reports, seed=2).degrade(gt)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
class TestDihedralAugmentation:
    """Item 1a's augmentation. Correctness matters: an augmentation that
    desynchronised the GT/NoisyLR pair would silently poison training."""

    @staticmethod
    def _dihedral(arr, k):
        from scripts.train_stage_a_aug_ema import dihedral
        return dihedral(arr, k)

    def test_all_eight_transforms_preserve_shape_and_content(self):
        rng = np.random.default_rng(0)
        arr = rng.random((16, 16)).astype(np.float32)
        for k in range(8):
            out = self._dihedral(arr, k)
            assert out.shape == arr.shape
            assert sorted(out.ravel().tolist()) == pytest.approx(sorted(arr.ravel().tolist()))

    def test_identity_at_k_zero(self):
        arr = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        assert np.array_equal(self._dihedral(arr, 0), arr)

    def test_transforms_are_distinct_on_an_asymmetric_input(self):
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        outs = [self._dihedral(arr, k).tobytes() for k in range(8)]
        assert len(set(outs)) == 8, "dihedral group should give 8 distinct results"

    def test_pair_stays_aligned_under_the_same_k(self):
        """GT and NoisyLR are at different resolutions; the same k must map
        corresponding regions to corresponding regions."""
        gt = np.zeros((8, 8), dtype=np.float32); gt[0, 0] = 1.0      # marker top-left
        noisy = np.zeros((4, 4), dtype=np.float32); noisy[0, 0] = 1.0
        for k in range(8):
            g, n = self._dihedral(gt, k), self._dihedral(noisy, k)
            gy, gx = np.argwhere(g == 1.0)[0]
            ny, nx = np.argwhere(n == 1.0)[0]
            # marker must land in the same corner in both (scaled by 2)
            assert (gy // 7, gx // 7) == (ny // 3, nx // 3), f"pair desynchronised at k={k}"

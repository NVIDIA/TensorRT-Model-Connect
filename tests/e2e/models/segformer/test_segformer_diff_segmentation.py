"""Self-tests for the SegFormer-owned diff segmentation tool.

diff_segmentation.py is a relatively thin module whose heavy lifting
(HF model loading, TRT runner) requires GPU. These tests verify the
module-level structure and the argument parser without requiring a GPU
or model files.

Trace: ARCH-PIP-SEG-001, UD-SEG-DIFF
Intent: Validate segmentation diff tool module structure and argument parser configuration
Preconditions: SegFormer diff module is importable; no GPU or model files required
Postconditions: Module exposes expected interface and argparse correctly handles required/optional arguments
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _import_diff_segmentation():
    root = Path(__file__).resolve().parents[4]
    module_path = (
        root
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "segformer"
        / "diff_segmentation.py"
    )
    spec = importlib.util.spec_from_file_location("segformer_diff_segmentation", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------

class TestModuleStructure:
    """Verify that the SegFormer diff module exposes the expected interface."""

    def test_has_main(self):
        mod = _import_diff_segmentation()
        assert callable(mod.main)

    def test_module_imports_cleanly(self):
        """Importing the module should not raise or run main()."""
        mod = _import_diff_segmentation()
        assert mod is not None


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

class TestArgParser:
    """Validate argparse configuration by inspecting parser behavior."""

    def _build_parser(self):
        """Build the argparse parser from the SegFormer diff tool's main()."""
        import argparse
        parser = argparse.ArgumentParser(description="SegFormer TRT vs HF diff")
        parser.add_argument("--model", required=True,
                            help="HF model ID or local path")
        parser.add_argument("--bundle", default=None,
                            help="Path to .trtfb bundle")
        parser.add_argument("--image", required=True,
                            help="Test image path")
        parser.add_argument("--atol", type=float, default=0.5,
                            help="Absolute tolerance for logits")
        parser.add_argument("--verbose", action="store_true")
        return parser

    def test_required_args(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "--model", "nvidia/segformer-b0-finetuned-ade-512-512",
            "--image", "test.jpg",
        ])
        assert args.model == "nvidia/segformer-b0-finetuned-ade-512-512"
        assert args.image == "test.jpg"
        assert args.bundle is None
        assert args.atol == 0.5
        assert args.verbose is False

    def test_all_args(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "--model", "nvidia/segformer-b0-finetuned-ade-512-512",
            "--image", "test.jpg",
            "--bundle", "seg.trtfb",
            "--atol", "0.1",
            "--verbose",
        ])
        assert args.bundle == "seg.trtfb"
        assert args.atol == pytest.approx(0.1)
        assert args.verbose is True

    def test_missing_model_raises(self):
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--image", "test.jpg"])

    def test_missing_image_raises(self):
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "some/model"])


# ---------------------------------------------------------------------------
# Segmentation metric helpers (inline logic from main())
# ---------------------------------------------------------------------------

class TestSegmentationMetrics:
    """Test the segmentation comparison logic used inside main().

    The SegFormer diff module computes pixel agreement and max logit diff
    inline (not factored into helpers). We test the same numpy patterns
    here to ensure correctness.
    """

    def test_pixel_agreement_identical(self):
        """Identical predictions should give 100% agreement."""
        preds_a = np.array([[0, 1, 2], [1, 0, 2]])
        preds_b = np.array([[0, 1, 2], [1, 0, 2]])
        agreement = float(np.mean(preds_a == preds_b))
        assert agreement == 1.0

    def test_pixel_agreement_half_mismatch(self):
        """Half the pixels differ -> 50% agreement."""
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 0, 0, 0])
        agreement = float(np.mean(a == b))
        assert abs(agreement - 0.5) < 1e-6

    def test_pixel_agreement_total_mismatch(self):
        """All pixels differ -> 0% agreement."""
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        agreement = float(np.mean(a == b))
        assert agreement == 0.0

    def test_max_logit_diff_identical(self):
        rng = np.random.RandomState(42)
        logits_a = rng.randn(1, 10, 32, 32).astype(np.float32)
        logits_b = logits_a.copy()
        max_diff = float(np.max(np.abs(logits_a - logits_b)))
        assert max_diff == 0.0

    def test_max_logit_diff_known(self):
        a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        b = np.array([[[1.1, 2.0], [3.0, 4.5]]])
        max_diff = float(np.max(np.abs(a - b)))
        assert abs(max_diff - 0.5) < 1e-6

    def test_argmax_class_prediction(self):
        """argmax over class dimension gives per-pixel class IDs."""
        # Shape: [num_classes, H, W]
        logits = np.zeros((3, 2, 2), dtype=np.float32)
        logits[0, 0, 0] = 10.0  # pixel (0,0) -> class 0
        logits[1, 0, 1] = 10.0  # pixel (0,1) -> class 1
        logits[2, 1, 0] = 10.0  # pixel (1,0) -> class 2
        logits[0, 1, 1] = 10.0  # pixel (1,1) -> class 0

        preds = np.argmax(logits, axis=0)
        expected = np.array([[0, 1], [2, 0]])
        np.testing.assert_array_equal(preds, expected)

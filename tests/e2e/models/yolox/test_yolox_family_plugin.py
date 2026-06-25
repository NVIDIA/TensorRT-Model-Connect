"""Tests for the YOLOX family plugin -- weight key mapping, detection config,
and CNN builder integration.

No GPU or TRT needed for weight loading tests.

Trace: ARCH-FAM-001, UD-FAM-YOLOX
Intent: Validate YOLOX object detection family plugin weight key mapping and CNN builder config
Preconditions: Synthetic safetensors with YOLOX conv/bn weight naming are available (currently skipped)
Postconditions: Plugin produces correct conv/bn weight keys for backbone, neck, and detection head
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

pytest.skip("YOLOX plugin not yet implemented", allow_module_level=True)

RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


def _conv_bn_tensors(prefix: str, in_ch: int, out_ch: int, ksize: int = 3) -> dict:
    """Generate conv + bn weight tensors for a given HF prefix."""
    t = {}
    t[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, ksize, ksize)
    t[f"{prefix}.bn.weight"] = _rand(out_ch)
    t[f"{prefix}.bn.bias"] = _rand(out_ch)
    t[f"{prefix}.bn.running_mean"] = _rand(out_ch)
    t[f"{prefix}.bn.running_var"] = np.abs(_rand(out_ch)) + 0.01
    return t


def _conv_only_tensors(prefix: str, in_ch: int, out_ch: int, ksize: int = 1) -> dict:
    """Generate conv-only weight tensors (weight + bias, no BN)."""
    t = {}
    t[f"{prefix}.weight"] = _rand(out_ch, in_ch, ksize, ksize)
    t[f"{prefix}.bias"] = _rand(out_ch)
    return t


def _bottleneck_tensors(prefix: str, ch: int) -> dict:
    """Generate bottleneck (2x conv-bn) tensors."""
    half = ch // 2
    t = {}
    t.update(_conv_bn_tensors(f"{prefix}.conv1", ch, half, 1))
    t.update(_conv_bn_tensors(f"{prefix}.conv2", half, ch, 3))
    return t


def _csp_tensors(prefix: str, in_ch: int, out_ch: int, num_blocks: int) -> dict:
    """Generate CSP layer tensors."""
    mid = out_ch // 2
    t = {}
    t.update(_conv_bn_tensors(f"{prefix}.main_conv", in_ch, mid, 1))
    t.update(_conv_bn_tensors(f"{prefix}.short_conv", in_ch, mid, 1))
    t.update(_conv_bn_tensors(f"{prefix}.final_conv", 2 * mid, out_ch, 1))
    for i in range(num_blocks):
        t.update(_bottleneck_tensors(f"{prefix}.m.{i}", mid))
    return t


def _make_yolox_tensors(
    num_classes: int = 4,
    depth_mul: float = 0.33,
    width_mul: float = 0.25,
) -> dict:
    """Generate a complete set of synthetic YOLOX weights matching HF key layout."""
    base_widths = [64, 128, 256, 512, 1024]
    widths = [max(round(w * width_mul), 1) for w in base_widths]
    base_depths = [3, 9, 9, 3]
    depths = [max(round(d * depth_mul), 1) for d in base_depths]

    stem_ch = widths[0]     # 16
    dark2_ch = widths[1]    # 32
    dark3_ch = widths[2]    # 64
    dark4_ch = widths[3]    # 128
    dark5_ch = widths[4]    # 256

    bb = "backbone.backbone"
    t = {}

    # Stem
    t.update(_conv_bn_tensors(f"{bb}.stem.0", 3, stem_ch, 6))

    # Dark2: downsample + CSP
    t.update(_conv_bn_tensors(f"{bb}.dark2.0", stem_ch, dark2_ch, 3))
    t.update(_csp_tensors(f"{bb}.dark2.1", dark2_ch, dark2_ch, depths[0]))

    # Dark3
    t.update(_conv_bn_tensors(f"{bb}.dark3.0", dark2_ch, dark3_ch, 3))
    t.update(_csp_tensors(f"{bb}.dark3.1", dark3_ch, dark3_ch, depths[1]))

    # Dark4
    t.update(_conv_bn_tensors(f"{bb}.dark4.0", dark3_ch, dark4_ch, 3))
    t.update(_csp_tensors(f"{bb}.dark4.1", dark4_ch, dark4_ch, depths[2]))

    # Dark5
    t.update(_conv_bn_tensors(f"{bb}.dark5.0", dark4_ch, dark5_ch, 3))
    t.update(_csp_tensors(f"{bb}.dark5.1", dark5_ch, dark5_ch, depths[3]))

    # FPN Neck
    neck = "backbone.neck"
    t.update(_conv_bn_tensors(f"{neck}.lateral_conv0", dark5_ch, dark4_ch, 1))
    t.update(_conv_bn_tensors(f"{neck}.reduce_conv1", dark4_ch, dark3_ch, 1))

    fpn_depth = max(round(3 * depth_mul), 1)
    t.update(_csp_tensors(f"{neck}.C3_p4", 2 * dark4_ch, dark4_ch, fpn_depth))
    t.update(_csp_tensors(f"{neck}.C3_p3", 2 * dark3_ch, dark3_ch, fpn_depth))
    t.update(_csp_tensors(f"{neck}.C3_n3", 2 * dark3_ch, dark3_ch, fpn_depth))
    t.update(_csp_tensors(f"{neck}.C3_n4", 2 * dark4_ch, dark4_ch, fpn_depth))

    t.update(_conv_bn_tensors(f"{neck}.bu_conv2", dark3_ch, dark3_ch, 3))
    t.update(_conv_bn_tensors(f"{neck}.bu_conv1", dark4_ch, dark4_ch, 3))

    # Detection Head (3 scales)
    head_ch = dark3_ch  # smallest FPN output channel count
    for scale_idx in range(3):
        if scale_idx == 0:
            in_ch = dark3_ch
        elif scale_idx == 1:
            in_ch = dark4_ch
        else:
            in_ch = dark5_ch
        # Stem
        t.update(_conv_bn_tensors(f"head.stems.{scale_idx}", in_ch, head_ch, 1))
        # Cls branch
        t.update(_conv_bn_tensors(f"head.cls_convs.{scale_idx}.0", head_ch, head_ch, 3))
        t.update(_conv_bn_tensors(f"head.cls_convs.{scale_idx}.1", head_ch, head_ch, 3))
        # Reg branch
        t.update(_conv_bn_tensors(f"head.reg_convs.{scale_idx}.0", head_ch, head_ch, 3))
        t.update(_conv_bn_tensors(f"head.reg_convs.{scale_idx}.1", head_ch, head_ch, 3))
        # Prediction heads (conv only)
        t.update(_conv_only_tensors(f"head.cls_preds.{scale_idx}", head_ch, num_classes))
        t.update(_conv_only_tensors(f"head.reg_preds.{scale_idx}", head_ch, 4))
        t.update(_conv_only_tensors(f"head.obj_preds.{scale_idx}", head_ch, 1))

    return t


def _make_config(num_classes: int = 4, depth_mul: float = 0.33,
                 width_mul: float = 0.25) -> dict:
    return {
        "model_type": "yolox",
        "num_labels": num_classes,
        "depth_multiplier": depth_mul,
        "width_multiplier": width_mul,
    }


# ===========================================================================
# Plugin matching
# ===========================================================================

class TestYoloxPluginMatch:
    """Verify YOLOX plugin matches the expected model types."""

    def test_matches_yolox(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert plugin.matches("yolox")

    def test_matches_yolox_document(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert plugin.matches("yolox_document")

    def test_matches_yolox_doc(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert plugin.matches("yolox-doc")

    def test_no_match_yolo(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert not plugin.matches("yolo")

    def test_no_match_detr(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert not plugin.matches("detr")

    def test_name(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert plugin.name == "yolox"

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.yolox import plugin
        assert plugin.runtime_strategy == "yolox_object_detection"


# ===========================================================================
# Weight loading
# ===========================================================================

class TestYoloxLoadWeights:
    """Verify weight key mapping from HF keys to engine keys."""

    NUM_CLASSES = 4

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.yolox import plugin

        config = _make_config(num_classes=self.NUM_CLASSES)
        tensors = _make_yolox_tensors(num_classes=self.NUM_CLASSES)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Backbone stem
        assert "backbone.stem.conv1.conv.weight" in weights
        assert "backbone.stem.conv1.bn.weight" in weights

        # Dark stages
        for stage in ("dark2", "dark3", "dark4", "dark5"):
            assert f"backbone.{stage}.downsample.conv.weight" in weights
            assert f"backbone.{stage}.csp.main_conv.conv.weight" in weights
            assert f"backbone.{stage}.csp.short_conv.conv.weight" in weights
            assert f"backbone.{stage}.csp.final_conv.conv.weight" in weights

        # FPN
        assert "fpn.lateral_conv0.conv.weight" in weights
        assert "fpn.reduce_conv1.conv.weight" in weights
        assert "fpn.c3_p3.main_conv.conv.weight" in weights
        assert "fpn.c3_p4.main_conv.conv.weight" in weights
        assert "fpn.c3_n3.main_conv.conv.weight" in weights
        assert "fpn.c3_n4.main_conv.conv.weight" in weights
        assert "fpn.bu_conv2.conv.weight" in weights
        assert "fpn.bu_conv1.conv.weight" in weights

        # Detection head (3 scales)
        for i in range(3):
            sp = f"head.heads.{i}"
            assert f"{sp}.stem.conv.weight" in weights
            assert f"{sp}.cls_convs.0.conv.weight" in weights
            assert f"{sp}.cls_convs.1.conv.weight" in weights
            assert f"{sp}.reg_convs.0.conv.weight" in weights
            assert f"{sp}.reg_convs.1.conv.weight" in weights
            assert f"{sp}.cls_pred.weight" in weights
            assert f"{sp}.cls_pred.bias" in weights
            assert f"{sp}.reg_pred.weight" in weights
            assert f"{sp}.obj_pred.weight" in weights

    def test_load_weights_shapes(self, tmp_path):
        """Verify weight tensor shapes are preserved correctly."""
        from tensorrt_model_connect.families.yolox import plugin

        config = _make_config(num_classes=self.NUM_CLASSES)
        tensors = _make_yolox_tensors(num_classes=self.NUM_CLASSES)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Stem conv: 3 -> stem_ch (16), 6x6 kernel
        assert weights["backbone.stem.conv1.conv.weight"].shape[1] == 3
        assert weights["backbone.stem.conv1.conv.weight"].shape[2] == 6

        # Cls prediction: head_ch -> num_classes, 1x1
        assert weights["head.heads.0.cls_pred.weight"].shape[0] == self.NUM_CLASSES
        assert weights["head.heads.0.cls_pred.bias"].shape[0] == self.NUM_CLASSES

        # Reg prediction: head_ch -> 4, 1x1
        assert weights["head.heads.0.reg_pred.weight"].shape[0] == 4

        # Obj prediction: head_ch -> 1, 1x1
        assert weights["head.heads.0.obj_pred.weight"].shape[0] == 1

    def test_bn_tensors_present(self, tmp_path):
        """Verify BN running_mean/running_var are loaded (needed for folding)."""
        from tensorrt_model_connect.families.yolox import plugin

        config = _make_config(num_classes=self.NUM_CLASSES)
        tensors = _make_yolox_tensors(num_classes=self.NUM_CLASSES)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Check BN stats for stem
        assert "backbone.stem.conv1.bn.running_mean" in weights
        assert "backbone.stem.conv1.bn.running_var" in weights

        # Check BN stats for a head stem
        assert "head.heads.0.stem.bn.running_mean" in weights
        assert "head.heads.0.stem.bn.running_var" in weights


# ===========================================================================
# Detection config
# ===========================================================================

class TestYoloxDetectionConfig:
    """Verify get_detection_config() returns correct values."""

    def test_default_config(self, tmp_path):
        from tensorrt_model_connect.families.yolox import plugin

        config = _make_config(num_classes=10)
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)

        # Set _resolved_image_size as load_weights would
        cfg.raw["_resolved_image_size"] = 640

        det_cfg = plugin.get_detection_config(cfg)
        assert det_cfg["det_num_classes"] == 10
        assert det_cfg["det_input_h"] == 640
        assert det_cfg["det_input_w"] == 640
        assert det_cfg["det_conf_threshold"] == 0.5
        assert det_cfg["det_nms_threshold"] == 0.45
        assert len(det_cfg["image_mean"]) == 3
        assert len(det_cfg["image_std"]) == 3

    def test_custom_image_size(self, tmp_path):
        from tensorrt_model_connect.families.yolox import plugin

        config = _make_config(num_classes=4)
        _write_config(tmp_path, config)
        cfg = ModelConfig.from_dir(tmp_path)
        cfg.raw["_resolved_image_size"] = 1024

        det_cfg = plugin.get_detection_config(cfg)
        assert det_cfg["det_input_h"] == 1024
        assert det_cfg["det_input_w"] == 1024


# ===========================================================================
# Image size resolution
# ===========================================================================

class TestResolveImageSize:
    """Verify _resolve_image_size reads preprocessor_config.json."""

    def test_from_preprocessor_config(self, tmp_path):
        from tensorrt_model_connect.families.yolox import _resolve_image_size

        pp = {"size": {"height": 1024, "width": 1024}}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(pp))
        assert _resolve_image_size(str(tmp_path)) == 1024

    def test_default_when_missing(self, tmp_path):
        from tensorrt_model_connect.families.yolox import _resolve_image_size

        assert _resolve_image_size(str(tmp_path)) == 640

    def test_int_size_format(self, tmp_path):
        from tensorrt_model_connect.families.yolox import _resolve_image_size

        pp = {"size": 512}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(pp))
        assert _resolve_image_size(str(tmp_path)) == 512


# ===========================================================================
# CNN builder BatchNorm folding
# ===========================================================================

class TestBatchNormFolding:
    """Verify BN folding produces correct fused weights."""

    def test_fuse_conv_bn(self):
        from tensorrt_model_connect.cnn_builder import _fuse_conv_bn

        out_ch, in_ch, kh, kw = 8, 3, 3, 3
        conv_w = _rand(out_ch, in_ch, kh, kw)
        conv_b = np.zeros(out_ch, dtype=np.float32)
        bn_gamma = np.ones(out_ch, dtype=np.float32)
        bn_beta = np.zeros(out_ch, dtype=np.float32)
        bn_mean = np.zeros(out_ch, dtype=np.float32)
        bn_var = np.ones(out_ch, dtype=np.float32)

        fused_w, fused_b = _fuse_conv_bn(
            conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var)

        # With identity BN (gamma=1, beta=0, mean=0, var=1), output == input
        np.testing.assert_allclose(fused_w, conv_w, atol=1e-4)
        np.testing.assert_allclose(fused_b, conv_b, atol=1e-4)

    def test_fuse_conv_bn_nontrivial(self):
        from tensorrt_model_connect.cnn_builder import _fuse_conv_bn

        out_ch, in_ch = 4, 3
        conv_w = np.ones((out_ch, in_ch, 1, 1), dtype=np.float32)
        conv_b = np.zeros(out_ch, dtype=np.float32)
        bn_gamma = 2.0 * np.ones(out_ch, dtype=np.float32)
        bn_beta = np.ones(out_ch, dtype=np.float32)
        bn_mean = np.zeros(out_ch, dtype=np.float32)
        bn_var = np.ones(out_ch, dtype=np.float32)

        fused_w, fused_b = _fuse_conv_bn(
            conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var)

        # BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
        # With mean=0, var=1, eps~0: y = gamma * x + beta
        # So fused_w = gamma * conv_w, fused_b = gamma * conv_b + beta
        np.testing.assert_allclose(fused_w, 2.0 * conv_w, atol=1e-5)
        np.testing.assert_allclose(fused_b, np.ones(out_ch), atol=1e-5)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for SegFormer (semantic segmentation).

Intention:
    Validate the SegFormer family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness.

    SegFormer is a hierarchical encoder-decoder architecture for semantic
    segmentation with 4 encoder stages and a lightweight All-MLP decode head.
    Each encoder stage has overlapping patch embeddings, transformer blocks
    with efficient self-attention (sequence reduction), and Mix-FFN with
    depthwise convolution.

    SegFormer does NOT use standard decoder keys (no embedding, no w_out,
    no KV cache, no position IDs). Instead, it uses pixel_values as input
    and produces segmentation logits [1, num_classes, H/4, W/4].

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    ALL of: spec, get_config_dict(), make_hf_tensors(), expected_weight_keys().
    Tier 2 is skipped because SegFormer uses a fully custom graph builder with
    4-stage hierarchical encoder + decode head (no standard decoder).

Trace: ARCH-FAM-001, UD-FAM-SEGFORMER-01
Intent: Validate the SegFormer family plugin weight loading for hierarchical 4-stage encoder with overlapping patch embeddings, efficient self-attention, Mix-FFN, and All-MLP decode head.
Preconditions: safetensors and tensorrt_model_connect are importable; no TRT or GPU required for weight-loading tests.
Postconditions: All SegFormer weight keys (patch embed, attention, Mix-FFN, decode head) are present with correct shapes for all 4 encoder stages.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
except (ImportError, ModuleNotFoundError):
    pytest.skip("safetensors not available", allow_module_level=True)

try:
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tensorrt_model_connect.models.segformer.tests._family_plugin_tester import (
    FamilyPluginTester,
    TinyModelSpec,
)
from tensorrt_model_connect.models.segformer.tests._family_plugin_test_mixin import (
    FamilyPluginTestMixin,
)


# SegFormer B0-like dimensions, kept tiny for fast tests.
_DEPTHS = [2, 2, 2, 2]
_HIDDEN_SIZES = [8, 16, 20, 32]
_NUM_HEADS = [1, 2, 5, 8]
_SR_RATIOS = [8, 4, 2, 1]
_MLP_RATIOS = [4, 4, 4, 4]
_PATCH_SIZES = [7, 3, 3, 3]
_STRIDES = [4, 2, 2, 2]
_NUM_CLASSES = 10
_DECODER_HIDDEN = 32
_IMAGE_SIZE = 64


class SegformerPluginTester(FamilyPluginTester):
    """Tester for the SegFormer family plugin.

    SegFormer uses:
      - 4-stage hierarchical encoder with overlapping patch embeddings
      - Efficient Self-Attention with Sequence Reduction (SR Conv2d)
      - Mix-FFN: FC1 -> DWConv3x3 -> GELU -> FC2
      - All-MLP decode head: per-stage linear projection, bilinear upsample,
        concat, fuse conv (1x1) -> BN -> ReLU -> classifier conv (1x1)
      - LayerNorm with bias everywhere
      - No positional encoding (overlapping patches provide position info)
    """

    plugin_module = "tensorrt_model_connect.models.segformer.model"
    model_type = "segformer"
    spec = TinyModelSpec(
        vocab_size=_NUM_CLASSES,
        hidden_size=_HIDDEN_SIZES[0],
        num_hidden_layers=sum(_DEPTHS),
        num_attention_heads=_NUM_HEADS[0],
        num_key_value_heads=_NUM_HEADS[0],
    )

    def get_config_dict(self) -> dict:
        """SegFormer config with hierarchical encoder + decode head fields."""
        return {
            "model_type": "segformer",
            "depths": _DEPTHS,
            "hidden_sizes": _HIDDEN_SIZES,
            "num_attention_heads": _NUM_HEADS,
            "sr_ratios": _SR_RATIOS,
            "mlp_ratios": _MLP_RATIOS,
            "patch_sizes": _PATCH_SIZES,
            "strides": _STRIDES,
            "num_labels": _NUM_CLASSES,
            "decoder_hidden_size": _DECODER_HIDDEN,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu",
        }

    def write_model_dir(self, tmp_path):
        """Write config.json, model.safetensors, and preprocessor_config.json."""
        path = super().write_model_dir(tmp_path)
        # SegFormer reads image size from preprocessor_config.json
        pp_config = {
            "size": {"height": _IMAGE_SIZE, "width": _IMAGE_SIZE},
        }
        (path / "preprocessor_config.json").write_text(json.dumps(pp_config))
        return path

    def prepare_config_and_weights(self, tmp_path):
        """Override to also write preprocessor_config.json."""
        raw_tensors = self.make_hf_tensors()
        config_dict = self.get_config_dict()
        (tmp_path / "config.json").write_text(json.dumps(config_dict))
        save_file(raw_tensors, str(tmp_path / "model.safetensors"))
        # Write preprocessor_config.json for _resolve_image_size
        pp_config = {
            "size": {"height": _IMAGE_SIZE, "width": _IMAGE_SIZE},
        }
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps(pp_config))

        config = ModelConfig.from_dir(tmp_path)
        plugin = self.get_plugin()
        weights = plugin.load_weights(str(tmp_path), config)
        return config, weights, raw_tensors

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching SegFormer's weight layout.

        Encoder per stage:
          - segformer.encoder.patch_embeddings.{s}.proj.weight [hidden, in_ch, ps, ps]
          - segformer.encoder.patch_embeddings.{s}.proj.bias [hidden]
          - segformer.encoder.patch_embeddings.{s}.layer_norm.weight/bias [hidden]
          - Per block:
            - segformer.encoder.block.{s}.{b}.layer_norm_1/2.weight/bias [hidden]
            - segformer.encoder.block.{s}.{b}.attention.self.query/key/value.weight [h, h]
            - segformer.encoder.block.{s}.{b}.attention.self.query/key/value.bias [h]
            - segformer.encoder.block.{s}.{b}.attention.output.dense.weight/bias [h, h]
            - segformer.encoder.block.{s}.{b}.attention.self.sr.weight [h, h, sr, sr]
              (only if sr > 1)
            - segformer.encoder.block.{s}.{b}.attention.self.sr.bias [h]
            - segformer.encoder.block.{s}.{b}.attention.self.layer_norm.weight/bias [h]
            - segformer.encoder.block.{s}.{b}.mlp.dense1.weight [ffn, h]
            - segformer.encoder.block.{s}.{b}.mlp.dense1.bias [ffn]
            - segformer.encoder.block.{s}.{b}.mlp.dense2.weight [h, ffn]
            - segformer.encoder.block.{s}.{b}.mlp.dense2.bias [h]
            - segformer.encoder.block.{s}.{b}.mlp.dwconv.dwconv.weight [ffn, 1, 3, 3]
            - segformer.encoder.block.{s}.{b}.mlp.dwconv.dwconv.bias [ffn]
          - segformer.encoder.layer_norm.{s}.weight/bias [hidden]

        Decode head:
          - decode_head.linear_c.{i}.proj.weight [decoder_hidden, stage_hidden]
          - decode_head.linear_c.{i}.proj.bias [decoder_hidden]
          - decode_head.linear_fuse.weight [decoder_hidden, 4*decoder_hidden, 1, 1]
          - decode_head.batch_norm.weight/bias/running_mean/running_var [decoder_hidden]
          - decode_head.classifier.weight [num_classes, decoder_hidden, 1, 1]
          - decode_head.classifier.bias [num_classes]
        """
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}

        for stage_idx in range(4):
            hidden = _HIDDEN_SIZES[stage_idx]
            n_blocks = _DEPTHS[stage_idx]
            sr = _SR_RATIOS[stage_idx]
            mlp_ratio = _MLP_RATIOS[stage_idx]
            ffn_hidden = hidden * mlp_ratio
            patch_size = _PATCH_SIZES[stage_idx]
            in_ch = 3 if stage_idx == 0 else _HIDDEN_SIZES[stage_idx - 1]

            # Overlap patch embedding
            pe = f"segformer.encoder.patch_embeddings.{stage_idx}"
            t[f"{pe}.proj.weight"] = rand(hidden, in_ch, patch_size, patch_size)
            t[f"{pe}.proj.bias"] = rand(hidden)
            t[f"{pe}.layer_norm.weight"] = rand(hidden)
            t[f"{pe}.layer_norm.bias"] = rand(hidden)

            for block_idx in range(n_blocks):
                bp = f"segformer.encoder.block.{stage_idx}.{block_idx}"

                # Layer norms
                t[f"{bp}.layer_norm_1.weight"] = rand(hidden)
                t[f"{bp}.layer_norm_1.bias"] = rand(hidden)
                t[f"{bp}.layer_norm_2.weight"] = rand(hidden)
                t[f"{bp}.layer_norm_2.bias"] = rand(hidden)

                # Attention Q/K/V
                for proj in ("query", "key", "value"):
                    t[f"{bp}.attention.self.{proj}.weight"] = rand(
                        hidden, hidden)
                    t[f"{bp}.attention.self.{proj}.bias"] = rand(hidden)
                # Output dense
                t[f"{bp}.attention.output.dense.weight"] = rand(hidden, hidden)
                t[f"{bp}.attention.output.dense.bias"] = rand(hidden)

                # SR (sequence reduction) if sr > 1
                if sr > 1:
                    t[f"{bp}.attention.self.sr.weight"] = rand(
                        hidden, hidden, sr, sr)
                    t[f"{bp}.attention.self.sr.bias"] = rand(hidden)
                    t[f"{bp}.attention.self.layer_norm.weight"] = rand(hidden)
                    t[f"{bp}.attention.self.layer_norm.bias"] = rand(hidden)

                # Mix-FFN
                t[f"{bp}.mlp.dense1.weight"] = rand(ffn_hidden, hidden)
                t[f"{bp}.mlp.dense1.bias"] = rand(ffn_hidden)
                t[f"{bp}.mlp.dense2.weight"] = rand(hidden, ffn_hidden)
                t[f"{bp}.mlp.dense2.bias"] = rand(hidden)
                # DWConv (depthwise)
                t[f"{bp}.mlp.dwconv.dwconv.weight"] = rand(ffn_hidden, 1, 3, 3)
                t[f"{bp}.mlp.dwconv.dwconv.bias"] = rand(ffn_hidden)

            # Per-stage final LayerNorm
            t[f"segformer.encoder.layer_norm.{stage_idx}.weight"] = rand(hidden)
            t[f"segformer.encoder.layer_norm.{stage_idx}.bias"] = rand(hidden)

        # --- Decode head ---
        for i in range(4):
            t[f"decode_head.linear_c.{i}.proj.weight"] = rand(
                _DECODER_HIDDEN, _HIDDEN_SIZES[i])
            t[f"decode_head.linear_c.{i}.proj.bias"] = rand(_DECODER_HIDDEN)

        # Fuse conv (1x1): [decoder_hidden, 4*decoder_hidden, 1, 1]
        t["decode_head.linear_fuse.weight"] = rand(
            _DECODER_HIDDEN, 4 * _DECODER_HIDDEN, 1, 1)

        # BatchNorm
        t["decode_head.batch_norm.weight"] = rand(_DECODER_HIDDEN)
        t["decode_head.batch_norm.bias"] = rand(_DECODER_HIDDEN)
        t["decode_head.batch_norm.running_mean"] = rand(_DECODER_HIDDEN)
        # running_var must be positive for BN
        t["decode_head.batch_norm.running_var"] = np.abs(
            rand(_DECODER_HIDDEN)) + 0.1

        # Classifier conv (1x1)
        t["decode_head.classifier.weight"] = rand(
            _NUM_CLASSES, _DECODER_HIDDEN, 1, 1)
        t["decode_head.classifier.bias"] = rand(_NUM_CLASSES)

        return t

    def expected_weight_keys(self) -> set[str]:
        """SegFormer weight keys: 4-stage encoder + decode head.

        Per stage:
          stage{s}.patch_embed.proj.weight/bias
          stage{s}.patch_embed.norm.weight/bias
          stage{s}.block{b}.norm1/2.weight/bias
          stage{s}.block{b}.attn.q/k/v/o.weight/bias
          stage{s}.block{b}.attn.sr.weight/bias (if sr > 1)
          stage{s}.block{b}.attn.sr_norm.weight/bias (if sr > 1)
          stage{s}.block{b}.mlp.fc1/2.weight/bias
          stage{s}.block{b}.mlp.dwconv.weight/bias
          stage{s}.final_norm.weight/bias

        Decode head:
          decode_head.linear_c{i}.weight/bias
          decode_head.fuse.weight/bias
          decode_head.bn.weight/bias/running_mean/running_var
          decode_head.classifier.weight/bias
        """
        keys: set[str] = set()

        for stage_idx in range(4):
            n_blocks = _DEPTHS[stage_idx]
            sr = _SR_RATIOS[stage_idx]

            keys.update({
                f"stage{stage_idx}.patch_embed.proj.weight",
                f"stage{stage_idx}.patch_embed.proj.bias",
                f"stage{stage_idx}.patch_embed.norm.weight",
                f"stage{stage_idx}.patch_embed.norm.bias",
            })

            for block_idx in range(n_blocks):
                wp = f"stage{stage_idx}.block{block_idx}"
                keys.update({
                    f"{wp}.norm1.weight", f"{wp}.norm1.bias",
                    f"{wp}.norm2.weight", f"{wp}.norm2.bias",
                    f"{wp}.attn.q.weight", f"{wp}.attn.q.bias",
                    f"{wp}.attn.k.weight", f"{wp}.attn.k.bias",
                    f"{wp}.attn.v.weight", f"{wp}.attn.v.bias",
                    f"{wp}.attn.o.weight", f"{wp}.attn.o.bias",
                    f"{wp}.mlp.fc1.weight", f"{wp}.mlp.fc1.bias",
                    f"{wp}.mlp.fc2.weight", f"{wp}.mlp.fc2.bias",
                    f"{wp}.mlp.dwconv.weight", f"{wp}.mlp.dwconv.bias",
                })
                if sr > 1:
                    keys.update({
                        f"{wp}.attn.sr.weight", f"{wp}.attn.sr.bias",
                        f"{wp}.attn.sr_norm.weight", f"{wp}.attn.sr_norm.bias",
                    })

            keys.update({
                f"stage{stage_idx}.final_norm.weight",
                f"stage{stage_idx}.final_norm.bias",
            })

        # Decode head
        for i in range(4):
            keys.update({
                f"decode_head.linear_c{i}.weight",
                f"decode_head.linear_c{i}.bias",
            })
        keys.update({
            "decode_head.fuse.weight",
            "decode_head.fuse.bias",
            "decode_head.bn.weight",
            "decode_head.bn.bias",
            "decode_head.bn.running_mean",
            "decode_head.bn.running_var",
            "decode_head.classifier.weight",
            "decode_head.classifier.bias",
        })

        return keys


class TestSegformerEngine(FamilyPluginTestMixin):
    """Engine tests for SegFormer family plugin.

    Tier 0 and Tier 1 tests run via the mixin. Tier 2 (engine build) is
    skipped because SegFormer uses a fully custom graph builder with a
    4-stage hierarchical encoder + decode head.
    """

    tester_class = SegformerPluginTester

    # --- Tier 2 skips ---
    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_build_engine_succeeds(self, tester, tmp_path):
        pass

    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_engine_io_tensor_names(self, tester, tmp_path):
        pass

    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_engine_logits_output_shape(self, tester, tmp_path):
        pass

    # --- Override mixin tests that assume standard decoder layout ---

    def test_load_weights_embedding_shape(self, tester, tmp_path):
        """Override: SegFormer has no token embedding.

        Verify the first patch embedding conv weight exists instead.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        key = "stage0.patch_embed.proj.weight"
        assert key in weights, f"Missing key: {key}"
        # Shape: [hidden, in_ch, patch_size, patch_size]
        assert weights[key].shape[0] == _HIDDEN_SIZES[0], (
            f"stage0 patch embed output channels {weights[key].shape[0]} != "
            f"expected {_HIDDEN_SIZES[0]}"
        )

    def test_load_weights_projections_transposed(self, tester, tmp_path):
        """Override: SegFormer uses stage/block-prefixed attention projections.

        Verify stage0.block0.attn.q.weight has shape[0] == hidden
        (transposed from [out, in] to [in, out]).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        key = "stage0.block0.attn.q.weight"
        assert key in weights, f"Missing key: {key}"
        h = _HIDDEN_SIZES[0]
        assert weights[key].shape[0] == h, (
            f"{key} shape[0] = {weights[key].shape[0]}, expected {h} "
            f"(projection should be transposed from HF [out, in] to [in, out])"
        )

    # --- SegFormer-specific Tier 1 tests ---

    def test_sr_weights_present_for_high_sr_stages(self, tester, tmp_path):
        """Validate that sequence reduction (SR) weights are loaded for stages with sr > 1.

        Intention:
            SegFormer stages with sr_ratio > 1 use a Conv2d to reduce the
            spatial resolution of K/V before attention. If the SR weights are
            missing, the model will fall back to full-sequence attention,
            which is both slower and numerically different.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each stage with sr > 1, verify attn.sr.weight/bias and
               attn.sr_norm.weight/bias exist.
            3. For stage with sr == 1, verify SR weights are absent.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for stage_idx in range(4):
            sr = _SR_RATIOS[stage_idx]
            for block_idx in range(_DEPTHS[stage_idx]):
                wp = f"stage{stage_idx}.block{block_idx}"
                if sr > 1:
                    assert f"{wp}.attn.sr.weight" in weights, (
                        f"Missing SR weight for {wp} (sr={sr})"
                    )
                    assert f"{wp}.attn.sr_norm.weight" in weights, (
                        f"Missing SR norm weight for {wp} (sr={sr})"
                    )
                else:
                    assert f"{wp}.attn.sr.weight" not in weights, (
                        f"Unexpected SR weight for {wp} (sr=1)"
                    )

    def test_decode_head_bn_weights(self, tester, tmp_path):
        """Validate that decode head BatchNorm weights are loaded.

        Intention:
            The decode head uses BatchNorm after the fuse convolution.
            Four tensors are required: weight, bias, running_mean, running_var.
            Missing any of these causes incorrect normalization.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify all four BN keys exist.
            3. Verify shapes are [decoder_hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            key = f"decode_head.bn.{suffix}"
            assert key in weights, f"Missing BN key: {key}"
            assert weights[key].shape == (_DECODER_HIDDEN,), (
                f"{key} shape {weights[key].shape} != "
                f"expected ({_DECODER_HIDDEN},)"
            )

    def test_dwconv_weights_per_block(self, tester, tmp_path):
        """Validate that depthwise convolution weights exist for each block.

        Intention:
            SegFormer's Mix-FFN uses a 3x3 depthwise convolution between
            FC1 and GELU. Each block has its own DWConv weights. If missing,
            the Mix-FFN is incomplete.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each stage/block, verify mlp.dwconv.weight exists.
            3. Verify shape [ffn_hidden, 1, 3, 3] (depthwise).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for stage_idx in range(4):
            hidden = _HIDDEN_SIZES[stage_idx]
            ffn_hidden = hidden * _MLP_RATIOS[stage_idx]
            for block_idx in range(_DEPTHS[stage_idx]):
                key = f"stage{stage_idx}.block{block_idx}.mlp.dwconv.weight"
                assert key in weights, f"Missing DWConv key: {key}"
                assert weights[key].shape == (ffn_hidden, 1, 3, 3), (
                    f"{key} shape {weights[key].shape} != "
                    f"expected ({ffn_hidden}, 1, 3, 3)"
                )

    def test_classifier_output_channels(self, tester, tmp_path):
        """Validate that the classifier conv has num_classes output channels.

        Intention:
            The final classifier conv (1x1) maps from decoder_hidden to
            num_classes. If the output channels are wrong, the segmentation
            map will have the wrong number of classes.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify classifier.weight shape[0] == num_classes.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        key = "decode_head.classifier.weight"
        assert key in weights, f"Missing key: {key}"
        assert weights[key].shape[0] == _NUM_CLASSES, (
            f"Classifier output channels {weights[key].shape[0]} != "
            f"expected {_NUM_CLASSES}"
        )

"""Tests for SAM family plugin -- weight loading and config parsing.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-SAM
Intent: Validate SAM prompted segmentation family plugin weight loading and encoder/decoder key mapping
Preconditions: Synthetic safetensors with SAM vision encoder and mask decoder weight naming are available
Postconditions: Plugin produces correct weight keys for image encoder, prompt encoder, and mask decoder
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


RNG = np.random.RandomState(99)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# SAM test dimensions (small for fast tests)
HIDDEN = 32
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS
MLP_DIM = 64
DECODER_HIDDEN = 16
DECODER_DEPTH = 2
NUM_MULTIMASK = 3
PATCH_SIZE = 16
IMAGE_SIZE = 64  # small for test
IMAGE_EMBED_SIZE = IMAGE_SIZE // PATCH_SIZE  # = 4


def _make_sam_config():
    return {
        "model_type": "sam",
        "vision_config": {
            "hidden_size": HIDDEN,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "mlp_dim": MLP_DIM,
            "window_size": 2,
            "global_attn_indexes": [1],
        },
        "prompt_encoder_config": {
            "hidden_size": DECODER_HIDDEN,
            "image_embedding_size": IMAGE_EMBED_SIZE,
            "mask_input_channels": 4,
        },
        "mask_decoder_config": {
            "hidden_size": DECODER_HIDDEN,
            "num_multimask_outputs": NUM_MULTIMASK,
            "num_attention_heads": 4,
            "depth": DECODER_DEPTH,
            "mlp_dim": DECODER_HIDDEN * 8,
        },
    }


def _make_sam_tensors():
    """Create synthetic SAM weight tensors matching HF naming expected by sam.py plugin."""
    t = {}

    # Vision encoder: patch embed
    t["vision_encoder.patch_embed.projection.weight"] = _rand(HIDDEN, 3, PATCH_SIZE, PATCH_SIZE)
    t["vision_encoder.patch_embed.projection.bias"] = _rand(HIDDEN)

    # Positional embedding (plugin reads vision_encoder.pos_embed)
    num_patches_h = IMAGE_SIZE // PATCH_SIZE
    t["vision_encoder.pos_embed"] = _rand(1, num_patches_h, num_patches_h, HIDDEN)

    # Shared image embedding for prompt encoder
    num_patches = num_patches_h ** 2
    t["shared_image_embedding.positional_embedding"] = _rand(num_patches, HIDDEN)

    # Vision encoder layers
    for i in range(NUM_LAYERS):
        p = f"vision_encoder.layers.{i}"
        t[f"{p}.layer_norm1.weight"] = _rand(HIDDEN)
        t[f"{p}.layer_norm1.bias"] = _rand(HIDDEN)
        t[f"{p}.layer_norm2.weight"] = _rand(HIDDEN)
        t[f"{p}.layer_norm2.bias"] = _rand(HIDDEN)
        # Fused QKV
        t[f"{p}.attn.qkv.weight"] = _rand(HIDDEN * 3, HIDDEN)
        t[f"{p}.attn.qkv.bias"] = _rand(HIDDEN * 3)
        t[f"{p}.attn.proj.weight"] = _rand(HIDDEN, HIDDEN)
        t[f"{p}.attn.proj.bias"] = _rand(HIDDEN)
        # MLP
        t[f"{p}.mlp.lin1.weight"] = _rand(MLP_DIM, HIDDEN)
        t[f"{p}.mlp.lin1.bias"] = _rand(MLP_DIM)
        t[f"{p}.mlp.lin2.weight"] = _rand(HIDDEN, MLP_DIM)
        t[f"{p}.mlp.lin2.bias"] = _rand(HIDDEN)

    # Neck: plugin reads vision_encoder.neck.conv1/conv2 and layer_norm1/layer_norm2
    t["vision_encoder.neck.conv1.weight"] = _rand(DECODER_HIDDEN, HIDDEN, 1, 1)
    t["vision_encoder.neck.conv1.bias"] = _rand(DECODER_HIDDEN)
    t["vision_encoder.neck.layer_norm1.weight"] = _rand(DECODER_HIDDEN)
    t["vision_encoder.neck.layer_norm1.bias"] = _rand(DECODER_HIDDEN)
    t["vision_encoder.neck.conv2.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN, 3, 3)
    t["vision_encoder.neck.conv2.bias"] = _rand(DECODER_HIDDEN)
    t["vision_encoder.neck.layer_norm2.weight"] = _rand(DECODER_HIDDEN)
    t["vision_encoder.neck.layer_norm2.bias"] = _rand(DECODER_HIDDEN)

    # Prompt encoder: plugin reads prompt_encoder.point_embed.{i}.weight
    for i in range(4):
        t[f"prompt_encoder.point_embed.{i}.weight"] = _rand(1, DECODER_HIDDEN)
    t["prompt_encoder.not_a_point_embed.weight"] = _rand(1, DECODER_HIDDEN)
    t["prompt_encoder.no_mask_embed.weight"] = _rand(1, DECODER_HIDDEN)

    # Mask decoder
    num_masks = NUM_MULTIMASK + 1
    t["mask_decoder.iou_token.weight"] = _rand(1, DECODER_HIDDEN)
    t["mask_decoder.mask_tokens.weight"] = _rand(num_masks, DECODER_HIDDEN)

    # Two-way transformer layers
    for i in range(DECODER_DEPTH):
        p = f"mask_decoder.transformer.layers.{i}"
        # Self-attention
        t[f"{p}.self_attn.q_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.self_attn.q_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.self_attn.k_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.self_attn.k_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.self_attn.v_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.self_attn.v_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.self_attn.out_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.self_attn.out_proj.bias"] = _rand(DECODER_HIDDEN)
        # Cross-attention (token to image)
        t[f"{p}.cross_attn_token_to_image.q_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.q_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.k_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.k_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.v_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.v_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.out_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_token_to_image.out_proj.bias"] = _rand(DECODER_HIDDEN)
        # Cross-attention (image to token)
        t[f"{p}.cross_attn_image_to_token.q_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.q_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.k_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.k_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.v_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.v_proj.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.out_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
        t[f"{p}.cross_attn_image_to_token.out_proj.bias"] = _rand(DECODER_HIDDEN)
        # LayerNorms (plugin reads layer_norm1..4)
        t[f"{p}.layer_norm1.weight"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm1.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm2.weight"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm2.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm3.weight"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm3.bias"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm4.weight"] = _rand(DECODER_HIDDEN)
        t[f"{p}.layer_norm4.bias"] = _rand(DECODER_HIDDEN)
        # MLP
        t[f"{p}.mlp.lin1.weight"] = _rand(DECODER_HIDDEN * 8, DECODER_HIDDEN)
        t[f"{p}.mlp.lin1.bias"] = _rand(DECODER_HIDDEN * 8)
        t[f"{p}.mlp.lin2.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN * 8)
        t[f"{p}.mlp.lin2.bias"] = _rand(DECODER_HIDDEN)

    # Final attention
    p = "mask_decoder.transformer.final_attn_token_to_image"
    t[f"{p}.q_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
    t[f"{p}.q_proj.bias"] = _rand(DECODER_HIDDEN)
    t[f"{p}.k_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
    t[f"{p}.k_proj.bias"] = _rand(DECODER_HIDDEN)
    t[f"{p}.v_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
    t[f"{p}.v_proj.bias"] = _rand(DECODER_HIDDEN)
    t[f"{p}.out_proj.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN)
    t[f"{p}.out_proj.bias"] = _rand(DECODER_HIDDEN)

    # Final LayerNorm
    t["mask_decoder.transformer.layer_norm_final_attn.weight"] = _rand(DECODER_HIDDEN)
    t["mask_decoder.transformer.layer_norm_final_attn.bias"] = _rand(DECODER_HIDDEN)

    # Output upscaling: plugin reads upscale_conv1, upscale_layer_norm, upscale_conv2
    t["mask_decoder.upscale_conv1.weight"] = _rand(DECODER_HIDDEN, DECODER_HIDDEN // 4, 2, 2)
    t["mask_decoder.upscale_conv1.bias"] = _rand(DECODER_HIDDEN // 4)
    t["mask_decoder.upscale_layer_norm.weight"] = _rand(DECODER_HIDDEN // 4)
    t["mask_decoder.upscale_layer_norm.bias"] = _rand(DECODER_HIDDEN // 4)
    t["mask_decoder.upscale_conv2.weight"] = _rand(DECODER_HIDDEN // 4, DECODER_HIDDEN // 8, 2, 2)
    t["mask_decoder.upscale_conv2.bias"] = _rand(DECODER_HIDDEN // 8)

    # Hypernetwork MLPs: plugin reads {proj_in, layers.0, proj_out}
    _hyper_layer_map = {0: "proj_in", 1: "layers.0", 2: "proj_out"}
    for i in range(num_masks):
        for j in range(3):
            in_dim = DECODER_HIDDEN if j == 0 else DECODER_HIDDEN
            out_dim = DECODER_HIDDEN if j < 2 else DECODER_HIDDEN // 8
            suffix = _hyper_layer_map[j]
            t[f"mask_decoder.output_hypernetworks_mlps.{i}.{suffix}.weight"] = _rand(out_dim, in_dim)
            t[f"mask_decoder.output_hypernetworks_mlps.{i}.{suffix}.bias"] = _rand(out_dim)

    # IoU prediction head: plugin reads {proj_in, layers.0, proj_out}
    _iou_layer_map = {0: "proj_in", 1: "layers.0", 2: "proj_out"}
    for j in range(3):
        in_dim = DECODER_HIDDEN if j == 0 else DECODER_HIDDEN
        out_dim = DECODER_HIDDEN if j < 2 else num_masks
        suffix = _iou_layer_map[j]
        t[f"mask_decoder.iou_prediction_head.{suffix}.weight"] = _rand(out_dim, in_dim)
        t[f"mask_decoder.iou_prediction_head.{suffix}.bias"] = _rand(out_dim)

    return t


class TestSamPlugin:
    """SAM plugin load_weights and matches tests."""

    def test_matches(self):
        from tensorrt_model_connect.families.sam import plugin
        assert plugin.matches("sam")
        assert not plugin.matches("qwen3")
        assert not plugin.matches("bert")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.sam import plugin
        assert plugin.runtime_strategy == "sam_prompted_segmentation"

    def test_load_weights_has_encoder_keys(self, tmp_path):
        from tensorrt_model_connect.families.sam import plugin

        config_dict = _make_sam_config()
        _write_config(tmp_path, config_dict)
        _write_safetensors(tmp_path, _make_sam_tensors())

        config = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), config)

        # Check encoder keys (plugin uses encoder.layer{i} without dot)
        assert "encoder.patch_embed.weight" in weights
        assert "encoder.patch_embed.bias" in weights
        assert "encoder.pos_embed" in weights

        for i in range(NUM_LAYERS):
            assert f"encoder.layer{i}.norm1.weight" in weights
            assert f"encoder.layer{i}.attn.q.weight" in weights
            assert f"encoder.layer{i}.attn.k.weight" in weights
            assert f"encoder.layer{i}.attn.v.weight" in weights
            assert f"encoder.layer{i}.attn.o.weight" in weights
            assert f"encoder.layer{i}.mlp.fc1.weight" in weights
            assert f"encoder.layer{i}.mlp.fc2.weight" in weights

    def test_load_weights_has_decoder_keys(self, tmp_path):
        from tensorrt_model_connect.families.sam import plugin

        config_dict = _make_sam_config()
        _write_config(tmp_path, config_dict)
        _write_safetensors(tmp_path, _make_sam_tensors())

        config = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), config)

        # Check decoder keys (plugin uses decoder.layer{i} without dot)
        assert "decoder.iou_token" in weights
        assert "decoder.mask_tokens" in weights
        assert "decoder.upscale.conv1.weight" in weights
        assert "decoder.iou_head.0.weight" in weights

        for i in range(DECODER_DEPTH):
            assert f"decoder.layer{i}.self_attn.q.weight" in weights
            assert f"decoder.layer{i}.cross_t2i.q.weight" in weights
            assert f"decoder.layer{i}.cross_i2t.q.weight" in weights

    def test_load_weights_has_prompt_encoder_keys(self, tmp_path):
        from tensorrt_model_connect.families.sam import plugin

        config_dict = _make_sam_config()
        _write_config(tmp_path, config_dict)
        _write_safetensors(tmp_path, _make_sam_tensors())

        config = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), config)

        # Check prompt encoder point embeddings
        assert "prompt.point_embed.0" in weights
        assert "prompt.point_embed.1" in weights
        assert "prompt.no_mask_embed" in weights

    def test_get_segmentation_config(self, tmp_path):
        from tensorrt_model_connect.families.sam import plugin

        config_dict = _make_sam_config()
        _write_config(tmp_path, config_dict)
        _write_safetensors(tmp_path, _make_sam_tensors())

        config = ModelConfig.from_dir(tmp_path)
        # load_weights populates _sam_config
        plugin.load_weights(str(tmp_path), config)

        get_seg = getattr(plugin, 'get_segmentation_config', None)
        assert get_seg is not None

        seg_cfg = get_seg(config)
        assert seg_cfg is not None
        assert seg_cfg["sam_image_embedding_size"] == IMAGE_EMBED_SIZE
        assert seg_cfg["sam_decoder_hidden_size"] == DECODER_HIDDEN
        assert seg_cfg["sam_num_mask_outputs"] == NUM_MULTIMASK + 1
        assert seg_cfg["input_image_h"] == IMAGE_SIZE

    def test_qkv_split(self, tmp_path):
        """Verify fused QKV weight is properly split into Q, K, V."""
        from tensorrt_model_connect.families.sam import plugin

        config_dict = _make_sam_config()
        _write_config(tmp_path, config_dict)

        tensors = _make_sam_tensors()
        # Make a known QKV weight to verify split
        known_qkv = np.arange(HIDDEN * 3 * HIDDEN, dtype=np.float32).reshape(HIDDEN * 3, HIDDEN)
        tensors["vision_encoder.layers.0.attn.qkv.weight"] = known_qkv
        _write_safetensors(tmp_path, tensors)

        config = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), config)

        # Plugin uses encoder.layer{i}.attn.{q,k,v}.weight (no dot before index)
        q_w = weights["encoder.layer0.attn.q.weight"]
        k_w = weights["encoder.layer0.attn.k.weight"]
        v_w = weights["encoder.layer0.attn.v.weight"]

        # QKV was split along dim 0 and transposed by _transpose_2d
        assert q_w.shape == (HIDDEN, HIDDEN)
        assert k_w.shape == (HIDDEN, HIDDEN)
        assert v_w.shape == (HIDDEN, HIDDEN)

        # Verify split matches expected slices (transposed for matmul)
        np.testing.assert_array_equal(q_w, known_qkv[:HIDDEN].T)
        np.testing.assert_array_equal(k_w, known_qkv[HIDDEN:HIDDEN*2].T)
        np.testing.assert_array_equal(v_w, known_qkv[HIDDEN*2:].T)

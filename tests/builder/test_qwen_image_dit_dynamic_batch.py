"""Smoke tests for the Qwen-Image dynamic-batch DiT builder path."""

from __future__ import annotations

import numpy as np

from .conftest import requires_trt


def _rng_for(key: str) -> np.random.Generator:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % (2**32)
    return np.random.default_rng(seed)


def _linear(weights: dict[str, np.ndarray], key: str, out_dim: int, in_dim: int) -> None:
    rng = _rng_for(key)
    weights[f"{key}.weight"] = rng.normal(0.0, 0.01, (out_dim, in_dim)).astype(np.float32)
    weights[f"{key}.bias"] = np.zeros((out_dim,), dtype=np.float32)


def _block_weights(prefix: str, *, hidden: int, head_dim: int, intermediate: int) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    _linear(weights, f"{prefix}.img_mod.1", 6 * hidden, hidden)
    _linear(weights, f"{prefix}.txt_mod.1", 6 * hidden, hidden)

    for name in ("to_q", "to_k", "to_v"):
        _linear(weights, f"{prefix}.attn.{name}", hidden, hidden)
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        _linear(weights, f"{prefix}.attn.{name}", hidden, hidden)
    _linear(weights, f"{prefix}.attn.to_out.0", hidden, hidden)
    _linear(weights, f"{prefix}.attn.to_add_out", hidden, hidden)

    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        weights[f"{prefix}.attn.{name}.weight"] = np.ones((head_dim,), dtype=np.float32)

    _linear(weights, f"{prefix}.img_mlp.net.0.proj", intermediate, hidden)
    _linear(weights, f"{prefix}.img_mlp.net.2", hidden, intermediate)
    _linear(weights, f"{prefix}.txt_mlp.net.0.proj", intermediate, hidden)
    _linear(weights, f"{prefix}.txt_mlp.net.2", hidden, intermediate)
    return weights


def _tiny_weights() -> dict[str, np.ndarray]:
    hidden = 12
    text_dim = 6
    in_channels = 4
    out_channels = 1
    patch = 2
    timestep_dim = 4
    intermediate = 24

    weights: dict[str, np.ndarray] = {}
    _linear(weights, "img_in", hidden, in_channels)
    weights["txt_norm.weight"] = np.ones((text_dim,), dtype=np.float32)
    _linear(weights, "txt_in", hidden, text_dim)
    _linear(weights, "time_text_embed.timestep_embedder.linear_1", hidden, timestep_dim)
    _linear(weights, "time_text_embed.timestep_embedder.linear_2", hidden, hidden)
    weights.update(_block_weights(
        "transformer_blocks.0", hidden=hidden, head_dim=6, intermediate=intermediate))
    _linear(weights, "norm_out.linear", 2 * hidden, hidden)
    _linear(weights, "proj_out", out_channels * patch * patch, hidden)
    return weights


@requires_trt
def test_qwen_image_dit_dynamic_batch_engine_builds_tiny_network(tmp_path):
    import tensorrt as trt
    from tensorrt_model_connect.families.qwen_image.qwen_image_dit_builder import (
        QwenImageDiTConfig,
        build_qwen_image_dit_engine,
    )

    cfg = QwenImageDiTConfig(
        in_channels=4,
        out_channels=1,
        patch_size=2,
        hidden_size=12,
        num_joint_blocks=1,
        num_attention_heads=2,
        attention_head_dim=6,
        intermediate_size=24,
        text_embed_dim=6,
        rope_axes_dim=[2, 2, 2],
        timestep_embed_dim=4,
        max_image_tokens=2,
        max_text_tokens=3,
    )
    out_path = tmp_path / "qwen_image_dit.plan"

    build_qwen_image_dit_engine(
        cfg,
        _tiny_weights(),
        out_path,
        h_lat=1,
        w_lat=2,
        n_text=3,
        batch_size=2,
        verbose=False,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(out_path.read_bytes())
    ctx = engine.create_execution_context()

    for batch in (1, 2):
        assert ctx.set_input_shape("img_patched", (batch, 2, 4))
        assert ctx.set_input_shape("txt_hidden", (batch, 3, 6))
        assert ctx.set_input_shape("timestep", (batch,))
        assert tuple(ctx.get_tensor_shape("noise_patched")) == (batch, 2, 4)

    assert ctx.set_input_shape("img_patched", (3, 2, 4)) is False

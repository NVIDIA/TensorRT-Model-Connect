# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GitHub ELF family plugin.

These tests validate the exact Flax-style weight naming used by
https://github.com/lillian039/ELF without depending on Hugging Face loaders.
"""

from __future__ import annotations

import json
import importlib
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.elf_flow import plugin
    from tensorrt_model_connect.families.elf_flow.config import (
        make_elf_rope_cache,
        resolve_elf_config,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect test dependencies unavailable", allow_module_level=True)


RNG = np.random.RandomState(7)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _cfg(**overrides) -> ModelConfig:
    data = {
        "model_type": "elf",
        "model": "ELF-B",
        "text_encoder_dim": 6,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 11,
        "max_length": 4,
        "max_position_embeddings": 4,
        "bottleneck_dim": 3,
        "num_time_tokens": 2,
        "num_self_cond_cfg_tokens": 1,
        "num_model_mode_tokens": 1,
        "self_cond_prob": 0.5,
    }
    data.update(overrides)
    return ModelConfig.from_json(json.dumps(data))


def _write_model(tmp_path: Path, tensors: dict[str, np.ndarray]) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_cfg().raw), encoding="utf-8")
    save_file(tensors, str(tmp_path / "model.safetensors"))


def _nest_tensor_tree(tensors: dict[str, np.ndarray]) -> dict:
    root: dict = {}
    for name, value in tensors.items():
        parts = name.split(".")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def _find_trtmc_binary() -> Path | None:
    env_path = os.environ.get("TRTMC_BINARY")
    candidates = [
        Path(env_path) if env_path else None,
        Path(__file__).resolve().parents[2] / "build" / "trtmc",
        Path("/tmp/trtmc-elf-build/trtmc"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _tiny_unigram_tokenizer_json(vocab_size: int) -> bytes:
    vocab = [["<unk>", 0.0]]
    vocab.extend([[f"\u2581{chr(ord('A') + idx - 1)}", -float(idx)] for idx in range(1, vocab_size)])
    return json.dumps({
        "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab},
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "add_prefix_space": True,
        },
    }).encode("utf-8")


def _decode_tiny_unigram(ids: np.ndarray, vocab_size: int) -> str:
    pieces = ["<unk>"]
    pieces.extend([f"\u2581{chr(ord('A') + idx - 1)}" for idx in range(1, vocab_size)])
    joined = "".join(pieces[int(idx)] for idx in ids)
    return joined.replace("\u2581", " ").lstrip(" ")


def _elf_tensors(*, layers: int = 2, scale: float = 1.0) -> dict[str, np.ndarray]:
    hidden = 8
    text_dim = 6
    bottleneck = 3
    input_dim = 12
    vocab = 11
    head_dim = 4
    actual_ffn = int(int(hidden * 4.0) * 2 / 3)
    tensors: dict[str, np.ndarray] = {
        "self_cond_proj.kernel": _rand(input_dim, text_dim),
        "self_cond_proj.bias": _rand(text_dim),
        "text_proj.proj1.kernel": _rand(text_dim, bottleneck),
        "text_proj.proj2.kernel": _rand(bottleneck, hidden),
        "text_proj.proj2.bias": _rand(hidden),
        "t_embedder.mlp_0.kernel": _rand(256, hidden),
        "t_embedder.mlp_0.bias": _rand(hidden),
        "t_embedder.mlp_2.kernel": _rand(hidden, hidden),
        "t_embedder.mlp_2.bias": _rand(hidden),
        "t_emb_tokens": _rand(1, 2, hidden),
        "self_cond_cfg_embedder.mlp_0.kernel": _rand(256, hidden),
        "self_cond_cfg_embedder.mlp_0.bias": _rand(hidden),
        "self_cond_cfg_embedder.mlp_2.kernel": _rand(hidden, hidden),
        "self_cond_cfg_embedder.mlp_2.bias": _rand(hidden),
        "self_cond_cfg_tokens": _rand(1, 1, hidden),
        "mode_tokens": _rand(1, 1, hidden),
        "proj_kernel": _rand(hidden, text_dim),
        "proj_bias": _rand(text_dim),
        "unembed_kernel": _rand(text_dim, vocab),
        "unembed_bias": _rand(vocab),
        "final_layer.norm_final.weight": _rand(hidden),
        "final_layer.linear.kernel": _rand(hidden, text_dim),
        "final_layer.linear.bias": _rand(text_dim),
    }
    if scale != 1.0:
        tensors = {key: value * scale for key, value in tensors.items()}
    for i in range(layers):
        p = f"blocks_{i}"
        tensors[f"{p}.norm1.weight"] = _rand(hidden)
        tensors[f"{p}.attn.qkv.kernel"] = _rand(hidden, 3 * hidden) * scale
        tensors[f"{p}.attn.qkv.bias"] = _rand(3 * hidden) * scale
        tensors[f"{p}.attn.q_norm.weight"] = _rand(head_dim)
        tensors[f"{p}.attn.k_norm.weight"] = _rand(head_dim)
        tensors[f"{p}.attn.proj.kernel"] = _rand(hidden, hidden) * scale
        tensors[f"{p}.attn.proj.bias"] = _rand(hidden) * scale
        tensors[f"{p}.norm2.weight"] = _rand(hidden)
        tensors[f"{p}.mlp.w12.kernel"] = _rand(hidden, 2 * actual_ffn) * scale
        tensors[f"{p}.mlp.w12.bias"] = _rand(2 * actual_ffn) * scale
        tensors[f"{p}.mlp.w3.kernel"] = _rand(actual_ffn, hidden) * scale
        tensors[f"{p}.mlp.w3.bias"] = _rand(hidden) * scale
    return tensors


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def _timestep_embedding(t: np.ndarray, dim: int = 256) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(0, half, dtype=np.float32) / half)
    args = t.reshape(-1, 1).astype(np.float32) * freqs.reshape(1, -1)
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)


def _timestep_mlp(t: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    emb = _timestep_embedding(t)
    emb = emb @ weights[f"{prefix}.mlp_0.w"] + weights[f"{prefix}.mlp_0.b"]
    emb = _silu(emb)
    return emb @ weights[f"{prefix}.mlp_2.w"] + weights[f"{prefix}.mlp_2.b"]


def _apply_github_rope(x: np.ndarray, *, max_length: int, prefix_tokens: int) -> np.ndarray:
    seq, _, head_dim = x.shape
    freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.arange(max_length, dtype=np.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    cos_main = np.repeat(np.cos(angles), 2, axis=-1)
    sin_main = np.repeat(np.sin(angles), 2, axis=-1)
    cos = np.concatenate([
        np.ones((prefix_tokens, head_dim), dtype=np.float32),
        cos_main,
    ], axis=0)[:seq]
    sin = np.concatenate([
        np.zeros((prefix_tokens, head_dim), dtype=np.float32),
        sin_main,
    ], axis=0)[:seq]
    pairs = x.reshape(seq, x.shape[1], head_dim // 2, 2)
    rotated = np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _numpy_elf_forward(
    weights: dict[str, np.ndarray],
    cfg: dict,
    latent: np.ndarray,
    timestep: float,
    self_cond_cfg_scale: float,
    decoder_mode: float,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_size = cfg["hidden_size"]
    text_dim = cfg["text_encoder_dim"]
    max_length = cfg["max_length"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    x = latent.astype(np.float32)
    if x.shape[-1] == 2 * text_dim:
        x = x @ weights["self_cond_proj.w"] + weights["self_cond_proj.b"]

    x = x @ weights["text_proj.proj1.w"]
    x = x @ weights["text_proj.proj2.w"] + weights["text_proj.proj2.b"]

    mode_tokens = cfg["num_model_mode_tokens"]
    if mode_tokens > 0:
        mode = weights["mode_tokens"].reshape(mode_tokens, hidden_size) * decoder_mode
        x = np.concatenate([mode, x], axis=0)

    t_emb = _timestep_mlp(np.array([timestep], dtype=np.float32), weights, "t_embedder")
    prefix_parts = [
        weights["t_emb_tokens"].reshape(cfg["num_time_tokens"], hidden_size) + t_emb
    ]
    if cfg["num_self_cond_cfg_tokens"] > 0:
        sc_emb = _timestep_mlp(
            np.array([self_cond_cfg_scale], dtype=np.float32),
            weights,
            "self_cond_cfg_embedder",
        )
        prefix_parts.append(
            weights["self_cond_cfg_tokens"].reshape(
                cfg["num_self_cond_cfg_tokens"], hidden_size) + sc_emb
        )
    prefix = np.concatenate(prefix_parts, axis=0)
    x = np.concatenate([prefix, x], axis=0)
    empty_tokens = prefix.shape[0] + mode_tokens

    for layer_idx in range(cfg["depth"]):
        p = f"layer.{layer_idx}"
        normed = _rms_norm(x, weights[f"{p}.norm1"])
        qkv = normed @ weights[f"{p}.attn.qkv.w"] + weights[f"{p}.attn.qkv.b"]
        qkv = qkv.reshape(x.shape[0], 3, num_heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = _rms_norm(q, weights[f"{p}.attn.q_norm"])
        k = _rms_norm(k, weights[f"{p}.attn.k_norm"])
        q = _apply_github_rope(q, max_length=max_length, prefix_tokens=empty_tokens)
        k = _apply_github_rope(k, max_length=max_length, prefix_tokens=empty_tokens)
        qh = np.transpose(q, (1, 0, 2))
        kh = np.transpose(k, (1, 0, 2))
        vh = np.transpose(v, (1, 0, 2))
        scores = (qh @ np.swapaxes(kh, -1, -2)) / np.sqrt(head_dim)
        attn = _softmax(scores, axis=-1) @ vh
        attn = np.transpose(attn, (1, 0, 2)).reshape(x.shape[0], hidden_size)
        attn = attn @ weights[f"{p}.attn.proj.w"] + weights[f"{p}.attn.proj.b"]
        x = x + attn

        normed = _rms_norm(x, weights[f"{p}.norm2"])
        fused = normed @ weights[f"{p}.mlp.w12.w"] + weights[f"{p}.mlp.w12.b"]
        x1, x2 = np.split(fused, 2, axis=-1)
        mlp = (_silu(x1) * x2) @ weights[f"{p}.mlp.w3.w"] + weights[f"{p}.mlp.w3.b"]
        x = x + mlp

    body = x[empty_tokens:]
    logits = _gelu_tanh(body @ weights["decoder.proj.w"] + weights["decoder.proj.b"])
    logits = logits @ weights["decoder.unembed.w"] + weights["decoder.unembed.b"]
    logits = logits * decoder_mode
    denoised = _rms_norm(body, weights["final.norm"])
    denoised = denoised @ weights["final.linear.w"] + weights["final.linear.b"]
    return denoised.astype(np.float32), logits.astype(np.float32)


def test_elf_plugin_matches_and_resolves_config() -> None:
    assert plugin.matches("elf")
    assert plugin.matches("embedded-language-flow")
    assert not plugin.matches("llama")

    cfg = resolve_elf_config(_cfg())
    assert cfg["hidden_size"] == 8
    assert cfg["depth"] == 2
    assert cfg["num_heads"] == 2
    assert cfg["text_encoder_dim"] == 6
    assert cfg["input_dim"] == 12
    assert cfg["max_length"] == 4


def test_model_config_from_dir_accepts_github_elf_yaml(tmp_path: Path) -> None:
    (tmp_path / "train_owt_ELF-B.yml").write_text(
        "\n".join([
            "model: ELF-B",
            "max_length: 1024",
            "encoder_model_name: t5-small",
            "denoiser_p_mean: -1.5",
            "denoiser_p_std: 0.8",
            "denoiser_noise_scale: 2.0",
            "self_cond_prob: 0.5",
            "num_time_tokens: 4",
            "num_self_cond_cfg_tokens: 4",
            "num_model_mode_tokens: 4",
        ]),
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)
    resolved = resolve_elf_config(cfg)

    assert cfg.model_type == "elf"
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert resolved["max_length"] == 1024
    assert resolved["denoiser_noise_scale"] == 2.0


def test_resolve_elf_config_honors_builder_max_length() -> None:
    cfg = _cfg(max_length=1024, max_position_embeddings=1024)

    resolved = resolve_elf_config(cfg, max_seq_length=128)

    assert resolved["max_length"] == 128


def test_load_weights_uses_github_flax_shapes_without_transpose(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    qkv = tensors["blocks_0.attn.qkv.kernel"].copy()
    _write_model(tmp_path, tensors)

    weights = plugin.load_weights(str(tmp_path), _cfg())

    assert weights["self_cond_proj.w"].shape == (12, 6)
    assert weights["text_proj.proj1.w"].shape == (6, 3)
    assert weights["layer.0.attn.qkv.w"].shape == (8, 24)
    np.testing.assert_allclose(weights["layer.0.attn.qkv.w"], qkv)
    assert weights["layer.0.attn.q_norm"].shape == (4,)
    assert weights["layer.0.mlp.w12.w"].shape == (8, 42)
    assert weights["decoder.unembed.w"].shape == (6, 11)
    assert weights["final.linear.w"].shape == (8, 6)


def test_load_weights_accepts_local_github_checkpoint_and_uses_ema_params(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    params = {key: value + 100.0 for key, value in tensors.items()}
    with (tmp_path / "checkpoint_42").open("wb") as f:
        pickle.dump(
            {
                "params": _nest_tensor_tree(params),
                "ema_params1": _nest_tensor_tree(tensors),
                "opt_state": {},
                "step": 42,
                "epoch": 0,
                "dropout_rng": np.array([0], dtype=np.uint32),
            },
            f,
        )

    weights = plugin.load_weights(str(tmp_path), _cfg())

    np.testing.assert_allclose(weights["layer.0.attn.qkv.w"], tensors["blocks_0.attn.qkv.kernel"])
    np.testing.assert_allclose(weights["decoder.unembed.w"], tensors["unembed_kernel"])


def test_orbax_checkpoint_loader_selects_ema_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.elf_flow.plugin")
    checkpoint = tmp_path / "checkpoint_0"
    checkpoint.mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = {
        "params": {"weight": expected + 100.0},
        "ema_params1": {"weight": expected},
    }

    fake_checkpoint = types.SimpleNamespace(
        PyTreeCheckpointer=lambda: types.SimpleNamespace(
            restore=lambda path: payload))
    fake_orbax = types.SimpleNamespace(checkpoint=fake_checkpoint)
    monkeypatch.setitem(sys.modules, "orbax", fake_orbax)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", fake_checkpoint)

    arrays = module._load_orbax_arrays(checkpoint)

    assert arrays is not None
    np.testing.assert_array_equal(arrays["weight"], expected)


def test_load_weights_infers_vocab_size_from_unembed_kernel(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    cfg = _cfg(vocab_size=0)

    weights = plugin.load_weights(str(tmp_path), cfg)

    assert cfg.vocab_size == tensors["unembed_kernel"].shape[1]
    assert cfg.raw["vocab_size"] == tensors["unembed_kernel"].shape[1]
    assert weights["decoder.unembed.w"].shape == tensors["unembed_kernel"].shape


def test_bundle_config_overrides_advertise_api_runtime() -> None:
    overrides = plugin.get_bundle_config_overrides(_cfg())
    assert overrides["runtime_strategy"] == "elf_flow"
    assert overrides["model_type"] == "elf"
    assert overrides["elf_max_length"] == 4
    assert overrides["elf_text_encoder_dim"] == 6
    assert overrides["elf_input_dim"] == 12
    assert overrides["elf_num_time_tokens"] == 2
    assert overrides["elf_num_self_cond_cfg_tokens"] == 1
    assert overrides["elf_num_model_mode_tokens"] == 1
    assert overrides["elf_denoiser_p_mean"] == -1.5
    assert overrides["elf_denoiser_p_std"] == 0.8
    assert overrides["elf_latent_mean"] == 0.0
    assert overrides["elf_latent_std"] == 0.2
    assert overrides["elf_encoder_model_name"] == "t5-small"
    assert overrides["elf_encoder_pad_token_id"] == 0
    assert overrides["elf_has_text_encoder"] == 0
    assert overrides["elf_runtime_contract"] == "api_path_denoise_or_decode_logits"
    assert overrides["elf_user_contract"] == "diffusion_text_generation"
    assert overrides["elf_output_schema"] == "jsonl_id_generated_after_sampler_decode"
    assert overrides["elf_max_input_length"] == 0


def test_load_weights_marks_official_jax_t5_encoder_checkpoint(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    (tmp_path / "t5_small_encoder_jax.pkl").write_bytes(b"encoder")
    cfg = _cfg(pad_token="eos", eos_token_id=1, latent_std=0.25)

    weights = plugin.load_weights(str(tmp_path), cfg)
    overrides = plugin.get_bundle_config_overrides(cfg)

    assert weights["_elf_encoder_checkpoint"].endswith("t5_small_encoder_jax.pkl")
    assert overrides["elf_has_text_encoder"] == 1
    assert overrides["elf_encoder_pad_token_id"] == 1
    assert overrides["elf_latent_std"] == 0.25


def test_build_extra_engines_compiles_official_jax_t5_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    (tmp_path / "t5_small_encoder_jax.pkl").write_bytes(b"encoder")
    cfg = _cfg(text_encoder_dim=512)
    weights = plugin.load_weights(str(tmp_path), cfg)

    def load_jax_t5_encoder_weights(path: str, **kwargs):
        calls["load"] = {"path": path, **kwargs}
        return {"shared.weight": np.zeros((32128, 512), dtype=np.float32)}

    def build_t5_encoder_engine(weights_arg, **kwargs):
        calls["build"] = {"weights": weights_arg, **kwargs}
        return b"t5-plan"

    fake_t5_builder = types.SimpleNamespace(
        load_jax_t5_encoder_weights=load_jax_t5_encoder_weights,
        build_t5_encoder_engine=build_t5_encoder_engine,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.elf_flow.t5_encoder_builder",
        fake_t5_builder,
    )

    out = plugin.build_extra_engines(cfg, weights, 128, precision="fp32", verbose=True)

    assert out == {"elf_text_encoder_plan": b"t5-plan"}
    assert calls["load"]["precision"] == "fp32"
    assert calls["load"]["num_layers"] == 6
    assert calls["build"]["d_model"] == 512
    assert calls["build"]["num_layers"] == 6
    assert calls["build"]["vocab_size"] == 32128
    assert calls["build"]["max_seq_len"] == 4
    assert "is_gated_act" not in calls["build"]


def test_elf_rope_cache_matches_github_empty_token_semantics() -> None:
    cos, sin = make_elf_rope_cache(max_length=3, head_dim=4, prefix_tokens=2)
    assert cos.shape == (1, 5, 2)
    assert sin.shape == (1, 5, 2)
    np.testing.assert_allclose(cos[0, :2], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_allclose(sin[0, :2], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(cos[0, 2], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(sin[0, 2], np.zeros(2, dtype=np.float32))

    freqs = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4))
    np.testing.assert_allclose(cos[0, 3], np.cos(freqs), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin[0, 3], np.sin(freqs), rtol=1e-6, atol=1e-6)


@pytest.mark.trt
def test_elf_trt_forward_matches_github_numpy_reference(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.elf_flow.debug_runner import VisionTrtRunner
    from tensorrt_model_connect.families.elf_flow.builder import build_elf_flow_engine

    cfg = _cfg(num_hidden_layers=1)
    tensors = _elf_tensors(layers=1, scale=0.05)
    _write_model(tmp_path, tensors)
    weights = plugin.load_weights(str(tmp_path), cfg)
    resolved = resolve_elf_config(cfg)

    plan = build_elf_flow_engine(cfg, weights, precision="fp32")
    runner = VisionTrtRunner(plan)

    latent = (np.arange(
        resolved["max_length"] * resolved["input_dim"], dtype=np.float32
    ).reshape(resolved["max_length"], resolved["input_dim"]) / 50.0) - 0.2
    timestep = np.array([0.35], dtype=np.float32)
    self_cond_cfg_scale = np.array([1.25], dtype=np.float32)

    outputs = runner.encode(
        latent=latent,
        timestep=timestep,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_mode=np.array([1.0], dtype=np.float32),
    )
    ref_denoised, ref_logits = _numpy_elf_forward(
        weights, resolved, latent, 0.35, 1.25, 1.0)
    np.testing.assert_allclose(outputs["denoised"], ref_denoised, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(outputs["decoder_logits"], ref_logits, rtol=2e-3, atol=2e-3)

    outputs_off = runner.encode(
        latent=latent,
        timestep=timestep,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_mode=np.array([0.0], dtype=np.float32),
    )
    ref_denoised_off, ref_logits_off = _numpy_elf_forward(
        weights, resolved, latent, 0.35, 1.25, 0.0)
    np.testing.assert_allclose(
        outputs_off["denoised"], ref_denoised_off, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        outputs_off["decoder_logits"], ref_logits_off, rtol=2e-3, atol=2e-3)


@pytest.mark.trt
def test_elf_trtmc_run_generates_text_from_diffusion_decode(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.families.elf_flow.builder import build_elf_flow_engine

    trtmc_binary = _find_trtmc_binary()
    if trtmc_binary is None:
        pytest.skip("trtmc binary not built")
    backend_dir = trtmc_binary.parent
    if not (backend_dir / "libtrtmc_backend_trt.so").exists():
        pytest.skip("trtmc_backend_trt DSO not built next to trtmc")

    cfg = _cfg(num_hidden_layers=1)
    tensors = _elf_tensors(layers=1, scale=0.05)
    _write_model(tmp_path, tensors)
    weights = plugin.load_weights(str(tmp_path), cfg)
    resolved = resolve_elf_config(cfg)
    plan = build_elf_flow_engine(cfg, weights, precision="fp32")

    config_json = {
        "runtime_strategy": "elf_flow",
        "engine_backend": "trt",
        "model_type": "elf",
        "vocab_size": resolved["vocab_size"],
        "elf_max_length": resolved["max_length"],
        "elf_text_encoder_dim": resolved["text_encoder_dim"],
        "elf_input_dim": resolved["input_dim"],
        "elf_denoiser_noise_scale": 2.0,
        "elf_denoiser_p_mean": -1.5,
        "elf_denoiser_p_std": 0.8,
        "elf_t_eps": 0.05,
    }
    bundle_path = tmp_path / "tiny-elf.trtfb"
    write_bundle(
        bundle_path,
        BundleInfo(
            model_id="tiny-elf",
            model_type="elf",
            family="elf_flow",
            runtime_strategy="elf_flow",
            vocab_size=resolved["vocab_size"],
            hidden_size=resolved["hidden_size"],
            num_layers=resolved["depth"],
            num_attention_heads=resolved["num_heads"],
            tokenizer_add_special_tokens=False,
        ),
        [
            BundleSection("engine_plan", plan),
            BundleSection("config.json", json.dumps(config_json).encode("utf-8")),
            BundleSection("tokenizer.json", _tiny_unigram_tokenizer_json(resolved["vocab_size"])),
        ],
    )

    z0 = (np.arange(
        resolved["max_length"] * resolved["text_encoder_dim"], dtype=np.float32
    ).reshape(resolved["max_length"], resolved["text_encoder_dim"]) / 20.0) - 0.5
    latents_path = tmp_path / "initial_latents.raw"
    z0.tofile(latents_path)

    zeros = np.zeros_like(z0)
    latent0 = np.concatenate([z0, zeros], axis=-1)
    denoised, _ = _numpy_elf_forward(weights, resolved, latent0, 0.0, 1.0, 0.0)
    latent1 = np.concatenate([denoised, zeros], axis=-1)
    _, logits = _numpy_elf_forward(weights, resolved, latent1, 1.0, 1.0, 1.0)
    expected_ids = np.argmax(logits, axis=-1).astype(np.int32)
    expected_text = _decode_tiny_unigram(expected_ids, resolved["vocab_size"])
    assert expected_text

    result = subprocess.run(
        [
            str(trtmc_binary),
            "run",
            str(bundle_path),
            "--num-steps",
            "1",
            "--guidance-scale",
            "1",
            "--cfg-scale",
            "2",
            "--sde-gamma",
            "0",
            "--seed",
            "123",
            "--initial-latents-raw",
            str(latents_path),
            "--backend-dir",
            str(backend_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\n") == expected_text

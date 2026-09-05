# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.elf_flow import model as elf_model
from families.elf_flow.config import make_elf_rope_cache, resolve_elf_config
from families.elf_flow.model_config import ModelConfig


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


def _elf_tensors(*, layers: int = 2) -> dict[str, np.ndarray]:
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
    for layer_idx in range(layers):
        prefix = f"blocks_{layer_idx}"
        tensors[f"{prefix}.norm1.weight"] = _rand(hidden)
        tensors[f"{prefix}.attn.qkv.kernel"] = _rand(hidden, 3 * hidden)
        tensors[f"{prefix}.attn.qkv.bias"] = _rand(3 * hidden)
        tensors[f"{prefix}.attn.q_norm.weight"] = _rand(head_dim)
        tensors[f"{prefix}.attn.k_norm.weight"] = _rand(head_dim)
        tensors[f"{prefix}.attn.proj.kernel"] = _rand(hidden, hidden)
        tensors[f"{prefix}.attn.proj.bias"] = _rand(hidden)
        tensors[f"{prefix}.norm2.weight"] = _rand(hidden)
        tensors[f"{prefix}.mlp.w12.kernel"] = _rand(hidden, 2 * actual_ffn)
        tensors[f"{prefix}.mlp.w12.bias"] = _rand(2 * actual_ffn)
        tensors[f"{prefix}.mlp.w3.kernel"] = _rand(actual_ffn, hidden)
        tensors[f"{prefix}.mlp.w3.bias"] = _rand(hidden)
    return tensors


def test_model_config_from_dir_accepts_github_elf_yaml(tmp_path: Path) -> None:
    (tmp_path / "train_owt_ELF-B.yml").write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
    )

    config = ModelConfig.from_dir(tmp_path)
    resolved = resolve_elf_config(config)

    assert config.model_type == "elf_flow"
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert resolved["max_length"] == 1024
    assert resolved["denoiser_noise_scale"] == 2.0


def test_resolve_elf_config_honors_builder_max_length() -> None:
    config = _cfg(max_length=1024, max_position_embeddings=1024)

    resolved = resolve_elf_config(config, max_seq_length=128)

    assert resolved["max_length"] == 128


def test_load_weights_uses_github_flax_shapes_without_transpose(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    qkv = tensors["blocks_0.attn.qkv.kernel"].copy()
    _write_model(tmp_path, tensors)

    weights = elf_model._ElfFlowModel().load_weights(str(tmp_path), _cfg())

    assert weights["self_cond_proj.w"].shape == (12, 6)
    assert weights["text_proj.proj1.w"].shape == (6, 3)
    assert weights["layer.0.attn.qkv.w"].shape == (8, 24)
    np.testing.assert_allclose(weights["layer.0.attn.qkv.w"], qkv)
    assert weights["layer.0.attn.q_norm"].shape == (4,)
    assert weights["layer.0.mlp.w12.w"].shape == (8, 42)
    assert weights["decoder.unembed.w"].shape == (6, 11)
    assert weights["final.linear.w"].shape == (8, 6)


def test_load_weights_accepts_local_github_checkpoint_and_uses_ema_params(
    tmp_path: Path,
) -> None:
    tensors = _elf_tensors()
    params = {key: value + 100.0 for key, value in tensors.items()}
    with (tmp_path / "checkpoint_42").open("wb") as checkpoint:
        pickle.dump(
            {
                "params": _nest_tensor_tree(params),
                "ema_params1": _nest_tensor_tree(tensors),
                "opt_state": {},
                "step": 42,
                "epoch": 0,
                "dropout_rng": np.array([0], dtype=np.uint32),
            },
            checkpoint,
        )

    weights = elf_model._ElfFlowModel().load_weights(str(tmp_path), _cfg())

    np.testing.assert_allclose(weights["layer.0.attn.qkv.w"], tensors["blocks_0.attn.qkv.kernel"])
    np.testing.assert_allclose(weights["decoder.unembed.w"], tensors["unembed_kernel"])


def test_orbax_checkpoint_loader_selects_ema_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("families.elf_flow.model")
    checkpoint = tmp_path / "checkpoint_0"
    checkpoint.mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = {
        "params": {"weight": expected + 100.0},
        "ema_params1": {"weight": expected},
    }

    fake_checkpoint = types.SimpleNamespace(
        PyTreeCheckpointer=lambda: types.SimpleNamespace(restore=lambda path: payload)
    )
    fake_orbax = types.SimpleNamespace(checkpoint=fake_checkpoint)
    monkeypatch.setitem(sys.modules, "orbax", fake_orbax)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", fake_checkpoint)

    arrays = module._load_orbax_arrays(checkpoint)

    assert arrays is not None
    np.testing.assert_array_equal(arrays["weight"], expected)


def test_load_weights_infers_vocab_size_from_unembed_kernel(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    config = _cfg(vocab_size=0)

    weights = elf_model._ElfFlowModel().load_weights(str(tmp_path), config)

    assert config.vocab_size == tensors["unembed_kernel"].shape[1]
    assert config.raw["vocab_size"] == tensors["unembed_kernel"].shape[1]
    assert weights["decoder.unembed.w"].shape == tensors["unembed_kernel"].shape


def test_bundle_config_overrides_advertise_runtime_contract() -> None:
    overrides = elf_model._ElfFlowModel().get_bundle_config_overrides(_cfg())

    assert overrides == {
        "max_length": 4,
        "max_input_length": 0,
        "input_dim": 12,
        "text_encoder_dim": 6,
        "vocab_size": 11,
        "denoiser_noise_scale": 1.0,
        "denoiser_p_mean": -1.5,
        "denoiser_p_std": 0.8,
        "timestep_epsilon": 0.05,
        "latent_mean": 0.0,
        "latent_std": 0.2,
        "encoder_pad_token_id": 0,
    }


def test_load_weights_marks_official_jax_t5_encoder_checkpoint(tmp_path: Path) -> None:
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    (tmp_path / "t5_small_encoder_jax.pkl").write_bytes(b"encoder")
    config = _cfg(pad_token="eos", eos_token_id=1, latent_std=0.25)
    model = elf_model._ElfFlowModel()

    weights = model.load_weights(str(tmp_path), config)
    overrides = model.get_bundle_config_overrides(config)

    assert weights["_elf_encoder_checkpoint"].endswith("t5_small_encoder_jax.pkl")
    assert overrides["encoder_pad_token_id"] == 1
    assert overrides["latent_std"] == 0.25


def test_build_extra_engines_compiles_official_jax_t5_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    tensors = _elf_tensors()
    _write_model(tmp_path, tensors)
    (tmp_path / "t5_small_encoder_jax.pkl").write_bytes(b"encoder")
    config = _cfg(text_encoder_dim=512)
    model = elf_model._ElfFlowModel()
    weights = model.load_weights(str(tmp_path), config)

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
    monkeypatch.setitem(sys.modules, "families.elf_flow.t5_encoder_builder", fake_t5_builder)

    out = model.build_extra_engines(config, weights, 128, precision="fp32", verbose=True)

    assert out == {"text_encoder.plan": b"t5-plan"}
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

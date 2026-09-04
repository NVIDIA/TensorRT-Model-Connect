# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF flow family plugin.

ELF is implemented from the GitHub source at https://github.com/lillian039/ELF.
The weight names below mirror the Flax module tree in ``src/modules/model.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_mapper import WeightDict
from .model_config import ModelConfig
from .config import resolve_elf_config


class _TensorStore:
    def __init__(self, model_dir: str | Path):
        from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors

        self._has_tensor = _has_tensor
        self._load_tensor = _load_tensor
        self._readers = None
        self._arrays: dict[str, np.ndarray] | None = None
        model_path = Path(model_dir)
        try:
            self._readers = _open_safetensors(model_path)
        except FileNotFoundError:
            self._arrays = _load_local_elf_arrays(model_path)
            if self._arrays is None:
                raise FileNotFoundError(
                    f"No ELF safetensors, npz, or local GitHub checkpoint found in {model_path}"
                )

    def has(self, name: str) -> bool:
        if self._arrays is not None:
            return name in self._arrays
        return bool(self._has_tensor(self._readers, name))

    def get(self, name: str) -> np.ndarray:
        if self._arrays is not None:
            return np.asarray(self._arrays[name], dtype=np.float32)
        return self._load_tensor(self._readers, name)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            arrays.update(_flatten_arrays(item, name))
        return arrays
    if prefix:
        try:
            arrays[prefix] = np.asarray(value)
        except (TypeError, ValueError):
            pass
    return arrays


def _select_upstream_params(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        if "ema_params1" in payload:
            return payload["ema_params1"]
        if "params" in payload:
            return payload["params"]
    return payload


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if not hasattr(loaded, "files"):
        return None
    return {key: loaded[key] for key in loaded.files}


def _load_pickle_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_flax_arrays(path: Path) -> dict[str, np.ndarray] | None:
    try:
        from flax import serialization
    except ImportError:
        return None

    if path.is_file():
        try:
            payload = serialization.msgpack_restore(path.read_bytes())
        except Exception:
            return None
    else:
        try:
            from flax.training import checkpoints

            payload = checkpoints.restore_checkpoint(str(path.resolve()), target=None)
        except Exception:
            return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_orbax_arrays(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_dir() or not (
        (path / "_CHECKPOINT_METADATA").exists() or (path / "manifest.ocdbt").exists()
    ):
        return None
    try:
        import orbax.checkpoint as ocp

        payload = ocp.PyTreeCheckpointer().restore(str(path.resolve()))
    except Exception:
        return None
    arrays = _flatten_arrays(_select_upstream_params(payload))
    return arrays or None


def _load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray] | None:
    if path.suffix == ".npz":
        arrays = _load_npz_arrays(path)
        if arrays:
            return arrays
    arrays = _load_orbax_arrays(path)
    if arrays:
        return arrays
    arrays = _load_pickle_arrays(path)
    if arrays:
        return arrays
    return _load_flax_arrays(path)


def _local_checkpoint_candidates(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        return []

    candidates: list[Path] = []
    for name in ("model.npz", "elf_params.npz"):
        candidate = model_path / name
        if candidate.exists():
            candidates.append(candidate)

    checkpoints = sorted(
        model_path.glob("checkpoint_*"),
        key=lambda item: (_checkpoint_step(item), item.name),
        reverse=True,
    )
    candidates.extend(checkpoints)
    return candidates


def _load_local_elf_arrays(model_path: Path) -> dict[str, np.ndarray] | None:
    for candidate in _local_checkpoint_candidates(model_path):
        arrays = _load_checkpoint_arrays(candidate)
        if arrays:
            return arrays
    return None


def _name_variants(name: str) -> list[str]:
    variants = [name]
    if "." in name:
        variants.append(name.replace(".", "/"))
    if "/" in name:
        variants.append(name.replace("/", "."))
    prefixed: list[str] = []
    for item in variants:
        prefixed.append(f"params.{item}")
        prefixed.append(f"params/{item}")
    out: list[str] = []
    for item in variants + prefixed:
        if item not in out:
            out.append(item)
    return out


def _load(store: _TensorStore, *names: str, dtype: np.dtype = np.float32) -> np.ndarray:
    for name in names:
        for candidate in _name_variants(name):
            if store.has(candidate):
                return np.ascontiguousarray(store.get(candidate), dtype=dtype)
    joined = ", ".join(names)
    raise KeyError(f"ELF tensor not found; tried: {joined}")


def _find_encoder_checkpoint(model_dir: str | Path) -> Path | None:
    model_path = Path(model_dir)
    for name in (
        "t5_small_encoder_jax.pkl",
        "encoder_checkpoint.pkl",
        "text_encoder.pkl",
        "t5_encoder.pkl",
    ):
        candidate = model_path / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_encoder_checkpoint(model_dir: str | Path) -> Path:
    local = _find_encoder_checkpoint(model_dir)
    if local is not None:
        return local
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            "embedded-language-flows/t5_small_encoder_jax",
            "t5_small_encoder_jax.pkl",
        )
    )


def _resolve_tokenizer_dir(model_dir: str | Path) -> Path:
    local = Path(model_dir)
    if (local / "tokenizer.json").is_file():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            "google-t5/t5-small",
            allow_patterns=(
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "spiece.model",
            ),
        )
    )


def _elf_encoder_pad_token_id(config: ModelConfig) -> int:
    raw = config.raw or {}
    explicit = raw.get("elf_encoder_pad_token_id", raw.get("encoder_pad_token_id"))
    if explicit is not None:
        return int(explicit)
    pad_token_id = raw.get("pad_token_id", config.pad_token_id)
    if isinstance(pad_token_id, int) and pad_token_id >= 0:
        return int(pad_token_id)
    if str(raw.get("pad_token", "")).lower() == "eos":
        eos_token_id = raw.get("eos_token_id", config.eos_token_id)
        return int(eos_token_id) if isinstance(eos_token_id, int) and eos_token_id >= 0 else 1
    return 0


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _ElfFlowModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        cfg = resolve_elf_config(config)
        store = _TensorStore(model_dir)
        # Keep source weights in FP32 so fp32_layers can preserve individual
        # blocks without first rounding their constants through FP16.
        target_dtype = np.float32
        weights = WeightDict()
        config.raw["_elf_model_dir"] = str(Path(model_dir).resolve())
        encoder_checkpoint = _find_encoder_checkpoint(model_dir)
        if encoder_checkpoint is not None:
            weights["_elf_encoder_checkpoint"] = str(encoder_checkpoint.resolve())
            config.raw["_elf_encoder_checkpoint"] = str(encoder_checkpoint.resolve())

        def proj(name: str, *aliases: str) -> np.ndarray:
            return _load(store, name, *aliases, dtype=target_dtype)

        def vec(name: str, *aliases: str) -> np.ndarray:
            return _load(store, name, *aliases, dtype=np.float32)

        if cfg["input_dim"] == 2 * cfg["text_encoder_dim"]:
            weights["self_cond_proj.w"] = proj("self_cond_proj.kernel")
            weights["self_cond_proj.b"] = proj("self_cond_proj.bias")

        weights["text_proj.proj1.w"] = proj("text_proj.proj1.kernel")
        weights["text_proj.proj2.w"] = proj("text_proj.proj2.kernel")
        weights["text_proj.proj2.b"] = proj("text_proj.proj2.bias")

        weights["t_embedder.mlp_0.w"] = proj("t_embedder.mlp_0.kernel")
        weights["t_embedder.mlp_0.b"] = proj("t_embedder.mlp_0.bias")
        weights["t_embedder.mlp_2.w"] = proj("t_embedder.mlp_2.kernel")
        weights["t_embedder.mlp_2.b"] = proj("t_embedder.mlp_2.bias")
        weights["t_emb_tokens"] = proj("t_emb_tokens")

        if cfg["num_self_cond_cfg_tokens"] > 0:
            weights["self_cond_cfg_embedder.mlp_0.w"] = proj("self_cond_cfg_embedder.mlp_0.kernel")
            weights["self_cond_cfg_embedder.mlp_0.b"] = proj("self_cond_cfg_embedder.mlp_0.bias")
            weights["self_cond_cfg_embedder.mlp_2.w"] = proj("self_cond_cfg_embedder.mlp_2.kernel")
            weights["self_cond_cfg_embedder.mlp_2.b"] = proj("self_cond_cfg_embedder.mlp_2.bias")
            weights["self_cond_cfg_tokens"] = proj("self_cond_cfg_tokens")

        if cfg["num_model_mode_tokens"] > 0:
            weights["mode_tokens"] = proj("mode_tokens")

        for layer_idx in range(cfg["depth"]):
            src = f"blocks_{layer_idx}"
            dst = f"layer.{layer_idx}"
            weights[f"{dst}.norm1"] = vec(f"{src}.norm1.weight")
            weights[f"{dst}.attn.qkv.w"] = proj(f"{src}.attn.qkv.kernel")
            weights[f"{dst}.attn.qkv.b"] = proj(f"{src}.attn.qkv.bias")
            weights[f"{dst}.attn.q_norm"] = vec(f"{src}.attn.q_norm.weight")
            weights[f"{dst}.attn.k_norm"] = vec(f"{src}.attn.k_norm.weight")
            weights[f"{dst}.attn.proj.w"] = proj(f"{src}.attn.proj.kernel")
            weights[f"{dst}.attn.proj.b"] = proj(f"{src}.attn.proj.bias")
            weights[f"{dst}.norm2"] = vec(f"{src}.norm2.weight")
            weights[f"{dst}.mlp.w12.w"] = proj(f"{src}.mlp.w12.kernel")
            weights[f"{dst}.mlp.w12.b"] = proj(f"{src}.mlp.w12.bias")
            weights[f"{dst}.mlp.w3.w"] = proj(f"{src}.mlp.w3.kernel")
            weights[f"{dst}.mlp.w3.b"] = proj(f"{src}.mlp.w3.bias")

        weights["decoder.proj.w"] = proj("proj_kernel")
        weights["decoder.proj.b"] = proj("proj_bias")
        weights["decoder.unembed.w"] = proj("unembed_kernel")
        weights["decoder.unembed.b"] = proj("unembed_bias")
        if config.vocab_size <= 0 and weights["decoder.unembed.w"].ndim == 2:
            config.vocab_size = int(weights["decoder.unembed.w"].shape[1])
            config.raw["vocab_size"] = config.vocab_size
        weights["final.norm"] = vec("final_layer.norm_final.weight")
        weights["final.linear.w"] = proj("final_layer.linear.kernel")
        weights["final.linear.b"] = proj("final_layer.linear.bias")
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        from .builder import build_elf_flow_engine

        return build_elf_flow_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict | None:
        del max_cache_length
        encoder_checkpoint = weights.get("_elf_encoder_checkpoint")
        if not encoder_checkpoint:
            return {}

        from .t5_encoder_builder import (
            build_t5_encoder_engine,
            load_jax_t5_encoder_weights,
        )

        cfg = resolve_elf_config(config)
        t5_weights = load_jax_t5_encoder_weights(
            str(encoder_checkpoint), precision=precision, num_layers=6
        )
        t5_plan = build_t5_encoder_engine(
            t5_weights,
            d_model=cfg["text_encoder_dim"],
            num_heads=8,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            vocab_size=32128,
            max_seq_len=cfg["max_length"],
            eps=1e-6,
            verbose=verbose,
        )
        return {"text_encoder.plan": t5_plan}

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = resolve_elf_config(config)
        raw = config.raw or {}
        return {
            "max_length": cfg["max_length"],
            "max_input_length": cfg["max_input_length"],
            "input_dim": cfg["input_dim"],
            "text_encoder_dim": cfg["text_encoder_dim"],
            "vocab_size": cfg["vocab_size"],
            "denoiser_noise_scale": cfg["denoiser_noise_scale"],
            "denoiser_p_mean": cfg["denoiser_p_mean"],
            "denoiser_p_std": cfg["denoiser_p_std"],
            "timestep_epsilon": cfg["t_eps"],
            "latent_mean": float(raw.get("latent_mean", 0.0)),
            "latent_std": float(raw.get("latent_std", 0.2)),
            "encoder_pad_token_id": _elf_encoder_pad_token_id(config),
        }


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one ELF Flow model and its text encoder."""
    if request.image_height is not None:
        raise NotImplementedError("elf_flow does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("elf_flow does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("elf_flow does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("elf_flow does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("elf_flow supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {
        "elf_flow",
        "elf",
        "embedded_language_flow",
        "embedded-language-flow",
    }:
        raise ValueError(f"ELF Flow does not support model_type={config.model_type!r}")
    resolved = resolve_elf_config(config)
    max_length = int(resolved["max_length"])
    if request.max_sequence_length is not None and request.max_sequence_length != max_length:
        raise ValueError(
            f"ELF Flow max_sequence_length must equal the checkpoint max_length ({max_length})"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError("ELF Flow precision must be fp32 or fp16")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("ELF Flow does not support tensor-parallel builds")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("ELF Flow does not support quantized builds")
    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    encoder_checkpoint = _resolve_encoder_checkpoint(model_dir)

    config.raw["_fp32_layers"] = list(request.fp32_layers)
    model = _ElfFlowModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    weights["_elf_encoder_checkpoint"] = str(encoder_checkpoint)
    plan = model.build_engine(
        config,
        weights,
        max_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
    )
    extra = model.build_extra_engines(
        config,
        weights,
        max_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    text_encoder = (extra or {}).get("text_encoder.plan")
    if text_encoder is None:
        raise FileNotFoundError("ELF Flow requires its family-owned T5 encoder checkpoint")

    writer.set_header(family="elf_flow", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes("text_encoder.plan", text_encoder)
    writer.add_json("runtime.json", model.get_bundle_config_overrides(config))
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if not path.is_file():
            path = tokenizer_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

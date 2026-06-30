"""ELF flow family plugin.

ELF is implemented from the GitHub source at https://github.com/lillian039/ELF.
The weight names below mirror the Flax module tree in ``src/modules/model.py``.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .weights import WeightDict
from .model.components.config import ModelConfig
from .config import resolve_elf_config


class _TensorStore:
    def __init__(self, model_dir: str | Path):
        from .weights import _has_tensor, _load_tensor, _open_safetensors

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


def _load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray] | None:
    if path.suffix == ".npz":
        arrays = _load_npz_arrays(path)
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


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision in ("fp16", "bf16") else np.float32


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


class ELFPlugin:
    name = "elf_flow"
    runtime_strategy = "elf_flow"

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt in ("elf", "embedded_language_flow", "embedded-language-flow")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        cfg = resolve_elf_config(config)
        store = _TensorStore(model_dir)
        target_dtype = _target_np_dtype(precision)
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
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del quant_ctx
        from .model.model import build_elf_flow_engine

        return build_elf_flow_engine(
            config, weights, max_cache_length, precision=precision, verbose=verbose
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict | None:
        del max_cache_length
        encoder_checkpoint = weights.get("_elf_encoder_checkpoint")
        if not encoder_checkpoint:
            return {}

        from ...build_timing import timed_trt_compile, timed_weight_loading
        from .model.components.text_encoder import (
            build_t5_encoder_engine,
            load_jax_t5_encoder_weights,
        )

        cfg = resolve_elf_config(config)
        with timed_weight_loading(build_timing, "elf_t5_encoder"):
            t5_weights = load_jax_t5_encoder_weights(
                str(encoder_checkpoint), precision=precision, num_layers=6
            )
        with timed_trt_compile(build_timing, "elf_t5_encoder"):
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
        return {"elf_text_encoder_plan": t5_plan}

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = resolve_elf_config(config)
        raw = config.raw or {}
        return {
            "runtime_strategy": self.runtime_strategy,
            "model_type": "elf",
            "hidden_size": cfg["hidden_size"],
            "num_hidden_layers": cfg["depth"],
            "num_attention_heads": cfg["num_heads"],
            "head_dim": cfg["head_dim"],
            "max_position_embeddings": cfg["max_length"],
            "vocab_size": cfg["vocab_size"],
            "elf_variant": cfg["variant"],
            "elf_max_length": cfg["max_length"],
            "elf_max_input_length": cfg["max_input_length"],
            "elf_text_encoder_dim": cfg["text_encoder_dim"],
            "elf_input_dim": cfg["input_dim"],
            "elf_bottleneck_dim": cfg["bottleneck_dim"],
            "elf_num_time_tokens": cfg["num_time_tokens"],
            "elf_num_self_cond_cfg_tokens": cfg["num_self_cond_cfg_tokens"],
            "elf_num_model_mode_tokens": cfg["num_model_mode_tokens"],
            "elf_denoiser_noise_scale": cfg["denoiser_noise_scale"],
            "elf_denoiser_p_mean": cfg["denoiser_p_mean"],
            "elf_denoiser_p_std": cfg["denoiser_p_std"],
            "elf_t_eps": cfg["t_eps"],
            "elf_latent_mean": float(raw.get("latent_mean", 0.0)),
            "elf_latent_std": float(raw.get("latent_std", 0.2)),
            "elf_encoder_model_name": raw.get("encoder_model_name", "t5-small"),
            "elf_encoder_max_length": cfg["max_length"],
            "elf_encoder_pad_token_id": _elf_encoder_pad_token_id(config),
            "elf_has_text_encoder": int(bool(raw.get("_elf_encoder_checkpoint"))),
            "elf_runtime_contract": "api_path_denoise_or_decode_logits",
            "elf_user_contract": "diffusion_text_generation",
            "elf_output_schema": "jsonl_id_generated_after_sampler_decode",
        }


plugin = ELFPlugin()

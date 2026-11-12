# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chroma checkpoint layout: validated configuration and lazy safetensors weights."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

_DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Chroma checkpoint is missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require(config: dict, key: str, expected, where: str) -> None:
    if config.get(key) != expected:
        raise ValueError(f"{where}: {key}={config.get(key)!r} is not supported (expected {expected!r})")


@dataclass(frozen=True)
class TransformerConfig:
    num_layers: int
    num_single_layers: int
    num_heads: int
    head_dim: int
    in_channels: int
    joint_attention_dim: int
    axes_dims_rope: tuple[int, int, int]
    approximator_channels: int
    approximator_hidden: int
    approximator_layers: int

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def modulation_rows(self) -> int:
        return 3 * self.num_single_layers + 12 * self.num_layers + 2


@dataclass(frozen=True)
class TextEncoderConfig:
    d_model: int
    d_kv: int
    d_ff: int
    num_heads: int
    num_layers: int
    vocab_size: int
    num_buckets: int
    max_distance: int
    eps: float


@dataclass(frozen=True)
class VaeConfig:
    latent_channels: int
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    groups: int
    scaling_factor: float
    shift_factor: float


@dataclass(frozen=True)
class SchedulerConfig:
    shift: float
    num_train_timesteps: int


@dataclass(frozen=True)
class ChromaCheckpoint:
    root: Path
    transformer: TransformerConfig
    text_encoder: TextEncoderConfig
    vae: VaeConfig
    scheduler: SchedulerConfig

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.vae.block_out_channels) - 1)


def load_checkpoint(root: Path) -> ChromaCheckpoint:
    """Read and validate every configuration file the family consumes."""
    root = Path(root)
    index = _read_json(root / "model_index.json")
    _require(index, "_class_name", "ChromaPipeline", "model_index.json")
    for component, expected in (
        ("transformer", ["diffusers", "ChromaTransformer2DModel"]),
        ("text_encoder", ["transformers", "T5EncoderModel"]),
        ("vae", ["diffusers", "AutoencoderKL"]),
        ("scheduler", ["diffusers", "FlowMatchEulerDiscreteScheduler"]),
    ):
        _require(index, component, expected, "model_index.json")
    for optional in ("image_encoder", "feature_extractor"):
        if index.get(optional) not in (None, [None, None]):
            raise ValueError(f"model_index.json: {optional} (IP-Adapter) is not supported")
    if index.get("tokenizer", [None, None])[1] not in ("T5Tokenizer", "T5TokenizerFast"):
        raise ValueError("model_index.json: the tokenizer must be a T5 tokenizer")

    t = _read_json(root / "transformer" / "config.json")
    _require(t, "patch_size", 1, "transformer")
    if t.get("out_channels") not in (None, t.get("in_channels")):
        raise ValueError("transformer: out_channels must equal in_channels")
    axes = tuple(int(value) for value in t.get("axes_dims_rope", ()))
    if len(axes) != 3 or sum(axes) != int(t["attention_head_dim"]):
        raise ValueError("transformer: axes_dims_rope must have three axes summing to attention_head_dim")
    if int(t["approximator_num_channels"]) != 64:
        raise ValueError("transformer: approximator_num_channels must be 64 (16 + 16 + 32 sinusoid channels)")
    transformer = TransformerConfig(
        num_layers=int(t["num_layers"]),
        num_single_layers=int(t["num_single_layers"]),
        num_heads=int(t["num_attention_heads"]),
        head_dim=int(t["attention_head_dim"]),
        in_channels=int(t["in_channels"]),
        joint_attention_dim=int(t["joint_attention_dim"]),
        axes_dims_rope=axes,  # type: ignore[arg-type]
        approximator_channels=int(t["approximator_num_channels"]),
        approximator_hidden=int(t["approximator_hidden_dim"]),
        approximator_layers=int(t["approximator_layers"]),
    )

    e = _read_json(root / "text_encoder" / "config.json")
    _require(e, "model_type", "t5", "text_encoder")
    _require(e, "feed_forward_proj", "gated-gelu", "text_encoder")
    _require(e, "dense_act_fn", "gelu_new", "text_encoder")
    text_encoder = TextEncoderConfig(
        d_model=int(e["d_model"]),
        d_kv=int(e["d_kv"]),
        d_ff=int(e["d_ff"]),
        num_heads=int(e["num_heads"]),
        num_layers=int(e["num_layers"]),
        vocab_size=int(e["vocab_size"]),
        num_buckets=int(e["relative_attention_num_buckets"]),
        max_distance=int(e["relative_attention_max_distance"]),
        eps=float(e["layer_norm_epsilon"]),
    )
    if text_encoder.d_model != transformer.joint_attention_dim:
        raise ValueError("text encoder width must equal the transformer joint_attention_dim")

    v = _read_json(root / "vae" / "config.json")
    _require(v, "_class_name", "AutoencoderKL", "vae")
    _require(v, "use_post_quant_conv", False, "vae")
    _require(v, "mid_block_add_attention", True, "vae")
    _require(v, "act_fn", "silu", "vae")
    if any(kind != "UpDecoderBlock2D" for kind in v.get("up_block_types", ())):
        raise ValueError("vae: only UpDecoderBlock2D decoder blocks are supported")
    if v.get("shift_factor") is None or v.get("scaling_factor") is None:
        raise ValueError("vae: scaling_factor and shift_factor are required")
    if v.get("latents_mean") is not None or v.get("latents_std") is not None:
        raise ValueError("vae: per-channel latent statistics are not supported")
    vae = VaeConfig(
        latent_channels=int(v["latent_channels"]),
        block_out_channels=tuple(int(value) for value in v["block_out_channels"]),
        layers_per_block=int(v["layers_per_block"]),
        groups=int(v["norm_num_groups"]),
        scaling_factor=float(v["scaling_factor"]),
        shift_factor=float(v["shift_factor"]),
    )
    if transformer.in_channels != 4 * vae.latent_channels:
        raise ValueError("transformer in_channels must be 4x the VAE latent channels (2x2 packing)")

    s = _read_json(root / "scheduler" / "scheduler_config.json")
    _require(s, "_class_name", "FlowMatchEulerDiscreteScheduler", "scheduler")
    for flag in (
        "use_dynamic_shifting",
        "invert_sigmas",
        "stochastic_sampling",
        "use_karras_sigmas",
        "use_exponential_sigmas",
        "use_beta_sigmas",
    ):
        if s.get(flag):
            raise ValueError(f"scheduler: {flag} is not supported")
    if s.get("shift_terminal"):
        raise ValueError("scheduler: shift_terminal is not supported")
    num_train_timesteps = int(s["num_train_timesteps"])
    if num_train_timesteps != 1000:
        # The runtime's timestep conditioning (t / 1000 * 1000 in the bundle's precision) assumes this value.
        raise ValueError(f"scheduler: num_train_timesteps must be 1000, got {num_train_timesteps}")
    scheduler = SchedulerConfig(shift=float(s.get("shift", 1.0)), num_train_timesteps=num_train_timesteps)

    return ChromaCheckpoint(root, transformer, text_encoder, vae, scheduler)


class SafetensorsDirectory:
    """Lazy, memory-mapped access to one component's (possibly sharded) safetensors."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        files = sorted(self.directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors files in {self.directory}")
        self._index: dict[str, tuple[Path, int, dict]] = {}
        for path in files:
            with path.open("rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            for name, entry in header.items():
                if name in self._index:
                    raise ValueError(f"tensor {name} appears in more than one shard")
                self._index[name] = (path, 8 + length, entry)
        self._maps: dict[Path, np.memmap] = {}

    def names(self) -> set[str]:
        return set(self._index)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __getitem__(self, name: str) -> np.ndarray:
        """The tensor in its stored dtype (BF16 as ``ml_dtypes.bfloat16``)."""
        if name not in self._index:
            raise KeyError(f"{self.directory.name}: missing tensor {name}")
        path, offset, entry = self._index[name]
        dtype = _DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{name}: unsupported stored dtype {entry['dtype']}")
        mapped = self._maps.get(path)
        if mapped is None:
            mapped = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = mapped
        begin, end = entry["data_offsets"]
        raw = mapped[offset + begin: offset + end]
        return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])

    def float32(self, name: str) -> np.ndarray:
        return self[name].astype(np.float32)

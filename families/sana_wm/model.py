# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM family plugin.

The public SANA-WM release ships a Sana-specific config.yaml plus DiT, LTX-2
VAE, and refiner weights.
"""

from __future__ import annotations

import json
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import WeightDict
from .config import ModelConfig

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


@contextmanager
def timed_build_phase(timing: dict | None, name: str):
    """Record one family-owned build phase when a caller requests it."""
    start = time.monotonic()
    yield
    if timing is not None:
        timing.setdefault("phases", {})[name] = time.monotonic() - start


_DEFAULT_TRANSLATION_SPEED = 0.055
_DEFAULT_ROTATION_SPEED_DEG = 1.2
_DEFAULT_NUM_FRAMES = 321
_DEFAULT_HEIGHT = 704
_DEFAULT_WIDTH = 1280
_DEFAULT_FPS = 16
_DEFAULT_NUM_STEPS = 60
_DEFAULT_GUIDANCE_SCALE = 5.0
_DEFAULT_VAE_STRIDE = (8, 32, 32)
_DEFAULT_REFINER_TEXT_MAX_LENGTH = 1024
_STAGE1_DIT_REL = Path("dit") / "sana_wm_1600m_720p.safetensors"
_STAGE1_TEXT_ENCODER_REL = Path("stage1_text_encoder")
_REFINER_TRANSFORMER_REL = Path("refiner") / "transformer" / "diffusion_pytorch_model.safetensors"
_REFINER_CONNECTORS_REL = Path("refiner") / "connectors" / "diffusion_pytorch_model.safetensors"
_REFINER_GEMMA_REL = Path("refiner") / "text_encoder"
_REQUIRED_MODEL_PATHS = (
    _STAGE1_DIT_REL,
    _STAGE1_TEXT_ENCODER_REL / "config.json",
    _STAGE1_TEXT_ENCODER_REL / "model.safetensors.index.json",
    _STAGE1_TEXT_ENCODER_REL / "tokenizer.json",
    Path("vae/config.json"),
    Path("vae/diffusion_pytorch_model.safetensors"),
    _REFINER_TRANSFORMER_REL,
    Path("refiner/transformer/config.json"),
    _REFINER_CONNECTORS_REL,
    Path("refiner/connectors/config.json"),
    _REFINER_GEMMA_REL / "config.json",
    _REFINER_GEMMA_REL / "model.safetensors.index.json",
    _REFINER_GEMMA_REL / "tokenizer.json",
)
_LTX2_VAE_PLAN_PRECISION = "bf16"
_LTX2_VAE_ENCODER_PLAN_PRECISION = "bf16"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)
_TOKENIZER_SECTION_SUFFIXES = {
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "vocab.json": "vocab.json",
    "merges.txt": "merges.txt",
    "special_tokens_map.json": "special_tokens_map.json",
    "tokenizer.model": "tokenizer.model",
}
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


def _vae_stride(raw_vae: dict, raw_config: dict) -> tuple[int, int, int]:
    stride = raw_vae.get("vae_stride", raw_config.get("vae_stride", _DEFAULT_VAE_STRIDE))
    if not isinstance(stride, (list, tuple)) or len(stride) == 0:
        return _DEFAULT_VAE_STRIDE
    values = [int(v) for v in stride]
    if len(values) == 1:
        values = [values[0], values[0], values[0]]
    if len(values) == 2:
        values = [values[0], values[1], values[1]]
    return values[0], values[1], values[2]


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"{path} has an invalid safetensors header size: {header_len}")
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path} ended before its safetensors header was complete")
    return json.loads(header)


def _tensor_shape(header: dict, name: str) -> list[int]:
    entry = header.get(name)
    if not isinstance(entry, dict) or "shape" not in entry:
        raise ValueError(f"SANA-WM DiT safetensors missing tensor {name!r}")
    return [int(v) for v in entry["shape"]]


def _block_count(header: dict) -> int:
    block_ids = set()
    for name in header:
        parts = name.split(".", 2)
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit():
            block_ids.add(int(parts[1]))
    return max(block_ids) + 1 if block_ids else 0


def _summarize_stage1_dit(path: Path) -> dict:
    header = _read_safetensors_header(path)
    tensor_count = sum(1 for name in header if name != "__metadata__")
    hidden_size, latent_channels, _, _, _ = _tensor_shape(header, "x_embedder.proj.weight")
    text_hidden, text_dim = _tensor_shape(header, "y_embedder.y_proj.fc1.weight")
    text_length, y_dim = _tensor_shape(header, "y_embedder.y_embedding")
    out_channels, out_hidden = _tensor_shape(header, "final_layer.linear.weight")
    plucker_hidden, chunk_plucker_channels, _, _, _ = _tensor_shape(
        header, "plucker_embedder.proj.weight"
    )
    raymap_hidden, raymap_channels, _, _, _ = _tensor_shape(header, "raymap_embedder.proj.weight")
    qkv_rows, qkv_cols = _tensor_shape(header, "blocks.0.attn.qkv.weight")

    if not (
        hidden_size == text_hidden == out_hidden == plucker_hidden == raymap_hidden == qkv_cols
    ):
        raise ValueError("SANA-WM DiT metadata has inconsistent hidden-size dimensions")
    if text_dim != y_dim:
        raise ValueError("SANA-WM DiT metadata has inconsistent text embedding dimensions")
    if out_channels != latent_channels:
        raise ValueError("SANA-WM DiT metadata has inconsistent latent channel dimensions")
    if qkv_rows != hidden_size * 3:
        raise ValueError("SANA-WM DiT qkv tensor does not match 3x hidden size")

    return {
        "tensor_count": tensor_count,
        "num_layers": _block_count(header),
        "hidden_size": hidden_size,
        "latent_channels": latent_channels,
        "text_max_length": text_length,
        "text_embed_dim": text_dim,
        "chunk_plucker_channels": chunk_plucker_channels,
        "raymap_channels": raymap_channels,
    }


def _load_vae_config(vae_dir: Path) -> dict:
    return _load_json_config(vae_dir / "config.json")


def _load_refiner_transformer_config(transformer_dir: Path | None) -> dict:
    if transformer_dir is None:
        raise FileNotFoundError("SANA-WM refiner/transformer is missing")
    return _load_json_config(transformer_dir / "config.json")


def _load_json_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"SANA-WM config must be a JSON object: {path}")
    return parsed


def _int_tuple(value, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    return tuple(int(v) for v in value)


def _bool_tuple(value, fallback: tuple[bool, ...]) -> tuple[bool, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    return tuple(bool(v) for v in value)


def _str_tuple(value, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    return tuple(str(v) for v in value)


def _join_chi_prompt(text_encoder: dict) -> str:
    chi_prompt = text_encoder.get("chi_prompt", [])
    if isinstance(chi_prompt, str):
        return chi_prompt
    if isinstance(chi_prompt, (list, tuple)):
        return "\n".join(str(line) for line in chi_prompt)
    return ""


def _stage1_chi_prompt_token_count(text_encoder_dir: Path, chi_prompt: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, local_files_only=True)
    return len(tokenizer.encode(chi_prompt))


def _tokenizer_adds_special_tokens(tokenizer_dir: Path) -> bool:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    return tokenizer.encode("hello") != tokenizer.encode("hello", add_special_tokens=False)


def _stage1_text_encoder_conditioning_length(
    text_encoder_dir: Path,
    raw_config: dict,
) -> int:
    text_encoder = raw_config.get("text_encoder", {})
    if not isinstance(text_encoder, dict):
        text_encoder = {}

    model_max_length = int(
        text_encoder.get(
            "model_max_length",
            raw_config.get("text_encoder_max_length", 300),
        )
    )
    chi_prompt = raw_config.get("sana_wm_chi_prompt", _join_chi_prompt(text_encoder))
    if not chi_prompt:
        return model_max_length

    chi_tokens = _stage1_chi_prompt_token_count(text_encoder_dir, str(chi_prompt))
    return max(model_max_length, model_max_length + max(0, chi_tokens - 2))


def _discover_tokenizer_sections(
    model_path: Path,
) -> dict[str, Path]:
    sections: dict[str, Path] = {}
    for prefix, directory in (
        ("sana_wm_stage1", model_path / _STAGE1_TEXT_ENCODER_REL),
        ("sana_wm_refiner", model_path / _REFINER_GEMMA_REL),
    ):
        tokenizer = directory / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(tokenizer)
        for name in _TOKENIZER_FILES:
            path = directory / name
            if path.is_file():
                sections[f"{prefix}_{_TOKENIZER_SECTION_SUFFIXES[name]}"] = path
    return sections


def _validate_native_tokenizer_sections(
    tokenizer_sections: dict[str, Path], *, require_refiner: bool
) -> None:
    if "sana_wm_stage1_tokenizer.json" not in tokenizer_sections:
        raise ValueError("SANA-WM requires stage1_text_encoder/tokenizer.json")
    if require_refiner and "sana_wm_refiner_tokenizer.json" not in tokenizer_sections:
        raise ValueError("SANA-WM requires refiner/text_encoder/tokenizer.json")


def _missing_prepared_model_paths(model_path: Path) -> list[str]:
    return [path.as_posix() for path in _REQUIRED_MODEL_PATHS if not (model_path / path).is_file()]


def _pre_scale_gemma_embedding_bf16(weights: WeightDict, hidden_size: int) -> None:
    import ml_dtypes

    embedding = np.asarray(weights["embedding"])
    embedding_bf16 = embedding.astype(ml_dtypes.bfloat16)
    scale_bf16 = np.asarray(np.sqrt(hidden_size), dtype=ml_dtypes.bfloat16)
    scaled_bf16 = (embedding_bf16 * scale_bf16).astype(ml_dtypes.bfloat16)
    weights["embedding"] = scaled_bf16.astype(np.float32).astype(embedding.dtype)
    weights.pop("_embedding_scale", None)


def _make_exact_gemma_rope_tables(
    text_config: ModelConfig,
    table_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SANA-WM exact Gemma RoPE table generation requires CUDA")
    head_dim = int(
        text_config.raw.get(
            "head_dim", text_config.attention_size // text_config.num_attention_heads
        )
    )
    rope_theta = float(text_config.rope_theta)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    position_ids = torch.arange(table_length, device="cuda", dtype=torch.long).unsqueeze(0)
    inv_freq_expanded = inv_freq.cuda()[None, :, None].float()
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type="cuda", enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.bfloat16).float().cpu().numpy()[0]
        sin = emb.sin().to(torch.bfloat16).float().cpu().numpy()[0]
    # Every BF16 value is exactly representable in fp16 in [-1, 1], which is
    # the storage dtype used by the generic BF16 TensorRT builder.
    return cos.astype(np.float16), sin.astype(np.float16)


def _make_exact_gemma3_rope_tables(
    text_encoder_dir: Path,
    table_length: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

    if not torch.cuda.is_available():
        raise RuntimeError("SANA-WM exact Gemma3 RoPE table generation requires CUDA")
    config = AutoConfig.from_pretrained(
        text_encoder_dir,
        local_files_only=True,
    ).text_config
    rotary = Gemma3RotaryEmbedding(config)
    positions = torch.arange(table_length, device="cuda", dtype=torch.long).unsqueeze(0)
    probe = torch.empty((1, 1, config.head_dim), device="cuda", dtype=torch.bfloat16)
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for layer_type in set(config.layer_types):
        cos, sin = rotary(probe, positions, layer_type)
        tables[layer_type] = (
            cos.float().cpu().numpy()[0].astype(np.float16),
            sin.float().cpu().numpy()[0].astype(np.float16),
        )
    return tables


def _prepare_exact_gemma_text_weights(
    text_config: ModelConfig,
    text_weights: WeightDict,
    max_cache_length: int,
    text_encoder_dir: Path | None = None,
) -> None:
    _pre_scale_gemma_embedding_bf16(text_weights, text_config.hidden_size)
    text_weights["_sana_wm_exact_gemma"] = True
    nested_text_config = text_config.raw.get("text_config", {})
    layer_types = nested_text_config.get("layer_types", [])
    if text_config.model_type.startswith("gemma3") and layer_types:
        if text_encoder_dir is None:
            raise ValueError("Gemma3 exact RoPE generation requires the text encoder directory")
        tables = _make_exact_gemma3_rope_tables(
            text_encoder_dir,
            2 * max_cache_length,
        )
        text_weights["_sana_wm_rope_tables"] = tables
        text_weights["_sana_wm_layer_types"] = list(layer_types[: text_config.num_hidden_layers])
    else:
        cos, sin = _make_exact_gemma_rope_tables(text_config, 2 * max_cache_length)
        text_weights["_sana_wm_rope_cos"] = cos
        text_weights["_sana_wm_rope_sin"] = sin


def _build_gemma_text_encoder_plan(
    text_encoder_dir: Path,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    label: str,
    debug_layer_outputs: bool = False,
) -> bytes:
    from .components.gemma.plugin import plugin as gemma_plugin
    from .components.gemma.standard_decoder_builder import build_standard_decoder_engine

    text_config = ModelConfig.from_dir(text_encoder_dir)
    if not gemma_plugin.matches(text_config.model_type):
        raise ValueError(
            f"SANA-WM {label} text encoder builder currently supports Gemma only; "
            f"found model_type={text_config.model_type!r} in {text_encoder_dir}"
        )
    load_precision = "fp32" if precision == "bf16" else precision
    text_weights = gemma_plugin.load_weights(
        str(text_encoder_dir),
        text_config,
        precision=load_precision,
    )
    if precision == "bf16":
        from .stage1_dit_builder import _get_sana_wm_plugin_creator

        trt_module = trt
        for plugin_name in (
            "SanaWmGemmaRmsNorm",
            "SanaWmGemmaGatedGelu",
            "SanaWmGemmaRope",
            "SanaWmGemmaAttention",
        ):
            if _get_sana_wm_plugin_creator(trt_module, plugin_name) is None:
                raise RuntimeError(
                    f"{plugin_name} is required for exact SANA-WM Gemma text encoding"
                )
        _prepare_exact_gemma_text_weights(
            text_config,
            text_weights,
            max_cache_length,
            text_encoder_dir,
        )
    text_config.raw["_decoder_engine_role"] = "decode"
    return build_standard_decoder_engine(
        text_config,
        text_weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        debug_layer_outputs=debug_layer_outputs,
        hidden_state_output=True,
    )


def _build_stage1_text_encoder_plan(
    text_encoder_dir: Path,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    return _build_gemma_text_encoder_plan(
        text_encoder_dir,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        label="stage-1",
    )


def _build_refiner_text_encoder_plan(
    text_encoder_dir: Path,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    return _build_gemma_text_encoder_plan(
        text_encoder_dir,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        label="refiner",
        debug_layer_outputs=True,
    )


def _build_sana_wm_stage1_denoiser_plan(
    dit_path: Path,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from .stage1_dit_builder import (
        build_sana_wm_stage1_dit_engine,
        load_sana_wm_stage1_dit_weights,
    )

    weights = load_sana_wm_stage1_dit_weights(dit_path, precision=precision)
    return build_sana_wm_stage1_dit_engine(
        weights,
        raw_config,
        precision=precision,
        verbose=verbose,
    )


def _build_sana_wm_refiner_denoiser_plan(
    transformer_dir: Path,
    raw_config: dict,
    transformer_config: dict | None = None,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from .refiner_dit_builder import (
        build_sana_wm_refiner_dit_engine,
        load_sana_wm_refiner_dit_weights,
    )

    cfg = transformer_config or _load_refiner_transformer_config(transformer_dir)
    weights = load_sana_wm_refiner_dit_weights(
        transformer_dir,
        num_layers=int(cfg.get("num_layers", 48)),
        precision=precision,
    )
    return build_sana_wm_refiner_dit_engine(
        weights,
        raw_config,
        cfg,
        precision=precision,
        verbose=verbose,
    )


def _build_sana_wm_refiner_text_connector_plan(
    connectors_dir: Path,
    raw_config: dict,
    connector_config: dict | None = None,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    from .refiner_text_connector_builder import (
        build_sana_wm_refiner_text_connector_engine,
        load_sana_wm_refiner_text_connector_weights,
        refiner_text_connector_shape_from_config,
    )

    cfg = connector_config or _load_json_config(connectors_dir / "config.json")
    shape = refiner_text_connector_shape_from_config(raw_config, cfg)
    weights = load_sana_wm_refiner_text_connector_weights(
        connectors_dir,
        num_layers=shape.num_layers,
        precision=precision,
    )
    return build_sana_wm_refiner_text_connector_engine(
        weights,
        raw_config,
        cfg,
        precision=precision,
        verbose=verbose,
    )


def _build_sana_wm_vae_encoder_plan(
    vae_dir: Path,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from .components.ltx_video.ltx_vae_builder import (
        build_ltx_vae_encoder_engine,
        load_ltx_vae_encoder_weights,
    )

    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    vae_config = _load_vae_config(vae_dir)
    video_height = int(raw_config.get("video_height", _DEFAULT_HEIGHT))
    video_width = int(raw_config.get("video_width", _DEFAULT_WIDTH))
    spatial_tiling = video_height > 512 or video_width > 512
    weights = load_ltx_vae_encoder_weights(vae_dir, precision=precision)
    return build_ltx_vae_encoder_engine(
        weights,
        sample_frames=1,
        sample_height=video_height,
        sample_width=video_width,
        in_channels=int(vae_config.get("in_channels", 3)),
        latent_channels=int(vae_config.get("latent_channels", vae.get("vae_latent_dim", 128))),
        block_out_channels=_int_tuple(vae_config.get("block_out_channels"), (128, 256, 512, 512)),
        layers_per_block=_int_tuple(vae_config.get("layers_per_block"), (4, 3, 3, 3, 4)),
        spatio_temporal_scaling=_bool_tuple(
            vae_config.get("spatio_temporal_scaling"), (True, True, True, False)
        ),
        downsample_type=_str_tuple(
            vae_config.get("downsample_type"), ("conv", "conv", "conv", "conv")
        ),
        patch_size=int(vae_config.get("patch_size", 4)),
        patch_size_t=int(vae_config.get("patch_size_t", 1)),
        precision=precision,
        normalize_output=True,
        scaling_factor=float(vae_config.get("scaling_factor", 1.0)),
        spatial_tiling=spatial_tiling,
        tile_sample_min_height=int(vae.get("tile_sample_min_height", 512)),
        tile_sample_min_width=int(vae.get("tile_sample_min_width", 512)),
        tile_sample_stride_height=int(vae.get("tile_sample_stride_height", 448)),
        tile_sample_stride_width=int(vae.get("tile_sample_stride_width", 448)),
        use_torch_conv3d=precision == "bf16",
        verbose=verbose,
    )


def _sana_wm_vae_tile_section_name(
    prefix: str,
    frames: int,
    height: int,
    width: int,
) -> str:
    return f"{prefix}_tile_t{frames}_h{height}_w{width}_plan"


def _sana_wm_vae_tile_shapes(raw_config: dict) -> list[tuple[int, int, int]]:
    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    if not (
        vae.get("use_framewise_decoding") is True
        or "tile_sample_min_num_frames" in vae
        or "tile_sample_stride_num_frames" in vae
    ):
        return []

    video_height = int(raw_config.get("video_height", _DEFAULT_HEIGHT))
    video_width = int(raw_config.get("video_width", _DEFAULT_WIDTH))
    video_num_frames = int(raw_config.get("video_num_frames", _DEFAULT_NUM_FRAMES))
    vae_stride = _vae_stride(vae, raw_config)
    latent_frames = (video_num_frames - 1) // vae_stride[0] + 1
    latent_height = video_height // vae_stride[-1]
    latent_width = video_width // vae_stride[-1]

    tile_latent_min_frames = max(1, int(vae.get("tile_sample_min_num_frames", 96)) // vae_stride[0])
    tile_latent_stride_frames = max(
        1, int(vae.get("tile_sample_stride_num_frames", 64)) // vae_stride[0]
    )
    tile_latent_min_height = max(1, int(vae.get("tile_sample_min_height", 512)) // vae_stride[-1])
    tile_latent_min_width = max(1, int(vae.get("tile_sample_min_width", 512)) // vae_stride[-1])
    tile_latent_stride_height = max(
        1, int(vae.get("tile_sample_stride_height", 448)) // vae_stride[-1]
    )
    tile_latent_stride_width = max(
        1, int(vae.get("tile_sample_stride_width", 448)) // vae_stride[-1]
    )

    temporal_tiles: list[int] = []
    if bool(vae.get("use_framewise_decoding", True)) and latent_frames > tile_latent_min_frames:
        for start in range(0, latent_frames, tile_latent_stride_frames):
            frames = min(tile_latent_min_frames + 1, latent_frames - start)
            if start > 0 and frames <= 1:
                continue
            temporal_tiles.append(frames)
    else:
        temporal_tiles.append(latent_frames)

    height_tiles: list[int] = []
    width_tiles: list[int] = []
    if bool(vae.get("use_tiling", True)) and (
        latent_height > tile_latent_min_height or latent_width > tile_latent_min_width
    ):
        for start in range(0, latent_height, tile_latent_stride_height):
            height_tiles.append(min(tile_latent_min_height, latent_height - start))
        for start in range(0, latent_width, tile_latent_stride_width):
            width_tiles.append(min(tile_latent_min_width, latent_width - start))
    else:
        height_tiles.append(latent_height)
        width_tiles.append(latent_width)

    return sorted(
        {
            (frames, height, width)
            for frames in temporal_tiles
            for height in height_tiles
            for width in width_tiles
        }
    )


def _build_sana_wm_vae_decoder_plan(
    vae_dir: Path,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from .components.ltx_video.ltx_vae_builder import (
        build_ltx_vae_decoder_engine,
        load_ltx_vae_weights,
    )

    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    video_height = int(raw_config.get("video_height", _DEFAULT_HEIGHT))
    video_width = int(raw_config.get("video_width", _DEFAULT_WIDTH))
    video_num_frames = int(raw_config.get("video_num_frames", _DEFAULT_NUM_FRAMES))
    vae_config = _load_vae_config(vae_dir)
    vae_stride = _vae_stride(vae, raw_config)
    latent_frames = (video_num_frames - 1) // vae_stride[0] + 1
    latent_height = video_height // vae_stride[-1]
    latent_width = video_width // vae_stride[-1]
    weights = load_ltx_vae_weights(vae_dir, precision=precision)
    return build_ltx_vae_decoder_engine(
        weights,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=int(
            vae_config.get(
                "latent_channels",
                vae.get("vae_latent_dim", raw_config.get("vae_latent_dim", 128)),
            )
        ),
        block_out_channels=_int_tuple(
            vae_config.get("decoder_block_out_channels") or vae_config.get("block_out_channels"),
            (128, 256, 512, 512),
        ),
        layers_per_block=_int_tuple(
            vae_config.get("decoder_layers_per_block") or vae_config.get("layers_per_block"),
            (4, 3, 3, 3, 4),
        ),
        spatio_temporal_scaling=_bool_tuple(
            vae_config.get("decoder_spatio_temporal_scaling")
            or vae_config.get("spatio_temporal_scaling"),
            (True, True, True, False),
        ),
        upsample_type=_str_tuple(
            vae_config.get("upsample_type"),
            ("spatiotemporal", "spatiotemporal", "spatiotemporal"),
        ),
        upsample_factor=_int_tuple(vae_config.get("upsample_factor"), (1, 1, 1)),
        upsample_residual=_bool_tuple(vae_config.get("upsample_residual"), (False, False, False)),
        patch_size=int(vae_config.get("patch_size", 4)),
        patch_size_t=int(vae_config.get("patch_size_t", 1)),
        out_channels=int(vae_config.get("out_channels", 3)),
        precision=precision,
        denormalize_input=True,
        scaling_factor=float(vae_config.get("scaling_factor", 1.0)),
        spatial_padding_mode=str(vae_config.get("decoder_spatial_padding_mode", "zeros")),
        verbose=verbose,
    )


def _build_sana_wm_vae_decoder_tile_plans(
    vae_dir: Path,
    raw_config: dict,
    *,
    section_prefix: str,
    precision: str = "fp16",
    verbose: bool = False,
) -> dict[str, bytes]:
    shapes = _sana_wm_vae_tile_shapes(raw_config)
    if not shapes:
        return {}
    from .components.ltx_video.ltx_vae_builder import (
        build_ltx_vae_decoder_engine,
        load_ltx_vae_weights,
    )

    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    vae_config = _load_vae_config(vae_dir)
    weights = load_ltx_vae_weights(vae_dir, precision=precision)
    result: dict[str, bytes] = {}
    for latent_frames, latent_height, latent_width in shapes:
        result[
            _sana_wm_vae_tile_section_name(
                section_prefix, latent_frames, latent_height, latent_width
            )
        ] = build_ltx_vae_decoder_engine(
            weights,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_channels=int(
                vae_config.get(
                    "latent_channels",
                    vae.get("vae_latent_dim", raw_config.get("vae_latent_dim", 128)),
                )
            ),
            block_out_channels=_int_tuple(
                vae_config.get("decoder_block_out_channels")
                or vae_config.get("block_out_channels"),
                (128, 256, 512, 512),
            ),
            layers_per_block=_int_tuple(
                vae_config.get("decoder_layers_per_block") or vae_config.get("layers_per_block"),
                (4, 3, 3, 3, 4),
            ),
            spatio_temporal_scaling=_bool_tuple(
                vae_config.get("decoder_spatio_temporal_scaling")
                or vae_config.get("spatio_temporal_scaling"),
                (True, True, True, False),
            ),
            upsample_type=_str_tuple(
                vae_config.get("upsample_type"),
                ("spatiotemporal", "spatiotemporal", "spatiotemporal"),
            ),
            upsample_factor=_int_tuple(vae_config.get("upsample_factor"), (1, 1, 1)),
            upsample_residual=_bool_tuple(
                vae_config.get("upsample_residual"), (False, False, False)
            ),
            patch_size=int(vae_config.get("patch_size", 4)),
            patch_size_t=int(vae_config.get("patch_size_t", 1)),
            out_channels=int(vae_config.get("out_channels", 3)),
            precision=precision,
            denormalize_input=True,
            scaling_factor=float(vae_config.get("scaling_factor", 1.0)),
            spatial_padding_mode=str(vae_config.get("decoder_spatial_padding_mode", "zeros")),
            verbose=verbose,
        )
    return result


class _SanaWmModel:

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_path = Path(model_dir)
        weights = WeightDict()
        weights["_stage1_dit_path"] = str(model_path / _STAGE1_DIT_REL)

        missing = _missing_prepared_model_paths(model_path)
        if missing:
            raise FileNotFoundError(
                "SANA-WM prepared model directory is missing: " + ", ".join(missing)
            )
        stage1_path = model_path / _STAGE1_DIT_REL
        summary = _summarize_stage1_dit(stage1_path)
        weights["_stage1_dit_summary"] = summary
        config.raw["_sana_wm_stage1_dit_summary"] = summary
        stage1_text_encoder_dir = model_path / _STAGE1_TEXT_ENCODER_REL
        refiner_text_encoder_dir = model_path / _REFINER_GEMMA_REL
        refiner_transformer_dir = model_path / "refiner/transformer"
        refiner_connectors_dir = model_path / "refiner/connectors"
        vae_dir = model_path / "vae"
        tokenizer_sections = _discover_tokenizer_sections(model_path)
        stage1_tokenizer = tokenizer_sections.get("sana_wm_stage1_tokenizer.json")
        _validate_native_tokenizer_sections(tokenizer_sections, require_refiner=True)
        if stage1_tokenizer is None:
            raise FileNotFoundError("SANA-WM stage1_text_encoder/tokenizer.json is missing")
        config.raw["tokenizer_add_special_tokens"] = int(
            _tokenizer_adds_special_tokens(stage1_tokenizer.parent)
        )
        weights["_stage1_text_encoder_dir"] = str(stage1_text_encoder_dir)
        weights["_refiner_text_encoder_dir"] = str(refiner_text_encoder_dir)
        weights["_refiner_transformer_dir"] = str(refiner_transformer_dir)
        transformer_config = _load_refiner_transformer_config(refiner_transformer_dir)
        weights["_refiner_transformer_config"] = transformer_config
        config.raw["_sana_wm_refiner_transformer_config"] = transformer_config
        weights["_refiner_connectors_dir"] = str(refiner_connectors_dir)
        connector_config = _load_json_config(refiner_connectors_dir / "config.json")
        weights["_refiner_connectors_config"] = connector_config
        config.raw["_sana_wm_refiner_connectors_config"] = connector_config
        weights["_sana_wm_vae_encoder_dir"] = str(vae_dir)
        weights["_sana_wm_vae_decoder_dir"] = str(vae_dir)
        weights["_tokenizer_sections"] = {
            section: str(path) for section, path in tokenizer_sections.items()
        }
        return weights

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict:
        from .native_plugin_builder import load_native_plugin

        load_native_plugin(verbose=verbose)
        result = {}
        text_encoder_dir = Path(str(weights["_stage1_text_encoder_dir"]))
        stage1_text_length = _stage1_text_encoder_conditioning_length(
            text_encoder_dir,
            config.raw,
        )
        with timed_build_phase(build_timing, "build_extra_sana_wm_stage1_text_encoder_s"):
            result["text_encoder_0_plan"] = _build_stage1_text_encoder_plan(
                text_encoder_dir,
                stage1_text_length,
                precision=precision,
                verbose=verbose,
            )
        stage1_dit_path = Path(str(weights["_stage1_dit_path"]))
        with timed_build_phase(build_timing, "build_extra_sana_wm_stage1_denoiser_s"):
            result["denoiser_plan"] = _build_sana_wm_stage1_denoiser_plan(
                stage1_dit_path,
                config.raw,
                precision=precision,
                verbose=verbose,
            )
        refiner_text_encoder_dir = Path(str(weights["_refiner_text_encoder_dir"]))
        refiner_connectors_dir = Path(str(weights["_refiner_connectors_dir"]))
        with timed_build_phase(build_timing, "build_extra_sana_wm_refiner_text_connector_s"):
            result["sana_wm_refiner_text_connector_plan"] = (
                _build_sana_wm_refiner_text_connector_plan(
                    refiner_connectors_dir,
                    config.raw,
                    weights["_refiner_connectors_config"],
                    precision="bf16",
                    verbose=verbose,
                )
            )
        with timed_build_phase(build_timing, "build_extra_sana_wm_refiner_text_encoder_s"):
            result["sana_wm_refiner_text_encoder_plan"] = _build_refiner_text_encoder_plan(
                refiner_text_encoder_dir,
                max(
                    max_cache_length,
                    int(
                        config.raw.get(
                            "sana_wm_refiner_text_max_length",
                            _DEFAULT_REFINER_TEXT_MAX_LENGTH,
                        ),
                    ),
                ),
                precision=precision,
                verbose=verbose,
            )
        refiner_transformer_dir = Path(str(weights["_refiner_transformer_dir"]))
        with timed_build_phase(build_timing, "build_extra_sana_wm_refiner_denoiser_s"):
            result["sana_wm_refiner_denoiser_plan"] = _build_sana_wm_refiner_denoiser_plan(
                refiner_transformer_dir,
                config.raw,
                weights["_refiner_transformer_config"],
                precision=precision,
                verbose=verbose,
            )
        vae_encoder_dir = Path(str(weights["_sana_wm_vae_encoder_dir"]))
        with timed_build_phase(build_timing, "build_extra_sana_wm_vae_encoder_s"):
            result["sana_wm_vae_encoder_plan"] = _build_sana_wm_vae_encoder_plan(
                vae_encoder_dir,
                config.raw,
                precision=_LTX2_VAE_ENCODER_PLAN_PRECISION,
                verbose=verbose,
            )
        vae_decoder_dir = Path(str(weights["_sana_wm_vae_decoder_dir"]))
        with timed_build_phase(build_timing, "build_extra_sana_wm_vae_decoder_s"):
            result["vae_decoder_plan"] = _build_sana_wm_vae_decoder_plan(
                vae_decoder_dir,
                config.raw,
                precision=_LTX2_VAE_PLAN_PRECISION,
                verbose=verbose,
            )
        with timed_build_phase(build_timing, "build_extra_sana_wm_vae_decoder_tiles_s"):
            result.update(
                _build_sana_wm_vae_decoder_tile_plans(
                    vae_decoder_dir,
                    config.raw,
                    section_prefix="sana_wm_vae_decoder",
                    precision=_LTX2_VAE_PLAN_PRECISION,
                    verbose=verbose,
                )
            )
        result["sana_wm_refiner_vae_decoder_plan"] = result["vae_decoder_plan"]
        result.update(
            {
                section: Path(path).read_bytes()
                for section, path in weights["_tokenizer_sections"].items()
            }
        )
        return result

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        raw = config.raw
        text_encoder = raw.get("text_encoder", {})
        scheduler = raw.get("scheduler", {})
        vae = raw.get("vae", {})
        if not isinstance(text_encoder, dict):
            text_encoder = {}
        if not isinstance(scheduler, dict):
            scheduler = {}
        if not isinstance(vae, dict):
            vae = {}

        video_height = int(raw.get("video_height", _DEFAULT_HEIGHT))
        video_width = int(raw.get("video_width", _DEFAULT_WIDTH))
        video_num_frames = int(raw.get("video_num_frames", _DEFAULT_NUM_FRAMES))
        vae_stride = _vae_stride(vae, raw)

        return {
            "sana_wm_translation_speed": float(
                raw.get("sana_wm_translation_speed", _DEFAULT_TRANSLATION_SPEED)
            ),
            "sana_wm_rotation_speed_deg": float(
                raw.get("sana_wm_rotation_speed_deg", _DEFAULT_ROTATION_SPEED_DEG)
            ),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "fps": int(raw.get("fps", _DEFAULT_FPS)),
            "num_inference_steps": int(raw.get("num_inference_steps", _DEFAULT_NUM_STEPS)),
            "guidance_scale": float(raw.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE)),
            "vae_latent_dim": int(vae.get("vae_latent_dim", raw.get("vae_latent_dim", 128))),
            "vae_time_stride": int(vae_stride[0]),
            "vae_spatial_stride": int(vae_stride[-1]),
            "vae_use_framewise_decoding": bool(vae.get("use_framewise_decoding", True)),
            "vae_use_spatial_tiling": bool(vae.get("use_tiling", True)),
            "vae_tile_sample_min_height": int(vae.get("tile_sample_min_height", 512)),
            "vae_tile_sample_min_width": int(vae.get("tile_sample_min_width", 512)),
            "vae_tile_sample_stride_height": int(vae.get("tile_sample_stride_height", 448)),
            "vae_tile_sample_stride_width": int(vae.get("tile_sample_stride_width", 448)),
            "vae_tile_sample_min_num_frames": int(vae.get("tile_sample_min_num_frames", 96)),
            "vae_tile_sample_stride_num_frames": int(vae.get("tile_sample_stride_num_frames", 64)),
            "text_encoder_max_length": int(text_encoder.get("model_max_length", 300)),
            "sana_wm_refiner_text_max_length": int(
                raw.get("sana_wm_refiner_text_max_length", _DEFAULT_REFINER_TEXT_MAX_LENGTH)
            ),
            "sana_wm_chi_prompt": str(
                raw.get("sana_wm_chi_prompt", _join_chi_prompt(text_encoder))
            ),
            "flow_shift": float(scheduler.get("inference_flow_shift", 9.8)),
        }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one SANA-WM world-model bundle."""
    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "world_model_generation":
        raise ValueError("sana_wm supports only task=world_model_generation")
    if request.precision != "bf16":
        raise ValueError("SANA-WM requires precision=bf16")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("SANA-WM requires tensor_parallel_size=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("SANA-WM requires max_batch_size=1")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("SANA-WM does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("SANA-WM does not support fp32_layers")

    import yaml

    model_dir = Path(request.model_dir)
    raw = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SANA-WM config.yaml must contain a mapping")
    raw.update(
        {
            "video_height": int(request.image_height or _DEFAULT_HEIGHT),
            "video_width": int(request.image_width or _DEFAULT_WIDTH),
            "video_num_frames": int(request.video_num_frames or _DEFAULT_NUM_FRAMES),
        }
    )
    config = ModelConfig(model_type="sana_wm", raw=raw)
    model = _SanaWmModel()
    weights = model.load_weights(str(model_dir), config)
    extras = model.build_extra_engines(
        config,
        weights,
        int(request.max_sequence_length or _DEFAULT_REFINER_TEXT_MAX_LENGTH),
        precision=request.precision,
        verbose=request.verbose,
    )
    section_map = {
        "text_encoder_0_plan": "text_encoder.0.plan",
        "denoiser_plan": "denoiser.plan",
        "sana_wm_vae_encoder_plan": "vae_encoder.plan",
        "vae_decoder_plan": "vae.plan",
        "sana_wm_refiner_text_encoder_plan": "refiner.text_encoder.plan",
        "sana_wm_refiner_text_connector_plan": "refiner.text_connector.plan",
        "sana_wm_refiner_denoiser_plan": "refiner.denoiser.plan",
        "sana_wm_refiner_vae_decoder_plan": "refiner.vae.plan",
        "sana_wm_stage1_tokenizer.json": "stage1.tokenizer.json",
        "sana_wm_refiner_tokenizer.json": "refiner.tokenizer.json",
    }
    missing = sorted(name for name in section_map if name not in extras)
    if missing:
        raise RuntimeError(f"SANA-WM build did not produce required sections: {missing}")

    writer.set_header(family="sana_wm", task=request.task, backend="trt")
    for source, destination in section_map.items():
        writer.add_bytes(destination, extras[source])

    fields = model.get_bundle_config_overrides(config)
    summary = config.raw.get("_sana_wm_stage1_dit_summary", {})
    runtime = {
        "num_frames": int(fields["video_num_frames"]),
        "height": int(fields["video_height"]),
        "width": int(fields["video_width"]),
        "fps": int(fields["fps"]),
        "num_steps": int(fields["num_inference_steps"]),
        "seed": 0,
        "refiner_seed": 0,
        "vae_latent_dim": int(fields["vae_latent_dim"]),
        "vae_time_stride": int(fields["vae_time_stride"]),
        "vae_spatial_stride": int(fields["vae_spatial_stride"]),
        "vae_tile_sample_min_height": int(fields["vae_tile_sample_min_height"]),
        "vae_tile_sample_min_width": int(fields["vae_tile_sample_min_width"]),
        "vae_tile_sample_stride_height": int(fields["vae_tile_sample_stride_height"]),
        "vae_tile_sample_stride_width": int(fields["vae_tile_sample_stride_width"]),
        "vae_tile_sample_min_num_frames": int(fields["vae_tile_sample_min_num_frames"]),
        "vae_tile_sample_stride_num_frames": int(fields["vae_tile_sample_stride_num_frames"]),
        "text_encoder_max_length": int(fields["text_encoder_max_length"]),
        "text_encoder_dim": int(summary.get("text_embed_dim", 0)),
        "refiner_text_encoder_max_length": int(fields["sana_wm_refiner_text_max_length"]),
        "translation_speed": float(fields["sana_wm_translation_speed"]),
        "rotation_speed_deg": float(fields["sana_wm_rotation_speed_deg"]),
        "cfg_scale": float(fields["guidance_scale"]),
        "flow_shift": float(fields["flow_shift"]),
        "vae_use_framewise_decoding": bool(fields["vae_use_framewise_decoding"]),
        "vae_use_spatial_tiling": bool(fields["vae_use_spatial_tiling"]),
        "chi_prompt": str(fields["sana_wm_chi_prompt"]),
        "no_refiner": False,
    }
    if runtime["text_encoder_dim"] <= 0:
        raise ValueError("SANA-WM stage-1 checkpoint does not define text_encoder_dim")
    writer.add_json("runtime.json", runtime)

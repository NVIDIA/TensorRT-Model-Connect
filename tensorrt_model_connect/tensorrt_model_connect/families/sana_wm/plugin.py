"""SANA-WM family plugin.

The public SANA-WM release is not a standard diffusers directory: it ships a
Sana-specific config.yaml plus DiT, LTX-2 VAE, and refiner weights. Local
directories may package prebuilt native TRT component plans under
``trtmc_engines/`` for the C++ runtime to load.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_HF_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
_DEFAULT_ACTION = "w-80,jw-40,w-40,lw-60,w-100"
_DEFAULT_TRANSLATION_SPEED = 0.055
_DEFAULT_ROTATION_SPEED_DEG = 1.2
_DEFAULT_NUM_FRAMES = 321
_DEFAULT_HEIGHT = 704
_DEFAULT_WIDTH = 1280
_DEFAULT_FPS = 16
_DEFAULT_NUM_STEPS = 60
_DEFAULT_GUIDANCE_SCALE = 5.0
_DEFAULT_DEMO_INTRINSICS = (797.87866, 830.0503, 844.2675, 463.7225)
_DEFAULT_VAE_STRIDE = (8, 32, 32)
_STAGE1_DIT_REL = Path("dit") / "sana_wm_1600m_720p.safetensors"
_STAGE1_TEXT_ENCODER_REL = Path("text_encoder")
_STAGE1_TEXT_ENCODER_HF_IDS = {
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}
_STAGE1_TEXT_ENCODER_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer*",
    "tokenizer.model",
    "special_tokens_map.json",
)
_REFINER_REL = Path("refiner") / "refiner.safetensors"
_REFINER_GEMMA_REL = Path("refiner") / "text_encoder"
_FULL_SNAPSHOT_REQUIRED_PATHS = (
    _STAGE1_DIT_REL,
    Path("vae"),
    _REFINER_REL,
    _REFINER_GEMMA_REL,
)
_NATIVE_BUILDER_COMPONENTS = (
    "stage-1 Gemma text encoder",
    "SanaMSVideoCamCtrl DiT with BidirectionalGDN camera-control blocks",
    "LTX-2 VAE encoder",
    "LTX-2/SANA VAE decoder or complete LTX-2 refiner stack",
)
_NATIVE_PLAN_DIR = Path("trtmc_engines")
_NATIVE_PLAN_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "sana_wm_vae_encoder_plan",
    "vae_decoder_plan",
    "sana_wm_refiner_text_encoder_plan",
    "sana_wm_refiner_denoiser_plan",
    "sana_wm_refiner_vae_decoder_plan",
)
_STAGE1_CORE_PLAN_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "sana_wm_vae_encoder_plan",
)
_REFINER_PLAN_SECTIONS = (
    "sana_wm_refiner_text_encoder_plan",
    "sana_wm_refiner_denoiser_plan",
    "sana_wm_refiner_vae_decoder_plan",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_NATIVE_ENGINE_MARKER = b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"


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


def _float_list(value, fallback: tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return list(fallback)


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


def _resolve_native_plan_path(model_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = model_path / path
    return path


def _append_native_plan_dir(candidates: list[Path], model_path: Path, value: object) -> None:
    if not value:
        return
    path = _resolve_native_plan_path(model_path, str(value))
    if path not in candidates:
        candidates.append(path)


def _append_model_native_plan_dir(candidates: list[Path], model_path: Path, value: object) -> None:
    if not value:
        return
    path = _resolve_native_plan_path(model_path, str(value)) / _NATIVE_PLAN_DIR
    if path not in candidates:
        candidates.append(path)


def _resolve_stage1_text_encoder_dir(model_path: Path, raw_config: dict) -> Path | None:
    text_encoder = raw_config.get("text_encoder", {})
    if not isinstance(text_encoder, dict):
        text_encoder = {}
    candidates: list[Path] = []
    for value in (
        raw_config.get("sana_wm_text_encoder_dir"),
        text_encoder.get("text_encoder_dir"),
        os.environ.get("SANA_WM_TEXT_ENCODER_DIR"),
    ):
        if value:
            candidates.append(_resolve_native_plan_path(model_path, str(value)))
    candidates.append(model_path / _STAGE1_TEXT_ENCODER_REL)

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    downloaded = _download_stage1_text_encoder_dir(raw_config)
    if downloaded is not None and (downloaded / "config.json").is_file():
        return downloaded
    return None


def _resolve_refiner_text_encoder_dir(model_path: Path, raw_config: dict) -> Path | None:
    candidates: list[Path] = []
    for value in (
        raw_config.get("sana_wm_refiner_text_encoder_dir"),
        os.environ.get("SANA_WM_REFINER_TEXT_ENCODER_DIR"),
    ):
        if value:
            candidates.append(_resolve_native_plan_path(model_path, str(value)))
    candidates.append(model_path / _REFINER_GEMMA_REL)

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    return None


def _has_safetensors_weight_file(path: Path) -> bool:
    return (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
        or (path / "diffusion_pytorch_model.safetensors").is_file()
        or (path / "diffusion_pytorch_model.safetensors.index.json").is_file()
        or any(path.glob("*.safetensors"))
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _stage1_text_encoder_hf_id(raw_config: dict) -> str:
    text_encoder = raw_config.get("text_encoder", {})
    if not isinstance(text_encoder, dict):
        text_encoder = {}
    configured = (
        raw_config.get("sana_wm_text_encoder_hf_id")
        or text_encoder.get("text_encoder_hf_id")
        or text_encoder.get("text_encoder_name")
        or text_encoder.get("model")
        or "gemma-2-2b-it"
    )
    name = str(configured)
    return _STAGE1_TEXT_ENCODER_HF_IDS.get(name, name)


def _download_stage1_text_encoder_dir(raw_config: dict) -> Path | None:
    if not _truthy_env("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS"):
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the SANA-WM stage-1 "
            "Gemma text encoder. Install it or set SANA_WM_TEXT_ENCODER_DIR."
        ) from exc
    return Path(
        snapshot_download(
            repo_id=_stage1_text_encoder_hf_id(raw_config),
            allow_patterns=list(_STAGE1_TEXT_ENCODER_ALLOW_PATTERNS),
        )
    )


def _resolve_vae_dir(model_path: Path, raw_config: dict) -> Path | None:
    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    candidates: list[Path] = []
    for value in (
        raw_config.get("sana_wm_vae_dir"),
        vae.get("vae_dir"),
        os.environ.get("SANA_WM_VAE_DIR"),
    ):
        if value:
            candidates.append(_resolve_native_plan_path(model_path, str(value)))
    candidates.append(model_path / "vae")

    for candidate in candidates:
        if candidate.is_dir() and _has_safetensors_weight_file(candidate):
            return candidate
    return None


def _resolve_vae_encoder_dir(model_path: Path, raw_config: dict) -> Path | None:
    return _resolve_vae_dir(model_path, raw_config)


def _resolve_vae_decoder_dir(model_path: Path, raw_config: dict) -> Path | None:
    return _resolve_vae_dir(model_path, raw_config)


def _load_vae_config(vae_dir: Path) -> dict:
    config_path = vae_dir / "config.json"
    if not config_path.is_file():
        return {}
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _int_tuple(value, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    return tuple(int(v) for v in value)


def _bool_tuple(value, fallback: tuple[bool, ...]) -> tuple[bool, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    return tuple(bool(v) for v in value)


def _discover_native_plan_dirs(model_path: Path, raw_config: dict) -> list[Path]:
    candidates: list[Path] = []
    _append_native_plan_dir(candidates, model_path, raw_config.get("sana_wm_native_plan_dir"))
    _append_native_plan_dir(candidates, model_path, os.environ.get("SANA_WM_NATIVE_PLAN_DIR"))
    _append_model_native_plan_dir(candidates, model_path, raw_config.get("sana_wm_model_dir"))
    _append_model_native_plan_dir(candidates, model_path, os.environ.get("SANA_WM_MODEL_DIR"))
    _append_native_plan_dir(candidates, model_path, _NATIVE_PLAN_DIR)
    return candidates


def _discover_native_plan_paths(model_path: Path, raw_config: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    configured = raw_config.get("sana_wm_native_plan_paths")
    if isinstance(configured, dict):
        for section in _NATIVE_PLAN_SECTIONS:
            value = configured.get(section)
            if value:
                path = _resolve_native_plan_path(model_path, str(value))
                if not path.is_file():
                    raise FileNotFoundError(
                        f"SANA-WM native plan {section!r} does not exist: {path}"
                    )
                paths[section] = path

    for plan_dir in _discover_native_plan_dirs(model_path, raw_config):
        for section in _NATIVE_PLAN_SECTIONS:
            if section in paths:
                continue
            path = plan_dir / f"{section}.plan"
            if path.is_file():
                paths[section] = path
    return paths


def _effective_native_sections(
    paths: dict[str, Path],
    *,
    can_build_stage1_text_encoder: bool,
    can_build_refiner_text_encoder: bool = False,
    can_build_vae_encoder: bool = False,
    can_build_vae_decoder: bool = False,
) -> list[str]:
    present = set(paths)
    if can_build_stage1_text_encoder:
        present.add("text_encoder_0_plan")
    if can_build_refiner_text_encoder:
        present.add("sana_wm_refiner_text_encoder_plan")
    if can_build_vae_encoder:
        present.add("sana_wm_vae_encoder_plan")
    if can_build_vae_decoder:
        present.add("vae_decoder_plan")
    return [section for section in _NATIVE_PLAN_SECTIONS if section in present]


def _native_sections_are_complete(sections: list[str]) -> bool:
    present = set(sections)
    if not all(section in present for section in _STAGE1_CORE_PLAN_SECTIONS):
        return False
    present_refiner = [section for section in _REFINER_PLAN_SECTIONS if section in present]
    if present_refiner:
        return all(section in present for section in _REFINER_PLAN_SECTIONS)
    return "vae_decoder_plan" in present


def _validate_native_plan_paths(
    paths: dict[str, Path],
    *,
    can_build_stage1_text_encoder: bool = False,
    can_build_refiner_text_encoder: bool = False,
    can_build_vae_encoder: bool = False,
    can_build_vae_decoder: bool = False,
) -> None:
    if not paths:
        return
    effective_sections = _effective_native_sections(
        paths,
        can_build_stage1_text_encoder=can_build_stage1_text_encoder,
        can_build_refiner_text_encoder=can_build_refiner_text_encoder,
        can_build_vae_encoder=can_build_vae_encoder,
        can_build_vae_decoder=can_build_vae_decoder,
    )
    missing_core = [
        section
        for section in _STAGE1_CORE_PLAN_SECTIONS
        if section not in effective_sections
    ]
    if missing_core:
        raise ValueError(
            "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
            f"missing {missing_core!r} with only {effective_sections!r} present"
        )

    present_refiner = [section for section in _REFINER_PLAN_SECTIONS if section in effective_sections]
    if present_refiner:
        missing_refiner = [
            section
            for section in _REFINER_PLAN_SECTIONS
            if section not in effective_sections
        ]
        if missing_refiner:
            raise ValueError(
                "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
                f"missing {missing_refiner!r} with only {effective_sections!r} present"
            )
        return

    if "vae_decoder_plan" not in effective_sections:
        raise ValueError(
            "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
            f"missing ['vae_decoder_plan'] with only {effective_sections!r} present"
        )


def _join_chi_prompt(text_encoder: dict) -> str:
    chi_prompt = text_encoder.get("chi_prompt", [])
    if isinstance(chi_prompt, str):
        return chi_prompt
    if isinstance(chi_prompt, (list, tuple)):
        return "\n".join(str(line) for line in chi_prompt)
    return ""


def _discover_tokenizer_sections(model_path: Path, raw_config: dict) -> dict[str, Path]:
    configured = raw_config.get("sana_wm_tokenizer_dir")
    candidates: list[Path] = []
    if configured:
        candidates.append(_resolve_native_plan_path(model_path, str(configured)))
    candidates.extend(
        [
            model_path / _STAGE1_TEXT_ENCODER_REL,
            model_path / _REFINER_GEMMA_REL,
        ]
    )

    for candidate in candidates:
        if not candidate.is_dir() or candidate == model_path:
            continue
        sections = {
            name: candidate / name
            for name in _TOKENIZER_FILES
            if (candidate / name).is_file()
        }
        if "tokenizer.json" in sections:
            return sections
    return {}


def _validate_native_tokenizer_sections(tokenizer_sections: dict[str, Path]) -> None:
    if not tokenizer_sections:
        raise ValueError(
            "SANA-WM native TensorRT bundles require tokenizer assets for the C++ "
            "text-encoder path. Place tokenizer.json under text_encoder/ or "
            "refiner/text_encoder/, or set sana_wm_tokenizer_dir."
        )


def _missing_full_snapshot_paths(model_path: Path) -> list[str]:
    missing: list[str] = []
    for rel in _FULL_SNAPSHOT_REQUIRED_PATHS:
        if not (model_path / rel).exists():
            missing.append(str(rel))
    return missing


def _missing_native_builder_components(weights: WeightDict) -> tuple[str, ...]:
    missing = list(_NATIVE_BUILDER_COMPONENTS)
    if weights.get("_stage1_text_encoder_dir"):
        missing = [
            component
            for component in missing
            if component != "stage-1 Gemma text encoder"
        ]
    if weights.get("_sana_wm_vae_encoder_dir"):
        missing = [
            component for component in missing if component != "LTX-2 VAE encoder"
        ]
    if weights.get("_sana_wm_vae_decoder_dir"):
        missing = [
            component
            for component in missing
            if component != "LTX-2/SANA VAE decoder or complete LTX-2 refiner stack"
        ]
    return tuple(missing)


def _native_build_error(weights: WeightDict) -> str:
    model_path = Path(str(weights.get("_model_dir", "")))
    components = "; ".join(_missing_native_builder_components(weights))
    message = (
        "SANA-WM pure C++ builds require native TensorRT component plans under "
        "trtmc_engines/ or sana_wm_native_plan_paths. Building those plans "
        "directly from raw SANA-WM weights is not implemented yet; missing "
        f"native builders: {components}."
    )
    missing = _missing_full_snapshot_paths(model_path)
    if missing:
        message += (
            " The resolved model snapshot is also missing raw SANA-WM weight "
            f"paths: {', '.join(missing)}. Set TRTMC_SANA_WM_DOWNLOAD_WEIGHTS=1 "
            "when resolving Efficient-Large-Model/SANA-WM_bidirectional, or "
            "point SANA_WM_MODEL_DIR at a full local snapshot."
        )
    return message


def _build_gemma_text_encoder_plan(
    text_encoder_dir: Path,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    label: str,
) -> bytes:
    from ..gemma.plugin import plugin as gemma_plugin
    from ..gemma.standard_decoder_builder import build_standard_decoder_engine

    text_config = ModelConfig.from_dir(text_encoder_dir)
    if not gemma_plugin.matches(text_config.model_type):
        raise ValueError(
            f"SANA-WM {label} text encoder builder currently supports Gemma only; "
            f"found model_type={text_config.model_type!r} in {text_encoder_dir}"
        )
    text_weights = gemma_plugin.load_weights(
        str(text_encoder_dir),
        text_config,
        precision=precision,
    )
    return build_standard_decoder_engine(
        text_config,
        text_weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
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
    )


def _build_sana_wm_vae_encoder_plan(
    vae_dir: Path,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from ..ltx_video.ltx_vae_builder import (
        build_ltx_vae_encoder_engine,
        load_ltx_vae_encoder_weights,
    )

    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    vae_config = _load_vae_config(vae_dir)
    video_height = int(raw_config.get("video_height", _DEFAULT_HEIGHT))
    video_width = int(raw_config.get("video_width", _DEFAULT_WIDTH))
    weights = load_ltx_vae_encoder_weights(vae_dir, precision=precision)
    return build_ltx_vae_encoder_engine(
        weights,
        sample_frames=1,
        sample_height=video_height,
        sample_width=video_width,
        in_channels=int(vae_config.get("in_channels", 3)),
        latent_channels=int(
            vae_config.get("latent_channels", vae.get("vae_latent_dim", 128))
        ),
        block_out_channels=_int_tuple(
            vae_config.get("block_out_channels"), (128, 256, 512, 512)
        ),
        layers_per_block=_int_tuple(
            vae_config.get("layers_per_block"), (4, 3, 3, 3, 4)
        ),
        spatio_temporal_scaling=_bool_tuple(
            vae_config.get("spatio_temporal_scaling"), (True, True, True, False)
        ),
        patch_size=int(vae_config.get("patch_size", 4)),
        patch_size_t=int(vae_config.get("patch_size_t", 1)),
        precision=precision,
        normalize_output=True,
        scaling_factor=float(vae_config.get("scaling_factor", 1.0)),
        verbose=verbose,
    )


def _build_sana_wm_vae_decoder_plan(
    vae_dir: Path,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    from ..ltx_video.ltx_vae_builder import (
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
            vae_config.get("decoder_block_out_channels")
            or vae_config.get("block_out_channels"),
            (128, 256, 512, 512),
        ),
        layers_per_block=_int_tuple(
            vae_config.get("decoder_layers_per_block")
            or vae_config.get("layers_per_block"),
            (4, 3, 3, 3, 4),
        ),
        spatio_temporal_scaling=_bool_tuple(
            vae_config.get("decoder_spatio_temporal_scaling")
            or vae_config.get("spatio_temporal_scaling"),
            (True, True, True, False),
        ),
        patch_size=int(vae_config.get("patch_size", 4)),
        patch_size_t=int(vae_config.get("patch_size_t", 1)),
        out_channels=int(vae_config.get("out_channels", 3)),
        precision=precision,
        denormalize_input=True,
        scaling_factor=float(vae_config.get("scaling_factor", 1.0)),
        verbose=verbose,
    )


class SanaWmPlugin:
    name = "sana_wm"
    runtime_strategy = "diffusion_sana_wm"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in (
            "sana_wm",
            "sana-wm",
            "sanamsvideocamctrl_1600m_p1_d20",
        )

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_path = Path(model_dir)
        weights = WeightDict()
        weights["_model_format"] = "sana_wm_yaml"
        weights["_model_dir"] = str(model_path)
        weights["_stage1_dit_path"] = str(model_path / _STAGE1_DIT_REL)
        weights["_vae_dir"] = str(model_path / "vae")
        weights["_refiner_checkpoint"] = str(model_path / _REFINER_REL)
        weights["_refiner_gemma_root"] = str(model_path / _REFINER_GEMMA_REL)

        stage1_path = model_path / _STAGE1_DIT_REL
        if stage1_path.is_file():
            summary = _summarize_stage1_dit(stage1_path)
            weights["_stage1_dit_summary"] = summary
            config.raw["_sana_wm_stage1_dit_summary"] = summary
        native_plan_paths = _discover_native_plan_paths(model_path, config.raw)
        stage1_text_encoder_dir = _resolve_stage1_text_encoder_dir(model_path, config.raw)
        can_build_stage1_text_encoder = stage1_text_encoder_dir is not None
        refiner_text_encoder_dir = _resolve_refiner_text_encoder_dir(model_path, config.raw)
        can_build_refiner_text_encoder = refiner_text_encoder_dir is not None
        vae_encoder_dir = _resolve_vae_encoder_dir(model_path, config.raw)
        can_build_vae_encoder = vae_encoder_dir is not None
        vae_decoder_dir = _resolve_vae_decoder_dir(model_path, config.raw)
        can_build_vae_decoder = vae_decoder_dir is not None
        _validate_native_plan_paths(
            native_plan_paths,
            can_build_stage1_text_encoder=can_build_stage1_text_encoder,
            can_build_refiner_text_encoder=can_build_refiner_text_encoder,
            can_build_vae_encoder=can_build_vae_encoder,
            can_build_vae_decoder=can_build_vae_decoder,
        )
        tokenizer_sections = _discover_tokenizer_sections(model_path, config.raw)
        if native_plan_paths:
            _validate_native_tokenizer_sections(tokenizer_sections)
        if native_plan_paths:
            weights["_native_plan_paths"] = {
                section: str(path) for section, path in native_plan_paths.items()
            }
            effective_sections = _effective_native_sections(
                native_plan_paths,
                can_build_stage1_text_encoder=can_build_stage1_text_encoder,
                can_build_refiner_text_encoder=can_build_refiner_text_encoder,
                can_build_vae_encoder=can_build_vae_encoder,
                can_build_vae_decoder=can_build_vae_decoder,
            )
            if _native_sections_are_complete(effective_sections):
                config.raw["_sana_wm_native_plan_sections"] = effective_sections
        if stage1_text_encoder_dir is not None:
            weights["_stage1_text_encoder_dir"] = str(stage1_text_encoder_dir)
        if refiner_text_encoder_dir is not None:
            weights["_refiner_text_encoder_dir"] = str(refiner_text_encoder_dir)
        if vae_encoder_dir is not None:
            weights["_sana_wm_vae_encoder_dir"] = str(vae_encoder_dir)
        if vae_decoder_dir is not None:
            weights["_sana_wm_vae_decoder_dir"] = str(vae_decoder_dir)
        if tokenizer_sections:
            weights["_tokenizer_sections"] = {
                section: str(path) for section, path in tokenizer_sections.items()
            }
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
        del max_cache_length, precision, quant_ctx, verbose
        native_plan_paths = weights.get("_native_plan_paths")
        effective_sections = _effective_native_sections(
            native_plan_paths if isinstance(native_plan_paths, dict) else {},
            can_build_stage1_text_encoder=bool(weights.get("_stage1_text_encoder_dir")),
            can_build_refiner_text_encoder=bool(weights.get("_refiner_text_encoder_dir")),
            can_build_vae_encoder=bool(weights.get("_sana_wm_vae_encoder_dir")),
            can_build_vae_decoder=bool(weights.get("_sana_wm_vae_decoder_dir")),
        )
        if not _native_sections_are_complete(effective_sections):
            raise NotImplementedError(_native_build_error(weights))
        # The runtime plugin ignores engine_plan. A small marker section keeps
        # the bundle shape compatible with the generic builder/writer path.
        return _NATIVE_ENGINE_MARKER

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict:
        result = {}
        plan_paths = weights.get("_native_plan_paths", {})
        if isinstance(plan_paths, dict):
            result.update(
                {
                    section: Path(path).read_bytes()
                    for section, path in plan_paths.items()
                    if section in _NATIVE_PLAN_SECTIONS
                }
            )
        text_encoder_dir = weights.get("_stage1_text_encoder_dir")
        if "text_encoder_0_plan" not in result and text_encoder_dir:
            result["text_encoder_0_plan"] = _build_stage1_text_encoder_plan(
                Path(str(text_encoder_dir)),
                max_cache_length,
                precision=precision,
                verbose=verbose,
            )
        refiner_text_encoder_dir = weights.get("_refiner_text_encoder_dir")
        if "sana_wm_refiner_text_encoder_plan" not in result and refiner_text_encoder_dir:
            result["sana_wm_refiner_text_encoder_plan"] = _build_refiner_text_encoder_plan(
                Path(str(refiner_text_encoder_dir)),
                max(max_cache_length, 256),
                precision=precision,
                verbose=verbose,
            )
        vae_encoder_dir = weights.get("_sana_wm_vae_encoder_dir")
        if "sana_wm_vae_encoder_plan" not in result and vae_encoder_dir:
            result["sana_wm_vae_encoder_plan"] = _build_sana_wm_vae_encoder_plan(
                Path(str(vae_encoder_dir)),
                config.raw,
                precision=precision,
                verbose=verbose,
            )
        vae_decoder_dir = weights.get("_sana_wm_vae_decoder_dir")
        if "vae_decoder_plan" not in result and vae_decoder_dir:
            result["vae_decoder_plan"] = _build_sana_wm_vae_decoder_plan(
                Path(str(vae_decoder_dir)),
                config.raw,
                precision=precision,
                verbose=verbose,
            )
        tokenizer_sections = weights.get("_tokenizer_sections", {})
        if isinstance(tokenizer_sections, dict):
            result.update(
                {
                    section: Path(path).read_bytes()
                    for section, path in tokenizer_sections.items()
                    if section in _TOKENIZER_FILES
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

        native_sections = raw.get("_sana_wm_native_plan_sections")
        has_native_sections = isinstance(native_sections, list) and len(native_sections) > 0

        overrides = {
            "model_type": "sana_wm",
            "runtime_strategy": self.runtime_strategy,
            "sana_wm_hf_id": _HF_ID,
            "sana_wm_config_path": f"hf://{_HF_ID}/config.yaml",
            "sana_wm_model_path": f"hf://{_HF_ID}/dit/sana_wm_1600m_720p.safetensors",
            "sana_wm_refiner_checkpoint": f"hf://{_HF_ID}/refiner/refiner.safetensors",
            "sana_wm_refiner_gemma_root": f"hf://{_HF_ID}/refiner/text_encoder",
            "sana_wm_action": str(raw.get("sana_wm_action", _DEFAULT_ACTION)),
            "sana_wm_translation_speed": float(
                raw.get("sana_wm_translation_speed", _DEFAULT_TRANSLATION_SPEED)
            ),
            "sana_wm_rotation_speed_deg": float(
                raw.get("sana_wm_rotation_speed_deg", _DEFAULT_ROTATION_SPEED_DEG)
            ),
            "sana_wm_default_intrinsics": _float_list(
                raw.get("sana_wm_default_intrinsics"), _DEFAULT_DEMO_INTRINSICS
            ),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "fps": int(raw.get("fps", _DEFAULT_FPS)),
            "num_inference_steps": int(
                raw.get("num_inference_steps", _DEFAULT_NUM_STEPS)
            ),
            "guidance_scale": float(
                raw.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE)
            ),
            "vae_latent_dim": int(vae.get("vae_latent_dim", raw.get("vae_latent_dim", 128))),
            "vae_downsample_rate": int(
                vae.get("vae_downsample_rate", raw.get("vae_downsample_rate", 32))
            ),
            "vae_time_stride": int(vae_stride[0]),
            "vae_spatial_stride": int(vae_stride[-1]),
            "text_encoder_name": str(
                text_encoder.get("text_encoder_name")
                or text_encoder.get("model")
                or "gemma-2-2b-it"
            ),
            "text_encoder_max_length": int(text_encoder.get("model_max_length", 300)),
            "sana_wm_chi_prompt": str(
                raw.get("sana_wm_chi_prompt", _join_chi_prompt(text_encoder))
            ),
            "flow_shift": float(scheduler.get("inference_flow_shift", 9.8)),
        }
        if not has_native_sections:
            overrides["engine_backend"] = "none"
        if has_native_sections:
            overrides["sana_wm_native_plan_sections"] = [str(v) for v in native_sections]
        stage1_summary = raw.get("_sana_wm_stage1_dit_summary")
        if isinstance(stage1_summary, dict):
            overrides.update(
                {
                    "sana_wm_dit_num_layers": int(stage1_summary.get("num_layers", 0)),
                    "sana_wm_dit_hidden_size": int(stage1_summary.get("hidden_size", 0)),
                    "sana_wm_dit_text_embed_dim": int(stage1_summary.get("text_embed_dim", 0)),
                    "sana_wm_dit_tensor_count": int(stage1_summary.get("tensor_count", 0)),
                }
            )
        return overrides


plugin = SanaWmPlugin()

"""SANA-WM family plugin.

The public SANA-WM release is not a standard diffusers directory: it ships a
Sana-specific config.yaml plus DiT, LTX-2 VAE, and refiner weights. Local
directories may package prebuilt native TRT component plans under
``trtmc_engines/`` for the C++ runtime to load. The official Python bridge is a
legacy compatibility path and is only bundled when explicitly enabled.
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
_REFINER_REL = Path("refiner") / "refiner.safetensors"
_REFINER_GEMMA_REL = Path("refiner") / "text_encoder"
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
_PYTHON_BRIDGE_ENGINE_MARKER = b"TRTMC_SANA_WM_PYTHON_BRIDGE\n"


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


def _allow_python_bridge(raw_config: dict) -> bool:
    return int(raw_config.get("sana_wm_allow_python_bridge", 0)) != 0


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


def _validate_native_plan_paths(paths: dict[str, Path]) -> None:
    if not paths:
        return
    missing_core = [section for section in _STAGE1_CORE_PLAN_SECTIONS if section not in paths]
    if missing_core:
        present = [section for section in _NATIVE_PLAN_SECTIONS if section in paths]
        raise ValueError(
            "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
            f"missing {missing_core!r} with only {present!r} present"
        )

    present_refiner = [section for section in _REFINER_PLAN_SECTIONS if section in paths]
    if present_refiner:
        missing_refiner = [section for section in _REFINER_PLAN_SECTIONS if section not in paths]
        if missing_refiner:
            present = [section for section in _NATIVE_PLAN_SECTIONS if section in paths]
            raise ValueError(
                "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
                f"missing {missing_refiner!r} with only {present!r} present"
            )
        return

    if "vae_decoder_plan" not in paths:
        present = [section for section in _NATIVE_PLAN_SECTIONS if section in paths]
        raise ValueError(
            "SANA-WM native TensorRT bundle requires a complete prebuilt plan set; "
            f"missing ['vae_decoder_plan'] with only {present!r} present"
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
        _validate_native_plan_paths(native_plan_paths)
        tokenizer_sections = _discover_tokenizer_sections(model_path, config.raw)
        if native_plan_paths:
            _validate_native_tokenizer_sections(tokenizer_sections)
        if native_plan_paths:
            weights["_native_plan_paths"] = {
                section: str(path) for section, path in native_plan_paths.items()
            }
            config.raw["_sana_wm_native_plan_sections"] = list(native_plan_paths)
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
        if not weights.get("_native_plan_paths") and not _allow_python_bridge(config.raw):
            raise NotImplementedError(
                "SANA-WM pure C++ builds require native TensorRT component plans under "
                "trtmc_engines/ or sana_wm_native_plan_paths. The Python bridge is disabled "
                "by default; set sana_wm_allow_python_bridge=1 only for legacy bridge testing."
            )
        # The runtime plugin ignores engine_plan. A small marker section keeps
        # the bundle shape compatible with the generic builder/writer path.
        if weights.get("_native_plan_paths"):
            return _NATIVE_ENGINE_MARKER
        return _PYTHON_BRIDGE_ENGINE_MARKER

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict:
        del config, max_cache_length, precision, verbose
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
            "sana_wm_require_official_script": int(
                raw.get("sana_wm_require_official_script", 1)
            ),
            "sana_wm_allow_python_bridge": int(_allow_python_bridge(raw)),
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

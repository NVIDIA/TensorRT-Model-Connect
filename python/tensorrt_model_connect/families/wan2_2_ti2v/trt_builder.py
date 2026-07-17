# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT component builder for Wan2.2 TI2V-5B.

Python and PyTorch are build-time checkpoint readers only.  The returned plans
contain the complete UMT5 encoder, 30-layer DiT, and FP32 VAE decoder; the
Model-Connect runtime executes only C++/CUDA/TensorRT.

Prebuilt plans are deliberately treated as qualified artifacts rather than
opaque byte strings.  Every prebuilt plan needs a sibling ``.manifest.json``
that binds its SHA256, component source identity, CUDA plugin bytes, and exact
1280x704/121-frame I/O contract to the current checkpoint and source tree.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from tensorrt_model_connect import trt_compat

from .model_config import WAN22_TI2V_5B, official_artifact_profile
from .plugin import WAN22_MODEL_OWNED_BUNDLE_SECTIONS


_FAMILY = "wan2_2_ti2v"
_PREBUILT_MANIFEST_SCHEMA = "trtmc.wan2_2_ti2v.prebuilt.v1"
_ARTIFACT_MANIFEST_SCHEMA = "trtmc.wan2_2_ti2v.bundle-artifacts.v2"
_PACKAGE_DIR = Path(__file__).resolve().parent
_FAMILIES_DIR = _PACKAGE_DIR.parent
_LOADED_PLUGIN_HANDLES: dict[Path, ctypes.CDLL] = {}
_FORBIDDEN_PLUGIN_DEPENDENCY_MARKERS = ("python", "torch", "c10")

_PLAN_COMPONENTS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
)

_COMPONENT_SOURCE_FILES = {
    "text_encoder_0_plan": (
        "wan2_2_ti2v/trt_builder.py",
        "wan2_2_ti2v/plugin.py",
        "wan2_2_ti2v/model_config.py",
        "wan2_2_ti2v/umt5_encoder_builder.py",
        "wan2_2_ti2v/umt5_cuda_plugin_builder.py",
        "wan2_2_ti2v/umt5_cuda_plugins/CMakeLists.txt",
        "wan2_2_ti2v/umt5_cuda_plugins/wan22_umt5_gelu_plugin.cu",
        "wan2_2_ti2v/trt_ops.py",
    ),
    "denoiser_plan": (
        "wan2_2_ti2v/trt_builder.py",
        "wan2_2_ti2v/plugin.py",
        "wan2_2_ti2v/model_config.py",
        "wan2_2_ti2v/checkpoint_mapper.py",
        "wan2_2_ti2v/dit_builder.py",
        "wan2_2_ti2v/dit_cuda_plugin_builder.py",
        "wan2_2_ti2v/trt_ops.py",
    ),
    "vae_decoder_plan": (
        "wan2_2_ti2v/trt_builder.py",
        "wan2_2_ti2v/plugin.py",
        "wan2_2_ti2v/model_config.py",
        "wan2_2_ti2v/checkpoint_mapper.py",
        "wan2_2_ti2v/vae_step_builder.py",
        "wan2_2_ti2v/vae_builder.py",
        "wan2_2_ti2v/vae_cuda_plugin_builder.py",
        "wan2_2_ti2v/graph_ops.py",
        "wan2_2_ti2v/graph_blocks.py",
    ),
    "vae_decoder_first_frame_plan": (
        "wan2_2_ti2v/trt_builder.py",
        "wan2_2_ti2v/plugin.py",
        "wan2_2_ti2v/model_config.py",
        "wan2_2_ti2v/checkpoint_mapper.py",
        "wan2_2_ti2v/vae_step_builder.py",
        "wan2_2_ti2v/vae_builder.py",
        "wan2_2_ti2v/vae_cuda_plugin_builder.py",
        "wan2_2_ti2v/graph_ops.py",
        "wan2_2_ti2v/graph_blocks.py",
    ),
}


# Keep provenance and sidecar validation importable in a minimal build
# environment. Heavy TensorRT/NumPy/PyTorch conversion modules are loaded only
# when their corresponding component is actually built.
def ensure_umt5_cuda_plugin(**kwargs):
    from .umt5_cuda_plugin_builder import ensure_umt5_cuda_plugin as implementation

    return implementation(**kwargs)


def ensure_dit_cuda_plugin(**kwargs):
    from .dit_cuda_plugin_builder import ensure_dit_cuda_plugin as implementation

    return implementation(**kwargs)


def ensure_vae_cuda_plugin(**kwargs):
    from .vae_cuda_plugin_builder import ensure_vae_cuda_plugin as implementation

    return implementation(**kwargs)


def build_native_umt5_encoder_engine(*args, **kwargs):
    from .umt5_encoder_builder import (
        build_native_umt5_encoder_engine as implementation,
    )

    return implementation(*args, **kwargs)


def build_dit_engine(*args, **kwargs):
    from .dit_builder import build_dit_engine as implementation

    return implementation(*args, **kwargs)


def build_vae_step_engine(*args, **kwargs):
    from .vae_step_builder import build_vae_step_engine as implementation

    return implementation(*args, **kwargs)


def load_vae_step_weights(*args, **kwargs):
    from .vae_step_builder import load_vae_step_weights as implementation

    return implementation(*args, **kwargs)


def official_vae_step_profile():
    from .vae_step_builder import OFFICIAL_VAE_STEP_PROFILE

    return OFFICIAL_VAE_STEP_PROFILE


def _official_profile() -> dict[str, Any]:
    return official_artifact_profile()


def _validate_requested_profile(config) -> None:
    raw = config.raw
    profile = _official_profile()
    requested = {
        "video_width": int(raw.get("video_width", profile["video_width"])),
        "video_height": int(raw.get("video_height", profile["video_height"])),
        "video_num_frames": int(raw.get("video_num_frames", profile["video_num_frames"])),
    }
    expected = {key: profile[key] for key in requested}
    if requested != expected:
        raise ValueError(
            "Wan2.2-TI2V-5B TensorRT engines are fixed to the official "
            "1280x704, 121-frame profile; requested "
            f"{requested['video_width']}x{requested['video_height']}, "
            f"{requested['video_num_frames']} frames"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, cache: dict[Path, str] | None = None) -> str:
    resolved = path.expanduser().resolve()
    if cache is not None and resolved in cache:
        return cache[resolved]
    if not resolved.is_file():
        raise FileNotFoundError(f"Wan2.2 source artifact not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    result = digest.hexdigest()
    if cache is not None:
        cache[resolved] = result
    return result


def _canonical_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _transformer_checkpoint_paths(root: Path) -> list[Path]:
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid Wan2.2 transformer index: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and name for name in shard_names):
            raise ValueError(f"Invalid Wan2.2 transformer shard name in {index_path}")
        paths = [index_path, *(root / name for name in shard_names)]
    else:
        paths = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No Wan2.2 transformer safetensors in {root}")
    return paths


def _checkpoint_sources(component: str, model_root: Path, weights: dict) -> list[tuple[str, Path]]:
    config_path = model_root / "config.json"
    if component == "text_encoder_0_plan":
        checkpoint = Path(
            weights.get(
                "_text_encoder_checkpoint",
                model_root / "models_t5_umt5-xxl-enc-bf16.pth",
            )
        )
        return [
            ("checkpoint/config.json", config_path),
            ("checkpoint/models_t5_umt5-xxl-enc-bf16.pth", checkpoint),
        ]
    if component == "denoiser_plan":
        result = [("checkpoint/config.json", config_path)]
        result.extend(
            (f"checkpoint/{path.relative_to(model_root).as_posix()}", path)
            for path in _transformer_checkpoint_paths(model_root)
        )
        return result
    if component in {"vae_decoder_plan", "vae_decoder_first_frame_plan"}:
        checkpoint = Path(weights.get("_vae_checkpoint", model_root / "Wan2.2_VAE.pth"))
        return [
            ("checkpoint/config.json", config_path),
            ("checkpoint/Wan2.2_VAE.pth", checkpoint),
        ]
    raise ValueError(f"Unknown Wan2.2 plan component: {component!r}")


def _component_plugin_sections(component: str) -> tuple[str, ...]:
    if component == "text_encoder_0_plan":
        return ("wan2_2_umt5_cuda_plugin_so",)
    if component == "denoiser_plan":
        return (
            "wan2_2_umt5_cuda_plugin_so",
            "wan2_2_dit_cuda_plugin_so",
        )
    if component in {"vae_decoder_plan", "vae_decoder_first_frame_plan"}:
        return ("wan2_2_vae_cuda_plugin_so",)
    raise ValueError(f"Unknown Wan2.2 plan component: {component!r}")


def _component_source_identity(
    component: str,
    model_dir: str | Path,
    weights: dict,
    plugin_payloads: dict[str, bytes],
    *,
    digest_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    inputs = [
        {"name": name, "sha256": _sha256_file(path, digest_cache)}
        for name, path in _checkpoint_sources(component, root, weights)
    ]
    inputs.extend(
        {
            "name": f"source/{filename}",
            "sha256": _sha256_file(_FAMILIES_DIR / filename, digest_cache),
        }
        for filename in _COMPONENT_SOURCE_FILES[component]
    )
    inputs.extend(
        {
            "name": f"bundle/{section}",
            "sha256": _sha256_bytes(plugin_payloads[section]),
        }
        for section in _component_plugin_sections(component)
    )
    inputs.sort(key=lambda item: item["name"])
    identity_document = {
        "family": _FAMILY,
        "component": component,
        "profile": _official_profile(),
        "inputs": inputs,
    }
    return {
        "sha256": _canonical_sha256(identity_document),
        "inputs": inputs,
    }


def _prebuilt_manifest_path(plan_path: Path) -> Path:
    return Path(f"{plan_path}.manifest.json")


def _prebuilt_manifest_payload(component: str, plan: bytes, source_sha256: str) -> dict[str, Any]:
    return {
        "schema": _PREBUILT_MANIFEST_SCHEMA,
        "family": _FAMILY,
        "component": component,
        "profile": _official_profile(),
        "plan_sha256": _sha256_bytes(plan),
        "source_sha256": source_sha256,
    }


def _read_prebuilt(
    config,
    config_key: str,
    environment_key: str,
    *,
    component: str,
    source_sha256: str,
    manifest_config_key: str,
    manifest_environment_key: str,
) -> bytes | None:
    configured = config.raw.get(config_key) or os.environ.get(environment_key)
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prebuilt Wan2.2 component not found: {path}")

    configured_manifest = config.raw.get(manifest_config_key) or os.environ.get(
        manifest_environment_key
    )
    manifest_path = (
        Path(configured_manifest).expanduser()
        if configured_manifest
        else _prebuilt_manifest_path(path)
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Qualified Wan2.2 prebuilt manifest not found: {manifest_path}. "
            "Generate it with write_wan22_prebuilt_manifest() after qualifying "
            "the plan on the target GPU."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid Wan2.2 prebuilt manifest: {manifest_path}") from exc
    expected_keys = {
        "schema",
        "family",
        "component",
        "profile",
        "plan_sha256",
        "source_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError(
            f"Wan2.2 prebuilt manifest {manifest_path} must contain exactly {sorted(expected_keys)}"
        )

    expected_static = {
        "schema": _PREBUILT_MANIFEST_SCHEMA,
        "family": _FAMILY,
        "component": component,
        "profile": _official_profile(),
        "source_sha256": source_sha256,
    }
    for key, expected in expected_static.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Wan2.2 prebuilt manifest {manifest_path} has mismatched {key}: "
                f"{manifest.get(key)!r}; expected {expected!r}"
            )

    plan = path.read_bytes()
    actual_plan_sha256 = _sha256_bytes(plan)
    if manifest["plan_sha256"] != actual_plan_sha256:
        raise ValueError(
            f"Wan2.2 prebuilt plan SHA256 mismatch for {path}: "
            f"manifest={manifest['plan_sha256']}, actual={actual_plan_sha256}"
        )
    return plan


def _enum_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].lower()


def _inspect_serialized_engine(plan: bytes) -> dict[str, tuple[str, tuple[int, ...], str]]:
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise ValueError("TensorRT could not deserialize the Wan2.2 prebuilt plan")
    result = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        result[name] = (
            _enum_name(engine.get_tensor_mode(name)),
            tuple(int(dim) for dim in engine.get_tensor_shape(name)),
            _enum_name(engine.get_tensor_dtype(name)),
        )
    return result


def _expected_engine_contract(
    component: str,
) -> dict[str, tuple[str, tuple[int, ...], str]]:
    arch = WAN22_TI2V_5B
    latent_shape = (
        1,
        arch.z_dim,
        arch.latent_frames,
        arch.latent_height,
        arch.latent_width,
    )
    if component == "text_encoder_0_plan":
        return {
            "input_ids": ("input", (1, arch.text_seq_len), "int32"),
            "attention_mask": ("input", (1, arch.text_seq_len), "int32"),
            "text_embeddings": (
                "output",
                (1, arch.text_seq_len, arch.text_dim),
                "float",
            ),
        }
    if component == "denoiser_plan":
        return {
            "latents": ("input", latent_shape, "float"),
            "time_features": ("input", (1, arch.freq_dim), "float"),
            "encoder_hidden_states": (
                "input",
                (1, arch.text_seq_len, arch.text_dim),
                "float",
            ),
            "noise_prediction": ("output", latent_shape, "float"),
        }
    if component in {"vae_decoder_plan", "vae_decoder_first_frame_plan"}:
        from .vae_step_builder import VAE_STEP_CACHE_SPECS

        profile = official_vae_step_profile()
        output_frames = 4 if component == "vae_decoder_plan" else 1
        contract = {
            "latent_frame": ("input", profile.latent_shape, "float"),
            "video_frame": (
                "output",
                profile.video_shape(first_frame_only=output_frames == 1),
                "float",
            ),
        }
        for spec in VAE_STEP_CACHE_SPECS:
            shape = spec.shape(profile)
            contract[f"cache_{spec.index}"] = ("input", shape, "float")
            contract[f"cache_out_{spec.index}"] = ("output", shape, "float")
        return contract
    raise ValueError(f"Unknown Wan2.2 plan component: {component!r}")


def _validate_serialized_engine_contract(plan: bytes, component: str) -> None:
    actual = _inspect_serialized_engine(plan)
    expected = _expected_engine_contract(component)
    if actual != expected:
        raise ValueError(
            f"Wan2.2 {component} TensorRT I/O contract mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )


def _register_plugin_library(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if resolved not in _LOADED_PLUGIN_HANDLES:
        _LOADED_PLUGIN_HANDLES[resolved] = ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)


def _validate_plugin_runtime_dependencies(path: Path) -> tuple[str, ...]:
    resolved = path.expanduser().resolve()
    try:
        result = subprocess.run(
            ["readelf", "-d", str(resolved)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Unable to inspect Wan2.2 CUDA plugin dependencies: {resolved}") from exc
    needed = tuple(
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*Shared library: \[([^]]+)\]", line))
    )
    forbidden = sorted(
        dependency
        for dependency in needed
        if any(marker in dependency.casefold() for marker in _FORBIDDEN_PLUGIN_DEPENDENCY_MARKERS)
    )
    if forbidden:
        raise ValueError(
            "Wan2.2 CUDA plugin links forbidden Python/PyTorch runtime "
            f"dependencies: path={resolved}, needed={forbidden}"
        )
    return needed


def _ensure_plugin_payloads(*, verbose: bool) -> tuple[dict[str, Path], dict[str, bytes]]:
    paths = {
        "wan2_2_umt5_cuda_plugin_so": ensure_umt5_cuda_plugin(verbose=verbose),
        "wan2_2_dit_cuda_plugin_so": ensure_dit_cuda_plugin(verbose=verbose),
        "wan2_2_vae_cuda_plugin_so": ensure_vae_cuda_plugin(verbose=verbose),
    }
    for path in paths.values():
        _validate_plugin_runtime_dependencies(path)
        _register_plugin_library(path)
    return paths, {section: path.read_bytes() for section, path in paths.items()}


def write_wan22_prebuilt_manifest(
    plan_path: str | Path,
    component: str,
    *,
    model_dir: str | Path,
    output_path: str | Path | None = None,
    verbose: bool = False,
) -> Path:
    """Write a source-bound sidecar for a target-qualified prebuilt plan.

    ``component`` is one of ``text_encoder_0_plan``, ``denoiser_plan``,
    ``vae_decoder_plan`` (the recurrent four-frame step), or
    ``vae_decoder_first_frame_plan`` (the one-frame initializer).  The helper
    validates the serialized TensorRT I/O contract before writing
    ``<plan>.manifest.json`` by default.
    """

    if component not in _PLAN_COMPONENTS:
        raise ValueError(
            f"Unknown Wan2.2 plan component {component!r}; expected one of {list(_PLAN_COMPONENTS)}"
        )
    root = Path(model_dir).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    if not plan_file.is_file():
        raise FileNotFoundError(plan_file)
    plugin_paths, plugin_payloads = _ensure_plugin_payloads(verbose=verbose)
    # Keep the exact plugin libraries registered while TensorRT deserializes.
    del plugin_paths
    weights = {
        "_text_encoder_checkpoint": str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        "_vae_checkpoint": str(root / "Wan2.2_VAE.pth"),
    }
    identity = _component_source_identity(
        component, root, weights, plugin_payloads, digest_cache={}
    )
    plan = plan_file.read_bytes()
    _validate_serialized_engine_contract(plan, component)
    manifest = _prebuilt_manifest_payload(component, plan, identity["sha256"])
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else _prebuilt_manifest_path(plan_file)
    )
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _artifact_manifest(
    section_payloads: dict[str, bytes],
    source_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_sections = set(WAN22_MODEL_OWNED_BUNDLE_SECTIONS)
    if set(section_payloads) != expected_sections:
        raise ValueError(
            "Wan2.2 artifact manifest section mismatch: "
            f"actual={sorted(section_payloads)}, expected={sorted(expected_sections)}"
        )
    sections = {}
    for name in WAN22_MODEL_OWNED_BUNDLE_SECTIONS:
        entry: dict[str, Any] = {
            "sha256": _sha256_bytes(section_payloads[name]),
            "size": len(section_payloads[name]),
        }
        if name in source_identities:
            entry["source_sha256"] = source_identities[name]["sha256"]
            entry["source_inputs"] = source_identities[name]["inputs"]
        sections[name] = entry
    return {
        "schema": _ARTIFACT_MANIFEST_SCHEMA,
        "family": _FAMILY,
        "profile": _official_profile(),
        "runtime": "native_cpp_cuda_tensorrt",
        "sections": sections,
    }


def build_wan22_components(
    model_dir: str,
    *,
    config,
    weights: dict,
    precision: str = "bf16",
    verbose: bool = False,
    **_kwargs,
) -> dict:
    """Build all fixed official-profile engines for the native checkpoint."""

    if precision.lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Wan2.2-TI2V-5B requires BF16 DiT/T5 precision")
    _validate_requested_profile(config)

    # Keep all creators registered while building or validating plans.  The
    # exact libraries are bundled and included in every dependent plan's
    # source identity.
    plugin_paths, plugin_payloads = _ensure_plugin_payloads(verbose=verbose)
    digest_cache: dict[Path, str] = {}
    source_identities = {
        component: _component_source_identity(
            component,
            model_dir,
            weights,
            plugin_payloads,
            digest_cache=digest_cache,
        )
        for component in _PLAN_COMPONENTS
    }

    text_encoder = _read_prebuilt(
        config,
        "_wan2_2_prebuilt_text_encoder",
        "WAN22_PREBUILT_TEXT_ENCODER",
        component="text_encoder_0_plan",
        source_sha256=source_identities["text_encoder_0_plan"]["sha256"],
        manifest_config_key="_wan2_2_prebuilt_text_encoder_manifest",
        manifest_environment_key="WAN22_PREBUILT_TEXT_ENCODER_MANIFEST",
    )
    if text_encoder is None:
        text_encoder = build_native_umt5_encoder_engine(
            weights["_text_encoder_checkpoint"],
            source_gelu_plugin=plugin_paths["wan2_2_umt5_cuda_plugin_so"],
            source_softmax=True,
            source_rmsnorm=True,
            verbose=verbose,
        )

    denoiser = _read_prebuilt(
        config,
        "_wan2_2_prebuilt_denoiser",
        "WAN22_PREBUILT_DENOISER",
        component="denoiser_plan",
        source_sha256=source_identities["denoiser_plan"]["sha256"],
        manifest_config_key="_wan2_2_prebuilt_denoiser_manifest",
        manifest_environment_key="WAN22_PREBUILT_DENOISER_MANIFEST",
    )
    if denoiser is None:
        # Deliberately omit the old source_attention_plugin argument.  The
        # production plan may contain TensorRT layers and Wan2.2-owned pure
        # CUDA plugins, but never the former ATen/libtorch plugin.
        denoiser = build_dit_engine(
            model_dir,
            latent_frames=WAN22_TI2V_5B.latent_frames,
            latent_height=WAN22_TI2V_5B.latent_height,
            latent_width=WAN22_TI2V_5B.latent_width,
            num_layers=WAN22_TI2V_5B.num_layers,
            source_attention_plugin=None,
            cuda_bf16_plugin=str(plugin_paths["wan2_2_umt5_cuda_plugin_so"]),
            dit_cuda_plugin=str(plugin_paths["wan2_2_dit_cuda_plugin_so"]),
            dit_bf16_linear=True,
            dit_time_silu=True,
            dit_time_linear2=True,
            dit_time_projection=True,
            dit_block_layer_norm=True,
            dit_adaptive_norm=True,
            dit_rms_norm=True,
            dit_self_gated_residual=True,
            dit_ffn_gated_residual=True,
            dit_cross_affine_layer_norm=True,
            dit_final_projection=True,
            verbose=verbose,
        )

    vae_decoder = _read_prebuilt(
        config,
        "_wan2_2_prebuilt_vae_decoder",
        "WAN22_PREBUILT_VAE_DECODER",
        component="vae_decoder_plan",
        source_sha256=source_identities["vae_decoder_plan"]["sha256"],
        manifest_config_key="_wan2_2_prebuilt_vae_decoder_manifest",
        manifest_environment_key="WAN22_PREBUILT_VAE_DECODER_MANIFEST",
    )
    vae_decoder_first_frame = _read_prebuilt(
        config,
        "_wan2_2_prebuilt_vae_decoder_first_frame",
        "WAN22_PREBUILT_VAE_DECODER_FIRST_FRAME",
        component="vae_decoder_first_frame_plan",
        source_sha256=source_identities["vae_decoder_first_frame_plan"]["sha256"],
        manifest_config_key="_wan2_2_prebuilt_vae_decoder_first_frame_manifest",
        manifest_environment_key="WAN22_PREBUILT_VAE_DECODER_FIRST_FRAME_MANIFEST",
    )
    if vae_decoder is None or vae_decoder_first_frame is None:
        vae_weights = load_vae_step_weights(weights["_vae_checkpoint"])
        profile = official_vae_step_profile()
        if vae_decoder is None:
            vae_decoder = build_vae_step_engine(
                vae_weights,
                profile=profile,
                first_frame_only=False,
                verbose=verbose,
            )
        if vae_decoder_first_frame is None:
            vae_decoder_first_frame = build_vae_step_engine(
                vae_weights,
                profile=profile,
                first_frame_only=True,
                verbose=verbose,
            )

    plans = {
        "text_encoder_0_plan": bytes(text_encoder),
        "denoiser_plan": bytes(denoiser),
        "vae_decoder_plan": bytes(vae_decoder),
        "vae_decoder_first_frame_plan": bytes(vae_decoder_first_frame),
    }
    for component, plan in plans.items():
        _validate_serialized_engine_contract(plan, component)

    tokenizer_path = Path(weights["_tokenizer_dir"]) / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Wan2.2 tokenizer.json not found: {tokenizer_path}")
    tokenizer_json = tokenizer_path.read_bytes()
    section_payloads = {
        **plugin_payloads,
        **plans,
        "tokenizer.json": tokenizer_json,
    }

    return {
        "umt5_cuda_plugin": plugin_payloads["wan2_2_umt5_cuda_plugin_so"],
        "dit_cuda_plugin": plugin_payloads["wan2_2_dit_cuda_plugin_so"],
        "vae_cuda_plugin": plugin_payloads["wan2_2_vae_cuda_plugin_so"],
        "text_encoders": [("umt5_xxl", plans["text_encoder_0_plan"])],
        "denoiser": plans["denoiser_plan"],
        "vae_decoder": plans["vae_decoder_plan"],
        "vae_decoder_first_frame": plans["vae_decoder_first_frame_plan"],
        "tokenizer_json": tokenizer_json,
        "artifact_manifest": _artifact_manifest(section_payloads, source_identities),
    }


__all__ = [
    "build_wan22_components",
    "write_wan22_prebuilt_manifest",
]

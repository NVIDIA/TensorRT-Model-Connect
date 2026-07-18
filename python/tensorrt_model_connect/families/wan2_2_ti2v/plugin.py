# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect plugin for the native Wan2.2 TI2V-5B release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .checkpoint_mapper import VAE22_CONFIG
from .model_config import (
    OFFICIAL_NEGATIVE_PROMPT,
    WAN22_TI2V_5B,
    official_artifact_profile,
    validate_native_config,
)


WAN22_MODEL_OWNED_BUNDLE_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
    "wan2_2_ti2v_plugins.so",
)
WAN22_REQUIRED_BUNDLE_SECTIONS = (
    *WAN22_MODEL_OWNED_BUNDLE_SECTIONS,
    "config.json",
)
WAN22_EAGER_BUNDLE_SECTIONS = (
    "tokenizer.json",
    "config.json",
)
WAN22_LAZY_BUNDLE_SECTIONS = (
    "wan2_2_ti2v_plugins.so",
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
)

_COMPONENT_KEYS = {
    "plugin_contract",
    "plugin_library",
    "text_encoders",
    "denoiser",
    "vae_decoder",
    "vae_decoder_first_frame",
    "tokenizer_json",
    "artifact_manifest",
}
_PLAN_SECTIONS = {
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
}
_ARTIFACT_MANIFEST_SCHEMA = "trtmc.wan2_2_ti2v.bundle-artifacts.v4"
_PLUGIN_CONTRACT_KEYS = {
    "schema",
    "family",
    "semantic_abi",
    "source_digest",
    "creator_set",
    "runtime_abi",
    "cuda_architectures",
}
_PLUGIN_RUNTIME_ABI_KEYS = {
    "tensorrt_major",
    "tensorrt_minor",
    "cuda_major",
    "cudnn_major",
}


def _official_artifact_profile() -> dict:
    return official_artifact_profile()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _component_section_payloads(components: dict) -> dict[str, bytes]:
    if set(components) != _COMPONENT_KEYS:
        raise ValueError(
            "Wan2.2 bundle components must contain exactly "
            f"{sorted(_COMPONENT_KEYS)}; got {sorted(components)}"
        )
    text_encoders = components["text_encoders"]
    if (
        not isinstance(text_encoders, list)
        or len(text_encoders) != 1
        or not isinstance(text_encoders[0], tuple)
        or len(text_encoders[0]) != 2
        or text_encoders[0][0] != "umt5_xxl"
    ):
        raise ValueError("Wan2.2 bundle requires exactly one UMT5-XXL text encoder")
    payloads = {
        "text_encoder_0_plan": text_encoders[0][1],
        "denoiser_plan": components["denoiser"],
        "vae_decoder_plan": components["vae_decoder"],
        "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
        "tokenizer.json": components["tokenizer_json"],
        "wan2_2_ti2v_plugins.so": components["plugin_library"],
    }
    for name, payload in payloads.items():
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise TypeError(f"Wan2.2 bundle section {name!r} must be non-empty bytes")
    return {name: bytes(payload) for name, payload in payloads.items()}


def _validated_plugin_contract(components: dict) -> dict:
    contract = components["plugin_contract"]
    if not isinstance(contract, dict) or set(contract) != _PLUGIN_CONTRACT_KEYS:
        raise ValueError("Wan2.2 plugin_contract has an unsupported schema")
    if contract["schema"] != 1 or contract["family"] != "wan2_2_ti2v":
        raise ValueError("Wan2.2 plugin_contract has an unsupported version or family")
    for key in ("semantic_abi", "creator_set"):
        if not isinstance(contract[key], str) or not contract[key]:
            raise ValueError(f"Wan2.2 plugin_contract {key} must be a non-empty string")
    if not _is_sha256(contract["source_digest"]):
        raise ValueError("Wan2.2 plugin_contract source_digest must be a lowercase SHA256")
    runtime_abi = contract["runtime_abi"]
    if not isinstance(runtime_abi, dict) or set(runtime_abi) != _PLUGIN_RUNTIME_ABI_KEYS:
        raise ValueError("Wan2.2 plugin_contract runtime_abi has an unsupported schema")
    if any(
        not isinstance(runtime_abi[key], int) or isinstance(runtime_abi[key], bool)
        for key in _PLUGIN_RUNTIME_ABI_KEYS
    ):
        raise ValueError("Wan2.2 plugin_contract runtime ABI values must be integers")
    if runtime_abi["tensorrt_minor"] < 0 or any(
        runtime_abi[key] < 1
        for key in ("tensorrt_major", "cuda_major", "cudnn_major")
    ):
        raise ValueError(
            "Wan2.2 plugin_contract ABI majors must be positive and TensorRT minor nonnegative"
        )
    if contract["cuda_architectures"] != [103, 110]:
        raise ValueError("Wan2.2 plugin_contract must bind the SM103/SM110 fat binary")
    # JSON round-trip returns a detached tree and proves the object contains no
    # Python-only values before it is embedded into config.json.
    return json.loads(json.dumps(contract, sort_keys=True, separators=(",", ":")))


def _validate_artifact_manifest(components: dict, payloads: dict[str, bytes]) -> None:
    manifest = components["artifact_manifest"]
    if not isinstance(manifest, dict):
        raise TypeError("Wan2.2 artifact_manifest must be a dictionary")
    if set(manifest) != {"schema", "family", "profile", "runtime", "sections"}:
        raise ValueError("Wan2.2 artifact_manifest has an unsupported schema")
    expected_header = {
        "schema": _ARTIFACT_MANIFEST_SCHEMA,
        "family": "wan2_2_ti2v",
        "profile": _official_artifact_profile(),
        "runtime": "native_cpp_cuda_tensorrt",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Wan2.2 artifact_manifest has mismatched {key}: "
                f"{manifest.get(key)!r}; expected {expected!r}"
            )
    sections = manifest["sections"]
    if not isinstance(sections, dict) or set(sections) != set(payloads):
        raise ValueError(
            "Wan2.2 artifact_manifest must describe exactly the model-owned "
            f"bundle sections {sorted(payloads)}"
        )
    plugin_elf_sha256 = _sha256(payloads["wan2_2_ti2v_plugins.so"])
    plugin_contract_sha256 = _sha256(
        json.dumps(
            _validated_plugin_contract(components),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name, payload in payloads.items():
        entry = sections[name]
        expected_entry_keys = (
            {"sha256", "size", "source_sha256", "source_inputs"}
            if name in _PLAN_SECTIONS
            else {"sha256", "size"}
        )
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
            raise ValueError(
                f"Wan2.2 artifact manifest entry {name!r} must contain exactly "
                f"{sorted(expected_entry_keys)}"
            )
        if entry["sha256"] != _sha256(payload):
            raise ValueError(f"Wan2.2 artifact SHA256 mismatch for {name}")
        if (
            not isinstance(entry["size"], int)
            or isinstance(entry["size"], bool)
            or entry["size"] != len(payload)
        ):
            raise ValueError(f"Wan2.2 artifact size mismatch for {name}")
        if name in _PLAN_SECTIONS:
            if not _is_sha256(entry["source_sha256"]):
                raise ValueError(f"Wan2.2 artifact source identity is invalid for {name}")
            source_inputs = entry["source_inputs"]
            if not isinstance(source_inputs, list) or not source_inputs:
                raise ValueError(f"Wan2.2 artifact source inputs are missing for {name}")
            for source in source_inputs:
                if (
                    not isinstance(source, dict)
                    or set(source) != {"name", "sha256"}
                    or not isinstance(source["name"], str)
                    or not source["name"]
                    or not _is_sha256(source["sha256"])
                ):
                    raise ValueError(
                        f"Wan2.2 artifact source input is invalid for {name}: {source!r}"
                    )
            source_digests = {source["name"]: source["sha256"] for source in source_inputs}
            if len(source_digests) != len(source_inputs):
                raise ValueError(f"Wan2.2 artifact source inputs contain duplicates for {name}")
            if source_digests.get("plugin/contract.json") != plugin_contract_sha256:
                raise ValueError(
                    f"Wan2.2 plan is bound to a different AOT plugin contract: {name}"
                )
            if source_digests.get("plugin/elf") != plugin_elf_sha256:
                raise ValueError(
                    f"Wan2.2 plan is bound to a different AOT plugin ELF: {name}"
                )
            source_document = {
                "family": "wan2_2_ti2v",
                "component": name,
                "profile": _official_artifact_profile(),
                "inputs": source_inputs,
            }
            canonical_source = json.dumps(
                source_document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            if entry["source_sha256"] != _sha256(canonical_source):
                raise ValueError(f"Wan2.2 artifact source identity mismatch for {name}")


class Wan22TI2VPlugin:
    name = "wan2_2_ti2v"
    runtime_strategy = "diffusion_wan2_2_ti2v"
    pipeline_classes = ("WanModel", "WanPipeline")
    requires_tokenizer = True

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {
            "ti2v",
            "ti2v_5b",
            "wan2.2_ti2v_5b",
            "wan2_2_ti2v",
            "wan2_2_ti2v_5b",
            "wanmodel",
            "wanpipeline",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        root = Path(model_dir)
        config_path = root / "config.json"
        if not config_path.exists():
            raise ValueError(f"Wan2.2 TI2V requires native config.json in {root}")
        native_config = json.loads(config_path.read_text())
        validate_native_config(native_config)

        required = {
            "_transformer_dir": root,
            "_vae_checkpoint": root / "Wan2.2_VAE.pth",
            "_text_encoder_checkpoint": root / "models_t5_umt5-xxl-enc-bf16.pth",
            "_tokenizer_dir": root / "google" / "umt5-xxl",
        }
        missing = [
            str(path)
            for key, path in required.items()
            if key != "_transformer_dir" and not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Incomplete Wan2.2-TI2V-5B checkpoint; missing: " + ", ".join(missing)
            )
        tokenizer_json = required["_tokenizer_dir"] / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise FileNotFoundError(
                f"Incomplete Wan2.2-TI2V-5B checkpoint; missing: {tokenizer_json}"
            )

        config.raw["_wan2_2_native_config"] = native_config
        config.raw["_vae_config"] = dict(VAE22_CONFIG)
        return {key: str(path) for key, path in required.items()}

    def build_engine(
        self,
        config,
        weights: dict,
        _max_cache_length: int,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        **kwargs,
    ) -> bytes:
        del config, weights, _max_cache_length, precision, verbose, kwargs
        raise NotImplementedError("Wan2.2 TI2V uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        from .trt_builder import build_wan22_components

        return build_wan22_components(
            model_dir,
            config=config,
            weights=weights,
            precision=precision,
            verbose=verbose,
            **kwargs,
        )

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        payloads = _component_section_payloads(components)
        _validated_plugin_contract(components)
        _validate_artifact_manifest(components, payloads)
        return [(name, payloads[name]) for name in WAN22_MODEL_OWNED_BUNDLE_SECTIONS]

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        payloads = _component_section_payloads(components)
        plugin_contract = _validated_plugin_contract(components)
        _validate_artifact_manifest(components, payloads)
        result = self.get_diffusion_config(config)
        result["num_text_encoders"] = len(components["text_encoders"])
        result["artifact_manifest"] = components["artifact_manifest"]
        result["_trtmc_wan22_plugin_contract"] = plugin_contract
        result["runtime_contract"] = {
            "implementation": "native_cpp_cuda_tensorrt",
            "artifact_integrity": "sha256_size_v1",
            "bundle_trust_model": "trusted_executable_artifact",
            "executable_bundle_sections": ["wan2_2_ti2v_plugins.so"],
            "required_bundle_sections": list(WAN22_REQUIRED_BUNDLE_SECTIONS),
            "runtime_dependencies": [
                "trtmc_core",
                "cuda",
                "tensorrt",
                "cudnn",
                "cublaslt",
                "nvrtc",
            ],
            "forbidden_runtime_dependencies": [
                "python",
                "pytorch",
                "libpython",
                "libtorch",
            ],
        }
        result["bundle_loading"] = {
            "mode": "staged",
            "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
            "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
        }
        return result

    def diffusion_tokenizer_add_special_tokens(
        self, model_dir_path, *, detect_tokenizer_add_special_tokens
    ) -> bool:
        del model_dir_path, detect_tokenizer_add_special_tokens
        return False

    def diffusion_tokenizer_bundle_sections(
        self, model_dir_path, *, ensure_tokenizer_json
    ) -> list[tuple[str, bytes]]:
        # tokenizer.json is already part of the source-bound component set.
        # Returning an empty list prevents a second unverified filesystem read
        # after the artifact manifest has been constructed.
        del model_dir_path, ensure_tokenizer_json
        return []

    def get_diffusion_config(self, config) -> dict:
        raw = config.raw
        arch = WAN22_TI2V_5B
        height = int(raw.get("video_height", arch.video_height))
        width = int(raw.get("video_width", arch.video_width))
        frames = int(raw.get("video_num_frames", arch.video_num_frames))
        if (height, width, frames) != (
            arch.video_height,
            arch.video_width,
            arch.video_num_frames,
        ):
            raise ValueError(
                "Initial Wan2.2-TI2V-5B support is fixed to the official "
                "1280x704, 121-frame profile"
            )
        return {
            "diffusion_backend_type": "wan2_2_ti2v",
            "scheduler": "unipc_flow",
            "prediction_type": "flow_prediction",
            "num_train_timesteps": arch.train_timesteps,
            "num_inference_steps": int(raw.get("num_inference_steps", arch.num_inference_steps)),
            "guidance_scale": float(raw.get("guidance_scale", arch.guidance_scale)),
            "flow_shift": float(raw.get("flow_shift", arch.flow_shift)),
            "expand_timesteps": True,
            "video_height": height,
            "video_width": width,
            "video_num_frames": frames,
            "frame_rate": int(raw.get("frame_rate", arch.frame_rate)),
            "negative_prompt": str(raw.get("negative_prompt", OFFICIAL_NEGATIVE_PROMPT)),
            "z_dim": arch.z_dim,
            "dit_dim": arch.dim,
            "dit_num_heads": arch.num_heads,
            "dit_num_layers": arch.num_layers,
            "patch_size": list(arch.patch_size),
            "scale_factor_temporal": arch.scale_factor_temporal,
            "scale_factor_spatial": arch.scale_factor_spatial,
            "text_seq_len": arch.text_seq_len,
            "text_encoder_dim": arch.text_dim,
            "latents_mean": list(VAE22_CONFIG["latents_mean"]),
            "latents_std": list(VAE22_CONFIG["latents_std"]),
            "seed": int(raw.get("seed", 42)),
        }

    def get_bundle_config_overrides(self, config) -> dict:
        """Expose the native-checkpoint diffusion contract to the C++ runtime."""

        return self.get_diffusion_config(config)


plugin = Wan22TI2VPlugin()


__all__ = [
    "WAN22_MODEL_OWNED_BUNDLE_SECTIONS",
    "WAN22_REQUIRED_BUNDLE_SECTIONS",
    "Wan22TI2VPlugin",
    "plugin",
]

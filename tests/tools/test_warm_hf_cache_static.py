"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

import ast
import fnmatch
import json
import pathlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
HELPER_FUNCTIONS = {
    "_allow_patterns_for",
    "_component_has_weight",
    "_diffusers_missing_weight_components",
    "_is_diffusers_component_enabled",
    "_is_cached",
    "_snapshot_has_required_files",
}


def _load_cache_helpers() -> dict:
    tree = ast.parse(WARM_HF_CACHE.read_text())
    namespace = {
        "fnmatch": fnmatch,
        "json": json,
        "pathlib": pathlib,
        "_DIFFUSERS_WEIGHT_COMPONENTS": {
            "controlnet",
            "image_encoder",
            "text_encoder",
            "text_encoder_2",
            "transformer",
            "unet",
            "vae",
        },
        "_REQUIRED_FILES_BY_HF_ID": {
            "nvidia/Nemotron-Labs-Diffusion-8B": [
                "linear_spec_lora/adapter_config.json",
                "linear_spec_lora/adapter_model.safetensors",
            ],
            "openbmb/VoxCPM2": [
                "audiovae.pth",
                "tokenization_voxcpm2.py",
            ],
        },
        "_ENTRYPOINT_PATTERNS": ["config.json", "model_index.json", "*/config.json"],
        "_WEIGHT_PATTERNS": ["*.safetensors", "*.bin", "*.nemo"],
        "_HF_ALLOW_PATTERNS": ["config.json", "model.safetensors"],
        "_HF_EXTRA_ALLOW_PATTERNS": ["*.nemo"],
        "_HF_EXTRA_ALLOW_PATTERNS_BY_HF_ID": {
            "openbmb/VoxCPM2": ["audiovae.pth"],
        },
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_FUNCTIONS:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(WARM_HF_CACHE), "exec"), namespace)
    return namespace


def test_magpie_reference_dependencies_are_warmed() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps" in text
    assert "google/byt5-small" in text
    assert "microsoft/wavlm-base-plus" in text


def test_voxcpm2_audio_vae_is_warmed() -> None:
    text = WARM_HF_CACHE.read_text()
    assert '"openbmb/VoxCPM2"' in text
    assert '"audiovae.pth"' in text
    assert '"tokenization_voxcpm2.py"' in text


def test_nemotron_labs_diffusion_lora_files_are_warmed() -> None:
    text = WARM_HF_CACHE.read_text()
    assert '"chat_template.jinja"' in text
    assert '"linear_spec_lora/**"' in text
    assert '"linear_spec_lora/adapter_config.json"' in text
    assert '"linear_spec_lora/adapter_model.safetensors"' in text


def test_nemo_archives_count_as_complete_snapshots() -> None:
    text = WARM_HF_CACHE.read_text()
    assert (
        'if any(fnmatch.fnmatch(name, "*.nemo") for name in files):\n'
        "        return True"
    ) in text


def test_diffusers_snapshot_requires_component_weights(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    (snapshot / "text_encoder").mkdir(parents=True)
    (snapshot / "transformer").mkdir()
    (snapshot / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (snapshot / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (snapshot / "transformer" / "config.json").write_text("{}")

    assert not helpers["_snapshot_has_required_files"](snapshot)
    assert helpers["_diffusers_missing_weight_components"](snapshot) == [
        "transformer",
        "vae",
    ]


def test_diffusers_snapshot_accepts_all_component_weights(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    (snapshot / "text_encoder").mkdir(parents=True)
    (snapshot / "transformer").mkdir()
    (snapshot / "vae").mkdir()
    (snapshot / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (snapshot / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (
        snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    ).write_bytes(b"weights")
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    assert helpers["_diffusers_missing_weight_components"](snapshot) == []
    assert helpers["_snapshot_has_required_files"](snapshot)


def test_nemotron_labs_diffusion_snapshot_requires_lora_adapter(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="nvidia/Nemotron-Labs-Diffusion-8B")

    lora_dir = snapshot / "linear_spec_lora"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text("{}")
    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="nvidia/Nemotron-Labs-Diffusion-8B")

    (lora_dir / "adapter_model.safetensors").write_bytes(b"weights")
    assert helpers["_snapshot_has_required_files"](
        snapshot, hf_id="nvidia/Nemotron-Labs-Diffusion-8B")


def test_voxcpm2_snapshot_requires_audio_vae_and_custom_tokenizer(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="openbmb/VoxCPM2")

    (snapshot / "tokenization_voxcpm2.py").write_text("# tokenizer")
    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="openbmb/VoxCPM2")

    (snapshot / "audiovae.pth").write_bytes(b"vae")
    assert helpers["_snapshot_has_required_files"](
        snapshot, hf_id="openbmb/VoxCPM2")


def test_voxcpm2_cache_allow_patterns_include_audio_vae() -> None:
    helpers = _load_cache_helpers()

    assert helpers["_allow_patterns_for"]("openbmb/VoxCPM2") == [
        "config.json",
        "model.safetensors",
        "*.nemo",
        "audiovae.pth",
    ]


def test_cache_skip_uses_hf_local_resolution(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls: list[dict] = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return str(snapshot)

    helpers["snapshot_download"] = fake_snapshot_download

    assert helpers["_is_cached"]("org/model")
    assert calls == [{
        "args": ("org/model",),
        "kwargs": {
            "allow_patterns": ["config.json", "model.safetensors", "*.nemo"],
            "local_files_only": True,
        },
    }]


def test_cache_skip_rejects_unresolvable_local_revision() -> None:
    helpers = _load_cache_helpers()

    def fake_snapshot_download(*args, **kwargs):
        raise RuntimeError("revision is not available offline")

    helpers["snapshot_download"] = fake_snapshot_download

    assert not helpers["_is_cached"]("org/model")

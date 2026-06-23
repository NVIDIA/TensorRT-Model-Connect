"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

import ast
import fnmatch
import json
import pathlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
FAMILIES = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
HELPER_FUNCTIONS = {
    "_component_has_weight",
    "_diffusers_missing_weight_components",
    "_is_hf_file_cached",
    "_is_diffusers_component_enabled",
    "_is_cached",
    "_snapshot_has_required_files",
}


def _family_metadata_values(field: str) -> list[str]:
    values: list[str] = []
    for model_toml in sorted(FAMILIES.glob("*/MODEL.toml")):
        data = tomllib.loads(model_toml.read_text(encoding="utf-8"))
        for spec in data.get(field, []):
            if not isinstance(spec, str):
                continue
            parts = [part for part in spec.split("|")[1:] if part]
            values.extend(parts)
    return values


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
            "org/adapter-model": [
                "linear_spec_lora/adapter_config.json",
                "linear_spec_lora/adapter_model.safetensors",
            ],
        },
        "_ENTRYPOINT_PATTERNS": ["config.json", "model_index.json", "*/config.json"],
        "_WEIGHT_PATTERNS": ["*.safetensors", "*.bin", "*.nemo"],
        "_HF_ALLOW_PATTERNS": ["config.json", "model.safetensors"],
        "_HF_EXTRA_ALLOW_PATTERNS": ["*.nemo"],
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_FUNCTIONS:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(WARM_HF_CACHE), "exec"), namespace)
    return namespace


def test_family_reference_dependencies_are_metadata_driven() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "_family_hf_warm_dependencies" in text
    assert "family_hf_warm_dependencies" in text
    for value in _family_metadata_values("hf_warm_dependencies"):
        assert value not in text


def test_required_hf_files_are_metadata_driven() -> None:
    text = WARM_HF_CACHE.read_text()
    assert '"chat_template.jinja"' in text
    assert '"linear_spec_lora/**"' in text
    assert "family_hf_required_files_by_id" in text
    assert "adapter_config.json" not in text
    assert "adapter_model.safetensors" not in text


def test_family_file_assets_are_metadata_driven() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "_family_hf_warm_files" in text
    assert "family_hf_warm_files" in text
    assert "Warming family file assets" in text
    for value in _family_metadata_values("hf_warm_files"):
        assert value not in text


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
        "_class_name": "SyntheticDiffusionPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "SyntheticTransformer2DModel"],
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
        "_class_name": "SyntheticDiffusionPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "SyntheticTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (snapshot / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (
        snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    ).write_bytes(b"weights")
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    assert helpers["_diffusers_missing_weight_components"](snapshot) == []
    assert helpers["_snapshot_has_required_files"](snapshot)


def test_snapshot_requires_declared_extra_files(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="org/adapter-model")

    lora_dir = snapshot / "linear_spec_lora"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text("{}")
    assert not helpers["_snapshot_has_required_files"](
        snapshot, hf_id="org/adapter-model")

    (lora_dir / "adapter_model.safetensors").write_bytes(b"weights")
    assert helpers["_snapshot_has_required_files"](
        snapshot, hf_id="org/adapter-model")


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


def test_hf_file_cache_skip_uses_hf_local_resolution() -> None:
    helpers = _load_cache_helpers()
    calls: list[dict] = []

    def fake_hf_hub_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "/tmp/hf-cache/open_clip_pytorch_model.bin"

    helpers["hf_hub_download"] = fake_hf_hub_download

    assert helpers["_is_hf_file_cached"]("org/model", "weights.bin")
    assert calls == [{
        "args": ("org/model",),
        "kwargs": {
            "filename": "weights.bin",
            "local_files_only": True,
        },
    }]


def test_hf_file_cache_skip_rejects_missing_local_file() -> None:
    helpers = _load_cache_helpers()

    def fake_hf_hub_download(*args, **kwargs):
        raise RuntimeError("file is not available offline")

    helpers["hf_hub_download"] = fake_hf_hub_download

    assert not helpers["_is_hf_file_cached"]("org/model", "weights.bin")

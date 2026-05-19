"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import pathlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
HELPER_FUNCTIONS = {
    "_allow_patterns_for_hf_id",
    "_component_has_weight",
    "_diffusers_missing_weight_components",
    "_is_diffusers_component_enabled",
    "_snapshot_has_required_files",
    "_truthy_env",
}
HELPER_CONSTANTS = {
    "_DIFFUSERS_WEIGHT_COMPONENTS",
    "_ENTRYPOINT_PATTERNS",
    "_HF_ALLOW_PATTERNS",
    "_HF_EXTRA_ALLOW_PATTERNS",
    "_SANA_WM_FULL_ALLOW_PATTERNS",
    "_SANA_WM_HF_ID",
    "_SANA_WM_METADATA_ALLOW_PATTERNS",
    "_WEIGHT_PATTERNS",
}


def _load_cache_helpers() -> dict:
    tree = ast.parse(WARM_HF_CACHE.read_text())
    namespace = {
        "fnmatch": fnmatch,
        "json": json,
        "os": os,
        "pathlib": pathlib,
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id in HELPER_CONSTANTS
            for target in node.targets
        ):
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(WARM_HF_CACHE), "exec"), namespace)
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


def test_sana_wm_cache_warm_uses_metadata_only_by_default(monkeypatch) -> None:
    helpers = _load_cache_helpers()
    monkeypatch.delenv("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS", raising=False)

    assert helpers["_allow_patterns_for_hf_id"](
        "Efficient-Large-Model/SANA-WM_bidirectional"
    ) == ["README.md", "config.yaml"]


def test_sana_wm_cache_warm_can_opt_into_full_weights(monkeypatch) -> None:
    helpers = _load_cache_helpers()
    monkeypatch.setenv("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS", "1")

    allow_patterns = helpers["_allow_patterns_for_hf_id"](
        "Efficient-Large-Model/SANA-WM_bidirectional"
    )

    assert allow_patterns != ["README.md", "config.yaml"]
    assert "asset/sana_wm/**" in allow_patterns
    assert "dit/**" in allow_patterns
    assert "refiner/**" in allow_patterns


def test_non_sana_wm_cache_warm_uses_standard_allow_patterns() -> None:
    helpers = _load_cache_helpers()

    assert helpers["_allow_patterns_for_hf_id"]("nvidia/test-model") == (
        helpers["_HF_ALLOW_PATTERNS"] + helpers["_HF_EXTRA_ALLOW_PATTERNS"]
    )

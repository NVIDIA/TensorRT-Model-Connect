# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

import ast
import fnmatch
import json
import pathlib
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
FAMILIES = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
HELPER_FUNCTIONS = {
    "_component_has_weight",
    "_cache_repository_manifest",
    "_diffusers_missing_weight_components",
    "_is_hf_file_cached",
    "_is_diffusers_component_enabled",
    "_is_cached",
    "_manifest_has_eligible_testcase",
    "_snapshot_has_required_files",
    "_warm_exit_code",
    "_warm_snapshot",
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


def _family_metadata_specs(field: str) -> list[str]:
    specs: list[str] = []
    for model_toml in sorted(FAMILIES.glob("*/MODEL.toml")):
        data = tomllib.loads(model_toml.read_text(encoding="utf-8"))
        specs.extend(
            spec for spec in data.get(field, []) if isinstance(spec, str)
        )
    return specs


def _literal_string_list(name: str) -> set[str]:
    tree = ast.parse(WARM_HF_CACHE.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        return {item for item in value if isinstance(item, str)}
    raise AssertionError(f"Missing literal string list {name}")


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
        "_ENTRYPOINT_PATTERNS": [
            "config.json",
            "model_index.json",
            "*.yml",
            "*.yaml",
            "*/config.json",
        ],
        "_WEIGHT_PATTERNS": [
            "*.safetensors",
            "*.bin",
            "*.nemo",
            "model.npz",
            "elf_params.npz",
            "checkpoint_*/manifest.ocdbt",
        ],
        "_HF_ALLOW_PATTERNS": ["config.json", "model.safetensors"],
        "_HF_FAMILY_ALLOW_PATTERNS": ["nested/**"],
        "_HF_EXTRA_ALLOW_PATTERNS": ["*.nemo"],
        "_HF_DOWNLOAD_PATTERNS": [
            "config.json",
            "model.safetensors",
            "nested/**",
            "*.nemo",
        ],
        "_TOKENIZER_DOWNLOAD_PATTERNS": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "HfHubHTTPError": RuntimeError,
        "repo_folder_name": (
            lambda *, repo_id, repo_type: f"{repo_type}s--{repo_id.replace('/', '--')}"
        ),
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


def test_family_allow_patterns_are_used_for_cache_warming() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "family_hf_allow_patterns" in text
    assert "_HF_DOWNLOAD_PATTERNS" in text


def test_family_file_assets_are_metadata_driven() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "_family_hf_warm_files" in text
    assert "family_hf_warm_files" in text
    assert "Warming family file assets" in text
    global_allow_patterns = _literal_string_list("_HF_ALLOW_PATTERNS")
    for spec in _family_metadata_specs("hf_warm_files"):
        asset_name, hf_id, filename = spec.split("|", 2)
        assert spec not in text
        assert asset_name not in text
        assert hf_id not in text
        # Generic filenames may already be part of the global snapshot schema;
        # only family-unique filenames prove model-specific hard-coding here.
        if filename not in global_allow_patterns:
            assert filename not in text


def test_family_file_asset_guard_allows_global_filename_collisions() -> None:
    """A generic weight name is not itself family-specific hard-coding."""
    assert "pytorch_model.bin" in _literal_string_list("_HF_ALLOW_PATTERNS")
    assert any(
        spec.endswith("|pytorch_model.bin")
        for spec in _family_metadata_specs("hf_warm_files")
    )


def test_nemo_archives_count_as_complete_snapshots() -> None:
    text = WARM_HF_CACHE.read_text()
    assert (
        'if any(fnmatch.fnmatch(name, "*.nemo") for name in files):\n'
        "        return True"
    ) in text


def test_orbax_checkpoint_counts_as_complete_snapshot(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    checkpoint = snapshot / "checkpoint_0"
    checkpoint.mkdir(parents=True)
    (snapshot / "ELF-B-de-en.yml").write_text("model: elf\n")
    (checkpoint / "manifest.ocdbt").write_bytes(b"checkpoint")

    assert helpers["_snapshot_has_required_files"](snapshot)


def test_tokenizer_dependency_does_not_require_model_weights(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text("{}")

    assert helpers["_snapshot_has_required_files"](
        snapshot,
        require_weights=False,
    )
    assert not helpers["_snapshot_has_required_files"](
        snapshot,
        require_weights=True,
    )


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
            "allow_patterns": [
                "config.json",
                "model.safetensors",
                "nested/**",
                "*.nemo",
            ],
            "local_files_only": True,
        },
    }]


def test_cache_skip_rejects_unresolvable_local_revision() -> None:
    helpers = _load_cache_helpers()

    def fake_snapshot_download(*args, **kwargs):
        raise RuntimeError("revision is not available offline")

    helpers["snapshot_download"] = fake_snapshot_download

    assert not helpers["_is_cached"]("org/model")


def test_cache_skip_rejects_unresolved_snapshots_parent(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshots = tmp_path / "models--org--model" / "snapshots"
    snapshot = snapshots / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    helpers["snapshot_download"] = lambda *args, **kwargs: str(snapshots)

    assert not helpers["_is_cached"]("org/model")


def test_selective_warm_of_cached_snapshot_makes_no_network_download() -> None:
    helpers = _load_cache_helpers()
    downloads: list[str] = []
    helpers["_is_cached"] = lambda _hf_id, **_kwargs: True
    helpers["snapshot_download"] = lambda hf_id, **_kwargs: downloads.append(hf_id)

    status, detail = helpers["_warm_snapshot"](
        "org/model",
        gated=True,
        token_available=False,
        selective=True,
        local_only=False,
    )

    assert (status, detail) == ("cached", "")
    assert downloads == []


def test_uncached_gated_snapshot_without_token_fails_before_download() -> None:
    helpers = _load_cache_helpers()
    downloads: list[str] = []
    helpers["_is_cached"] = lambda _hf_id, **_kwargs: False
    helpers["snapshot_download"] = lambda hf_id, **_kwargs: downloads.append(hf_id)

    status, detail = helpers["_warm_snapshot"](
        "org/gated-model",
        gated=True,
        token_available=False,
        selective=True,
        local_only=False,
    )

    assert status == "failed"
    assert "no HF token" in detail
    assert downloads == []


def test_local_only_uncached_snapshot_never_downloads() -> None:
    helpers = _load_cache_helpers()
    downloads: list[str] = []
    helpers["_is_cached"] = lambda _hf_id, **_kwargs: False
    helpers["snapshot_download"] = lambda hf_id, **_kwargs: downloads.append(hf_id)

    status, detail = helpers["_warm_snapshot"](
        "org/model",
        gated=False,
        token_available=False,
        selective=True,
        local_only=True,
    )

    assert status == "failed"
    assert "not available in the local cache" in detail
    assert downloads == []


def test_strict_warm_failure_returns_nonzero() -> None:
    exit_code = _load_cache_helpers()["_warm_exit_code"]
    text = WARM_HF_CACHE.read_text(encoding="utf-8")

    assert exit_code(True, ["org/missing"]) == 1
    assert exit_code(True, []) == 0
    assert exit_code(False, ["org/missing"]) == 0
    assert (
        "strict_exit_code = _warm_exit_code(args.strict or bool(args.emit_cache_repos), warned)"
        in text
    )
    assert "if strict_exit_code:\n    sys.exit(strict_exit_code)" in text


def test_selected_cache_repository_manifest_is_unique_and_canonical(
    tmp_path: Path,
) -> None:
    manifest = _load_cache_helpers()["_cache_repository_manifest"]
    hub = tmp_path / "hub"
    for folder in ("models--org--one", "models--org--two"):
        (hub / folder).mkdir(parents=True)

    payload = manifest(
        ["org/one", "org/one", "org/two"],
        hub_cache=hub,
    )

    assert payload == {
        "schema_version": 1,
        "hub_cache": str(hub.resolve()),
        "repositories": [
            {
                "repo_id": "org/one",
                "repo_type": "model",
                "cache_folder": "models--org--one",
                "cache_path": str((hub / "models--org--one").resolve()),
            },
            {
                "repo_id": "org/two",
                "repo_type": "model",
                "cache_folder": "models--org--two",
                "cache_path": str((hub / "models--org--two").resolve()),
            },
        ],
    }


def test_selected_cache_repository_manifest_rejects_missing_or_linked_repo(
    tmp_path: Path,
) -> None:
    manifest = _load_cache_helpers()["_cache_repository_manifest"]
    hub = tmp_path / "hub"
    hub.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (hub / "models--org--linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="missing or not a directory"):
        manifest(["org/missing"], hub_cache=hub)
    with pytest.raises(RuntimeError, match="missing or not a directory"):
        manifest(["org/linked"], hub_cache=hub)


def test_cache_repository_evidence_cli_is_fail_closed() -> None:
    text = WARM_HF_CACHE.read_text(encoding="utf-8")

    assert '"--emit-cache-repos"' in text
    assert "if args.emit_cache_repos and not warned:" in text
    assert 'warned.append("cache-repository-evidence")' in text


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


def test_manifest_tier_filter_keeps_models_with_any_eligible_testcase() -> None:
    eligible = _load_cache_helpers()["_manifest_has_eligible_testcase"]
    excluded = {"nightly_only", "multi_device"}

    assert eligible(
        {
            "testcases": [
                {"name": "base"},
                {"name": "probe", "ci_tier": "nightly_only"},
            ]
        },
        excluded,
    )
    assert not eligible(
        {"testcases": [{"name": "tp", "ci_tier": "multi_device"}]},
        excluded,
    )
    assert not eligible({"ci_tier": "nightly_only"}, excluded)

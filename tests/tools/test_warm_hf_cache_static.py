# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract checks for scripts/warm_hf_cache.py."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
DOWNLOAD_WORKER = REPO_ROOT / "scripts" / "hf_cache_download_worker.py"
FAMILIES = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
HELPER_FUNCTIONS = {
    "_component_has_weight",
    "_cache_repository_manifest",
    "_diffusers_missing_weight_components",
    "_is_hf_file_cached",
    "_is_diffusers_component_enabled",
    "_is_cached",
    "_manifest_has_eligible_testcase",
    "_run_download_attempts",
    "_run_download_worker",
    "_snapshot_has_required_files",
    "_download_validation_error",
    "_warm_file",
    "_warm_exit_code",
    "_warm_snapshot",
    "_worker_command",
    "_worker_failure_detail",
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
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "time": time,
        "_DOWNLOAD_WORKER": DOWNLOAD_WORKER,
        "_DOWNLOAD_ATTEMPTS": 2,
        "_DEFAULT_ATTEMPT_TIMEOUT_SECONDS": 600.0,
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


def _write_fake_hub_package(root: Path, init_source: str) -> Path:
    package = root / "huggingface_hub"
    package.mkdir()
    (package / "__init__.py").write_text(init_source, encoding="utf-8")
    (package / "constants.py").write_text(
        "import os\nHF_HUB_CACHE = os.environ.get('FAKE_HF_CACHE', '/tmp/fake-hub')\n",
        encoding="utf-8",
    )
    (package / "file_download.py").write_text(
        "def repo_folder_name(*, repo_id, repo_type):\n"
        "    return f\"{repo_type}s--{repo_id.replace('/', '--')}\"\n",
        encoding="utf-8",
    )
    return package


def test_family_reference_dependencies_are_metadata_driven() -> None:
    text = WARM_HF_CACHE.read_text()
    assert "_family_hf_warm_dependencies" in text
    assert "family_hf_warm_dependencies" in text
    for value in _family_metadata_values("hf_warm_dependencies"):
        assert value not in text


def test_parent_process_only_uses_hub_downloads_for_local_resolution() -> None:
    tree = ast.parse(WARM_HF_CACHE.read_text(encoding="utf-8"))
    hub_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"snapshot_download", "hf_hub_download"}
    ]

    assert len(hub_calls) == 2
    for call in hub_calls:
        local_only = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "local_files_only"),
            None,
        )
        assert isinstance(local_only, ast.Constant)
        assert local_only.value is True

    worker_text = DOWNLOAD_WORKER.read_text(encoding="utf-8")
    assert "snapshot_download(args.repo_id" in worker_text
    assert "hf_hub_download(args.repo_id" in worker_text


def test_online_download_contract_is_bounded_and_retryable() -> None:
    text = WARM_HF_CACHE.read_text(encoding="utf-8")

    assert "_DOWNLOAD_ATTEMPTS = 2" in text
    assert "_DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 600.0" in text
    assert '"--attempt-timeout-seconds"' in text
    assert 'environment["HF_HUB_DISABLE_XET"] = "1"' in text
    assert 'environment.pop("HF_HUB_DISABLE_XET", None)' in text
    assert 'parser.error("--fail-fast requires --strict or --emit-cache-repos")' in text


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


def test_cache_warm_patterns_cover_shared_snapshot_contract() -> None:
    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    warm_patterns = _literal_string_list("_HF_ALLOW_PATTERNS")
    warm_patterns.update(_family_metadata_specs("hf_allow_patterns"))
    warm_patterns.add("*.nemo")

    assert set(hf_snapshot_allow_patterns()) <= warm_patterns


def test_shared_cache_patterns_cover_builder_and_offline_snapshot_metadata() -> None:
    tree = ast.parse(WARM_HF_CACHE.read_text())
    allow_patterns = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_HF_ALLOW_PATTERNS"
            for target in node.targets
        )
    )

    assert "processor_config.json" in allow_patterns
    assert ".gitattributes" in allow_patterns


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
    helpers["_run_download_attempts"] = (
        lambda _operation, hf_id, **_kwargs: downloads.append(hf_id)
    )

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
    helpers["_run_download_attempts"] = (
        lambda _operation, hf_id, **_kwargs: downloads.append(hf_id)
    )

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
    helpers["_run_download_attempts"] = (
        lambda _operation, hf_id, **_kwargs: downloads.append(hf_id)
    )

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


def test_file_cache_and_local_only_paths_never_launch_worker() -> None:
    helpers = _load_cache_helpers()
    downloads: list[str] = []
    helpers["_run_download_attempts"] = (
        lambda _operation, hf_id, **_kwargs: downloads.append(hf_id)
    )
    helpers["_is_hf_file_cached"] = lambda _hf_id, _filename: True

    assert helpers["_warm_file"](
        "org/model",
        "weights.bin",
        selective=True,
        local_only=False,
    ) == ("cached", "")

    helpers["_is_hf_file_cached"] = lambda _hf_id, _filename: False
    status, detail = helpers["_warm_file"](
        "org/model",
        "weights.bin",
        selective=False,
        local_only=True,
    )
    assert status == "failed"
    assert "local cache" in detail
    assert downloads == []


def test_download_attempts_retry_once_with_xet_disabled(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls: list[bool] = []

    def fake_worker(*_args, disable_xet: bool, **_kwargs):
        calls.append(disable_xet)
        if len(calls) == 1:
            return None, "synthetic transport failure"
        return str(snapshot), ""

    helpers["_run_download_worker"] = fake_worker
    path, detail = helpers["_run_download_attempts"](
        "snapshot",
        "org/model",
        timeout_seconds=17,
        allow_patterns=["config.json", "model.safetensors"],
    )

    assert path == str(snapshot)
    assert calls == [False, True]
    assert "attempt 1/2 (default transfer backend" in detail
    assert "synthetic transport failure" in detail
    assert "attempt 2/2 (Xet disabled" in detail
    assert detail.endswith("succeeded")


def test_download_attempts_reject_incomplete_success_twice(tmp_path: Path) -> None:
    helpers = _load_cache_helpers()
    incomplete = tmp_path / "snapshots" / "partial"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}")
    calls: list[bool] = []

    def fake_worker(*_args, disable_xet: bool, **_kwargs):
        calls.append(disable_xet)
        return str(incomplete), ""

    helpers["_run_download_worker"] = fake_worker
    path, detail = helpers["_run_download_attempts"](
        "snapshot",
        "org/model",
        timeout_seconds=17,
        allow_patterns=["config.json", "model.safetensors"],
    )

    assert path is None
    assert calls == [False, True]
    assert detail.count("failed validation") == 2
    assert "downloaded snapshot is incomplete" in detail


def test_worker_timeout_is_bounded_and_second_attempt_environment_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers = _load_cache_helpers()
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    default_path, default_detail = helpers["_run_download_worker"](
        "file",
        "org/model",
        timeout_seconds=3.5,
        disable_xet=False,
        filename="weights.bin",
    )
    fallback_path, fallback_detail = helpers["_run_download_worker"](
        "file",
        "org/model",
        timeout_seconds=3.5,
        disable_xet=True,
        filename="weights.bin",
    )

    assert default_path is fallback_path is None
    assert "timed out and was killed after 3.5s" in default_detail
    assert "timed out and was killed after 3.5s" in fallback_detail
    assert len(calls) == 2
    assert calls[0]["timeout"] == 3.5
    assert "HF_HUB_DISABLE_XET" not in calls[0]["env"]
    assert calls[1]["env"]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[0]["command"][0] == sys.executable
    assert calls[0]["command"][1] == str(DOWNLOAD_WORKER)


@pytest.mark.parametrize(
    "worker_args, expected_path_env",
    [
        (
            [
                "--operation",
                "snapshot",
                "--repo-id",
                "org/model",
                "--allow-patterns-json",
                '["config.json", "model.safetensors"]',
            ],
            "FAKE_SNAPSHOT_PATH",
        ),
        (
            [
                "--operation",
                "file",
                "--repo-id",
                "org/model",
                "--filename",
                "weights.bin",
            ],
            "FAKE_FILE_PATH",
        ),
    ],
)
def test_download_worker_sets_request_defaults_before_hub_import(
    tmp_path: Path,
    worker_args: list[str],
    expected_path_env: str,
) -> None:
    _write_fake_hub_package(
        tmp_path,
        """
import os

assert os.environ.get("HF_HUB_ETAG_TIMEOUT") == "30"
assert os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") == "60"
assert os.environ.get("HF_HUB_DISABLE_XET") == "1"

def snapshot_download(repo_id, *, allow_patterns):
    assert repo_id == "org/model"
    assert allow_patterns == ["config.json", "model.safetensors"]
    return os.environ["FAKE_SNAPSHOT_PATH"]

def hf_hub_download(repo_id, *, filename):
    assert repo_id == "org/model"
    assert filename == "weights.bin"
    return os.environ["FAKE_FILE_PATH"]
""",
    )
    snapshot = tmp_path / "snapshot"
    downloaded_file = tmp_path / "weights.bin"
    environment = os.environ.copy()
    environment.pop("HF_HUB_ETAG_TIMEOUT", None)
    environment.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["PYTHONPATH"] = str(tmp_path)
    environment["FAKE_SNAPSHOT_PATH"] = str(snapshot)
    environment["FAKE_FILE_PATH"] = str(downloaded_file)

    completed = subprocess.run(
        [sys.executable, str(DOWNLOAD_WORKER), *worker_args],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"path": environment[expected_path_env]}


def test_fail_fast_requires_fail_closed_mode_and_stops_after_two_attempts(
    tmp_path: Path,
) -> None:
    log = tmp_path / "online-calls.log"
    _write_fake_hub_package(
        tmp_path,
        """
import os

def snapshot_download(repo_id, **kwargs):
    if kwargs.get("local_files_only"):
        raise RuntimeError("not cached")
    with open(os.environ["FAKE_HF_CALL_LOG"], "a", encoding="utf-8") as stream:
        stream.write(f"{repo_id}|{os.environ.get('HF_HUB_DISABLE_XET', '')}\\n")
    raise RuntimeError("synthetic online failure")

def hf_hub_download(repo_id, **kwargs):
    if kwargs.get("local_files_only"):
        raise RuntimeError("not cached")
    raise RuntimeError(f"unexpected file download for {repo_id}")
""",
    )
    models_file = tmp_path / "models.txt"
    models_file.write_text("albert-base\ngpt2-125m\n", encoding="utf-8")
    environment = os.environ.copy()
    # Simulate a runner/image override. The parent must still make attempt one
    # use the default backend and only disable Xet for attempt two.
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["PYTHONPATH"] = str(tmp_path)
    environment["FAKE_HF_CALL_LOG"] = str(log)
    environment["FAKE_HF_CACHE"] = str(tmp_path / "hub")

    warning_mode = subprocess.run(
        [sys.executable, str(WARM_HF_CACHE), "--fail-fast"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )
    assert warning_mode.returncode == 2
    assert "--fail-fast requires --strict or --emit-cache-repos" in warning_mode.stderr
    assert not log.exists()

    fail_closed_modes = [
        ["--strict"],
        ["--emit-cache-repos", str(tmp_path / "cache-evidence.json")],
    ]
    for mode_args in fail_closed_modes:
        log.unlink(missing_ok=True)
        fail_closed = subprocess.run(
            [
                sys.executable,
                str(WARM_HF_CACHE),
                "--models-file",
                str(models_file),
                *mode_args,
                "--fail-fast",
                "--attempt-timeout-seconds",
                "5",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=15,
        )

        assert fail_closed.returncode == 1
        assert "Fail-fast: stopping after exhausted item" in fail_closed.stderr
        calls = log.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 2
        assert calls[0].split("|", 1)[0] == calls[1].split("|", 1)[0]
        assert calls[0].endswith("|")
        assert calls[1].endswith("|1")


def test_strict_warm_failure_returns_nonzero() -> None:
    exit_code = _load_cache_helpers()["_warm_exit_code"]
    text = WARM_HF_CACHE.read_text(encoding="utf-8")

    assert exit_code(True, ["org/missing"]) == 1
    assert exit_code(True, []) == 0
    assert exit_code(False, ["org/missing"]) == 0
    assert "fail_closed = args.strict or bool(args.emit_cache_repos)" in text
    assert "strict_exit_code = _warm_exit_code(fail_closed, warned)" in text
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

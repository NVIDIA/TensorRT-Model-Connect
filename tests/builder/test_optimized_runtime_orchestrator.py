# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.optimized_runtime.manifest import (
    AmbiguousImplementationError,
    ImplementationRequest,
    ManifestDiscoveryError,
    ManifestValidationError,
    load_implementation_manifest,
)
from tensorrt_model_connect.optimized_runtime.orchestrator import (
    _probe_model_default_revision,
    discover_family_implementations_for_model,
    select_delegated_build,
    try_build_optimized_runtime,
)
from tensorrt_model_connect.optimized_runtime.target import (
    TargetResolutionError,
    normalize_target_descriptor,
)
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC


_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _model_revision(_model_id: str) -> str:
    return _REVISION


def _read_bundle(path: Path) -> tuple[dict, dict[str, bytes]]:
    with path.open("rb") as source:
        assert source.read(8) == BUNDLE_MAGIC
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
        payload_offset = 16 + header_length
        sections: dict[str, bytes] = {}
        for name, metadata in header["sections"].items():
            source.seek(payload_offset + metadata["offset"])
            sections[name] = source.read(metadata["size"])
    return header, sections


_ADAPTER = r"""import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("operation")
parser.add_argument("--request", required=True)
parser.add_argument("--output")
args = parser.parse_args()
request = json.loads(Path(args.request).read_text(encoding="utf-8"))
if request["parameters"].get("verify_launch_context"):
    assert os.environ.get("TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE") == "1"
    assert "active_device_ordinal" not in request["target"]
    assert "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in json.dumps(request)
if args.operation == "probe":
    print(json.dumps({
        "schema_version": 1,
        "supported": request["parameters"].get("precision") == "fp16",
        **({
            "profile_id": "a100-fp16-b4",
        } if request["parameters"].get("precision") == "fp16" else {
            "reason": "only the fp16 profile is supported",
        }),
    }))
elif args.operation == "build":
    output = Path(args.output)
    artifacts = output / "artifacts"
    (artifacts / "engine.dir").mkdir(parents=True)
    (artifacts / "engine.dir" / "engine.plan").write_bytes(b"engine")
    (artifacts / "libtrtmc_impl_fake.so").write_bytes(b"fake-dso")
    descriptor = {
        "schema_version": 1,
        "build_binding": request["build_binding"],
        "bundle_config": {
            "capsule_owned": True,
        },
        "bundle_info": {
            "family": "fake-optimized-family",
            "precision": "capsule-defined",
        },
        "private": {"capsule_owns_this": True},
    }
    (output / "descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")
    print(json.dumps({
        "schema_version": 1,
        "descriptor": "descriptor.json",
        "artifacts": "artifacts",
    }))
"""


def _capsule(
    root: Path,
    *,
    name: str = "fake",
    implementation_id: str = "fake-a100-runtime",
    model_id: str = "Example/Model",
) -> Path:
    capsule = root / name
    (capsule / "builder").mkdir(parents=True)
    (capsule / "builder" / "adapter.py").write_text(_ADAPTER, encoding="utf-8")
    manifest = capsule / "IMPLEMENTATION.toml"
    manifest.write_text(
        f'''schema_version = 1
implementation_id = "{implementation_id}"
runtime_kind = "optimized"
downstream_runtime = "fake-runtime"
downstream_version = "1.2.3"
downstream_commit = "0123456789abcdef"

[model]
id = "{model_id}"
revisions = ["0123456789abcdef0123456789abcdef01234567"]

[target]
os = "linux"
architecture = "x86_64"
gpu_architecture = "sm80"

[build]
entrypoint = "builder/adapter.py"
timeout_seconds = 30

[runtime]
library = "libtrtmc_impl_fake.so"
abi = 1
artifact_layout = "directory-tree-v1"
''',
        encoding="utf-8",
    )
    return manifest


def _make_invalid_toml(manifest: Path) -> None:
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\nbroken = [\n",
        encoding="utf-8",
    )


def _target() -> dict[str, object]:
    return {
        "os": "linux",
        "architecture": "x86_64",
        "platform_kind": "discrete",
        "gpu_architecture": "sm80",
        "gpu_memory_mib": 81920,
        "gpu_count": 1,
        "gpu_name": "NVIDIA A100 80GB PCIe",
    }


def test_target_descriptor_normalizes_aliases_without_runtime_knowledge() -> None:
    facts = normalize_target_descriptor(
        {
            "schema_version": 1,
            "target_id": "a100-node",
            "gpu": {
                "name": "NVIDIA A100 80GB PCIe",
                "compute_capability": "8.0",
                "memory_mib": 81920,
                "count": 1,
            },
            "system": {"os": "GNU/Linux", "arch": "amd64", "kind": "dgpu"},
        }
    )
    assert facts == {
        **_target(),
        "target_id": "a100-node",
    }


def test_family_discovery_rejects_nested_adapter_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.optimized_runtime.orchestrator as orchestrator

    family_root = tmp_path / "example_family"
    _capsule(family_root / "nested", name="runtime")
    monkeypatch.setattr(
        orchestrator,
        "family_implementation_roots",
        lambda _family: (family_root,),
    )

    with pytest.raises(ManifestDiscoveryError, match="<family>/<adapter>"):
        discover_family_implementations_for_model("example_family", "Example/Model")


def test_target_descriptor_rejects_unknown_facts() -> None:
    with pytest.raises(TargetResolutionError, match="Unknown target facts"):
        normalize_target_descriptor(
            {
                "schema_version": 1,
                "gpu": {
                    "name": "GPU",
                    "compute_capability": 80,
                    "memory_mib": 1,
                    "count": 1,
                    "vendor_policy": "capsule-specific",
                },
                "system": {"os": "linux", "arch": "x86_64", "kind": "discrete"},
            }
        )


def test_full_generic_build_writes_self_contained_delegated_bundle(
    tmp_path: Path,
) -> None:
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision="",
        parameters={"precision": "fp16", "quantization": "none"},
        manifests=(manifest,),
        current_target_probe=_target,
        model_revision_probe=_model_revision,
    )

    assert selection is not None
    assert selection.manifest.implementation_id == "fake-a100-runtime"
    header, sections = _read_bundle(output)
    config = json.loads(sections["config.json"])
    descriptor = json.loads(sections["optimized_runtime.json"])
    private = json.loads(sections["implementation.json"])
    assert header["model_type"] == "optimized_runtime"
    assert config == {"capsule_owned": True}
    assert header["family"] == "fake-optimized-family"
    assert header["precision"] == "capsule-defined"
    assert descriptor["implementation_id"] == "fake-a100-runtime"
    assert descriptor["runtime_library"] == "libtrtmc_impl_fake.so"
    assert descriptor["factory_abi"] == 1
    assert descriptor["artifact"]["file_count"] == 2
    assert private["private"] == {"capsule_owns_this": True}
    assert sections["optimized_runtime_artifacts/engine.dir/engine.plan"] == b"engine"
    assert sections["optimized_runtime_artifacts/libtrtmc_impl_fake.so"] == b"fake-dso"


def test_current_target_preserves_active_device_ordinal_only_as_launch_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.optimized_runtime.target as target_module

    class HeterogeneousCudaRuntime:
        cudaError_t = SimpleNamespace(cudaSuccess=0)

        @staticmethod
        def cudaGetDevice():
            return 0, 1

        @staticmethod
        def cudaGetDeviceProperties(device: int):
            assert device == 1
            return 0, SimpleNamespace(
                name=b"NVIDIA A100 80GB PCIe\x00",
                totalGlobalMem=81920 * 1024 * 1024,
                major=8,
                minor=0,
            )

        @staticmethod
        def cudaGetDeviceCount():
            return 0, 2

    monkeypatch.setattr(target_module, "_cuda_runtime", lambda: HeterogeneousCudaRuntime)
    monkeypatch.setattr(target_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(target_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-slow,GPU-a100")
    monkeypatch.delenv("TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE", raising=False)
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision="",
        parameters={"precision": "fp16", "verify_launch_context": True},
        manifests=(manifest,),
        model_revision_probe=_model_revision,
    )

    assert selection is not None
    request_payload = selection.request.to_json()
    assert request_payload["target"]["gpu_count"] == 2
    assert request_payload["target"]["gpu_name"] == "NVIDIA A100 80GB PCIe"
    assert "active_device_ordinal" not in request_payload["target"]
    assert "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in json.dumps(request_payload)
    assert b"TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in output.read_bytes()


def test_selected_family_fails_closed_for_any_malformed_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.optimized_runtime.orchestrator as orchestrator

    root = tmp_path / "capsules"
    _capsule(
        root,
        name="requested",
        implementation_id="requested-runtime",
        model_id="Example/Requested-Model",
    )
    unrelated = _capsule(
        root,
        name="unrelated",
        implementation_id="unrelated-runtime",
        model_id="Example/Unrelated-Model",
    )
    _make_invalid_toml(unrelated)
    monkeypatch.setattr(orchestrator, "family_implementation_roots", lambda _family: (root,))

    with pytest.raises(ManifestValidationError, match="invalid TOML"):
        try_build_optimized_runtime(
            "Example/Requested-Model",
            tmp_path / "model.trtfb",
            target="current",
            configured_revision="0123456789abcdef0123456789abcdef01234567",
            family_name="example",
            parameters={"precision": "fp16"},
            current_target_probe=_target,
        )


def test_public_native_build_without_an_owning_family_skips_adapter_discovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capsules"
    unrelated = _capsule(
        root,
        model_id="Example/Unrelated-Model",
    )
    _make_invalid_toml(unrelated)
    selection = try_build_optimized_runtime(
        "Example/Native-Only-Model",
        tmp_path / "native.trtfb",
        target="current",
        configured_revision="",
        current_target_probe=lambda: pytest.fail("native-only request resolved a target"),
    )

    assert selection is None


def test_public_build_fails_closed_for_malformed_model_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.optimized_runtime.orchestrator as orchestrator

    root = tmp_path / "capsules"
    candidate = _capsule(
        root,
        model_id="Example/Requested-Model",
    )
    _make_invalid_toml(candidate)
    monkeypatch.setattr(orchestrator, "family_implementation_roots", lambda _family: (root,))

    with pytest.raises(ManifestValidationError, match="invalid TOML"):
        try_build_optimized_runtime(
            "Example/Requested-Model",
            tmp_path / "model.trtfb",
            target="current",
            configured_revision="0123456789abcdef0123456789abcdef01234567",
            family_name="example",
            parameters={"precision": "fp16"},
            current_target_probe=_target,
        )


def test_unsupported_probe_returns_native_fallback_without_building(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "model.trtfb"
    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision="",
        parameters={"precision": "fp8"},
        manifests=(manifest,),
        current_target_probe=_target,
        model_revision_probe=_model_revision,
    )
    assert selection is None
    assert not output.exists()


def test_bare_model_id_does_not_select_a_stale_capsule_revision(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "native.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision=None,
        parameters={"precision": "fp16"},
        manifests=(manifest,),
        current_target_probe=lambda: pytest.fail("stale profile resolved a target"),
        model_revision_probe=lambda _model_id: "f" * 40,
    )

    assert selection is None
    assert not output.exists()


def test_bare_model_id_revision_lookup_failure_retains_native_path(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "native.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision=None,
        parameters={"precision": "fp16"},
        manifests=(manifest,),
        current_target_probe=lambda: pytest.fail("unresolved model resolved a target"),
        model_revision_probe=lambda _model_id: None,
    )

    assert selection is None
    assert not output.exists()


def test_bare_model_id_resolves_the_warmed_snapshot_when_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType

    snapshot = tmp_path / "models--Example--Model" / "snapshots" / _REVISION
    snapshot.mkdir(parents=True)

    class OfflineApi:
        def model_info(self, _model_id: str):
            raise RuntimeError("offline")

    hub = ModuleType("huggingface_hub")
    hub.HfApi = OfflineApi
    hub.snapshot_download = lambda _model_id, *, local_files_only: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert _probe_model_default_revision("Example/Model") == _REVISION


def test_bare_model_id_selects_only_the_resolved_revision_with_multiple_capsules(
    tmp_path: Path,
) -> None:
    first_path = _capsule(
        tmp_path / "capsules",
        name="first",
        implementation_id="first-revision-runtime",
    )
    second_path = _capsule(
        tmp_path / "capsules",
        name="second",
        implementation_id="second-revision-runtime",
    )
    second_revision = "abcdef0123456789abcdef0123456789abcdef01"
    second_path.write_text(
        second_path.read_text(encoding="utf-8").replace(_REVISION, second_revision),
        encoding="utf-8",
    )
    manifests = tuple(load_implementation_manifest(path) for path in (first_path, second_path))
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target="current",
        configured_revision=None,
        parameters={"precision": "fp16"},
        manifests=manifests,
        current_target_probe=_target,
        model_revision_probe=lambda _model_id: second_revision,
    )

    assert selection is not None
    assert selection.manifest.implementation_id == "second-revision-runtime"
    assert selection.request.model_revision == second_revision


def test_unavailable_active_target_returns_native_fallback(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_capsule(tmp_path / "capsules"))
    output = tmp_path / "native.trtfb"

    def unavailable_target() -> dict[str, object]:
        raise TargetResolutionError("CUDA target probing is unavailable")

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        target=None,
        configured_revision="",
        parameters={"precision": "fp16"},
        manifests=(manifest,),
        current_target_probe=unavailable_target,
        model_revision_probe=_model_revision,
    )

    assert selection is None
    assert not output.exists()


def test_two_supported_capsules_fail_authoritative_runtime_invariant(
    tmp_path: Path,
) -> None:
    first = load_implementation_manifest(
        _capsule(tmp_path, name="first", implementation_id="first-runtime")
    )
    second = load_implementation_manifest(
        _capsule(tmp_path, name="second", implementation_id="second-runtime")
    )
    request = ImplementationRequest(
        model_id="Example/Model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        target=_target(),
        parameters={"precision": "fp16"},
    )
    with pytest.raises(AmbiguousImplementationError, match="first-runtime.*second-runtime"):
        select_delegated_build((second, first), request)

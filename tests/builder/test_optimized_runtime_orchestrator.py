# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.runtime_provider.manifest import (
    AmbiguousImplementationError,
    ImplementationRequest,
    load_implementation_manifest,
)
from tensorrt_model_connect.runtime_provider.provider_process import (
    BuildAdapterError,
    ProbeResult,
)
from tensorrt_model_connect.runtime_provider.orchestrator import (
    build_selected_implementation,
    discover_family_implementations,
    select_delegated_build,
    try_build_optimized_runtime,
)
from tensorrt_model_connect.runtime_provider.target import (
    TargetResolutionError,
)
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC


_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _snapshot(
    root: Path,
    *,
    model_id: str = "Example/Model",
    revision: str = _REVISION,
) -> Path:
    organization, name = model_id.split("/", 1)
    snapshot = root / f"models--{organization}--{name}" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


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
profile_matches = (
    request["model"]["id"] == "__SUPPORTED_MODEL_ID__"
    and request["model"]["revision"] == "__SUPPORTED_REVISION__"
    and request["parameters"].get("precision") == "fp16"
)
if request["parameters"].get("verify_launch_context"):
    assert os.environ.get("TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE") == "1"
    assert "active_device_ordinal" not in request["target"]
    assert "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in json.dumps(request)
if args.operation == "probe":
    print(json.dumps({
        "schema_version": 1,
        "supported": profile_matches,
        **({
            "profile_id": "a100-fp16-b4",
            "profile_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        } if profile_matches else {
            "reason": "no qualified profile matches this request",
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
    downstream_runtime: str = "fake-runtime",
    model_id: str = "Example/Model",
    revision: str = _REVISION,
) -> Path:
    capsule = root / name
    (capsule / "builder").mkdir(parents=True)
    (capsule / "builder" / "adapter.py").write_text(
        _ADAPTER.replace("__SUPPORTED_MODEL_ID__", model_id).replace(
            "__SUPPORTED_REVISION__", revision
        ),
        encoding="utf-8",
    )
    profiles = capsule / "profiles"
    profiles.mkdir()
    (profiles / "a100-fp16.toml").write_text(
        f'''schema_version = 1
profile_id = "a100-fp16-b4"

[model]
id = "{model_id}"
revisions = ["{revision}"]

[target]
gpu_architecture = "sm80"
''',
        encoding="utf-8",
    )
    manifest = capsule / "IMPLEMENTATION.toml"
    manifest.write_text(
        f'''schema_version = 2
implementation_id = "{implementation_id}"
downstream_runtime = "{downstream_runtime}"
downstream_version = "1.2.3"
downstream_commit = "0123456789abcdef"

[build]
entrypoint = "builder/adapter.py"
timeout_seconds = 30

[runtime]
library = "libtrtmc_impl_fake.so"
abi = 1
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


def test_family_discovery_ignores_nested_non_adapter_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "example_family"
    _capsule(family_root / "nested", name="runtime")
    monkeypatch.setattr(
        orchestrator,
        "family_implementation_root",
        lambda _family: family_root,
    )

    assert not discover_family_implementations("example_family")


def test_family_discovery_uses_the_already_selected_family_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "example"
    manifest = _capsule(family_root)
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)

    discovered = discover_family_implementations("example")

    assert [item.path for item in discovered] == [manifest.resolve()]


def test_full_generic_build_writes_self_contained_delegated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "family"
    _capsule(family_root)
    model_source = _snapshot(tmp_path / "cache")
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)
    monkeypatch.setattr(
        orchestrator,
        "_probe_current_target_with_device",
        lambda: (_target(), 0),
    )
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        str(model_source),
        output,
        family_name="example",
        parameters={"precision": "fp16", "quantization": "none"},
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
    import tensorrt_model_connect.runtime_provider.target as target_module

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
    family_root = tmp_path / "family"
    _capsule(family_root)
    model_source = _snapshot(tmp_path / "cache")
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        str(model_source),
        output,
        family_name="example",
        parameters={"precision": "fp16", "verify_launch_context": True},
    )

    assert selection is not None
    request_payload = selection.request.to_json()
    assert request_payload["target"]["gpu_count"] == 2
    assert request_payload["target"]["gpu_name"] == "NVIDIA A100 80GB PCIe"
    assert "active_device_ordinal" not in request_payload["target"]
    assert "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in json.dumps(request_payload)
    assert b"TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE" not in output.read_bytes()


def test_malformed_sibling_in_the_same_family_is_isolated_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

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
    model_source = _snapshot(
        tmp_path / "cache",
        model_id="Example/Requested-Model",
    )
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: root)
    monkeypatch.setattr(
        orchestrator,
        "_probe_current_target_with_device",
        lambda: (_target(), 0),
    )

    selection = try_build_optimized_runtime(
        str(model_source),
        tmp_path / "model.trtfb",
        family_name="example",
        parameters={"precision": "fp16"},
    )

    assert selection is not None
    assert selection.manifest.implementation_id == "requested-runtime"
    assert "Ignoring invalid optimized-runtime manifest" in caplog.text


@pytest.mark.parametrize(
    "failure",
    (
        "probe response returned invalid JSON",
        "probe timed out after 30s",
    ),
)
def test_probe_failure_or_timeout_is_isolated_from_supported_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    broken = load_implementation_manifest(
        _capsule(tmp_path, name="broken", implementation_id="broken-runtime")
    )
    healthy = load_implementation_manifest(
        _capsule(tmp_path, name="healthy", implementation_id="healthy-runtime")
    )
    request = ImplementationRequest(
        model_id="Example/Model",
        model_revision=_REVISION,
        target=_target(),
        parameters={"precision": "fp16"},
    )

    def probe(manifest, _request, **_kwargs):
        if manifest is broken:
            raise BuildAdapterError(failure)
        return ProbeResult(
            supported=True,
            profile_id="a100-fp16-b4",
            profile_sha256="a" * 64,
        )

    monkeypatch.setattr(orchestrator, "run_probe", probe)

    selection = select_delegated_build((broken, healthy), request)

    assert selection is not None
    assert selection.manifest is healthy
    assert failure in caplog.text


def test_selected_adapter_build_failure_remains_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    manifest = load_implementation_manifest(_capsule(tmp_path))
    request = ImplementationRequest(
        model_id="Example/Model",
        model_revision=_REVISION,
        target=_target(),
        parameters={"precision": "fp16"},
    )
    selection = select_delegated_build((manifest,), request)
    assert selection is not None

    def fail_build(*_args, **_kwargs):
        raise BuildAdapterError("selected adapter build failed")

    monkeypatch.setattr(orchestrator, "run_build", fail_build)

    with pytest.raises(BuildAdapterError, match="selected adapter build failed"):
        build_selected_implementation(selection, tmp_path / "model.trtfb")


def test_public_native_build_without_an_owning_family_skips_adapter_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    model_source = _snapshot(
        tmp_path / "cache",
        model_id="Example/Native-Only-Model",
    )
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: None)
    monkeypatch.setattr(
        orchestrator,
        "_probe_current_target_with_device",
        lambda: pytest.fail("native-only request resolved a target"),
    )
    selection = try_build_optimized_runtime(
        str(model_source),
        tmp_path / "native.trtfb",
        family_name="example",
    )

    assert selection is None


def test_malformed_unselected_capsule_preserves_native_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    root = tmp_path / "capsules"
    candidate = _capsule(
        root,
        model_id="Example/Requested-Model",
    )
    _make_invalid_toml(candidate)
    model_source = _snapshot(
        tmp_path / "cache",
        model_id="Example/Requested-Model",
    )
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: root)

    selection = try_build_optimized_runtime(
        str(model_source),
        tmp_path / "model.trtfb",
        family_name="example",
        parameters={"precision": "fp16"},
    )

    assert selection is None
    assert "Ignoring invalid optimized-runtime manifest" in caplog.text


def test_unsupported_probe_returns_native_fallback_without_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "family"
    _capsule(family_root)
    model_source = _snapshot(tmp_path / "cache")
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)
    monkeypatch.setattr(
        orchestrator,
        "_probe_current_target_with_device",
        lambda: (_target(), 0),
    )
    output = tmp_path / "model.trtfb"
    selection = try_build_optimized_runtime(
        str(model_source),
        output,
        family_name="example",
        parameters={"precision": "fp8"},
    )
    assert selection is None
    assert not output.exists()


def test_non_snapshot_model_source_retains_native_path(tmp_path: Path) -> None:
    output = tmp_path / "native.trtfb"

    selection = try_build_optimized_runtime(
        "Example/Model",
        output,
        family_name="example",
        parameters={"precision": "fp16"},
    )

    assert selection is None
    assert not output.exists()


def test_snapshot_selects_only_its_exact_revision_with_multiple_capsules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "family"
    _capsule(
        family_root,
        name="first",
        implementation_id="first-revision-runtime",
        downstream_runtime="first-runtime",
    )
    second_revision = "abcdef0123456789abcdef0123456789abcdef01"
    _capsule(
        family_root,
        name="second",
        implementation_id="second-revision-runtime",
        downstream_runtime="second-runtime",
        revision=second_revision,
    )
    model_source = _snapshot(tmp_path / "cache", revision=second_revision)
    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)
    monkeypatch.setattr(
        orchestrator,
        "_probe_current_target_with_device",
        lambda: (_target(), 0),
    )
    output = tmp_path / "model.trtfb"

    selection = try_build_optimized_runtime(
        str(model_source),
        output,
        family_name="example",
        parameters={"precision": "fp16"},
    )

    assert selection is not None
    assert selection.manifest.implementation_id == "second-revision-runtime"
    assert selection.request.model_revision == second_revision


def test_unavailable_active_target_returns_native_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    family_root = tmp_path / "family"
    _capsule(family_root)
    model_source = _snapshot(tmp_path / "cache")
    output = tmp_path / "native.trtfb"

    def unavailable_target() -> tuple[dict[str, object], int]:
        raise TargetResolutionError("CUDA target probing is unavailable")

    monkeypatch.setattr(orchestrator, "family_implementation_root", lambda _family: family_root)
    monkeypatch.setattr(orchestrator, "_probe_current_target_with_device", unavailable_target)
    selection = try_build_optimized_runtime(
        str(model_source),
        output,
        family_name="example",
        parameters={"precision": "fp16"},
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

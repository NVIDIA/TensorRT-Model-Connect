# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for isolated optimized-runtime implementation capsules."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorrt_model_connect.optimized_runtime.build_adapter import (
    BuildAdapterError,
    ProbeResult,
    run_build,
    run_probe,
)
from tensorrt_model_connect.optimized_runtime.manifest import (
    ImplementationRequest,
    ManifestDiscoveryError,
    ManifestValidationError,
    discover_implementations_for_model,
    load_implementation_manifest,
)


_DEFAULT_ADAPTER = r"""#!/usr/bin/env python3
import json
import pathlib
import sys

operation = sys.argv[1]
request_path = pathlib.Path(sys.argv[sys.argv.index("--request") + 1])
request = json.loads(request_path.read_text(encoding="utf-8"))

if operation == "probe":
    if request["target"]["gpu_architecture"] == "sm80":
        print(json.dumps({
            "schema_version": 1,
            "supported": True,
            "profile_id": "a100-fp16",
        }))
    else:
        print(json.dumps({
            "schema_version": 1,
            "supported": False,
            "reason": "target is not supported",
        }))
elif operation == "build":
    output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
    artifacts = output / "artifacts"
    engine = artifacts / "engine.dir"
    engine.mkdir(parents=True)
    (engine / "engine.plan").write_bytes(b"opaque-engine")
    descriptor = {
        "schema_version": 1,
        "build_binding": request["build_binding"],
        "implementation_id": request["implementation_id"],
        "model": request["model"],
        "target": request["target"],
    }
    (output / "descriptor.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    print(json.dumps({
        "schema_version": 1,
        "descriptor": "descriptor.json",
        "artifacts": "artifacts",
    }))
else:
    raise SystemExit(64)
"""


def _toml_value(value: object) -> str:
    # JSON scalars and arrays are also valid TOML values for this test schema.
    return json.dumps(value)


def _write_capsule(
    root: Path,
    *,
    capsule_name: str = "example-optimized-a100",
    implementation_id: str = "example-model.acme-runtime.a100-fp16",
    model_id: str = "Example/Optimized-Model",
    revisions: tuple[str, ...] = ("revision-a",),
    target: dict[str, object] | None = None,
    adapter: str = _DEFAULT_ADAPTER,
) -> Path:
    capsule = root / capsule_name
    builder_dir = capsule / "builder"
    builder_dir.mkdir(parents=True)
    adapter_path = builder_dir / "adapter.py"
    adapter_path.write_text(adapter, encoding="utf-8")

    target = target or {
        "os": "linux",
        "architecture": "x86_64",
        "gpu_architecture": "sm80",
    }
    target_lines = "\n".join(f"{key} = {_toml_value(value)}" for key, value in target.items())
    manifest = capsule / "IMPLEMENTATION.toml"
    manifest.write_text(
        f"""schema_version = 1
implementation_id = {_toml_value(implementation_id)}
runtime_kind = "optimized"
downstream_runtime = "acme-optimized-runtime"
downstream_version = "1.2.3"
downstream_commit = "0123456789abcdef"

[model]
id = {_toml_value(model_id)}
revisions = {_toml_value(list(revisions))}

[target]
{target_lines}

[build]
entrypoint = "builder/adapter.py"
timeout_seconds = 30

[runtime]
library = "libtrtmc_impl_example_optimized_runtime.so"
abi = 1
artifact_layout = "directory-tree-v1"
""",
        encoding="utf-8",
    )
    return manifest


def _request(
    *,
    model_id: str = "Example/Optimized-Model",
    revision: str = "revision-a",
    target: dict[str, object] | None = None,
) -> ImplementationRequest:
    return ImplementationRequest(
        model_id=model_id,
        model_revision=revision,
        target=target
        or {
            "os": "linux",
            "architecture": "x86_64",
            "gpu_architecture": "sm80",
            "gpu_memory_mib": 81920,
        },
    )


def _supported_probe() -> ProbeResult:
    return ProbeResult(
        supported=True,
        profile_id="a100-fp16",
    )


def test_model_scoped_discovery_is_recursive_stable_and_uncached(tmp_path: Path) -> None:
    first = _write_capsule(
        tmp_path / "root-b",
        capsule_name="z-capsule",
        implementation_id="z-runtime",
    )
    discovered = discover_implementations_for_model(
        [tmp_path / "root-b"], "Example/Optimized-Model"
    )
    assert [manifest.path for manifest in discovered] == [first.resolve()]

    _write_capsule(
        tmp_path / "root-a",
        capsule_name="a-capsule",
        implementation_id="a-runtime",
    )
    discovered = discover_implementations_for_model(
        [tmp_path / "root-b", tmp_path / "root-a"], "Example/Optimized-Model"
    )
    assert [manifest.implementation_id for manifest in discovered] == [
        "a-runtime",
        "z-runtime",
    ]


def test_discovery_rejects_duplicate_implementation_ids(tmp_path: Path) -> None:
    _write_capsule(tmp_path, capsule_name="one", implementation_id="duplicate")
    _write_capsule(tmp_path, capsule_name="two", implementation_id="duplicate")

    with pytest.raises(ManifestDiscoveryError, match="duplicate.*one.*two"):
        discover_implementations_for_model([tmp_path], "Example/Optimized-Model")


def test_model_scoped_discovery_filters_valid_family_owned_adapters(tmp_path: Path) -> None:
    requested = _write_capsule(
        tmp_path,
        capsule_name="requested",
        implementation_id="requested-runtime",
        model_id="Example/Requested-Model",
    )
    unrelated = _write_capsule(
        tmp_path,
        capsule_name="unrelated",
        implementation_id="unrelated-runtime",
        model_id="Example/Unrelated-Model",
    )
    discovered = discover_implementations_for_model([tmp_path], "Example/Requested-Model")

    assert [manifest.path for manifest in discovered] == [requested.resolve()]
    assert unrelated.resolve() not in {manifest.path for manifest in discovered}


def test_model_scoped_discovery_ignores_malformed_unrelated_sibling(
    tmp_path: Path,
) -> None:
    requested = _write_capsule(
        tmp_path,
        capsule_name="requested",
        implementation_id="requested-runtime",
        model_id="Example/Requested-Model",
    )
    unrelated = _write_capsule(
        tmp_path,
        capsule_name="unrelated",
        implementation_id="unrelated-runtime",
        model_id="Example/Unrelated-Model",
    )
    unrelated.write_text(
        unrelated.read_text(encoding="utf-8").replace(
            'revisions = ["revision-a"]',
            'revisions = [',
        ),
        encoding="utf-8",
    )

    discovered = discover_implementations_for_model(
        [tmp_path], "Example/Requested-Model"
    )

    assert [manifest.path for manifest in discovered] == [requested.resolve()]


def test_model_scoped_discovery_fails_closed_for_malformed_candidate(
    tmp_path: Path,
) -> None:
    requested = _write_capsule(
        tmp_path,
        capsule_name="requested",
        implementation_id="requested-runtime",
        model_id="Example/Requested-Model",
    )
    requested.write_text(
        requested.read_text(encoding="utf-8") + "\nbroken = [\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="invalid TOML"):
        discover_implementations_for_model([tmp_path], "Example/Requested-Model")


def test_model_discovery_supports_family_directory_and_direct_manifest(
    tmp_path: Path,
) -> None:
    manifest = _write_capsule(
        tmp_path,
        implementation_id="requested-runtime",
        model_id="Example/Requested-Model",
    )
    discovered = discover_implementations_for_model([tmp_path], "Example/Requested-Model")
    assert [item.path for item in discovered] == [manifest.resolve()]
    direct = discover_implementations_for_model([manifest], "Example/Requested-Model")
    assert [item.path for item in direct] == [manifest.resolve()]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda text: text + "\nunexpected = true\n", "unknown field"),
        (
            lambda text: text.replace(
                'implementation_id = "example-model.acme-runtime.a100-fp16"',
                'implementation_id = "Invalid Runtime"',
            ),
            "implementation_id",
        ),
        (
            lambda text: text.replace("abi = 1", "abi = true"),
            "runtime.abi must be an integer",
        ),
        (
            lambda text: text.replace(
                'artifact_layout = "directory-tree-v1"',
                'artifact_layout = "provider-specific-layout"',
            ),
            "unsupported runtime.artifact_layout",
        ),
    ],
)
def test_manifest_schema_is_strict(tmp_path: Path, mutation, message: str) -> None:
    path = _write_capsule(tmp_path)
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=message):
        load_implementation_manifest(path)


def test_manifest_rejects_entrypoint_path_escape(tmp_path: Path) -> None:
    path = _write_capsule(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'entrypoint = "builder/adapter.py"', 'entrypoint = "../outside.py"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="capsule-relative"):
        load_implementation_manifest(path)


def test_manifest_rejects_entrypoint_symlink_escape(tmp_path: Path) -> None:
    path = _write_capsule(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    adapter = path.parent / "builder" / "adapter.py"
    adapter.unlink()
    adapter.symlink_to(outside)

    with pytest.raises(ManifestValidationError, match="outside the capsule"):
        load_implementation_manifest(path)


def test_exact_matching_has_no_cross_model_or_cross_target_leakage(
    tmp_path: Path,
) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path))

    assert manifest.matches(_request())
    assert not manifest.matches(_request(model_id="Example/Other-Model"))
    assert not manifest.matches(_request(revision="revision-b"))
    assert not manifest.matches(
        _request(
            target={
                "os": "linux",
                "architecture": "x86_64",
                "gpu_architecture": "sm90",
            }
        )
    )


def test_target_matching_is_type_strict(tmp_path: Path) -> None:
    path = _write_capsule(
        tmp_path,
        target={"sm": 80, "integrated": False},
    )
    manifest = load_implementation_manifest(path)

    assert manifest.matches_target({"sm": 80, "integrated": False})
    assert not manifest.matches_target({"sm": 80.0, "integrated": False})
    assert not manifest.matches_target({"sm": 80, "integrated": 0})


def test_probe_and_build_use_versioned_json_protocol(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path))
    request = _request()

    probe = run_probe(manifest, request)
    assert probe.supported
    assert probe.profile_id == "a100-fp16"

    artifact = run_build(manifest, request, tmp_path / "staging", probe=probe)
    assert artifact.descriptor == {
        "schema_version": 1,
        "build_binding": artifact.descriptor["build_binding"],
        "implementation_id": manifest.implementation_id,
        "model": {"id": request.model_id, "revision": request.model_revision},
        "target": dict(request.target),
    }
    assert (artifact.artifacts_path / "engine.dir" / "engine.plan").read_bytes() == (
        b"opaque-engine"
    )


def _binding_adapter(mutation: str, *, include_binding: bool = True) -> str:
    binding_entry = '"build_binding": binding,' if include_binding else ""
    return f"""import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
artifacts = output / "artifacts"
artifacts.mkdir(parents=True)
(artifacts / "runtime.so").write_bytes(b"runtime")
binding = dict(request["build_binding"])
{mutation}
descriptor = {{"schema_version": 1, {binding_entry}"private": {{"opaque": True}}}}
(output / "descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")
print(json.dumps({{
    "schema_version": 1,
    "descriptor": "descriptor.json",
    "artifacts": "artifacts",
}}))
"""


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ('binding["profile_id"] = "different-profile"', "build_binding.profile_id"),
        ('binding["request_sha256"] = "0" * 64', "build_binding.request_sha256"),
        ('binding["manifest_sha256"] = "0" * 64', "build_binding.manifest_sha256"),
        ('binding["implementation_id"] = "different-runtime"', "build_binding.implementation_id"),
    ],
)
def test_build_rejects_binding_that_disagrees_with_selection(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest = load_implementation_manifest(
        _write_capsule(tmp_path, adapter=_binding_adapter(mutation))
    )

    with pytest.raises(BuildAdapterError, match=message):
        run_build(
            manifest,
            _request(),
            tmp_path / "staging",
            probe=_supported_probe(),
        )


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (_binding_adapter('binding.pop("request_sha256")'), "missing field.*request_sha256"),
        (_binding_adapter('binding["unexpected"] = True'), "unknown field.*unexpected"),
        (_binding_adapter("", include_binding=False), "build_binding must be a JSON object"),
    ],
)
def test_build_binding_schema_is_strict(tmp_path: Path, adapter: str, message: str) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match=message):
        run_build(
            manifest,
            _request(),
            tmp_path / "staging",
            probe=_supported_probe(),
        )


def test_build_requires_the_supported_probe_selected_by_the_router(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path))

    with pytest.raises(BuildAdapterError, match="requires a supported probe"):
        run_build(
            manifest,
            _request(),
            tmp_path / "staging",
            probe=ProbeResult(supported=False, reason="profile is not supported"),
        )

    assert not (tmp_path / "staging").exists()


def test_adapter_is_not_invoked_for_a_nonmatching_request(tmp_path: Path) -> None:
    marker = tmp_path / "invoked"
    adapter = f"""import pathlib
pathlib.Path({str(marker)!r}).write_text("invoked", encoding="utf-8")
"""
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match="does not match"):
        run_probe(manifest, _request(revision="other"))
    assert not marker.exists()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not json", "invalid JSON"),
        ('{"schema_version":1,"supported":NaN}', "non-finite JSON"),
        (
            json.dumps({"schema_version": 1, "supported": True}),
            "profile_id",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "supported": False,
                    "reason": "no",
                    "extra": True,
                }
            ),
            "unknown field",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "supported": False,
                    "reason": "profile is not supported",
                    "profile_id": 0,
                }
            ),
            "profile_id",
        ),
    ],
)
def test_probe_rejects_malformed_adapter_responses(
    tmp_path: Path, response: str, message: str
) -> None:
    adapter = f"print({response!r})\n"
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match=message):
        run_probe(manifest, _request())


def test_probe_reports_adapter_exit_failure(tmp_path: Path) -> None:
    adapter = """import sys
print("adapter detail", file=sys.stderr)
raise SystemExit(23)
"""
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match="exit code 23: adapter detail"):
        run_probe(manifest, _request())


def test_explicit_zero_timeout_is_rejected_instead_of_using_manifest_default(
    tmp_path: Path,
) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path))

    with pytest.raises(ValueError, match="positive integer"):
        run_probe(manifest, _request(), timeout_seconds=0)


def test_build_descriptor_rejects_nonfinite_json(tmp_path: Path) -> None:
    adapter = _binding_adapter("").replace(
        '"private": {"opaque": True}',
        '"private": {"score": float("nan")}',
    )
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match="non-finite JSON"):
        run_build(
            manifest,
            _request(),
            tmp_path / "staging",
            probe=_supported_probe(),
        )


@pytest.mark.skipif(os.name != "posix", reason="optimized-runtime capsules target Linux")
def test_probe_timeout_terminates_adapter_process_group(tmp_path: Path) -> None:
    ready = tmp_path / "child-ready"
    terminated = tmp_path / "child-terminated"
    child = f"""import pathlib
import signal
import time

ready = pathlib.Path({str(ready)!r})
terminated = pathlib.Path({str(terminated)!r})

def stop(_signal, _frame):
    terminated.write_text("terminated", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready.write_text("ready", encoding="utf-8")
time.sleep(60)
"""
    adapter = f"""import pathlib
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", {child!r}])
ready = pathlib.Path({str(ready)!r})
while not ready.exists():
    time.sleep(0.01)
time.sleep(60)
"""
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match="timed out after 1s"):
        run_probe(manifest, _request(), timeout_seconds=1)

    assert ready.is_file()
    assert terminated.read_text(encoding="utf-8") == "terminated"


def test_build_rejects_artifacts_outside_staging_directory(tmp_path: Path) -> None:
    adapter = r"""import json
import pathlib
import sys

output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
(output / "descriptor.json").write_text("{}", encoding="utf-8")
(output.parent / "outside").mkdir()
print(json.dumps({
    "schema_version": 1,
    "descriptor": "descriptor.json",
    "artifacts": "../outside",
}))
"""
    manifest = load_implementation_manifest(_write_capsule(tmp_path, adapter=adapter))

    with pytest.raises(BuildAdapterError, match="output-relative"):
        run_build(manifest, _request(), tmp_path / "staging", probe=_supported_probe())


def test_build_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    manifest = load_implementation_manifest(_write_capsule(tmp_path))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "stale").write_text("stale", encoding="utf-8")

    with pytest.raises(BuildAdapterError, match="must be empty"):
        run_build(manifest, _request(), staging, probe=_supported_probe())

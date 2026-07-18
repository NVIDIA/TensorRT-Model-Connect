# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-artifact tests for the SAM3 TensorRT-only runtime contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest


_AUDITOR_PATH = Path(__file__).with_name("audit_runtime_contract.py")
_AUDITOR_SPEC = importlib.util.spec_from_file_location(
    "sam3_runtime_contract_auditor",
    _AUDITOR_PATH,
)
assert _AUDITOR_SPEC is not None
assert _AUDITOR_SPEC.loader is not None
audit = importlib.util.module_from_spec(_AUDITOR_SPEC)
sys.modules[_AUDITOR_SPEC.name] = audit
_AUDITOR_SPEC.loader.exec_module(audit)


def test_sam3_runtime_auditor_runs_directly_from_clean_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(_AUDITOR_PATH), "--help"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Audit a production SAM3 bundle" in completed.stdout
    assert "--cmake-cache" in completed.stdout


def _valid_section_payloads() -> dict[str, bytes]:
    payloads = {
        name: (
            b"ftrt native TensorRT plan " + name.encode("ascii")
            if name in audit.SAM3_PLAN_SECTIONS
            else b"{}"
        )
        for name in sorted(audit.SAM3_BUNDLE_SECTIONS)
    }
    payloads["config.json"] = json.dumps(
        {"detector_config": {"text_config": {"vocab_size": 3}}}
    ).encode()
    payloads["tokenizer.json"] = json.dumps(
        {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"],
            }
        }
    ).encode()
    return payloads


def _write_bundle(
    path: Path,
    *,
    section_payloads: dict[str, bytes] | None = None,
    runtime_strategy: str = "sam3_prompted_segmentation",
    model_type: str = "sam3_video",
    gap_before_section: str | None = None,
    gap_payload: bytes = b"",
    trailing_payload: bytes = b"",
) -> Path:
    payloads = _valid_section_payloads()
    if section_payloads is not None:
        payloads = dict(section_payloads)

    offset = 0
    section_table: dict[str, dict[str, int]] = {}
    payload_chunks: list[bytes] = []
    for name, payload in payloads.items():
        if name == gap_before_section:
            payload_chunks.append(gap_payload)
            offset += len(gap_payload)
        section_table[name] = {"offset": offset, "size": len(payload)}
        payload_chunks.append(payload)
        offset += len(payload)
    header = {
        "model_type": model_type,
        "runtime_strategy": runtime_strategy,
        "sections": section_table,
    }
    encoded_header = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(b"TRTFB\x00\x01\x00")
        handle.write(struct.pack("<Q", len(encoded_header)))
        handle.write(encoded_header)
        for payload in payload_chunks:
            handle.write(payload)
        handle.write(trailing_payload)
    return path


def _write_runtime_dsos(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (
        tmp_path / "libtrtmc_core.so",
        tmp_path / "libtrtmc_backend_trt.so",
        tmp_path / "libtrtmc_model_sam3.so",
    )
    for path in paths:
        path.write_bytes(b"\x7fELF" + path.name.encode("ascii"))
    return paths


class _FakeElfTools:
    def __init__(
        self,
        *,
        injected_output: dict[str | tuple[str, str], str] | None = None,
        backend_has_tensorrt: bool = True,
    ) -> None:
        self.injected_output = injected_output or {}
        self.backend_has_tensorrt = backend_has_tensorrt

    def __call__(self, command: Sequence[str], stdin: str | None) -> str:
        tool = command[0]
        path_key = (tool, Path(command[-1]).name) if len(command) > 1 else None
        injected = self.injected_output.get(path_key) if path_key is not None else None
        if injected is None:
            injected = self.injected_output.get(tool)
        if injected is not None:
            return injected
        if tool == "readelf":
            path = Path(command[-1])
            needed = ["libcudart.so.13", "libstdc++.so.6"]
            if path.name.startswith("libtrtmc_backend_trt") and self.backend_has_tensorrt:
                needed.append("libnvinfer.so.10")
            return "\n".join(
                f"0x0000000000000001 (NEEDED) Shared library: [{name}]" for name in needed
            )
        if tool == "ldd":
            return "libcudart.so.13 => /usr/local/cuda/lib64/libcudart.so.13"
        if tool == "nm":
            return "0000000000000000 T trtmc_sam3_create"
        if tool == "c++filt":
            return stdin or ""
        if tool == "strings":
            return "TRTMC SAM3 native TensorRT runtime"
        raise AssertionError(f"Unexpected audit tool: {command!r}")


def test_sam3_runtime_artifact_audit_accepts_native_tensorrt_release(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "sam3.trtfb")
    dsos = _write_runtime_dsos(tmp_path)
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:BOOL=OFF\nTRTMC_ENABLE_TVM_FFI:BOOL=OFF\n",
        encoding="utf-8",
    )

    report = audit.audit_sam3_runtime_artifacts(
        bundle_path=bundle,
        dso_paths=dsos,
        cmake_cache=cache,
        tool_runner=_FakeElfTools(),
    )

    assert report.plan_count == 12
    assert set(report.bundle_sections) == audit.SAM3_BUNDLE_SECTIONS
    assert len(report.bundle_sha256) == 64
    assert set(report.dso_sha256) == {str(path) for path in dsos}
    assert all(len(digest) == 64 for digest in report.dso_sha256.values())


def test_sam3_bundle_audit_requires_exact_native_plan_and_asset_sections(tmp_path: Path) -> None:
    payloads = _valid_section_payloads()
    payloads["tracker_aoti.pt2"] = b"PK\x03\x04"
    bundle = _write_bundle(tmp_path / "sam3-extra.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match="unexpected=.*tracker_aoti.pt2"):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize("payload_location", ["gap", "trailing"])
def test_sam3_bundle_audit_rejects_unaccounted_payload_bytes(
    tmp_path: Path,
    payload_location: str,
) -> None:
    kwargs = (
        {
            "gap_before_section": "sam3_tracker_step_engine_plan",
            "gap_payload": b"hidden-aoti-package",
        }
        if payload_location == "gap"
        else {"trailing_payload": b"hidden-aoti-package"}
    )
    bundle = _write_bundle(tmp_path / f"sam3-{payload_location}.trtfb", **kwargs)

    with pytest.raises(audit.Sam3RuntimeContractError, match="unaccounted payload bytes"):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize("payload", [b"PK\x03\x04aoti", b"not-an-engine", b"onnx-protobuf"])
def test_sam3_bundle_audit_rejects_non_tensorrt_plan_payload(
    tmp_path: Path,
    payload: bytes,
) -> None:
    payloads = _valid_section_payloads()
    payloads["sam3_tracker_step_engine_plan"] = payload
    bundle = _write_bundle(tmp_path / "sam3-bad-plan.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match="not a serialized TensorRT plan"):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize(
    "marker",
    [
        b"AOTInductorModelContainer",
        b"TvmFfiKernel",
        b"libtorch.so",
        b"libpython3.12.so",
        b"onnxruntime",
        b"tracker.pt2",
    ],
)
def test_sam3_bundle_audit_rejects_bridge_marker_inside_plan(
    tmp_path: Path,
    marker: bytes,
) -> None:
    payloads = _valid_section_payloads()
    payloads["sam3_tracker_step_engine_plan"] = b"ftrt" + marker
    bundle = _write_bundle(tmp_path / "sam3-bridge-plan.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match="forbidden marker"):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize(
    "marker",
    [
        b"AOTInductorModelContainer",
        b"TvmFfiKernel",
        b"libtorch.so",
        b"libpython3.12.so",
        b"onnxruntime",
    ],
)
def test_sam3_bundle_audit_rejects_bridge_marker_inside_asset(
    tmp_path: Path,
    marker: bytes,
) -> None:
    payloads = _valid_section_payloads()
    payloads["config.json"] = b'{"hidden_runtime": "' + marker + b'"}'
    bundle = _write_bundle(tmp_path / "sam3-bridge-asset.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match="forbidden marker"):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize(
    ("asset", "payload", "message"),
    [
        ("config.json", b"not-json", "not valid JSON"),
        ("tokenizer.json", b"[]", "must contain an object"),
        ("merges.txt", b"\xff", "not valid UTF-8"),
        ("merges.txt", b"version\x00payload", "binary NUL"),
    ],
)
def test_sam3_bundle_audit_rejects_malformed_asset_payload(
    tmp_path: Path,
    asset: str,
    payload: bytes,
    message: str,
) -> None:
    payloads = _valid_section_payloads()
    payloads[asset] = payload
    bundle = _write_bundle(tmp_path / "sam3-malformed-asset.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match=message):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize(
    ("vocab", "merges", "message"),
    [
        ({"a": 0, "b": 2}, ["a b"], "unique and dense"),
        ({"a": 0, "b": 1, "ab": 2}, ["not-a-pair"], "invalid BPE merge"),
    ],
)
def test_sam3_bundle_audit_rejects_unusable_native_bpe_tokenizer(
    tmp_path: Path,
    vocab: dict[str, int],
    merges: list[str],
    message: str,
) -> None:
    payloads = _valid_section_payloads()
    payloads["tokenizer.json"] = json.dumps(
        {"model": {"type": "BPE", "vocab": vocab, "merges": merges}}
    ).encode()
    bundle = _write_bundle(tmp_path / "sam3-invalid-bpe.trtfb", section_payloads=payloads)

    with pytest.raises(audit.Sam3RuntimeContractError, match=message):
        audit.audit_sam3_bundle(bundle)


@pytest.mark.parametrize(
    "cache_text",
    [
        "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:BOOL=ON\nTRTMC_ENABLE_TVM_FFI:BOOL=OFF\n",
        "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:BOOL=OFF\nTRTMC_ENABLE_TVM_FFI:BOOL=ON\n",
        "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:BOOL=OFF\n",
    ],
)
def test_sam3_build_cache_audit_rejects_enabled_or_missing_bridges(
    tmp_path: Path,
    cache_text: str,
) -> None:
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(cache_text, encoding="utf-8")

    with pytest.raises(audit.Sam3RuntimeContractError, match="must be OFF"):
        audit.audit_sam3_build_cache(cache)


@pytest.mark.parametrize(
    ("tool", "marker"),
    [
        ("readelf", "Shared library: [libtorch.so]"),
        ("readelf", "Shared library: [libc10.so]"),
        ("readelf", "Shared library: [libpython3.12.so]"),
        ("ldd", "libonnxruntime.so => /runtime/libonnxruntime.so"),
        ("c++filt", "torch::Tensor tracker_step"),
        ("c++filt", "c10::Tensor tracker_step"),
        ("c++filt", "at::Tensor tracker_step"),
        ("c++filt", "PyObject_Call tracker_step"),
        ("c++filt", "_Py_Dealloc tracker_step"),
        ("strings", "AOTI runtime package"),
        ("strings", "TvmFfiKernel"),
        ("c++filt", "tvm::ffi::Function tracker_step"),
    ],
)
def test_sam3_runtime_dependency_audit_rejects_external_frameworks(
    tmp_path: Path,
    tool: str,
    marker: str,
) -> None:
    dsos = _write_runtime_dsos(tmp_path)
    runner = _FakeElfTools(injected_output={tool: marker})

    with pytest.raises(audit.Sam3RuntimeContractError, match="Forbidden runtime dependency"):
        audit.audit_sam3_runtime_dependencies(dsos, tool_runner=runner)


def test_sam3_runtime_dependency_audit_rejects_unresolved_library(tmp_path: Path) -> None:
    dsos = _write_runtime_dsos(tmp_path)
    runner = _FakeElfTools(injected_output={"ldd": "libcuda.so.1 => not found"})

    with pytest.raises(audit.Sam3RuntimeContractError, match="unresolved library"):
        audit.audit_sam3_runtime_dependencies(dsos, tool_runner=runner)


def test_sam3_runtime_dependency_audit_scans_transitive_model_connect_dso(
    tmp_path: Path,
) -> None:
    dsos = _write_runtime_dsos(tmp_path)
    transitive = tmp_path / "libtrtmc_tracker_bridge.so"
    transitive.write_bytes(b"\x7fELFtransitive")
    runner = _FakeElfTools(
        injected_output={
            (
                "ldd",
                "libtrtmc_model_sam3.so",
            ): f"libtrtmc_tracker_bridge.so => {transitive} (0x0000000000010000)",
            ("strings", transitive.name): "AOTI runtime package",
        }
    )

    with pytest.raises(audit.Sam3RuntimeContractError, match="Forbidden runtime dependency"):
        audit.audit_sam3_runtime_dependencies(dsos, tool_runner=runner)


def test_sam3_runtime_dependency_audit_rejects_non_tensorrt_framework(
    tmp_path: Path,
) -> None:
    dsos = _write_runtime_dsos(tmp_path)
    runner = _FakeElfTools(
        injected_output={
            "ldd": "libtensorflow.so.2 => /runtime/libtensorflow.so.2",
        }
    )

    with pytest.raises(audit.Sam3RuntimeContractError, match="Non-TensorRT runtime libraries"):
        audit.audit_sam3_runtime_dependencies(dsos, tool_runner=runner)


def test_sam3_runtime_dependency_audit_requires_backend_link_to_tensorrt(
    tmp_path: Path,
) -> None:
    dsos = _write_runtime_dsos(tmp_path)

    with pytest.raises(audit.Sam3RuntimeContractError, match="does not depend on libnvinfer"):
        audit.audit_sam3_runtime_dependencies(
            dsos,
            tool_runner=_FakeElfTools(backend_has_tensorrt=False),
        )


def test_sam3_runtime_audit_cli_requires_cmake_cache() -> None:
    parser = audit._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--bundle",
                "sam3.trtfb",
                "--dso",
                "libtrtmc_core.so",
            ]
        )

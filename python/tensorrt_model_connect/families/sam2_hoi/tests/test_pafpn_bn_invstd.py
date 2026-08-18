# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi import pafpn_bn_invstd


def _identity(nvcc: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "helper_version": pafpn_bn_invstd.HELPER_VERSION,
        "source": {
            "path": str(Path(pafpn_bn_invstd.__file__).with_name("pafpn_bn_invstd_helper.cu")),
            "size": 2276,
            "sha256": pafpn_bn_invstd.HELPER_SOURCE_SHA256,
        },
        "compiler": {
            "path": str(nvcc),
            "arguments": ["--version"],
            "returncode": 0,
            "output": "reviewed nvcc",
        },
        "cuda_architectures": ["89", "100"],
        "compile_flags": list(pafpn_bn_invstd._compile_flags()),
        "kernel_expression": "rsqrtf(__fadd_rn(variance[index], epsilon))",
        "execution_scope": "builder_time_only",
        "inference_runtime_launch_added": False,
    }


def test_helper_source_is_exact_builder_only_cuda_contract() -> None:
    source = Path(pafpn_bn_invstd.__file__).with_name("pafpn_bn_invstd_helper.cu")
    payload = source.read_bytes()
    assert len(payload) == 2276
    assert hashlib.sha256(payload).hexdigest() == pafpn_bn_invstd.HELPER_SOURCE_SHA256
    text = payload.decode("utf-8")
    assert "rsqrtf(summed)" in text
    assert "__fadd_rn(variance[index], epsilon)" in text
    assert pafpn_bn_invstd.HELPER_VERSION in text
    assert "cudaMemcpyHostToDevice" in text and "cudaMemcpyDeviceToHost" in text

    family_root = source.parent
    cmake = (family_root / "native_plugins/CMakeLists.txt").read_text(encoding="utf-8")
    assert source.name not in cmake
    assert "pafpn_bn_invstd" not in cmake


def test_helper_build_is_atomic_cached_and_receipted(monkeypatch, tmp_path: Path) -> None:
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("reviewed compiler", encoding="utf-8")
    identity = _identity(nvcc)
    monkeypatch.setattr(pafpn_bn_invstd, "_configured_build_base", lambda: tmp_path / "cache")
    monkeypatch.setattr(pafpn_bn_invstd, "_build_identity", lambda: identity)
    commands = []

    def fake_run(command, *, check, **_kwargs):
        assert check is True
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"reviewed-helper-dso")

    monkeypatch.setattr(pafpn_bn_invstd.subprocess, "run", fake_run)

    output, receipt, observed = pafpn_bn_invstd._build_helper(verbose=False)
    assert observed == identity
    assert output.read_bytes() == b"reviewed-helper-dso"
    assert output.stat().st_mode & 0o777 == 0o700
    assert commands[0][0] == str(nvcc)
    assert commands[0][1:-3] == list(pafpn_bn_invstd._compile_flags())
    assert commands[0][-3] == str(
        Path(pafpn_bn_invstd.__file__).with_name("pafpn_bn_invstd_helper.cu")
    )
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "identity": identity,
        "output_sha256": hashlib.sha256(b"reviewed-helper-dso").hexdigest(),
    }

    assert pafpn_bn_invstd._build_helper(verbose=False) == (output, receipt, identity)
    assert len(commands) == 1


def test_helper_compile_failure_removes_partial_output(monkeypatch, tmp_path: Path) -> None:
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("reviewed compiler", encoding="utf-8")
    monkeypatch.setattr(pafpn_bn_invstd, "_configured_build_base", lambda: tmp_path / "cache")
    monkeypatch.setattr(pafpn_bn_invstd, "_build_identity", lambda: _identity(nvcc))
    monkeypatch.setattr(
        pafpn_bn_invstd.subprocess,
        "run",
        lambda command, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, command, output="compile failed")
        ),
    )

    with pytest.raises(RuntimeError, match="helper build failed"):
        pafpn_bn_invstd._build_helper(verbose=False)
    cache_files = [path for path in (tmp_path / "cache").rglob("*") if path.is_file()]
    assert all(path.name.startswith(".") and path.name.endswith(".lock") for path in cache_files)


def test_helper_receipt_failure_removes_promoted_dso(monkeypatch, tmp_path: Path) -> None:
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("reviewed compiler", encoding="utf-8")
    monkeypatch.setattr(pafpn_bn_invstd, "_configured_build_base", lambda: tmp_path / "cache")
    monkeypatch.setattr(pafpn_bn_invstd, "_build_identity", lambda: _identity(nvcc))

    def fake_run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"reviewed-helper-dso")

    monkeypatch.setattr(pafpn_bn_invstd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pafpn_bn_invstd.native_plugin_builder,
        "_write_build_receipt",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("receipt failure")),
    )
    with pytest.raises(RuntimeError, match="receipt failure"):
        pafpn_bn_invstd._build_helper(verbose=False)
    cache_files = [path for path in (tmp_path / "cache").rglob("*") if path.is_file()]
    assert all(path.name.startswith(".") and path.name.endswith(".lock") for path in cache_files)


def test_compute_invstd_validates_and_propagates_helper_status(monkeypatch, tmp_path: Path) -> None:
    helper = tmp_path / "helper.so"
    helper.write_bytes(b"helper")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        pafpn_bn_invstd,
        "_build_helper",
        lambda *, verbose: (helper, receipt, {}),
    )
    statuses = iter((0, 17))

    def fake_load(_path, _digest):
        status = next(statuses)

        def compute(variance_pointer, count, epsilon, output_pointer):
            assert count.value == 2 and epsilon.value == pytest.approx(1.0e-5)
            variance = np.ctypeslib.as_array(variance_pointer, shape=(count.value,))
            output = np.ctypeslib.as_array(output_pointer, shape=(count.value,))
            output[:] = variance + np.float32(1.0) if status == 0 else np.nan
            return status

        return object(), compute

    monkeypatch.setattr(pafpn_bn_invstd, "_load_functions", fake_load)
    variance = np.asarray([3.0, 8.0], dtype=np.float32)
    np.testing.assert_array_equal(
        pafpn_bn_invstd.compute_invstd(variance, epsilon=1.0e-5),
        np.asarray([4.0, 9.0], dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="CUDA status 17"):
        pafpn_bn_invstd.compute_invstd(variance, epsilon=1.0e-5)

    for invalid in (np.ones((1, 2), dtype=np.float32), np.asarray([-1.0], dtype=np.float32)):
        with pytest.raises(ValueError):
            pafpn_bn_invstd.compute_invstd(invalid, epsilon=1.0e-5)
    with pytest.raises(ValueError, match="epsilon"):
        pafpn_bn_invstd.compute_invstd(np.ones(2, dtype=np.float32), epsilon=0.0)


def test_helper_receipt_preflights_version_and_invalid_argument_contract(
    monkeypatch, tmp_path: Path
) -> None:
    helper = tmp_path / "helper.so"
    helper.write_bytes(b"helper")
    identity = {"execution_scope": "builder_time_only"}
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "identity": identity,
                "output_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pafpn_bn_invstd,
        "_build_helper",
        lambda *, verbose: (helper, receipt, identity),
    )
    calls = []

    def compute(input_pointer, count, epsilon, output_pointer):
        calls.append((input_pointer, count.value, epsilon.value, output_pointer))
        return -1

    monkeypatch.setattr(
        pafpn_bn_invstd,
        "_load_functions",
        lambda _path, _digest: (object(), compute),
    )
    observed = pafpn_bn_invstd.helper_build_receipt()
    assert observed["payload"]["identity"] == identity
    assert observed["invalid_argument_contract_status"] == -1
    assert calls == [(None, 0, pytest.approx(1.0e-5), None)]


def test_helper_ctypes_contract_is_float32_only() -> None:
    assert ctypes.sizeof(ctypes.c_float) == 4
    assert pafpn_bn_invstd._CUDA_ARCHITECTURES == ("89", "100")
    assert pafpn_bn_invstd._compile_flags().count("-gencode") == 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from typing import Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.dynamic_memory_contract import (  # noqa: E402
    module_residency_plan_set_sha256,
    qualified_runtime_stack_sha256,
)


MODULE_PATH = REPO_ROOT / "tools" / "seal_native_dynamic_memory_bundle.py"
SPEC = importlib.util.spec_from_file_location(
    "seal_native_dynamic_memory_bundle",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
seal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seal
SPEC.loader.exec_module(seal)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]


def _v1_contract() -> dict:
    return {
        "contract_version": 1,
        "qualified_model_id": "Qwen/Qwen3-0.6B",
        "qualified_model_revision": (
            "c1899de289a04d12100db370d81485cdf75e47ca"
        ),
        "qualified_config_sha256": (
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
        ),
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": (
                "7b9b711c22b6823e87150213ecd8449260db8610"
            ),
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": 40960,
        "prefill_chunk_limit": 2048,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114688,
        "active_kv_profile_limits": [128, 512, 2048, 8192, 32768, 40960],
        "runtime_owned": True,
    }


def _section_payloads() -> dict[str, bytes]:
    return {
        "engine_plan": b"serialized decode engine plan",
        "prefill_engine_plan": b"serialized prefill engine plan",
        "config.json": (
            b'{\n  "model_type": "qwen3",\n'
            b'  "runtime_strategy": "native"\n}\n'
        ),
        "weights": (b"unchanged-model-weights-" * 4096) + b"tail",
    }


def _calibration(
    contract: dict,
    section_payloads: dict[str, bytes],
) -> dict:
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(
                section_payloads["engine_plan"]
            ).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(
                contract["active_kv_profile_limits"]
            ),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(
                section_payloads["prefill_engine_plan"]
            ).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    return {
        "schema_version": 1,
        "measurement_kind": "nvml_process_cumulative_first_use",
        "cuda_module_loading_mode": "lazy",
        "evidence_provenance": "external_manifest_v1",
        "qualified_runtime_stack_sha256": qualified_runtime_stack_sha256(
            contract["qualified_runtime_stack"]
        ),
        "plan_set_sha256": module_residency_plan_set_sha256(plans),
        "plans": plans,
        "profile_reserves": [
            {
                "covering_profile_limit": limit,
                "cumulative_reserve_bytes": (index + 1) * 16 * 1024 * 1024,
            }
            for index, limit in enumerate(
                contract["active_kv_profile_limits"]
            )
        ],
        "evidence_sha256": hashlib.sha256(
            b"qualification evidence"
        ).hexdigest(),
    }


def _header_and_payload(
    *,
    contract: dict | None = None,
    section_payloads: dict[str, bytes] | None = None,
) -> tuple[dict, bytes]:
    runtime_memory = _v1_contract() if contract is None else contract
    payloads = _section_payloads() if section_payloads is None else section_payloads
    sections: dict[str, dict[str, int]] = {}
    body = bytearray()
    for name, value in payloads.items():
        sections[name] = {"offset": len(body), "size": len(value)}
        body.extend(value)
    header = {
        "format_version": 1,
        "model_id": runtime_memory["qualified_model_id"],
        "family": "qwen",
        "precision": "bf16",
        "max_cache_length": runtime_memory["model_context_limit"],
        "runtime_memory": runtime_memory,
        "sections": sections,
        "metadata": {
            "must_survive": True,
            "nested": ["all", "non-runtime", "header", "fields"],
        },
    }
    return header, bytes(body)


def _write_raw_bundle(path: Path, header: dict, payload: bytes) -> None:
    raw_header = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    path.write_bytes(
        seal.BUNDLE_MAGIC
        + struct.pack("<Q", len(raw_header))
        + raw_header
        + payload
    )


def _write_bundle(
    path: Path,
    *,
    contract: dict | None = None,
    section_payloads: dict[str, bytes] | None = None,
) -> tuple[dict, bytes]:
    header, payload = _header_and_payload(
        contract=contract,
        section_payloads=section_payloads,
    )
    _write_raw_bundle(path, header, payload)
    return header, payload


def _read_bundle(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as stream:
        assert stream.read(8) == seal.BUNDLE_MAGIC
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        payload = stream.read()
    return header, payload


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "records": records}),
        encoding="utf-8",
    )


def _qualified_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict, bytes, dict]:
    input_path = tmp_path / "input.trtfb"
    output_path = tmp_path / "output.trtfb"
    manifest_path = tmp_path / "MODULE_RESIDENCY_CALIBRATIONS.json"
    header, payload = _write_bundle(input_path)
    calibration = _calibration(header["runtime_memory"], _section_payloads())
    _write_manifest(manifest_path, [calibration])
    return (
        input_path,
        output_path,
        manifest_path,
        header,
        payload,
        calibration,
    )


def test_reseal_streams_payload_and_preserves_every_non_runtime_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        input_header,
        input_payload,
        calibration,
    ) = _qualified_fixture(tmp_path)
    input_bytes = input_path.read_bytes()
    output_path.write_bytes(b"replace me atomically")
    output_path.chmod(0o640)

    original_read = seal.os.read
    requested_read_sizes: list[int] = []

    def recording_read(descriptor: int, size: int) -> bytes:
        requested_read_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(seal, "_IO_CHUNK_BYTES", 31)
    monkeypatch.setattr(seal.os, "read", recording_read)

    receipt = seal.reseal_bundle(
        input_path,
        output_path,
        family="qwen",
        manifest_path=manifest_path,
    )

    output_header, output_payload = _read_bundle(output_path)
    output_bytes = output_path.read_bytes()
    input_non_runtime = dict(input_header)
    output_non_runtime = dict(output_header)
    input_non_runtime.pop("runtime_memory")
    output_non_runtime.pop("runtime_memory")

    assert input_path.read_bytes() == input_bytes
    assert output_payload == input_payload
    assert output_non_runtime == input_non_runtime
    assert output_header["sections"] == input_header["sections"]
    assert output_header["runtime_memory"]["contract_version"] == 2
    assert output_header["runtime_memory"]["runtime_config_sha256"] == (
        hashlib.sha256(_section_payloads()["config.json"]).hexdigest()
    )
    assert (
        output_header["runtime_memory"]["module_residency_calibration"]
        == calibration
    )
    assert (
        output_header["runtime_memory"]["module_residency_calibration"][
            "evidence_provenance"
        ]
        == "external_manifest_v1"
    )
    assert receipt["input_bundle_size_bytes"] == len(input_bytes)
    assert receipt["input_bundle_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert receipt["output_bundle_size_bytes"] == len(output_bytes)
    assert receipt["output_bundle_sha256"] == hashlib.sha256(
        output_bytes
    ).hexdigest()
    assert receipt["payload_size_bytes"] == len(input_payload)
    assert receipt["payload_sha256"] == hashlib.sha256(input_payload).hexdigest()
    assert receipt["plan_section_sha256"] == {
        name: hashlib.sha256(_section_payloads()[name]).hexdigest()
        for name in ("engine_plan", "prefill_engine_plan")
    }
    assert receipt["plan_set_sha256"] == calibration["plan_set_sha256"]
    assert receipt["evidence_sha256"] == calibration["evidence_sha256"]
    assert receipt["cuda_module_loading_mode"] == "lazy"
    assert max(requested_read_sizes) <= 31
    assert output_path.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(".output.trtfb.tmp.*"))


def test_reseal_binds_exact_serialized_runtime_config_bytes(
    tmp_path: Path,
) -> None:
    runtime_configs = (
        b'{"model_type":"qwen3","runtime_strategy":"native"}',
        b'{\n  "model_type": "qwen3",\n'
        b'  "runtime_strategy": "native"\n}\n',
    )
    observed: list[str] = []
    for index, runtime_config in enumerate(runtime_configs):
        section_payloads = {
            **_section_payloads(),
            "config.json": runtime_config,
        }
        input_path = tmp_path / f"input-{index}.trtfb"
        output_path = tmp_path / f"output-{index}.trtfb"
        manifest_path = tmp_path / f"manifest-{index}.json"
        header, _payload = _write_bundle(
            input_path,
            section_payloads=section_payloads,
        )
        _write_manifest(
            manifest_path,
            [_calibration(header["runtime_memory"], section_payloads)],
        )

        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )
        output_header, _output_payload = _read_bundle(output_path)
        observed.append(
            output_header["runtime_memory"]["runtime_config_sha256"]
        )

    assert observed == [
        hashlib.sha256(runtime_config).hexdigest()
        for runtime_config in runtime_configs
    ]
    assert observed[0] != observed[1]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda header, _payload: header["sections"].pop("config.json"),
            "config.json runtime configuration",
        ),
        (
            lambda header, _payload: header["sections"].pop(
                "prefill_engine_plan"
            ),
            "exactly engine_plan and prefill_engine_plan",
        ),
        (
            lambda header, _payload: header["sections"].update(
                vision_plan={"offset": 0, "size": 1}
            ),
            "extra=",
        ),
        (
            lambda header, _payload: header["sections"][
                "prefill_engine_plan"
            ].update(offset=1),
            "overlaps",
        ),
        (
            lambda header, payload: header["sections"]["engine_plan"].update(
                size=len(payload) + 1
            ),
            "beyond the payload",
        ),
    ),
)
def test_reseal_rejects_invalid_plan_section_layouts(
    tmp_path: Path,
    mutate: Callable[[dict, bytes], object],
    message: str,
) -> None:
    input_path = tmp_path / "input.trtfb"
    header, payload = _header_and_payload()
    mutate(header, payload)
    _write_raw_bundle(input_path, header, payload)

    with pytest.raises(seal.BundleResealError, match=message):
        seal.reseal_bundle(
            input_path,
            tmp_path / "output.trtfb",
            family="qwen",
            manifest_path=tmp_path / "unused.json",
        )


def test_reseal_rejects_already_sealed_v2_input(tmp_path: Path) -> None:
    section_payloads = _section_payloads()
    contract = _v1_contract()
    contract = {
        **contract,
        "contract_version": 2,
        "runtime_config_sha256": hashlib.sha256(
            section_payloads["config.json"]
        ).hexdigest(),
        "module_residency_calibration": _calibration(
            contract,
            section_payloads,
        ),
    }
    input_path = tmp_path / "input.trtfb"
    _write_bundle(
        input_path,
        contract=contract,
        section_payloads=section_payloads,
    )

    with pytest.raises(seal.BundleResealError, match="already"):
        seal.reseal_bundle(
            input_path,
            tmp_path / "output.trtfb",
            family="qwen",
            manifest_path=tmp_path / "unused.json",
        )


def test_reseal_rejects_plan_hash_without_exact_calibration(
    tmp_path: Path,
) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        header,
        _payload,
        _calibration_record,
    ) = _qualified_fixture(tmp_path)
    different_payloads = {
        **_section_payloads(),
        "engine_plan": b"a different serialized plan",
    }
    _write_manifest(
        manifest_path,
        [_calibration(header["runtime_memory"], different_payloads)],
    )

    with pytest.raises(seal.BundleResealError, match="exact stack\\+plan.*absent"):
        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )
    assert not output_path.exists()


def test_reseal_rejects_ambiguous_exact_calibration(tmp_path: Path) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        _header,
        _payload,
        calibration,
    ) = _qualified_fixture(tmp_path)
    _write_manifest(manifest_path, [calibration, calibration])

    with pytest.raises(seal.BundleResealError, match="ambiguous"):
        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )


def test_reseal_rejects_invalid_selected_calibration(tmp_path: Path) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        _header,
        _payload,
        calibration,
    ) = _qualified_fixture(tmp_path)
    calibration["plan_set_sha256"] = "0" * 64
    _write_manifest(manifest_path, [calibration])

    with pytest.raises(
        seal.BundleResealError,
        match="selected module-residency calibration is invalid",
    ):
        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )


def test_reseal_refuses_in_place_and_hard_link_outputs(tmp_path: Path) -> None:
    (
        input_path,
        _output_path,
        manifest_path,
        _header,
        _payload,
        _calibration_record,
    ) = _qualified_fixture(tmp_path)

    with pytest.raises(seal.BundleResealError, match="paths must differ"):
        seal.reseal_bundle(
            input_path,
            input_path,
            family="qwen",
            manifest_path=manifest_path,
        )

    hard_link = tmp_path / "hard-link.trtfb"
    os.link(input_path, hard_link)
    with pytest.raises(seal.BundleResealError, match="same file"):
        seal.reseal_bundle(
            input_path,
            hard_link,
            family="qwen",
            manifest_path=manifest_path,
        )


def test_reseal_refuses_symlink_inputs_and_outputs(tmp_path: Path) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        _header,
        _payload,
        _calibration_record,
    ) = _qualified_fixture(tmp_path)
    input_link = tmp_path / "input-link.trtfb"
    input_link.symlink_to(input_path)

    with pytest.raises(seal.BundleResealError, match="non-symlink regular"):
        seal.reseal_bundle(
            input_link,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )

    output_target = tmp_path / "output-target.trtfb"
    output_target.write_bytes(b"existing output")
    output_path.symlink_to(output_target)
    with pytest.raises(seal.BundleResealError, match="non-symlink regular"):
        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )


def test_reseal_refuses_nonregular_inputs_and_outputs(tmp_path: Path) -> None:
    directory_input = tmp_path / "input-dir"
    directory_input.mkdir()
    with pytest.raises(seal.BundleResealError, match="regular file"):
        seal.reseal_bundle(
            directory_input,
            tmp_path / "unused-output.trtfb",
            family="qwen",
        )

    (
        input_path,
        _output_path,
        manifest_path,
        _header,
        _payload,
        _calibration_record,
    ) = _qualified_fixture(tmp_path)
    directory_output = tmp_path / "output-dir"
    directory_output.mkdir()
    with pytest.raises(seal.BundleResealError, match="regular file"):
        seal.reseal_bundle(
            input_path,
            directory_output,
            family="qwen",
            manifest_path=manifest_path,
        )


def test_atomic_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        input_path,
        output_path,
        manifest_path,
        _header,
        _payload,
        _calibration_record,
    ) = _qualified_fixture(tmp_path)
    old_output = b"existing output must survive"
    output_path.write_bytes(old_output)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise seal.BundleResealError("injected copy failure")

    monkeypatch.setattr(seal, "_copy_payload", fail_copy)
    with pytest.raises(seal.BundleResealError, match="injected copy failure"):
        seal.reseal_bundle(
            input_path,
            output_path,
            family="qwen",
            manifest_path=manifest_path,
        )

    assert output_path.read_bytes() == old_output
    assert not list(tmp_path.glob(".output.trtfb.tmp.*"))


def test_cli_accepts_only_input_output_and_family_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "receipt_schema_version": 1,
        "output_bundle_sha256": "a" * 64,
    }
    observed: dict[str, object] = {}

    def fake_reseal(
        input_bundle: Path,
        output_bundle: Path,
        *,
        family: str,
    ) -> dict:
        observed.update(
            input_bundle=input_bundle,
            output_bundle=output_bundle,
            family=family,
        )
        return expected

    monkeypatch.setattr(seal, "reseal_bundle", fake_reseal)
    input_path = tmp_path / "input.trtfb"
    output_path = tmp_path / "output.trtfb"

    assert (
        seal.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--family",
                "qwen",
            ]
        )
        == 0
    )
    assert observed == {
        "input_bundle": input_path,
        "output_bundle": output_path,
        "family": "qwen",
    }
    assert json.loads(capsys.readouterr().out) == expected

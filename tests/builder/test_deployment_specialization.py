"""Unit tests for deployment specialization bundle metadata."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.cli import _cmd_inspect
from tensorrt_model_connect.deployment import (
    DEPLOYMENT_MANIFEST_SECTION,
    directory_sections,
    edge_llm_manifest,
    ffi_attention_manifest,
    kernel_sections,
    manifest_section,
    parse_kernel_artifacts,
)
from tensorrt_model_connect.edge_llm_provider import build_edge_llm_bundle
from tensorrt_model_connect.runtime_config import resolve_cli_config
from tensorrt_model_connect.runtime_config.schemas import load_all


def _read_header(path: Path) -> dict:
    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[8:16])[0]
    return json.loads(data[16:16 + header_len])


def _read_section(path: Path, name: str) -> bytes:
    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[8:16])[0]
    header = json.loads(data[16:16 + header_len])
    section = header["sections"][name]
    body = 16 + header_len
    start = body + section["offset"]
    return data[start:start + section["size"]]


class _DeploymentConfig:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get(self, namespace: str, field: str) -> object:
        assert namespace == "deployment"
        return self._values[field]


def test_deployment_schema_accepts_exit_criteria_flags() -> None:
    load_all()
    cfg = resolve_cli_config(
        config_path=None,
        set_tokens=[
            "deployment.provider=tensorrt-edge-llm",
            "deployment.target=jetson-thor",
            "deployment.enable_ffi_attention=true",
            "deployment.ffi_kernel_artifacts=flashinfer.decode_f16_d64=/tmp/k.so",
        ],
    )

    assert cfg.get("deployment", "provider") == "tensorrt-edge-llm"
    assert cfg.get("deployment", "target") == "jetson-thor"
    assert cfg.get("deployment", "enable_ffi_attention") is True
    assert (
        cfg.get("deployment", "ffi_kernel_artifacts")
        == "flashinfer.decode_f16_d64=/tmp/k.so"
    )


def test_kernel_artifacts_and_ffi_manifest_round_trip(tmp_path: Path) -> None:
    so_path = tmp_path / "kernel.so"
    so_path.write_bytes(b"\x7fELFfake")
    artifacts = parse_kernel_artifacts(f"flashinfer.decode_f16_d64={so_path}")
    sections, manifest_json = kernel_sections(
        artifacts,
        section_prefix="deployment/variants/ffi_attention/",
    )
    manifest = ffi_attention_manifest(
        target="gb300",
        runtime_strategy="decoder_kv_cache",
        performance={
            "throughput_tokens_per_s": 123.0,
            "peak_memory_mb": 456.0,
        },
    )
    manifest_dict = manifest.to_dict()
    ffi_variant = manifest_dict["variants"][1]
    assert ffi_variant["compatibility"] == {"platform": ["gb300"]}
    assert ffi_variant["performance"]["target_id"] == "gb300"
    assert ffi_variant["performance"]["variant_id"] == "ffi_attention"
    assert ffi_variant["performance"]["provider"] == "tvm_ffi"
    assert ffi_variant["performance"]["scope"] == "kernel"
    bundle_path = tmp_path / "ffi.trtfb"
    write_bundle(
        bundle_path,
        BundleInfo(model_id="ffi", runtime_strategy="decoder_kv_cache"),
        [
            BundleSection("engine_plan", b"native"),
            BundleSection("deployment/variants/ffi_attention/engine_plan", b"ffi"),
            *sections,
            BundleSection(
                "deployment/variants/ffi_attention/kernel_manifest.json",
                manifest_json,
            ),
            manifest_section(manifest),
        ],
    )

    header = _read_header(bundle_path)
    assert DEPLOYMENT_MANIFEST_SECTION in header["sections"]
    assert "deployment/variants/ffi_attention/engine_plan" in header["sections"]
    assert "deployment/variants/ffi_attention/kernel_flashinfer_decode_f16_d64.so" in (
        header["sections"]
    )


def test_edge_llm_directory_packaging_and_inspect(tmp_path: Path, capsys) -> None:
    engine_dir = tmp_path / "engine"
    (engine_dir / "nested").mkdir(parents=True)
    (engine_dir / "llm.engine").write_bytes(b"plan")
    (engine_dir / "config.json").write_text("{}", encoding="utf-8")
    (engine_dir / "nested" / "tokenizer.json").write_text("{}", encoding="utf-8")

    prefix = "providers/edgellm/engine_dir/"
    bundle_path = tmp_path / "edge.trtfb"
    write_bundle(
        bundle_path,
        BundleInfo(model_id="edge", family="tensorrt-edge-llm"),
        [
            *directory_sections(engine_dir, section_prefix=prefix),
            manifest_section(edge_llm_manifest(
                target="jetson-thor",
                engine_section_prefix=prefix,
                performance={
                    "throughput_tokens_per_s": 321.0,
                    "peak_memory_mb": 654.0,
                },
            )),
            BundleSection("config.json", b'{"runtime_strategy":"text_generation"}'),
        ],
    )

    rc = _cmd_inspect(argparse.Namespace(
        bundle_path=str(bundle_path),
        list_engines=False,
        deployment=True,
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert "provider: tensorrt-edge-llm" in out
    assert 'compatibility: {"platform": ["jetson-thor"]}' in out
    assert '"target_id": "jetson-thor"' in out
    assert '"variant_id": "edge_llm"' in out
    assert "section_prefix=providers/edgellm/engine_dir/" in out


def test_edge_llm_bundle_sanitizes_absolute_chat_template_model_path(tmp_path: Path) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "llm.engine").write_bytes(b"plan")
    (engine_dir / "config.json").write_text("{}", encoding="utf-8")
    (engine_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (engine_dir / "processed_chat_template.json").write_text(
        json.dumps({
            "model_path": str(tmp_path / "source-model"),
            "roles": {},
        }),
        encoding="utf-8",
    )

    bundle_path = tmp_path / "edge.trtfb"
    build_edge_llm_bundle(
        model_dir="Qwen/Qwen3-0.6B",
        output_path=str(bundle_path),
        max_cache_length=128,
        precision="fp16",
        deployment_config=_DeploymentConfig({
            "target": "gb300",
            "edge_llm_workspace": "",
            "edge_llm_engine_dir": str(engine_dir),
        }),
    )

    section = _read_section(
        bundle_path,
        "providers/edgellm/engine_dir/processed_chat_template.json",
    )
    template = json.loads(section.decode("utf-8"))
    assert template["model_path"] == "bundle://providers/edgellm/engine_dir"
    assert str(tmp_path) not in section.decode("utf-8")

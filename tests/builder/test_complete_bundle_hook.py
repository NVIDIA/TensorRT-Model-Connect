# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for family-owned complete bundle builders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct

import pytest

try:
    import tensorrt_model_connect.engine_builder as engine_builder
    from tensorrt_model_connect.families.base import CompleteBundleBuildRequest
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency-only
    pytest.skip(
        "tensorrt_model_connect build dependencies are unavailable",
        allow_module_level=True,
    )


_FAMILY = "complete_bundle_test"
_RUNTIME_STRATEGY = "complete_bundle_test_runtime"


def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": _FAMILY,
                "architectures": ["CompleteBundleTestModel"],
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def _write_test_bundle(
    path: Path,
    *,
    family: str = _FAMILY,
    runtime_strategy: str = _RUNTIME_STRATEGY,
    magic: bytes = b"BUNDLE\x01\x00",
    header_bytes: bytes | None = None,
) -> None:
    if header_bytes is None:
        header_bytes = _test_header_bytes(
            family=family,
            runtime_strategy=runtime_strategy,
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(magic)
        output.write(struct.pack("<Q", len(header_bytes)))
        output.write(header_bytes)
        output.write(b"plan")


def _test_header_bytes(
    *,
    family: str = _FAMILY,
    runtime_strategy: str = _RUNTIME_STRATEGY,
    sections: object = None,
) -> bytes:
    if sections is None:
        sections = {"engine_plan": {"offset": 0, "size": 4}}
    return json.dumps(
        {
            "model_id": "complete-bundle-test",
            "model_type": _FAMILY,
            "family": family,
            "runtime_strategy": runtime_strategy,
            "sections": sections,
        }
    ).encode("utf-8")


class _CompleteBundlePlugin:
    name = _FAMILY
    runtime_strategy = _RUNTIME_STRATEGY

    def __init__(self, action=None) -> None:
        self.action = action or (lambda request: _write_test_bundle(request.output_path))
        self.requests: list[CompleteBundleBuildRequest] = []

    def matches(self, model_type: str) -> bool:
        return model_type == self.name

    def build_complete_bundle(self, request: CompleteBundleBuildRequest) -> None:
        self.requests.append(request)
        return self.action(request)

    def load_weights(self, *_args, **_kwargs):
        raise AssertionError("complete bundle path must not load weights")

    def build_engine(self, *_args, **_kwargs):
        raise AssertionError("complete bundle path must not build an engine")


def _install_complete_plugin(monkeypatch, plugin: object) -> None:
    monkeypatch.setattr(
        engine_builder,
        "family_has_capability",
        lambda _config, capability: capability == "complete_bundle_builder",
    )
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)


def _forbid_legacy_path(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy builder path must not run")

    monkeypatch.setattr(engine_builder, "_setup_trt_import", forbidden)
    monkeypatch.setattr(engine_builder, "_load_plugin_weights", forbidden)
    monkeypatch.setattr(engine_builder, "write_bundle", forbidden)
    monkeypatch.setattr(engine_builder, "_prepare_tokenizer_special_frame", forbidden)


def test_complete_bundle_hook_receives_full_request_before_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    output = tmp_path / "result.bundle"
    plugin = _CompleteBundlePlugin()
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)
    monkeypatch.setattr(
        engine_builder.build_bundle,
        "_fp8_scales",
        {"layer": {"input_scale": 1.25}},
        raising=False,
    )
    monkeypatch.setattr(
        engine_builder.build_bundle,
        "_save_fp8_scales",
        "/tmp/scales.json",
        raising=False,
    )

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2)
    engine_builder.build_bundle(
        str(model_dir),
        str(output),
        384,
        decoder_engine_layout="dual_profile",
        dynamic_kv_cache=True,
        dynamic_kv_profile_rows_override=[32, 128],
        precision="bf16",
        fp32_layers=[1, 3],
        quantize="int8",
        quant_scales="quant.json",
        quant_calibration_samples=17,
        verbose=True,
        kernel_artifacts=[("owned.kernel", "kernel.so")],
        rtx=True,
        triattention_stats_path="stats.json",
        triattention_kv_budget=128,
        triattention_divide_length=64,
        triattention_recent_window=16,
        triattention_score_aggregation="max",
        triattention_count_prompt_tokens=False,
        triattention_protect_prefill=False,
        triattention_disable_mlr=True,
        triattention_disable_trig=True,
        family_build_options={"sampler": {"mode": "fixed"}},
        parallel_config=parallel,
        diffusion_overrides={"image_height": 1024},
        build_timing_path="timing.json",
        max_batch_size=3,
        tokenizer_source_model_id_or_path="org/source-model",
        tokenizer_source_revision="source-revision",
    )

    assert output.is_file()
    assert len(plugin.requests) == 1
    request = plugin.requests[0]
    assert request.model_dir == model_dir
    assert request.output_path == output
    assert request.config.model_type == _FAMILY
    assert request.max_cache_length == 384
    assert request.decoder_engine_layout == "dual_profile"
    assert request.dynamic_kv_cache is True
    assert request.dynamic_kv_profile_rows_override == (32, 128)
    assert request.precision == "bf16"
    assert request.fp32_layers == (1, 3)
    assert request.quantize == "int8"
    assert request.quant_scales == "quant.json"
    assert request.quant_calibration_samples == 17
    assert request.verbose is True
    assert request.kernel_artifacts == (("owned.kernel", "kernel.so"),)
    assert request.rtx is True
    assert request.fp8_scales == {"layer": {"input_scale": 1.25}}
    assert request.save_fp8_scales == "/tmp/scales.json"
    assert request.triattention_stats_path == "stats.json"
    assert request.triattention_kv_budget == 128
    assert request.triattention_divide_length == 64
    assert request.triattention_recent_window == 16
    assert request.triattention_score_aggregation == "max"
    assert request.triattention_count_prompt_tokens is False
    assert request.triattention_protect_prefill is False
    assert request.triattention_disable_mlr is True
    assert request.triattention_disable_trig is True
    assert request.family_build_options == {"sampler": {"mode": "fixed"}}
    assert request.parallel_config == parallel
    assert request.diffusion_overrides == {"image_height": 1024}
    assert request.build_timing_path == "timing.json"
    assert request.max_batch_size == 3
    assert request.source_model_id_or_path == "org/source-model"
    assert request.source_revision == "source-revision"


def test_complete_bundle_hook_must_create_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    plugin = _CompleteBundlePlugin(action=lambda _request: None)
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(FileNotFoundError, match="did not create"):
        engine_builder.build_bundle(str(model_dir), str(tmp_path / "missing.bundle"))


@pytest.mark.parametrize(
    ("action", "error"),
    [
        (
            lambda request: _write_test_bundle(request.output_path, magic=b"NOTABNDL"),
            "invalid bundle magic/header",
        ),
        (
            lambda request: _write_test_bundle(request.output_path, header_bytes=b"not-json"),
            "invalid JSON header",
        ),
        (
            lambda request: _write_test_bundle(request.output_path, family="other"),
            "family mismatch",
        ),
        (
            lambda request: _write_test_bundle(request.output_path, runtime_strategy="other"),
            "runtime_strategy mismatch",
        ),
    ],
)
def test_complete_bundle_hook_rejects_invalid_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action,
    error: str,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    plugin = _CompleteBundlePlugin(action=action)
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)
    output = tmp_path / "bad.bundle"

    with pytest.raises(ValueError, match=error):
        engine_builder.build_bundle(str(model_dir), str(output))

    assert output.is_file(), "invalid hook artifacts are preserved for diagnosis"


def test_complete_bundle_hook_rejects_duplicate_header_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    duplicate_header = (
        b'{"family":"complete_bundle_test","family":"complete_bundle_test",'
        b'"runtime_strategy":"complete_bundle_test_runtime",'
        b'"sections":{"engine_plan":{"offset":0,"size":4}}}'
    )
    model_dir = _make_model_dir(tmp_path)
    plugin = _CompleteBundlePlugin(
        action=lambda request: _write_test_bundle(
            request.output_path,
            header_bytes=duplicate_header,
        )
    )
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(ValueError, match="duplicate JSON object key 'family'"):
        engine_builder.build_bundle(str(model_dir), str(tmp_path / "duplicate.bundle"))


@pytest.mark.parametrize(
    "sections",
    [
        {},
        [{"name": "engine_plan", "offset": 0, "size": 4}],
        {"engine_plan": {"offset": 1, "size": 4}},
        {
            "engine_plan": {"offset": 0, "size": 3},
            "config.json": {"offset": 2, "size": 2},
        },
    ],
)
def test_complete_bundle_hook_rejects_invalid_section_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sections: object,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    plugin = _CompleteBundlePlugin(
        action=lambda request: _write_test_bundle(
            request.output_path,
            header_bytes=_test_header_bytes(sections=sections),
        )
    )
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(ValueError, match="sections|section"):
        engine_builder.build_bundle(str(model_dir), str(tmp_path / "bad-sections.bundle"))


def test_complete_bundle_hook_allows_format_compatible_empty_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sections = {
        "empty_metadata": {"offset": 0, "size": 0},
        "engine_plan": {"offset": 0, "size": 4},
    }
    model_dir = _make_model_dir(tmp_path)
    plugin = _CompleteBundlePlugin(
        action=lambda request: _write_test_bundle(
            request.output_path,
            header_bytes=_test_header_bytes(sections=sections),
        )
    )
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    engine_builder.build_bundle(str(model_dir), str(tmp_path / "empty-section.bundle"))


def test_complete_bundle_hook_refuses_existing_output_without_calling_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    output = tmp_path / "existing.bundle"
    output.write_bytes(b"do-not-replace")
    plugin = _CompleteBundlePlugin()
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(FileExistsError, match="already exists"):
        engine_builder.build_bundle(str(model_dir), str(output))

    assert output.read_bytes() == b"do-not-replace"
    assert plugin.requests == []


def test_complete_bundle_hook_refuses_existing_symlink_without_calling_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    target = tmp_path / "target.bundle"
    target.write_bytes(b"do-not-replace")
    output = tmp_path / "linked.bundle"
    output.symlink_to(target)
    plugin = _CompleteBundlePlugin()
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(FileExistsError, match="already exists"):
        engine_builder.build_bundle(str(model_dir), str(output))

    assert output.is_symlink()
    assert target.read_bytes() == b"do-not-replace"
    assert plugin.requests == []


def test_complete_bundle_hook_rejects_symlink_created_by_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    target = tmp_path / "target.bundle"
    _write_test_bundle(target)

    def create_symlink(request: CompleteBundleBuildRequest) -> None:
        request.output_path.symlink_to(target)

    plugin = _CompleteBundlePlugin(action=create_symlink)
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(ValueError, match="not a regular file"):
        output = tmp_path / "linked.bundle"
        engine_builder.build_bundle(str(model_dir), str(output))

    assert output.is_symlink(), "invalid hook artifacts are preserved for diagnosis"


def test_complete_bundle_hook_rejects_hard_link_created_by_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    existing = tmp_path / "existing.bundle"
    _write_test_bundle(existing)

    def create_hard_link(request: CompleteBundleBuildRequest) -> None:
        os.link(existing, request.output_path)

    plugin = _CompleteBundlePlugin(action=create_hard_link)
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(ValueError, match="not published exclusively"):
        engine_builder.build_bundle(str(model_dir), str(tmp_path / "hard-linked.bundle"))


def test_complete_bundle_hook_rejects_named_path_replacement_after_final_fstat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = _make_model_dir(tmp_path)
    output = tmp_path / "replace-race.bundle"
    replacement = tmp_path / "replacement.bundle"
    plugin = _CompleteBundlePlugin()
    _install_complete_plugin(monkeypatch, plugin)
    _forbid_legacy_path(monkeypatch)

    real_fstat = os.fstat
    fstat_calls = 0

    def replace_name_after_second_fstat(descriptor: int):
        nonlocal fstat_calls
        result = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            _write_test_bundle(replacement)
            os.replace(replacement, output)
        return result

    monkeypatch.setattr(engine_builder.os, "fstat", replace_name_after_second_fstat)

    with pytest.raises(ValueError, match="changed during validation"):
        engine_builder.build_bundle(str(model_dir), str(output))

    assert fstat_calls == 2
    assert output.is_file()


def test_complete_bundle_capability_requires_a_real_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class MissingHookPlugin:
        name = _FAMILY
        runtime_strategy = _RUNTIME_STRATEGY

    model_dir = _make_model_dir(tmp_path)
    _install_complete_plugin(monkeypatch, MissingHookPlugin())
    _forbid_legacy_path(monkeypatch)

    with pytest.raises(TypeError, match="does not implement"):
        engine_builder.build_bundle(str(model_dir), str(tmp_path / "out.bundle"))


def test_legacy_plugin_still_uses_load_build_and_shared_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class LegacyPlugin:
        name = _FAMILY
        runtime_strategy = ""
        requires_tokenizer = False

        def matches(self, model_type: str) -> bool:
            return model_type == self.name

        def load_weights(self, _model_dir, _config, *, precision="fp32"):
            order.append("load_weights")
            return {}

        def build_engine(
            self,
            _config,
            _weights,
            _max_cache_length,
            *,
            precision="fp32",
            verbose=False,
        ):
            order.append("build_engine")
            return b"legacy-plan"

    plugin = LegacyPlugin()
    model_dir = _make_model_dir(tmp_path)
    output = tmp_path / "legacy.bundle"

    monkeypatch.setattr(engine_builder, "family_has_capability", lambda *_args: False)

    def setup_trt(_rtx: bool) -> None:
        order.append("setup_trt")

    def find_plugin(_config):
        order.append("find_plugin")
        return plugin

    def shared_writer(path, _info, _sections) -> None:
        order.append("write_bundle")
        assert path == str(output)

    monkeypatch.setattr(engine_builder, "_setup_trt_import", setup_trt)
    monkeypatch.setattr(engine_builder, "find_plugin", find_plugin)
    monkeypatch.setattr(engine_builder.trt_compat, "resolved_summary", lambda: "stub TensorRT")
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "11.1.0")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "stub GPU")
    monkeypatch.setattr(
        engine_builder, "_detect_tokenizer_special_frame", lambda *_args, **_kwargs: ([], [])
    )
    monkeypatch.setattr(engine_builder, "write_bundle", shared_writer)

    engine_builder.build_bundle(str(model_dir), str(output), 32)

    assert order == [
        "setup_trt",
        "find_plugin",
        "load_weights",
        "build_engine",
        "write_bundle",
    ]

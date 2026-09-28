# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Turbo base precision is carried through offline build routing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from families.minimax_h3 import delivery, model, staged_build
from families.minimax_h3.runtime_config_schema import normalize_build_options


@pytest.mark.parametrize("precision", ("bf16", "int8"))
def test_base_precision_requires_turbo(precision):
    options = {"turbo_base_precision": precision}
    with pytest.raises(ValueError, match="require turbo=true"):
        normalize_build_options(options)
    assert normalize_build_options({"turbo": True, **options}) == {"turbo": True, **options}


@pytest.mark.parametrize("precision", ("fp8", "nvfp4", "pruned_int8", True, 8, None))
def test_unknown_turbo_base_precision_fails(precision):
    with pytest.raises(ValueError, match="invalid MiniMax-H3 option"):
        normalize_build_options({"turbo": True, "turbo_base_precision": precision})


def test_int8_source_resolution_keeps_text_and_lora_unquantized(tmp_path, monkeypatch):
    import huggingface_hub

    text, lora, base = (tmp_path / name for name in ("text", "lora", "base"))
    for path in (text, lora, base):
        path.write_bytes(b"fixture")
    observed = []

    def download(repository, filename, *, revision):
        observed.append((repository, filename, revision))
        return str(base)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = delivery.resolve_turbo_sources(
        tmp_path,
        {"turbo_base_precision": "int8", "turbo_text_encoder": str(text), "turbo_lora": str(lora)},
    )
    assert observed == [
        (
            delivery.COMFY_REPOSITORY,
            delivery.QUANTIZED_SOURCES["quantized_transformer"],
            delivery.COMFY_REVISION,
        )
    ]
    assert result == {"turbo_transformer": base, "turbo_text_encoder": text, "turbo_lora": lora}


def test_plugin_passes_base_precision_without_changing_runtime_defaults(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in delivery.TURBO_SOURCES}
    observed = {}
    monkeypatch.setattr(delivery, "resolve_turbo_sources", lambda *_args: paths)
    monkeypatch.setattr(delivery, "resolve_super_resolution_sources", lambda *_args: (None, None))
    monkeypatch.setattr(
        staged_build, "build_staged_bundle", lambda *_args, **kwargs: observed.update(kwargs)
    )
    options = {"turbo": True, "turbo_base_precision": "int8"}
    model.plugin.build_staged_bundle(
        str(tmp_path),
        object(),
        SimpleNamespace(raw={"_family_build_options": {"minimax_h3": options}}),
        {"_model_dir": str(tmp_path)},
        plans_dir=tmp_path / "plans",
        precision="bf16",
    )
    assert observed["turbo_base_precision"] == "int8"
    assert {key: observed[key] for key in paths} == paths
    assert observed["runtime_defaults"]["num_frames"] == 124


def test_staged_child_carries_explicit_precision(tmp_path, monkeypatch):
    output = tmp_path / "head.plan"
    observed = []

    def run(command, *, check):
        assert check
        observed.extend(command)
        output.write_bytes(b"plan fixture")

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    staged_build._run_component(
        "denoiser_head",
        tmp_path,
        output,
        verbose=False,
        turbo_transformer_path=tmp_path / "base",
        turbo_lora_path=tmp_path / "lora",
        turbo_base_precision="int8",
    )
    index = observed.index("--turbo-base-precision")
    assert observed[index + 1] == "int8"


def test_child_cli_passes_precision_to_builder(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setattr(
        staged_build, "_build_component", lambda *_args, **kwargs: observed.update(kwargs)
    )
    assert (
        staged_build._main(
            [
                "--child",
                "--component",
                "denoiser_head",
                "--model-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "head.plan"),
                "--turbo-transformer",
                str(tmp_path / "base"),
                "--turbo-lora",
                str(tmp_path / "lora"),
                "--turbo-base-precision",
                "int8",
            ]
        )
        == 0
    )
    assert observed["turbo_base_precision"] == "int8"
    assert observed["turbo_transformer_path"] == Path(tmp_path / "base")

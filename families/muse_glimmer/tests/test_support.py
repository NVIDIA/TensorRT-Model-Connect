# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer identity and family-owned Edge route tests."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.muse_glimmer.edge_llm import builder, dispatch


def raw_config():
    return {
        "model_type": "muse_glimmer",
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "quantization_config": {"quant_algo": "MIXED_PRECISION"},
        "text_config": {
            "model_type": "muse_glimmer_text",
            "num_hidden_layers": 52,
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 202048,
            "max_position_embeddings": 131072,
        },
    }


def quant_config():
    return {
        "quantization": {
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {"quant_algo": "NVFP4", "group_size": 16},
                "model.layers.0.mlp.down_proj": {"quant_algo": "MXFP8"},
            }
        }
    }


def test_support_identity():
    family, support = resolve_family(ModelMetadata(raw_config(), {}))
    assert family == "muse_glimmer"
    assert support.default_task == "text_generation"
    assert support.default_precision == "fp16"


@pytest.fixture
def request_and_route(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(raw_config()))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config()))
    request = BuildRequest(
        tmp_path,
        tmp_path / "model.bundle",
        "muse_glimmer",
        "text_generation",
        "fp16",
        max_sequence_length=1024,
        quantization="nvfp4",
    )
    target = {
        "os": "linux",
        "arch": "x86_64",
        "sm": 120,
        "cuda_version": "13.3",
        "tensorrt_version": "11.1.0.106",
    }
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "local_target", lambda: target)
    return request, target


def test_documented_mixed_checkpoint_is_candidate(request_and_route):
    request, _ = request_and_route
    assert dispatch.candidate(request, raw_config())
    assert builder.request_weight_format(request, raw_config()) == "nvfp4"


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", "fp32"),
        ("tensor_parallel_size", 2),
        ("dynamic_kv_cache", True),
        ("quantization", "fp8"),
    ],
)
def test_non_candidate_uses_native_without_probe(request_and_route, monkeypatch, field, value):
    request, _ = request_and_route
    request = replace(request, **{field: value})
    monkeypatch.setattr(builder, "local_target", lambda: pytest.fail("unexpected GPU probe"))
    calls = []
    dispatch.build(request, object(), lambda *args: calls.append(args[0]))
    assert calls == [request]


def test_edge_failure_retries_unchanged_native_once(request_and_route, monkeypatch, caplog):
    request, _ = request_and_route
    writer = object()
    calls = []

    def failed(*args):
        raise RuntimeError("upstream builder error")

    monkeypatch.setitem(dispatch.EDGE_DISPATCH, ("linux", "x86_64", 120, "nvfp4"), failed)
    dispatch.build(request, writer, lambda r, w: calls.append((r, w)))
    assert calls == [(request, writer)]
    assert "Retrying native once" in caplog.text


def test_publication_failure_does_not_retry_native(request_and_route, monkeypatch):
    request, _ = request_and_route
    monkeypatch.setitem(
        dispatch.EDGE_DISPATCH, ("linux", "x86_64", 120, "nvfp4"), lambda *args: ({}, {})
    )
    monkeypatch.setattr(builder, "publish", lambda *args: (_ for _ in ()).throw(OSError("publish")))
    with pytest.raises(OSError, match="publish"):
        dispatch.build(request, object(), lambda *args: pytest.fail("must not retry"))


def test_compatible_wheel_selected_and_sdk_python_is_fallback(monkeypatch):
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps(["0.11.0", "11.1.0.106"]))

    monkeypatch.setattr(builder.subprocess, "run", probe)
    package = {"python": "/sdk/python"}
    assert (
        builder.exporter_python(package, {"tensorrt_version": "11.1.0.106"})
        == builder.sys.executable
    )
    assert "import tensorrt_edgellm.scripts.export" in commands[0][-1]
    assert builder.exporter_python(package, {"tensorrt_version": "11.0.0.0"}) == package["python"]


def test_prepare_maps_to_wheel_exporter_and_native_onnx_builder(
    request_and_route, tmp_path, monkeypatch
):
    request, target = request_and_route
    prefix = tmp_path / "prefix"
    paths = {
        "python": prefix / "bin/python",
        "plugin": prefix / "lib/plugin.so",
        "onnx_builder": prefix / "bin/onnx-build",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    package = {name: str(path) for name, path in paths.items()}
    monkeypatch.setattr(builder, "installed_package", lambda target: package)
    monkeypatch.setattr(builder, "exporter_python", lambda package, target: package["python"])
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs.get("env")))
        if "tensorrt_edgellm.scripts.export" in command:
            out = Path(command[command.index("tensorrt_edgellm.scripts.export") + 2]) / "llm"
            out.mkdir(parents=True)
        else:
            out = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--engineDir=")))
            out.mkdir(parents=True)
            for name in (
                "llm.engine",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "embedding.safetensors",
            ):
                (out / name).write_text("x")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", run)
    staging = tmp_path / "staging"
    staging.mkdir()
    log = tmp_path / "build.log"
    files, marker = builder.prepare(request, raw_config(), target, staging, log)
    assert seen[0][0][2:4] == ["-m", "tensorrt_edgellm.scripts.export"]
    assert seen[1][0][0] == package["onnx_builder"]
    assert seen[1][1]["EDGELLM_PLUGIN_PATH"] == package["plugin"]
    assert marker["weight_format"] == "nvfp4"
    assert "edge_llm/engine/llm.engine" in files

    # Exercise both documented companions through the same owning build mapper.
    from families.muse_glimmer.build_request import coerce_request
    from families.muse_glimmer.edge_llm.paired import companion_version

    legacy = coerce_request(request)
    assert legacy.execution_variant == "autoregressive" and legacy.companion is None
    with pytest.raises(ValueError, match="requires exactly one companion"):
        replace(legacy, execution_variant="dflash")

    for version, architecture in ((1, "MuseGlimmerAssistantModel"), (2, "DFlash2DraftModel")):
        draft = tmp_path / f"draft-{version}"
        draft.mkdir()
        policy = {"block_size": 16, "mask_token_id": 201818,
                  "target_layer_ids": [1, 13, 25, 37, 49]}
        draft_config = {"architectures": [architecture], "num_hidden_layers": 5,
                        "hidden_size": 6656, "intermediate_size": 19968,
                        "num_attention_heads": 32, "num_key_value_heads": 8,
                        "head_dim": 128}
        draft_config.update(policy if version == 1 else {"dflash_config": policy})
        (draft / "config.json").write_text(json.dumps(draft_config))
        (draft / "model.safetensors").write_bytes(b"fixture")
        assert companion_version(draft) == version
        paired_request = replace(legacy, execution_variant="dflash", companion=draft)
        paired_commands = []
        include_selector = True

        def run_pair(command, **kwargs):
            paired_commands.append(command)
            if "tensorrt_edgellm.scripts.export" in command:
                output = Path(command[command.index("tensorrt_edgellm.scripts.export") + 2])
                output.mkdir()
                assert command[-2:] == ["--dflash-draft-dir", str(draft)]
            else:
                output = Path(command[command.index("--engineDir") + 1])
                output.mkdir(exist_ok=True, parents=True)
                role = "base" if "--specBase" in command else "draft"
                (output / f"spec_{role}.engine").write_bytes(b"engine")
                config = {"spec_decode_type": "dflash",
                          "dflash_config": {**policy, "version": version}}
                if version == 2 and role == "draft":
                    config["dflash_config"]["selector_file"] = "dflash2_selector.safetensors"
                    if include_selector:
                        (output / "dflash2_selector.safetensors").write_bytes(b"fixture")
                (output / f"{role}_config.json").write_text(json.dumps(config))
                for name in ("embedding.safetensors", "tokenizer.json",
                             "tokenizer_config.json", "chat_template.jinja"):
                    (output / name).write_bytes(b"fixture")
                assert kwargs["env"]["EDGELLM_PLUGIN_PATH"] == package["plugin"]
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(builder.subprocess, "run", run_pair)
        destination = tmp_path / f"paired-{version}"
        destination.mkdir()
        pair_files, pair_marker = builder.prepare(
            paired_request, raw_config(), target, destination, log
        )
        assert len(paired_commands) == 4
        assert pair_marker["execution_variant"] == "dflash"
        assert pair_marker["dflash_version"] == version
        assert pair_marker["dflash_block_size"] == 16
        assert "edge_llm/engine/spec_base.engine" in pair_files
        assert "edge_llm/engine/spec_draft.engine" in pair_files
        assert "edge_llm/engine/llm.engine" not in pair_files
        if version == 2:
            assert "edge_llm/engine/dflash2_selector.safetensors" in pair_files
            include_selector = False
            missing_selector = tmp_path / "missing-selector"
            missing_selector.mkdir()
            with pytest.raises(ValueError, match="paired artifact missing: dflash2_selector"):
                builder.prepare(paired_request, raw_config(), target, missing_selector, log)
        policy["target_layer_ids"] = [52]
        draft_config.update(policy if version == 1 else {"dflash_config": policy})
        (draft / "config.json").write_text(json.dumps(draft_config))
        with pytest.raises(ValueError, match="matched block-16"):
            companion_version(draft)

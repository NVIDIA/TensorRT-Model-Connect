# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Gemma generations this family will build."""

from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError
import re
from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from families.gemma.model import _SUPPORTED_MODEL_TYPES, build as build_family
    from families.gemma.support import describe
    from tensorrt_model_connect import BuildRequest
    from tensorrt_model_connect.model_support import ModelMetadata
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


from families.gemma.edge_llm.config import (
    BuildExecutionInputs, GemmaBuildRequest, NamedCheckpoint, with_execution,
)

build_core = importlib.import_module("tensorrt_model_connect.build")


def _model_dir(tmp_path: Path, model_type: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "vocab_size": 32,
                "max_position_embeddings": 256,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _build(model_dir: Path) -> None:
    build_family(
        BuildRequest(
            model_dir=model_dir,
            output_path=model_dir / "out.bundle",
            family="gemma",
            task="text_generation",
            precision="fp16",
        ),
        writer=None,
    )


def test_the_gate_matches_what_support_claims() -> None:
    """The builder and the identity check must name the same generations."""
    for model_type in _SUPPORTED_MODEL_TYPES:
        assert (
            describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is not None
        )


def test_a_later_gemma_generation_is_refused(tmp_path: Path) -> None:
    """Gemma 3n and 4 need machinery this family still does not have.

    Gemma 4 adds vision and audio towers, per-layer input embeddings and
    KV-shared layers; Gemma 3n is its own architecture again. Neither is built through the ordinary native entrypoint;
    explicit paired Edge offload is separate, so a prefix check would let them build a full-attention graph and
    generate quietly wrong text. The refusal names the type so the message is
    actionable. Gemma 3 text is supported and is covered below.
    """
    for model_type in ("gemma3n", "gemma4", "gemma4_text", "gemma4_unified"):
        directory = _model_dir(tmp_path / model_type.replace("_", ""), model_type)
        with pytest.raises(ValueError, match=re.escape(f"model_type={model_type!r}")):
            _build(directory)


def test_an_unrelated_model_type_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path / "llama", "llama")
    with pytest.raises(ValueError, match="does not support model_type"):
        _build(directory)


def test_the_supported_generations_pass_the_gate(tmp_path: Path) -> None:
    """gemma and gemma2 must get past the type check and fail later instead.

    The directory holds no weights, so the build cannot finish; what matters is
    that it stops for a reason other than the model type.
    """
    for model_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        directory = _model_dir(tmp_path / model_type, model_type)
        with pytest.raises(Exception) as caught:  # noqa: PT011 - any later failure will do
            _build(directory)
        assert "does not support model_type" not in str(caught.value)


def test_gemma3_refuses_fp16(tmp_path: Path) -> None:
    """Gemma 3 activations exceed the fp16 range, so fp16 is refused.

    Measured on the reference in fp32, largest absolute value leaving a decoder
    layer against the fp16 maximum of 65504: gemma-3-270m peaks at 102956 and
    gemma-3-4b at 298680, both of which overflow and make the engine emit token
    0 repeatedly. gemma-3-1b peaks at 61040, inside the range by 7%, which is
    luck rather than headroom.
    """
    for model_type in ("gemma3", "gemma3_text"):
        directory = _model_dir(tmp_path / f"fp16-{model_type}", model_type)
        with pytest.raises(NotImplementedError, match="does not support fp16"):
            _build_with(directory, precision="fp16")


def test_gemma2_keeps_fp16(tmp_path: Path) -> None:
    """Gemma 2 peaks at 4060, sixteen times inside the fp16 range."""
    directory = _model_dir(tmp_path / "fp16-gemma2", "gemma2")
    with pytest.raises(Exception) as caught:  # noqa: PT011 - a later failure is fine
        _build_with(directory, precision="fp16")
    assert "does not support fp16" not in str(caught.value)


def _build_with(model_dir: Path, *, precision: str) -> None:
    build_family(
        BuildRequest(
            model_dir=model_dir,
            output_path=model_dir / "out.bundle",
            family="gemma",
            task="text_generation",
            precision=precision,
        ),
        writer=None,
    )


def execution_request(root: Path) -> BuildRequest:
    return BuildRequest(root, root / "model.bundle", "gemma", "text_generation", "fp16")


def inputs(root: Path) -> BuildExecutionInputs:
    return BuildExecutionInputs("mtp", (NamedCheckpoint("draft", root),))


def test_execution_inputs_are_immutable(tmp_path):
    execution = inputs(tmp_path)
    with pytest.raises(FrozenInstanceError):
        execution.variant = "other"
    with pytest.raises(FrozenInstanceError):
        execution.checkpoints[0].role = "other"
    assert execution.checkpoints[0].model_dir is tmp_path


@pytest.mark.parametrize("value", ["", "../bad", "UPPER", "a-b", "a.b"])
def test_invalid_role_and_variant(tmp_path, value):
    with pytest.raises(ValueError, match="lowercase identifier"):
        NamedCheckpoint(value, tmp_path)
    with pytest.raises(ValueError, match="lowercase identifier"):
        BuildExecutionInputs(value)


def test_execution_requires_immutable_typed_companions(tmp_path):
    checkpoint = NamedCheckpoint("draft", tmp_path)
    with pytest.raises(TypeError, match="tuple"):
        BuildExecutionInputs("mtp", [checkpoint])
    with pytest.raises(TypeError, match="NamedCheckpoint"):
        BuildExecutionInputs("mtp", (object(),))
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("mtp", (checkpoint, checkpoint))
    with pytest.raises(TypeError, match="Path"):
        NamedCheckpoint("draft", str(tmp_path))


def test_local_checkpoint_required_and_rechecked(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="existing local directory"):
        NamedCheckpoint("draft", tmp_path / "missing")
    file = tmp_path / "file"
    file.write_text("not a directory")
    with pytest.raises(ValueError, match="existing local directory"):
        NamedCheckpoint("draft", file)
    directory = tmp_path / "companion"
    directory.mkdir()
    execution = inputs(directory)
    directory.rmdir()
    monkeypatch.setattr(build_core, "_select_backend", lambda _: pytest.fail("backend touched"))
    with pytest.raises(ValueError, match="existing local directory"):
        with_execution(execution_request(tmp_path), execution)


def test_untyped_execution_fails_before_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(build_core, "_select_backend", lambda _: pytest.fail("backend touched"))
    with pytest.raises(TypeError, match="BuildExecutionInputs"):
        with_execution(execution_request(tmp_path), {"variant": "paired"})




@pytest.mark.parametrize("variant", ["mtp", "dspark"])
def test_edge_cli_routes_through_the_ordinary_family_entrypoint(tmp_path, monkeypatch, variant):
    from families.gemma.edge_llm import builder as edge_builder
    from tensorrt_model_connect import family_cli as build_cli

    source = _model_dir(tmp_path / "target", "gemma4_unified")
    draft = tmp_path / "draft"
    draft.mkdir()
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, GemmaBuildRequest)
        assert request.execution is execution
        seen.append(execution)
        writer.set_header(family="gemma", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(edge_builder, "build", paired)
    assert build_cli.main(["gemma",
        "build", str(source), "--precision", "fp16",
        "-o", str(output), "--execution-variant", variant,
        "--companion", f"draft={draft}",
    ]) == 0
    assert seen == [BuildExecutionInputs(variant, (NamedCheckpoint("draft", draft),))]
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "mtp", "--companion", "missing_separator"],
    ["--execution-variant", "mtp", "--companion", "=path"],
    ["--execution-variant", "mtp", "--companion", "draft="],
    ["--execution-variant", "mtp", "--companion", "draft=https://example.com/model"],
])
def test_bad_edge_cli_inputs_fail_before_backend_and_bundle(tmp_path, monkeypatch, options):
    from tensorrt_model_connect import family_cli as build_cli

    from families.gemma import cli as owner

    source = _model_dir(tmp_path / "target", "gemma4_unified")
    output = tmp_path / "pair.bundle"
    monkeypatch.setattr(owner, "select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(owner, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["gemma", "build", str(source), "-o", str(output), *options])
    assert not output.exists()


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_family_failure_preserves_existing_publication(tmp_path, monkeypatch, failure):
    from families.gemma.edge_llm import builder as edge_builder

    request = with_execution(execution_request(tmp_path), inputs(tmp_path))
    request.output_path.write_bytes(b"previous valid publication")

    def fail(actual, writer, execution):
        writer.set_header(family="gemma", task=actual.task, backend=actual.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(edge_builder, "build", fail)
    with pytest.raises(type(failure)) as caught:
        build_core.build(request)
    assert caught.value is failure
    assert request.output_path.read_bytes() == b"previous valid publication"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.bundle"]


def test_edge_request_keeps_graph_callback_and_all_ordinary_fields(tmp_path):
    from dataclasses import fields, replace

    def callback(layer):
        return layer
    ordinary = replace(execution_request(tmp_path), graph_transform=callback, max_sequence_length=128)
    extended = with_execution(ordinary, inputs(tmp_path))
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(ordinary, field.name)
    with pytest.raises(FrozenInstanceError):
        extended.execution = None


def test_family_cli_without_extra_options_preserves_native_request(tmp_path):
    from families.gemma.edge_llm import cli

    assert cli.execution_inputs(None) is None


def test_paired_request_cannot_be_dispatched_to_another_family(tmp_path):
    from dataclasses import replace

    request = replace(execution_request(tmp_path), family="another_owner")
    with pytest.raises(ValueError, match="requires the gemma family"):
        with_execution(request, inputs(tmp_path))


@pytest.mark.parametrize("options", [[], ["--precision", "fp16", "--max-sequence-length", "64"]])
def test_declared_build_matches_legacy_request(tmp_path, monkeypatch, options):
    """Owner command preserves ordinary request defaults and explicit controls."""
    import json
    from families.gemma import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "gemma"}))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["gemma", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "gemma"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.gemma.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "gemma"
    assert captured[0].task == "text_generation"
    assert captured[0].precision == ("fp16" if options else "fp32")
    assert not output.exists()


def test_declared_help_is_offline_and_dependency_free():
    """Actual child-process help needs neither a checkpoint nor GPU imports."""
    import subprocess
    import sys

    code = """
import sys
from tensorrt_model_connect.family_cli import main
try:
    main(["gemma", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.gemma.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc gemma build" in result.stdout

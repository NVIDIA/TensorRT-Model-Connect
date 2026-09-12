# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import json

import pytest
from tensorrt_model_connect import BuildRequest, build
from tensorrt_model_connect.model_support import ModelMetadata
from families.cosyvoice3 import model, support
from families.cosyvoice3.tests.test_bundle import read_bundle_section


FILES = (
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json",
)


@pytest.mark.parametrize("model_type", ["", "cosyvoice3"])
def test_exact_support(model_type):
    result = support.describe(ModelMetadata(dict(model_type=model_type), {}, FILES))
    assert result.default_task == "audio_generation"
    assert result.default_precision == "fp32"
    for missing in FILES:
        assert (
            support.describe(ModelMetadata({}, {}, tuple(f for f in FILES if f != missing))) is None
        )
    assert support.describe(ModelMetadata(dict(model_type="cosyvoice2"), {}, FILES)) is None


@pytest.mark.parametrize(
    "options",
    [
        dict(precision="fp16"),
        dict(task="text_generation"),
        dict(max_sequence_length=15),
        dict(max_sequence_length=2049),
        dict(max_batch_size=2),
        dict(dynamic_kv_cache=True),
    ],
)
def test_build_rejects_unsupported_before_execution(tmp_path, monkeypatch, options):
    monkeypatch.setattr(model, "_build_bundle", lambda *args: pytest.fail("builder ran"))
    kwargs = dict(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="cosyvoice3",
        task="audio_generation",
        precision="fp32",
    )
    kwargs.update(options)
    with pytest.raises((ValueError, NotImplementedError)):
        build(BuildRequest(**kwargs))
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("verbose", [False, True])
def test_standard_build_writes_six_engines_without_fixed_voice(tmp_path, monkeypatch, capsys, verbose):
    from families.cosyvoice3 import __main__ as components, config, reference, tts

    monkeypatch.setattr(config, "read_config", lambda _: None)
    monkeypatch.setattr(reference, "coefficients", lambda: {"schema": 1})
    monkeypatch.setattr(
        tts,
        "text_tokenizer",
        lambda _: SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "{}")),
    )
    seen = []

    def component(args):
        args.output.mkdir()
        name = args.output.name
        seen.append((name, vars(args).copy()))
        (args.output / (name + ".plan")).write_bytes(b"fixture-" + name.encode())
        (args.output / "manifest.json").write_text("{}")

    monkeypatch.setattr(components, "build", component)
    monkeypatch.setattr(components, "build_conditioner", component)
    monkeypatch.setattr(components, "build_speech_component", component)
    target = tmp_path / "model.bundle"
    build(
        BuildRequest(
            model_dir=tmp_path,
            output_path=target,
            family="cosyvoice3",
            task="audio_generation",
            precision="fp32",
            max_sequence_length=128,
            verbose=verbose,
        )
    )
    result = json.loads(read_bundle_section(target, "config.json"))
    assert result["cosyvoice3_schema"] == 2
    assert result["reference_voice"] == "per_request"
    assert result["cosyvoice3"]["total_tokens"] == 128
    assert "speaker" not in result["cosyvoice3"]
    assert [name for name, _ in seen] == [
        "llm",
        "conditioning",
        "flow",
        "hift",
        "campplus",
        "speech_tokenizer",
    ]
    for name, _ in seen:
        assert read_bundle_section(target, name + ".plan") == b"fixture-" + name.encode()
    assert seen[2][1]["max_frames"] == 256
    assert seen[0][1]["max_context"] == 512
    assert ("CosyVoice3: building llm" in capsys.readouterr().out) == verbose

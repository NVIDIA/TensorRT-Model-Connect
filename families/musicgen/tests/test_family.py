# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from families.musicgen.model import relative_buckets, validate_request
from families.musicgen.support import describe


def metadata():
    return {
        "model_type": "musicgen",
        "decoder": {
            "hidden_size": 1536,
            "num_hidden_layers": 48,
            "num_codebooks": 4,
            "vocab_size": 2048,
        },
        "text_encoder": {"model_type": "t5", "d_model": 768},
        "audio_encoder": {"model_type": "encodec", "audio_channels": 1, "sampling_rate": 32000},
    }


def test_exact_owner_and_task():
    family, support = resolve_family(ModelMetadata(metadata(), {}))
    assert family == "musicgen"
    assert support.tasks == ("text_to_audio",)
    assert support.default_task == "text_to_audio"
    assert support.default_precision == "fp32"


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("decoder", "hidden_size", 1024),
        ("decoder", "num_hidden_layers", 24),
        ("decoder", "num_codebooks", 8),
        ("audio_encoder", "audio_channels", 2),
        ("audio_encoder", "sampling_rate", 24000),
        ("text_encoder", "model_type", "umt5"),
    ],
)
def test_other_topologies_are_not_claimed(part, key, value):
    config = copy.deepcopy(metadata())
    config[part][key] = value
    assert describe(ModelMetadata(config, {})) is None
    assert describe(ModelMetadata({"model_type": "musicgen_melody"}, {})) is None


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "fp16"},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
        {"max_batch_size": 2},
        {"task": "text_to_speech"},
        {"quantization": "fp8"},
        {"fp32_layers": (0,)},
        {"mtp_seq_len": 2},
        {"image_height": 32},
        {"max_sequence_length": 10},
        {"max_sequence_length": 1505},
        {"backend": "trt_rtx"},
    ],
)
def test_unsupported_build_options_fail_before_weights(options):
    kwargs = dict(
        model_dir=Path("unused"),
        output_path=Path("unused.bundle"),
        family="musicgen",
        task="text_to_audio",
        precision="fp32",
    )
    kwargs.update(options)
    with pytest.raises(ValueError):
        validate_request(BuildRequest(**kwargs))


def test_relative_position_buckets_match_official_t5():
    import torch
    from transformers.models.t5.modeling_t5 import T5Attention

    positions = torch.arange(64)[None, :] - torch.arange(64)[:, None]
    expected = T5Attention._relative_position_bucket(positions, True, 32, 128)
    assert (relative_buckets(64) == expected.numpy()).all()


def test_manifest_revision_and_duration_are_explicit():
    manifest = json.loads(
        (Path(__file__).parent / "manifests/musicgen-medium-fp32.json").read_text()
    )
    assert len(manifest["hf_revision"]) == 40
    assert (manifest["max_sequence_length"] - 4) * 640 == 32000


def test_native_tokenizer_matches_official_checkpoint(tmp_path):
    from transformers import AutoTokenizer

    if "TRTMC_MUSICGEN_MODEL_DIR" not in os.environ:
        pytest.skip("native tokenizer parity requires the pinned checkpoint and native runtime")
    model_dir = Path(os.environ["TRTMC_MUSICGEN_MODEL_DIR"])
    binary = Path(os.environ["TRTMC_RUNTIME_ROOT"]) / "test_musicgen_tokenizer"
    prompts = [
        "A mellow jazz piano with soft drums.",
        "Energetic rock music with electric guitar.",
        "  jazz\t piano\n drums  ",
        "Café music",
        "Cafe\u0301 music",
        "Ｆｕｎｋ music",
        "warm\u00a0bass",
        "",
        "\u2460 guitar \ufb01ngerstyle",
        "gentle \u200b piano",
        "piano <extra_id_0> drums",
        "piano<extra_id_0>drums",
        "jazz </s> piano",
        "<pad> soft music",
        "<unk><extra_id_99></s>",
        "guitar 🎹🎶",
    ]
    official = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(official.backend_tokenizer.to_str())
    expected = [official.encode(prompt) for prompt in prompts]
    result = subprocess.run(
        [str(binary), str(tokenizer_json), *prompts],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == expected


def test_bundle_preserves_the_official_tokenizer_backend():
    from transformers import AutoTokenizer

    if "TRTMC_MUSICGEN_BUNDLE" not in os.environ:
        pytest.skip("bundle tokenizer regression check requires a built bundle")
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["TRTMC_MUSICGEN_MODEL_DIR"], local_files_only=True
    )
    with Path(os.environ["TRTMC_MUSICGEN_BUNDLE"]).open("rb") as bundle:
        assert bundle.read(8) == b"BUNDLE\x01\x00"
        header_size = struct.unpack("<Q", bundle.read(8))[0]
        header = json.loads(bundle.read(header_size))
        assert (header["family"], header["task"], header["backend"]) == (
            "musicgen",
            "text_to_audio",
            "trt",
        )
        section = header["sections"]["tokenizer.json"]
        bundle.seek(16 + header_size + section["offset"])
        actual = json.loads(bundle.read(section["length"]))
    assert actual == json.loads(tokenizer.backend_tokenizer.to_str())

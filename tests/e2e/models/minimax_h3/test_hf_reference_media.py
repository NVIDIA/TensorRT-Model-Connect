# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for model-owned MiniMax-H3 HF reference media decoding."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import wave

import numpy as np
from PIL import Image
import pytest
import torch

from tests.e2e.models.minimax_h3 import hf_reference


def _write_pcm_wav(path: Path, samples: list[tuple[int, int]], sample_rate: int) -> None:
    payload = b"".join(struct.pack("<hh", *frame) for frame in samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(payload)


def test_decode_reference_wav_preserves_pcm_samples_and_rate(tmp_path: Path) -> None:
    path = tmp_path / "reference.wav"
    samples = [(-32768, 32767), (-16384, 16384), (0, -1)]
    _write_pcm_wav(path, samples, 22050)

    waveform, sample_rate = hf_reference._decode_reference_wav(path)

    expected = torch.tensor(samples, dtype=torch.float32).T / 32768.0
    assert sample_rate == 22050
    assert waveform.dtype == torch.float32
    assert waveform.is_contiguous()
    torch.testing.assert_close(waveform, expected, rtol=0.0, atol=0.0)


def test_official_references_receive_ordered_decoded_media(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    _write_pcm_wav(audio_path, [(-32768, 32767), (0, 16384)], 32000)
    first_pixels = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    second_pixels = np.array(
        [[[21, 22, 23], [24, 25, 26]], [[27, 28, 29], [30, 31, 32]]], dtype=np.uint8
    )
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(first_pixels).save(first_path)
    Image.fromarray(second_pixels).save(second_path)
    video_path = tmp_path / "video.json"
    video_path.write_text(
        json.dumps(
            {
                "fps": 1.5,
                "frames": [first_path.name, second_path.name],
                "audio": audio_path.name,
            }
        ),
        encoding="utf-8",
    )

    class FakeMiniMaxH3Reference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def load_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    references = hf_reference.build_official_references(
        [
            ("audio", audio_path),
            ("image", first_path),
            ("video", video_path),
            ("image", second_path),
        ],
        FakeMiniMaxH3Reference,
        load_image,
    )

    assert [
        "video"
        if "video" in ref.kwargs
        else next(key for key in ("audio", "image") if key in ref.kwargs)
        for ref in references
    ] == [
        "audio",
        "image",
        "video",
        "image",
    ]
    assert references[0].kwargs["sample_rate"] == 32000
    assert isinstance(references[0].kwargs["audio"], torch.Tensor)
    assert np.array_equal(np.asarray(references[1].kwargs["image"]), first_pixels)
    video = references[2].kwargs
    assert video["fps"] == 1.5
    assert video["sample_rate"] == 32000
    assert [np.asarray(frame).tolist() for frame in video["video"]] == [
        first_pixels.tolist(),
        second_pixels.tolist(),
    ]
    torch.testing.assert_close(video["audio"], references[0].kwargs["audio"], rtol=0.0, atol=0.0)
    assert np.array_equal(np.asarray(references[3].kwargs["image"]), second_pixels)


def test_video_reference_receipt_binds_nested_media_and_revalidates_it(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(np.full((2, 2, 3), 17, dtype=np.uint8)).save(first_path)
    Image.fromarray(np.full((2, 2, 3), 31, dtype=np.uint8)).save(second_path)
    audio_path = tmp_path / "soundtrack.wav"
    _write_pcm_wav(audio_path, [(0, 1), (2, 3)], 32000)
    manifest_path = tmp_path / "video.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fps": 2,
                "frames": [first_path.name, second_path.name],
                "audio": audio_path.name,
            }
        ),
        encoding="utf-8",
    )

    records, identities, resolved_videos = hf_reference.bind_reference_input_records(
        [("video", manifest_path)]
    )

    assert len(records) == 1
    assert records[0]["kind"] == "video"
    assert [frame["path"] for frame in records[0]["frames"]] == [
        first_path.name,
        second_path.name,
    ]
    assert all(frame["bytes"] > 0 and len(frame["sha256"]) == 64 for frame in records[0]["frames"])
    assert records[0]["soundtrack"]["path"] == audio_path.name
    assert records[0]["soundtrack"]["bytes"] == audio_path.stat().st_size
    assert len(records[0]["soundtrack"]["sha256"]) == 64
    assert [label for _path, _identity, label in identities] == [
        "reference 0 video",
        "reference 0 video frame 0",
        "reference 0 video frame 1",
        "reference 0 video soundtrack",
    ]

    loaded_paths = []

    class FakeMiniMaxH3Reference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def load_image(path: str) -> str:
        loaded_paths.append(Path(path))
        return path

    references = hf_reference.build_official_references(
        [("video", manifest_path)],
        FakeMiniMaxH3Reference,
        load_image,
        resolved_videos=resolved_videos,
    )

    assert loaded_paths == [first_path, second_path]
    assert references[0].kwargs["sample_rate"] == 32000
    hf_reference.validate_bound_reference_identities(identities)

    second_path.write_bytes(b"changed after generation")
    with pytest.raises(ValueError, match="reference 0 video frame 1"):
        hf_reference.validate_bound_reference_identities(identities)

    _, rebound_identities, _ = hf_reference.bind_reference_input_records([("video", manifest_path)])
    _write_pcm_wav(audio_path, [(4, 5), (6, 7), (8, 9)], 32000)
    with pytest.raises(ValueError, match="reference 0 video soundtrack"):
        hf_reference.validate_bound_reference_identities(rebound_identities)


@pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
def test_t2va_and_fl2va_keep_modular_model_index_construction(workflow: str) -> None:
    manager = object()

    class FakeModularPipeline:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return "fl-partition"

    class RefBlocksMustNotConstruct:
        def __init__(self):
            raise AssertionError("T2VA/FL2VA must not construct Ref2VA blocks")

    result = hf_reference.create_official_pipeline(
        workflow=workflow,
        model_path=Path("/model"),
        components_manager=manager,
        modular_pipeline_type=FakeModularPipeline,
        ref2va_blocks_type=RefBlocksMustNotConstruct,
    )

    assert result == "fl-partition"
    assert FakeModularPipeline.calls == [
        (
            (Path("/model"),),
            {"components_manager": manager},
        )
    ]


def test_ref2va_uses_official_ref_blocks_init_pipeline() -> None:
    manager = object()

    class ModularPipelineMustNotLoad:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("Ref2VA must not use the FL2VA modular index pipeline")

    class FakeRef2VABlocks:
        instances = []

        def __init__(self):
            self.calls = []
            self.instances.append(self)

        def init_pipeline(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "ref-partition"

    result = hf_reference.create_official_pipeline(
        workflow="ref2va",
        model_path=Path("/model"),
        components_manager=manager,
        modular_pipeline_type=ModularPipelineMustNotLoad,
        ref2va_blocks_type=FakeRef2VABlocks,
    )

    assert result == "ref-partition"
    assert len(FakeRef2VABlocks.instances) == 1
    assert FakeRef2VABlocks.instances[0].calls == [
        ((Path("/model"),), {"components_manager": manager})
    ]


@pytest.mark.parametrize(
    ("workflow", "components", "inputs", "expected_transformer"),
    [
        ("t2va", ["transformer"], ["prompt"], "transformer"),
        ("fl2va", ["transformer"], ["prompt", "image"], "transformer"),
        ("ref2va", ["transformer_ref"], ["prompt", "references"], "transformer_ref"),
    ],
)
def test_pipeline_partition_validation_records_exact_workflow(
    workflow: str,
    components: list[str],
    inputs: list[str],
    expected_transformer: str,
) -> None:
    blocks_name = "MiniMaxH3Ref2VABlocks" if workflow == "ref2va" else "FakeBlocks"
    pipeline_name = "MiniMaxH3Ref2VAModularPipeline" if workflow == "ref2va" else "FakePipeline"
    blocks_type = type(blocks_name, (), {"input_names": inputs})
    pipeline_type = type(
        pipeline_name,
        (),
        {"component_names": ["vae", *components], "_blocks": blocks_type()},
    )

    record = hf_reference.validate_official_pipeline_partition(pipeline_type(), workflow)

    assert expected_transformer in record["component_names"]
    assert record["block_inputs"] == sorted(inputs)


@pytest.mark.parametrize(
    ("workflow", "component"),
    [
        ("t2va", "transformer"),
        ("fl2va", "transformer"),
        ("ref2va", "transformer_ref"),
    ],
)
def test_compile_updates_exact_workflow_transformer(workflow: str, component: str) -> None:
    original = object()
    compiled = object()
    compile_calls = []

    class FakePipeline:
        component_names = ["vae", component]

        def __init__(self):
            setattr(self, component, original)
            self.update_calls = []

        def update_components(self, **kwargs):
            self.update_calls.append(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)

    def compile_function(value, **kwargs):
        compile_calls.append((value, kwargs))
        return compiled

    pipe = FakePipeline()
    selected = hf_reference.compile_official_transformer(
        pipe,
        workflow,
        mode="max-autotune-no-cudagraphs",
        compile_function=compile_function,
    )

    assert selected == component
    assert compile_calls == [(original, {"mode": "max-autotune-no-cudagraphs", "dynamic": False})]
    assert pipe.update_calls == [{component: compiled}]
    assert getattr(pipe, component) is compiled


@pytest.mark.parametrize(
    ("workflow", "wrong_component"),
    [("t2va", "transformer_ref"), ("fl2va", "transformer_ref"), ("ref2va", "transformer")],
)
def test_compile_rejects_wrong_workflow_partition(
    workflow: str,
    wrong_component: str,
) -> None:
    class FakePipeline:
        component_names = ["vae", wrong_component]

        def __init__(self):
            setattr(self, wrong_component, object())

        def update_components(self, **_kwargs):
            raise AssertionError("wrong partition must not be updated")

    def compile_function(*_args, **_kwargs):
        raise AssertionError("wrong partition must not be compiled")

    with pytest.raises(ValueError, match="wrong checkpoint partition"):
        hf_reference.compile_official_transformer(
            FakePipeline(),
            workflow,
            mode="default",
            compile_function=compile_function,
        )


def test_compile_rejects_pipeline_that_ignores_component_update() -> None:
    original = object()

    class FakePipeline:
        component_names = ["vae", "transformer_ref"]
        transformer_ref = original

        def update_components(self, **_kwargs):
            pass

    with pytest.raises(ValueError, match="did not install compiled component transformer_ref"):
        hf_reference.compile_official_transformer(
            FakePipeline(),
            "ref2va",
            mode="default",
            compile_function=lambda *_args, **_kwargs: object(),
        )


@pytest.mark.parametrize(
    ("workflow", "components", "inputs", "message"),
    [
        ("ref2va", ["transformer"], ["prompt", "references"], "wrong checkpoint partition"),
        ("ref2va", ["transformer_ref"], ["prompt"], "does not consume references"),
        ("t2va", ["transformer_ref"], ["prompt"], "wrong checkpoint partition"),
    ],
)
def test_pipeline_partition_validation_rejects_wrong_runtime(
    workflow: str,
    components: list[str],
    inputs: list[str],
    message: str,
) -> None:
    blocks_type = type("MiniMaxH3Ref2VABlocks", (), {"input_names": inputs})
    pipeline_type = type(
        "MiniMaxH3Ref2VAModularPipeline",
        (),
        {"component_names": ["vae", *components], "_blocks": blocks_type()},
    )

    with pytest.raises(ValueError, match=message):
        hf_reference.validate_official_pipeline_partition(pipeline_type(), workflow)


def test_ref2va_partition_validation_rejects_wrong_official_types() -> None:
    class MiniMaxH3Blocks:
        input_names = ["prompt", "references"]

    class MiniMaxH3ModularPipeline:
        component_names = ["vae", "transformer_ref"]
        _blocks = MiniMaxH3Blocks()

    with pytest.raises(ValueError, match="wrong official pipeline types"):
        hf_reference.validate_official_pipeline_partition(MiniMaxH3ModularPipeline(), "ref2va")

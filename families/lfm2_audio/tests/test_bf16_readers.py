# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 checkpoint reads must work without unrelated module import side effects."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
BF16_BITS = [0x3F80, 0xC020, 0x8000, 0x3F00]
EXPECTED = {
    "bf16": [value << 16 for value in BF16_BITS],
    "fp32": [0x3E000000, 0xC0E00000, 0x80000000],
}


def checkpoint_tensors():
    return {
        "bf16": ("BF16", [4], struct.pack("<4H", *BF16_BITS)),
        "fp32": ("F32", [3], struct.pack("<3I", *EXPECTED["fp32"])),
    }


def write_checkpoint(path, tensors=None):
    header, data = {}, bytearray()
    for name, (dtype, shape, raw) in (tensors or checkpoint_tensors()).items():
        start = len(data)
        data.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    header = json.dumps(header, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def write_layout(root, stem, indexed, tensors=None):
    tensors = tensors or checkpoint_tensors()
    if not indexed:
        write_checkpoint(root / (stem + ".safetensors"), tensors)
        return
    weight_map = {}
    for index in range(2):
        filename = f"shard-{index + 1:05d}.safetensors"
        entries = dict(list(tensors.items())[index::2])
        write_checkpoint(root / filename, entries)
        weight_map.update(dict.fromkeys(entries, filename))
    (root / (stem + ".safetensors.index.json")).write_text(json.dumps({"weight_map": weight_map}))


def run_reader(path, *, direct):
    # A fresh interpreter prevents checkpoint_mapper's ml_dtypes import from
    # accidentally making a broken standalone read_weights test pass.
    script = """
import json, sys
from pathlib import Path
import numpy as np
assert 'ml_dtypes' not in sys.modules
if sys.argv[2] == 'direct':
    from families.lfm2_audio.model import read_weights
    assert 'families.lfm2_audio.checkpoint_mapper' not in sys.modules
    values = read_weights(Path(sys.argv[1]))
else:
    from families.lfm2_audio.checkpoint_mapper import _open_safetensors, _load_tensor
    readers = _open_safetensors(Path(sys.argv[1]))
    values = {name: _load_tensor(readers, name) for name in readers.tensor_map}
assert 'torch' not in sys.modules
assert all(value.dtype == np.float32 for value in values.values())
print(json.dumps({name: value.view(np.uint32).tolist() for name, value in values.items()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), "direct" if direct else "mapper"],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_standalone_read_weights_registers_bf16(tmp_path):
    path = tmp_path / "model.safetensors"
    write_checkpoint(path)
    assert run_reader(path, direct=True) == EXPECTED


@pytest.mark.parametrize("stem", ["model", "diffusion_pytorch_model"])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_readers_load_bf16_checkpoint_layouts(tmp_path, stem, indexed, direct):
    write_layout(tmp_path, stem, indexed)
    assert run_reader(tmp_path, direct=direct) == EXPECTED


@pytest.mark.parametrize("filter_error", [False, True])
def test_directory_filter_runs_before_loading_and_closes_shards(
    monkeypatch, tmp_path, filter_error
):
    import numpy as np

    from families.lfm2_audio import checkpoint_mapper, model

    write_layout(tmp_path, "model", True)
    actual = checkpoint_mapper._open_safetensors(tmp_path)
    loaded, closed = [], []

    class Reader:
        def __init__(self, reader):
            self.reader = reader

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            closed.append(self.reader)
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            assert name == "bf16", "excluded tensor was materialized"
            loaded.append(name)
            return self.reader.get_tensor(name)

    readers = checkpoint_mapper._ReaderCollection([Reader(reader) for reader in actual])
    monkeypatch.setattr(checkpoint_mapper, "_open_safetensors", lambda _: readers)

    def keep(name):
        if filter_error and name == "fp32":
            raise ValueError("filter failed")
        return name == "bf16"

    if filter_error:
        with pytest.raises(ValueError, match="filter failed"):
            model.read_weights(tmp_path, keep=keep)
    else:
        values = model.read_weights(tmp_path, keep=keep)
        assert set(values) == {"bf16"}
        assert values["bf16"].view(np.uint32).tolist() == EXPECTED["bf16"]
    assert loaded == ["bf16"]
    assert len(closed) == 2 and {id(reader) for reader in closed} == {
        id(reader) for reader in actual
    }


@pytest.mark.parametrize("indexed", [False, True])
def test_build_materializes_only_consumed_weights_and_releases_detokenizer(
    monkeypatch, tmp_path, indexed
):
    import gc
    from types import ModuleType
    import weakref

    import ml_dtypes
    import numpy as np

    from families.lfm2_audio import checkpoint_mapper, model

    fixtures = Path(__file__).parent / "fixtures"
    raw = json.loads((fixtures / "config.json").read_text())
    detok_raw = json.loads((fixtures / "detokenizer_config.json").read_text())
    monkeypatch.setattr(model, "load_config", lambda _: (raw, detok_raw))
    names = [
        "lfm.embed_tokens.weight",
        "audio_embedding.embedding.weight",
        "conformer.pre_encode.out.weight",
        "audio_adapter.model.0.weight",
        "depth_linear.weight",
        "depthformer.layers.0.ffn_norm.weight",
        "future_non_backbone.weight",
        *(f"depth_embeddings.{i}.embedding.weight" for i in range(8)),
    ]
    values, expected = {}, {}
    for index, name in enumerate([*names, "emb.emb.weight"]):
        bits = np.array([0x3F80 + index, 0xC020, 0x8000, 0x3F00], dtype=np.uint16).reshape(2, 2)
        values[name] = bits.view(ml_dtypes.bfloat16)
        expected[name] = bits.astype(np.uint32) << 16
    values["conformer.pre_encode.out.weight"] = (
        np.array([0x3E000000, 0xC0E00000, 0x80000000, 0x3F800000], dtype=np.uint32)
        .reshape(2, 2)
        .view(np.float32)
    )
    expected["conformer.pre_encode.out.weight"] = (
        values["conformer.pre_encode.out.weight"].view(np.uint32).copy()
    )
    loaded, mapped, stages = {"root": [], "detokenizer": []}, {}, []
    opened, closed = [], []
    unused_root = ["lfm.layers.0.feed_forward.w1.weight", "lfm.embedding_norm.weight"]
    detok_names = ["lin.weight", "emb.emb.weight", "lfm.layers.0.feed_forward.w1.weight"]

    def tensors(selected):
        return {
            name: (
                "BF16" if values.get(name, values[names[0]]).dtype.name == "bfloat16" else "F32",
                [2, 2],
                values.get(name, values[names[0]]).tobytes(),
            )
            for name in selected
        }

    write_layout(tmp_path, "model", indexed, tensors([*names, *unused_root]))
    detok_root = tmp_path / "audio_detokenizer"
    detok_root.mkdir()
    write_layout(
        detok_root, "diffusion_pytorch_model" if indexed else "model", indexed, tensors(detok_names)
    )
    original_open = checkpoint_mapper.safe_open

    class Reader:
        def __init__(self, which, reader, path):
            self.which = which
            self.reader = reader
            self.path = path

        def __enter__(self):
            self.reader.__enter__()
            opened.append(self.path)
            return self

        def __exit__(self, *_):
            closed.append(self.path)
            return self.reader.__exit__(*_)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            allowed = names if self.which == "root" else ["emb.emb.weight"]
            assert name in allowed, f"unused tensor materialized: {name}"
            loaded[self.which].append(name)
            return self.reader.get_tensor(name)

    def open_weights(path, *, framework):
        assert framework == "numpy"
        which = "detokenizer" if Path(path).parent.name == "audio_detokenizer" else "root"
        if which == "detokenizer":
            gc.collect()
            assert mapped["detokenizer"]() is None, "serialized detokenizer weights still live"
        return Reader(which, original_open(path, framework=framework), str(path))

    monkeypatch.setattr(checkpoint_mapper, "safe_open", open_weights)

    class WeightMap(dict):
        pass

    def map_weights(path, *_args, **_kwargs):
        which = "detokenizer" if Path(path).name == "audio_detokenizer" else "backbone"
        weights = WeightMap(which=which)
        mapped[which] = weakref.ref(weights)
        return weights

    def engine(_config, weights, _length, **_kwargs):
        assert sorted(opened) == sorted(closed), "checkpoint readers still open during engine build"
        stages.append(weights["which"])
        return weights["which"].encode()

    def consume(stage, weights, *_args, **_kwargs):
        assert mapped["backbone"]() is None
        assert set(weights) == set(names)
        for name, value in weights.items():
            assert value.dtype == np.float32
            np.testing.assert_array_equal(value.view(np.uint32), expected[name])
        stages.append(stage)
        return stage.encode()

    for module_name, function_name, function in (
        ("backbone", "build_lfm2_engine", engine),
        ("conformer", "build_encoder", lambda *args, **kwargs: consume("encoder", *args, **kwargs)),
        (
            "depthformer",
            "build_depthformer",
            lambda *args, **kwargs: consume("depthformer", *args, **kwargs),
        ),
    ):
        module = ModuleType(f"families.lfm2_audio.{module_name}")
        setattr(module, function_name, function)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(checkpoint_mapper, "load_lfm2_weights", map_weights)

    class Writer:
        def __init__(self):
            self.outputs = {}

        def set_header(self, **_kwargs):
            pass

        def add_bytes(self, name, data):
            self.outputs[name] = data

        def add_json(self, name, data):
            self.outputs[name] = data

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        (tmp_path / name).write_text(name)
    writer = Writer()
    model.build(model.BuildRequest(tmp_path, max_sequence_length=16), writer)
    assert sorted(loaded["root"]) == sorted(names)
    assert loaded["detokenizer"] == ["emb.emb.weight"]
    assert len(opened) == len(closed) == (4 if indexed else 2)
    assert sorted(opened) == sorted(closed)
    assert stages == ["backbone", "encoder", "depthformer", "detokenizer"]
    for output, name in (
        ("text_embeddings.f32", "lfm.embed_tokens.weight"),
        ("audio_embeddings.f32", "audio_embedding.embedding.weight"),
        ("detokenizer_embeddings.f32", "emb.emb.weight"),
    ):
        assert writer.outputs[output] == expected[name].astype("<u4").tobytes()
    assert (
        writer.outputs["depth_embeddings.f32"]
        == np.stack([expected[f"depth_embeddings.{i}.embedding.weight"] for i in range(8)])
        .astype("<u4")
        .tobytes()
    )

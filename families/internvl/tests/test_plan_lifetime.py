# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep serialized engines out of RAM while building the next network."""

import gc
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tiles", [1, 12])
def test_each_plan_is_staged_and_released_before_the_next_build(tmp_path, monkeypatch, tiles):
    pytest.importorskip("tensorrt")
    from families.internvl import model
    from families.internvl.edge_llm import dispatch
    from tensorrt_model_connect.bundle_writer import BundleWriter

    live = set()
    built = []

    class Plan(bytes):
        def __new__(cls, role):
            value = super().__new__(cls, role.encode())
            value.role = role
            live.add(role)
            return value

        def __del__(self):
            live.remove(self.role)

    def next_plan(role):
        gc.collect()
        assert not live, f"previous serialized engine retained during {role}: {live}"
        built.append(role)
        return Plan(role)

    class Model:
        @staticmethod
        def load_weights(*_args, **_kwargs):
            return {}

        @staticmethod
        def build_engine(config, *_args, **_kwargs):
            return next_plan(config.raw["_decoder_engine_role"])

        @staticmethod
        def build_vision_engine(*_args, **_kwargs):
            return next_plan("vision")

        @staticmethod
        def get_vl_config(_config):
            return {"image_token_id": 3, "vision_output_dim": 4, "num_image_pad_tokens": 256,
                    "max_image_tiles": tiles}

    monkeypatch.setattr(model, "_InternVLModel", Model)
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda _directory: {})
    monkeypatch.setattr(dispatch, "build", lambda request, writer, native: native(request, writer))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "internvl", "hidden_size": 4, "num_hidden_layers": 1,
        "vocab_size": 8, "num_attention_heads": 1, "num_key_value_heads": 1,
    }))
    request = SimpleNamespace(
        mtp_seq_len=None, dynamic_kv_cache=False, image_height=None, image_width=None,
        video_num_frames=None, max_batch_size=1, context_parallel_size=1,
        task="vision_language_generation", tensor_parallel_size=1, quantization=None,
        fp32_layers=(), model_dir=tmp_path, precision="fp16",
        max_sequence_length=4096 if tiles > 1 else 768,
        verbose=False, backend="trt", family="internvl",
        output_path=tmp_path / "model.bundle", graph_transform=None,
    )
    writer = BundleWriter(tmp_path / "model.bundle")
    staged = {}
    add_bytes = writer.add_bytes

    def stage(name, value):
        if name.endswith(".plan"):
            staged[name] = bytes(value)
        add_bytes(name, value)

    monkeypatch.setattr(writer, "add_bytes", stage)
    try:
        model.build(request, writer)
        writer.finish()
    finally:
        writer.abort()
    gc.collect()
    assert built == ["dual_profile", "vision"]
    assert staged == {"engine.plan": b"dual_profile", "vision.plan": b"vision"}
    assert not live
    assert (tmp_path / "model.bundle").is_file()

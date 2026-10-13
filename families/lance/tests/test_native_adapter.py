# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.lance.reference import adapter


class Host:
    Error = RuntimeError

    @staticmethod
    def required(request, name):
        return request[name]

    @staticmethod
    def timed(call):
        return call(), 2.5

    @staticmethod
    def invocation(observation, elapsed):
        return SimpleNamespace(observation=observation, model_call_ms=elapsed)


def upstream_fixture(monkeypatch, *, fail=False):
    loads, generations = [], []

    class Dataset:
        def __init__(self):
            self.data = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    class Loader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset

        def __iter__(self):
            return iter([self.dataset[0]])

    tokenizer = SimpleNamespace(decode=lambda ids: "A<|im_end|>")
    original_decode = tokenizer.decode

    def validate(*, val_data_cpu, inference_args, tokenizer):
        prompt = val_data_cpu["data"]["interleave_array"][1][1]
        generations.append((prompt, upstream.MAX_GENERATION_LENGTH))
        if fail:
            raise ValueError("upstream generation failed")
        assert tokenizer.decode(np.asarray([32, 151645])) == "A<|im_end|>"
        inference_args.prompt_data_dict["0"] = prompt

    def main():
        loads.append("load")
        loader = upstream.DataLoader(Dataset(), batch_size=1)
        iterator = iter(loader)
        inference = SimpleNamespace(prompt_data_dict={})
        for _ in upstream.trange(len(loader)):
            try:
                data = next(iterator)
            except StopIteration:
                break
            upstream.validate_on_fixed_batch(val_data_cpu=data, inference_args=inference, tokenizer=tokenizer)
            assert tokenizer.decode is original_decode

    upstream = SimpleNamespace(DataLoader=Loader, validate_on_fixed_batch=validate, main=main,
                               simple_custom_collate=lambda values: values[0],
                               normalize_understanding_answer=lambda text: text.strip())
    monkeypatch.setattr(adapter.reference, "_model_paths", lambda args: (Path("model"), Path("vit"), Path("vae")))
    monkeypatch.setattr(adapter.reference, "_require_reference", lambda root: root)
    monkeypatch.setattr(adapter.reference, "_configure_upstream_vae", lambda *args: None)
    monkeypatch.setattr(adapter.reference, "_load_upstream", lambda root: upstream)
    monkeypatch.setattr(adapter.reference, "_sdpa_vit_path", lambda root, vit: vit)
    monkeypatch.setattr(adapter.reference, "_working_directory", lambda root: nullcontext())
    return loads, generations


def test_native_reuses_one_model_and_keeps_each_prompt_and_generation_length(monkeypatch, tmp_path):
    loads, generations = upstream_fixture(monkeypatch)
    spec = SimpleNamespace(mode="eager", precision="bf16", model="checkpoint", revision="pinned")
    native = adapter.Adapter(spec, Host())
    root = native.root
    try:
        for prompt, length in [("first", 7), ("second", 11)]:
            result = native.invoke({"prompt": prompt, "image_path": "/image.png", "max_new_tokens": length}, tmp_path)
            assert result.observation["text"] == prompt and result.model_call_ms == 2.5
            assert result.observation["token_ids"] == [32, 151645]
            assert result.observation["output_tokens"] == 2
            assert result.observation["input_image_height"] == result.observation["input_image_width"] == 448
        assert loads == ["load"] and generations == [("first", 7), ("second", 11)]
    finally:
        native.close()
    assert not native.worker.is_alive() and not root.exists()


def test_native_generation_failure_is_returned_and_stops_further_requests(monkeypatch, tmp_path):
    upstream_fixture(monkeypatch, fail=True)
    native = adapter.Adapter(SimpleNamespace(mode="eager", precision="bf16", model="checkpoint", revision=None), Host())
    try:
        request = {"prompt": "question", "image_path": "/image.png"}
        with pytest.raises(RuntimeError, match="upstream generation failed"):
            native.invoke(request, tmp_path)
        with pytest.raises(RuntimeError, match="worker stopped"):
            native.invoke(request, tmp_path)
    finally:
        native.close()


def test_image_input_matches_fixed_square_task_geometry():
    from PIL import Image

    values = adapter.image_values(Image.new("RGB", (39, 21), (0, 0, 0)))
    assert values.shape == (3, 448, 448) and values.dtype == np.float32 and values.flags.c_contiguous
    np.testing.assert_allclose(values[:, 10, 20], [-1.7922626, -1.7520971, -1.4802198], rtol=1e-6)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from trtmc_perf_serving import openai
from trtmc_perf_serving.files import FILE_KEY, inline_files, materialize_files


def test_inline_and_materialize_round_trip_nested_path_fields(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"png-bytes")
    request = {"image_path": str(image), "frame_paths": [str(image), str(image)], "prompt": str(image),
               "nested": {"depth_path": str(image)}, "missing_path": "/does/not/exist.png"}

    inlined = inline_files(request)

    assert inlined["image_path"][FILE_KEY]["suffix"] == ".png"
    assert inlined["prompt"] == str(image)  # only *_path / *_paths fields are files
    assert inlined["missing_path"] == "/does/not/exist.png"
    restored = materialize_files(inlined, tmp_path / "server")
    for path in (restored["image_path"], *restored["frame_paths"], restored["nested"]["depth_path"]):
        assert open(path, "rb").read() == b"png-bytes"
    assert len({restored["image_path"], *restored["frame_paths"]}) == 3


@pytest.mark.parametrize("reference", [
    {FILE_KEY: {"b64": "not base64!"}},
    {FILE_KEY: {"b64": "", "suffix": "/../../etc"}},
    {FILE_KEY: {"b64": ""}, "other": 1},
])
def test_materialize_rejects_malformed_references(tmp_path, reference):
    with pytest.raises(ValueError):
        materialize_files({"image_path": reference}, tmp_path)


def test_completion_maps_sampling_and_rejects_unknown_fields():
    request = openai.completion({"model": "m", "prompt": "hi", "max_tokens": 8, "stream": True,
                                 "stream_options": {"include_usage": True}}, {"prompt": "base", "seed": 1})
    assert request == {"prompt": "hi", "seed": 1, "max_new_tokens": 8, "use_chat_template": False}
    with pytest.raises(openai.RequestError, match="ignore_eos"):
        openai.completion({"prompt": "hi", "ignore_eos": True}, {})


def test_chat_maps_single_image_to_inline_file():
    data = base64.b64encode(b"jpeg").decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        {"type": "text", "text": "describe"}]}], "max_completion_tokens": 4}
    request = openai.chat(body, {"image_path": "/base.png"})
    assert request["prompt"] == "describe"
    assert request["use_chat_template"] is True
    assert request["image_path"] == {FILE_KEY: {"suffix": ".jpg", "b64": data}}
    assert request["max_new_tokens"] == 4


def test_multi_message_chat_is_rendered_once_for_every_backend():
    body = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
                         {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}],
            "enable_thinking": False, "stop": ["\n"], "max_completion_tokens": 5}
    seen = []

    def renderer(messages, thinking):
        seen.append((messages, thinking))
        return "<rendered>"

    request = openai.chat(body, {"image_path": "/base.png"}, renderer)
    assert request == {"prompt": "<rendered>", "use_chat_template": False, "max_new_tokens": 5}
    assert seen == [([{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}], False)]
    with pytest.raises(openai.RequestError, match="renderer"):
        openai.chat(body, {})


def test_stop_sequences_truncate_text():
    assert openai.stop_sequences({"stop": "\n"}) == ("\n",)
    assert openai.truncate_at_stop(" B\nQuestion: next", ("\n", "Question")) == " B"
    assert openai.truncate_at_stop("no stop here", ("\n",)) == "no stop here"
    with pytest.raises(openai.RequestError):
        openai.stop_sequences({"stop": [""]})


def test_ranking_and_image_generation_mapping():
    ranking = openai.ranking({"query": {"text": "q"}, "passages": [{"text": "a"}, {"text": "b"}]}, {})
    assert ranking == {"query": "q", "documents": ["a", "b"]}
    image = openai.image_generation({"prompt": "cat", "size": "512x256", "num_inference_steps": "4"}, {"seed": 0})
    assert image == {"prompt": "cat", "seed": 0, "width": 512, "height": 256, "num_steps": 4}

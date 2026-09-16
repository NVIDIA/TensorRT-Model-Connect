# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trtmc_benchmark.task_adapters import resolve_task_case
from trtmc_benchmark.types import BenchmarkError


@pytest.mark.parametrize("task", [
    "text_continuation", "conditional_text_generation",
    "text_to_pooled_features", "text_to_token_features", "text_to_head_scores",
])
def test_text_source_keeps_ids_empty_inputs_and_config(tmp_path, task):
    ids = [-(1 << 31), 0, (1 << 31) - 1]
    case = {"inputs": {"token_ids": ids}, "config": {"mode": "", "enabled": False}}
    resolved = resolve_task_case(task, case, tmp_path)
    assert resolved.request == {"token_ids": ids, "config": case["config"]}
    assert resolved.request["token_ids"] is not ids
    assert resolve_task_case(task, {"token_ids": []}, tmp_path).request == {"token_ids": []}
    assert resolve_task_case(task, {"prompt": ""}, tmp_path).request == {"prompt": ""}


@pytest.mark.parametrize("task", ["text_continuation", "conditional_text_generation"])
@pytest.mark.parametrize("invalid", [[True], [1.5], [1 << 31], [-(1 << 31) - 1], "[1]", None])
def test_generation_rejects_invalid_id_representation(tmp_path, task, invalid):
    with pytest.raises(BenchmarkError, match="int32 array"):
        resolve_task_case(task, {"token_ids": invalid}, tmp_path)


@pytest.mark.parametrize("source", [
    {"prompt": ""}, {"test_prompt": ""}, {"source_text": ""},
    {"prompt_file": "prompt.txt"}, {"prompt_repeat": {"text": "x", "count": 2}},
])
def test_generation_does_not_choose_between_ids_and_text(tmp_path, source):
    with pytest.raises(BenchmarkError, match="exactly one"):
        resolve_task_case("text_continuation", {"token_ids": [], **source}, tmp_path)


@pytest.mark.parametrize("task", [
    "text_generation", "vision_language_generation", "corrupted_text_reconstruction",
    "text_summarization", "images_text_to_text", "unconditional_text_generation", "text_to_embedding",
])
def test_other_input_contracts_do_not_discard_token_ids(tmp_path, task):
    with pytest.raises(BenchmarkError):
        resolve_task_case(task, {"prompt": "x", "token_ids": [7]}, tmp_path)


@pytest.mark.parametrize("task", [
    "text_translation", "text_to_audio", "text_to_speech", "streaming_text_to_speech",
])
@pytest.mark.parametrize("nested", [False, True])
def test_utf8_only_tasks_do_not_drop_ids_from_valid_text_requests(tmp_path, task, nested):
    inputs = {"prompt": "hello"}
    case = {"inputs": inputs} if nested else inputs
    resolved = resolve_task_case(task, case, tmp_path)
    text_key = "source_text" if task == "text_translation" else "prompt"
    assert resolved.request[text_key] == "hello"
    inputs["token_ids"] = [7]
    with pytest.raises(BenchmarkError, match="token_ids"):
        resolve_task_case(task, case, tmp_path)


def test_existing_embedding_encode_operation_uses_pooled_input_contract(tmp_path):
    case = {"token_ids": [7,9], "config": {"scale": 2.0}}
    resolved = resolve_task_case("text_to_embedding", case, tmp_path, operation="encode")
    assert resolved.operation == "encode"
    assert resolved.request == case
    with pytest.raises(BenchmarkError, match="embedding role"):
        resolve_task_case("text_to_embedding", {"prompt": "x", "role": "query"}, tmp_path,
                          operation="encode")


@pytest.mark.parametrize("name", [
    "max_new_tokens", "temperature", "top_k", "top_p", "min_p", "seed",
    "repetition_penalty", "use_chat_template", "enable_thinking",
])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [0, False, ""])
def test_semantic_text_controls_preserve_explicit_values(tmp_path, name, nested, value):
    case = {"prompt": "hello", "inputs": {name: value}} if nested else {
        "prompt": "hello", name: value,
    }
    resolved = resolve_task_case("text_continuation", case, tmp_path)
    # Transport must not coerce invalid values before the family validates them.
    assert resolved.request == {"prompt": "hello", name: value}
    assert type(resolved.request[name]) is type(value)


@pytest.mark.parametrize("name", [
    "max_new_tokens", "temperature", "top_k", "top_p", "min_p", "seed",
    "repetition_penalty", "use_chat_template", "enable_thinking",
])
def test_semantic_text_controls_reject_duplicate_levels(tmp_path, name):
    case = {"prompt": "hello", name: 0, "inputs": {name: 0}}
    with pytest.raises(BenchmarkError, match=f"duplicate input/control for {name}"):
        resolve_task_case("text_continuation", case, tmp_path)


def test_semantic_text_controls_do_not_inject_defaults(tmp_path):
    assert resolve_task_case("text_continuation", {"prompt": "hello"}, tmp_path).request == {
        "prompt": "hello",
    }


def test_legacy_text_controls_keep_nested_temperature_and_other_defaults(tmp_path):
    case = {
        "prompt": "hello",
        "inputs": {
            "max_new_tokens": 0, "temperature": "0.5", "top_k": 7, "top_p": 0.8,
            "min_p": 0.1, "seed": 17, "repetition_penalty": 2.0,
            "use_chat_template": True, "enable_thinking": False,
        },
    }
    assert resolve_task_case("text_generation", case, tmp_path).request == {
        "prompt": "hello", "max_new_tokens": 128, "temperature": 0.5, "top_k": 1,
        "top_p": 1.0, "min_p": 0.0, "seed": -1, "repetition_penalty": 1.0,
        "use_chat_template": False, "enable_thinking": True,
    }


def test_legacy_text_controls_keep_coercions_and_top_level_precedence(tmp_path):
    case = {
        "prompt": "hello", "max_new_tokens": "7", "temperature": "0.5", "top_k": "3",
        "top_p": "0.8", "min_p": "0.1", "seed": "17", "repetition_penalty": "1.5",
        "use_chat_template": 0, "enable_thinking": "", "inputs": {"temperature": "0.9"},
    }
    assert resolve_task_case("text_generation", case, tmp_path).request == {
        "prompt": "hello", "max_new_tokens": 7, "temperature": 0.5, "top_k": 3,
        "top_p": 0.8, "min_p": 0.1, "seed": 17, "repetition_penalty": 1.5,
        "use_chat_template": False, "enable_thinking": False,
    }

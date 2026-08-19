# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static acceptance contracts for issue #928 and the pinned model card."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from tests.e2e_harness.manifest_loader import load_model_manifest

_ROOT = Path(__file__).resolve().parent
_PINNED_350M_REVISION = "f37d3f5c8c5484bc01dad379a595cf4c68c4e70e"
_STRATEGY = "lfm2_hybrid_conv_attention"


def _read_manifest(name: str) -> dict:
    return json.loads((_ROOT / "manifests" / name).read_text(encoding="utf-8"))


def test_model_root_registers_every_dense_manifest() -> None:
    model = tomllib.loads((_ROOT / "MODEL.toml").read_text(encoding="utf-8"))
    assert model["id"] == "lfm2"
    assert model["plugin"] == "lfm2"
    assert model["test_manifests"] == [
        "manifests/lfm2-350m-fp16.json",
        "manifests/lfm2-350m-bf16-model-card.json",
        "manifests/lfm2-700m.json",
        "manifests/lfm2-1.2b.json",
        "manifests/lfm2-2.6b.json",
    ]


def test_core_fp16_and_model_card_bf16_are_revision_pinned() -> None:
    fp16 = _read_manifest("lfm2-350m-fp16.json")
    bf16 = _read_manifest("lfm2-350m-bf16-model-card.json")
    for manifest, precision in ((fp16, "fp16"), (bf16, "bf16")):
        assert manifest["hf_id"] == "LiquidAI/LFM2-350M"
        assert manifest["hf_revision"] == _PINNED_350M_REVISION
        assert manifest["family"] == "lfm2"
        assert manifest["runtime_strategy"] == _STRATEGY
        assert manifest["task_strategy"] == "text_generation_causal"
        assert manifest["precision"] == precision
        assert manifest["trust_remote_code"] is False

        loaded = load_model_manifest(_ROOT / "manifests" / f"{manifest['name']}.json")
        assert loaded.hf_revision == _PINNED_350M_REVISION
        assert loaded.build_case.runtime_strategy == _STRATEGY


def test_core_cases_cover_exact_continuation_and_chat_template() -> None:
    manifest = _read_manifest("lfm2-350m-fp16.json")
    cases = {case["name"]: case for case in manifest["testcases"]}
    assert set(cases) == {"lfm2-350m-greedy", "lfm2-350m-chat"}

    greedy = cases["lfm2-350m-greedy"]
    assert greedy["reference_family"] == "lfm2_greedy_continuation"
    assert greedy["inputs"]["do_sample"] is False

    chat = cases["lfm2-350m-chat"]
    assert chat["reference_family"] == "lfm2_chat_template"
    assert chat["contract_config"]["use_chat_template"] is True
    assert chat["expected_prompt_token_ids"] == [
        1,
        6,
        6423,
        708,
        3493,
        856,
        779,
        5706,
        803,
        4481,
        540,
        42275,
        797,
        1235,
        2359,
        523,
        7,
        708,
        6,
        64015,
        708,
    ]
    assert chat["expected_continuation_token_ids"] == [41677, 7]
    assert chat["expected_continuation_text"] == "Paris"


def test_bf16_case_matches_documented_sampling_recipe() -> None:
    manifest = _read_manifest("lfm2-350m-bf16-model-card.json")
    assert manifest["max_cache_length"] == 32768
    assert manifest["model_card_contract"] == {
        "context_length": 32768,
        "vocab_size": 65536,
        "checkpoint_dtype": "bf16",
        "bos_token_id": 1,
        "eos_token_id": 7,
        "pad_token_id": 0,
        "use_default_system_prompt": False,
        "tool_token_ids": {
            "tool_list_start": 8,
            "tool_list_end": 9,
            "tool_call_start": 10,
            "tool_call_end": 11,
            "tool_response_start": 12,
            "tool_response_end": 13,
        },
    }
    usage = manifest["model_card_usage"]
    assert usage["trtmc_runtime_equivalents"] == [
        "causal_continuation",
        "chat_template_generation",
        "temperature_min_p_repetition_sampling",
        "raw_prompt_generation",
        "tool_token_generation_from_prerendered_prompt",
    ]
    assert "JSON_schema_and_Python_function_tool_serialization" in usage["hf_preprocessing_only"]
    assert usage["external_wrappers_not_linked_by_trtmc"] == [
        "transformers_pipeline",
        "device_map_auto",
        "flash_attention_2_selector",
        "vllm_server_and_SamplingParams",
        "sglang_server",
        "OpenAI_compatible_HTTP_API",
        "Docker_Model_Runner",
        "llama_cpp_GGUF",
        "Colab_TRL_Unsloth_DPO",
    ]
    [case] = manifest["testcases"]
    assert case["ci_tier"] == "nightly_only"
    assert case["reference_family"] == "lfm2_model_card_sampling"
    assert case["user_contract"] == "sampling"
    assert case["max_new_tokens"] == 512
    assert case["temperature"] == 0.3
    assert case["min_p"] == 0.15
    assert case["top_k"] == 50
    assert case["seed"] == 42
    assert case["inputs"] == {
        "do_sample": True,
        "repetition_penalty": 1.05,
    }
    assert case["contract_config"]["use_chat_template"] is True


def test_followup_dense_variants_are_nightly_and_cover_both_layer_schemas() -> None:
    expected = {
        "lfm2-700m.json": (
            "LiquidAI/LFM2-700M",
            "86f49fc9a3800c3a325b7320bde179c318062583",
            16,
            1536,
            6912,
            [2, 5, 8, 10, 12, 14],
        ),
        "lfm2-1.2b.json": (
            "LiquidAI/LFM2-1.2B",
            "40f3da0d0164913923aee9462c23077868b816a3",
            16,
            2048,
            8192,
            [2, 5, 8, 10, 12, 14],
        ),
        "lfm2-2.6b.json": (
            "LiquidAI/LFM2-2.6B",
            "36ed799f7024ef169ba92ffe184821835e15cc66",
            30,
            2048,
            10752,
            [2, 5, 9, 13, 17, 21, 24, 27],
        ),
    }
    for filename, contract in expected.items():
        manifest = _read_manifest(filename)
        [case] = manifest["testcases"]
        hf_id, revision, layers, hidden, intermediate, attention_layers = contract
        assert manifest["hf_id"] == hf_id
        assert manifest["hf_revision"] == revision
        assert manifest["trust_remote_code"] is False
        assert manifest["runtime_strategy"] == _STRATEGY
        assert case["ci_tier"] == "nightly_only"
        assert case["architecture_contract"] == {
            "layers": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_layers": attention_layers,
        }


def test_v1_manifests_exclude_moe_vl_and_external_runtime_dependencies() -> None:
    manifests = [_read_manifest(path.name) for path in sorted((_ROOT / "manifests").glob("*.json"))]
    model_ids = {manifest["hf_id"] for manifest in manifests}
    assert all("8B-A1B" not in model_id for model_id in model_ids)
    assert all("-VL" not in model_id for model_id in model_ids)

    local_runtime_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (_ROOT / "e2e_plugins").rglob("*.py")
    ).lower()
    for external_runtime in ("import vllm", "import sglang", "llama.cpp", "trtexec", "onnxruntime"):
        assert external_runtime not in local_runtime_sources


def test_reference_and_runner_preserve_model_card_parameters() -> None:
    reference = (_ROOT / "e2e_plugins" / "references" / "hf_transformers.py").read_text(
        encoding="utf-8"
    )
    runner = (_ROOT / "e2e_plugins" / "runners" / "text_generation.py").read_text(encoding="utf-8")

    assert "return_dict=True" in reference
    assert 'inputs["input_ids"]' in reference
    assert '"torch_dtype":' in reference
    assert 'model_kwargs["revision"] = revision' in reference
    for parameter in (
        "do_sample",
        "temperature",
        "min_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
    ):
        assert parameter in reference

    assert "from tensorrt_model_connect.families.lfm2.debug_runner import" in runner
    assert "family_runner_from_bundle(" in runner
    for flag in (
        "--chat-template",
        "--temperature",
        "--min-p",
        "--top-k",
        "--seed",
        "--repetition-penalty",
    ):
        assert flag in runner


def test_every_case_has_a_family_local_threshold_sidecar() -> None:
    for manifest_path in sorted((_ROOT / "manifests").glob("*.json")):
        loaded = load_model_manifest(manifest_path)
        manifest = _read_manifest(manifest_path.name)
        assert {case.name for case in loaded.testcases} == {
            case["name"] for case in manifest["testcases"]
        }
        for case in manifest["testcases"]:
            threshold = _ROOT / "thresholds" / f"{case['name']}.json"
            assert threshold.is_file(), threshold

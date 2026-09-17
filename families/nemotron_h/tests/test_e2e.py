# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import gc
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "text_generation", path
        assert isinstance(manifest["precision"], str), path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert isinstance(manifest["tensor_parallel_size"], int), path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> set[str]:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        selected |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selection(config)
    enabled = os.environ.get("TRTMC_E2E") == "1"
    if not enabled and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _required_environment(tp_size: int):
    binary_value = os.environ.get("TRTMC_BINARY")
    runtime_value = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert binary_value, "selected E2E requires TRTMC_BINARY"
    assert runtime_value, "selected E2E requires TRTMC_RUNTIME_ROOT"

    binary = Path(binary_value)
    runtime_root = Path(runtime_value)
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert runtime_root.is_dir(), runtime_root
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root

    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    assert torch.cuda.device_count() >= tp_size, (
        f"{_FAMILY} TP{tp_size} requires {tp_size} visible GPUs; found {torch.cuda.device_count()}"
    )
    if tp_size > 1:
        assert shutil.which("mpirun"), "selected TP E2E requires mpirun"
    return binary, runtime_root, torch


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
        )
    )
    assert (path / "config.json").is_file(), path
    return path


def _prompt(case: dict) -> str:
    if "prompt" in case:
        prompt = case["prompt"]
        assert isinstance(prompt, str) and prompt, case["name"]
        return prompt
    repeated = case["prompt_repeat"]
    count = int(repeated["count"])
    assert count > 0
    return str(repeated["separator"]).join([str(repeated["text"])] * count) + str(
        repeated.get("suffix", "")
    )


def _thresholds(case_name: str) -> dict[str, float]:
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload["threshold_overrides"]
    assert isinstance(thresholds, dict) and thresholds, path
    return thresholds


def _build_bundle(manifest: dict, model_dir: Path, bundle: Path, *, execution=None) -> None:
    quantization = manifest.get("quantization")
    assert quantization is None or isinstance(quantization, str)
    fp32_layers = tuple(manifest.get("fp32_layers", ()))
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task="text_generation",
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            quantization=quantization,
            fp32_layers=fp32_layers,
        ),
        execution=execution,
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _assert_rank_sections(binary: Path, bundle: Path, tp_size: int) -> None:
    inspected = subprocess.run(
        [str(binary), "inspect", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    record_evidence("commands", {"argv": getattr(inspected, "args", None)})
    record_evidence("native", {"stdout": getattr(inspected, "stdout", None), "stderr": getattr(inspected, "stderr", None)})
    payload = json.loads(inspected.stdout)
    assert payload["family"] == _FAMILY
    assert payload["task"] == "text_generation"
    if tp_size > 1:
        rank_sections = {
            name
            for name in payload["sections"]
            if name.startswith("engine.rank") and name.endswith(".plan")
        }
        assert rank_sections == {f"engine.rank{rank}.plan" for rank in range(tp_size)}


def _native_arguments(bundle: Path, runtime_root: Path, prompt: str, case: dict) -> list[str]:
    arguments = [
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(case["max_new_tokens"]),
    ]
    options = {
        "temperature": "--temperature",
        "top_k": "--top-k",
        "top_p": "--top-p",
        "min_p": "--min-p",
        "seed": "--seed",
        "repetition_penalty": "--repetition-penalty",
        "use_chat_template": "--use-chat-template",
        "enable_thinking": "--enable-thinking",
    }
    for field, option in options.items():
        if field not in case:
            continue
        value = case[field]
        if isinstance(value, bool):
            value = "true" if value else "false"
        arguments.extend([option, str(value)])
    return arguments


def _run_native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    prompt: str,
    case: dict,
    tp_size: int,
    tmp_path: Path,
) -> dict:
    command = [str(binary), *_native_arguments(bundle, runtime_root, prompt, case)]
    environment = dict(os.environ)
    if tp_size > 1:
        environment["TRTMC_NCCL_RENDEZVOUS"] = str(tmp_path / "nccl.rendezvous")
        prefix = ["mpirun", "--tag-output", "-np", str(tp_size)]
        for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "TRTMC_NCCL_RENDEZVOUS"):
            if name in environment:
                prefix.extend(["-x", name])
        command = [*prefix, *command]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
    if tp_size == 1:
        return json.loads(completed.stdout)

    rank_zero_payloads = []
    for line in completed.stdout.splitlines():
        match = _MPI_RANK_ZERO.fullmatch(line)
        if match and match.group(1).lstrip().startswith("{"):
            rank_zero_payloads.append(match.group(1))
    assert len(rank_zero_payloads) == 1, completed.stdout
    return json.loads(rank_zero_payloads[0])


def _render_prompt(tokenizer, prompt: str, case: dict):
    if not case.get("use_chat_template", False):
        return tokenizer(prompt, return_tensors="pt")
    options = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in case:
        options["enable_thinking"] = case["enable_thinking"]
    messages = []
    template = getattr(tokenizer, "chat_template", "") or ""
    modern_chatml = (
        isinstance(template, str)
        and "<|im_start|>system" in template
        and "<think></think>" in template
    )
    # The modern source template honors enable_thinking itself, without /no_think.
    if case.get("enable_thinking") is False and not modern_chatml:
        messages.append({"role": "system", "content": "/no_think"})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        **options,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def _is_sampling(case: dict) -> bool:
    return bool(
        case.get("do_sample", False)
        or float(case.get("temperature", 1.0)) not in {0.0, 1.0}
        or int(case.get("top_k", 1)) > 1
        or float(case.get("top_p", 1.0)) < 1.0
        or float(case.get("min_p", 0.0)) > 0.0
    )


def _apply_repetition_penalty(logits, token_ids: list[int], penalty: float):
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for token_id in set(token_ids):
        logits[token_id] = (
            logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
        )
    return logits


def _allowed_tokens(torch, logits, case: dict, history: list[int]):
    penalty = float(case.get("repetition_penalty", 1.0))
    logits = _apply_repetition_penalty(logits.float(), history, penalty)
    temperature = float(case.get("temperature", 1.0))
    if temperature > 0.0:
        logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    allowed = torch.ones_like(probabilities, dtype=torch.bool)

    top_k = int(case.get("top_k", 0))
    if top_k > 0 and top_k < probabilities.numel():
        top_indices = torch.topk(probabilities, top_k).indices
        top_mask = torch.zeros_like(allowed)
        top_mask[top_indices] = True
        allowed &= top_mask

    top_p = float(case.get("top_p", 1.0))
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        keep = torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities < top_p
        top_p_mask = torch.zeros_like(allowed)
        top_p_mask[sorted_indices[keep]] = True
        allowed &= top_p_mask

    min_p = float(case.get("min_p", 0.0))
    if min_p > 0.0:
        allowed &= probabilities >= probabilities.max() * min_p
    return allowed


def _record_text_diagnostics(native_ids: list[int], reference_ids: list[int]) -> None:
    common = min(len(native_ids), len(reference_ids))
    prefix = next((index for index in range(common) if native_ids[index] != reference_ids[index]), common)
    identical = prefix == len(native_ids) == len(reference_ids)
    record_evidence("diagnostics", {
        "matching_prefix_tokens": prefix,
        "native_token_count": len(native_ids),
        "reference_token_count": len(reference_ids),
        "first_difference": None if identical else {
            "index": prefix,
            "native_token_id": native_ids[prefix] if prefix < len(native_ids) else None,
            "reference_token_id": reference_ids[prefix] if prefix < len(reference_ids) else None,
        },
    })


def _hf_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    prompt: str,
    actual_ids: list[int],
    torch,
) -> tuple[list[int], str, float | None, str]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    reference_precision = case.get(
        "reference_precision",
        manifest.get("reference_precision", manifest["precision"]),
    )
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    assert reference_precision in dtypes, reference_precision
    reference_options = {}
    if "reference_use_mamba_kernels" in case:
        value = case["reference_use_mamba_kernels"]
        assert isinstance(value, bool), "reference_use_mamba_kernels must be boolean"
        reference_options["use_mamba_kernels"] = value
    if case.get("reference_decode_modelopt_nvfp4", False):
        # The declared BF16 oracle needs decoded weights, not packed NVFP4 bytes.
        # This does not emulate compiled activation or KV rounding.
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
        from modelopt.torch.export.quant_utils import QUANTIZATION_FP8, from_quantized_weight
        from safetensors.torch import load_file

        assert not trust_remote_code and reference_precision == "bf16"
        quant = json.loads((model_dir / "hf_quant_config.json").read_text())["quantization"]
        assert quant["quant_algo"] in {"NVFP4", "MIXED_PRECISION"}
        mixed = quant["quant_algo"] == "MIXED_PRECISION"
        layers = quant.get("quantized_layers", {})
        if mixed:
            assert layers and all(
                policy["quant_algo"] == "FP8"
                or (policy["quant_algo"] == "W4A16_NVFP4" and policy["group_size"] == 16)
                for policy in layers.values()
            )
        else:
            assert quant["group_size"] == 16
        state = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            tensors = load_file(str(shard), device="cpu")
            assert not state.keys() & tensors.keys(), "duplicate checkpoint tensors"
            state.update(tensors)
        packed = {key for key, value in state.items() if value.dtype == torch.uint8}
        assert packed, "NVFP4 reference requires actual packed weights"
        fp8 = {
            key for key, value in state.items()
            if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn
        }
        quantized = packed | fp8
        if mixed:
            assert quantized == {key + ".weight" for key in layers}
            assert all(layers[key.removesuffix(".weight")]["quant_algo"] == "FP8" for key in fp8)
            assert all(
                layers[key.removesuffix(".weight")]["quant_algo"] == "W4A16_NVFP4"
                for key in packed
            )
        else:
            assert not fp8
        for key in packed:
            assert key.endswith(".weight"), key
            weight = state[key]
            assert weight.ndim == 2 and weight.shape[-1] % 8 == 0, key
            prefix = key.removesuffix("weight")
            scale = state[prefix + "weight_scale"]
            double_scale = state[prefix + "weight_scale_2"]
            shape = (weight.shape[0], weight.shape[1] * 2)
            assert scale.dtype == torch.float8_e4m3fn
            assert scale.shape == (shape[0], shape[1] // 16), key
            assert torch.isfinite(scale.float()).all() and (scale.float() >= 0).all()
            assert double_scale.numel() == 1 and torch.isfinite(double_scale).all()
            assert (double_scale > 0).all()
            state[key] = NVFP4QTensor(shape, torch.bfloat16, weight).dequantize(
                dtype=torch.bfloat16, scale=scale, double_scale=double_scale,
                block_sizes={-1: 16}, fast=False,
            )
            assert state[key].shape == shape and torch.isfinite(state[key]).all(), key
        for key in fp8:
            weight = state[key]
            scale = state[key.removesuffix("weight") + "weight_scale"]
            assert weight.ndim == 2 and scale.numel() == 1, key
            assert torch.isfinite(scale).all() and (scale > 0).all(), key
            state[key] = from_quantized_weight(
                weight, scale, QUANTIZATION_FP8, torch.bfloat16,
            )
            assert state[key].shape == weight.shape and torch.isfinite(state[key]).all(), key
        for key in list(state):
            if key.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                assert key.rsplit(".", 1)[0] + ".weight" in quantized, key
                scale = state.pop(key)
                assert torch.isfinite(scale.float()).all() and (scale.float() >= 0).all(), key
        # ModelOpt exports FP8 KV calibration buffers as k_scale/v_scale.
        # The floating BF16 oracle does not use quantized KV; these are not weights.
        kv_scales = {
            key for key in state if key.endswith((".k_proj.k_scale", ".v_proj.v_scale"))
        }
        if kv_scales:
            assert quant["kv_cache_quant_algo"] == "FP8"
            k_prefixes = {
                key.removesuffix(".k_proj.k_scale")
                for key in kv_scales if key.endswith(".k_proj.k_scale")
            }
            v_prefixes = {
                key.removesuffix(".v_proj.v_scale")
                for key in kv_scales if key.endswith(".v_proj.v_scale")
            }
            assert k_prefixes == v_prefixes, "unpaired FP8 KV calibration"
            for key in kv_scales:
                scale = state[key]
                weight = state[key.rsplit(".", 1)[0] + ".weight"]
                assert weight.is_floating_point() and weight.ndim == 2, key
                assert scale.dtype == torch.float32 and scale.numel() == 1, key
                assert torch.isfinite(scale).all() and (scale > 0).all(), key
                del state[key]
        config = AutoConfig.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, **reference_options,
        )
        embedded = getattr(config, "quantization_config", None)
        if mixed:
            assert embedded["quant_method"] == "modelopt"
            assert embedded["quant_algo"] == quant["quant_algo"]
            assert embedded["quantized_layers"] == layers
            # All declared packed/FP8 matrices are now independently decoded.
            # Do not ask HF to quantize the already decoded BF16 state again.
            del config.quantization_config
        else:
            assert not embedded
        # Keep Transformers' official checkpoint-name conversion (backbone -> model).
        from transformers import NemotronHForCausalLM

        assert isinstance(config, NemotronHForCausalLM.config_class)
        model, loading = NemotronHForCausalLM.from_pretrained(
            None, config=config, state_dict=state, torch_dtype=torch.bfloat16,
            attn_implementation="eager", output_loading_info=True,
        )
        assert all(not loading.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
        )), loading
        if (model_dir / "generation_config.json").is_file():
            model.generation_config = GenerationConfig.from_pretrained(
                model_dir, local_files_only=True,
            )
        del state, tensors
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtypes[reference_precision],
            attn_implementation="eager",
            **reference_options,
        )
    reference_device = case.get("reference_device", "cuda")
    assert reference_device in {"cpu", "cuda"}, reference_device
    model = model.eval().to(reference_device)
    inputs = _render_prompt(tokenizer, prompt, case).to(model.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    if "expected_prompt_token_ids" in case:
        assert prompt_ids == case["expected_prompt_token_ids"]

    sampling_support = None
    reference_ids: list[int] = []
    reference_text = ""
    with torch.inference_mode():
        if _is_sampling(case):
            output = model(**inputs, use_cache=True)
            past = output.past_key_values
            history = list(prompt_ids)
            attention_mask = inputs.get("attention_mask")
            accepted = 0
            for token_id in actual_ids:
                allowed = _allowed_tokens(torch, output.logits[0, -1], case, history)
                accepted += int(0 <= token_id < allowed.numel() and allowed[token_id].item())
                history.append(token_id)
                next_id = torch.tensor([[token_id]], dtype=torch.long, device=model.device)
                next_inputs = {
                    "input_ids": next_id,
                    "past_key_values": past,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device=model.device,
                            ),
                        ],
                        dim=1,
                    )
                    next_inputs["attention_mask"] = attention_mask
                output = model(**next_inputs)
                past = output.past_key_values
            sampling_support = accepted / len(actual_ids)
        else:
            generate_options = {
                "max_new_tokens": int(case["max_new_tokens"]),
                "do_sample": False,
                "use_cache": False,
            }
            if "repetition_penalty" in case:
                generate_options["repetition_penalty"] = float(case["repetition_penalty"])
            generated = model.generate(**inputs, **generate_options)
            reference_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
            reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()

    actual_decoded = tokenizer.decode(actual_ids, skip_special_tokens=True).strip()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reference_ids, reference_text, sampling_support, actual_decoded


def _normalized_edit_distance(left: str, right: str) -> float:
    left = " ".join(left.casefold().split())
    right = " ".join(right.casefold().split())
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _text_threshold(thresholds: dict[str, float]) -> float:
    _evidence_threshold = float(thresholds.get("contract_ned_threshold", 0.15))
    record_evidence("thresholds", {**thresholds, "contract_ned_threshold": _evidence_threshold})
    return _evidence_threshold


def _reference_backend(case: dict) -> str:
    backend = case.get("reference_backend")
    assert backend in {"golden_snapshot", "hf_transformers"}, (
        f"unsupported {_FAMILY} reference backend: {backend!r}"
    )
    return backend


def _golden_reference(case: dict) -> str:
    metadata = case.get("metadata")
    assert isinstance(metadata, dict), f"{case['name']} golden reference requires metadata"
    value = metadata.get("golden_snapshot_path")
    assert isinstance(value, str) and value, (
        f"{case['name']} golden reference requires metadata.golden_snapshot_path"
    )
    path = Path(value)
    if not path.is_absolute():
        path = _TEST_DIR / path
    assert path.is_file(), f"{case['name']} golden reference does not exist: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{case['name']} golden reference must be an object: {path}"
    text = payload.get("text")
    assert isinstance(text, str) and text.strip(), (
        f"{case['name']} golden reference requires non-empty text: {path}"
    )
    return text.strip()


def _assert_golden_correctness(
    payload: dict,
    case: dict,
    thresholds: dict[str, float],
    reference_text: str,
) -> None:
    actual_ids = payload["token_ids"]
    assert isinstance(actual_ids, list) and actual_ids
    assert all(isinstance(token_id, int) for token_id in actual_ids)
    assert len(actual_ids) <= int(case["max_new_tokens"]), (
        "Task API token_ids must contain generated tokens only"
    )
    actual_text = str(payload["text"]).strip()
    assert actual_text
    if case.get("enable_thinking") is False:
        assert "<think>" not in actual_text.casefold()
    assert _normalized_edit_distance(actual_text, reference_text) <= _text_threshold(thresholds)


def _assert_correctness(
    payload: dict,
    case: dict,
    thresholds: dict[str, float],
    reference_ids: list[int],
    reference_text: str,
    sampling_support: float | None,
    actual_decoded: str,
) -> None:
    actual_ids = payload["token_ids"]
    assert isinstance(actual_ids, list) and actual_ids
    assert all(isinstance(token_id, int) for token_id in actual_ids)
    assert len(actual_ids) <= int(case["max_new_tokens"]), (
        "Task API token_ids must contain generated tokens only"
    )
    actual_text = str(payload["text"]).strip()
    if case.get("enable_thinking") is False:
        assert "<think>" not in actual_text.casefold()
    if "expected_continuation_token_ids" in case:
        assert actual_ids == case["expected_continuation_token_ids"]
    if "expected_continuation_text" in case:
        assert case["expected_continuation_text"].casefold() in actual_decoded.casefold()
    expected_answers = case.get("expected_answers", ())
    if expected_answers:
        assert any(answer.casefold() in actual_decoded.casefold() for answer in expected_answers)

    del sampling_support
    assert actual_text
    assert _normalized_edit_distance(actual_text, reference_text) <= _text_threshold(thresholds)


def test_reference_routes_keep_tp4_on_golden_and_single_gpu_on_hf() -> None:
    expected_backends = {
        "nemotron-h-nano-9b": "hf_transformers",
        "nemotron-h-nano-9b-tp4": "golden_snapshot",
    }
    assert {name: case["reference_backend"] for name, (_, case) in _CASES.items()} == (
        expected_backends
    )
    source = Path(__file__).read_text(encoding="utf-8")
    for required in (
        '{"role": "system", "content": "/no_think"}',
        "torch_dtype=dtypes[reference_precision]",
        'attn_implementation="eager"',
        '"use_cache": False',
    ):
        assert required in source
    for template, modern in (
        ("<SPECIAL_10>", False),
        ("<|im_start|>", False),
        ("<|im_start|>system <think></think>", True),
    ):
        tokenizer = Mock(chat_template=template)
        tokenizer.apply_chat_template.return_value = "rendered"
        _render_prompt(
            tokenizer, "prompt", {"use_chat_template": True, "enable_thinking": False}
        )
        messages = [{"role": "user", "content": "prompt"}]
        if not modern:
            messages.insert(0, {"role": "system", "content": "/no_think"})
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokenizer.assert_called_once_with(
            "rendered", return_tensors="pt", add_special_tokens=False
        )
    _, case = _CASES["nemotron-h-nano-9b-tp4"]
    assert _reference_backend(case) == "golden_snapshot"
    assert _golden_reference(case) == "Paris"
    _assert_golden_correctness(
        {"token_ids": [1], "text": "Paris"},
        case,
        _thresholds(case["name"]),
        "Paris",
    )
    with pytest.raises(AssertionError, match="unsupported"):
        _reference_backend({"name": "missing-backend"})


def test_golden_reference_fails_closed_for_missing_or_invalid_data(tmp_path: Path) -> None:
    case = {
        "name": "broken-golden",
        "metadata": {"golden_snapshot_path": str(tmp_path / "missing.json")},
    }
    with pytest.raises(AssertionError, match="does not exist"):
        _golden_reference(case)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not JSON", encoding="utf-8")
    case["metadata"]["golden_snapshot_path"] = str(invalid)
    with pytest.raises(json.JSONDecodeError):
        _golden_reference(case)

    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="requires non-empty text"):
        _golden_reference(case)


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": _CASES[case_name][-1]})
    tp_size = manifest["tensor_parallel_size"]
    binary, runtime_root, torch = _required_environment(tp_size)
    model_dir = _checkpoint(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
    prompt = _prompt(case)
    record_evidence("inputs", {"prompt": prompt})
    bundle = tmp_path / manifest["bundle"]

    with evidence_stage("build"):
        _build_bundle(manifest, model_dir, bundle)
    with evidence_stage("compare"):
        _assert_rank_sections(binary, bundle, tp_size)
    with evidence_stage("native"):
        payload = _run_native(
            binary,
            runtime_root,
            bundle,
            prompt,
            case,
            tp_size,
            tmp_path,
        )
    record_evidence("native", payload)
    reruns = int(case.get("determinism_reruns", 0))
    for _ in range(reruns):
        with evidence_stage("native"):
            repeated = _run_native(
                binary,
                runtime_root,
                bundle,
                prompt,
                case,
                tp_size,
                tmp_path,
            )
        record_evidence("native", repeated)
        with evidence_stage("compare"):
            assert repeated["token_ids"] == payload["token_ids"]
        with evidence_stage("compare"):
            assert repeated["text"] == payload["text"]

    backend = _reference_backend(case)
    if backend == "golden_snapshot":
        with evidence_stage("compare"):
            _assert_golden_correctness(
                payload,
                case,
                record_evidence("thresholds", _thresholds(case_name)),
                record_evidence("reference", _golden_reference(case)),
            )
        return

    with evidence_stage("reference"):
        reference = _hf_reference(
            model_dir,
            manifest,
            case,
            prompt,
            payload["token_ids"],
            torch,
        )
    record_evidence("reference", {"reference_ids": reference[0], "reference_text": reference[1], "sampling_support": reference[2], "actual_decoded": reference[3]})
    if not _is_sampling(case):
        _record_text_diagnostics(payload["token_ids"], reference[0])
    with evidence_stage("compare"):
        _assert_correctness(payload, case, record_evidence("thresholds", _thresholds(case_name)), *reference)

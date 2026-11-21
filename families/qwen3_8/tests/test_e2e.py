# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import gc
import json
import os
import subprocess
from pathlib import Path

import pytest

from families.qwen3_8.cli import build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] in {"text_generation", "vision_language_generation"}, path
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
    return binary, runtime_root, torch


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import constants, snapshot_download

    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=constants.HF_HUB_OFFLINE,
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
        model=str(model_dir),
        output=bundle,
        task=manifest.get("task", "text_generation"),
        precision=manifest["precision"],
        max_sequence_length=manifest["max_sequence_length"],
        tensor_parallel_size=manifest["tensor_parallel_size"],
        quantization=quantization,
        fp32_layers=fp32_layers,
        execution_variant=execution.variant if execution is not None else None,
        companion=tuple(
            f"{checkpoint.role}={checkpoint.model_dir}" for checkpoint in execution.checkpoints
        ) if execution is not None else (),
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _assert_rank_sections(binary: Path, bundle: Path, tp_size: int,
                          task: str = "text_generation") -> bool:
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
    edge = "edge_llm.json" in payload["sections"]
    media = edge or "native_visual.json" in payload["sections"]
    expected_task = "images_text_to_text" if media and task == "vision_language_generation" else task
    # Previously published single-image bundles retain the legacy task ID.
    assert payload["task"] in {task, expected_task}
    if edge:
        if tp_size == 2:
            for rank in range(tp_size):
                assert f"edge_llm/engine/llm_world2_rank{rank}.engine" in payload["sections"]
        if task == "vision_language_generation":
            assert "edge_llm/engine/visual/visual.engine" in payload["sections"]
    elif tp_size > 1:
        rank_sections = {
            name
            for name in payload["sections"]
            if name.startswith("engine.rank") and name.endswith(".plan")
        }
        assert rank_sections == {f"engine.rank{rank}.plan" for rank in range(tp_size)}
    if not edge and "native_visual.json" in payload["sections"]:
        expected_visual = ({f"native_visual.rank{rank}.plan" for rank in range(tp_size)}
                           if tp_size > 1 else {"native_visual.plan"})
        assert expected_visual <= payload["sections"].keys()
    return edge


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
    if "video_frames" in case:
        raise ValueError("Native video arguments are not mapped by this E2E harness")
    if "image" in case and "images" in case:
        raise ValueError("Specify image or ordered images, not both")
    if "image" in case:
        arguments.extend(["--image", str(case["image"])])
    if "images" in case:
        if not case["images"]:
            raise ValueError("Ordered images cannot be empty")
        arguments.extend(["--images", json.dumps(list(map(str, case["images"])))])
    return arguments


def _run_native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    prompt: str,
    case: dict,
    tp_size: int,
    tmp_path: Path,
    *, edge: bool = False,
) -> dict:
    command = [str(binary), *_native_arguments(bundle, runtime_root, prompt, case)]
    environment = dict(os.environ)
    # Both Edge and the family-native TP2 runtime own their rank threads and
    # NCCL communicators in one process. An MPI wrapper would load duplicate
    # two-GPU models and is not the family runtime contract.

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
    (tmp_path / "native.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (tmp_path / "native.stderr.log").write_text(completed.stderr, encoding="utf-8")
    completed.check_returncode()
    return json.loads(completed.stdout)


def _render_prompt(tokenizer, prompt: str, case: dict):
    if not case.get("use_chat_template", False):
        return tokenizer(prompt, return_tensors="pt")
    options = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in case:
        options["enable_thinking"] = case["enable_thinking"]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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


def _reference_inputs(model_dir: Path, manifest: dict, tokenizer, prompt: str, case: dict):
    """Validate every supplied modality before loading large source weights."""
    has_images = "image" in case or "images" in case
    has_video = "video_frames" in case
    if not has_images and not has_video:
        return _render_prompt(tokenizer, prompt, case)

    import math
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    assert case.get("use_chat_template", True), "Media requires the source chat template"
    if "image" in case and "images" in case:
        raise ValueError("Specify image or ordered images, not both")
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=bool(manifest.get("trust_remote_code", False)),
    )
    content, images, arguments = [], [], {}
    try:
        if has_images:
            factor = processor.image_processor.patch_size * processor.image_processor.merge_size
            processor.image_processor.size = {
                "shortest_edge": 128 * factor * factor,
                "longest_edge": 1280 * factor * factor,
            }
            image_paths = case["images"] if "images" in case else [case["image"]]
            if not image_paths:
                raise ValueError("Ordered images cannot be empty")
            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                content.append({"type": "image"})
            arguments["images"] = images
        if has_video:
            paths = case["video_frames"]
            fps = float(case["video_fps"])
            assert paths and math.isfinite(fps) and fps > 0
            video_processor = processor.video_processor
            factor = video_processor.patch_size * video_processor.merge_size
            temporal = video_processor.temporal_patch_size
            # Preserve the existing per-media temporal pixel budget.
            video_processor.size = {
                "shortest_edge": 128 * temporal * factor * factor,
                "longest_edge": 1280 * temporal * factor * factor,
            }
            frames = []
            for path in paths:
                with Image.open(path) as image:
                    frames.append(np.asarray(image.convert("RGB")))
            frames = np.stack(frames)
            arguments.update(
                videos=[frames],
                video_metadata=[VideoMetadata(
                    total_num_frames=len(frames), fps=fps,
                    height=frames.shape[1], width=frames.shape[2],
                    duration=len(frames) / fps, frames_indices=list(range(len(frames))),
                )],
                do_sample_frames=False,
            )
            content.append({"type": "video"})
        content.append({"type": "text", "text": prompt})
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            enable_thinking=case.get("enable_thinking", False),
        )
        inputs = processor(text=[rendered], return_tensors="pt", **arguments)
    finally:
        for image in images:
            image.close()
    if has_images:
        record_evidence("inputs", {"images": list(map(str, image_paths)),
                                  "image_grid_thw": inputs["image_grid_thw"].tolist()})
    if has_video:
        record_evidence("inputs", {
            "video_frames": list(map(str, paths)), "video_fps": fps,
            "video_grid_thw": inputs["video_grid_thw"].tolist(),
        })
    return inputs


def _load_reference_model(model_dir: Path, manifest: dict, case: dict, torch):
    """Load the declared source oracle once; callers may reuse it across requests."""
    from transformers import AutoModelForCausalLM

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
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
    if case.get("reference_decode_modelopt_mixed", False):
        # Preserve the declared FP32 oracle; packed bytes are not floating weights.
        # This does not emulate compiled activation or KV quantization.
        from modelopt.torch.export.quant_utils import QUANTIZATION_FP8, from_quantized_weight
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
        from safetensors.torch import load_file
        from transformers import AutoConfig, GenerationConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

        assert not trust_remote_code and reference_precision == "fp32"
        raw = json.loads((model_dir / "config.json").read_text())
        quant = json.loads((model_dir / "hf_quant_config.json").read_text())["quantization"]
        embedded = raw["quantization_config"]
        assert embedded["quant_method"] == "modelopt"
        assert embedded["quant_algo"] == quant["quant_algo"] == "MIXED_PRECISION"
        layers = quant["quantized_layers"]
        assert layers and embedded["quantized_layers"] == layers
        assert all(
            policy["quant_algo"] == "FP8"
            or (policy["quant_algo"] == "NVFP4" and policy["group_size"] == 16)
            for policy in layers.values()
        )
        state = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            tensors = load_file(str(shard), device="cpu")
            assert not state.keys() & tensors.keys(), "duplicate checkpoint tensors"
            state.update(tensors)
        packed = {key for key, value in state.items() if value.dtype == torch.uint8}
        fp8 = {
            key for key, value in state.items()
            if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn
        }
        quantized = packed | fp8
        assert packed and fp8 and quantized == {key + ".weight" for key in layers}
        assert all(layers[key.removesuffix(".weight")]["quant_algo"] == "NVFP4" for key in packed)
        assert all(layers[key.removesuffix(".weight")]["quant_algo"] == "FP8" for key in fp8)
        for key in packed:
            weight = state[key]
            assert key.endswith(".weight") and weight.ndim == 2, key
            assert weight.shape[-1] % 8 == 0, key
            prefix = key.removesuffix("weight")
            scale = state[prefix + "weight_scale"]
            double_scale = state[prefix + "weight_scale_2"]
            shape = (weight.shape[0], weight.shape[1] * 2)
            assert scale.dtype == torch.float8_e4m3fn
            assert scale.shape == (shape[0], shape[1] // 16), key
            assert torch.isfinite(scale.float()).all() and (scale.float() >= 0).all(), key
            assert double_scale.numel() == 1 and torch.isfinite(double_scale).all(), key
            assert (double_scale > 0).all(), key
            state[key] = NVFP4QTensor(shape, torch.float32, weight).dequantize(
                dtype=torch.float32, scale=scale, double_scale=double_scale,
                block_sizes={-1: 16}, fast=False,
            )
            assert state[key].shape == shape and torch.isfinite(state[key]).all(), key
        for key in fp8:
            weight = state[key]
            scale = state[key.removesuffix("weight") + "weight_scale"]
            assert weight.ndim == 2 and scale.numel() == 1, key
            assert torch.isfinite(scale).all() and (scale > 0).all(), key
            state[key] = from_quantized_weight(
                weight, scale, QUANTIZATION_FP8, torch.float32,
            )
            assert state[key].shape == weight.shape and torch.isfinite(state[key]).all(), key
        for key in list(state):
            if key.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                assert key.rsplit(".", 1)[0] + ".weight" in quantized, key
                scale = state.pop(key)
                assert torch.isfinite(scale.float()).all() and (scale.float() >= 0).all(), key
        # Use the complete official source model for media. All packed tensors
        # have already been checked and dequantized above; remove only the
        # quantizer instruction, never weights or missing-key checks.
        config = AutoConfig.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
        )
        if isinstance(raw.get("vision_config"), dict):
            assert config.model_type == "qwen3_5"
            del config.quantization_config
            model_class = Qwen3_5ForConditionalGeneration
        else:
            config = config.get_text_config(decoder=True)
            assert config.model_type == "qwen3_5_text"
            model_class = Qwen3_5ForCausalLM
        assert not getattr(config, "quantization_config", None)
        model, loading = model_class.from_pretrained(
            None, config=config, state_dict=state, dtype=torch.float32,
            output_loading_info=True,
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
        raw = json.loads((model_dir / "config.json").read_text())
        model_class = AutoModelForCausalLM
        if isinstance(raw.get("vision_config"), dict):
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        model = model_class.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            dtype=dtypes[reference_precision],
            device_map=manifest.get("reference_device_map"),
        )
    reference_device = case.get("reference_device", "cuda")
    assert reference_device in {"cpu", "cuda"}, reference_device
    model = model.eval()
    if not manifest.get("reference_device_map") or case.get("reference_decode_modelopt_mixed", False):
        model = model.to(reference_device)
    return model


def _hf_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    prompt: str,
    actual_ids: list[int],
    torch,
    *,
    reference_model=None,
) -> tuple[list[int], str, float | None, str]:
    from transformers import AutoTokenizer

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    inputs = _reference_inputs(model_dir, manifest, tokenizer, prompt, case)
    model = reference_model
    if model is None:
        model = _load_reference_model(model_dir, manifest, case, torch)
    inputs = inputs.to(model.device)
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

    for keyword in case.get("expected_keywords", ()):
        assert keyword.casefold() in actual_text.casefold(), keyword

    del sampling_support
    assert actual_text
    assert _normalized_edit_distance(actual_text, reference_text) <= _text_threshold(thresholds)


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
        edge = _assert_rank_sections(binary, bundle, tp_size, manifest["task"])
    with evidence_stage("native"):
        payload = _run_native(
            binary,
            runtime_root,
            bundle,
            prompt,
            case,
            tp_size,
            tmp_path, edge=edge,
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
                tmp_path, edge=edge,
            )
        record_evidence("native", repeated)
        with evidence_stage("compare"):
            assert repeated["token_ids"] == payload["token_ids"]
        with evidence_stage("compare"):
            assert repeated["text"] == payload["text"]

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

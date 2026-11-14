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

from tensorrt_model_connect import BuildRequest, build


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


def _build_bundle(manifest: dict, model_dir: Path, bundle: Path) -> None:
    quantization = manifest.get("quantization")
    assert quantization is None or isinstance(quantization, str)
    fp32_layers = tuple(manifest.get("fp32_layers", ()))
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            quantization=quantization,
            fp32_layers=fp32_layers,
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _assert_rank_sections(binary: Path, bundle: Path, tp_size: int,
                          task: str = "text_generation", *, require_edge: bool = False) -> bool:
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
    assert edge or not require_edge, "This provisioned case requires Edge-LLM, not native fallback"
    expected_task = "images_text_to_text" if task == "vision_language_generation" else task
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
    if not edge and task == "vision_language_generation":
        visual_plans = ({"native_visual.plan"} if tp_size == 1 else
                        {f"native_visual.rank{rank}.plan" for rank in range(tp_size)})
        assert visual_plans | {"native_visual.json"} <= set(payload["sections"])
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
    if "image" in case and "images" in case:
        raise ValueError("Specify image or ordered images, not both")
    if "images" in case:
        if not case["images"]:
            raise ValueError("Ordered images cannot be empty")
        arguments.extend(["--images", json.dumps(case["images"])])
    elif "image" in case:
        arguments.extend(["--image", str(case["image"])])
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
    *, edge: bool = False,
) -> dict:
    command = [str(binary), *_native_arguments(bundle, runtime_root, prompt, case)]
    environment = dict(os.environ)
    # Native TP owns both rank tasks and communicator lifetime in one process.
    # An MPI wrapper would start two independent TP2 models, not one two-rank task.

    record_evidence("commands", {"argv": command})
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired stores partial output as bytes even with text=True.
        partial = {}
        for name in ("stdout", "stderr"):
            value = getattr(error, name) or ""
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (tmp_path / f"native.{name}.log").write_text(value, encoding="utf-8")
            partial[name] = value
        record_evidence("native", partial)
        raise
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
            arguments["images_kwargs"] = {
                "min_pixels": processor.image_processor.size["shortest_edge"],
                "max_pixels": processor.image_processor.size["longest_edge"],
            }
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
        template_options = {}
        if "enable_thinking" in case:
            template_options["enable_thinking"] = case["enable_thinking"]
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            chat_template=processor.chat_template or tokenizer.chat_template,
            **template_options,
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    # Large-model references may distribute FP32 weights without relaxing the oracle.
    device_map = manifest.get("reference_device_map")
    reference_options = {} if device_map is None else {"device_map": device_map}
    if "reference_max_memory" in manifest:
        assert device_map is not None, "reference_max_memory requires reference_device_map"
        reference_options["max_memory"] = {
            int(device) if str(device).isdigit() else device: budget
            for device, budget in manifest["reference_max_memory"].items()
        }
    inputs = _reference_inputs(model_dir, manifest, tokenizer, prompt, case)
    model = reference_model
    if model is None:
        model_class = AutoModelForCausalLM
        if "image" in case or "images" in case or "video_frames" in case:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        model = model_class.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            dtype=dtypes[reference_precision],
            **reference_options,
        ).eval()
        if device_map is None:
            model = model.to("cuda")
    else:
        assert next(model.parameters()).dtype == dtypes[reference_precision]
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

    del sampling_support
    assert actual_text
    assert _normalized_edit_distance(actual_text, reference_text) <= _text_threshold(thresholds)


def _assert_reference_free(payload: dict, case: dict) -> None:
    """Check a generation that has no reference model: shape, length and expected answer."""
    expected_answers = case.get("expected_answers")
    assert expected_answers, "a case with reference=none must set expected_answers"
    token_ids = payload["token_ids"]
    assert isinstance(token_ids, list) and token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert len(token_ids) <= int(case["max_new_tokens"])
    text = str(payload["text"]).strip()
    if case.get("enable_thinking") is False:
        assert "<think>" not in text.casefold()
    assert any(answer.casefold() in text.casefold() for answer in expected_answers), text


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
        edge = _assert_rank_sections(
            binary, bundle, tp_size, manifest["task"],
            require_edge=bool(manifest.get("require_edge_llm", False)),
        )
    with evidence_stage("native"):
        payload = _run_native(
            binary,
            runtime_root,
            bundle,
            prompt,
            case,
            tp_size,
            tmp_path,
            edge=edge,
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
                edge=edge,
            )
        record_evidence("native", repeated)
        with evidence_stage("compare"):
            assert repeated["token_ids"] == payload["token_ids"]
        with evidence_stage("compare"):
            assert repeated["text"] == payload["text"]

    if case.get("reference") == "none":
        # ModelOpt NVFP4 checkpoints cannot be run by transformers, so there is no
        # reference model; the engine output is checked against the expected answer.
        record_evidence("reference", {"mode": "none", "oracle": "expected_answers"})
        with evidence_stage("compare"):
            _assert_reference_free(payload, case)
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

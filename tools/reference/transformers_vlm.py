#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run vision-language references directly through Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trtmc.native-reference-reproduction/v1"
REPOSITORY = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = REPOSITORY / "python"
for source_root in (REPOSITORY, PYTHON_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _selected_indices(
    rows: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[int]:
    selected = [
        index
        for index, row in enumerate(rows)
        if not sample_id or str(row.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float32":
        return torch_module.float32
    return "auto"


def _entrypoint_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reproduction_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        arguments.model,
        "--prompts",
        "{work_dir}/prompts.jsonl",
        "--answers",
        "{work_dir}/answers.json",
        "--manifest",
        "{work_dir}/manifest.json",
        "--predictions",
        "{reference_predictions_json}",
        "--raw-output",
        "{reference_raw_jsonl}",
        "--dtype",
        arguments.dtype,
        "--device",
        arguments.device,
        "--sample-id",
        "{sample_id}",
    ]
    for flag, value in (
        ("--model-revision", arguments.model_revision),
        ("--reference-family", arguments.reference_family),
        ("--device-map", arguments.device_map),
        ("--attn-impl", arguments.attn_impl),
        ("--max-new-tokens", arguments.max_new_tokens),
        ("--temperature", arguments.temperature),
        ("--top-k", arguments.top_k),
        ("--top-p", arguments.top_p),
        ("--seed", arguments.seed),
    ):
        if value not in (None, ""):
            command.extend([flag, str(value)])
    for enabled, flag in (
        (arguments.trust_remote_code, "--trust-remote-code"),
        (arguments.local_files_only, "--local-files-only"),
        (arguments.do_sample, "--do-sample"),
    ):
        if enabled:
            command.append(flag)
    return command


def _write_reproduction_metadata(arguments: argparse.Namespace) -> None:
    if arguments.repro_metadata is None:
        return
    arguments.repro_metadata.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "backend": "transformers_vlm",
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": _entrypoint_sha256(),
                "command": _reproduction_command(arguments),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "native VLM reference requires torch and transformers"
        ) from exc
    return torch, transformers, AutoProcessor


def _model_classes(transformers_module: Any) -> list[Any]:
    classes = [
        getattr(transformers_module, name, None)
        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        )
    ]
    available = [model_class for model_class in classes if model_class is not None]
    if not available:
        raise RuntimeError(
            "Transformers installation does not expose a VLM-capable AutoModel class"
        )
    return available


def _load_model(
    transformers_module: Any,
    model_id: str,
    model_kwargs: Mapping[str, Any],
) -> Any:
    errors = []
    for model_class in _model_classes(transformers_module):
        try:
            return model_class.from_pretrained(
                model_id,
                **dict(model_kwargs),
            ).eval()
        except ValueError as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    raise RuntimeError(
        f"Could not load VLM reference model {model_id!r}: "
        + " | ".join(errors)
    )


def _load_images(image_paths: Sequence[str]) -> list[Any]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("native VLM reference requires Pillow") from exc
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def _has_image_placeholder(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<|image_pad|>",
            "<|vision_start|>",
            "<image>",
            "<IMG_CONTEXT>",
        )
    )


def _apply_chat_template(obj: Any, messages: Sequence[Any]) -> str:
    if not hasattr(obj, "apply_chat_template"):
        return ""
    try:
        return str(
            obj.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except ValueError as exc:
        if "chat_template" in str(exc):
            return ""
        raise


def _chat_text(
    processor: Any,
    request: Mapping[str, Any],
    fallback_prompt: str,
    fallback_template: str,
) -> str:
    messages = request.get("messages")
    messages = messages if isinstance(messages, list) else []
    rendered = _apply_chat_template(processor, messages)
    tokenizer = getattr(processor, "tokenizer", None)
    if not rendered and tokenizer is not None:
        rendered = _apply_chat_template(tokenizer, messages)
    if rendered or _has_image_placeholder(fallback_prompt):
        return rendered or fallback_prompt
    return (
        fallback_template.replace("{prompt}", fallback_prompt)
        if fallback_template
        else fallback_prompt
    )


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _generation_settings(
    arguments: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = manifest.get("generation", {})
    defaults = defaults if isinstance(defaults, dict) else {}
    return {
        "max_new_tokens": (
            arguments.max_new_tokens
            if arguments.max_new_tokens is not None
            else int(defaults.get("max_new_tokens", 8))
        ),
        "temperature": (
            arguments.temperature
            if arguments.temperature is not None
            else float(defaults.get("temperature", 1.0))
        ),
        "top_k": (
            arguments.top_k
            if arguments.top_k is not None
            else int(defaults.get("top_k", 1))
        ),
        "top_p": (
            arguments.top_p
            if arguments.top_p is not None
            else float(defaults.get("top_p", 1.0))
        ),
        "seed": (
            arguments.seed
            if arguments.seed is not None
            else int(defaults.get("seed", -1))
        ),
        "do_sample": arguments.do_sample
        or bool(defaults.get("do_sample", False)),
    }


def _is_locateanything(arguments: argparse.Namespace) -> bool:
    return (
        str(arguments.reference_family).lower() == "locateanything"
        or "locateanything" in str(arguments.model).lower()
    )


def _install_locateanything_compat() -> None:
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self: Any) -> tuple[Any, ...]:
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = to_legacy_cache
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls: Any, past_key_values: Any = None) -> Any:
            cache = cls()
            for layer_index, values in enumerate(past_key_values or ()):
                cache.update(values[0], values[1], layer_index)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    original = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original is None or getattr(original, "_trtmc_locateanything_compat", False):
        return

    def get_expanded_tied_weights_keys(self: Any, all_submodels: bool = False) -> Any:
        tied = getattr(self, "_tied_weights_keys", None)
        if not isinstance(tied, list):
            return original(self, all_submodels=all_submodels)
        if all_submodels:
            expanded = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if not isinstance(submodule, PreTrainedModel):
                    continue
                values = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                if prefix:
                    values = {
                        f"{prefix}.{key}": f"{prefix}.{value}"
                        for key, value in values.items()
                    }
                expanded.update(values)
            return expanded
        if not getattr(self.config, "tie_word_embeddings", False):
            return {}
        return (
            {"lm_head.weight": "model.embed_tokens.weight"}
            if "lm_head.weight" in tied
            else {}
        )

    get_expanded_tied_weights_keys._trtmc_locateanything_compat = True
    PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys


def _locateanything_model_ref(arguments: argparse.Namespace) -> str:
    if Path(arguments.model).is_dir():
        return str(Path(arguments.model).resolve())
    from huggingface_hub import snapshot_download
    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    return snapshot_download(
        repo_id=arguments.model,
        revision=arguments.model_revision or None,
        local_files_only=arguments.local_files_only,
        allow_patterns=hf_snapshot_allow_patterns(),
    )


def _locateanything_config(
    model_ref: str,
    arguments: argparse.Namespace,
    transformers_module: Any,
) -> Any:
    config = transformers_module.AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=True,
    )
    raw = _load_json(Path(model_ref) / "config.json")
    if hasattr(config, "text_config") and not hasattr(config.text_config, "rope_theta"):
        text_config = raw.get("text_config", {})
        rope_theta = text_config.get("rope_theta")
        for key in ("rope_parameters", "rope_scaling"):
            nested = text_config.get(key, {})
            if rope_theta is None and isinstance(nested, Mapping):
                rope_theta = nested.get("rope_theta")
        config.text_config.rope_theta = float(rope_theta or raw.get("rope_theta", 10_000.0))
    return config


def _locateanything_tokenizer(
    model_ref: str,
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
) -> Any:
    try:
        return transformers_module.AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=arguments.trust_remote_code,
            local_files_only=True,
        )
    except (KeyError, TypeError, ValueError, OSError) as auto_error:
        try:
            from tokenizers import Tokenizer

            raw_tokenizer = Tokenizer.from_file(str(Path(model_ref) / "tokenizer.json"))
        except Exception as fallback_error:
            raise RuntimeError(
                "LocateAnything tokenizer could not be loaded by AutoTokenizer or tokenizer.json"
            ) from fallback_error
        tokenizer_config_path = Path(model_ref) / "tokenizer_config.json"
        tokenizer_config = (
            _load_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
        )

        class TokenizersWrapper:
            model_max_length = int(tokenizer_config.get("model_max_length", 16_384))

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return raw_tokenizer.encode(
                    text, add_special_tokens=add_special_tokens
                ).ids

            def __call__(self, text: str, return_tensors: str | None = None) -> Any:
                ids = self.encode(text)
                if return_tensors == "pt":
                    input_ids = torch_module.tensor([ids], dtype=torch_module.long)
                    return {
                        "input_ids": input_ids,
                        "attention_mask": torch_module.ones_like(input_ids),
                    }
                return {"input_ids": ids}

            def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
                if torch_module.is_tensor(ids):
                    ids = ids.detach().cpu().tolist()
                if isinstance(ids, int):
                    ids = [ids]
                return raw_tokenizer.decode(
                    [int(token) for token in ids],
                    skip_special_tokens=skip_special_tokens,
                )

            def batch_decode(
                self, batch_ids: Any, skip_special_tokens: bool = False
            ) -> list[str]:
                return [
                    self.decode(ids, skip_special_tokens=skip_special_tokens)
                    for ids in batch_ids
                ]

        print(
            f"warning: LocateAnything AutoTokenizer failed ({auto_error}); "
            "using tokenizer.json",
            file=sys.stderr,
        )
        return TokenizersWrapper()


def _repair_locateanything_rotary_buffers(model: Any, torch_module: Any) -> None:
    repaired = 0
    model_device = next(model.parameters()).device
    for module in model.language_model.modules():
        if not all(
            hasattr(module, name) for name in ("_set_cos_sin_cache", "base", "dim")
        ):
            continue
        device = getattr(getattr(module, "inv_freq", None), "device", model_device)
        inv_freq = 1.0 / (
            float(module.base)
            ** (
                torch_module.arange(
                    0,
                    int(module.dim),
                    2,
                    device=device,
                    dtype=torch_module.float32,
                )
                / float(module.dim)
            )
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        sequence_length = int(
            getattr(
                module,
                "max_position_embeddings",
                getattr(model.config.text_config, "max_position_embeddings", 32_768),
            )
        )
        module._set_cos_sin_cache(
            seq_len=sequence_length,
            device=inv_freq.device,
            dtype=torch_module.float32,
        )
        repaired += 1
    if repaired == 0:
        raise RuntimeError("LocateAnything reference did not find rotary buffers to repair")


def _load_locateanything_runtime(
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[Any, Any, Any]:
    transformers_module.logging.set_verbosity_error()
    if torch_module.cuda.is_available():
        torch_module.backends.cudnn.enabled = False
    _install_locateanything_compat()
    model_ref = _locateanything_model_ref(arguments)
    config = _locateanything_config(model_ref, arguments, transformers_module)
    tokenizer = _locateanything_tokenizer(
        model_ref, arguments, torch_module, transformers_module
    )
    model_dtype = _model_dtype(torch_module, arguments.dtype)
    model_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": model_dtype,
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": True,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    model = transformers_module.AutoModel.from_pretrained(
        model_ref, **model_kwargs
    ).eval()
    if arguments.device_map:
        device = model.device
    else:
        device = torch_module.device(arguments.device)
        if model_dtype == "auto":
            model.to(device)
        else:
            model.to(device=device, dtype=model_dtype)
    _repair_locateanything_rotary_buffers(model, torch_module)
    return tokenizer, model, device


def _load_runtime(
    arguments: argparse.Namespace,
    torch_module: Any,
    transformers_module: Any,
    processor_class: Any,
) -> tuple[Any, Any, Any]:
    if _is_locateanything(arguments):
        return _load_locateanything_runtime(arguments, torch_module, transformers_module)
    transformers_module.logging.set_verbosity_error()
    processor = processor_class.from_pretrained(
        arguments.model,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
        **(
            {"revision": arguments.model_revision}
            if arguments.model_revision
            else {}
        ),
    )
    model_dtype = _model_dtype(torch_module, arguments.dtype)
    model_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.attn_impl:
        model_kwargs["attn_implementation"] = arguments.attn_impl
    if arguments.model_revision:
        model_kwargs["revision"] = arguments.model_revision
    model = _load_model(
        transformers_module,
        arguments.model,
        model_kwargs,
    )
    if arguments.device_map:
        device = model.device
    else:
        device = torch_module.device(arguments.device)
        if model_dtype == "auto":
            model.to(device)
        else:
            model.to(device=device, dtype=model_dtype)
    return processor, model, device


def _is_deepseek_ocr(model_id: str, model: Any) -> bool:
    return "deepseek-ocr" in model_id.lower() and hasattr(model, "infer")


def _deepseek_prompt(prompt: str) -> str:
    return prompt if "<image>" in prompt else f"<image>\n{prompt}"


def _deepseek_response(
    *,
    model: Any,
    processor: Any,
    prompt_row: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    images = [str(path) for path in prompt_row.get("images", [])]
    if len(images) != 1:
        raise ValueError("DeepSeek-OCR reference expects exactly one image")
    sample_id = str(prompt_row.get("sample_id", "vlm"))
    start = time.perf_counter()
    output_text = model.infer(
        processor,
        prompt=_deepseek_prompt(str(prompt_row.get("prompt", ""))),
        image_file=images[0],
        output_path=str(output_root / sample_id),
        save_results=False,
        eval_mode=True,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    output_text = "" if output_text is None else str(output_text)
    try:
        token_ids = [
            int(token_id)
            for token_id in processor(
                output_text,
                add_special_tokens=False,
            ).input_ids
        ]
    except Exception:
        token_ids = []
    return {
        "sample_id": sample_id,
        "output_text": output_text,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "wall_ms": wall_ms,
        "source": "hf",
    }


def _locateanything_response(
    *,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    prompt_row: Mapping[str, Any],
    source_index: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    from tensorrt_model_connect.families.locateanything.vl_debug_runner import (
        preprocess_image_inputs_for_trt,
    )

    image_paths = [str(path) for path in prompt_row.get("images", [])]
    if len(image_paths) != 1:
        raise ValueError("LocateAnything reference expects exactly one image")
    image_inputs = preprocess_image_inputs_for_trt(
        image_paths[0],
        preprocessor_type="patchify_chw",
        fixed_image_size=448,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        patch_size=14,
        interpolation="bicubic",
    )
    pixel_values = torch_module.from_numpy(image_inputs["pixel_values"]).to(device)
    image_grid_hws = torch_module.from_numpy(image_inputs["image_grid_hws"]).to(
        device=device, dtype=torch_module.int32
    )
    prompt = str(prompt_row.get("prompt", ""))
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<img>"
        + "<IMG_CONTEXT>" * 256
        + f"</img>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seed = int(settings["seed"])
    if seed >= 0:
        torch_module.manual_seed(seed + source_index)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed + source_index)
    generate_kwargs: dict[str, Any] = {
        "pixel_values": pixel_values,
        "image_grid_hws": image_grid_hws,
        "input_ids": input_ids,
        "tokenizer": tokenizer,
        "max_new_tokens": int(settings["max_new_tokens"]),
        "use_cache": True,
        "generation_mode": "slow",
        "do_sample": False,
    }
    if inputs.get("attention_mask") is not None:
        generate_kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    start = time.perf_counter()
    with torch_module.inference_mode():
        output = model.generate(**generate_kwargs)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if isinstance(output, str):
        output_text = output
        token_ids = tokenizer.encode(output_text, add_special_tokens=False)
    elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
        output_text = output[0]
        token_ids = tokenizer.encode(output_text, add_special_tokens=False)
    else:
        sequence = output[0] if output.ndim > 1 else output
        generated = sequence[input_ids.shape[-1] :]
        token_ids = [int(token) for token in generated.detach().cpu().tolist()]
        output_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {
        "sample_id": prompt_row.get("sample_id", f"vlm_{source_index:06d}"),
        "output_text": output_text,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "wall_ms": wall_ms,
        "source": "hf",
    }


def _generate_kwargs(
    processor: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer is not None and pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    kwargs = {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["do_sample"],
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "num_beams": 1,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    return kwargs


def _generated_text(processor: Any, generated: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            generated.unsqueeze(0),
            skip_special_tokens=False,
        )[0]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(generated, skip_special_tokens=False)
    return str(generated.tolist())


def _vlm_response(
    *,
    torch_module: Any,
    processor: Any,
    model: Any,
    device: Any,
    request: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    source_index: int,
    settings: Mapping[str, Any],
    fallback_template: str,
) -> dict[str, Any]:
    image_paths = [str(path) for path in prompt_row.get("images", [])]
    if len(image_paths) != 1:
        raise ValueError("VLM reference expects exactly one image")
    prompt = _chat_text(
        processor,
        request,
        str(prompt_row.get("prompt", "")),
        fallback_template,
    )
    inputs = processor(
        text=[prompt],
        images=_load_images(image_paths),
        padding=True,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, device)
    seed = int(settings["seed"])
    if seed >= 0:
        torch_module.manual_seed(seed + source_index)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed + source_index)
    start = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            **inputs,
            **_generate_kwargs(processor, settings),
        )
    wall_ms = (time.perf_counter() - start) * 1000.0
    generated = output_ids[0, int(inputs["input_ids"].shape[1]) :]
    return {
        "sample_id": prompt_row.get("sample_id", f"vlm_{source_index:06d}"),
        "output_text": _generated_text(processor, generated),
        "generated_tokens": int(generated.shape[0]),
        "generated_token_ids": [
            int(token_id) for token_id in generated.tolist()
        ],
        "wall_ms": wall_ms,
        "source": "hf",
    }


def run(arguments: argparse.Namespace) -> None:
    torch, transformers, processor_class = _runtime_dependencies()
    manifest = _load_json(arguments.manifest)
    answers = _load_json(arguments.answers)
    prompt_rows = _load_jsonl(arguments.prompts)
    requests = answers.get("requests", [])
    if not isinstance(requests, list) or len(requests) != len(prompt_rows):
        raise ValueError("answers and prepared prompts must contain the same samples")
    selected = _selected_indices(prompt_rows, arguments.sample_id)
    processor, model, device = _load_runtime(
        arguments,
        torch,
        transformers,
        processor_class,
    )
    settings = _generation_settings(arguments, manifest)
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    fallback_template = str(
        task_config.get("vlm_fallback_prompt_template", "") or ""
    )
    responses = []
    for run_index, source_index in enumerate(selected):
        prompt_row = prompt_rows[source_index]
        request = requests[source_index]
        if not isinstance(request, dict):
            raise ValueError(f"request {source_index} must be a JSON object")
        if _is_locateanything(arguments):
            response = _locateanything_response(
                torch_module=torch,
                tokenizer=processor,
                model=model,
                device=device,
                prompt_row=prompt_row,
                source_index=source_index,
                settings=settings,
            )
        elif _is_deepseek_ocr(arguments.model, model):
            response = _deepseek_response(
                model=model,
                processor=processor,
                prompt_row=prompt_row,
                output_root=arguments.predictions.parent
                / "hf_deepseek_ocr_outputs",
            )
        else:
            response = _vlm_response(
                torch_module=torch,
                processor=processor,
                model=model,
                device=device,
                request=request,
                prompt_row=prompt_row,
                source_index=source_index,
                settings=settings,
                fallback_template=fallback_template,
            )
        responses.append(response)
        print(
            f"[transformers.native_vlm] sample={run_index + 1}/{len(selected)}",
            file=sys.stderr,
        )
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for response in responses:
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
    arguments.predictions.write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VLM directly through Transformers."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--reference-family", default="")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--repro-metadata", type=Path)
    parser.add_argument("--sample-id", default="")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

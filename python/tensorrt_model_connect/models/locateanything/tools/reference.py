# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything's owner-specific Transformers reference."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }.get(name, "auto")


def _install_transformers_compat() -> None:
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: tuple(
            (layer.keys, layer.values) for layer in self.layers
        )
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

    def expanded(self: Any, all_submodels: bool = False) -> Any:
        tied = getattr(self, "_tied_weights_keys", None)
        if not isinstance(tied, list):
            return original(self, all_submodels=all_submodels)
        if all_submodels:
            result = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if not isinstance(submodule, PreTrainedModel):
                    continue
                values = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                if prefix:
                    values = {
                        f"{prefix}.{key}": f"{prefix}.{value}"
                        for key, value in values.items()
                    }
                result.update(values)
            return result
        if not getattr(self.config, "tie_word_embeddings", False):
            return {}
        return (
            {"lm_head.weight": "model.embed_tokens.weight"}
            if "lm_head.weight" in tied
            else {}
        )

    expanded._trtmc_locateanything_compat = True
    PreTrainedModel.get_expanded_tied_weights_keys = expanded


def _model_ref(arguments: Any) -> str:
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


def _tokenizer(model_ref: str, arguments: Any, torch: Any, transformers: Any) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=arguments.trust_remote_code,
            local_files_only=True,
        )
    except (KeyError, TypeError, ValueError, OSError) as auto_error:
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(str(Path(model_ref) / "tokenizer.json"))
        config_path = Path(model_ref) / "tokenizer_config.json"
        config = _load_json(config_path) if config_path.is_file() else {}

        class Wrapper:
            model_max_length = int(config.get("model_max_length", 16_384))

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return raw.encode(text, add_special_tokens=add_special_tokens).ids

            def __call__(self, text: str, return_tensors: str | None = None) -> Any:
                ids = self.encode(text)
                if return_tensors == "pt":
                    input_ids = torch.tensor([ids], dtype=torch.long)
                    return {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                    }
                return {"input_ids": ids}

            def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
                if torch.is_tensor(ids):
                    ids = ids.detach().cpu().tolist()
                return raw.decode(
                    [int(token) for token in ([ids] if isinstance(ids, int) else ids)],
                    skip_special_tokens=skip_special_tokens,
                )

        print(
            f"warning: LocateAnything AutoTokenizer failed ({auto_error}); using tokenizer.json",
            file=sys.stderr,
        )
        return Wrapper()


def _repair_rotary_buffers(model: Any, torch: Any) -> None:
    repaired = 0
    model_device = next(model.parameters()).device
    for module in model.language_model.modules():
        if not all(hasattr(module, name) for name in ("_set_cos_sin_cache", "base", "dim")):
            continue
        device = getattr(getattr(module, "inv_freq", None), "device", model_device)
        inv_freq = 1.0 / (
            float(module.base)
            ** (
                torch.arange(0, int(module.dim), 2, device=device, dtype=torch.float32)
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
            dtype=torch.float32,
        )
        repaired += 1
    if not repaired:
        raise RuntimeError("LocateAnything reference did not find rotary buffers to repair")


def _load_runtime(arguments: Any) -> tuple[Any, Any, Any, Any]:
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    _install_transformers_compat()
    model_ref = _model_ref(arguments)
    config = transformers.AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=True,
    )
    raw_config = _load_json(Path(model_ref) / "config.json")
    if hasattr(config, "text_config") and not hasattr(config.text_config, "rope_theta"):
        text_config = raw_config.get("text_config", {})
        rope_theta = text_config.get("rope_theta")
        for key in ("rope_parameters", "rope_scaling"):
            nested = text_config.get(key, {})
            if rope_theta is None and isinstance(nested, Mapping):
                rope_theta = nested.get("rope_theta")
        config.text_config.rope_theta = float(
            rope_theta or raw_config.get("rope_theta", 10_000.0)
        )
    tokenizer = _tokenizer(model_ref, arguments, torch, transformers)
    dtype = _model_dtype(torch, arguments.dtype)
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": True,
    }
    if arguments.device_map:
        kwargs["device_map"] = arguments.device_map
    model = transformers.AutoModel.from_pretrained(model_ref, **kwargs).eval()
    device = model.device if arguments.device_map else torch.device(arguments.device)
    if not arguments.device_map:
        model.to(device) if dtype == "auto" else model.to(device=device, dtype=dtype)
    _repair_rotary_buffers(model, torch)
    return torch, tokenizer, model, device


def _settings(arguments: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    defaults = manifest.get("generation", {})
    defaults = defaults if isinstance(defaults, Mapping) else {}
    return {
        "max_new_tokens": arguments.max_new_tokens or int(defaults.get("max_new_tokens", 8)),
        "seed": arguments.seed if arguments.seed is not None else int(defaults.get("seed", -1)),
    }


def _response(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    prompt_row: Mapping[str, Any],
    source_index: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    from tensorrt_model_connect.models.locateanything.vl_debug_runner import (
        preprocess_image_inputs_for_trt,
    )

    images = [str(path) for path in prompt_row.get("images", [])]
    if len(images) != 1:
        raise ValueError("LocateAnything reference expects exactly one image")
    image_inputs = preprocess_image_inputs_for_trt(
        images[0],
        preprocessor_type="patchify_chw",
        fixed_image_size=448,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        patch_size=14,
        interpolation="bicubic",
    )
    pixel_values = torch.from_numpy(image_inputs["pixel_values"]).to(device)
    image_grid_hws = torch.from_numpy(image_inputs["image_grid_hws"]).to(
        device=device, dtype=torch.int32
    )
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<img>"
        + "<IMG_CONTEXT>" * 256
        + f"</img>{prompt_row.get('prompt', '')}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seed = int(settings["seed"])
    if seed >= 0:
        torch.manual_seed(seed + source_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + source_index)
    kwargs: dict[str, Any] = {
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
        kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    wall_ms = (time.perf_counter() - started) * 1000.0
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


def run(
    arguments: Any,
    manifest: Mapping[str, Any],
    _answers: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the owner-specific upstream reference."""

    torch, tokenizer, model, device = _load_runtime(arguments)
    settings = _settings(arguments, manifest)
    selected = [
        index
        for index, row in enumerate(prompt_rows)
        if not arguments.sample_id or str(row.get("sample_id", "")) == arguments.sample_id
    ]
    if arguments.sample_id and not selected:
        raise ValueError(f"sample_id {arguments.sample_id!r} is not present in the prepared prompts")
    return [
        _response(torch, tokenizer, model, device, prompt_rows[index], index, settings)
        for index in selected
    ]

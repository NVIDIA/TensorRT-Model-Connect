# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete GPT-2 checkpoint-to-bundle build path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, _open_safetensors
from .config import ModelConfig
from .default_decoder import build_standard_decoder_engine
from .default_dual_profile_decoder_tp import build_dual_profile_tp_decoder_engine
from .parallel import ParallelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _detect_prefix(readers) -> str:
    if _has_tensor(readers, "wte.weight"):
        return ""
    if _has_tensor(readers, "transformer.wte.weight"):
        return "transformer"
    raise KeyError("Tensor not found: wte.weight")


def _key(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def load_weights(model_dir: Path, config: ModelConfig) -> WeightDict:
    """Load GPT-2 Conv1D weights without a shared checkpoint mapper."""

    readers = _open_safetensors(model_dir)
    hidden = config.hidden_size
    vocab = config.vocab_size
    root = _detect_prefix(readers)
    weights = WeightDict()

    embedding = _load_tensor(readers, _key(root, "wte.weight"))
    if tuple(embedding.shape) != (vocab, hidden):
        raise ValueError(f"GPT-2 embedding shape {tuple(embedding.shape)} != ({vocab}, {hidden})")
    weights["embedding"] = embedding.astype(np.float32)
    weights["position_embedding"] = _load_tensor(readers, _key(root, "wpe.weight")).astype(
        np.float32
    )

    mlp_size = 0
    for layer_index in range(config.num_hidden_layers):
        prefix = f"layer.{layer_index}"
        checkpoint_prefix = _key(root, f"h.{layer_index}")
        weights[f"{prefix}.input_norm"] = _load_tensor(
            readers, f"{checkpoint_prefix}.ln_1.weight"
        ).astype(np.float32)
        weights[f"{prefix}.input_norm_beta"] = _load_tensor(
            readers, f"{checkpoint_prefix}.ln_1.bias"
        ).astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = _load_tensor(
            readers, f"{checkpoint_prefix}.ln_2.weight"
        ).astype(np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = _load_tensor(
            readers, f"{checkpoint_prefix}.ln_2.bias"
        ).astype(np.float32)

        fused_qkv = _load_tensor(readers, f"{checkpoint_prefix}.attn.c_attn.weight")
        fused_qkv_bias = _load_tensor(readers, f"{checkpoint_prefix}.attn.c_attn.bias")
        expected_qkv_shape = (hidden, 3 * hidden)
        if tuple(fused_qkv.shape) != expected_qkv_shape:
            raise ValueError(
                f"GPT-2 fused QKV shape {tuple(fused_qkv.shape)} != {expected_qkv_shape}"
            )
        for offset, name in enumerate(("q", "k", "v")):
            start = offset * hidden
            stop = start + hidden
            weights[f"{prefix}.w_{name}"] = np.ascontiguousarray(
                fused_qkv[:, start:stop], dtype=np.float32
            )
            weights[f"{prefix}.{name}_bias"] = np.ascontiguousarray(
                fused_qkv_bias[start:stop], dtype=np.float32
            )

        weights[f"{prefix}.w_o"] = np.ascontiguousarray(
            _load_tensor(readers, f"{checkpoint_prefix}.attn.c_proj.weight"),
            dtype=np.float32,
        )
        weights[f"{prefix}.o_bias"] = _load_tensor(
            readers, f"{checkpoint_prefix}.attn.c_proj.bias"
        ).astype(np.float32)

        fc1 = _load_tensor(readers, f"{checkpoint_prefix}.mlp.c_fc.weight")
        if mlp_size == 0:
            mlp_size = int(fc1.shape[1])
        weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1, dtype=np.float32)
        weights[f"{prefix}.fc1_bias"] = _load_tensor(
            readers, f"{checkpoint_prefix}.mlp.c_fc.bias"
        ).astype(np.float32)
        weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(
            _load_tensor(readers, f"{checkpoint_prefix}.mlp.c_proj.weight"),
            dtype=np.float32,
        )
        weights[f"{prefix}.fc2_bias"] = _load_tensor(
            readers, f"{checkpoint_prefix}.mlp.c_proj.bias"
        ).astype(np.float32)

    weights["final_norm"] = _load_tensor(readers, _key(root, "ln_f.weight")).astype(np.float32)
    weights["final_norm_beta"] = _load_tensor(readers, _key(root, "ln_f.bias")).astype(np.float32)
    if _has_tensor(readers, "lm_head.weight"):
        output = _load_tensor(readers, "lm_head.weight")
    else:
        output = embedding
    weights["w_out"] = np.ascontiguousarray(output.T, dtype=np.float32)
    weights["_attention_size"] = hidden
    weights["_kv_attention_size"] = hidden
    weights["_mlp_size"] = mlp_size
    return weights


def _build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_sequence_length: int,
    *,
    precision: str,
    verbose: bool,
    parallel: ParallelConfig,
) -> bytes:
    if parallel.enabled:
        return build_dual_profile_tp_decoder_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="learned",
            activation="gelu_new",
            verbose=verbose,
            parallel_config=parallel,
        )
    return build_standard_decoder_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        norm_type="layernorm",
        mlp_type="gelu_fc",
        position_type="learned",
        activation="gelu_new",
        verbose=verbose,
    )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one GPT-2 bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("gpt2 does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("gpt2 does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("gpt2 does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("gpt2 does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("gpt2 does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("gpt2 supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "gpt2":
        raise ValueError(f"GPT-2 does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("GPT-2 precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError(
            "GPT-2 max_sequence_length exceeds learned position capacity: "
            f"{max_sequence_length} > {config.max_position_embeddings}"
        )
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("GPT-2 has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("GPT-2 does not expose mixed-precision layer selection")

    tp_size = _positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    parallel = ParallelConfig(tp_size=tp_size)
    parallel.validate()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = load_weights(model_dir, config)

    writer.set_header(family="gpt2", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = _build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                verbose=bool(request.verbose),
                parallel=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = _build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            verbose=bool(request.verbose),
            parallel=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = _build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            verbose=bool(request.verbose),
            parallel=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"

    runtime = _runtime_config(
        model_dir,
        config,
        precision=precision,
        max_cache_length=max_sequence_length,
        decoder_engine_layout=layout,
        tensor_parallel_size=parallel.tp_size,
        tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
    )
    writer.add_json("runtime.json", runtime)
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

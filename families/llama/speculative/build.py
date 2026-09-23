# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a standalone Llama 3.1 / EAGLE3 bundle via a family-owned command."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tensorrt_model_connect.bundle_writer import BundleWriter

from ..checkpoint_mapper import load_standard_weights
from ..config import ModelConfig
from ..model import _BUNDLE_FILES, _chat_template, _runtime_config
from .contract import EngineContract, ExecutionProfile


def load_draft(path: Path, embedding: np.ndarray):
    """Read the original EAGLE3 checkpoint without executing checkpoint code."""
    import torch
    from safetensors.torch import load_file

    config = ModelConfig.from_dir(path)
    if config.hidden_size != 4096 or config.num_hidden_layers != 1:
        raise ValueError("first scope requires the one-layer Llama 3.1 8B EAGLE3 draft")
    files = sorted(path.glob("*.safetensors"))
    if files:
        tensors = {}
        for filename in files:
            tensors.update(load_file(str(filename), device="cpu"))
    else:
        tensors = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)

    def get(key, transpose=False):
        value = tensors[key].float().numpy()
        return np.ascontiguousarray(value.T if transpose else value,
                                    dtype=np.float16 if transpose else np.float32)

    weights = {"embedding": embedding, "fc": get("fc.weight", True),
               "final_norm": get("norm.weight"), "w_out": get("lm_head.weight", True)}
    prefix = "midlayer" if "midlayer.hidden_norm.weight" in tensors else "layers.0"
    for logical, source in {
        "input_norm": "input_layernorm", "hidden_norm": "hidden_norm",
        "post_attn_norm": "post_attention_layernorm",
        "w_q": "self_attn.q_proj", "w_k": "self_attn.k_proj",
        "w_v": "self_attn.v_proj", "w_o": "self_attn.o_proj",
        "w_gate": "mlp.gate_proj", "w_up": "mlp.up_proj", "w_down": "mlp.down_proj",
    }.items():
        weights[f"layer.0.{logical}"] = get(f"{prefix}.{source}.weight", logical.startswith("w_"))
    draft_vocab = weights["w_out"].shape[1]
    raw = tensors["d2t"].numpy().astype(np.int64)
    # This checkpoint stores offsets from each draft vocabulary index.
    mapping = raw + np.arange(draft_vocab, dtype=np.int64)
    if (mapping.shape != (draft_vocab,) or np.any(mapping < 0)
            or np.any(mapping >= embedding.shape[0]) or len(np.unique(mapping)) != draft_vocab):
        raise ValueError("invalid EAGLE3 offset vocabulary map")
    if weights["fc"].shape != (3 * embedding.shape[1], config.hidden_size):
        raise ValueError("EAGLE3 feature projection does not match the target")
    return config, weights, mapping.astype(np.int32)


def build_speculative(*, model_dir: Path, draft_dir: Path, output: Path,
                      max_sequence_length: int = 2048, max_query: int = 64,
                      draft_depth: int = 4, spec_dec: str = "eagle3", verbose: bool = False,
                      execution_profiles: str = "single", prefill_query: int = 64) -> int:
    from .graph import build_draft, build_target

    if spec_dec != "eagle3":
        raise ValueError("only EAGLE3 is implemented in this prototype")
    config = ModelConfig.from_dir(model_dir)
    if (config.model_type != "llama" or config.hidden_size != 4096
            or config.num_hidden_layers != 32 or config.vocab_size != 128256
            or config.num_attention_heads != 32 or config.num_key_value_heads != 8):
        raise ValueError("first scope requires Llama 3.1 8B with its full vocabulary")
    if not 1 <= draft_depth or not 2 * draft_depth + 1 <= max_query <= 64:
        raise ValueError("require draft_depth >= 1 and 2 * draft_depth + 1 <= max_query <= 64")
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("capacity exceeds target context window")
    if execution_profiles not in {"single", "split"}:
        raise ValueError("execution_profiles must be single or split")
    target_profiles, draft_profiles = (), ()
    target_max, draft_max = max_query, max_query
    if execution_profiles == "split":
        if not 1 <= prefill_query <= max_sequence_length:
            raise ValueError("prefill_query exceeds cache capacity")
        prefill = ExecutionProfile("prefill", (1, prefill_query, prefill_query), (1, 1, 1))
        verify = (1, draft_depth + 1, 2 * draft_depth + 1)
        target_profiles = (prefill, ExecutionProfile("decode", verify, verify))
        draft_profiles = (prefill, ExecutionProfile("decode", (1, 1, draft_depth + 1), (1, 1, 1)))
        target_max = max(prefill_query, verify[2])
        draft_max = max(prefill_query, draft_depth + 1)
    target_contract = EngineContract(
        "target", config.num_hidden_layers, config.hidden_size, config.num_key_value_heads,
        config.head_dim, config.vocab_size, max_sequence_length, target_max, 3 * config.hidden_size,
        execution_profiles=target_profiles,
    )
    weights = load_standard_weights(model_dir, config, precision="fp16")
    draft_config, draft_weights, mapping = load_draft(draft_dir, weights["embedding"])
    # Preserve the draft's own RoPE parameters, including its defaults when
    # absent. This checkpoint does not inherit the target's scaled RoPE.
    draft_contract = EngineContract(
        "draft", draft_config.num_hidden_layers, draft_config.hidden_size,
        draft_config.num_key_value_heads, draft_config.head_dim, len(mapping),
        max_sequence_length, draft_max, 3 * config.hidden_size,
        execution_profiles=draft_profiles,
    )
    writer = BundleWriter(output)
    runtime_metadata = _runtime_config(model_dir, config)
    try:
        writer.set_header(family="llama", task="text_generation", backend="trt")
        metadata = {
            "version": 1, "method": spec_dec, "draft_depth": draft_depth,
            "target": target_contract.to_dict(), "draft": draft_contract.to_dict(),
            "target_feature_indices": [2, 16, 28], "d2t": mapping.tolist(),
            "stop_token_ids": runtime_metadata.get("eos_token_ids", [runtime_metadata["eos_token_id"]]),
            "attention_lowering": "tensorrt_primitives",
            "target_config": config.raw, "draft_config": draft_config.raw,
        }
        metadata["device_policy"] = {"version": 1}
        writer.add_json("speculative.json", metadata)
        print("Compiling speculative target...", flush=True)
        writer.add_bytes("target.plan", build_target(config, weights, target_contract, verbose=verbose))
        del weights
        print("Compiling EAGLE3 draft...", flush=True)
        writer.add_bytes("draft.plan", build_draft(draft_config, draft_weights, draft_contract, verbose=verbose))
        from .selection import add_selection_plans
        add_selection_plans(writer, target_contract, draft_contract, verbose=verbose)
        from .device_policy import add_device_policy_plans
        add_device_policy_plans(writer, target_contract, draft_depth,
                                mapping, metadata["stop_token_ids"])
        for filename in _BUNDLE_FILES:
            path = model_dir / filename
            if path.is_file():
                writer.add_bytes(filename, path.read_bytes())
        template = _chat_template(model_dir)
        if template is not None:
            writer.add_bytes("chat_template.jinja", template)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return 0

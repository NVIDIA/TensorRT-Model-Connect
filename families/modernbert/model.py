# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT family plugin -- encoder-only transformer with modern design.

ModernBERT differs significantly from classic BERT:
  - PRE-norm with LayerNorm (no bias) -- NOT RMSNorm despite weight naming
  - Fused QKV projection (Wqkv) -- split into Q/K/V
  - GeGLU MLP (fused Wi gate+up, Wo down) -- split Wi into gate/up
  - RoPE position encoding with per-layer theta (full_attention=160000, sliding=10000)
  - No token type embeddings
  - No attention bias, no MLP bias
  - Layer 0 has no attn_norm (identity)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys
from pathlib import Path

import numpy as np

from .config import ModelConfig, resolve_attention_contract
from .parallel import ParallelConfig
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from .graph import model as graph_ops
from .parallel import normalize_parallel_config

import tensorrt as trt


def _add_layernorm_no_bias(
    network,
    inp,
    hidden_size,
    gamma,
    eps,
    *,
    dtype=np.float32,
):
    """LayerNorm without bias via TRT native normalization.

    ModernBERT uses nn.LayerNorm(bias=False) which still mean-centers,
    unlike RMSNorm which does not.
    """
    beta = np.zeros(hidden_size, dtype=dtype)
    return graph_ops.add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _ModernBertModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        num_layers = config.num_hidden_layers
        intermediate = config.intermediate_size

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, "model.embeddings.tok_embeddings.weight")
        assert embedding.shape == (config.vocab_size, hidden)
        weights["embedding"] = embedding.astype(np.float32)

        # Embedding LayerNorm (no bias)
        weights["embed_norm"] = _load_tensor(readers, "model.embeddings.norm.weight").astype(
            np.float32
        )

        # Final LayerNorm
        weights["final_norm"] = _load_tensor(readers, "model.final_norm.weight").astype(np.float32)

        # MLM head weights (optional)
        if _has_tensor(readers, "head.dense.weight"):
            weights["head_dense_w"] = np.ascontiguousarray(
                _load_tensor(readers, "head.dense.weight").T.astype(np.float32)
            )
        if _has_tensor(readers, "head.norm.weight"):
            weights["head_norm"] = _load_tensor(readers, "head.norm.weight").astype(np.float32)
        if _has_tensor(readers, "decoder.bias"):
            weights["decoder_bias"] = _load_tensor(readers, "decoder.bias").astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Attention LayerNorm (layer 0 has no attn_norm)
            attn_norm_key = f"{hf_prefix}.attn_norm.weight"
            if _has_tensor(readers, attn_norm_key):
                weights[f"{prefix}.attn_norm"] = _load_tensor(readers, attn_norm_key).astype(
                    np.float32
                )

            # Fused QKV: [3*hidden, hidden] -> split into Q, K, V
            wqkv = _load_tensor(readers, f"{hf_prefix}.attn.Wqkv.weight")
            assert wqkv.shape == (3 * hidden, hidden)
            q_w, k_w, v_w = np.split(wqkv, 3, axis=0)
            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # Output projection
            wo = _load_tensor(readers, f"{hf_prefix}.attn.Wo.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(wo.T.astype(np.float32))

            # MLP LayerNorm
            weights[f"{prefix}.mlp_norm"] = _load_tensor(
                readers, f"{hf_prefix}.mlp_norm.weight"
            ).astype(np.float32)

            # GeGLU MLP: Wi [2*intermediate, hidden] -> split into input, gate
            wi = _load_tensor(readers, f"{hf_prefix}.mlp.Wi.weight")
            assert wi.shape == (2 * intermediate, hidden)
            input_w, gate_w = np.split(wi, 2, axis=0)
            weights[f"{prefix}.w_mlp_input"] = np.ascontiguousarray(input_w.T.astype(np.float32))
            weights[f"{prefix}.w_mlp_gate"] = np.ascontiguousarray(gate_w.T.astype(np.float32))

            # Down projection
            mlp_wo = _load_tensor(readers, f"{hf_prefix}.mlp.Wo.weight")
            weights[f"{prefix}.w_down"] = np.ascontiguousarray(mlp_wo.T.astype(np.float32))

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("ModernBERT tensor-parallel builds do not support quantization")
            from .model.parallel import build_tp_modernbert_engine

            return build_tp_modernbert_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads
        intermediate = config.intermediate_size
        eps = config.raw.get("norm_eps", config.rms_norm_eps)
        max_seq = max_cache_length
        if max_seq < 1:
            raise ValueError("ModernBERT max_cache_length must be positive")
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported ModernBERT precision: {precision}")

        attention_contract = resolve_attention_contract(config)
        layer_types = attention_contract.layer_types
        full_theta = attention_contract.full_rope_theta
        sliding_theta = attention_contract.sliding_rope_theta

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        # Inputs
        input_ids = network.add_input("input_ids", trt.int32, (-1,))
        attention_mask_input = network.add_input("attention_mask", trt.int32, (-1,))
        profile = builder.create_optimization_profile()
        opt_seq = min(16, max_seq)
        profile.set_shape("input_ids", (1,), (opt_seq,), (max_seq,))
        profile.set_shape("attention_mask", (1,), (opt_seq,), (max_seq,))
        trt_config.add_optimization_profile(profile)
        sequence_shape = network.add_shape(input_ids).get_output(0)

        # Attention mask: [seq] int -> [1, 1, 1, seq] additive float mask.
        mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
        ones_c = graph_ops.add_constant(
            network, (1,), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
        )
        mask_penalty = -1e4 if precision == "fp16" else -1e10
        neg_large = graph_ops.add_constant(
            network, (1,), np.array([mask_penalty], dtype=work_np_dtype), dtype=work_np_dtype
        )
        inv_mask = network.add_elementwise(
            ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB
        )
        pad_penalty = network.add_elementwise(
            inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
        )
        pad_mask_4d = network.add_shuffle(pad_penalty.get_output(0))
        pad_mask_4d.reshape_dims = (1, 1, 1, -1)

        # Pre-compute RoPE tables for both theta values
        rope_tables = {}
        for theta in set([full_theta, sliding_theta]):
            cos = graph_ops.add_constant(
                network,
                (max_seq, head_dim // 2),
                graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=True),
                dtype=work_np_dtype,
            )
            sin = graph_ops.add_constant(
                network,
                (max_seq, head_dim // 2),
                graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=False),
                dtype=work_np_dtype,
            )
            rope_tables[theta] = (cos, sin)

        all_pos_indices = graph_ops.add_constant(
            network, (max_seq,), np.arange(max_seq, dtype=np.int32), dtype=np.int32
        )
        pos_slice = network.add_slice(all_pos_indices, start=(0,), shape=(1,), stride=(1,))
        pos_slice.set_input(2, sequence_shape)
        pos_indices = pos_slice.get_output(0)

        # Embedding
        embed_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
        )
        word_embed = network.add_gather(embed_table, input_ids, 0)
        hidden_state = _add_layernorm_no_bias(
            network,
            word_embed.get_output(0),
            hidden,
            weights["embed_norm"],
            eps,
            dtype=work_np_dtype,
        )

        # Encoder layers
        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # Determine RoPE theta for this layer
            lt = layer_types[layer_idx]
            if lt in ("full_attention", "global_attention"):
                theta = full_theta
            else:
                theta = sliding_theta
            cos_table, sin_table = rope_tables[theta]

            # Pre-norm attention
            has_attn_norm = f"{prefix}.attn_norm" in weights
            if has_attn_norm:
                attn_input = _add_layernorm_no_bias(
                    network,
                    hidden_state,
                    hidden,
                    weights[f"{prefix}.attn_norm"],
                    eps,
                    dtype=work_np_dtype,
                )
            else:
                attn_input = hidden_state

            # QKV projections
            q = graph_ops.add_matmul_rhs_constant(
                network, attn_input, hidden, hidden, weights[f"{prefix}.w_q"], dtype=work_np_dtype
            )
            k = graph_ops.add_matmul_rhs_constant(
                network, attn_input, hidden, hidden, weights[f"{prefix}.w_k"], dtype=work_np_dtype
            )
            v = graph_ops.add_matmul_rhs_constant(
                network, attn_input, hidden, hidden, weights[f"{prefix}.w_v"], dtype=work_np_dtype
            )

            # RoPE
            q = graph_ops.add_apply_rope_native(
                network,
                q,
                num_heads,
                head_dim,
                cos_table,
                sin_table,
                pos_indices,
                head_dim,
                sequence_length=None,
            )
            k = graph_ops.add_apply_rope_native(
                network,
                k,
                num_heads,
                head_dim,
                cos_table,
                sin_table,
                pos_indices,
                head_dim,
                sequence_length=None,
            )

            context_flat = graph_ops.add_attention_from_rows(
                network,
                q,
                k,
                v,
                num_heads=num_heads,
                head_dim=head_dim,
                q_seq=None,
                kv_seq=None,
                mask=pad_mask_4d.get_output(0),
            )

            attn_out = graph_ops.add_matmul_rhs_constant(
                network, context_flat, hidden, hidden, weights[f"{prefix}.w_o"], dtype=work_np_dtype
            )

            # Residual
            res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = res1.get_output(0)

            # Pre-norm GeGLU MLP
            mlp_input = _add_layernorm_no_bias(
                network,
                hidden_state,
                hidden,
                weights[f"{prefix}.mlp_norm"],
                eps,
                dtype=work_np_dtype,
            )

            # GeGLU: act(input) * gate
            inp_proj = graph_ops.add_matmul_rhs_constant(
                network,
                mlp_input,
                hidden,
                intermediate,
                weights[f"{prefix}.w_mlp_input"],
                dtype=work_np_dtype,
            )
            gate_proj = graph_ops.add_matmul_rhs_constant(
                network,
                mlp_input,
                hidden,
                intermediate,
                weights[f"{prefix}.w_mlp_gate"],
                dtype=work_np_dtype,
            )
            inp_act = graph_ops.add_gelu_erf(network, inp_proj, dtype=work_np_dtype)
            gated = network.add_elementwise(inp_act, gate_proj, trt.ElementWiseOperation.PROD)

            down = graph_ops.add_matmul_rhs_constant(
                network,
                gated.get_output(0),
                intermediate,
                hidden,
                weights[f"{prefix}.w_down"],
                dtype=work_np_dtype,
            )

            res2 = network.add_elementwise(hidden_state, down, trt.ElementWiseOperation.SUM)
            hidden_state = res2.get_output(0)

        # Final norm
        hidden_state = _add_layernorm_no_bias(
            network, hidden_state, hidden, weights["final_norm"], eps, dtype=work_np_dtype
        )

        public_output = hidden_state
        if public_output.dtype != trt.float32:
            public_output = network.add_cast(public_output, trt.float32).get_output(0)
        public_output.name = "hidden_states"
        network.mark_output(public_output)

        if verbose:
            print(
                f"[trtmc build] Building ModernBERT encoder TRT engine "
                f"({num_layers} layers, hidden={hidden}, seq_len={max_seq}, "
                f"precision={precision}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(plan)


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
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


def _tokenizer_runtime_contract(model_dir: Path) -> dict[str, object]:
    """Resolve this family's exact native-tokenizer framing."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=True,
    )
    default_ids = list(tokenizer.encode("hello"))
    plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    if default_ids == plain_ids:
        prefix_ids, suffix_ids = [], []
    elif not plain_ids:
        prefix_ids, suffix_ids = default_ids, []
    else:
        frame = next(
            (
                start
                for start in range(len(default_ids) - len(plain_ids) + 1)
                if default_ids[start : start + len(plain_ids)] == plain_ids
            ),
            None,
        )
        if frame is None:
            raise RuntimeError("tokenizer special-token framing is not a prefix/suffix")
        prefix_ids = default_ids[:frame]
        suffix_ids = default_ids[frame + len(plain_ids) :]
    return {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": prefix_ids,
        "tokenizer_suffix_ids": suffix_ids,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.dynamic_kv_cache:
        raise NotImplementedError("modernbert does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("modernbert does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("modernbert does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("modernbert does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("modernbert does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("modernbert task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("modernbert"):
        raise ValueError(
            f"ModernBERT builder requires model_type='modernbert', got {config.model_type!r}"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("ModernBERT precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("ModernBERT max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("ModernBERT has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("ModernBERT does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _ModernBertModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="modernbert", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            **_tokenizer_runtime_contract(model_dir),
            "tensor_parallel_size": parallel.tp_size,
        },
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

"""DeepSeek-OCR-2 family plugin — VL model: SAM ViT + Qwen2 encoder + MoE decoder.

DeepSeek-OCR-2 is a VL model with a DeepSeek-V2-style language decoder. Unlike
the full DeepSeek-V2 which uses Multi-head Latent Attention (MLA), OCR-2 uses
standard Llama-style multi-head attention (use_mla=False). The MLP layers use
DeepSeek-V2 MoE with shared experts for layers >= first_k_dense_replace, and
dense SwiGLU for earlier layers.

Vision pipeline:
  1. SAM ViT-B: [1, 3, 1024, 1024] -> patch embed -> 12 blocks (window/global
     attention with relative position biases) -> neck + downsample convs
     -> [1, 896, 16, 16]
  2. Qwen2 Decoder-as-Encoder: flatten SAM features [1, 256, 896], concat with
     learned queries [256, 896], run 24 Qwen2 layers with mixed attention
     (bidirectional for image, causal for queries), take query outputs
     -> [1, 256, 896]
  3. Linear Projector: [1, 256, 896] -> [1, 256, 1280]
  4. View Separator: append [1, 1280] -> total 257 vision tokens

Architecture:
  - Attention: Standard Q/K/V/O (no biases, no GQA — heads == kv_heads)
  - RoPE: Standard rotary position embeddings
  - Layer 0: Dense SwiGLU MLP (intermediate_size=6848)
  - Layers 1-11: MoE (64 experts, top-6, intermediate=896) + shared experts (2, intermediate=1792)
  - Norm: RMSNorm

Operational note:
  - DeepSeek-OCR VL prefill injects 257 image tokens before user text.
  - Very small max_cache_length (especially <=257) can degrade OCR output
    (prompt echo / repeated "skip" style tokens). Use 4096 for stable OCR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .standard_decoder_builder import _apply_norm, _mark_debug_output


trt = trt_compat.get_trt()

class DeepSeekOCRPlugin:
    name = "deepseek_ocr"
    runtime_strategy = "vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("deepseek_vl_v2",)

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        # MoE config from raw
        raw = config.raw
        n_routed_experts = raw.get("n_routed_experts", 64)
        n_shared_experts = raw.get("n_shared_experts", 2)
        num_experts_per_tok = raw.get("num_experts_per_tok", 6)
        first_k_dense_replace = raw.get("first_k_dense_replace", 1)
        moe_intermediate_size = raw.get("moe_intermediate_size", 896)
        shared_intermediate = moe_intermediate_size * n_shared_experts

        weights = WeightDict()

        # Weight key prefix: DeepSeek-VL2 stores language decoder weights
        # under "language.model.*" rather than bare "model.*".
        lang_prefix = ""
        if _has_tensor(readers, "language.model.embed_tokens.weight"):
            lang_prefix = "language."

        # Embedding
        embedding = _load_tensor(readers, f"{lang_prefix}model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = 0
        kv_attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{lang_prefix}model.layers.{layer_idx}"

            # RMSNorm weights
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # Standard Q/K/V/O attention projections (no biases)
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]
            if kv_attention_size == 0:
                kv_attention_size = k_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")
            del q_raw, k_raw, v_raw, o_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # MLP: dense or MoE depending on layer
            is_moe_layer = layer_idx >= first_k_dense_replace

            if is_moe_layer:
                # Router weight
                router_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate.weight")
                weights[f"{prefix}.router"] = _transpose_2d(
                    router_raw, "router")
                del router_raw

                # Per-expert weights
                for e in range(n_routed_experts):
                    exp_hf = f"{hf_prefix}.mlp.experts.{e}"
                    gate_raw = _load_tensor(
                        readers, f"{exp_hf}.gate_proj.weight")
                    up_raw = _load_tensor(
                        readers, f"{exp_hf}.up_proj.weight")
                    down_raw = _load_tensor(
                        readers, f"{exp_hf}.down_proj.weight")

                    weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate")
                    weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(
                        up_raw, f"expert_{e}_up")
                    weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(
                        down_raw, f"expert_{e}_down")
                    del gate_raw, up_raw, down_raw

                # Shared expert weights
                shared_hf = f"{hf_prefix}.mlp.shared_experts"
                s_gate_raw = _load_tensor(
                    readers, f"{shared_hf}.gate_proj.weight")
                s_up_raw = _load_tensor(
                    readers, f"{shared_hf}.up_proj.weight")
                s_down_raw = _load_tensor(
                    readers, f"{shared_hf}.down_proj.weight")

                weights[f"{prefix}.shared.w_gate"] = _transpose_2d(
                    s_gate_raw, "shared_gate")
                weights[f"{prefix}.shared.w_up"] = _transpose_2d(
                    s_up_raw, "shared_up")
                weights[f"{prefix}.shared.w_down"] = _transpose_2d(
                    s_down_raw, "shared_down")
                del s_gate_raw, s_up_raw, s_down_raw
            else:
                # Dense SwiGLU MLP
                gate_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(
                    gate_raw, "gate_proj")
                weights[f"{prefix}.w_up"] = _transpose_2d(
                    up_raw, "up_proj")
                weights[f"{prefix}.w_down"] = _transpose_2d(
                    down_raw, "down_proj")
                del gate_raw, up_raw, down_raw

        # Final norm
        final_norm_key = f"{lang_prefix}model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = f"{lang_prefix}lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        # Store metadata
        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
        weights["_n_routed_experts"] = n_routed_experts  # type: ignore[assignment]
        weights["_n_shared_experts"] = n_shared_experts  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = num_experts_per_tok  # type: ignore[assignment]
        weights["_first_k_dense_replace"] = first_k_dense_replace  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = moe_intermediate_size  # type: ignore[assignment]
        weights["_shared_intermediate_size"] = shared_intermediate  # type: ignore[assignment]
        weights["_norm_topk_prob"] = raw.get("norm_topk_prob", False)  # type: ignore[assignment]
        weights["_routed_scaling_factor"] = raw.get("routed_scaling_factor", 1.0)  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="DeepSeek-OCR tensor-parallel builds")
            if debug_layer_outputs:
                raise ValueError(
                    "DeepSeek-OCR tensor-parallel builds do not support "
                    "debug layer outputs")
            from .tp_builder import build_deepseek_ocr_tp_engine
            return build_deepseek_ocr_tp_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel)

        image_prefill_tokens = 257
        if max_cache_length <= image_prefill_tokens:
            print(
                "[trtmc build] WARNING: DeepSeek-OCR-2 uses 257 image prefill tokens. "
                f"max_cache_length={max_cache_length} is too small and can cause "
                "prompt echo / repeated skip-like tokens. Use --max-cache-length 4096.",
                file=sys.stderr,
            )
        elif max_cache_length < 4096:
            print(
                "[trtmc build] NOTE: DeepSeek-OCR-2 is more stable with "
                "--max-cache-length 4096.",
                file=sys.stderr,
            )

        attention_size: int = weights.get("_attention_size", config.attention_size)
        n_routed_experts: int = weights["_n_routed_experts"]
        num_experts_per_tok: int = weights["_num_experts_per_tok"]
        first_k_dense_replace: int = weights["_first_k_dense_replace"]
        moe_intermediate: int = weights["_moe_intermediate_size"]
        shared_intermediate: int = weights["_shared_intermediate_size"]
        norm_topk_prob: bool = weights.get("_norm_topk_prob", False)
        routed_scaling_factor: float = weights.get("_routed_scaling_factor", 1.0)
        dense_intermediate = config.intermediate_size

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        # VL embed_input: allow vision features to override embedding
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (1,))

        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])

        graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
        cos_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        # -----------------------------------------------------------
        # Embedding with input_embed override for VL
        # -----------------------------------------------------------
        gather = network.add_gather(embedding_table, token_id, 0)
        token_embed = gather.get_output(0)

        # Conditional: (1 - flag) * token_embed + flag * input_embed
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=np.float32))
        inv_flag = network.add_elementwise(
            one_const, flag_broadcast.get_output(0),
            trt.ElementWiseOperation.SUB)
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed,
            trt.ElementWiseOperation.PROD)
        embed_part = network.add_elementwise(
            flag_broadcast.get_output(0), input_embed_tensor,
            trt.ElementWiseOperation.PROD)
        hidden_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = hidden_sum.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # -----------------------------------------------------------
        # Decoder layers
        # -----------------------------------------------------------
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            is_moe_layer = layer_idx >= first_k_dense_replace

            result = _add_decoder_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx],
                cache_v=cache_v_inputs[layer_idx],
                attention_mask=attention_mask,
                position_id=position_id,
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                eps_tensor=eps_tensor,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                attention_size=attention_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_cache_length=max_cache_length,
                is_moe_layer=is_moe_layer,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate=moe_intermediate,
                shared_intermediate=shared_intermediate,
                dense_intermediate=dense_intermediate,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
            )

            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])

            if debug_layer_outputs:
                _mark_debug_output(
                    network, result["post_attn"],
                    f"debug_post_attn_{layer_idx}")
                _mark_debug_output(
                    network, hidden_state,
                    f"debug_hidden_{layer_idx}")

        # -----------------------------------------------------------
        # Final norm
        # -----------------------------------------------------------
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = _apply_norm(
                network, hidden_state, hidden, final_norm,
                None, eps_tensor, "rmsnorm")

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"])
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

        logits.name = "logits"
        network.mark_output(logits)

        # -----------------------------------------------------------
        # Present K/V outputs
        # -----------------------------------------------------------
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        # -----------------------------------------------------------
        # Build engine
        # -----------------------------------------------------------
        if verbose:
            print(f"[trtmc build] Building DeepSeek-OCR TRT engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, heads={num_heads}, "
                  f"experts={n_routed_experts}, top_k={num_experts_per_tok}, "
                  f"moe_inter={moe_intermediate}, "
                  f"shared_inter={shared_intermediate}, "
                  f"dense_inter={dense_intermediate}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build SAM ViT-B + Qwen2 encoder + projector + view_separator as TRT engine.

        Native TRT API implementation (no ONNX). Full pipeline:
        SAM ViT-B -> downsample convs -> Qwen2 encoder -> projector -> view_sep.

        Input:  pixel_values  [1, 3, 1024, 1024] float32
        Output: image_features [257, 1280] float32  (256 projected + 1 view_sep)
        """
        return _build_deepseek_ocr_vision_engine(model_dir, config, verbose=verbose)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Promote language_config fields to top level for C++ fast_path_config."""
        lang = config.raw.get("language_config", {})
        return {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "intermediate_size": config.intermediate_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
            "n_routed_experts": lang.get("n_routed_experts", 64),
            "n_shared_experts": lang.get("n_shared_experts", 2),
            "num_experts_per_tok": lang.get("num_experts_per_tok", 6),
            "first_k_dense_replace": lang.get("first_k_dense_replace", 1),
            "moe_intermediate_size": lang.get("moe_intermediate_size", 896),
        }

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return VL configuration for the bundle's config.json."""
        hidden = config.hidden_size  # 1280 (language decoder hidden)
        return {
            "image_token_id": config.raw.get("image_token_id", 128815),
            "fixed_image_size": 1024,
            "num_image_pad_tokens": 257,  # 256 projected + 1 view_separator
            "vision_output_dim": hidden,
            "preprocessor_type": "pad_center_chw",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "interpolation": "bicubic",
            "temporal_patch_size": 1,  # SAM ViT-B: no temporal tiling
            "vl_prompt_template": "{image_pads}\n{prompt}",
            "image_token_str": "<image>",
            "tokenizer_add_special_tokens": 1,
        }


# ---------------------------------------------------------------------------
# Helper: single SwiGLU expert
# ---------------------------------------------------------------------------

def _add_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down)
    return down


# ---------------------------------------------------------------------------
# MoE with shared experts
# ---------------------------------------------------------------------------

def _add_moe_with_shared_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    moe_intermediate: int,
    num_experts_per_tok: int,
    shared_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
) -> trt.ITensor:
    """MoE block with shared experts (DeepSeek-V2 style).

    Matches HF DeepSeekV2 MoEGate routing:
    1. Router logits -> softmax -> top-k selection
    2. If norm_topk_prob: renormalize selected weights to sum to 1.0
       Else: scale by routed_scaling_factor (default 1.0, i.e. use raw softmax probs)
    3. Compute all routed expert outputs, select top-k, weighted sum
    4. Compute shared expert output (always active, applied to original input)
    5. Final = routed_output + shared_output
    """
    # 1. Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, n_routed_experts,
        weights[f"{prefix}.router"])

    # 2. Softmax
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1

    # 3. TopK
    topk = network.add_topk(
        sm.get_output(0), trt.TopKOperation.MAX,
        num_experts_per_tok, 1 << 1)
    top_values = topk.get_output(0)   # [1, top_k]
    top_indices = topk.get_output(1)  # [1, top_k]

    # 4. Weight normalization (matches HF MoEGate):
    #    norm_topk_prob=True  -> renormalize: values / sum(values)
    #    norm_topk_prob=False -> scale by routed_scaling_factor (raw softmax probs)
    if norm_topk_prob:
        sum_val = network.add_reduce(
            top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
        final_weights = network.add_elementwise(
            top_values, sum_val.get_output(0),
            trt.ElementWiseOperation.DIV).get_output(0)
    elif routed_scaling_factor != 1.0:
        scale_c = graph_ops.add_constant(
            network, (1, 1),
            np.array([routed_scaling_factor], dtype=np.float32))
        final_weights = network.add_elementwise(
            top_values, scale_c,
            trt.ElementWiseOperation.PROD).get_output(0)
    else:
        final_weights = top_values

    # 5. Compute ALL routed expert outputs and stack
    expert_outputs = []
    for e in range(n_routed_experts):
        exp_out = _add_swiglu_expert(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [n_routed_experts, hidden_size]

    # 6. Gather selected experts, scale, and sum
    result = None
    for k in range(num_experts_per_tok):
        idx_slice = network.add_slice(
            top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        w_slice = network.add_slice(
            final_weights,
            start=(0, k), shape=(1, 1), stride=(1, 1))

        expert_out = network.add_gather(
            stacked_out, idx_flat.get_output(0), 0)

        scaled_expert = network.add_elementwise(
            expert_out.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)

        if result is None:
            result = scaled_expert.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                result, scaled_expert.get_output(0),
                trt.ElementWiseOperation.SUM)
            result = sum_layer.get_output(0)

    # 7. Shared expert output (always active, applied to original input)
    shared_out = _add_swiglu_expert(
        network, inp, hidden_size, shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
    )

    # 8. Combine: routed_output + shared_output
    combined = network.add_elementwise(
        result, shared_out, trt.ElementWiseOperation.SUM)

    return combined.get_output(0)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
) -> dict[str, trt.ITensor]:
    """One decoder layer: standard MHA + (dense MLP or MoE with shared experts)."""
    attention_window = max_cache_length + 1

    # Pre-attention RMSNorm
    norm1 = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        None, eps_tensor, "rmsnorm")

    # QKV projections (no biases)
    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"])

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)

    # Save present K/V
    present_k = k
    present_v = v

    # Reshape for cache concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=1, kv_seq=attention_window,
        mask=mask_4d)

    # Output projection
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat,
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"])

    # Residual
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None, eps_tensor, "rmsnorm")

    # MLP: dense or MoE with shared experts
    if is_moe_layer:
        mlp_out = _add_moe_with_shared_experts(
            network, norm2, weights, prefix,
            hidden_size, n_routed_experts, moe_intermediate,
            num_experts_per_tok, shared_intermediate,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor)
    else:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=dense_intermediate)

    # Residual
    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


# ---------------------------------------------------------------------------
# Vision engine builder (native TRT API)
# ---------------------------------------------------------------------------


def _get_rel_pos(q_size: int, k_size: int, rel_pos: np.ndarray) -> np.ndarray:
    """Get relative positional embeddings, matching SAM's get_rel_pos."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, rel_pos.shape[0])
        x_new = np.linspace(0, 1, max_rel_dist)
        f = interp1d(x_old, rel_pos, axis=0, kind='linear')
        rel_pos_resized = f(x_new).astype(np.float32)
    else:
        rel_pos_resized = rel_pos
    q_coords = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    indices = np.clip(relative_coords.astype(np.int64), 0, max_rel_dist - 1)
    return rel_pos_resized[indices]


def _build_sam_attention(
    network: trt.INetworkDefinition,
    inp_4d: trt.ITensor,
    weights: dict,
    w_prefix: str,
    grid_size: int,
    hidden: int,
    num_heads: int,
    head_dim: int,
    eps_t: trt.ITensor,
    window_size: int = 0,
) -> trt.ITensor:
    """Build SAM ViT attention block with decomposed relative position biases.

    Args:
        window_size: If > 0, apply windowed attention (partition into windows
            of this size, attend within each window, then unpartition).
            If 0, use global attention over the full grid.
    """
    # Determine effective spatial size for attention
    if window_size > 0:
        # Window partition: pad H,W to multiple of window_size, then reshape
        # into (num_windows, window_size, window_size, hidden).
        pad_h = (window_size - grid_size % window_size) % window_size
        pad_w = (window_size - grid_size % window_size) % window_size
        Hp = grid_size + pad_h
        Wp = grid_size + pad_w
        num_win_h = Hp // window_size
        num_win_w = Wp // window_size
        num_windows = num_win_h * num_win_w
        attn_spatial = window_size
        attn_seq = window_size * window_size
        attn_batch = num_windows  # treat windows as batch

        # Pad input: [1, H, W, C] -> [1, Hp, Wp, C] with zeros
        if pad_h > 0 or pad_w > 0:
            # Transpose to NCHW for TRT padding, then back to NHWC
            to_nchw = network.add_shuffle(inp_4d)
            to_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])
            pad_layer = network.add_padding_nd(
                to_nchw.get_output(0),
                pre_padding=(0, 0),
                post_padding=(pad_h, pad_w))
            to_nhwc = network.add_shuffle(pad_layer.get_output(0))
            to_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
            padded = to_nhwc.get_output(0)
        else:
            padded = inp_4d

        # Reshape to windows: [1, Hp, Wp, C] -> [1, nH, ws, nW, ws, C]
        #                    -> [nH*nW, ws, ws, C]
        r1 = network.add_shuffle(padded)
        r1.reshape_dims = (1, num_win_h, window_size, num_win_w, window_size, hidden)
        r1.second_transpose = trt.Permutation([0, 1, 3, 2, 4, 5])
        r2 = network.add_shuffle(r1.get_output(0))
        r2.reshape_dims = (num_windows, window_size, window_size, hidden)
        attn_input = r2.get_output(0)  # [num_windows, ws, ws, C]
    else:
        attn_spatial = grid_size
        attn_seq = grid_size * grid_size
        attn_batch = 1
        attn_input = inp_4d  # [1, H, W, C]

    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Flatten spatial dims: [B, H, W, C] -> [B*H*W, C]
    flat = network.add_shuffle(attn_input)
    flat.reshape_dims = (attn_batch * attn_seq, hidden)

    q = graph_ops.add_matmul_rhs_constant(
        network, flat.get_output(0), hidden, hidden,
        weights[f"{w_prefix}.attn.q.weight"])
    q = graph_ops.add_bias_sum(
        network, q, hidden, weights[f"{w_prefix}.attn.q.bias"])
    k = graph_ops.add_matmul_rhs_constant(
        network, flat.get_output(0), hidden, hidden,
        weights[f"{w_prefix}.attn.k.weight"])
    k = graph_ops.add_bias_sum(
        network, k, hidden, weights[f"{w_prefix}.attn.k.bias"])
    v = graph_ops.add_matmul_rhs_constant(
        network, flat.get_output(0), hidden, hidden,
        weights[f"{w_prefix}.attn.v.weight"])
    v = graph_ops.add_bias_sum(
        network, v, hidden, weights[f"{w_prefix}.attn.v.bias"])

    # Reshape to [B*num_heads, seq, head_dim] for batched matmul
    q_h = network.add_shuffle(q)
    q_h.reshape_dims = (attn_batch, attn_seq, num_heads, head_dim)
    q_h.second_transpose = trt.Permutation([0, 2, 1, 3])
    q_r = network.add_shuffle(q_h.get_output(0))
    q_r.reshape_dims = (attn_batch * num_heads, attn_seq, head_dim)

    k_h = network.add_shuffle(k)
    k_h.reshape_dims = (attn_batch, attn_seq, num_heads, head_dim)
    k_h.second_transpose = trt.Permutation([0, 2, 1, 3])
    k_r = network.add_shuffle(k_h.get_output(0))
    k_r.reshape_dims = (attn_batch * num_heads, attn_seq, head_dim)

    v_h = network.add_shuffle(v)
    v_h.reshape_dims = (attn_batch, attn_seq, num_heads, head_dim)
    v_h.second_transpose = trt.Permutation([0, 2, 1, 3])
    v_r = network.add_shuffle(v_h.get_output(0))
    v_r.reshape_dims = (attn_batch * num_heads, attn_seq, head_dim)

    score = network.add_matrix_multiply(
        q_r.get_output(0), trt.MatrixOperation.NONE,
        k_r.get_output(0), trt.MatrixOperation.TRANSPOSE)
    scale_c = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    scaled = network.add_elementwise(
        score.get_output(0), scale_c, trt.ElementWiseOperation.PROD)

    # Add decomposed relative position bias (using attn_spatial, not grid_size)
    rp_h_key = f"{w_prefix}.attn.rel_pos_h"
    if rp_h_key in weights:
        rp_h = _get_rel_pos(attn_spatial, attn_spatial, weights[rp_h_key])
        rp_w = _get_rel_pos(attn_spatial, attn_spatial,
                             weights[f"{w_prefix}.attn.rel_pos_w"])

        # q for rel_pos: [B*N, seq, D] -> [B*N, H, W, D]
        q_4d = network.add_shuffle(q_r.get_output(0))
        q_4d.reshape_dims = (attn_batch * num_heads, attn_spatial, attn_spatial, head_dim)

        # H-axis: [H, B*N*W, D] @ [H, D, K] -> [H, B*N*W, K]
        q_perm_h = network.add_shuffle(q_4d.get_output(0))
        q_perm_h.first_transpose = trt.Permutation([1, 0, 2, 3])
        q_perm_h.reshape_dims = (attn_spatial, attn_batch * num_heads * attn_spatial, head_dim)
        rp_h_t = rp_h.transpose(0, 2, 1).astype(np.float32)
        rp_h_c = graph_ops.add_constant(network, rp_h_t.shape, rp_h_t)
        rel_h_mm = network.add_matrix_multiply(
            q_perm_h.get_output(0), trt.MatrixOperation.NONE,
            rp_h_c, trt.MatrixOperation.NONE)
        rel_h_4d = network.add_shuffle(rel_h_mm.get_output(0))
        rel_h_4d.reshape_dims = (attn_spatial, attn_batch * num_heads, attn_spatial, attn_spatial)
        rel_h_4d.second_transpose = trt.Permutation([1, 0, 2, 3])

        # W-axis: [W, B*N*H, D] @ [W, D, K] -> [W, B*N*H, K]
        q_perm_w = network.add_shuffle(q_4d.get_output(0))
        q_perm_w.first_transpose = trt.Permutation([2, 0, 1, 3])
        q_perm_w.reshape_dims = (attn_spatial, attn_batch * num_heads * attn_spatial, head_dim)
        rp_w_t = rp_w.transpose(0, 2, 1).astype(np.float32)
        rp_w_c = graph_ops.add_constant(network, rp_w_t.shape, rp_w_t)
        rel_w_mm = network.add_matrix_multiply(
            q_perm_w.get_output(0), trt.MatrixOperation.NONE,
            rp_w_c, trt.MatrixOperation.NONE)
        rel_w_4d = network.add_shuffle(rel_w_mm.get_output(0))
        rel_w_4d.reshape_dims = (attn_spatial, attn_batch * num_heads, attn_spatial, attn_spatial)
        rel_w_4d.second_transpose = trt.Permutation([1, 2, 0, 3])

        # Combine: [B*N,H,W,K,1] + [B*N,H,W,1,K] -> [B*N,H*W,K*K]
        rel_h_5d = network.add_shuffle(rel_h_4d.get_output(0))
        rel_h_5d.reshape_dims = (attn_batch * num_heads, attn_spatial, attn_spatial, attn_spatial, 1)
        rel_w_5d = network.add_shuffle(rel_w_4d.get_output(0))
        rel_w_5d.reshape_dims = (attn_batch * num_heads, attn_spatial, attn_spatial, 1, attn_spatial)
        rel_bias = network.add_elementwise(
            rel_h_5d.get_output(0), rel_w_5d.get_output(0),
            trt.ElementWiseOperation.SUM)
        rel_bias_flat = network.add_shuffle(rel_bias.get_output(0))
        rel_bias_flat.reshape_dims = (attn_batch * num_heads, attn_seq, attn_seq)
        scaled = network.add_elementwise(
            scaled.get_output(0), rel_bias_flat.get_output(0),
            trt.ElementWiseOperation.SUM)

    # SAM-style 2D relative bias is query-dependent and is added before
    # softmax, which native IAttention cannot model as an additive mask.
    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2
    ctx = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_r.get_output(0), trt.MatrixOperation.NONE)

    # Reshape back: [B*N, seq, D] -> [B, N, H, W, D] -> [B, H, W, N*D]
    ctx_r = network.add_shuffle(ctx.get_output(0))
    ctx_r.reshape_dims = (attn_batch, num_heads, attn_spatial, attn_spatial, head_dim)
    ctx_r.second_transpose = trt.Permutation([0, 2, 3, 1, 4])
    ctx_flat = network.add_shuffle(ctx_r.get_output(0))
    ctx_flat.reshape_dims = (attn_batch * attn_seq, hidden)

    out = graph_ops.add_matmul_rhs_constant(
        network, ctx_flat.get_output(0), hidden, hidden,
        weights[f"{w_prefix}.attn.o.weight"])
    out = graph_ops.add_bias_sum(
        network, out, hidden, weights[f"{w_prefix}.attn.o.bias"])

    if window_size > 0:
        # Window unpartition: [nH*nW, ws, ws, C] -> [1, H, W, C]
        out_4d = network.add_shuffle(out)
        out_4d.reshape_dims = (num_win_h, num_win_w, window_size, window_size, hidden)
        out_4d.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
        out_merged = network.add_shuffle(out_4d.get_output(0))
        out_merged.reshape_dims = (1, Hp, Wp, hidden)

        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            crop = network.add_slice(
                out_merged.get_output(0),
                start=(0, 0, 0, 0),
                shape=(1, grid_size, grid_size, hidden),
                stride=(1, 1, 1, 1))
            return crop.get_output(0)
        else:
            return out_merged.get_output(0)
    else:
        out_4d = network.add_shuffle(out)
        out_4d.reshape_dims = (1, grid_size, grid_size, hidden)
        return out_4d.get_output(0)


def _load_vision_weights(
    model_dir: str,
) -> dict[str, np.ndarray]:
    """Load all vision pipeline weights from safetensors."""
    readers = _open_safetensors(Path(model_dir))
    vw: dict[str, np.ndarray] = {}

    # --- SAM ViT-B ---
    # Patch embed
    vw["sam.patch_embed.weight"] = _load_tensor(
        readers, "model.sam_model.patch_embed.proj.weight").astype(np.float32)
    vw["sam.patch_embed.bias"] = _load_tensor(
        readers, "model.sam_model.patch_embed.proj.bias").astype(np.float32)
    vw["sam.pos_embed"] = _load_tensor(
        readers, "model.sam_model.pos_embed").astype(np.float32)

    # Blocks
    for i in range(12):
        hf = f"model.sam_model.blocks.{i}"
        wp = f"sam.block{i}"
        vw[f"{wp}.norm1.weight"] = _load_tensor(readers, f"{hf}.norm1.weight").astype(np.float32)
        vw[f"{wp}.norm1.bias"] = _load_tensor(readers, f"{hf}.norm1.bias").astype(np.float32)
        vw[f"{wp}.norm2.weight"] = _load_tensor(readers, f"{hf}.norm2.weight").astype(np.float32)
        vw[f"{wp}.norm2.bias"] = _load_tensor(readers, f"{hf}.norm2.bias").astype(np.float32)
        # Fused QKV -> split
        qkv_w = _load_tensor(readers, f"{hf}.attn.qkv.weight").astype(np.float32)
        qkv_b = _load_tensor(readers, f"{hf}.attn.qkv.bias").astype(np.float32)
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
        vw[f"{wp}.attn.q.weight"] = _transpose_2d(q_w, "q")
        vw[f"{wp}.attn.q.bias"] = q_b.flatten().astype(np.float32)
        vw[f"{wp}.attn.k.weight"] = _transpose_2d(k_w, "k")
        vw[f"{wp}.attn.k.bias"] = k_b.flatten().astype(np.float32)
        vw[f"{wp}.attn.v.weight"] = _transpose_2d(v_w, "v")
        vw[f"{wp}.attn.v.bias"] = v_b.flatten().astype(np.float32)
        o_w = _load_tensor(readers, f"{hf}.attn.proj.weight").astype(np.float32)
        o_b = _load_tensor(readers, f"{hf}.attn.proj.bias").astype(np.float32)
        vw[f"{wp}.attn.o.weight"] = _transpose_2d(o_w, "o")
        vw[f"{wp}.attn.o.bias"] = o_b.flatten().astype(np.float32)
        # Rel pos
        if _has_tensor(readers, f"{hf}.attn.rel_pos_h"):
            vw[f"{wp}.attn.rel_pos_h"] = _load_tensor(
                readers, f"{hf}.attn.rel_pos_h").astype(np.float32)
            vw[f"{wp}.attn.rel_pos_w"] = _load_tensor(
                readers, f"{hf}.attn.rel_pos_w").astype(np.float32)
        # MLP
        fc1_w = _load_tensor(readers, f"{hf}.mlp.lin1.weight")
        fc1_b = _load_tensor(readers, f"{hf}.mlp.lin1.bias")
        fc2_w = _load_tensor(readers, f"{hf}.mlp.lin2.weight")
        fc2_b = _load_tensor(readers, f"{hf}.mlp.lin2.bias")
        vw[f"{wp}.mlp.fc1.weight"] = _transpose_2d(fc1_w, "fc1")
        vw[f"{wp}.mlp.fc1.bias"] = fc1_b.flatten().astype(np.float32)
        vw[f"{wp}.mlp.fc2.weight"] = _transpose_2d(fc2_w, "fc2")
        vw[f"{wp}.mlp.fc2.bias"] = fc2_b.flatten().astype(np.float32)

    # Neck
    vw["sam.neck.conv1.weight"] = _load_tensor(
        readers, "model.sam_model.neck.0.weight").astype(np.float32)
    vw["sam.neck.ln1.weight"] = _load_tensor(
        readers, "model.sam_model.neck.1.weight").astype(np.float32)
    vw["sam.neck.ln1.bias"] = _load_tensor(
        readers, "model.sam_model.neck.1.bias").astype(np.float32)
    vw["sam.neck.conv2.weight"] = _load_tensor(
        readers, "model.sam_model.neck.2.weight").astype(np.float32)
    vw["sam.neck.ln2.weight"] = _load_tensor(
        readers, "model.sam_model.neck.3.weight").astype(np.float32)
    vw["sam.neck.ln2.bias"] = _load_tensor(
        readers, "model.sam_model.neck.3.bias").astype(np.float32)

    # Downsample convs
    vw["sam.net_2.weight"] = _load_tensor(
        readers, "model.sam_model.net_2.weight").astype(np.float32)
    vw["sam.net_3.weight"] = _load_tensor(
        readers, "model.sam_model.net_3.weight").astype(np.float32)

    # --- Qwen2 encoder ---
    vw["qwen2.queries"] = _load_tensor(
        readers, "model.qwen2_model.query_1024.weight").astype(np.float32)
    vw["qwen2.final_norm"] = _load_tensor(
        readers, "model.qwen2_model.model.model.norm.weight").astype(np.float32)

    for i in range(24):
        hf = f"model.qwen2_model.model.model.layers.{i}"
        wp = f"qwen2.layer{i}"
        vw[f"{wp}.input_norm"] = _load_tensor(
            readers, f"{hf}.input_layernorm.weight").astype(np.float32)
        vw[f"{wp}.post_attn_norm"] = _load_tensor(
            readers, f"{hf}.post_attention_layernorm.weight").astype(np.float32)
        # Attention
        q_w = _load_tensor(readers, f"{hf}.self_attn.q_proj.weight")
        k_w = _load_tensor(readers, f"{hf}.self_attn.k_proj.weight")
        v_w = _load_tensor(readers, f"{hf}.self_attn.v_proj.weight")
        o_w = _load_tensor(readers, f"{hf}.self_attn.o_proj.weight")
        vw[f"{wp}.w_q"] = _transpose_2d(q_w, "q")
        vw[f"{wp}.w_k"] = _transpose_2d(k_w, "k")
        vw[f"{wp}.w_v"] = _transpose_2d(v_w, "v")
        vw[f"{wp}.w_o"] = _transpose_2d(o_w, "o")
        # Attention biases (Qwen2 has Q/K/V biases)
        vw[f"{wp}.q_bias"] = _load_tensor(
            readers, f"{hf}.self_attn.q_proj.bias").astype(np.float32)
        vw[f"{wp}.k_bias"] = _load_tensor(
            readers, f"{hf}.self_attn.k_proj.bias").astype(np.float32)
        vw[f"{wp}.v_bias"] = _load_tensor(
            readers, f"{hf}.self_attn.v_proj.bias").astype(np.float32)
        # MLP (SwiGLU)
        gate_w = _load_tensor(readers, f"{hf}.mlp.gate_proj.weight")
        up_w = _load_tensor(readers, f"{hf}.mlp.up_proj.weight")
        down_w = _load_tensor(readers, f"{hf}.mlp.down_proj.weight")
        vw[f"{wp}.w_gate"] = _transpose_2d(gate_w, "gate")
        vw[f"{wp}.w_up"] = _transpose_2d(up_w, "up")
        vw[f"{wp}.w_down"] = _transpose_2d(down_w, "down")

    # --- Projector ---
    vw["proj.weight"] = _transpose_2d(
        _load_tensor(readers, "model.projector.layers.weight"), "proj")
    vw["proj.bias"] = _load_tensor(
        readers, "model.projector.layers.bias").astype(np.float32)

    # --- View separator ---
    vw["view_sep"] = _load_tensor(
        readers, "model.view_seperator").astype(np.float32)

    return vw


def _build_deepseek_ocr_vision_engine(
    model_dir: str, config: ModelConfig, *, verbose: bool = False,
) -> bytes:
    """Build DeepSeek-OCR-2 vision engine using native TRT API.

    Pipeline: SAM ViT-B -> downsample -> Qwen2 encoder -> projector -> view_sep

    Input:  pixel_values  [1, 3, 1024, 1024] float32
    Output: image_features [257, 1280] float32 (256 projected + 1 view_sep)
    """
    print("[trtmc build] Building DeepSeek-OCR-2 vision engine (native TRT) ...",
          file=sys.stderr)

    vw = _load_vision_weights(model_dir)

    # SAM config
    sam_hidden = 768
    sam_layers = 12
    sam_heads = 12
    sam_head_dim = sam_hidden // sam_heads  # 64
    sam_mlp_dim = int(sam_hidden * 3.7362)  # ~2869, but actual is 3072 from weights
    sam_mlp_dim = vw["sam.block0.mlp.fc1.bias"].shape[0]  # 3072
    image_size = 1024
    patch_size = 16
    grid_size = image_size // patch_size  # 64
    seq_len = grid_size * grid_size  # 4096
    global_attn_indexes = {2, 5, 8, 11}

    # Qwen2 config
    qwen2_hidden = 896
    qwen2_layers = 24
    qwen2_heads = 14  # Q heads (896/64=14)
    qwen2_kv_heads = 2  # KV heads (128/64=2)
    qwen2_head_dim = 64
    qwen2_q_dim = qwen2_heads * qwen2_head_dim  # 896
    qwen2_kv_dim = qwen2_kv_heads * qwen2_head_dim  # 128
    qwen2_mlp_dim = vw["qwen2.layer0.w_gate"].shape[1]  # 4864
    qwen2_num_queries = vw["qwen2.queries"].shape[0]  # 256
    qwen2_total_seq = seq_len // (4 * 4) + qwen2_num_queries  # 256 + 256 = 512
    # Wait, SAM output is [1, 896, 16, 16] -> flatten -> [1, 256, 896]
    sam_out_spatial = 16  # after neck + downsample
    sam_out_seq = sam_out_spatial * sam_out_spatial  # 256
    qwen2_total_seq = sam_out_seq + qwen2_num_queries  # 512

    # Projector config
    proj_in = qwen2_hidden  # 896
    proj_out = 1280

    # RoPE config for Qwen2 (default theta=1000000.0 for Qwen2-0.5B)
    rope_theta = 1000000.0

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-6], dtype=np.float32))
    rms_eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-6], dtype=np.float32))

    # ===================================================================
    # Stage 1: SAM ViT-B
    # ===================================================================
    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, image_size, image_size))

    # Patch embedding: Conv2d [1, 3, 1024, 1024] -> [1, 768, 64, 64]
    pe_w = vw["sam.patch_embed.weight"]
    pe_b = vw["sam.patch_embed.bias"]
    patch_conv = network.add_convolution_nd(
        pixel_values, num_output_maps=sam_hidden,
        kernel_shape=(patch_size, patch_size),
        kernel=trt.Weights(np.ascontiguousarray(pe_w)),
        bias=trt.Weights(np.ascontiguousarray(pe_b)))
    patch_conv.stride_nd = (patch_size, patch_size)

    # NCHW -> NHWC: [1, 768, 64, 64] -> [1, 64, 64, 768]
    to_nhwc = network.add_shuffle(patch_conv.get_output(0))
    to_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])

    # Add position embedding
    pos_c = graph_ops.add_constant(
        network, (1, grid_size, grid_size, sam_hidden), vw["sam.pos_embed"])
    pos_sum = network.add_elementwise(
        to_nhwc.get_output(0), pos_c, trt.ElementWiseOperation.SUM)
    hidden_state = pos_sum.get_output(0)

    # Transformer blocks
    window_size = 14  # SAM ViT-B window size
    for layer_idx in range(sam_layers):
        wp = f"sam.block{layer_idx}"
        is_global = layer_idx in global_attn_indexes

        # Pre-attention LayerNorm
        reshape_2d = network.add_shuffle(hidden_state)
        reshape_2d.reshape_dims = (seq_len, sam_hidden)
        normed = graph_ops.add_layer_norm(
            network, reshape_2d.get_output(0), sam_hidden,
            vw[f"{wp}.norm1.weight"], vw[f"{wp}.norm1.bias"], eps_t)
        normed_4d = network.add_shuffle(normed)
        normed_4d.reshape_dims = (1, grid_size, grid_size, sam_hidden)

        # Attention: global for layers in global_attn_indexes, windowed otherwise
        attn_out_4d = _build_sam_attention(
            network, normed_4d.get_output(0), vw, wp,
            grid_size, sam_hidden, sam_heads, sam_head_dim, eps_t,
            window_size=0 if is_global else window_size)

        # Residual
        res1 = network.add_elementwise(
            hidden_state, attn_out_4d, trt.ElementWiseOperation.SUM)

        # Post-attention MLP
        res1_2d = network.add_shuffle(res1.get_output(0))
        res1_2d.reshape_dims = (seq_len, sam_hidden)
        normed2 = graph_ops.add_layer_norm(
            network, res1_2d.get_output(0), sam_hidden,
            vw[f"{wp}.norm2.weight"], vw[f"{wp}.norm2.bias"], eps_t)
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, sam_hidden, sam_mlp_dim,
            vw[f"{wp}.mlp.fc1.weight"])
        fc1 = graph_ops.add_bias_sum(
            network, fc1, sam_mlp_dim, vw[f"{wp}.mlp.fc1.bias"])
        gelu = graph_ops.add_gelu_erf(network, fc1)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu, sam_mlp_dim, sam_hidden,
            vw[f"{wp}.mlp.fc2.weight"])
        fc2 = graph_ops.add_bias_sum(
            network, fc2, sam_hidden, vw[f"{wp}.mlp.fc2.bias"])
        fc2_4d = network.add_shuffle(fc2)
        fc2_4d.reshape_dims = (1, grid_size, grid_size, sam_hidden)
        res2 = network.add_elementwise(
            res1.get_output(0), fc2_4d.get_output(0), trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    # Neck: NHWC -> NCHW, Conv1x1(768->256), LN, Conv3x3(256->256), LN
    to_nchw = network.add_shuffle(hidden_state)
    to_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])

    neck_c1 = network.add_convolution_nd(
        to_nchw.get_output(0), num_output_maps=256,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(vw["sam.neck.conv1.weight"])),
        bias=trt.Weights(np.zeros(256, dtype=np.float32)))

    # LN1: NCHW -> NHWC -> LN -> NCHW
    n1_nhwc = network.add_shuffle(neck_c1.get_output(0))
    n1_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
    n1_flat = network.add_shuffle(n1_nhwc.get_output(0))
    n1_flat.reshape_dims = (seq_len, 256)
    ln1 = graph_ops.add_layer_norm(
        network, n1_flat.get_output(0), 256,
        vw["sam.neck.ln1.weight"], vw["sam.neck.ln1.bias"], eps_t)
    ln1_4d = network.add_shuffle(ln1)
    ln1_4d.reshape_dims = (1, grid_size, grid_size, 256)
    ln1_nchw = network.add_shuffle(ln1_4d.get_output(0))
    ln1_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])

    neck_c2 = network.add_convolution_nd(
        ln1_nchw.get_output(0), num_output_maps=256,
        kernel_shape=(3, 3),
        kernel=trt.Weights(np.ascontiguousarray(vw["sam.neck.conv2.weight"])),
        bias=trt.Weights(np.zeros(256, dtype=np.float32)))
    neck_c2.padding_nd = (1, 1)

    # LN2
    n2_nhwc = network.add_shuffle(neck_c2.get_output(0))
    n2_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
    n2_flat = network.add_shuffle(n2_nhwc.get_output(0))
    n2_flat.reshape_dims = (seq_len, 256)
    ln2 = graph_ops.add_layer_norm(
        network, n2_flat.get_output(0), 256,
        vw["sam.neck.ln2.weight"], vw["sam.neck.ln2.bias"], eps_t)
    ln2_4d = network.add_shuffle(ln2)
    ln2_4d.reshape_dims = (1, grid_size, grid_size, 256)
    ln2_nchw = network.add_shuffle(ln2_4d.get_output(0))
    ln2_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])
    # SAM neck output: [1, 256, 64, 64]

    # Downsample: net_2 Conv2d(256->512, 3x3, stride=2, pad=1)
    net2 = network.add_convolution_nd(
        ln2_nchw.get_output(0), num_output_maps=512,
        kernel_shape=(3, 3),
        kernel=trt.Weights(np.ascontiguousarray(vw["sam.net_2.weight"])),
        bias=trt.Weights(np.zeros(512, dtype=np.float32)))
    net2.stride_nd = (2, 2)
    net2.padding_nd = (1, 1)
    # [1, 512, 32, 32]

    # Downsample: net_3 Conv2d(512->896, 3x3, stride=2, pad=1)
    net3 = network.add_convolution_nd(
        net2.get_output(0), num_output_maps=896,
        kernel_shape=(3, 3),
        kernel=trt.Weights(np.ascontiguousarray(vw["sam.net_3.weight"])),
        bias=trt.Weights(np.zeros(896, dtype=np.float32)))
    net3.stride_nd = (2, 2)
    net3.padding_nd = (1, 1)
    # SAM final output: [1, 896, 16, 16]

    # ===================================================================
    # Stage 2: Qwen2 Decoder-as-Encoder
    # ===================================================================
    # Flatten SAM features: [1, 896, 16, 16] -> NHWC -> [1, 256, 896]
    sam_nhwc = network.add_shuffle(net3.get_output(0))
    sam_nhwc.first_transpose = trt.Permutation([0, 2, 3, 1])
    sam_flat = network.add_shuffle(sam_nhwc.get_output(0))
    sam_flat.reshape_dims = (1, sam_out_seq, qwen2_hidden)  # [1, 256, 896]

    # Learned queries: [256, 896] -> [1, 256, 896]
    queries_c = graph_ops.add_constant(
        network, (1, qwen2_num_queries, qwen2_hidden), vw["qwen2.queries"])

    # Concatenate: [1, 256, 896] + [1, 256, 896] -> [1, 512, 896]
    enc_concat = network.add_concatenation(
        [sam_flat.get_output(0), queries_c])
    enc_concat.axis = 1
    enc_input = enc_concat.get_output(0)  # [1, 512, 896]

    # Flatten to 2D for transformer: [512, 896]
    enc_2d = network.add_shuffle(enc_input)
    enc_2d.reshape_dims = (qwen2_total_seq, qwen2_hidden)

    # Precompute native RoPE half-dim caches for Qwen2 positions 0..511.
    cos_half_np = graph_ops.make_rope_table_half_dim(
        qwen2_total_seq, qwen2_head_dim, rope_theta, True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        qwen2_total_seq, qwen2_head_dim, rope_theta, False)
    cos_half_c = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np)
    sin_half_c = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np)
    position_ids_c = graph_ops.add_constant(
        network,
        (qwen2_total_seq,),
        np.arange(qwen2_total_seq, dtype=np.int32),
        dtype=np.int32)

    enc_state = enc_2d.get_output(0)  # [512, 896]

    attn_scale = 1.0 / np.sqrt(qwen2_head_dim)

    # Standard causal mask (lower-triangular). HF Qwen2 SDPA uses is_causal=True
    # which creates this mask internally. All tokens attend only to preceding tokens.
    qwen2_mask_np = np.full(
        (qwen2_total_seq, qwen2_total_seq), -1e9, dtype=np.float32)
    for i in range(qwen2_total_seq):
        for j in range(i + 1):
            qwen2_mask_np[i, j] = 0.0
    qwen2_mask_c = graph_ops.add_constant(
        network, (1, qwen2_total_seq, qwen2_total_seq),
        qwen2_mask_np.reshape(1, qwen2_total_seq, qwen2_total_seq))


    for layer_idx in range(qwen2_layers):
        wp = f"qwen2.layer{layer_idx}"

        # Pre-attention RMSNorm
        norm1 = _apply_norm(
            network, enc_state, qwen2_hidden,
            vw[f"{wp}.input_norm"], None, rms_eps_t, "rmsnorm")

        # Q projection: [512, 896] @ [896, 896] -> [512, 896]
        q = graph_ops.add_matmul_rhs_constant(
            network, norm1, qwen2_hidden, qwen2_q_dim, vw[f"{wp}.w_q"])
        q = graph_ops.add_bias_sum(network, q, qwen2_q_dim, vw[f"{wp}.q_bias"])

        # K projection: [512, 896] @ [896, 128] -> [512, 128]
        k = graph_ops.add_matmul_rhs_constant(
            network, norm1, qwen2_hidden, qwen2_kv_dim, vw[f"{wp}.w_k"])
        k = graph_ops.add_bias_sum(network, k, qwen2_kv_dim, vw[f"{wp}.k_bias"])

        # V projection: [512, 896] @ [896, 128] -> [512, 128]
        v = graph_ops.add_matmul_rhs_constant(
            network, norm1, qwen2_hidden, qwen2_kv_dim, vw[f"{wp}.w_v"])
        v = graph_ops.add_bias_sum(network, v, qwen2_kv_dim, vw[f"{wp}.v_bias"])

        q_rope = graph_ops.add_apply_rope_native(
            network, q, qwen2_heads, qwen2_head_dim,
            cos_half_c, sin_half_c, position_ids_c, qwen2_head_dim,
            sequence_length=qwen2_total_seq)
        k_rope = graph_ops.add_apply_rope_native(
            network, k, qwen2_kv_heads, qwen2_head_dim,
            cos_half_c, sin_half_c, position_ids_c, qwen2_head_dim,
            sequence_length=qwen2_total_seq)

        mask_4d = graph_ops.add_3d_mask_to_4d(network, qwen2_mask_c)
        ctx_flat = graph_ops.add_attention_from_rows(
            network, q_rope, k_rope, v,
            num_heads=qwen2_heads,
            num_kv_heads=qwen2_kv_heads,
            head_dim=qwen2_head_dim,
            q_seq=qwen2_total_seq,
            kv_seq=qwen2_total_seq,
            mask=mask_4d,
            scale=attn_scale)

        # Output projection: [512, 896] @ [896, 896] -> [512, 896]
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx_flat,
            qwen2_q_dim, qwen2_hidden, vw[f"{wp}.w_o"])

        # Residual
        res1 = network.add_elementwise(
            enc_state, attn_out, trt.ElementWiseOperation.SUM)

        # Post-attention RMSNorm + SwiGLU MLP
        norm2 = _apply_norm(
            network, res1.get_output(0), qwen2_hidden,
            vw[f"{wp}.post_attn_norm"], None, rms_eps_t, "rmsnorm")

        gate = graph_ops.add_matmul_rhs_constant(
            network, norm2, qwen2_hidden, qwen2_mlp_dim, vw[f"{wp}.w_gate"])
        up = graph_ops.add_matmul_rhs_constant(
            network, norm2, qwen2_hidden, qwen2_mlp_dim, vw[f"{wp}.w_up"])
        sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(
            gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        gated = network.add_elementwise(
            swish.get_output(0), up, trt.ElementWiseOperation.PROD)
        down = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), qwen2_mlp_dim, qwen2_hidden,
            vw[f"{wp}.w_down"])

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), down, trt.ElementWiseOperation.SUM)
        enc_state = res2.get_output(0)

    # Final RMSNorm
    enc_state = _apply_norm(
        network, enc_state, qwen2_hidden,
        vw["qwen2.final_norm"], None, rms_eps_t, "rmsnorm")

    # Extract query outputs: last 256 tokens from [512, 896]
    query_out = network.add_slice(
        enc_state,
        start=(sam_out_seq, 0),
        shape=(qwen2_num_queries, qwen2_hidden),
        stride=(1, 1))
    # [256, 896]

    # ===================================================================
    # Stage 3: Linear Projector
    # ===================================================================
    projected = graph_ops.add_matmul_rhs_constant(
        network, query_out.get_output(0),
        proj_in, proj_out, vw["proj.weight"])
    projected = graph_ops.add_bias_sum(
        network, projected, proj_out, vw["proj.bias"])
    # [256, 1280]

    # ===================================================================
    # Stage 4: View Separator
    # ===================================================================
    view_sep_c = graph_ops.add_constant(
        network, (1, proj_out), vw["view_sep"].reshape(1, -1))
    final_concat = network.add_concatenation(
        [projected, view_sep_c])
    final_concat.axis = 0
    # [257, 1280]

    output = final_concat.get_output(0)
    output.name = "image_features"
    network.mark_output(output)

    if verbose:
        print(f"[trtmc build] Building vision TRT engine "
              f"(SAM: {sam_layers} blocks, Qwen2: {qwen2_layers} layers, "
              f"output: 257x{proj_out}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Vision TRT engine build failed")

    return bytes(plan)


plugin = DeepSeekOCRPlugin()

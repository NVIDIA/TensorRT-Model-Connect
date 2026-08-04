# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native dual-stream TensorRT denoiser for Cosmos3-Nano T2V."""

from __future__ import annotations

import sys
from typing import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat

from . import trt_ops as op
from .checkpoint_mapper import load_component_state_dict
from .model_config import COSMOS3_NANO, Cosmos3NanoConfig


trt = trt_compat.get_trt()


def required_transformer_tensor_names(
    profile: Cosmos3NanoConfig = COSMOS3_NANO,
) -> tuple[str, ...]:
    names = [
        "embed_tokens.weight",
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
        "norm.weight",
        "norm_moe_gen.weight",
    ]
    for index in range(profile.num_hidden_layers):
        prefix = f"layers.{index}"
        names.extend(
            (
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.input_layernorm_moe_gen.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.post_attention_layernorm_moe_gen.weight",
                f"{prefix}.self_attn.to_q.weight",
                f"{prefix}.self_attn.to_k.weight",
                f"{prefix}.self_attn.to_v.weight",
                f"{prefix}.self_attn.to_out.weight",
                f"{prefix}.self_attn.add_q_proj.weight",
                f"{prefix}.self_attn.add_k_proj.weight",
                f"{prefix}.self_attn.add_v_proj.weight",
                f"{prefix}.self_attn.to_add_out.weight",
                f"{prefix}.self_attn.norm_q.weight",
                f"{prefix}.self_attn.norm_k.weight",
                f"{prefix}.self_attn.norm_added_q.weight",
                f"{prefix}.self_attn.norm_added_k.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp_moe_gen.gate_proj.weight",
                f"{prefix}.mlp_moe_gen.up_proj.weight",
                f"{prefix}.mlp_moe_gen.down_proj.weight",
            )
        )
    return tuple(names)


def validate_transformer_state_dict(
    state: Mapping[str, object],
    profile: Cosmos3NanoConfig = COSMOS3_NANO,
) -> None:
    missing = [name for name in required_transformer_tensor_names(profile) if name not in state]
    if missing:
        raise KeyError("Cosmos3-Nano checkpoint is missing tensors: " + ", ".join(missing))


def _array(state: Mapping[str, object], name: str) -> np.ndarray:
    value = state[name]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _parallel_size(parallel_config) -> int:
    if parallel_config is None:
        return 1
    mode = str(getattr(parallel_config, "mode", "single"))
    if mode == "single":
        return 1
    if mode != "context_parallel":
        raise ValueError("Cosmos3-Nano supports single-device or context-parallel builds")
    size = int(getattr(parallel_config, "cp_size", 1))
    if size not in (2, 4, 8):
        raise ValueError("Cosmos3-Nano context parallel size must be 2, 4, or 8")
    if COSMOS3_NANO.num_attention_heads % size or COSMOS3_NANO.num_key_value_heads % size:
        raise ValueError("Cosmos3-Nano CP size must divide both query and KV heads")
    return size


def select_cp_execution_sizes(
    compute_capability: tuple[int, int],
    *,
    requested_cp_size: int,
) -> tuple[int, int]:
    """Return ``(denoiser_cp_size, classifier_free_parallel_size)``.

    Qualified B200 CP4 and CP8 launches split the launcher world into two
    independent denoiser groups, one for each classifier-free guidance
    branch. Other devices and single-device builds retain their existing
    execution topology.
    """

    if requested_cp_size not in (1, 2, 4, 8):
        raise ValueError("Cosmos3-Nano requested CP size must be 1, 2, 4, or 8")
    if compute_capability == (10, 0) and requested_cp_size in (4, 8):
        return requested_cp_size // 2, 2
    return requested_cp_size, 1


def select_cp_vision_query_chunk_size(
    compute_capability: tuple[int, int],
    *,
    cp_size: int,
    local_vision_length: int,
) -> int | None:
    """Select the exact full-query guard for B200 distributed graphs.

    Supplying the explicit query length also selects the primitive causal-text
    softmax in ``ulysses_dual_attention``. Without it, TensorRT 11.1 re-fuses
    that region into a kernel requiring 0x40194 bytes of shared memory, above
    B200's 0x38c00-byte per-block ceiling.
    """

    if compute_capability == (10, 0) and cp_size in (2, 4, 8):
        return local_vision_length * cp_size
    return None


def select_attention_decomposition(
    compute_capability: tuple[int, int],
    *,
    cp_size: int,
) -> bool:
    """Allow TensorRT attention fallback only on qualified B200 graphs."""

    return compute_capability == (10, 0) and cp_size in (1, 8)


def select_cp_rank_local_sharding(
    compute_capability: tuple[int, int],
    *,
    cp_size: int,
) -> bool:
    """Enable rank-local input sharding on qualified B200 CFG-split CP graphs."""

    return compute_capability == (10, 0) and cp_size in (4, 8)


def build_cosmos3_transformer_engine(
    transformer_dir: str,
    *,
    profile: Cosmos3NanoConfig = COSMOS3_NANO,
    parallel_config=None,
    verbose: bool = False,
) -> bytes:
    """Build one rank-dynamic SD or Ulysses CP denoiser plan."""

    requested_cp_size = _parallel_size(parallel_config)

    # Reuse the family-owned CUDA query already used to qualify Cosmos3 VAE
    # tactics. Qualified B200 CP4 and CP8 launches split into two denoiser
    # groups so their conditional and unconditional branches run concurrently.
    from .vae_step_builder import _current_cuda_device_profile

    compute_capability, _ = _current_cuda_device_profile()
    cp_size, classifier_free_parallel_size = select_cp_execution_sizes(
        compute_capability,
        requested_cp_size=requested_cp_size,
    )
    text_length = profile.max_text_seq_len
    vision_length = profile.num_vision_tokens
    if text_length % cp_size or vision_length % cp_size:
        raise ValueError("Cosmos3-Nano fixed engine token counts must divide the CP size")
    local_text_length = text_length // cp_size
    local_vision_length = vision_length // cp_size
    vision_query_chunk_size = None
    rank_local_sharding = False

    # Attention decomposition is required by B200 SD and the original CP8
    # topology. The CP4 execution graph keeps its already-qualified fused
    # attention policy.
    allow_attention_decomposition = select_attention_decomposition(
        compute_capability,
        cp_size=cp_size,
    )
    if cp_size > 1:
        vision_query_chunk_size = select_cp_vision_query_chunk_size(
            compute_capability,
            cp_size=cp_size,
            local_vision_length=local_vision_length,
        )
        rank_local_sharding = select_cp_rank_local_sharding(
            compute_capability,
            cp_size=requested_cp_size,
        )

    state = load_component_state_dict(transformer_dir)
    validate_transformer_state_dict(state, profile)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    input_ids = network.add_input("input_ids", trt.int32, (text_length,))
    vision_patches = network.add_input(
        "vision_patches", trt.float32, (vision_length, profile.patch_latent_dim)
    )
    timestep_features = network.add_input(
        "timestep_features", trt.float32, (1, profile.timestep_dim)
    )
    text_cos = network.add_input("text_rotary_cos", trt.float32, (text_length, profile.head_dim))
    text_sin = network.add_input("text_rotary_sin", trt.float32, (text_length, profile.head_dim))
    vision_cos = network.add_input(
        "vision_rotary_cos", trt.float32, (vision_length, profile.head_dim)
    )
    vision_sin = network.add_input(
        "vision_rotary_sin", trt.float32, (vision_length, profile.head_dim)
    )
    generation_mask = network.add_input(
        "generation_attention_mask",
        trt.float32,
        (1, 1, 1, text_length + vision_length),
    )
    context_parallel_rank = None
    if rank_local_sharding:
        context_parallel_rank = network.add_input("context_parallel_rank", trt.int32, (1,))
    text_causal_mask = None
    if cp_size > 1:
        text_causal_mask_array = np.triu(
            np.full(
                (text_length, text_length),
                np.finfo(np.float32).min,
                dtype=np.float32,
            ),
            k=1,
        ).reshape(1, 1, text_length, text_length)
        text_causal_mask = op.constant(network, text_causal_mask_array)

    embedding = op.constant(network, _array(state, "embed_tokens.weight"))
    embedding = op.cast(network, embedding, trt.bfloat16)
    text = network.add_gather(embedding, input_ids, 0).get_output(0)

    vision = op.linear(
        network,
        vision_patches,
        _array(state, "proj_in.weight"),
        _array(state, "proj_in.bias"),
    )
    timestep = op.linear(
        network,
        timestep_features,
        _array(state, "time_embedder.linear_1.weight"),
        _array(state, "time_embedder.linear_1.bias"),
        bf16=False,
    )
    timestep = op.silu(network, timestep)
    timestep = op.linear(
        network,
        timestep,
        _array(state, "time_embedder.linear_2.weight"),
        _array(state, "time_embedder.linear_2.bias"),
        bf16=False,
    )
    timestep = op.cast(network, timestep, trt.bfloat16)
    vision = network.add_elementwise(vision, timestep, trt.ElementWiseOperation.SUM).get_output(0)

    if cp_size > 1:
        if rank_local_sharding:
            if context_parallel_rank is None:
                raise RuntimeError("Cosmos3 rank-local sharding requires a rank input")
            text = op.select_replicated_rows(network, text, context_parallel_rank, cp_size)
            vision = op.select_replicated_rows(network, vision, context_parallel_rank, cp_size)
            text_cos = op.select_replicated_rows(network, text_cos, context_parallel_rank, cp_size)
            text_sin = op.select_replicated_rows(network, text_sin, context_parallel_rank, cp_size)
            vision_cos = op.select_replicated_rows(
                network, vision_cos, context_parallel_rank, cp_size
            )
            vision_sin = op.select_replicated_rows(
                network, vision_sin, context_parallel_rank, cp_size
            )
        else:
            text = op.reduce_scatter_replicated(network, text, cp_size)
            vision = op.reduce_scatter_replicated(network, vision, cp_size)
            text_cos = op.reduce_scatter_replicated(network, text_cos, cp_size)
            text_sin = op.reduce_scatter_replicated(network, text_sin, cp_size)
            vision_cos = op.reduce_scatter_replicated(network, vision_cos, cp_size)
            vision_sin = op.reduce_scatter_replicated(network, vision_sin, cp_size)

    for index in range(profile.num_hidden_layers):
        prefix = f"layers.{index}"
        text_norm = op.rms_norm(
            network,
            text,
            _array(state, f"{prefix}.input_layernorm.weight"),
            profile.hidden_size,
            profile.rms_norm_eps,
        )
        vision_norm = op.rms_norm(
            network,
            vision,
            _array(state, f"{prefix}.input_layernorm_moe_gen.weight"),
            profile.hidden_size,
            profile.rms_norm_eps,
        )

        def _project(stream, generation: bool, sequence_length: int):
            projection_names = (
                ("add_q_proj", "add_k_proj", "add_v_proj")
                if generation
                else ("to_q", "to_k", "to_v")
            )
            q_name, k_name, v_name = projection_names
            q = op.linear(network, stream, _array(state, f"{prefix}.self_attn.{q_name}.weight"))
            k = op.linear(network, stream, _array(state, f"{prefix}.self_attn.{k_name}.weight"))
            v = op.linear(network, stream, _array(state, f"{prefix}.self_attn.{v_name}.weight"))
            q_norm_name = "norm_added_q" if generation else "norm_q"
            k_norm_name = "norm_added_k" if generation else "norm_k"
            q = op.rms_norm_per_head(
                network,
                q,
                _array(state, f"{prefix}.self_attn.{q_norm_name}.weight"),
                sequence_length=sequence_length,
                num_heads=profile.num_attention_heads,
                head_dim=profile.head_dim,
                eps=profile.rms_norm_eps,
            )
            k = op.rms_norm_per_head(
                network,
                k,
                _array(state, f"{prefix}.self_attn.{k_norm_name}.weight"),
                sequence_length=sequence_length,
                num_heads=profile.num_key_value_heads,
                head_dim=profile.head_dim,
                eps=profile.rms_norm_eps,
            )
            return q, k, v

        text_q, text_k, text_v = _project(text_norm, False, local_text_length)
        vision_q, vision_k, vision_v = _project(vision_norm, True, local_vision_length)
        text_q = op.apply_rotate_half_rope(
            network,
            text_q,
            text_cos,
            text_sin,
            sequence_length=local_text_length,
            num_heads=profile.num_attention_heads,
            head_dim=profile.head_dim,
        )
        text_k = op.apply_rotate_half_rope(
            network,
            text_k,
            text_cos,
            text_sin,
            sequence_length=local_text_length,
            num_heads=profile.num_key_value_heads,
            head_dim=profile.head_dim,
        )
        vision_q = op.apply_rotate_half_rope(
            network,
            vision_q,
            vision_cos,
            vision_sin,
            sequence_length=local_vision_length,
            num_heads=profile.num_attention_heads,
            head_dim=profile.head_dim,
        )
        vision_k = op.apply_rotate_half_rope(
            network,
            vision_k,
            vision_cos,
            vision_sin,
            sequence_length=local_vision_length,
            num_heads=profile.num_key_value_heads,
            head_dim=profile.head_dim,
        )

        if cp_size == 1:
            text_context = op.attention(
                network,
                text_q,
                text_k,
                text_v,
                q_sequence_length=text_length,
                kv_sequence_length=text_length,
                num_heads=profile.num_attention_heads,
                num_kv_heads=profile.num_key_value_heads,
                head_dim=profile.head_dim,
                causal=True,
                decomposable=allow_attention_decomposition,
            )
            all_k = network.add_concatenation([text_k, vision_k])
            all_k.axis = 0
            all_v = network.add_concatenation([text_v, vision_v])
            all_v.axis = 0
            vision_context = op.attention(
                network,
                vision_q,
                all_k.get_output(0),
                all_v.get_output(0),
                q_sequence_length=vision_length,
                kv_sequence_length=text_length + vision_length,
                num_heads=profile.num_attention_heads,
                num_kv_heads=profile.num_key_value_heads,
                head_dim=profile.head_dim,
                causal=False,
                mask=generation_mask,
                decomposable=allow_attention_decomposition,
            )
        else:
            text_context, vision_context = op.ulysses_dual_attention(
                network,
                text_q,
                text_k,
                text_v,
                vision_q,
                vision_k,
                vision_v,
                local_text_length=local_text_length,
                local_vision_length=local_vision_length,
                num_heads=profile.num_attention_heads,
                num_kv_heads=profile.num_key_value_heads,
                head_dim=profile.head_dim,
                world_size=cp_size,
                generation_mask=generation_mask,
                text_causal_mask=text_causal_mask,
                vision_query_chunk_size=vision_query_chunk_size,
                allow_attention_decomposition=allow_attention_decomposition,
                # Qualify the packed layout with the B200 CP8 rank-local
                # input path.  Other targets retain per-tensor collectives.
                coalesce_collectives=rank_local_sharding,
            )

        text = op.residual(
            network,
            text,
            op.linear(
                network,
                text_context,
                _array(state, f"{prefix}.self_attn.to_out.weight"),
            ),
        )
        vision = op.residual(
            network,
            vision,
            op.linear(
                network,
                vision_context,
                _array(state, f"{prefix}.self_attn.to_add_out.weight"),
            ),
        )
        text_ffn_input = op.rms_norm(
            network,
            text,
            _array(state, f"{prefix}.post_attention_layernorm.weight"),
            profile.hidden_size,
            profile.rms_norm_eps,
        )
        vision_ffn_input = op.rms_norm(
            network,
            vision,
            _array(state, f"{prefix}.post_attention_layernorm_moe_gen.weight"),
            profile.hidden_size,
            profile.rms_norm_eps,
        )
        text = op.residual(
            network,
            text,
            op.swiglu_mlp(
                network,
                text_ffn_input,
                _array(state, f"{prefix}.mlp.gate_proj.weight"),
                _array(state, f"{prefix}.mlp.up_proj.weight"),
                _array(state, f"{prefix}.mlp.down_proj.weight"),
            ),
        )
        vision = op.residual(
            network,
            vision,
            op.swiglu_mlp(
                network,
                vision_ffn_input,
                _array(state, f"{prefix}.mlp_moe_gen.gate_proj.weight"),
                _array(state, f"{prefix}.mlp_moe_gen.up_proj.weight"),
                _array(state, f"{prefix}.mlp_moe_gen.down_proj.weight"),
            ),
        )

    vision = op.rms_norm(
        network,
        vision,
        _array(state, "norm_moe_gen.weight"),
        profile.hidden_size,
        profile.rms_norm_eps,
    )
    output = op.linear(
        network,
        vision,
        _array(state, "proj_out.weight"),
        _array(state, "proj_out.bias"),
    )
    output = op.cast(network, output, trt.float32)
    if cp_size > 1:
        output = op.add_collective(network, output, trt.CollectiveOperation.ALL_GATHER, cp_size)
    output.name = "noise_prediction_patches"
    network.mark_output(output)

    print(
        "[cosmos3] building dual-stream denoiser: "
        f"layers={profile.num_hidden_layers}, text={text_length}, "
        f"vision={vision_length}, requested_cp={requested_cp_size}, "
        f"denoiser_cp={cp_size}, cfg_parallel={classifier_free_parallel_size}, "
        f"vision_query_chunk={vision_query_chunk_size or 'full'}, "
        f"attention_decomposition={allow_attention_decomposition}, "
        f"rank_local_sharding={rank_local_sharding}, "
        f"coalesced_collectives={rank_local_sharding}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build Cosmos3-Nano denoiser")
    return bytes(plan)

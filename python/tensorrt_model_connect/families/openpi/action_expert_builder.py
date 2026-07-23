# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builder for the OpenPI π0.5 action expert.

The engine consumes the compact prefix K/V cache produced by the OpenPI
prefill engine and executes one Euler flow step.  Keeping the Euler update in
the plan lets the C++ runtime alternate two device buffers without copying the
action trajectory through host memory between denoising steps.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping

import numpy as np

from .model_config import OpenPIProfile


_DROID_TIME_CONDITION_WEIGHT_SHA256 = {
    "projections.time_mlp_in.weight": (
        "85ecc978190f79949312a4a752f478f4b62c78dcd21b3f1dba29524ef832c193"
    ),
    "projections.time_mlp_in.bias": (
        "11d898afc211d8bb47dd2839a15d4bdebb0ee61d73f618d05ed65a8fc9b10701"
    ),
    "projections.time_mlp_out.weight": (
        "9296cdb4f07c09c1c76176b45082666af029bad61753b998c1f95dfa30005512"
    ),
    "projections.time_mlp_out.bias": (
        "f0827b9c8a0913d42ff07eb7dc90af1e0b326232b3943a6f0b25fbe67295c935"
    ),
}

_DROID_ADAPTIVE_MODULATION_WEIGHT_SHA256 = {
    "action.layer.6.pre_ffw_norm.dense.weight": (
        "232cce0d38e54b88025a944b28a9f79bdf19f567788cc26eda48b51f9e59d9e7"
    ),
    "action.layer.6.pre_ffw_norm.dense.bias": (
        "02d543a1031d35b756d68d9a37fd14c0dbc547ec69c61ec659ce572cbc60521d"
    ),
    "action.layer.11.pre_attention_norm.dense.weight": (
        "eb241f79e1a1cdd4e23efcafd294c5546ac8a40d0cb1abfa41d201cc97809708"
    ),
    "action.layer.11.pre_attention_norm.dense.bias": (
        "f727a3fdbbfb2be46d4413b04d72dee62b7a8b34efb14a878f5a57fadb5e1d83"
    ),
    "action.layer.12.pre_ffw_norm.dense.weight": (
        "e54d6f1684b717677e546f1f25a8fa0683cffd2b6b774db01d46c3362dba5005"
    ),
    "action.layer.12.pre_ffw_norm.dense.bias": (
        "458bbf1cf284ae151861f84421d803845480093934dc0c213504ac1a95a8011a"
    ),
    "action.layer.14.pre_ffw_norm.dense.weight": (
        "55bb753eaa9d89f6337eb2286cdb579b339d4b007514534bedec6d22503f6cfd"
    ),
    "action.layer.14.pre_ffw_norm.dense.bias": (
        "746b2d2509799c4b895aab029305c5a17f22cdb8b21c13e8ff9166dfccefaab3"
    ),
}


def _weight(
    weights: Mapping[str, object],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return one mapped weight after enforcing its production shape."""
    if name not in weights:
        raise ValueError(f"OpenPI action engine is missing weight {name!r}")
    value = np.asarray(weights[name])
    if tuple(value.shape) != shape:
        raise ValueError(
            f"OpenPI action weight {name!r} has shape {tuple(value.shape)}, expected {shape}"
        )
    if value.dtype.kind != "f" and value.dtype.name != "bfloat16":
        raise ValueError(f"OpenPI action weight {name!r} must be floating point")
    return np.ascontiguousarray(value)


def required_action_weight_shapes(profile: OpenPIProfile) -> dict[str, tuple[int, ...]]:
    """Return the complete action-plan weight inventory."""
    cfg = profile.action_expert
    shapes: dict[str, tuple[int, ...]] = {
        "projections.action_in.weight": (profile.action_dim, cfg.width),
        "projections.action_in.bias": (cfg.width,),
        "projections.time_mlp_in.weight": (cfg.width, cfg.width),
        "projections.time_mlp_in.bias": (cfg.width,),
        "projections.time_mlp_out.weight": (cfg.width, cfg.width),
        "projections.time_mlp_out.bias": (cfg.width,),
        "projections.action_out.weight": (cfg.width, profile.action_dim),
        "projections.action_out.bias": (profile.action_dim,),
        "action.final_norm.dense.weight": (cfg.width, 3 * cfg.width),
        "action.final_norm.dense.bias": (3 * cfg.width,),
    }
    for layer in range(cfg.depth):
        prefix = f"action.layer.{layer}"
        shapes.update(
            {
                f"{prefix}.pre_attention_norm.dense.weight": (
                    cfg.width,
                    3 * cfg.width,
                ),
                f"{prefix}.pre_attention_norm.dense.bias": (3 * cfg.width,),
                f"{prefix}.attention.q.weight": (
                    cfg.width,
                    cfg.attention_width,
                ),
                f"{prefix}.attention.k.weight": (cfg.width, cfg.kv_width),
                f"{prefix}.attention.v.weight": (cfg.width, cfg.kv_width),
                f"{prefix}.attention.o.weight": (
                    cfg.attention_width,
                    cfg.width,
                ),
                f"{prefix}.pre_ffw_norm.dense.weight": (
                    cfg.width,
                    3 * cfg.width,
                ),
                f"{prefix}.pre_ffw_norm.dense.bias": (3 * cfg.width,),
                f"{prefix}.mlp.gate.weight": (cfg.width, cfg.mlp_dim),
                f"{prefix}.mlp.up.weight": (cfg.width, cfg.mlp_dim),
                f"{prefix}.mlp.down.weight": (cfg.mlp_dim, cfg.width),
            }
        )
    return shapes


def _validate_action_weights(
    weights: Mapping[str, object], profile: OpenPIProfile
) -> dict[str, np.ndarray]:
    return {
        name: _weight(weights, name, shape)
        for name, shape in required_action_weight_shapes(profile).items()
    }


def _uses_exact_droid_final_norm(profile: OpenPIProfile, *, precision: str) -> bool:
    """Return whether the audited fusion.144 contract applies exactly."""
    return (
        profile.name == "pi05_droid"
        and precision == "bf16"
        and profile.action_horizon == 15
        and profile.action_expert.width == 1024
        and np.float32(profile.rms_norm_epsilon) == np.float32(1.0e-6)
    )


def _uses_exact_droid_pre_attention_norm(
    profile: OpenPIProfile,
    *,
    precision: str,
    layer: int,
) -> bool:
    """Return whether the audited fixed-shape pre-attention norm applies."""
    cfg = profile.action_expert
    return (
        0 <= layer < cfg.depth
        and profile.name == "pi05_droid"
        and precision == "bf16"
        and profile.action_horizon == 15
        and cfg.depth == 18
        and cfg.width == 1024
        and np.float32(profile.rms_norm_epsilon) == np.float32(1.0e-6)
    )


def _uses_exact_droid_action_contract(
    profile: OpenPIProfile,
    *,
    precision: str,
) -> bool:
    """Return whether the fully qualified fixed-shape DROID contract applies."""
    cfg = profile.action_expert
    prefix = profile.prefix
    return (
        profile.name == "pi05_droid"
        and precision == "bf16"
        and profile.prefix_length == 968
        and profile.action_horizon == 15
        and profile.action_dim == 32
        and profile.denoise_steps == 10
        and cfg.depth == 18
        and cfg.width == 1024
        and cfg.mlp_dim == 4096
        and cfg.num_heads == 8
        and cfg.num_kv_heads == 1
        and cfg.head_dim == 256
        and prefix.depth == 18
        and prefix.width == 2048
        and prefix.num_heads == 8
        and prefix.num_kv_heads == 1
        and prefix.head_dim == 256
        and np.float32(profile.rms_norm_epsilon) == np.float32(1.0e-6)
    )


def _uses_exact_droid_action_mlp_closure(
    profile: OpenPIProfile,
    *,
    precision: str,
    layer: int,
) -> bool:
    """Return whether the qualified exact DROID MLP closure applies."""
    return 0 <= layer < profile.action_expert.depth and _uses_exact_droid_action_contract(
        profile, precision=precision
    )


def _uses_exact_droid_action_output_projection(
    profile: OpenPIProfile,
    *,
    precision: str,
) -> bool:
    """Return whether the audited padded action-output GEMM applies."""
    return _uses_exact_droid_action_contract(profile, precision=precision)


def _uses_exact_droid_time_condition_corrections(
    profile: OpenPIProfile,
    *,
    precision: str,
) -> bool:
    """Return whether the pinned DROID BF16 condition corrections apply."""
    return _uses_exact_droid_action_contract(profile, precision=precision)


def _uses_exact_droid_adaptive_modulation_corrections(
    profile: OpenPIProfile,
    *,
    precision: str,
) -> bool:
    """Return whether the pinned DROID modulation corrections apply."""
    return _uses_exact_droid_time_condition_corrections(profile, precision=precision)


def _require_exact_droid_time_condition_weights(
    weights: Mapping[str, object],
) -> None:
    """Fail unless the correction seam sees the audited BF16 checkpoint."""
    mismatches = []
    for name, expected in _DROID_TIME_CONDITION_WEIGHT_SHA256.items():
        value = weights.get(name)
        if value is None:
            mismatches.append(f"{name}=missing")
            continue
        payload = np.ascontiguousarray(value, dtype=np.float32).tobytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            mismatches.append(f"{name}={actual}")
    if mismatches:
        raise ValueError(
            "OpenPI DROID time-condition corrections require the audited "
            "BF16-rounded checkpoint weights: " + ", ".join(mismatches)
        )


def _require_exact_droid_adaptive_modulation_weights(
    weights: Mapping[str, object],
) -> None:
    """Fail unless sparse modulation corrections see their audited weights."""
    mismatches = []
    for name, expected in _DROID_ADAPTIVE_MODULATION_WEIGHT_SHA256.items():
        value = weights.get(name)
        if value is None:
            mismatches.append(f"{name}=missing")
            continue
        payload = np.ascontiguousarray(value, dtype=np.float32).tobytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            mismatches.append(f"{name}={actual}")
    if mismatches:
        raise ValueError(
            "OpenPI DROID adaptive-modulation corrections require the audited "
            "BF16-rounded checkpoint weights: " + ", ".join(mismatches)
        )


def _uses_action_attention_context(
    profile: OpenPIProfile,
    *,
    precision: str,
    layer: int,
) -> bool:
    """Return whether the qualified exact DROID attention topology applies."""
    cfg = profile.action_expert
    return (
        0 <= layer < cfg.depth
        and profile.name == "pi05_droid"
        and precision == "bf16"
        and profile.prefix_length == 968
        and profile.action_horizon == 15
        and cfg.depth > 1
        and cfg.width == 1024
        and cfg.num_heads == 8
        and cfg.num_kv_heads == 1
        and cfg.head_dim == 256
    )


def _combined_attention_mask(network, prefix_mask, profile: OpenPIProfile, *, graph_ops, trt):
    """Build bool ``[1,1,H,P+H]`` suffix-to-prefix/full-suffix mask."""
    horizon = profile.action_horizon
    prefix_length = profile.prefix_length
    prefix_shuffle = network.add_shuffle(prefix_mask)
    prefix_shuffle.reshape_dims = (1, 1, 1, prefix_length)
    query_rows = graph_ops.constant(
        network,
        np.ones((1, 1, horizon, 1), dtype=np.bool_),
        dtype=trt.bool,
    )
    prefix_part = network.add_elementwise(
        prefix_shuffle.get_output(0),
        query_rows,
        trt.ElementWiseOperation.AND,
    ).get_output(0)
    suffix_part = graph_ops.constant(
        network,
        np.ones((1, 1, horizon, horizon), dtype=np.bool_),
        dtype=trt.bool,
    )
    combined = network.add_concatenation([prefix_part, suffix_part])
    combined.axis = 3
    return combined.get_output(0)


def build_action_expert_engine(
    profile: OpenPIProfile,
    weights: Mapping[str, object],
    *,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build the one-step π0.5 action expert with TensorRT primitives only."""
    # Keep TensorRT lazy so profile/config inspection works in host-only build
    # environments.
    from . import graph_ops

    trt = graph_ops.trt
    mapped = _validate_action_weights(weights, profile)
    cfg = profile.action_expert
    horizon = profile.action_horizon
    prefix_length = profile.prefix_length
    _, work_dtype = graph_ops.precision_types(precision)
    builder, network, builder_config = graph_ops.create_builder_context(
        verbose=verbose,
        workspace_bytes=8 << 30,
    )

    noisy_actions = network.add_input(
        "noisy_actions", trt.float32, (1, horizon, profile.action_dim)
    )
    timestep = network.add_input("timestep", trt.float32, (1,))
    step_size = network.add_input("step_size", trt.float32, (1,))
    prefix_mask = network.add_input("prefix_mask", trt.bool, (1, prefix_length))
    suffix_position_ids = network.add_input("suffix_position_ids", trt.int32, (1, horizon))

    prefix_cache: list[tuple[object, object]] = []
    for layer in range(cfg.depth):
        prefix_k = network.add_input(
            f"prefix_k_{layer}",
            work_dtype,
            (1, prefix_length, cfg.num_kv_heads, cfg.head_dim),
        )
        prefix_v = network.add_input(
            f"prefix_v_{layer}",
            work_dtype,
            (1, prefix_length, cfg.num_kv_heads, cfg.head_dim),
        )
        prefix_k_shuffle = network.add_shuffle(prefix_k)
        prefix_k_shuffle.first_transpose = (0, 2, 1, 3)
        prefix_v_shuffle = network.add_shuffle(prefix_v)
        prefix_v_shuffle.first_transpose = (0, 2, 1, 3)
        prefix_cache.append((prefix_k_shuffle.get_output(0), prefix_v_shuffle.get_output(0)))

    # NNX projections outside PaliGemma run at the FP32 boundary. The expert
    # module casts action tokens to its configured embedding dtype on entry.
    hidden = graph_ops.linear(
        network,
        noisy_actions,
        mapped["projections.action_in.weight"],
        mapped["projections.action_in.bias"],
        dtype=trt.float32,
    )
    hidden = graph_ops.cast(network, hidden, work_dtype)

    condition = graph_ops.add_sinusoidal_embedding(network, timestep, cfg.width)
    condition = graph_ops.linear(
        network,
        condition,
        mapped["projections.time_mlp_in.weight"],
        mapped["projections.time_mlp_in.bias"],
        dtype=trt.float32,
    )
    condition = graph_ops._xla_fp32_silu(network, condition)
    condition = graph_ops.linear(
        network,
        condition,
        mapped["projections.time_mlp_out.weight"],
        mapped["projections.time_mlp_out.bias"],
        dtype=trt.float32,
    )
    condition = graph_ops._xla_fp32_silu(network, condition)
    if _uses_exact_droid_time_condition_corrections(profile, precision=precision):
        _require_exact_droid_time_condition_weights(mapped)
        condition = graph_ops.correct_droid_time_condition_bf16_boundaries(
            network,
            condition,
            timestep,
        )
    use_droid_modulation_corrections = _uses_exact_droid_adaptive_modulation_corrections(
        profile,
        precision=precision,
    )
    if use_droid_modulation_corrections:
        _require_exact_droid_adaptive_modulation_weights(mapped)
    attention_mask = _combined_attention_mask(
        network, prefix_mask, profile, graph_ops=graph_ops, trt=trt
    )

    for layer in range(cfg.depth):
        layer_prefix = f"action.layer.{layer}"
        if _uses_exact_droid_pre_attention_norm(
            profile,
            precision=precision,
            layer=layer,
        ):
            normed, gate = graph_ops.pre_attention_adaptive_rms_norm(
                network,
                hidden,
                condition,
                mapped[f"{layer_prefix}.pre_attention_norm.dense.weight"],
                mapped[f"{layer_prefix}.pre_attention_norm.dense.bias"],
                epsilon=profile.rms_norm_epsilon,
                layer_name=f"openpi_pre_attention_rms_norm_layer_{layer}",
                droid_timestep=timestep if use_droid_modulation_corrections else None,
                droid_layer=layer if use_droid_modulation_corrections else None,
            )
        else:
            normed, gate = graph_ops.adaptive_rms_norm(
                network,
                hidden,
                condition,
                mapped[f"{layer_prefix}.pre_attention_norm.dense.weight"],
                mapped[f"{layer_prefix}.pre_attention_norm.dense.bias"],
                epsilon=profile.rms_norm_epsilon,
            )
        q_weight = np.ascontiguousarray(
            mapped[f"{layer_prefix}.attention.q.weight"]
            .reshape(cfg.width, cfg.num_heads, cfg.head_dim)
            .transpose(1, 0, 2)
        )
        kv_weights = [
            np.ascontiguousarray(
                mapped[f"{layer_prefix}.attention.{kind}.weight"]
                .reshape(cfg.width, cfg.num_kv_heads, cfg.head_dim)
                .transpose(1, 0, 2)
            )
            for kind in ("k", "v")
        ]
        q_tokens = network.add_einsum(
            [normed, graph_ops.constant(network, q_weight, dtype=work_dtype)],
            "btd,ndh->btnh",
        ).get_output(0)
        kv_tokens = [
            network.add_einsum(
                [normed, graph_ops.constant(network, weight, dtype=work_dtype)],
                "btd,kdh->btkh",
            ).get_output(0)
            for weight in kv_weights
        ]
        q_shuffle = network.add_shuffle(q_tokens)
        q_shuffle.first_transpose = (0, 2, 1, 3)
        k_shuffle = network.add_shuffle(kv_tokens[0])
        k_shuffle.first_transpose = (0, 2, 1, 3)
        v_shuffle = network.add_shuffle(kv_tokens[1])
        v_shuffle.first_transpose = (0, 2, 1, 3)
        q = q_shuffle.get_output(0)
        suffix_k = k_shuffle.get_output(0)
        suffix_v = v_shuffle.get_output(0)
        q, suffix_k = graph_ops.apply_action_rope_qk(
            network,
            q,
            suffix_k,
            suffix_position_ids,
            max_positions=prefix_length + horizon,
            head_dim=cfg.head_dim,
        )
        cached_k, cached_v = prefix_cache[layer]
        k_concat = network.add_concatenation([cached_k, suffix_k])
        k_concat.axis = 2
        v_concat = network.add_concatenation([cached_v, suffix_v])
        v_concat.axis = 2
        if _uses_action_attention_context(
            profile,
            precision=precision,
            layer=layer,
        ):
            attended = graph_ops.action_attention_context(
                network,
                q,
                k_concat.get_output(0),
                v_concat.get_output(0),
                attention_mask,
                layer_name=f"openpi_action_attention_context_layer_{layer}",
            )
        else:
            attended = graph_ops.attention_from_rotated(
                network,
                q,
                k_concat.get_output(0),
                v_concat.get_output(0),
                attention_mask,
                num_query_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                fp32_logits=True,
            )
        attended_heads = network.add_shuffle(attended)
        attended_heads.reshape_dims = (
            1,
            horizon,
            cfg.num_heads,
            cfg.head_dim,
        )
        o_weight = np.ascontiguousarray(
            mapped[f"{layer_prefix}.attention.o.weight"].reshape(
                cfg.num_heads, cfg.head_dim, cfg.width
            )
        )
        attended = network.add_einsum(
            [
                attended_heads.get_output(0),
                graph_ops.constant(network, o_weight, dtype=work_dtype),
            ],
            "btnh,nhd->btd",
        ).get_output(0)
        attention_residual = hidden
        attention_update = attended
        attention_gate = gate
        hidden = graph_ops.gated_residual(
            network,
            attention_residual,
            attention_update,
            attention_gate,
            rounding_output_name=f"rounding_attention_update_{layer}",
        )

        normed, gate = graph_ops.post_attention_adaptive_rms_norm(
            network,
            attention_residual,
            attention_update,
            attention_gate,
            condition,
            mapped[f"{layer_prefix}.pre_ffw_norm.dense.weight"],
            mapped[f"{layer_prefix}.pre_ffw_norm.dense.bias"],
            epsilon=profile.rms_norm_epsilon,
            droid_timestep=timestep if use_droid_modulation_corrections else None,
            droid_layer=layer if use_droid_modulation_corrections else None,
        )
        if _uses_exact_droid_action_mlp_closure(
            profile,
            precision=precision,
            layer=layer,
        ):
            hidden = graph_ops.action_layer0_mlp_closure(
                network,
                hidden,
                normed,
                gate,
                mapped[f"{layer_prefix}.mlp.gate.weight"],
                mapped[f"{layer_prefix}.mlp.up.weight"],
                mapped[f"{layer_prefix}.mlp.down.weight"],
                layer_name=f"openpi_action_mlp_closure_layer_{layer}",
            )
        else:
            gate_projection = network.add_einsum(
                [
                    normed,
                    graph_ops.constant(
                        network,
                        mapped[f"{layer_prefix}.mlp.gate.weight"],
                        dtype=work_dtype,
                    ),
                ],
                "btd,df->btf",
            ).get_output(0)
            gated = graph_ops.gelu_tanh(
                network,
                gate_projection,
                rounding_output_name=f"rounding_mlp_gelu_cubic_{layer}",
            )
            up = network.add_einsum(
                [
                    normed,
                    graph_ops.constant(
                        network,
                        mapped[f"{layer_prefix}.mlp.up.weight"],
                        dtype=work_dtype,
                    ),
                ],
                "btd,df->btf",
            ).get_output(0)
            fused = network.add_elementwise(gated, up, trt.ElementWiseOperation.PROD).get_output(0)
            mlp = network.add_einsum(
                [
                    fused,
                    graph_ops.constant(
                        network,
                        mapped[f"{layer_prefix}.mlp.down.weight"],
                        dtype=work_dtype,
                    ),
                ],
                "btf,fd->btd",
            ).get_output(0)
            hidden = graph_ops.gated_residual(
                network,
                hidden,
                mlp,
                gate,
                rounding_output_name=f"rounding_mlp_update_{layer}",
            )
    if _uses_exact_droid_final_norm(profile, precision=precision):
        hidden = graph_ops.final_adaptive_rms_norm(
            network,
            hidden,
            condition,
            mapped["action.final_norm.dense.weight"],
            mapped["action.final_norm.dense.bias"],
            epsilon=profile.rms_norm_epsilon,
        )
    else:
        hidden, _ = graph_ops.adaptive_rms_norm(
            network,
            hidden,
            condition,
            mapped["action.final_norm.dense.weight"],
            mapped["action.final_norm.dense.bias"],
            epsilon=profile.rms_norm_epsilon,
        )
    if _uses_exact_droid_action_output_projection(profile, precision=precision):
        velocity_bf16 = graph_ops.action_output_projection(
            network,
            hidden,
            mapped["projections.action_out.weight"],
            mapped["projections.action_out.bias"],
        )
    else:
        velocity_bf16 = network.add_einsum(
            [
                hidden,
                graph_ops.constant(
                    network,
                    mapped["projections.action_out.weight"],
                    dtype=work_dtype,
                ),
            ],
            "btd,df->btf",
        ).get_output(0)
        output_bias = graph_ops.constant(
            network,
            mapped["projections.action_out.bias"].reshape(1, 1, profile.action_dim),
            dtype=work_dtype,
        )
        velocity_bf16 = network.add_elementwise(
            velocity_bf16,
            output_bias,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
    # Upstream action_out_proj returns BF16. The Euler state is FP32, so expose
    # the rounded velocity through an FP32 binding before the update.
    velocity = graph_ops.cast(network, velocity_bf16, trt.float32)
    velocity.name = "velocity"
    network.mark_output(velocity)

    dt_shuffle = network.add_shuffle(step_size)
    dt_shuffle.reshape_dims = (1, 1, 1)
    step_bf16 = graph_ops.cast(network, dt_shuffle.get_output(0), work_dtype)
    delta_bf16 = network.add_elementwise(
        velocity_bf16,
        step_bf16,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    delta = graph_ops.cast(network, delta_bf16, trt.float32)
    next_actions = network.add_elementwise(
        noisy_actions, delta, trt.ElementWiseOperation.SUM
    ).get_output(0)
    next_actions.name = "next_actions"
    network.mark_output(next_actions)

    if verbose:
        print(
            "[trtmc build] Building OpenPI action expert "
            f"(profile={profile.name}, precision={precision}, layers={cfg.depth}, "
            f"horizon={horizon}, prefix={prefix_length})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT OpenPI action engine build failed")
    return bytes(plan)

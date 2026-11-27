# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint layout of ESMFold2: the exact tensor set, validated from the file headers before any load."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .config import EsmFold2Config


class CheckpointError(ValueError):
    """The safetensors checkpoint does not match the configuration."""


def _pair_update_layer(prefix: str, config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    c = config.pair_size
    shapes: dict[str, tuple[int, ...]] = {}
    for side in ("tri_mul_out", "tri_mul_in"):
        for norm in ("norm_start", "norm_mix"):
            shapes[f"{prefix}.{side}.{norm}.weight"] = (c,)
            shapes[f"{prefix}.{side}.{norm}.bias"] = (c,)
        shapes[f"{prefix}.{side}.proj_bundle.weight"] = (4 * c, c)
        shapes[f"{prefix}.{side}.proj_emit.weight"] = (c, c)
        shapes[f"{prefix}.{side}.proj_gate.weight"] = (c, c)
    shapes.update(_transition(f"{prefix}.pair_transition", c, config.pair_transition_size))
    return shapes


def _transition(prefix: str, width: int, intermediate: int) -> dict[str, tuple[int, ...]]:
    return {
        f"{prefix}.norm.weight": (width,),
        f"{prefix}.norm.bias": (width,),
        f"{prefix}.mlp.gate_up_proj.weight": (2 * intermediate, width),
        f"{prefix}.mlp.down_proj.weight": (width, intermediate),
    }


def _atom_layers(prefix: str, width: int, intermediate: int, count: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(count):
        base = f"{prefix}.layers.{layer}"
        shapes[f"{base}.adaln_linear.weight"] = (6 * width, width)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            shapes[f"{base}.self_attn.{name}.weight"] = (width, width)
        shapes[f"{base}.mlp.gate_up_proj.weight"] = (2 * intermediate, width)
        shapes[f"{base}.mlp.down_proj.weight"] = (width, intermediate)
    return shapes


def _atom_encoder(prefix: str, config: EsmFold2Config, *, diffusion: bool) -> dict[str, tuple[int, ...]]:
    stack = config.diffusion_atoms if diffusion else config.trunk_atoms
    shapes = {
        f"{prefix}.atom_linear.weight": (stack.hidden_size, config.atom_feature_dim),
        f"{prefix}.atom_norm.weight": (stack.hidden_size,),
        f"{prefix}.atom_norm.bias": (stack.hidden_size,),
        f"{prefix}.atom_to_token_linear.weight": (stack.output_dim, stack.hidden_size),
    }
    shapes.update(_atom_layers(prefix, stack.hidden_size, stack.intermediate_size, stack.num_layers))
    return shapes


def _esmc(config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    e = config.esmc
    shapes = {"esmc.embed_tokens.weight": (e.vocab_size, e.hidden_size), "esmc.norm.weight": (e.hidden_size,)}
    for layer in range(e.num_layers):
        base = f"esmc.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{base}.{norm}.weight"] = (e.hidden_size,)
            shapes[f"{base}.{norm}.bias"] = (e.hidden_size,)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[f"{base}.self_attn.{name}.weight"] = (e.hidden_size, e.hidden_size)
        shapes[f"{base}.self_attn.q_norm.weight"] = (e.hidden_size,)
        shapes[f"{base}.self_attn.k_norm.weight"] = (e.hidden_size,)
        shapes[f"{base}.mlp.gate_proj.weight"] = (e.intermediate_size, e.hidden_size)
        shapes[f"{base}.mlp.up_proj.weight"] = (e.intermediate_size, e.hidden_size)
        shapes[f"{base}.mlp.down_proj.weight"] = (e.hidden_size, e.intermediate_size)
    return shapes


def _msa(config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    c, m = config.pair_size, config.msa_hidden_size
    inner = config.msa_heads * config.msa_head_dim
    shapes = {
        "msa_encoder.embed.weight": (m, config.num_res_types + 2),
        "msa_encoder.project_inputs.weight": (m, config.single_inputs_size),
    }
    for layer in range(config.msa_layers):
        base = f"msa_encoder.layers.{layer}"
        shapes.update(_pair_update_layer(base, config))
        opm = f"{base}.outer_product_mean"
        shapes[f"{opm}.norm.weight"] = (m,)
        shapes[f"{opm}.norm.bias"] = (m,)
        shapes[f"{opm}.input_proj.weight"] = (2 * config.msa_outer_size, m)
        shapes[f"{opm}.output_proj.weight"] = (c, config.msa_outer_size**2)
        shapes[f"{opm}.output_proj.bias"] = (c,)
        if layer == config.msa_layers - 1:
            continue
        pwa = f"{base}.msa_pair_weighted_averaging"
        shapes[f"{pwa}.norm_single.weight"] = (m,)
        shapes[f"{pwa}.norm_single.bias"] = (m,)
        shapes[f"{pwa}.bias_norm.weight"] = (c,)
        shapes[f"{pwa}.bias_norm.bias"] = (c,)
        shapes[f"{pwa}.bias_proj.weight"] = (config.msa_heads, c)
        shapes[f"{pwa}.v_proj.weight"] = (inner, m)
        shapes[f"{pwa}.gate_proj.weight"] = (inner, m)
        shapes[f"{pwa}.o_proj.weight"] = (m, inner)
        shapes.update(_transition(f"{base}.msa_transition", m, config.msa_intermediate_size))
    return shapes


def _diffusion(config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    c, h = config.pair_size, config.diffusion_hidden_size
    s, f = config.single_inputs_size, config.fourier_dim
    atom = config.diffusion_atoms.hidden_size
    cond = "structure_head.conditioning"
    shapes = {
        f"{cond}.fourier.frequencies": (f,),
        f"{cond}.fourier.phases": (f,),
        f"{cond}.noise_norm.weight": (f,),
        f"{cond}.noise_norm.bias": (f,),
        f"{cond}.noise_proj.weight": (h, f),
        f"{cond}.pair_input_norm.weight": (2 * c,),
        f"{cond}.pair_input_norm.bias": (2 * c,),
        f"{cond}.pair_proj.weight": (c, 2 * c),
        f"{cond}.single_input_norm.weight": (s,),
        f"{cond}.single_input_norm.bias": (s,),
        f"{cond}.single_proj.weight": (h, s),
        "structure_head.coords_linear.weight": (atom, 6),
        "structure_head.single_step_norm.weight": (h,),
        "structure_head.single_step_norm.bias": (h,),
        "structure_head.single_to_token.weight": (h, h),
        "structure_head.token_norm.weight": (h,),
        "structure_head.token_norm.bias": (h,),
        "structure_head.atom_decoder.token_to_atom_linear.weight": (atom, h),
        "structure_head.atom_decoder.norm.weight": (atom,),
        "structure_head.atom_decoder.norm.bias": (atom,),
        "structure_head.atom_decoder.output_linear.weight": (3, atom),
    }
    for index in (0, 1):
        shapes.update(_transition(f"{cond}.pair_transition_{index}", c, config.diffusion_pair_intermediate_size))
        shapes.update(_transition(f"{cond}.single_transition_{index}", h, config.diffusion_intermediate_size))
    shapes.update(_atom_encoder("structure_head.atom_encoder", config, diffusion=True))
    shapes.update(
        _atom_layers(
            "structure_head.atom_decoder",
            config.diffusion_atoms.hidden_size,
            config.diffusion_atoms.intermediate_size,
            config.diffusion_atoms.num_layers,
        )
    )
    for layer in range(config.diffusion_layers):
        base = f"structure_head.token_transformer.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{base}.{norm}.cond_norm.weight"] = (h,)
            shapes[f"{base}.{norm}.gate_proj.weight"] = (h, h)
            shapes[f"{base}.{norm}.gate_proj.bias"] = (h,)
            shapes[f"{base}.{norm}.shift_proj.weight"] = (h, h)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            shapes[f"{base}.self_attn.{name}.weight"] = (h, h)
        shapes[f"{base}.self_attn.q_proj.bias"] = (h,)
        for gate in ("attn_gate", "mlp_gate"):
            shapes[f"{base}.{gate}.weight"] = (h, h)
            shapes[f"{base}.{gate}.bias"] = (h,)
        shapes[f"{base}.mlp.gate_up_proj.weight"] = (2 * config.diffusion_intermediate_size, h)
        shapes[f"{base}.mlp.down_proj.weight"] = (h, config.diffusion_intermediate_size)
        shapes[f"{base}.pair_norm.weight"] = (c,)
        shapes[f"{base}.pair_norm.bias"] = (c,)
        shapes[f"{base}.pair_bias_proj.weight"] = (config.diffusion_heads, c)
    return shapes


def _confidence(config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    c, s, d = config.pair_size, config.single_inputs_size, config.hidden_size
    head = "confidence_head"
    shapes = {
        f"{head}.boundaries": (config.confidence_distance_bins - 1,),
        f"{head}.dist_bin_pairwise_embed.weight": (config.confidence_distance_bins, c),
        f"{head}.input_embedder.single_inputs_norm.weight": (s,),
        f"{head}.input_embedder.single_inputs_norm.bias": (s,),
        f"{head}.input_embedder.pair_norm.weight": (c,),
        f"{head}.input_embedder.pair_norm.bias": (c,),
        f"{head}.row_attention_pooling.attn_proj.weight": (1, c),
        f"{head}.row_attention_pooling.out_proj.weight": (d, c),
        f"{head}.plddt_layernorm.weight": (d,),
        f"{head}.plddt_layernorm.bias": (d,),
        f"{head}.plddt_weight": (config.max_atoms_per_token, d, config.plddt_bins),
        f"{head}.pae_layernorm.weight": (c,),
        f"{head}.pae_layernorm.bias": (c,),
        f"{head}.pae_head.weight": (config.pae_bins, c),
        f"{head}.pde_layernorm.weight": (c,),
        f"{head}.pde_layernorm.bias": (c,),
        f"{head}.pde_head.weight": (config.pde_bins, c),
        f"{head}.resolved_layernorm.weight": (d,),
        f"{head}.resolved_layernorm.bias": (d,),
        f"{head}.resolved_weight": (config.max_atoms_per_token, d, 2),
    }
    for name in ("single_to_pair", "single_to_pair_transpose", "single_to_pair_prod_in1", "single_to_pair_prod_in2"):
        shapes[f"{head}.input_embedder.{name}.weight"] = (c, s)
    shapes[f"{head}.input_embedder.single_to_pair_prod_out.weight"] = (c, c)
    for layer in range(config.confidence_layers):
        shapes.update(_pair_update_layer(f"{head}.folding_trunk.layers.{layer}", config))
    return shapes


def expected_tensors(config: EsmFold2Config) -> dict[str, tuple[int, ...]]:
    """The complete tensor set of a released checkpoint with this configuration, name -> shape."""

    c, s = config.pair_size, config.single_inputs_size
    shapes = _esmc(config)
    shapes.update(_atom_encoder("input_embedder.atom_encoder", config, diffusion=False))
    shapes.update(
        {
            "input_embedder.pair_init_1.weight": (c, s),
            "input_embedder.pair_init_2.weight": (c, s),
            "input_embedder.rel_pos.embed.weight": (
                c,
                2 * (2 * config.residue_bins + 2) + 1 + (2 * config.chain_bins + 2),
            ),
            "input_embedder.token_bonds.weight": (c, 1),
            "language_model.layer_weights": (config.lm_states,),
            "language_model.pair_input_norm.weight": (config.esmc.hidden_size,),
            "language_model.pair_input_norm.bias": (config.esmc.hidden_size,),
            "language_model.pair_proj.weight": (c, config.esmc.hidden_size),
            "language_model.single_to_pair.downproject.weight": (c, c),
            "language_model.single_to_pair.downproject.bias": (c,),
            "language_model.single_to_pair.output_fc1.weight": (c, 2 * c),
            "language_model.single_to_pair.output_fc1.bias": (c,),
            "language_model.single_to_pair.output_fc2.weight": (c, c),
            "language_model.single_to_pair.output_fc2.bias": (c,),
            "language_model.pair_output_norm.weight": (c,),
            "language_model.pair_output_norm.bias": (c,),
            "parcae.input_norm.weight": (c,),
            "parcae.input_norm.bias": (c,),
            "parcae.log_state_decay": (c,),
            "parcae.log_delta": (c,),
            "parcae.input_matrix_continuous": (c, c),
            "parcae.out_proj.weight": (c, c),
        }
    )
    for layer in range(config.trunk_layers):
        shapes.update(_pair_update_layer(f"folding_trunk.layers.{layer}", config))
    for layer in range(config.lm_layers):
        shapes.update(_pair_update_layer(f"lm_encoder.layers.{layer}", config))
    for layer in range(config.coda_layers):
        shapes.update(_pair_update_layer(f"parcae.output_stack.layers.{layer}", config))
    if config.msa_enabled:
        shapes.update(_msa(config))
    shapes.update(_diffusion(config))
    shapes.update(_confidence(config))
    return shapes


class Checkpoint:
    """A validated, lazily read safetensors checkpoint."""

    def __init__(self, model_dir: Path, config: EsmFold2Config) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.model_dir = Path(model_dir)
        self._files = self._shard_map()
        self._expected = expected_tensors(config)
        self._validate_headers()

    def _shard_map(self) -> dict[str, Path]:
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise CheckpointError("model.safetensors.index.json has no weight_map")
            files: dict[str, Path] = {}
            for name, shard in weight_map.items():
                if not isinstance(name, str) or not isinstance(shard, str) or "/" in shard or shard.startswith("."):
                    raise CheckpointError(f"invalid weight_map entry {name!r} -> {shard!r}")
                files[name] = self.model_dir / shard
            return files
        single = self.model_dir / "model.safetensors"
        if not single.is_file():
            raise CheckpointError(f"no safetensors checkpoint in {self.model_dir}")
        with self._safe_open(str(single), framework="numpy") as handle:
            return {name: single for name in handle.keys()}

    def _validate_headers(self) -> None:
        names = set(self._files)
        missing = sorted(set(self._expected) - names)
        if missing:
            raise CheckpointError(f"checkpoint lacks {len(missing)} tensors, first: {missing[:5]}")
        # A tensor the released graph does not have means another architecture: refuse it.
        extra = sorted(names - set(self._expected) - _UNUSED_TENSORS)
        if extra:
            raise CheckpointError(f"checkpoint has {len(extra)} unknown tensors, first: {extra[:5]}")
        by_file: dict[Path, list[str]] = {}
        for name in self._expected:
            by_file.setdefault(self._files[name], []).append(name)
        for path, members in by_file.items():
            with self._safe_open(str(path), framework="numpy") as handle:
                present = set(handle.keys())
                for name in members:
                    if name not in present:
                        raise CheckpointError(f"{path.name} does not contain {name}")
                    view = handle.get_slice(name)
                    dtype, shape = view.get_dtype(), tuple(view.get_shape())
                    if dtype != "F32":
                        raise CheckpointError(f"{name} must be F32, got {dtype}")
                    if shape != self._expected[name]:
                        raise CheckpointError(f"{name} has shape {shape}, expected {self._expected[name]}")

    def tensor(self, name: str) -> np.ndarray:
        """Read one validated tensor as a C-contiguous float32 array."""

        if name not in self._expected:
            raise CheckpointError(f"{name} is not a tensor of the validated graph")
        with self._safe_open(str(self._files[name]), framework="numpy") as handle:
            value = handle.get_tensor(name)
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.shape != self._expected[name]:
            raise CheckpointError(f"{name} changed shape between header and data")
        if not np.isfinite(array).all():
            raise CheckpointError(f"{name} holds non-finite values")
        return array

    def tensors(self, prefix: str) -> Iterator[tuple[str, np.ndarray]]:
        for name in sorted(self._expected):
            if name.startswith(prefix):
                yield name, self.tensor(name)


# Tensors of the released checkpoint that the structure_prediction path never reads, so their shapes do not depend on
# the parsed configuration: the distogram head (its logits are not part of the Task result).
_UNUSED_TENSORS: frozenset[str] = frozenset({"distogram_head.weight", "distogram_head.bias"})


def describe_tensors(config: EsmFold2Config) -> dict[str, Any]:
    """Counts used in the build report."""

    expected = expected_tensors(config)
    return {
        "tensors": len(expected),
        "parameters": int(sum(int(np.prod(shape)) for shape in expected.values())),
    }

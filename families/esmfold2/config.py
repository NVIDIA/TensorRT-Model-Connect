# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict reader of the ESMFold2 ``config.json``.

Every value the graphs or the runtime use is read here with its exact type and checked against the relations the
architecture needs. A value the family cannot honor is rejected, never ignored.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """The checkpoint configuration is not one this family builds."""


def _object(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    if key not in parent:
        raise ConfigError(f"{where} is missing {key!r}")
    value = parent[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be an object")
    return value


def _int(parent: dict[str, Any], key: str, where: str, *, minimum: int = 1) -> int:
    if key not in parent:
        raise ConfigError(f"{where} is missing {key!r}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    if value < minimum or value > 2**31 - 1:
        raise ConfigError(f"{where}.{key} is out of range: {value}")
    return value


def _float(parent: dict[str, Any], key: str, where: str, *, positive: bool = False) -> float:
    if key not in parent:
        raise ConfigError(f"{where} is missing {key!r}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or abs(number) > 3.0e38:
        raise ConfigError(f"{where}.{key} must be a finite float32 value, got {value!r}")
    if positive and number <= 0.0:
        raise ConfigError(f"{where}.{key} must be positive, got {value!r}")
    return number


def _bool(parent: dict[str, Any], key: str, where: str) -> bool:
    if key not in parent:
        raise ConfigError(f"{where} is missing {key!r}")
    value = parent[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be a boolean, got {value!r}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class AtomStackConfig:
    hidden_size: int
    output_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    spatial_rope_base: float
    spatial_pairs_per_axis: int
    uid_pairs: int
    uid_rope_base: float

    @property
    def inactive_rope_pairs(self) -> int:
        return self.head_dim // 2 - 3 * self.spatial_pairs_per_axis - self.uid_pairs


@dataclass(frozen=True)
class EsmcConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_positions: int
    rope_theta: float
    bos_id: int
    eos_id: int
    pad_id: int


@dataclass(frozen=True)
class EsmFold2Config:
    hidden_size: int
    pair_size: int
    single_inputs_size: int
    pair_transition_size: int
    atom_window_radius: int
    residue_bins: int
    chain_bins: int
    num_res_types: int
    max_atomic_number: int
    char_vocab_size: int
    max_chars: int
    max_atoms_per_token: int
    atom_feature_dim: int
    trunk_layers: int
    coda_layers: int
    lm_layers: int
    lm_dropout: float
    per_loop_lm_dropout: bool
    msa_enabled: bool
    msa_hidden_size: int
    msa_outer_size: int
    msa_layers: int
    msa_heads: int
    msa_head_dim: int
    msa_intermediate_size: int
    confidence_layers: int
    plddt_bins: int
    pae_bins: int
    pde_bins: int
    confidence_distance_bins: int
    confidence_eps: float
    sigma_data: float
    diffusion_hidden_size: int
    fourier_dim: int
    diffusion_layers: int
    diffusion_heads: int
    diffusion_head_dim: int
    diffusion_intermediate_size: int
    diffusion_pair_intermediate_size: int
    num_diffusion_samples: int
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float
    sigma_max_ratio: float
    sigma_min_ratio: float
    schedule_exponent: float
    inference_num_steps: int
    sigma_cap: float
    trunk_atoms: AtomStackConfig
    diffusion_atoms: AtomStackConfig
    esmc: EsmcConfig

    @property
    def lm_states(self) -> int:
        """ESMC hidden states mixed by the LM encoder: the embeddings and every layer output."""

        return self.esmc.num_layers + 1


def _atom_stack(raw: dict[str, Any], where: str) -> AtomStackConfig:
    stack = AtomStackConfig(
        hidden_size=_int(raw, "hidden_size", where),
        output_dim=_int(raw, "output_dim", where),
        num_layers=_int(raw, "num_hidden_layers", where),
        num_heads=_int(raw, "num_attention_heads", where),
        head_dim=_int(raw, "head_dim", where),
        intermediate_size=_int(raw, "intermediate_size", where),
        spatial_rope_base=_float(raw, "spatial_rope_base_frequency", where, positive=True),
        spatial_pairs_per_axis=_int(raw, "num_spatial_rope_pairs_per_axis", where),
        uid_pairs=_int(raw, "num_uid_rope_pairs", where),
        uid_rope_base=_float(raw, "uid_rope_base_frequency", where, positive=True),
    )
    _require(
        stack.num_heads * stack.head_dim == stack.hidden_size,
        f"{where}: heads * head_dim must equal hidden_size",
    )
    _require(stack.head_dim % 2 == 0, f"{where}: head_dim must be even for rotary embeddings")
    _require(stack.inactive_rope_pairs >= 0, f"{where}: rotary pairs exceed head_dim / 2")
    return stack


def _esmc(raw: dict[str, Any]) -> EsmcConfig:
    where = "esmc_config"
    _require(raw.get("model_type") == "esmc", "esmc_config.model_type must be 'esmc'")
    _require(raw.get("hidden_act") == "silu", "esmc_config.hidden_act must be 'silu'")
    _require(raw.get("attention_bias") is False, "esmc_config.attention_bias must be false")
    _require(raw.get("mlp_bias") is False, "esmc_config.mlp_bias must be false")
    rope = _object(raw, "rope_parameters", where)
    _require(rope.get("rope_type") == "default", "esmc_config rope_type must be 'default'")
    config = EsmcConfig(
        vocab_size=_int(raw, "vocab_size", where),
        hidden_size=_int(raw, "hidden_size", where),
        intermediate_size=_int(raw, "intermediate_size", where),
        num_layers=_int(raw, "num_hidden_layers", where),
        num_heads=_int(raw, "num_attention_heads", where),
        head_dim=_int(raw, "head_dim", where),
        max_positions=_int(raw, "max_position_embeddings", where),
        rope_theta=_float(rope, "rope_theta", f"{where}.rope_parameters", positive=True),
        bos_id=_int(raw, "bos_token_id", where, minimum=0),
        eos_id=_int(raw, "eos_token_id", where, minimum=0),
        pad_id=_int(raw, "pad_token_id", where, minimum=0),
    )
    _require(
        _int(raw, "num_key_value_heads", where) == config.num_heads,
        "esmc_config: grouped-query attention is not supported",
    )
    _require(
        config.num_heads * config.head_dim == config.hidden_size,
        "esmc_config: heads * head_dim must equal hidden_size",
    )
    _require(config.head_dim % 2 == 0, "esmc_config: head_dim must be even")
    for name in ("bos_id", "eos_id", "pad_id"):
        _require(getattr(config, name) < config.vocab_size, f"esmc_config.{name} is outside the vocabulary")
    return config


def parse_config(raw: dict[str, Any]) -> EsmFold2Config:
    """Validate a parsed ``config.json``; raise :class:`ConfigError` on anything unsupported."""

    _require(raw.get("model_type") == "esmfold2", "model_type must be 'esmfold2'")
    _require(raw.get("architectures") == ["EsmFold2Model"], "architectures must be ['EsmFold2Model']")
    if "type" in raw:
        _require(raw["type"] == "release", f"only the released ESMFold2 graph is supported, got type={raw['type']!r}")
    root = "config"
    structure = _object(raw, "structure_head", root)
    diffusion = _object(structure, "diffusion_module", "structure_head")
    confidence = _object(raw, "confidence_head", root)
    msa = _object(raw, "msa_encoder", root)
    lm = _object(raw, "lm_encoder", root)
    parcae = _object(raw, "parcae", root)

    _require(_bool(confidence, "enabled", "confidence_head"), "confidence_head.enabled must be true")
    _require(_bool(lm, "enabled", "lm_encoder"), "lm_encoder.enabled must be true")
    _require(_bool(parcae, "enabled", "parcae"), "parcae.enabled must be true")

    config = EsmFold2Config(
        hidden_size=_int(raw, "hidden_size", root),
        pair_size=_int(raw, "pairwise_hidden_size", root),
        single_inputs_size=_int(raw, "single_inputs_size", root),
        pair_transition_size=_int(raw, "pair_transition_intermediate_size", root),
        atom_window_radius=_int(raw, "sliding_window", root, minimum=2) // 2,
        residue_bins=_int(raw, "num_relative_residx_bins", root),
        chain_bins=_int(raw, "num_relative_chain_bins", root),
        num_res_types=_int(raw, "num_res_types", root),
        max_atomic_number=_int(raw, "max_atomic_number", root),
        char_vocab_size=_int(raw, "char_vocab_size", root),
        max_chars=_int(raw, "max_chars", root),
        max_atoms_per_token=_int(raw, "max_atoms_per_token", root),
        atom_feature_dim=_int(raw, "atom_feature_dim", root),
        trunk_layers=_int(raw, "folding_trunk_num_hidden_layers", root),
        coda_layers=_int(raw, "parcae_num_coda_layers", root),
        lm_layers=_int(lm, "num_hidden_layers", "lm_encoder"),
        lm_dropout=_float(lm, "lm_dropout", "lm_encoder"),
        per_loop_lm_dropout=_bool(lm, "per_loop_lm_dropout", "lm_encoder"),
        msa_enabled=_bool(msa, "enabled", "msa_encoder"),
        msa_hidden_size=_int(msa, "hidden_size", "msa_encoder"),
        msa_outer_size=_int(msa, "outer_hidden_size", "msa_encoder"),
        msa_layers=_int(msa, "num_hidden_layers", "msa_encoder"),
        msa_heads=_int(msa, "num_attention_heads", "msa_encoder"),
        msa_head_dim=_int(msa, "head_dim", "msa_encoder"),
        msa_intermediate_size=_int(msa, "intermediate_size", "msa_encoder"),
        confidence_layers=_int(confidence, "num_hidden_layers", "confidence_head"),
        plddt_bins=_int(confidence, "num_plddt_bins", "confidence_head", minimum=2),
        pae_bins=_int(confidence, "num_pae_bins", "confidence_head", minimum=2),
        pde_bins=_int(confidence, "num_pde_bins", "confidence_head", minimum=2),
        confidence_distance_bins=_int(confidence, "distogram_bins", "confidence_head", minimum=2),
        confidence_eps=_float(confidence, "eps", "confidence_head", positive=True),
        sigma_data=_float(diffusion, "sigma_data", "diffusion_module", positive=True),
        diffusion_hidden_size=_int(diffusion, "hidden_size", "diffusion_module"),
        fourier_dim=_int(diffusion, "fourier_dim", "diffusion_module"),
        diffusion_layers=_int(diffusion, "num_hidden_layers", "diffusion_module"),
        diffusion_heads=_int(diffusion, "num_attention_heads", "diffusion_module"),
        diffusion_head_dim=_int(diffusion, "head_dim", "diffusion_module"),
        diffusion_intermediate_size=_int(diffusion, "intermediate_size", "diffusion_module"),
        diffusion_pair_intermediate_size=_int(diffusion, "pair_intermediate_size", "diffusion_module"),
        num_diffusion_samples=_int(structure, "num_diffusion_samples", "structure_head"),
        gamma_0=_float(structure, "gamma_0", "structure_head"),
        gamma_min=_float(structure, "gamma_min", "structure_head"),
        noise_scale=_float(structure, "noise_scale", "structure_head"),
        step_scale=_float(structure, "step_scale", "structure_head"),
        sigma_max_ratio=_float(structure, "inference_sigma_max_ratio", "structure_head", positive=True),
        sigma_min_ratio=_float(structure, "inference_sigma_min_ratio", "structure_head", positive=True),
        schedule_exponent=_float(structure, "inference_exponent", "structure_head", positive=True),
        inference_num_steps=_int(structure, "inference_num_steps", "structure_head"),
        sigma_cap=_float(structure, "inference_sigma_cap", "structure_head", positive=True),
        trunk_atoms=_atom_stack(_object(raw, "atom_encoder", root), "atom_encoder"),
        diffusion_atoms=_atom_stack(_object(diffusion, "atom_encoder", "diffusion_module"), "diffusion atom_encoder"),
        esmc=_esmc(_object(raw, "esmc_config", root)),
    )
    _validate_relations(config, raw, parcae, lm)
    return config


def _validate_relations(
    config: EsmFold2Config, raw: dict[str, Any], parcae: dict[str, Any], lm: dict[str, Any]
) -> None:
    _require(
        config.atom_feature_dim
        == 3 + 1 + 1 + config.max_atomic_number + config.char_vocab_size * config.max_chars,
        "atom_feature_dim must equal xyz + charge + mask + element one-hot + atom-name one-hots",
    )
    _require(
        config.trunk_atoms.output_dim == config.single_inputs_size - (2 * config.num_res_types + 1),
        "atom_encoder.output_dim must equal single_inputs_size - (2 * num_res_types + 1)",
    )
    _require(
        config.diffusion_atoms.output_dim == config.diffusion_hidden_size,
        "the diffusion atom encoder must project to the diffusion hidden size",
    )
    _require(
        config.diffusion_heads * config.diffusion_head_dim == config.diffusion_hidden_size,
        "diffusion heads * head_dim must equal the diffusion hidden size",
    )
    _require(
        config.trunk_atoms.hidden_size == config.diffusion_atoms.hidden_size,
        "the trunk and diffusion atom stacks must share one width (the decoder reuses the encoder's states)",
    )
    _require(
        config.msa_heads * config.msa_head_dim == config.msa_hidden_size or not config.msa_enabled,
        "msa_encoder heads * head_dim must equal its hidden size",
    )
    _require(
        _int(parcae, "coda_n_layers", "parcae") == config.coda_layers,
        "parcae.coda_n_layers must equal parcae_num_coda_layers",
    )
    _require(0.0 <= config.lm_dropout < 1.0, "lm_encoder.lm_dropout must lie in [0, 1)")
    _require(config.noise_scale >= 0.0, "structure_head.noise_scale must not be negative")
    _require(config.gamma_0 >= 0.0, "structure_head.gamma_0 must not be negative")
    _require(config.sigma_min_ratio < config.sigma_max_ratio, "the noise schedule bounds are inverted")
    _require(config.confidence_distance_bins >= 2, "confidence distogram_bins must be at least 2")
    _require(config.max_atoms_per_token >= 14, "max_atoms_per_token must cover the largest amino acid")
    _require(
        config.esmc.hidden_size == raw.get("lm_d_model") and config.esmc.num_layers == raw.get("lm_num_layers"),
        "lm_d_model / lm_num_layers must describe the bundled ESMC trunk",
    )
    _require(lm.get("per_loop_lm_dropout") is True, "only per-loop LM dropout is supported")


def load_config(model_dir: Path) -> tuple[EsmFold2Config, dict[str, Any]]:
    """Read and validate ``config.json``; return the parsed config and the raw object."""

    path = Path(model_dir) / "config.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"ESMFold2 checkpoint has no config.json: {path}") from error

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"config.json repeats key {key!r}")
            result[key] = value
        return result

    raw = json.loads(text, object_pairs_hook=reject_duplicates)
    if not isinstance(raw, dict):
        raise ConfigError("config.json must hold an object")
    return parse_config(raw), raw

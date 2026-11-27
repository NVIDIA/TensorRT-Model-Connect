# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one ESMFold2 bundle: the TensorRT plans of a single-chain protein structure predictor."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

from . import plans
from .checkpoint import Checkpoint
from .config import EsmFold2Config, load_config
from .graph import build_plan, import_tensorrt
from .layers import ATOM_QUERY_BLOCK, lm_layer_mix_weights

FAMILY = "esmfold2"
TASK = "structure_prediction"
PRECISIONS = ("bf16", "fp32")
DEFAULT_MAX_TOKENS = 512
# Larger maxima are not validated: the plans reserve activation memory for their largest profile, which grows with
# the square of the maximum (about 79 GiB at 1024 residues and 32 samples), and TensorRT cannot build the plans at
# 2046 residues.
MAX_TOKENS = 1024
MIN_TOKENS = 2
# Heavy atoms of the largest standard residue (tryptophan); bounds the padded atom axis of a sequence.
MAX_ATOMS_PER_RESIDUE = 14
RUNTIME_FORMAT = 1


def token_limits(config: EsmFold2Config, max_sequence_length: int | None) -> tuple[int, int]:
    """``(max_tokens, max_atoms)`` of the bundle; the ESMC context bounds the sequence (BOS + EOS)."""

    limit = min(config.esmc.max_positions - 2, MAX_TOKENS)
    if max_sequence_length is not None and (
        isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int)
    ):
        raise ValueError(f"esmfold2 max_sequence_length must be an integer, got {max_sequence_length!r}")
    max_tokens = DEFAULT_MAX_TOKENS if max_sequence_length is None else max_sequence_length
    if max_tokens < MIN_TOKENS or max_tokens > limit:
        raise ValueError(f"esmfold2 max_sequence_length must lie in [{MIN_TOKENS}, {limit}], got {max_tokens}")
    max_atoms = math.ceil(max_tokens * MAX_ATOMS_PER_RESIDUE / ATOM_QUERY_BLOCK) * ATOM_QUERY_BLOCK
    return max_tokens, max_atoms


def _profiles(config: EsmFold2Config, max_tokens: int, max_atoms: int) -> dict[str, dict[str, tuple]]:
    opt_tokens = min(max_tokens, 256)
    opt_atoms = min(max_atoms, math.ceil(opt_tokens * 8 / ATOM_QUERY_BLOCK) * ATOM_QUERY_BLOCK)
    samples = config.num_diffusion_samples
    c, s, d = config.pair_size, config.single_inputs_size, config.diffusion_hidden_size
    width = config.diffusion_atoms.hidden_size
    head = config.diffusion_atoms.head_dim
    layers, heads = config.diffusion_layers, config.diffusion_heads

    def tokens(*tail: int, lead: tuple[int, ...] = (1,)) -> tuple:
        return (lead + (MIN_TOKENS,) + tail, lead + (opt_tokens,) + tail, lead + (max_tokens,) + tail)

    def pair(*tail: int) -> tuple:
        return (
            (1, MIN_TOKENS, MIN_TOKENS) + tail,
            (1, opt_tokens, opt_tokens) + tail,
            (1, max_tokens, max_tokens) + tail,
        )

    def atoms(*tail: int, lead: tuple[int, ...] = (1,)) -> tuple:
        return (lead + (ATOM_QUERY_BLOCK,) + tail, lead + (opt_atoms,) + tail, lead + (max_atoms,) + tail)

    atom_inputs = {
        "ref_pos": atoms(3),
        "ref_charge": atoms(),
        "atom_mask": atoms(),
        "ref_element": atoms(),
        "ref_name_chars": atoms(4),
        "ref_space_uid": atoms(),
        "atom_to_token": atoms(),
    }
    token_ids = {name: tokens() for name in ("res_type", "residue_index", "token_index", "asym_id", "sym_id",
                                             "entity_id")}
    conditioning_atoms = {name: shape for name, shape in atom_inputs.items() if name != "atom_to_token"}
    return {
        "esmc": {"input_ids": ((1, MIN_TOKENS + 2), (1, opt_tokens + 2), (1, max_tokens + 2))},
        "embed": {"lm_single": tokens(c), **token_ids, **atom_inputs},
        "msa": {"initial_pair": pair(c), "single_inputs": tokens(s), "res_type": tokens()},
        "loop": {"state": pair(c), "base_pair": pair(c), "lm_pair": pair(c), "lm_keep": pair(c)},
        "decode": {"state": pair(c)},
        "conditioning": {"pair_trunk": pair(c), "relpos": pair(c), "single_inputs": tokens(s),
                         **conditioning_atoms},
        "denoise": {
            "noisy_coords": ((1, ATOM_QUERY_BLOCK, 3), (samples, opt_atoms, 3), (samples, max_atoms, 3)),
            "noise_level": ((1, 1), (samples, 1), (samples, 1)),
            "token_bias": (
                (layers, heads, MIN_TOKENS, MIN_TOKENS),
                (layers, heads, opt_tokens, opt_tokens),
                (layers, heads, max_tokens, max_tokens),
            ),
            "cond_single": tokens(d),
            "atom_embeds": atoms(width),
            "rope_cos": atoms(head),
            "rope_sin": atoms(head),
            "atom_mask": atoms(),
            "atom_to_token": atoms(),
        },
        "confidence": {
            "single_inputs": tokens(s),
            "pair_trunk": pair(c),
            "relpos": pair(c),
            "coords": atoms(3),
            "rep_atom": tokens(),
            "atom_to_token": atoms(),
            "intra_index": atoms(),
        },
    }


def _runtime_json(config: EsmFold2Config, precision: str, max_tokens: int, max_atoms: int) -> dict[str, Any]:
    return {
        "format": RUNTIME_FORMAT,
        "precision": precision,
        "max_tokens": max_tokens,
        "max_atoms": max_atoms,
        "max_samples": config.num_diffusion_samples,
        "pair_size": config.pair_size,
        "single_inputs_size": config.single_inputs_size,
        "diffusion_hidden_size": config.diffusion_hidden_size,
        "diffusion_layers": config.diffusion_layers,
        "diffusion_heads": config.diffusion_heads,
        "atom_hidden_size": config.diffusion_atoms.hidden_size,
        "atom_head_dim": config.diffusion_atoms.head_dim,
        "max_atoms_per_token": config.max_atoms_per_token,
        "pae_bins": config.pae_bins,
        "msa_encoder": config.msa_enabled,
        "lm_dropout": config.lm_dropout,
        "trunk_init_std": math.sqrt(2.0 / (5.0 * config.pair_size)),
        "esmc_bos_id": config.esmc.bos_id,
        "esmc_eos_id": config.esmc.eos_id,
        "num_diffusion_samples": config.num_diffusion_samples,
        "confidence_eps": config.confidence_eps,
        "sampler": {
            "sigma_data": config.sigma_data,
            "gamma_0": config.gamma_0,
            "gamma_min": config.gamma_min,
            "noise_scale": config.noise_scale,
            "step_scale": config.step_scale,
            "sigma_max_ratio": config.sigma_max_ratio,
            "sigma_min_ratio": config.sigma_min_ratio,
            "exponent": config.schedule_exponent,
            "num_steps": config.inference_num_steps,
            "sigma_cap": config.sigma_cap,
        },
    }


# The ranges runtime/bundle_config.cpp accepts: a bundle outside them would build and then fail to load.
_RUNTIME_INTEGERS = {
    "max_tokens": (MIN_TOKENS, 4094), "max_atoms": (32, 1 << 20), "max_samples": (1, 1024), "pair_size": (1, 4096),
    "single_inputs_size": (1, 8192), "diffusion_hidden_size": (1, 8192), "diffusion_layers": (1, 256),
    "diffusion_heads": (1, 256), "atom_hidden_size": (1, 4096), "atom_head_dim": (2, 1024),
    "max_atoms_per_token": (14, 64), "pae_bins": (2, 4096), "esmc_bos_id": (0, 1 << 16), "esmc_eos_id": (0, 1 << 16),
}
_RUNTIME_REALS = {"lm_dropout": (0.0, 0.999), "trunk_init_std": (1.0e-6, 10.0), "confidence_eps": (1.0e-30, 1.0)}
_SAMPLER_REALS = {
    "sigma_data": (1.0e-6, 1.0e6), "gamma_0": (0.0, 1.0e3), "gamma_min": (0.0, 1.0e6), "noise_scale": (0.0, 1.0e3),
    "step_scale": (0.0, 1.0e3), "sigma_max_ratio": (1.0e-6, 1.0e6), "sigma_min_ratio": (1.0e-12, 1.0e6),
    "exponent": (1.0e-3, 1.0e3), "sigma_cap": (1.0e-6, 1.0e9),
}


def check_runtime_json(runtime: dict[str, Any]) -> None:
    """Reject what the runtime's load check would reject, before any plan is built."""

    def bounded(values: dict[str, Any], name: str, low: float, high: float, integer: bool) -> None:
        value = values[name]
        valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
        if isinstance(value, bool) or not valid_type or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"esmfold2 runtime value {name}={value!r} lies outside [{low}, {high}]")

    for name, (low, high) in _RUNTIME_INTEGERS.items():
        bounded(runtime, name, low, high, True)
    for name, (low, high) in _RUNTIME_REALS.items():
        bounded(runtime, name, low, high, False)
    bounded(runtime, "num_diffusion_samples", 1, runtime["max_samples"], True)
    for name, (low, high) in _SAMPLER_REALS.items():
        bounded(runtime["sampler"], name, low, high, False)
    bounded(runtime["sampler"], "num_steps", 1, 10000, True)
    if runtime["max_atoms"] % ATOM_QUERY_BLOCK or runtime["max_atoms"] < 4 * runtime["max_tokens"]:
        raise ValueError("esmfold2 max_atoms is inconsistent with max_tokens")
    if runtime["sampler"]["sigma_min_ratio"] >= runtime["sampler"]["sigma_max_ratio"]:
        raise ValueError("esmfold2 noise schedule bounds are inverted")


def build(request: Any, writer: Any) -> None:
    """Build every plan of an ESMFold2 / ESMFold2-Fast bundle from a local checkpoint directory."""

    if request.task != TASK:
        raise ValueError(f"esmfold2 supports only task={TASK!r}, got {request.task!r}")
    if request.backend != "trt":
        raise ValueError("esmfold2 supports only the trt backend")
    if request.precision not in PRECISIONS:
        raise ValueError(f"esmfold2 precision must be one of {PRECISIONS}, got {request.precision!r}")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("esmfold2 has no quantized build")
    if request.fp32_layers:
        raise NotImplementedError("esmfold2 does not take fp32 layer overrides")
    if request.max_batch_size != 1:
        raise NotImplementedError("esmfold2 folds one sequence per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("esmfold2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("esmfold2 does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("esmfold2 does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("esmfold2 has no image input")
    if request.video_num_frames is not None:
        raise NotImplementedError("esmfold2 has no video input")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("esmfold2 does not support MTP speculative decoding")
    model_dir = Path(request.model_dir)
    config, _ = load_config(model_dir)
    checkpoint = Checkpoint(model_dir, config)
    max_tokens, max_atoms = token_limits(config, request.max_sequence_length)
    trt = import_tensorrt()
    bf16 = request.precision == "bf16"
    compute = trt.bfloat16 if bf16 else trt.float32
    profiles = _profiles(config, max_tokens, max_atoms)

    layer_mix = lm_layer_mix_weights(checkpoint.tensor("language_model.layer_weights"), bf16)
    decay, matrix = plans.parcae_discretization(
        {name: checkpoint.tensor(name)
         for name in ("parcae.log_delta", "parcae.log_state_decay", "parcae.input_matrix_continuous")},
        bf16,
    )
    definitions = [
        ("esmc", partial(plans.define_esmc, config=config, layer_mix=layer_mix)),
        ("embed", partial(plans.define_embed, config=config)),
        ("loop", partial(plans.define_loop, config=config, decay=decay, matrix=matrix)),
        ("decode", partial(plans.define_decode, config=config)),
        ("conditioning", partial(plans.define_conditioning, config=config)),
        ("denoise", partial(plans.define_denoise, config=config)),
        ("confidence", partial(plans.define_confidence, config=config)),
    ]
    if config.msa_enabled:
        definitions.insert(2, ("msa", partial(plans.define_msa, config=config)))

    runtime = _runtime_json(config, request.precision, max_tokens, max_atoms)
    check_runtime_json(runtime)
    writer.set_header(family="esmfold2", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    for name, define in definitions:
        plan = build_plan(
            trt,
            define,
            compute=compute,
            weight=checkpoint.tensor,
            profiles=[profiles[name]],
            verbose=bool(request.verbose),
        )
        writer.add_bytes(f"esmfold2_{name}_plan", memoryview(plan))
        del plan

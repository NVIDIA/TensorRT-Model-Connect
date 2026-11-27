# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity, default-task and configuration policy of ESMFold2."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from families.esmfold2 import model
from families.esmfold2.config import ConfigError, parse_config
from families.esmfold2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _atom_stack(output_dim: int) -> dict:
    return {
        "hidden_size": 128, "output_dim": output_dim, "num_hidden_layers": 3, "num_attention_heads": 4,
        "head_dim": 32, "intermediate_size": 256, "spatial_rope_base_frequency": 20.0,
        "num_spatial_rope_pairs_per_axis": 2, "num_uid_rope_pairs": 10, "uid_rope_base_frequency": 10000.0,
    }


def released_config() -> dict:
    """The values of the released ``biohub/ESMFold2`` configuration that the family reads."""

    return {
        "model_type": "esmfold2", "architectures": ["EsmFold2Model"], "type": "release",
        "hidden_size": 384, "pairwise_hidden_size": 256, "single_inputs_size": 451,
        "pair_transition_intermediate_size": 1024, "sliding_window": 128, "num_relative_residx_bins": 32,
        "num_relative_chain_bins": 2, "num_res_types": 33, "max_atomic_number": 128, "char_vocab_size": 64,
        "max_chars": 4, "max_atoms_per_token": 23, "atom_feature_dim": 389, "folding_trunk_num_hidden_layers": 48,
        "parcae_num_coda_layers": 2, "lm_d_model": 2560, "lm_num_layers": 80,
        "atom_encoder": _atom_stack(384),
        "structure_head": {
            "diffusion_module": {
                "sigma_data": 16.0, "hidden_size": 768, "fourier_dim": 256, "num_hidden_layers": 12,
                "num_attention_heads": 16, "head_dim": 48, "intermediate_size": 1536, "pair_intermediate_size": 512,
                "atom_encoder": _atom_stack(768),
            },
            "num_diffusion_samples": 32, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5,
            "inference_sigma_max_ratio": 160.0, "inference_sigma_min_ratio": 0.0004, "inference_exponent": 7.0,
            "inference_num_steps": 14, "inference_sigma_cap": 256.0,
        },
        "confidence_head": {
            "num_hidden_layers": 4, "num_plddt_bins": 50, "num_pde_bins": 64, "num_pae_bins": 64,
            "distogram_bins": 39, "eps": 1e-06, "enabled": True,
        },
        "msa_encoder": {
            "hidden_size": 128, "outer_hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 8,
            "head_dim": 16, "intermediate_size": 512, "enabled": True,
        },
        "lm_encoder": {"num_hidden_layers": 4, "lm_dropout": 0.25, "per_loop_lm_dropout": True, "enabled": True},
        "parcae": {"enabled": True, "coda_n_layers": 2},
        "esmc_config": {
            "model_type": "esmc", "vocab_size": 64, "hidden_size": 2560, "intermediate_size": 6912,
            "num_hidden_layers": 80, "num_attention_heads": 40, "num_key_value_heads": 40, "head_dim": 64,
            "hidden_act": "silu", "max_position_embeddings": 2048,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "attention_bias": False,
            "mlp_bias": False, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2,
        },
    }


def test_structure_prediction_is_offered() -> None:
    for config in ({"model_type": "esmfold2"}, {"architectures": ["EsmFold2Model"]}):
        support = describe(ModelMetadata(config=config, model_index={}))
        assert support is not None
        assert support.tasks == ("structure_prediction",)
        assert support.default_task == "structure_prediction"
        assert support.default_precision == "bf16"
        assert {(variant.precision, variant.status) for variant in support.variants} == {
            ("bf16", "supported"), ("fp32", "experimental")
        }


def test_the_language_model_families_are_not_claimed() -> None:
    for model_type in ("esm", "esmc", "boltz2", "openfold3"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_released_configuration_parses() -> None:
    config = parse_config(released_config())
    assert config.trunk_layers == 48 and config.msa_enabled
    assert config.atom_window_radius == 64 and config.lm_states == 81
    assert config.trunk_atoms.inactive_rope_pairs == 0
    fast = released_config()
    fast["folding_trunk_num_hidden_layers"] = 24
    fast["msa_encoder"]["enabled"] = False
    assert not parse_config(fast).msa_enabled


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("hidden_size",), 384.0),  # integers are not coerced
        (("pairwise_hidden_size",), True),
        (("type",), "experimental"),
        (("atom_feature_dim",), 390),
        (("single_inputs_size",), 450),
        (("structure_head", "gamma_0"), math.inf),
        (("structure_head", "inference_sigma_cap"), None),
        (("structure_head", "inference_sigma_min_ratio"), 200.0),
        (("lm_encoder", "lm_dropout"), 1.0),
        (("lm_encoder", "per_loop_lm_dropout"), False),
        (("confidence_head", "enabled"), False),
        (("esmc_config", "num_key_value_heads"), 8),
        (("esmc_config", "hidden_act"), "gelu"),
        (("esmc_config", "rope_parameters", "rope_type"), "yarn"),
        (("structure_head", "diffusion_module", "head_dim"), 32),
        (("parcae", "coda_n_layers"), 3),
        (("esmc_config", "rope_parameters", "rope_theta"), 1.0e-100),  # float32 frequencies overflow
        (("atom_encoder", "spatial_rope_base_frequency"), 1.0e-100),
        (("atom_encoder", "uid_rope_base_frequency"), 1.0e-100),
        (("structure_head", "diffusion_module", "atom_encoder", "uid_rope_base_frequency"), 1.0e-100),
    ],
)
def test_configurations_the_graph_cannot_honor_are_rejected(path: tuple[str, ...], value: object) -> None:
    raw = copy.deepcopy(released_config())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_missing_values_are_rejected() -> None:
    raw = released_config()
    del raw["structure_head"]["noise_scale"]
    with pytest.raises(ConfigError, match="noise_scale"):
        parse_config(raw)


def _request(**overrides: object) -> SimpleNamespace:
    values = {
        "model_dir": "/nonexistent/esmfold2", "output_path": "/nonexistent/esmfold2.bundle", "family": "esmfold2",
        "task": "structure_prediction", "precision": "bf16", "backend": "trt", "max_sequence_length": None,
        "image_height": None, "image_width": None, "video_num_frames": None, "max_batch_size": 1,
        "tensor_parallel_size": 1, "context_parallel_size": 1, "quantization": None, "fp32_layers": (),
        "dynamic_kv_cache": False, "verbose": False, "graph_transform": None, "mtp_seq_len": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"task": "classification"}, "task="),
        ({"backend": "rtx"}, "trt backend"),
        ({"precision": "fp16"}, "precision"),
        ({"quantization": "fp8"}, "quantized"),
        ({"fp32_layers": (0,)}, "fp32 layer"),
        ({"max_batch_size": 2}, "max_batch_size"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
        ({"context_parallel_size": 2}, "context parallelism"),
        ({"dynamic_kv_cache": True}, "dynamic_kv_cache"),
        ({"image_height": 224}, "image"),
        ({"video_num_frames": 8}, "video"),
        ({"mtp_seq_len": 2}, "MTP"),
    ],
)
def test_unsupported_build_requests_are_rejected_before_loading(override: dict, message: str) -> None:
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(_request(**override), writer=None)


def test_the_maximum_sequence_length_is_bounded() -> None:
    config = parse_config(released_config())
    assert model.token_limits(config, None) == (512, 7168)
    assert model.token_limits(config, 1024) == (1024, 14336)
    for value in (1, 1025, 2046, 512.5, True):
        with pytest.raises(ValueError, match="max_sequence_length"):
            model.token_limits(config, value)


def test_runtime_values_the_loader_rejects_fail_the_build() -> None:
    config = parse_config(released_config())
    runtime = model._runtime_json(config, "bf16", 512, 7168)
    model.check_runtime_json(runtime)
    for path, value in (
        (("lm_dropout",), 0.9999),
        (("confidence_eps",), 2.0),
        (("max_atoms",), 7169),
        (("num_diffusion_samples",), 33),
        (("pae_bins",), 64.0),
        (("sampler", "num_steps"), 0),
        (("sampler", "exponent"), math.nan),
        (("sampler", "sigma_min_ratio"), 200.0),
    ):
        broken = copy.deepcopy(runtime)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            model.check_runtime_json(broken)


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        # Each edit keeps every older relation (feature widths follow the changed depth), so only the bound rejects it.
        ({("num_res_types",): 22, ("single_inputs_size",): 429}, "unknown residue type"),
        ({("max_chars",): 3, ("atom_feature_dim",): 325}, "four name characters"),
        ({("char_vocab_size",): 58, ("atom_feature_dim",): 365}, "name character 58"),
        ({("max_atomic_number",): 16, ("atom_feature_dim",): 277}, "sulfur"),
        ({("esmc_config", "vocab_size"): 23}, "residue tokens"),
        # Spatial RoPE positions are conformer coordinates (up to 4.52), not token positions (4 here).
        (
            {("esmc_config", "max_position_embeddings"): 4, ("atom_encoder", "spatial_rope_base_frequency"): 1.6e-76},
            "spatial_rope_base_frequency gives RoPE angles",
        ),
    ],
)
def test_one_hot_depths_must_cover_the_featurizer_tables(edits: dict, message: str) -> None:
    raw = copy.deepcopy(released_config())
    for path, value in edits.items():
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_rope_checks_do_not_scale_with_the_head_width() -> None:
    import tracemalloc

    raw = copy.deepcopy(released_config())
    width = 1 << 24
    raw["esmc_config"].update(head_dim=width, num_attention_heads=1, num_key_value_heads=1, hidden_size=width)
    raw["lm_d_model"] = width
    tracemalloc.start()
    try:
        config = parse_config(raw)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert config.esmc.head_dim == width  # the checkpoint's tensor shapes reject it later, not this parser
    assert peak < 1 << 20, peak  # one float per RoPE pair would be 8 M floats

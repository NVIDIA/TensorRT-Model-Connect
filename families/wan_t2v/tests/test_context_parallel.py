# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused family-owned contracts for Wan2.1 context parallelism."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import types

import pytest

try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    fake_tensorrt = types.ModuleType("tensorrt")
    fake_tensorrt.__version__ = "11.0.0"
    sys.modules["tensorrt"] = fake_tensorrt

try:
    import ml_dtypes  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ml_dtypes"] = types.ModuleType("ml_dtypes")

from tensorrt_model_connect.build import BuildRequest

from families.wan_t2v import model
from families.wan_t2v.parallel import ParallelConfig
from families.wan_t2v.standard_dit_cp_builder import (
    _round_up_to_multiple,
    _validate_context_parallel,
)
from families.wan_t2v.tests.test_e2e import MPI_RANK_ZERO
from families.wan_t2v import standard_dit_cp_builder


class _Writer:
    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.sections: dict[str, bytes] = {}
        self.json_sections: dict[str, dict] = {}

    def set_header(self, **header: str) -> None:
        self.header = header

    def add_bytes(self, name: str, payload: bytes) -> None:
        self.sections[name] = payload

    def add_json(self, name: str, payload: dict) -> None:
        self.json_sections[name] = payload


@pytest.mark.parametrize(
    ("size", "local_patches", "routed_heads"),
    [(2, 1008, 12), (4, 504, 12), (8, 252, 16)],
)
def test_context_parallel_config_supports_current_wan_worlds(
    size: int, local_patches: int, routed_heads: int
) -> None:
    parallel = ParallelConfig(cp_size=size)

    parallel.validate()
    assert parallel.cp_enabled
    assert not parallel.enabled
    assert parallel.distributed
    assert parallel.mode == "context_parallel"
    assert parallel.world_size == size
    assert 2016 // size == local_patches
    assert _round_up_to_multiple(12, size) == routed_heads


@pytest.mark.parametrize("size", [0, 3, 16])
def test_context_parallel_config_rejects_unsupported_worlds(size: int) -> None:
    with pytest.raises(ValueError, match="context_parallel_size"):
        ParallelConfig(cp_size=size).validate()


def test_context_parallel_cannot_be_combined_with_tensor_parallel() -> None:
    with pytest.raises(ValueError, match="cannot be enabled together"):
        ParallelConfig(tp_size=2, cp_size=4).validate()


def test_context_parallel_requires_even_patch_shards() -> None:
    _validate_context_parallel(ParallelConfig(cp_size=8), num_patches=2016)
    with pytest.raises(ValueError, match="num_patches divisible"):
        _validate_context_parallel(ParallelConfig(cp_size=8), num_patches=2017)


def test_cp8_routes_padded_heads_and_all_collective_phases() -> None:
    assert _round_up_to_multiple(12, 8) == 16
    source = inspect.getsource(standard_dit_cp_builder)
    assert "CollectiveOperation.REDUCE_SCATTER" in source
    assert "CollectiveOperation.ALL_TO_ALL" in source
    assert "CollectiveOperation.ALL_GATHER" in source
    assert "_pad_attention_heads(" in source


def test_runtime_uses_one_explicit_openmpi_nccl_contract() -> None:
    source = (Path(__file__).resolve().parents[1] / "runtime/distributed_runtime.cpp").read_text(
        encoding="utf-8"
    )

    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "TRTMC_NCCL_RENDEZVOUS",
    ):
        assert f'require_env_int("{name}")' in source or (
            name == "TRTMC_NCCL_RENDEZVOUS" and f'std::getenv("{name}")' in source
        )
    assert source.count('dlopen("libnccl.so.2"') == 1
    for forbidden in (
        'dlopen("libnccl.so"',
        '"PMI_SIZE"',
        '"WORLD_SIZE"',
        '"LOCAL_RANK"',
        "temp_directory_path",
        "TRTMC_NCCL_SKIP_DESTROY",
    ):
        assert forbidden not in source


def test_mpirun_output_selects_only_rank_zero_payload() -> None:
    rank_zero = MPI_RANK_ZERO.fullmatch('[job,0]<stdout>:{"output":"frames"}')

    assert rank_zero is not None
    assert json.loads(rank_zero.group(1)) == {"output": "frames"}
    assert MPI_RANK_ZERO.fullmatch('[job,1]<stdout>:{"worker":true}') is None


@pytest.mark.parametrize("size", [2, 4, 8])
def test_build_writes_one_shared_cp_plan_and_new_runtime_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int
) -> None:
    model_dir = tmp_path / "model"
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    observed: dict[str, ParallelConfig] = {}

    monkeypatch.setattr(
        model._WanT2VModel,
        "load_weights",
        lambda _self, _model_dir, _config: {},
    )

    def build_components(_self, _model_dir, _config, _weights, **kwargs):
        observed["parallel"] = kwargs["parallel_config"]
        return {
            "text_encoders": [("t5", b"t5")],
            "denoiser": b"cp-dit",
            "vae_decoder": b"vae",
            "vae_decoder_first_frame": b"vae-first",
            "preprocessor_weights": b"weights",
        }

    monkeypatch.setattr(model._WanT2VModel, "build_components", build_components)
    monkeypatch.setattr(
        model._WanT2VModel,
        "get_diffusion_config",
        lambda _self, _config: {},
    )

    writer = _Writer()
    model.build(
        BuildRequest(
            model_dir=model_dir,
            output_path=tmp_path / "wan.bundle",
            family="wan_t2v",
            task="image_generation",
            precision="fp16",
            image_height=384,
            image_width=672,
            video_num_frames=5,
            context_parallel_size=size,
        ),
        writer,
    )

    parallel = observed["parallel"]
    assert parallel.cp_size == size
    assert parallel.tp_size == 1
    assert writer.sections["denoiser.plan"] == b"cp-dit"
    assert not any(name.startswith("denoiser.rank") for name in writer.sections)
    assert writer.json_sections["runtime.json"]["parallel_mode"] == "context_parallel"
    assert writer.json_sections["runtime.json"]["parallel_size"] == size


def test_cp4_manifest_uses_build_request_fields() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads(
        (root / "manifests/wan21-t2v-1.3b-l0-cp4.json").read_text(encoding="utf-8")
    )
    assert manifest["tensor_parallel_size"] == 1
    assert manifest["context_parallel_size"] == 4
    assert manifest["bundle"].endswith(".bundle")
    assert manifest["task"] == "image_generation"

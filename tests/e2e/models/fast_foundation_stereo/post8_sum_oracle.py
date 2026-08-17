# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone full-volume bitwise oracle for the post8 sum IPluginV3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_SHAPE = (1, 28, 48, 176, 176)
_TILE_POSITIONS = (32, 64, 128, 256)
_PLUGIN_LIBRARY_ENV = "TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_LIBRARY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pin_plugin_library(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    existing = os.environ.get(_PLUGIN_LIBRARY_ENV)
    if existing is not None and Path(existing).expanduser().resolve() != resolved:
        raise RuntimeError(f"{_PLUGIN_LIBRARY_ENV} already selects a different DSO: {existing}")
    os.environ[_PLUGIN_LIBRARY_ENV] = str(resolved)
    return resolved


def _validate_tile_positions(tile_positions: int) -> int:
    if tile_positions not in _TILE_POSITIONS:
        raise ValueError(f"tile positions must be one of {_TILE_POSITIONS}, got {tile_positions}")
    return tile_positions


def _build_engine(
    tile_positions: int,
    branches: tuple[str, ...] = ("reference", "candidate"),
) -> bytes:
    from tensorrt_model_connect.families.fast_foundation_stereo.builder import (
        _create_network,
        _serialize_network,
    )
    from tensorrt_model_connect.families.fast_foundation_stereo.native_plugin_builder import (
        add_post8_sum_plugin,
        load_native_plugin,
    )

    tile_positions = _validate_tile_positions(tile_positions)
    unknown = set(branches) - {"reference", "candidate"}
    if not branches or unknown:
        raise ValueError(f"invalid standalone branches: {branches}")
    override = os.environ.get(_PLUGIN_LIBRARY_ENV)
    if override is None or not Path(override).expanduser().resolve().is_file():
        raise RuntimeError(
            f"standalone post8 build requires a pinned native plugin DSO via {_PLUGIN_LIBRARY_ENV}"
        )
    load_native_plugin(verbose=True)
    trt, builder, network = _create_network(verbose=True, strongly_typed=True)
    linear = network.add_input("linear", trt.float16, _SHAPE)
    skip = network.add_input("skip", trt.float16, _SHAPE)
    if linear is None or skip is None:
        raise RuntimeError("failed to add standalone post8 inputs")

    if "reference" in branches:
        layer = network.add_elementwise(linear, skip, trt.ElementWiseOperation.SUM)
        if layer is None:
            raise RuntimeError("failed to add standalone post8 control sum")
        reference = layer.get_output(0)
        reference.name = "reference"
        network.mark_output(reference)

    if "candidate" in branches:
        candidate = add_post8_sum_plugin(
            network,
            linear,
            skip,
            trt_module=trt,
            name="candidate",
            tile_positions=tile_positions,
        )
        candidate.name = "candidate"
        network.mark_output(candidate)

    return _serialize_network(
        trt,
        builder,
        network,
        fp16=True,
        strongly_typed=True,
        default_optimization_level=4,
        default_aux_streams=0,
        verbose=True,
    )


class _Runner:
    def __init__(self, engine_path: Path, output_names: tuple[str, ...]):
        import tensorrt as trt
        import torch

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create standalone post8 execution context")
        self.trt = trt
        self.output_names = output_names
        self.stream = torch.cuda.Stream()
        for name in ("linear", "skip", *output_names):
            if self.engine.get_tensor_dtype(name) != trt.float16:
                raise RuntimeError(f"standalone tensor {name} is not FP16")
            if self.engine.get_tensor_format(name) not in {
                trt.TensorFormat.LINEAR,
                trt.TensorFormat.DHWC8,
            }:
                raise RuntimeError(
                    f"standalone external tensor {name} must be LINEAR or DHWC8, got "
                    f"{self.engine.get_tensor_format_desc(name)}"
                )

    def _physical_tensor(self, name: str, logical: torch.Tensor) -> torch.Tensor:
        import torch

        logical = logical.to(dtype=torch.float16, device="cuda").contiguous()
        if tuple(logical.shape) != _SHAPE:
            raise ValueError(f"{name} shape {tuple(logical.shape)} != {_SHAPE}")
        if self.engine.get_tensor_format(name) == self.trt.TensorFormat.LINEAR:
            return logical
        physical = torch.zeros(
            (_SHAPE[0], _SHAPE[2], _SHAPE[3], _SHAPE[4], 32),
            dtype=torch.float16,
            device="cuda",
        )
        physical[..., : _SHAPE[1]] = logical.permute(0, 2, 3, 4, 1)
        return physical

    def _logical_tensor(self, name: str, physical: torch.Tensor) -> torch.Tensor:
        import torch

        if self.engine.get_tensor_format(name) == self.trt.TensorFormat.LINEAR:
            return physical
        if name == "candidate":
            tail_bits = physical[..., _SHAPE[1] :].view(torch.int16)
            nonzero_tail_bits = int(torch.count_nonzero(tail_bits))
            if nonzero_tail_bits:
                raise RuntimeError(f"candidate DHWC8 tail has {nonzero_tail_bits} nonzero lanes")
        return physical[..., : _SHAPE[1]].permute(0, 4, 1, 2, 3).contiguous()

    def _bind(self, linear: torch.Tensor, skip: torch.Tensor) -> dict[str, torch.Tensor]:
        import torch

        prepared = {
            "linear": self._physical_tensor("linear", linear),
            "skip": self._physical_tensor("skip", skip),
        }
        for name, value in prepared.items():
            if not self.context.set_tensor_address(name, int(value.data_ptr())):
                raise RuntimeError(f"failed to bind standalone input {name}")
        outputs = {}
        for name in self.output_names:
            shape = (
                _SHAPE
                if self.engine.get_tensor_format(name) == self.trt.TensorFormat.LINEAR
                else (_SHAPE[0], _SHAPE[2], _SHAPE[3], _SHAPE[4], 32)
            )
            outputs[name] = torch.empty(shape, dtype=torch.float16, device="cuda")
        for name, value in outputs.items():
            if not self.context.set_tensor_address(name, int(value.data_ptr())):
                raise RuntimeError(f"failed to bind standalone output {name}")
        self._prepared = prepared
        return outputs

    def run_once(self, linear: torch.Tensor, skip: torch.Tensor) -> dict[str, torch.Tensor]:
        import torch

        outputs = self._bind(linear, skip)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("standalone post8 enqueue failed")
        self.stream.synchronize()
        return {name: self._logical_tensor(name, output) for name, output in outputs.items()}


def _cases() -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    yield (
        "random",
        torch.randn(_SHAPE, generator=generator, device="cuda", dtype=torch.float16),
        torch.randn(_SHAPE, generator=generator, device="cuda", dtype=torch.float16),
    )
    elements = 1
    for dimension in _SHAPE:
        elements *= dimension
    structured = torch.linspace(-16.0, 16.0, elements, device="cuda", dtype=torch.float32)
    structured = structured.to(torch.float16).reshape(_SHAPE)
    yield "structured", structured, torch.flip(structured, dims=(1,))
    zeros = torch.zeros(_SHAPE, device="cuda", dtype=torch.float16)
    yield "signed_zero", zeros, -zeros


def _bitwise_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, int | bool]:
    import torch

    candidate_bits = candidate.view(torch.int16)
    reference_bits = reference.view(torch.int16)
    mismatch_count = int((candidate_bits != reference_bits).sum())
    return {
        "elements": candidate.numel(),
        "mismatch_count": mismatch_count,
        "bitwise_equal": mismatch_count == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tile-positions", type=int, default=32)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    tile_positions = _validate_tile_positions(args.tile_positions)

    from tests.e2e.models.fast_foundation_stereo.trt_runner import (
        load_native_plugin_libraries,
    )

    plugin_library = _pin_plugin_library(args.plugin_library)
    loaded = load_native_plugin_libraries([plugin_library])
    if args.build:
        args.engine.parent.mkdir(parents=True, exist_ok=True)
        args.engine.write_bytes(_build_engine(tile_positions))
    if not args.engine.is_file():
        raise FileNotFoundError(args.engine)

    runner = _Runner(args.engine.resolve(), ("reference", "candidate"))
    cases = {}
    for name, linear, skip in _cases():
        outputs = runner.run_once(linear, skip)
        metrics = _bitwise_metrics(outputs["candidate"], outputs["reference"])
        if not metrics["bitwise_equal"]:
            raise RuntimeError(f"{name} failed post8 full-volume bitwise oracle: {metrics}")
        cases[name] = metrics

    receipt = {
        "contract": "control-elementwise-sum-vs-post8-plugin-full-volume-bitwise",
        "shape": list(_SHAPE),
        "tile_positions": tile_positions,
        "plugin_libraries": loaded,
        "plugin_library_sha256": _sha256(plugin_library),
        "engine": str(args.engine.resolve()),
        "engine_sha256": _sha256(args.engine.resolve()),
        "cases": cases,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

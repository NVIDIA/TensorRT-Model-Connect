# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localize the failed full trained decoder's first three graph boundaries.

Reuse campaign 318's complete latent fixture and exact original checkpoint.
Only the unchanged post-quantization/projection/token-packing prefix executes;
no transformer layer, decoded pixel, full Ref2VA result or quality is qualified.
Each baseline/candidate and batch-1/batch-28 prefix is rebuilt three times.
Extra outputs and prefix pruning can change TensorRT tactics: these are
diagnostics, not a replacement for the owning full-video numerical gate.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from families.minimax_h3 import vae_builder as candidate
from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
from families.minimax_h3.tests.test_trained_video_decoder import _ATOL, _RTOL, _SHARDS
from families.minimax_h3.tests.vae_encoder_checkpoint_oracle import CONFIG_SHA256, REVISION


# Isolate convolution from the already-restored FP32 register concatenation.
BASELINE_SHA = "9f9ce39ae65bed193b05f15341aa3f72ff4db0c9b0df188eecce658b1de130c3"
ROWS_SHA = "c9d0c8c9403e95ba3b80ca5f4d4fbfc433aa713e3a25142763524eaebd5eed27"
UPSTREAM_SHA = "4c3c9745ee27d16ff343c4998244bad41cd8f4213f0029cf7ce11ebb6d72ca1b"
KEYS = (
    "post_quant_conv.weight",
    "post_quant_conv.bias",
    "decoder.proj_in.weight",
    "decoder.proj_in.bias",
    "decoder.register_tokens",
)
BOUNDARIES = ("post_quant_conv", "proj_in", "packed_tokens")


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compare(actual, expected):
    assert actual.shape == expected.shape
    # Widen FP16 losslessly; retain and separately compare the real ABI dtype.
    a, b = actual.astype(np.float32), expected.astype(np.float32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = np.abs(a - b)
    return dict(
        shape=list(a.shape),
        actual_dtype=str(actual.dtype),
        expected_dtype=str(expected.dtype),
        dtype_equal=actual.dtype == expected.dtype,
        byte_exact=actual.dtype == expected.dtype and np.array_equal(actual, expected),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        mismatched_elements=int(np.count_nonzero(error > _ATOL + _RTOL * np.abs(b))),
        compared_elements=a.size,
        rtol=_RTOL,
        atol=_ATOL,
    )


def save_array(root, label, value, saved):
    """Retain each unique full array once, recording exact aliases in reports.

    Identical rebuilds reference the already-written complete payload. Nothing
    is deleted, overwritten, rounded, sampled, compressed or hard-linked.
    Different values, shapes or dtypes always create a separate evidence file.
    """
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
    key = (value.dtype.str, tuple(value.shape), digest)
    if key not in saved:
        path = root / (label + ".npy")
        with path.open("xb") as file:
            np.save(file, value, allow_pickle=False)
        saved[key] = path.name
    return dict(
        path=saved[key], content_sha256=digest, dtype=str(value.dtype), shape=list(value.shape)
    )


def build_prefix(module, weights, batch):
    import tensorrt as trt

    real_builder = trt.Builder
    real_linear = module.op.linear
    real_convolution = getattr(module, "_post_quant_conv", None)
    recorded = {}
    tensors = {}

    class CapturedPrefix(Exception):
        pass

    class Builder:
        def __init__(self, logger):
            self.impl = real_builder(logger)
            recorded["builder"] = self.impl

        def create_network(self, flags):
            recorded["network"] = self.impl.create_network(flags)
            return recorded["network"]

        def create_builder_config(self):
            recorded["config"] = self.impl.create_builder_config()
            return recorded["config"]

    def linear(network, *args, **kwargs):
        result = real_linear(network, *args, **kwargs)
        assert len(tensors) < 2
        tensors[BOUNDARIES[len(tensors)]] = result
        return result

    def convolution(network, *args, **kwargs):
        result = real_convolution(network, *args, **kwargs)
        assert not tensors
        tensors["post_quant_conv"] = result
        return result

    def capture(network, value, *_args, **_kwargs):
        tensors["packed_tokens"] = value
        assert tuple(tensors) == BOUNDARIES
        for name, tensor in tensors.items():
            output = network.add_identity(tensor).get_output(0)
            output.name = name
            network.mark_output(output)
        raise CapturedPrefix

    try:
        with (
            patch.object(trt, "Builder", Builder),
            patch.object(module.op, "linear", linear),
            patch.object(module.op, "rms_norm", capture),
            patch.object(module, "_post_quant_conv", convolution)
            if real_convolution is not None
            else nullcontext(),
        ):
            try:
                module.build_vae_tile_decoder_engine(
                    weights, batch_size=batch, workspace_bytes=8 << 30
                )
            except CapturedPrefix:
                pass
            else:
                raise AssertionError("decoder unexpectedly passed the first transformer boundary")
        network, config = recorded["network"], recorded["config"]
        assert not config.get_flag(trt.BuilderFlag.TF32)
        plan = recorded["builder"].build_serialized_network(network, config)
        assert plan is not None
        return bytes(plan)
    finally:
        if "network" in recorded:
            module.op.release_weight_buffers(recorded["network"])


def execute(plan, tiles):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    assert engine.num_io_tensors == 4
    assert context.set_tensor_address("latent_tiles", tiles.data_ptr())
    outputs = {}
    for name in BOUNDARIES:
        dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[
            engine.get_tensor_dtype(name)
        ]
        outputs[name] = torch.empty(
            tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
        )
        assert context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    cold = {name: value.cpu().numpy().copy() for name, value in outputs.items()}
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert all(np.array_equal(cold[name], value.cpu().numpy()) for name, value in outputs.items())
    return cold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--baseline-builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    assert args.checkpoint.name == REVISION and sha(args.rows) == ROWS_SHA
    assert sha(args.baseline_builder) == BASELINE_SHA
    assert sha(args.checkpoint / "vae/config.json") == CONFIG_SHA256
    for name, digest in _SHARDS.items():
        assert sha(args.checkpoint / "vae" / name) == digest
    args.output.mkdir(mode=0o700)
    spec = importlib.util.spec_from_file_location(
        "families.minimax_h3._decoder_baseline", args.baseline_builder
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert sha(inspect.getfile(AutoencoderKLMiniMaxH3)) == UPSTREAM_SHA
    torch.backends.cuda.matmul.allow_tf32 = False
    # The first norm's weight is evaluated as a Python call argument before
    # instrumentation stops graph construction; that norm is never executed.
    state = load_selected_component_state_dict(
        args.checkpoint / "vae", (*KEYS, "decoder.transformer_blocks.0.norm1.weight")
    )
    assert len(state) == 6 and all(v.dtype == torch.float32 for v in state.values())
    config = json.loads((args.checkpoint / "vae/config.json").read_text())
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3.from_config(config)
    loaded = model.load_state_dict(state, strict=False, assign=True)
    assert not loaded.unexpected_keys
    model.post_quant_conv.cuda()
    model.decoder.proj_in.cuda()
    model.decoder.register_tokens = torch.nn.Parameter(
        model.decoder.register_tokens.cuda(), requires_grad=False
    )
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(48, theta=100.0).cuda()
    model.eval()
    normalized = np.random.default_rng(831).normal(0, 1, (1, 24, 37, 84, 48)).astype(np.float32)
    packed = patchify_video_latents(torch.from_numpy(normalized), (1, 2, 2)).numpy()
    assert hashlib.sha256(packed.tobytes()).hexdigest() == ROWS_SHA
    np.testing.assert_array_equal(packed, np.fromfile(args.rows, np.float32).reshape(37296, 96))
    mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 24, 1, 1, 1)
    std = torch.tensor(config["latents_std"], device="cuda").view(1, 24, 1, 1, 1)
    denormalized = torch.from_numpy(normalized).cuda() * std + mean
    ys = model._split_tiles(1344, 256, 64)[0]
    xs = model._split_tiles(768, 256, 64)[0]
    tiles = torch.cat(
        [
            denormalized[:, :, :7, y // 16 : y // 16 + 16, x // 16 : x // 16 + 16]
            for y in ys
            for x in xs
        ]
    ).contiguous()
    assert tiles.shape == (28, 24, 7, 16, 16)
    observed = {}

    class CapturedReference(Exception):
        pass

    def hook(name):
        def capture(_module, _inputs, output):
            if name == "post_quant_conv":
                output = output.permute(0, 2, 3, 4, 1).reshape(-1, 1792, 24)
            observed[name] = output.detach().cpu().numpy().copy()

        return capture

    def packed_hook(_module, inputs):
        observed["packed_tokens"] = inputs[0].detach().cpu().numpy().copy()
        raise CapturedReference

    hooks = [
        model.post_quant_conv.register_forward_hook(hook("post_quant_conv")),
        model.decoder.proj_in.register_forward_hook(hook("proj_in")),
        model.decoder.transformer_blocks[0].register_forward_pre_hook(packed_hook),
    ]

    def reference(value):
        observed.clear()
        try:
            with (
                torch.no_grad(),
                torch.backends.cudnn.flags(allow_tf32=False),
                torch.autocast("cuda", dtype=torch.float16),
            ):
                model.decoder(model.post_quant_conv(value))
        except CapturedReference:
            pass
        else:
            raise AssertionError("upstream unexpectedly executed the first transformer block")
        assert tuple(observed) == BOUNDARIES
        return dict(observed)

    sequential = [reference(tiles[i : i + 1]) for i in range(28)]
    canonical = {
        name: np.concatenate([row[name] for row in sequential], axis=0) for name in BOUNDARIES
    }
    del sequential
    batched = reference(tiles)
    for handle in hooks:
        handle.remove()
    saved, reference_arrays = {}, {}
    for kind, values in [("upstream-sequential", canonical), ("upstream-batched", batched)]:
        reference_arrays[kind] = {}
        for name, value in values.items():
            reference_arrays[kind][name] = save_array(args.output, kind + "." + name, value, saved)
    report = dict(
        source_checkpoint_revision=REVISION,
        source_input_sha256=ROWS_SHA,
        baseline_builder_sha256=BASELINE_SHA,
        candidate_builder_sha256=sha(inspect.getfile(candidate)),
        upstream_source_sha256=UPSTREAM_SHA,
        loaded_trained_tensors=len(state),
        executed_prefix_tensors=len(KEYS),
        tiles_shape=list(tiles.shape),
        repeats_per_variant=3,
        prefix_only=True,
        full_trained_decoder_qualified=False,
        full_ref2va_qualified=False,
        reference_arrays=reference_arrays,
        batch_effect={name: compare(batched[name], canonical[name]) for name in BOUNDARIES},
        executions=[],
    )
    model = None
    del normalized, packed, denormalized, mean, std
    gc.collect()
    weights = numpy_state(state)
    with (args.output / "boundaries.jsonl").open("x") as trace:
        for batch in (1, 28):
            for variant, module in [("baseline", baseline), ("candidate", candidate)]:
                for rebuild in range(3):
                    label = f"{variant}-batch{batch}-rebuild{rebuild}"
                    plan = build_prefix(module, weights, batch)
                    (args.output / (label + ".plan")).write_bytes(plan)
                    actual = execute(plan, tiles[:batch])
                    row = dict(
                        variant=variant,
                        batch=batch,
                        rebuild=rebuild,
                        cold_warm_byte_exact=True,
                        plan_sha256=hashlib.sha256(plan).hexdigest(),
                        plan_bytes=len(plan),
                        boundaries={},
                        actual_arrays={},
                    )
                    for name in BOUNDARIES:
                        row["actual_arrays"][name] = save_array(
                            args.output, label + "." + name, actual[name], saved
                        )
                        row["boundaries"][name] = compare(actual[name], canonical[name][:batch])
                    row["first_dtype_mismatch"] = next(
                        (name for name in BOUNDARIES if not row["boundaries"][name]["dtype_equal"]),
                        None,
                    )
                    row["first_numeric_mismatch"] = next(
                        (
                            name
                            for name in BOUNDARIES
                            if row["boundaries"][name]["mismatched_elements"]
                        ),
                        None,
                    )
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                    report["executions"].append(row)
                    del plan, actual
                    gc.collect()
    assert len(report["executions"]) == 12
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    inventory = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(args.output.iterdir())
        if p.is_file()
    }
    with (args.output / "files.json").open("x") as file:
        json.dump(inventory, file, indent=2, allow_nan=False)
    print(json.dumps(dict(executions=12, prefix_only=True, full_trained_decoder_qualified=False)))


if __name__ == "__main__":
    main()

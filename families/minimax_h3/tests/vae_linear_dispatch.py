# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe actual upstream linear dispatch, without inferring it from layout.

Execute the original first clip through its first trained block at batches
1/28. Retain operator/dtype/stride metadata only. No TensorRT engine is built,
pruned or executed, and no complete-decoder or Ref2VA qualification is inferred.
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import inspect
import json
import os
from pathlib import Path


CASES = ("upstream-linear-path-batch1", "upstream-linear-path-batch28")


def tensor_metadata(value):
    return dict(
        dtype=str(value.dtype),
        device=str(value.device),
        shape=list(value.shape),
        stride=list(value.stride()),
        contiguous=value.is_contiguous(),
    )


@contextmanager
def capture_linear_calls(modules, records):
    """Transparent eager tracing; every hook is removed, including on failure."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    active = None
    handles = []

    def before(name):
        def hook(module, inputs):
            nonlocal active
            assert active is None and len(records) < 32
            active = dict(
                module=name,
                input=tensor_metadata(inputs[0]),
                weight=tensor_metadata(module.weight),
                bias=tensor_metadata(module.bias) if module.bias is not None else None,
                operators=[],
                output=None,
            )
            records.append(active)

        return hook

    def after(_module, _inputs, output):
        nonlocal active
        assert active is not None
        active["output"] = tensor_metadata(output) if isinstance(output, torch.Tensor) else None
        active = None

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if active is not None:
                assert len(active["operators"]) < 256
                active["operators"].append(
                    dict(
                        operator=str(func),
                        schema=str(func._schema),
                        operands=[
                            dict(index=index, **tensor_metadata(value))
                            for index, value in enumerate(args)
                            if isinstance(value, torch.Tensor)
                        ],
                        output=tensor_metadata(output)
                        if isinstance(output, torch.Tensor)
                        else None,
                    )
                )
            return output

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after, always_call=True))
        with Capture():
            yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(dict(cases=list(CASES))))
        return
    assert args.checkpoint is not None and args.rows is not None and args.output is not None
    assert importlib.metadata.version("diffusers") == "0.40.0"

    import numpy as np
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    from families.minimax_h3.checkpoint import load_selected_component_state_dict
    from families.minimax_h3.tests.test_trained_video_decoder import _SHARDS
    from families.minimax_h3.tests.vae_decoder_prefix_localize import (
        KEYS,
        ROWS_SHA,
        UPSTREAM_SHA,
        sha,
    )
    from families.minimax_h3.tests.vae_encoder_checkpoint_oracle import CONFIG_SHA256, REVISION
    from families.minimax_h3.vae_builder import checkpoint_keys

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert args.checkpoint.name == REVISION and sha(args.rows) == ROWS_SHA
    assert sha(inspect.getfile(AutoencoderKLMiniMaxH3)) == UPSTREAM_SHA
    assert sha(args.checkpoint / "vae/config.json") == CONFIG_SHA256
    for name, digest in _SHARDS.items():
        assert sha(args.checkpoint / "vae" / name) == digest
    selected = (
        *KEYS,
        *(name for name in checkpoint_keys() if name.startswith("decoder.transformer_blocks.0.")),
    )
    assert len(selected) == len(set(selected)) == 21
    state = load_selected_component_state_dict(args.checkpoint / "vae", selected)
    assert len(state) == 21 and all(value.dtype == torch.float32 for value in state.values())
    config = json.loads((args.checkpoint / "vae/config.json").read_text())
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3.from_config(config)
    assert len(model.decoder.transformer_blocks) == 36
    expected = model.state_dict()
    assert all(tuple(expected[name].shape) == tuple(value.shape) for name, value in state.items())
    loaded = model.load_state_dict(state, strict=False, assign=True)
    assert not loaded.unexpected_keys and set(loaded.missing_keys) == set(expected) - set(state)
    del expected
    model.post_quant_conv.cuda()
    model.decoder.proj_in.cuda()
    first_block = model.decoder.transformer_blocks[0]
    first_block.cuda()
    model.decoder.register_tokens = torch.nn.Parameter(
        model.decoder.register_tokens.cuda(), requires_grad=False
    )
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(48, theta=100.0).cuda()
    model.eval()
    modules = {"decoder.proj_in": model.decoder.proj_in}
    modules.update(
        {
            "decoder.transformer_blocks.0." + name: module
            for name, module in first_block.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
    )
    assert len(modules) == 7
    normalized = np.random.default_rng(831).normal(0, 1, (1, 24, 37, 84, 48)).astype(np.float32)
    packed = patchify_video_latents(torch.from_numpy(normalized), (1, 2, 2)).numpy()
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
    report = dict(
        cases=[],
        checkpoint_revision=REVISION,
        checkpoint_shards=_SHARDS,
        upstream_source_sha256=UPSTREAM_SHA,
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        linear_flatten_3d_environment=os.environ.get("TORCH_LINEAR_FLATTEN_3D"),
        fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        trained_tensors_loaded=len(state),
        diagnostic_only=True,
        trt_engines_built_or_executed=0,
        full_trained_decoder_qualified=False,
        full_ref2va_qualified=False,
    )

    class FirstBlockObserved(Exception):
        pass

    def stop_after_first_block(_module, _inputs, _output):
        raise FirstBlockObserved

    handle = first_block.register_forward_hook(stop_after_first_block)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    try:
        for label, batch in zip(CASES, (1, 28), strict=True):
            records = []
            with (
                torch.no_grad(),
                torch.backends.cudnn.flags(allow_tf32=False),
                torch.autocast("cuda", dtype=torch.float16),
                capture_linear_calls(modules, records),
            ):
                try:
                    model.decoder(model.post_quant_conv(tiles[:batch]))
                except FirstBlockObserved:
                    pass
                else:
                    raise AssertionError("upstream passed the declared first-block stop")
            assert len(records) == len(modules)
            assert {row["module"] for row in records} == set(modules)
            assert all(row["operators"] and row["output"] is not None for row in records)
            report["cases"].append(
                dict(
                    case=label,
                    tiles=list(tiles[:batch].shape),
                    trained_blocks_executed=1,
                    linears=records,
                )
            )
    finally:
        handle.remove()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    args.output.mkdir(mode=0o700)
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(dict(cases=list(CASES), linears_per_trace=len(modules), diagnostic_only=True)))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the first failed trained-video boundary without rebuilding engines.

Use the complete original 28-tile first clip from the failed full-video case,
the original trained convolution and retained prefix results. Counterfactuals
are diagnostics, not proposed model changes or full-decoder qualification.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np

from families.minimax_h3.checkpoint import load_selected_component_state_dict
from families.minimax_h3.tests.test_trained_video_decoder import _SHARDS
from families.minimax_h3.tests.vae_decoder_prefix_localize import (
    ROWS_SHA,
    UPSTREAM_SHA,
    compare,
    save_array,
    sha,
)
from families.minimax_h3.tests.vae_encoder_checkpoint_oracle import CONFIG_SHA256, REVISION


VARIANTS = (
    "original-autocast",
    "explicit-half-convolution",
    "half-operands-float-accumulation",
    "half-convolution-separate-bias",
    "original-float-convolution",
)
CASES = tuple(
    f"{variant}-batch{batch}-repeat{repeat}"
    for batch in (1, 28)
    for variant in VARIANTS
    for repeat in range(3)
)
PRIOR_REPORT_SHA = "17d4fabac122811b4565e26c34d38aa432f4b6b66f0039a2e627b5be04c8803c"


def checked_array(root, record):
    """Verify retained full values, shape and dtype before using any evidence."""
    name = Path(record["path"])
    assert name.name == str(name) and not name.is_absolute()
    value = np.load(root / name, allow_pickle=False)
    assert str(value.dtype) == record["dtype"] and list(value.shape) == record["shape"]
    assert (
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        == record["content_sha256"]
    )
    assert np.isfinite(value).all()
    return value


def convolution_counterfactual(variant, module, value):
    import torch
    from torch.nn import functional as functional

    if variant == "original-autocast":
        with torch.autocast("cuda", dtype=torch.float16):
            return module(value)
    with torch.autocast("cuda", enabled=False):
        weight, bias = module.weight, module.bias
        if variant == "explicit-half-convolution":
            return functional.conv3d(value.half(), weight.half(), bias.half())
        if variant == "half-operands-float-accumulation":
            return functional.conv3d(
                value.half().float(), weight.half().float(), bias.half().float()
            ).half()
        if variant == "half-convolution-separate-bias":
            return functional.conv3d(value.half(), weight.half()) + bias.half().view(1, -1, 1, 1, 1)
        if variant == "original-float-convolution":
            return functional.conv3d(value, weight, bias).half()
    raise ValueError(f"Unknown convolution counterfactual: {variant}")


def capture_dispatch(module, value):
    """Observe actual ATen operands after autocast, not Python module inputs."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    events, tensors = [], {}

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if "convolution" in name or "_to_copy" in name:
                operands = []
                for index, operand in enumerate(args):
                    if isinstance(operand, torch.Tensor):
                        label = f"event{len(events)}-arg{index}"
                        tensors[label] = operand
                        operands.append(
                            dict(
                                index=index,
                                label=label,
                                dtype=str(operand.dtype),
                                shape=list(operand.shape),
                                stride=list(operand.stride()),
                                device=str(operand.device),
                            )
                        )
                events.append(dict(operator=name, schema=str(func._schema), operands=operands))
            return func(*args, **(kwargs or {}))

    with Capture(), torch.no_grad():
        output = convolution_counterfactual("original-autocast", module, value)
    assert any("convolution" in event["operator"] for event in events)
    # Copies and hashes happen after leaving dispatch, avoiding recursive tracing.
    arrays = {name: operand.detach().cpu().numpy().copy() for name, operand in tensors.items()}
    return events, arrays, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    assert args.checkpoint.name == REVISION and sha(args.rows) == ROWS_SHA
    assert sha(args.checkpoint / "vae/config.json") == CONFIG_SHA256
    assert sha(args.prior / "report.json") == PRIOR_REPORT_SHA
    for name, digest in _SHARDS.items():
        assert sha(args.checkpoint / "vae" / name) == digest
    prior = json.loads((args.prior / "report.json").read_text())
    expected = checked_array(
        args.prior, prior["reference_arrays"]["upstream-sequential"]["post_quant_conv"]
    )
    assert expected.shape == (28, 1792, 24) and expected.dtype == np.float16
    retained = [
        (row, checked_array(args.prior, row["actual_arrays"]["post_quant_conv"]))
        for row in prior["executions"]
    ]
    assert len(retained) == 12

    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert sha(inspect.getfile(AutoencoderKLMiniMaxH3)) == UPSTREAM_SHA
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = load_selected_component_state_dict(
        args.checkpoint / "vae", ("post_quant_conv.weight", "post_quant_conv.bias")
    )
    assert len(state) == 2 and all(t.dtype == torch.float32 for t in state.values())
    config = json.loads((args.checkpoint / "vae/config.json").read_text())
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3.from_config(config)
    loaded = model.load_state_dict(state, strict=False, assign=True)
    assert not loaded.unexpected_keys
    module = model.post_quant_conv.cuda().eval()
    normalized = np.random.default_rng(831).normal(0, 1, (1, 24, 37, 84, 48)).astype(np.float32)
    packed = patchify_video_latents(torch.from_numpy(normalized), (1, 2, 2)).numpy()
    assert hashlib.sha256(packed.tobytes()).hexdigest() == ROWS_SHA
    np.testing.assert_array_equal(packed, np.fromfile(args.rows, np.float32).reshape(37296, 96))
    mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 24, 1, 1, 1)
    std = torch.tensor(config["latents_std"], device="cuda").view(1, 24, 1, 1, 1)
    denormalized = torch.from_numpy(normalized).cuda() * std + mean
    ys, xs = model._split_tiles(1344, 256, 64)[0], model._split_tiles(768, 256, 64)[0]
    tiles = torch.cat(
        [
            denormalized[:, :, :7, y // 16 : y // 16 + 16, x // 16 : x // 16 + 16]
            for y in ys
            for x in xs
        ]
    ).contiguous()
    assert tiles.shape == (28, 24, 7, 16, 16)
    args.output.mkdir(mode=0o700)
    saved = {}
    report = dict(
        checkpoint_revision=REVISION,
        checkpoint_shards=_SHARDS,
        source_input_sha256=ROWS_SHA,
        prior_report_sha256=PRIOR_REPORT_SHA,
        upstream_source_sha256=UPSTREAM_SHA,
        loaded_trained_tensors=2,
        tiles_shape=list(tiles.shape),
        TF32=False,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        diagnostic_only=True,
        full_trained_decoder_qualified=False,
        full_ref2va_qualified=False,
        dispatch={},
        executions=[],
    )
    with (
        torch.no_grad(),
        torch.backends.cudnn.flags(allow_tf32=False),
        (args.output / "comparisons.jsonl").open("x") as trace,
    ):
        for batch in (1, 28):
            events, arrays, traced = capture_dispatch(module, tiles[:batch])
            assert traced.dtype == torch.float16
            report["dispatch"][str(batch)] = dict(
                events=events,
                operands={
                    name: save_array(args.output, f"batch{batch}-{name}", array, saved)
                    for name, array in arrays.items()
                },
            )
            for variant in VARIANTS:
                for repeat in range(3):
                    label = f"{variant}-batch{batch}-repeat{repeat}"
                    output = convolution_counterfactual(variant, module, tiles[:batch])
                    assert output.dtype == torch.float16
                    actual = output.permute(0, 2, 3, 4, 1).reshape(batch, 1792, 24)
                    actual = actual.cpu().numpy().copy()
                    comparison = compare(actual, expected[:batch])
                    if variant == "original-autocast":
                        assert comparison["byte_exact"], "original reference no longer reproduces"
                    row = dict(
                        case=label,
                        variant=variant,
                        batch=batch,
                        repeat=repeat,
                        output=save_array(args.output, label, actual, saved),
                        upstream=comparison,
                        retained_trt=[
                            dict(
                                variant=old["variant"],
                                rebuild=old["rebuild"],
                                plan_sha256=old["plan_sha256"],
                                comparison=compare(actual, value),
                            )
                            for old, value in retained
                            if old["batch"] == batch
                        ],
                    )
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()
                    report["executions"].append(row)
    assert tuple(row["case"] for row in report["executions"]) == CASES
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    inventory = {
        path.name: dict(bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    with (args.output / "files.json").open("x") as file:
        json.dump(inventory, file, indent=2, allow_nan=False)
    print(json.dumps(dict(cases=len(CASES), diagnostic_only=True, full_ref2va_qualified=False)))


if __name__ == "__main__":
    main()

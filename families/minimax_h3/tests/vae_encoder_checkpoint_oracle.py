# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit trained-weight tile comparison, not the public family E2E test.

Run after the synthetic GPU oracle. Only the exact cached original checkpoint
is accepted. No download, decoder construction or whole-video generation is
performed. Reports preserve the comparison arrays even on a numerical failure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
from families.minimax_h3.vae_encoder_builder import (
    VideoEncoderConfig,
    build_vae_tile_encoder_engine,
    checkpoint_keys,
    checkpoint_shapes,
)


REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
CONFIG_SHA256 = "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745"
RTOL, ATOL = 1.3e-6, 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-image", required=True, type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    assert args.checkpoint.name == REVISION
    vae = args.checkpoint / "vae"
    config_bytes = (vae / "config.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == CONFIG_SHA256
    raw_config = json.loads(config_bytes)
    config = VideoEncoderConfig()
    for name in (
        "in_channels",
        "latent_channels",
        "layers_per_block",
        "norm_num_groups",
        "norm_eps",
    ):
        assert raw_config[name] == getattr(config, name), name
    for name in ("block_out_channels", "spatial_downsample_factors", "temporal_downsample_factors"):
        assert tuple(raw_config[name]) == getattr(config, name), name
    assert hashlib.sha256(args.reference_image.read_bytes()).hexdigest() == (
        "b1257de01d8440baea06865ea35ce557d885b6dda700281ba25202193c58285c"
    )
    args.output.mkdir(mode=0o700)

    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d
    from PIL import Image

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    # Preserve a real tile of the reference input without downsampling. Whole
    # reference resizing/tiling is outside this component comparison.
    with Image.open(args.reference_image) as image:
        pixels = np.array(image.convert("RGB"), copy=True)[:256, :256]
    assert pixels.shape == (256, 256, 3)
    pixels = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].float()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    pixels = ((pixels / 255.0 - mean) / std).contiguous()
    np.save(args.output / "pixels.npy", pixels.numpy(), allow_pickle=False)
    started = time.perf_counter()
    state = load_selected_component_state_dict(vae, checkpoint_keys())
    assert {name: tuple(value.shape) for name, value in state.items()} == checkpoint_shapes()
    assert all(value.dtype == torch.float32 for value in state.values())
    torch.cuda.reset_peak_memory_stats()
    with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
        with torch.device("meta"):
            original = torch.nn.Module()
            original.encoder = MiniMaxH3VideoEncoder3d()
            original.quant_conv = torch.nn.Conv3d(48, 48, 1)
        original.load_state_dict(state, strict=True, assign=True)
        original.eval().cuda()
        device_pixels = pixels.cuda()
        expected = original.quant_conv(original.encoder(device_pixels)).cpu().numpy()
        del original
        gc.collect()
        torch.cuda.empty_cache()
        # Selective CPU checkpoint tensors remain alive until serialization.
        plan_started = time.perf_counter()
        plan = build_vae_tile_encoder_engine(numpy_state(state), max_frames=1)
        build_seconds = time.perf_counter() - plan_started
        del state
        gc.collect()
        with (args.output / "tile-encoder-fp32.plan").open("xb") as stream:
            stream.write(plan)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        context = engine.create_execution_context()
        assert context is not None
        assert context.set_input_shape("pixels", tuple(device_pixels.shape))
        output = torch.empty(
            tuple(context.get_tensor_shape("moments")), device="cuda", dtype=torch.float32
        )
        assert context.set_tensor_address("pixels", device_pixels.data_ptr())
        assert context.set_tensor_address("moments", output.data_ptr())
        stream = torch.cuda.current_stream()
        infer_started = time.perf_counter()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        inference_seconds = time.perf_counter() - infer_started
        actual = output.cpu().numpy()
    np.save(args.output / "upstream-moments.npy", expected, allow_pickle=False)
    np.save(args.output / "native-moments.npy", actual, allow_pickle=False)
    assert actual.shape == expected.shape == (1, 48, 1, 16, 16)
    error = np.abs(actual - expected)
    allowed = ATOL + RTOL * np.abs(expected)
    report = dict(
        checkpoint_revision=REVISION,
        config_sha256=CONFIG_SHA256,
        tensor_count=len(checkpoint_keys()),
        tile_shape=list(pixels.shape),
        moments_shape=list(actual.shape),
        precision="fp32",
        tf32=False,
        rtol=RTOL,
        atol=ATOL,
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        mismatched_elements=int(np.count_nonzero(error > allowed)),
        compared_elements=int(error.size),
        plan_bytes=len(plan),
        plan_sha256=hashlib.sha256(plan).hexdigest(),
        build_seconds=build_seconds,
        inference_seconds=inference_seconds,
        total_seconds=time.perf_counter() - started,
        # These are Torch allocator observations, not total TRT/driver peaks.
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        tile_moments_only=True,
        full_reference_latents_qualified=False,
        native_Ref2VA_video_qualified=False,
        full_app_qualified=False,
    )
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

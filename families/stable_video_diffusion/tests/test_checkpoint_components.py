# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call of each engine built from the real checkpoint against its reference module.

Runs only with TRTMC_STABLE_VIDEO_DIFFUSION_MODEL_DIR pointing at the checkpoint
and a CUDA GPU. Shapes are reduced (the graphs are shape-specialized, the weights
are the real ones). The oracles are relative to the reference itself: an FP32
engine's error against a float64 run of the reference must stay within a small
factor of the reference's own FP32 error, and an FP16 engine's error against the
reference's FP32 run within a per-component factor of the reference's own FP16
error.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from families.stable_video_diffusion import clip_builder, unet_builder, vae_builder  # noqa: E402
from families.stable_video_diffusion import config as config_module  # noqa: E402
from families.stable_video_diffusion.model import _compile, _load_weights  # noqa: E402

MODEL_DIR = os.environ.get("TRTMC_STABLE_VIDEO_DIFFUSION_MODEL_DIR")
pytestmark = pytest.mark.skipif(
    not MODEL_DIR or not torch.cuda.is_available(),
    reason="needs TRTMC_STABLE_VIDEO_DIFFUSION_MODEL_DIR and a CUDA GPU")
FRAMES = 3
LATENT = (8, 16)
# Allowed ratio of the engine's error to the reference's own error at the same precision.
# An FP16 engine rounds along other paths than the reference, so its error is of the order
# of the reference's FP16 error, not equal to it. Each FP16 bound lies between the healthy
# engine and the same engine with plain FP16 linear and convolution layers, whose
# FP16-accumulating tactics it must reject (relative L2 ratios, healthy / plain FP16):
# image encoder 1.5 / 2.9 (1.5 with every attention and activation precision tried),
# VAE encoder 0.8 / 2.1, VAE decoder 0.8 / 2.9. The UNet measures 1.0 / 1.1 at these
# shapes, so its bound only limits general precision loss; FP16 accumulation in the UNet
# is caught by the end-to-end FP16 shortfall bound.
FP32_RATIO = 2.0
FP16_RATIOS = {"image encoder": 2.0, "VAE encoder": 1.5, "VAE decoder": 1.5, "UNet": 1.5}


@pytest.fixture(autouse=True)
def _exact_fp32():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _run(plan: bytes, inputs: dict, outputs: dict) -> dict:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    keep = {}
    for name, value in inputs.items():
        keep[name] = value.detach().float().contiguous().cuda()
        context.set_tensor_address(name, keep[name].data_ptr())
    results = {}
    for name, shape in outputs.items():
        results[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(name, results[name].data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return results


def _relative_error(actual, expected) -> float:
    return float((actual.double() - expected.double()).abs().max() / expected.double().abs().max())


def _relative_l2(actual, expected) -> float:
    return float((actual.double() - expected.double()).norm() / expected.double().norm())


def _work(precision: str):
    return trt.float16 if precision == "fp16" else trt.float32


def _reference_run(module, call, precision: str):
    """(reference output at the engine's precision, the stricter reference it is judged against)."""
    with torch.no_grad():
        float32 = call(module.float())
        if precision == "fp32":
            return float32, call(module.double())
        return call(module.half()), float32


def _check(name: str, precision: str, actual, reference, baseline) -> None:
    """FP32 is judged on the largest error; FP16 on the relative L2 error, which does not
    swing with the single worst element as FP16 tactics change between builds."""
    errors = {metric: (function(actual, baseline), function(reference, baseline))
              for metric, function in (("max", _relative_error), ("l2", _relative_l2))}
    print(f"{name} {precision}: " + ", ".join(
        f"{metric} engine {engine:.3e} reference {ref:.3e}" for metric, (engine, ref) in errors.items()))
    if precision == "fp32":
        engine, ref = errors["max"]
        assert engine <= FP32_RATIO * ref
    else:
        engine, ref = errors["l2"]
        assert engine <= FP16_RATIOS[name] * ref


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_image_encoder_matches_the_checkpoint(precision):
    model_dir = Path(MODEL_DIR)
    cfg = config_module.resolve(model_dir)
    weights = _load_weights(model_dir, "image_encoder", "model.safetensors")
    work = _work(precision)
    plan = _compile(lambda network: clip_builder.build_image_encoder(
        network, weights, cfg["image_encoder"], work), work=work, verbose=False)
    module = transformers.CLIPVisionModelWithProjection.from_pretrained(
        model_dir / "image_encoder", torch_dtype=torch.float32).cuda().eval()
    pixels = torch.randn((1, 3, 224, 224), generator=torch.Generator().manual_seed(1)).cuda()
    reference, baseline = _reference_run(
        module, lambda m: m(pixels.to(next(m.parameters()).dtype)).image_embeds, precision)
    actual = _run(plan, {"pixel_values": pixels}, {"image_embeds": tuple(baseline.shape)})
    _check("image encoder", precision, actual["image_embeds"], reference, baseline)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_vae_matches_the_checkpoint(precision):
    model_dir = Path(MODEL_DIR)
    cfg = config_module.resolve(model_dir)
    weights = _load_weights(model_dir, "vae", "diffusion_pytorch_model.safetensors")
    height, width = LATENT[0] * 8, LATENT[1] * 8
    work = _work(precision)
    encoder_plan = _compile(lambda network: vae_builder.build_encoder(
        network, weights, cfg["vae"], height, width, work), work=work, verbose=False)
    decoder_plan = _compile(lambda network: vae_builder.build_decoder(
        network, weights, cfg["vae"], FRAMES, *LATENT, work), work=work, verbose=False)
    module = diffusers.AutoencoderKLTemporalDecoder.from_pretrained(
        model_dir / "vae", torch_dtype=torch.float32).cuda().eval()
    generator = torch.Generator().manual_seed(2)
    image = torch.rand((1, 3, height, width), generator=generator).cuda() * 2 - 1
    latents = torch.randn((FRAMES, 4, *LATENT), generator=generator).cuda()

    def dtype(m):
        return next(m.parameters()).dtype

    reference, baseline = _reference_run(
        module, lambda m: m.encode(image.to(dtype(m))).latent_dist.mode(), precision)
    actual = _run(encoder_plan, {"image": image}, {"image_latent": tuple(baseline.shape)})
    _check("VAE encoder", precision, actual["image_latent"], reference, baseline)
    reference, baseline = _reference_run(
        module, lambda m: m.decode(latents.to(dtype(m)), num_frames=FRAMES).sample, precision)
    actual = _run(decoder_plan, {"latents": latents}, {"frames": tuple(baseline.shape)})
    _check("VAE decoder", precision, actual["frames"], reference, baseline)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_unet_matches_the_checkpoint(precision):
    model_dir = Path(MODEL_DIR)
    cfg = config_module.resolve(model_dir)
    weights = _load_weights(model_dir, "unet", "diffusion_pytorch_model.safetensors")
    work = _work(precision)
    plan = _compile(lambda network: unet_builder.build_unet(
        network, weights, cfg["unet"], FRAMES, *LATENT, work), work=work, verbose=False)
    module = diffusers.UNetSpatioTemporalConditionModel.from_pretrained(
        model_dir / "unet", torch_dtype=torch.float32).cuda().eval()
    generator = torch.Generator().manual_seed(3)
    sample = torch.randn((2, FRAMES, 8, *LATENT), generator=generator).cuda()
    context = torch.randn((2, 1, 1024), generator=generator).cuda()
    context[0] = 0.0
    sample[0, :, 4:] = 0.0
    timestep = torch.tensor([0.25 * math.log(30.0)], dtype=torch.float32).cuda()
    time_ids = torch.tensor([[6.0, 127.0, 0.02]], dtype=torch.float32).cuda()

    def call(m):
        dtype = next(m.parameters()).dtype
        return m(sample.to(dtype), timestep[0].to(dtype), encoder_hidden_states=context.to(dtype),
                 added_time_ids=time_ids.repeat(2, 1).to(dtype)).sample

    reference, baseline = _reference_run(module, call, precision)
    actual = _run(plan, {"sample": sample, "timestep": timestep, "encoder_hidden_states": context,
                         "added_time_ids": time_ids}, {"noise_pred": tuple(baseline.shape)})
    _check("UNet", precision, actual["noise_pred"], reference, baseline)

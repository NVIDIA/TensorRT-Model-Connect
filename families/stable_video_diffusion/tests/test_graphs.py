# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each TensorRT graph against its diffusers / transformers module, on small random models.

The modules keep the published block layout (four UNet levels with three
cross-attention levels, two VAE resnets per level, the CLIP pre/post norms) at
reduced widths. Every parameter is randomized, including the AlphaBlender mix
factors and the norm affines, so no path hides behind an identity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from families.stable_video_diffusion import clip_builder, unet_builder, vae_builder  # noqa: E402
from families.stable_video_diffusion.model import _compile  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

FRAMES = 3


@pytest.fixture(autouse=True)
def _exact_fp32():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _randomize(module, seed: int):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            if name.endswith("mix_factor"):
                parameter.copy_(noise)
            elif parameter.ndim == 1 and "norm" in name and name.endswith("weight"):
                parameter.copy_(1.0 + 0.1 * noise)
            elif parameter.ndim == 1:
                parameter.copy_(0.05 * noise)
            else:
                fan_in = parameter[0].numel()
                parameter.copy_(noise / math.sqrt(fan_in))
    return module.eval()


def _weights(module) -> dict:
    return {key: value.detach().cpu().float().numpy() for key, value in module.state_dict().items()}


def _run(plan: bytes, inputs: dict, outputs: dict) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
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
    return float((actual.float() - expected.float()).abs().max() / expected.float().abs().max())


@pytest.mark.parametrize(("precision", "tolerance"), [("fp32", 5e-5), ("fp16", 2e-2)])
def test_clip_image_encoder_graph(precision, tolerance):
    config = transformers.CLIPVisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        image_size=56, patch_size=14, projection_dim=48, hidden_act="gelu", layer_norm_eps=1e-5)
    model = _randomize(transformers.CLIPVisionModelWithProjection(config), 1).cuda()
    cfg = {"image_size": 56, "patch_size": 14, "hidden_size": 64, "num_hidden_layers": 2,
           "num_attention_heads": 4, "projection_dim": 48, "layer_norm_eps": 1e-5}
    work = trt.float16 if precision == "fp16" else trt.float32
    weights = _weights(model)
    plan = _compile(lambda network: clip_builder.build_image_encoder(network, weights, cfg, work),
                    work=work, verbose=False)
    pixels = torch.randn((1, 3, 56, 56), generator=torch.Generator().manual_seed(2)).cuda()
    with torch.no_grad():
        expected = model(pixels).image_embeds
    actual = _run(plan, {"pixel_values": pixels}, {"image_embeds": (1, 48)})["image_embeds"]
    assert _relative_error(actual, expected) <= tolerance


def _tiny_vae():
    model = diffusers.AutoencoderKLTemporalDecoder(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 3,
        block_out_channels=(32, 64, 64), layers_per_block=2, latent_channels=4)
    return _randomize(model, 3).cuda()


@pytest.mark.parametrize(("precision", "tolerance"), [("fp32", 5e-5), ("fp16", 2e-2)])
def test_vae_encoder_graph(precision, tolerance):
    model = _tiny_vae()
    cfg = {"in_channels": 3, "block_out_channels": [32, 64, 64], "layers_per_block": 2,
           "latent_channels": 4}
    work = trt.float16 if precision == "fp16" else trt.float32
    weights = _weights(model)
    plan = _compile(lambda network: vae_builder.build_encoder(network, weights, cfg, 32, 48, work),
                    work=work, verbose=False)
    image = torch.rand((1, 3, 32, 48), generator=torch.Generator().manual_seed(4)).cuda() * 2 - 1
    with torch.no_grad():
        expected = model.encode(image).latent_dist.mode()
    actual = _run(plan, {"image": image}, {"image_latent": (1, 4, 8, 12)})["image_latent"]
    assert _relative_error(actual, expected) <= tolerance


@pytest.mark.parametrize(("precision", "tolerance"), [("fp32", 5e-5), ("fp16", 2e-2)])
def test_temporal_vae_decoder_graph(precision, tolerance):
    model = _tiny_vae()
    cfg = {"in_channels": 3, "block_out_channels": [32, 64, 64], "layers_per_block": 2,
           "latent_channels": 4}
    work = trt.float16 if precision == "fp16" else trt.float32
    weights = _weights(model)
    plan = _compile(lambda network: vae_builder.build_decoder(network, weights, cfg, FRAMES, 8, 12,
                                                              work), work=work, verbose=False)
    latents = torch.randn((FRAMES, 4, 8, 12), generator=torch.Generator().manual_seed(5)).cuda()
    with torch.no_grad():
        expected = model.decode(latents, num_frames=FRAMES).sample
    actual = _run(plan, {"latents": latents}, {"frames": (FRAMES, 3, 32, 48)})["frames"]
    assert _relative_error(actual, expected) <= tolerance


@pytest.mark.parametrize(("precision", "tolerance"), [("fp32", 5e-5), ("fp16", 2e-2)])
def test_spatio_temporal_unet_graph(precision, tolerance):
    model = diffusers.UNetSpatioTemporalConditionModel(
        sample_size=16, in_channels=8, out_channels=4,
        down_block_types=("CrossAttnDownBlockSpatioTemporal",) * 3 + ("DownBlockSpatioTemporal",),
        up_block_types=("UpBlockSpatioTemporal",) + ("CrossAttnUpBlockSpatioTemporal",) * 3,
        block_out_channels=(32, 32, 64, 64), addition_time_embed_dim=16,
        projection_class_embeddings_input_dim=48, layers_per_block=2, cross_attention_dim=24,
        transformer_layers_per_block=1, num_attention_heads=(1, 2, 2, 4), num_frames=FRAMES)
    model = _randomize(model, 6).cuda()
    cfg = {"in_channels": 8, "out_channels": 4, "block_out_channels": [32, 32, 64, 64],
           "down_block_types": list(model.config.down_block_types),
           "up_block_types": list(model.config.up_block_types), "layers_per_block": 2,
           "num_attention_heads": [1, 2, 2, 4], "cross_attention_dim": 24,
           "addition_time_embed_dim": 16, "added_time_ids": 3, "num_frames": FRAMES}
    work = trt.float16 if precision == "fp16" else trt.float32
    weights = _weights(model)
    plan = _compile(lambda network: unet_builder.build_unet(network, weights, cfg, FRAMES, 16, 24,
                                                            work), work=work, verbose=False)
    generator = torch.Generator().manual_seed(7)
    sample = torch.randn((2, FRAMES, 8, 16, 24), generator=generator).cuda()
    context = torch.randn((2, 1, 24), generator=generator).cuda()
    context[0] = 0.0
    timestep = torch.tensor([0.25 * math.log(30.0)], dtype=torch.float32).cuda()
    time_ids = torch.tensor([[6.0, 127.0, 0.02]], dtype=torch.float32).cuda()
    with torch.no_grad():
        expected = model(sample, timestep[0], encoder_hidden_states=context,
                         added_time_ids=time_ids.repeat(2, 1)).sample
    actual = _run(plan, {"sample": sample, "timestep": timestep, "encoder_hidden_states": context,
                         "added_time_ids": time_ids},
                  {"noise_pred": (2, FRAMES, 4, 16, 24)})["noise_pred"]
    assert _relative_error(actual, expected) <= tolerance


@pytest.mark.parametrize("shape", [(6, 32, 16, 24), (6, 32, 8, 12), (6, 32, 2, 3),
                                   (2, 32, 3, 4, 6), (2, 64, 5, 9, 16)])
def test_group_norm_matches_torch_for_small_and_large_groups(shape):
    from families.stable_video_diffusion import graph

    generator = torch.Generator().manual_seed(8)
    values = (torch.randn(shape, generator=generator) * 3 + 1).cuda()
    gamma = torch.randn(shape[1], generator=generator)
    beta = torch.randn(shape[1], generator=generator)
    expected = torch.nn.functional.group_norm(values, 32, gamma.cuda(), beta.cuda(), 1e-6)

    def populate(network):
        out = graph.group_norm(network, network.add_input("x", trt.float32, shape), gamma.numpy(),
                               beta.numpy(), 32, 1e-6, trt.float32)
        out.name = "y"
        network.mark_output(out)

    actual = _run(_compile(populate, work=trt.float32, verbose=False), {"x": values},
                  {"y": shape})["y"]
    assert _relative_error(actual, expected) <= 1e-6


def test_fp16_vae_attention_keeps_fp32_scores_and_sums():
    """The VAE's one-head, 512-wide attention over the 9216 positions of a 576 x 1024 frame,
    in an FP16 network.

    TensorRT has no fused attention kernel at this head width. Its decomposition of an
    attention layer accumulates the score and value products in FP16 and runs the softmax in
    FP16: a nearly uniform softmax then comes out percent-level wrong. diffusers' FP16
    attention keeps scores, softmax and sums in FP32, and so must the graph.
    """
    from families.stable_video_diffusion import blocks
    from families.stable_video_diffusion import graph as g

    channels, height, width, prefix, eps = 512, 96, 96, "attention", 1e-6
    generator = torch.Generator().manual_seed(10)
    x = 0.01 * torch.randn((1, channels, height, width), generator=generator)
    eye = torch.eye(channels)
    # Tiny query and key projections give a nearly uniform softmax; the value bias keeps every
    # value near 1, so the result is a mean over all positions that an FP16 sum cannot hold.
    weights = {
        f"{prefix}.group_norm.weight": torch.ones(channels),
        f"{prefix}.group_norm.bias": torch.zeros(channels),
        f"{prefix}.to_q.weight": 1e-3 * torch.randn((channels, channels), generator=generator),
        f"{prefix}.to_q.bias": torch.zeros(channels),
        f"{prefix}.to_k.weight": 1e-3 * torch.randn((channels, channels), generator=generator),
        f"{prefix}.to_k.bias": torch.zeros(channels),
        f"{prefix}.to_v.weight": 0.1 * torch.randn((channels, channels), generator=generator)
        / math.sqrt(channels),
        f"{prefix}.to_v.bias": torch.ones(channels),
        f"{prefix}.to_out.0.weight": eye,
        f"{prefix}.to_out.0.bias": torch.zeros(channels),
    }
    weights = {key: value.half().float() for key, value in weights.items()}

    def populate(network):
        image = g.cast(network, network.add_input("x", trt.float32, tuple(x.shape)), trt.float16)
        out = g.cast(network, blocks.map_attention(
            network, image, {key: value.numpy() for key, value in weights.items()}, prefix, eps,
            trt.float16), trt.float32)
        out.name = "y"
        network.mark_output(out)

    actual = _run(_compile(populate, work=trt.float16, verbose=False), {"x": x.half().float()},
                  {"y": tuple(x.shape)})["y"]
    exact = {key: value.double() for key, value in weights.items()}

    def project(name, values):
        return values @ exact[f"{prefix}.{name}.weight"].T + exact[f"{prefix}.{name}.bias"]

    with torch.no_grad():
        image = x.half().double()
        normed = torch.nn.functional.group_norm(image, 32, exact[f"{prefix}.group_norm.weight"],
                                                exact[f"{prefix}.group_norm.bias"], eps)
        tokens = normed.flatten(2).transpose(1, 2)
        scores = project("to_q", tokens) @ project("to_k", tokens).transpose(1, 2)
        attended = torch.softmax(scores / math.sqrt(channels), dim=-1) @ project("to_v", tokens)
        expected = project("to_out.0", attended).transpose(1, 2).reshape(image.shape) + image
    # Measured on a B300, seeds 10-12: at most 9.9e-4 with FP32 scores, softmax and sums;
    # 3.3e-2 to 3.4e-2 with the decomposed FP16 attention layer the graph used before.
    assert _relative_error(actual.cpu(), expected) <= 4e-3


def test_fp16_geglu_keeps_a_product_beyond_the_fp16_range():
    """An FP16 GEGLU feed-forward whose hidden and gate values are near 300: their product,
    about 90000, exceeds the FP16 maximum (65504) while the projected result (about 1400)
    does not. An FP16 product turns it into infinity."""
    from families.stable_video_diffusion import graph as g

    tokens, width, inner, prefix = 4, 8, 16, "ff"
    generator = torch.Generator().manual_seed(12)
    x = (1.0 + 0.01 * torch.randn((1, tokens, width), generator=generator)).half().float()
    weights = {
        f"{prefix}.net.0.proj.weight": torch.full((2 * inner, width), 37.5),
        f"{prefix}.net.0.proj.bias": torch.zeros(2 * inner),
        f"{prefix}.net.2.weight": torch.full((width, inner), 1.0 / 1024),
        f"{prefix}.net.2.bias": torch.zeros(width),
    }

    def populate(network):
        tensor = g.cast(network, network.add_input("x", trt.float32, tuple(x.shape)), trt.float16)
        out = g.cast(network, g.geglu_feed_forward(
            network, tensor, {key: value.numpy() for key, value in weights.items()}, prefix,
            trt.float16), trt.float32)
        out.name = "y"
        network.mark_output(out)

    actual = _run(_compile(populate, work=trt.float16, verbose=False), {"x": x},
                  {"y": (1, tokens, width)})["y"].cpu()
    exact = {key: value.double() for key, value in weights.items()}
    # The graph rounds the projection to FP16, as the reference module stores it.
    projected = (x.double() @ exact[f"{prefix}.net.0.proj.weight"].T
                 + exact[f"{prefix}.net.0.proj.bias"]).half().double()
    hidden, gate = projected[..., :inner], projected[..., inner:]
    product = hidden * torch.nn.functional.gelu(gate)
    expected = product @ exact[f"{prefix}.net.2.weight"].T + exact[f"{prefix}.net.2.bias"]
    assert float(product.min()) > 65504.0
    assert bool(torch.isfinite(actual).all()), actual
    # What remains: a TF32 kernel's rounding of the product (2^-11, or 2^-10 if it truncates)
    # and the FP16 rounding of the ~1400 result (3.6e-4). Measured on a B300: 7.8e-5.
    assert _relative_error(actual, expected) <= 2e-3


def test_fp16_unet_attention_sums_in_fp32():
    """The UNet's FP16 self-attention at its full-size shape: five 64-wide heads over the 9216
    positions of a 576 x 1024 frame. TensorRT runs it in its fused attention kernel, which
    accumulates in FP32, so a uniform softmax returns the mean of the values to FP16 output
    rounding; a decomposed attention layer with FP16 sums is percent-level off (see the VAE
    attention test above)."""
    from families.stable_video_diffusion import graph as g

    batch, heads, tokens, dim = 2, 5, 9216, 64
    shape = (batch, tokens, heads * dim)
    generator = torch.Generator().manual_seed(13)
    inputs = {"q": torch.zeros(shape), "k": torch.randn(shape, generator=generator).half().float(),
              "v": (1.0 + 0.01 * torch.randn(shape, generator=generator)).half().float()}

    def populate(network):
        operands = [g.split_heads(network, g.cast(network, network.add_input(
            name, trt.float32, shape), trt.float16), heads) for name in inputs]
        out = g.cast(network, g.merge_heads(network, g.attention(network, *operands)), trt.float32)
        out.name = "y"
        network.mark_output(out)

    actual = _run(_compile(populate, work=trt.float16, verbose=False), inputs, {"y": shape})["y"]
    expected = inputs["v"].double().mean(dim=1, keepdim=True).expand(shape)
    # Measured on a B300: 3.5e-4; a decomposed FP16 attention gives 5e-2 at this length.
    assert _relative_error(actual.cpu(), expected) <= 1e-3


@pytest.mark.parametrize("component", ["image_encoder", "vae_decoder"])
def test_fp16_networks_keep_clip_and_vae_attention_explicit(component):
    """FP16 networks compute the CLIP and VAE attention with explicit FP32 matrix multiplies
    around an FP32 softmax (graph.fp32_attention), not with an attention layer, whose FP16
    decomposition TensorRT may pick even for FP32 operands (the VAE attention test above). At
    the CLIP's head width TensorRT 11.4 picks a fused kernel that accumulates in FP32, so only
    the network shows the difference there."""
    if component == "image_encoder":
        config = transformers.CLIPVisionConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
            image_size=56, patch_size=14, projection_dim=48, hidden_act="gelu", layer_norm_eps=1e-5)
        weights = _weights(_randomize(transformers.CLIPVisionModelWithProjection(config), 1))
        cfg = {"image_size": 56, "patch_size": 14, "hidden_size": 64, "num_hidden_layers": 1,
               "num_attention_heads": 4, "projection_dim": 48, "layer_norm_eps": 1e-5}

        def populate(network, work):
            clip_builder.build_image_encoder(network, weights, cfg, work)
    else:
        weights = _weights(_tiny_vae())
        cfg = {"in_channels": 3, "block_out_channels": [32, 64, 64], "layers_per_block": 2,
               "latent_channels": 4}

        def populate(network, work):
            vae_builder.build_decoder(network, weights, cfg, FRAMES, 8, 12, work)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    attention = {trt.LayerType.ATTENTION_INPUT, trt.LayerType.ATTENTION_OUTPUT}
    found = {}
    for work in (trt.float16, trt.float32):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        populate(network, work)
        found[work] = sum(network.get_layer(index).type in attention
                          for index in range(network.num_layers))
    assert found[trt.float16] == 0
    assert found[trt.float32] > 0  # FP32 networks keep the attention layer, computed in FP32


def test_frame_position_table_matches_diffusers_timesteps():
    from diffusers.models.embeddings import get_timestep_embedding

    from families.stable_video_diffusion import graph

    expected = get_timestep_embedding(torch.arange(25), 320, flip_sin_to_cos=True,
                                      downscale_freq_shift=0).numpy()
    np.testing.assert_allclose(graph.sinusoid_table(np.arange(25), 320), expected, atol=2e-6)

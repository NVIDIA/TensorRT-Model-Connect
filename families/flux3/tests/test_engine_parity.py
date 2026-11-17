# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine graphs against tiny, randomly initialised official modules (GPU).

Each flux3 graph is built from a ``save_pretrained`` directory of a tiny
official module and compared on one forward pass: in float32 with TF32 off on
both sides (exactness of weight mapping, masks, positions and modulation), and
in bfloat16 against the bfloat16 module (the precision the bundle runs).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

FP32_LIMIT = 2e-5
BF16_LIMIT = 0.03


def _cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _run_engine(plan: bytes, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    for name, tensor in inputs.items():
        assert context.set_input_shape(name, tuple(tensor.shape)), name
        context.set_tensor_address(name, tensor.data_ptr())
    outputs = {}
    dtypes = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.int32: torch.int32}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        outputs[name] = torch.empty(
            shape, dtype=dtypes[engine.get_tensor_dtype(name)], device="cuda"
        )
        context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # inputs were produced on the current stream
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, e = actual.double(), expected.double()
    return float(torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e))


def _bf16_value(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float())


def _timestep_proj(t_feed: float, channels: int) -> np.ndarray:
    half = channels // 2
    exponent = (np.float32(-9.210340371976184) * np.arange(half, dtype=np.float32)) / np.float32(
        half
    )
    freqs = np.exp(exponent.astype(np.float32)).astype(np.float32)
    args = (np.float32(t_feed) * freqs).astype(np.float32)
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32).reshape(1, -1)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_denoiser_matches_tiny_flux3_transformer(tmp_path, precision):
    _cuda_or_skip()
    diffusers = pytest.importorskip("diffusers")
    from families.flux3 import rope
    from families.flux3.dit_builder import build_denoiser
    from families.flux3.weights import ComponentWeights

    torch.manual_seed(0)
    config = dict(
        in_channels=16,
        cond_in_channels=16,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=8 * 24,
        timestep_channels=256,
        mlp_ratio=3.0,
        axes_dims_rope=(8, 8, 8, 8),
        rope_theta=10000,
        eps=1e-6,
        flow_map=True,
    )
    module = diffusers.Flux3Transformer2DModel(**config)
    module.save_pretrained(tmp_path)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    compute = trt.float32 if precision == "fp32" else trt.bfloat16
    module = module.to("cuda", dtype)
    height, width, text_len = 4, 6, 160
    plan = build_denoiser(
        ComponentWeights(tmp_path),
        module.config,
        latent_height=height,
        latent_width=width,
        rope_scale=0.5,
        max_tokens=240,
        opt_tokens=160,
        min_tokens=80,
        compute=compute,
        optimization_level=3,
    )
    latents = torch.randn(1, height * width, 16, device="cuda").to(dtype)
    text = torch.randn(1, text_len, 8 * 24, device="cuda").to(dtype)
    sigma, target = 0.731, 0.512
    if precision == "bf16":
        t_feed = _bf16_value(_bf16_value(sigma) * 1000.0)
        target_feed = _bf16_value(_bf16_value(target) * 1000.0)
    else:
        t_feed = float(np.float32(sigma) * np.float32(1000.0))
        target_feed = float(np.float32(target) * np.float32(1000.0))
    with torch.no_grad():
        expected = module(
            hidden_states=latents,
            encoder_hidden_states=text,
            timestep=torch.tensor([sigma], device="cuda", dtype=dtype),
            target_timestep=torch.tensor([target], device="cuda", dtype=dtype),
            img_ids=torch.from_numpy(rope.image_ids(height, width, 0.5)).cuda(),
            txt_ids=torch.from_numpy(rope.text_ids(text_len)).cuda(),
            return_dict=False,
        )[0]
    out = _run_engine(
        plan,
        {
            "latents": latents.float().contiguous(),
            "text_embeds": text.contiguous(),
            "timestep_proj": torch.from_numpy(_timestep_proj(t_feed, 256)).cuda(),
            "target_timestep_proj": torch.from_numpy(_timestep_proj(target_feed, 256)).cuda(),
        },
    )["velocity"]
    error = _rel(out, expected.float())
    assert error < (FP32_LIMIT if precision == "fp32" else BF16_LIMIT), error


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_text_encoder_matches_tiny_qwen3_vl(tmp_path, precision):
    _cuda_or_skip()
    transformers = pytest.importorskip("transformers")
    from families.flux3.text_encoder_builder import PREFIX, build_text_encoder
    from families.flux3.weights import ComponentWeights

    # 34 layers: hidden_states[-1] carries the final norm, the stacked indices (<= 32) must not.
    torch.manual_seed(0)
    text_config = dict(
        hidden_size=64,
        num_hidden_layers=34,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    vision_config = dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        deepstack_visual_indexes=[0],
    )
    config = transformers.Qwen3VLConfig(text_config=text_config, vision_config=vision_config)
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    model.save_pretrained(tmp_path)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    model = model.to("cuda", dtype)
    layers = (4, 8, 12, 16, 20, 24, 28, 32)
    plan = build_text_encoder(
        ComponentWeights(tmp_path, prefix=PREFIX),
        model.config.text_config.to_dict(),
        output_layers=layers,
        max_tokens=240,
        opt_tokens=160,
        min_tokens=80,
        compute=trt.float32 if precision == "fp32" else trt.bfloat16,
        optimization_level=3,
    )
    for real, window in ((57, 80), (95, 160)):
        ids = torch.randint(0, 1024, (1, window), device="cuda", dtype=torch.int64)
        ids[:, real:] = 7
        mask = torch.zeros(1, window, dtype=torch.int64, device="cuda")
        mask[:, :real] = 1
        positions = torch.arange(window, device="cuda")[None].masked_fill(mask == 0, 1)
        with torch.no_grad():
            hidden = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
        expected = (
            torch.stack([hidden[k] for k in layers], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(1, window, -1)
        )
        out = _run_engine(
            plan,
            {
                "input_ids": ids.to(torch.int32).contiguous(),
                "valid_length": torch.tensor([real], dtype=torch.int32, device="cuda"),
            },
        )["text_embeds"]
        error = _rel(out.float(), expected.float())
        assert error < (FP32_LIMIT if precision == "fp32" else BF16_LIMIT), (real, window, error)


def test_vae_matches_tiny_autoencoder(tmp_path):
    _cuda_or_skip()
    diffusers = pytest.importorskip("diffusers")
    from families.flux3.vae_builder import build_vae_decoder
    from families.flux3.weights import ComponentWeights

    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        patch_size=(2, 2),
        batch_norm_eps=1e-4,
    ).eval()
    vae.bn.running_mean.copy_(torch.randn(16) * 0.2)
    vae.bn.running_var.copy_(torch.rand(16) + 0.5)
    vae.save_pretrained(tmp_path)
    vae = vae.cuda()
    height, width = 3, 5
    plan = build_vae_decoder(
        ComponentWeights(tmp_path),
        dict(vae.config),
        latent_height=height,
        latent_width=width,
        latent_dtype=trt.float32,
        optimization_level=3,
    )
    tokens = torch.randn(1, height * width, 16, device="cuda")
    with torch.no_grad():
        grid = tokens.view(1, height, width, 16).permute(0, 3, 1, 2)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        grid = grid * std + vae.bn.running_mean.view(1, -1, 1, 1)
        grid = (
            grid.reshape(1, 4, 2, 2, height, width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(1, 4, 2 * height, 2 * width)
        )
        expected = vae.decode(grid, return_dict=False)[0]
    out = _run_engine(plan, {"latents": tokens.contiguous()})["image"]
    assert out.shape == expected.shape
    assert _rel(out, expected) < FP32_LIMIT

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine parity on small randomly initialised models, without a checkpoint download.

Each test saves a tiny instance of the official module in the checkpoint layout, builds the family
engine from it and compares one forward pass with the official module in FP32.

Every graph is built twice. With FP32 as its compute type the engine has no BF16 rounding, so it
must reproduce the reference to FP32 accuracy; this is the check that pins the graph itself (weight
mapping, masks, position tables, modulation). With BF16, the type bundles are built in, the same
comparison bounds the rounding error.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

# Compute type -> (largest relative error, largest absolute VAE pixel error) against the FP32
# reference. BF16 carries 8 significant bits; two blocks of a random model keep the engine within a
# few rounding steps of the reference, while a wrong weight mapping, mask or position table does
# not. The FP32 limits leave room for kernel-level reordering only.
LIMITS = {
    "bf16": (trt.bfloat16, 0.03, 0.1),
    "fp32": (trt.float32, 2.0e-5, 1.0e-4),
}
both_compute_types = pytest.mark.parametrize("precision", sorted(LIMITS))


def _run(plan: bytes, **inputs):
    """Execute a serialised engine on CUDA tensors and return its outputs as float32 CPU tensors."""
    dtypes = {trt.float32: torch.float32, trt.int32: torch.int32}
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    bound = []
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = dtypes[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = inputs[name].to(device="cuda", dtype=dtype).contiguous()
            assert tuple(tensor.shape) == shape, name
        else:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            outputs[name] = tensor
        bound.append(tensor)
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return {name: tensor.float().cpu() for name, tensor in outputs.items()}


def _perturb(module, generator) -> None:
    """Move every parameter off its initial value so that no weight can be mapped silently."""
    with torch.no_grad():
        for parameter in module.parameters():
            scale = 0.1 if parameter.ndim == 1 else 0.02
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)


def _assert_close(actual, expected, precision: str) -> None:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    error = float((actual - expected).norm() / expected.norm())
    print(f"{precision} relative_error={error:.3e}")
    assert error <= LIMITS[precision][1], error


@both_compute_types
def test_denoiser_matches_the_reference_transformer(tmp_path: Path, precision: str) -> None:
    diffusers = pytest.importorskip("diffusers")
    from families.qwen_image2_1.dit_builder import build_denoiser_plan, load_denoiser

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    reference = diffusers.QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=32,
        num_attention_heads=2,
        context_in_dim=24,
        mlp_ratio=3,
        axes_dims_rope=(8, 12, 12),
    ).eval()
    _perturb(reference, generator)
    reference.save_pretrained(tmp_path)

    height, width, text_slots = 4, 6, 9
    tokens = height * width
    config, weights = load_denoiser(tmp_path)
    plan = io.BytesIO()
    build_denoiser_plan(
        config,
        weights,
        plan,
        latent_height=height,
        latent_width=width,
        text_tokens=text_slots,
        verbose=False,
        compute=LIMITS[precision][0],
    )

    for length in (5, text_slots):
        latents = torch.randn((1, tokens, 8), generator=generator)
        text = torch.randn((1, length, 24), generator=generator)
        sigma = 0.625
        with torch.no_grad():
            expected = reference(
                hidden_states=latents,
                encoder_hidden_states=text,
                timestep=torch.tensor([sigma]),
                img_shapes=[[(1, height, width)]],
                img_mask=torch.cat(
                    [torch.zeros(1, length, dtype=torch.bool), torch.ones(1, tokens // 4, dtype=torch.bool)],
                    dim=1,
                ),
                return_dict=False,
            )[0][:, -tokens:]
        # Slots beyond the prompt hold arbitrary values; the engine must ignore them.
        padded = torch.randn((1, text_slots, 24), generator=generator) * 5.0
        padded[:, :length] = text
        actual = _run(
            plan.getvalue(),
            latents=latents,
            timestep=torch.tensor([sigma]),
            text_hidden=padded,
            text_length=torch.tensor([length]),
        )["velocity"]
        _assert_close(actual, expected, precision)


@both_compute_types
def test_text_encoder_matches_the_reference_language_model(tmp_path: Path, precision: str) -> None:
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    from families.qwen_image2_1.text_encoder_builder import (
        build_text_encoder_plan,
        load_text_encoder,
    )

    text_config = {
        "attention_bias": False,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "vocab_size": 50,
    }
    torch.manual_seed(1)
    reference = transformers.Qwen3VLTextModel(transformers.Qwen3VLTextConfig(**text_config)).eval()
    generator = torch.Generator().manual_seed(1)
    _perturb(reference, generator)
    save_file(
        {f"model.language_model.{name}": value for name, value in reference.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "text_config": text_config})
    )

    prefix, text_slots, length = 3, 8, 6
    config, weights = load_text_encoder(tmp_path)
    plan = io.BytesIO()
    build_text_encoder_plan(
        config,
        weights,
        plan,
        prefix_tokens=prefix,
        text_tokens=text_slots,
        verbose=False,
        compute=LIMITS[precision][0],
    )

    ids = torch.randint(0, 50, (1, prefix + length), generator=generator)
    # The pipeline reads the last decoder layer's output before the final norm.
    hook = reference.norm.register_forward_hook(lambda module, args, output: args[0])
    with torch.no_grad():
        expected = reference(input_ids=ids).last_hidden_state[0, prefix:]
    hook.remove()
    padded = torch.full((prefix + text_slots,), 7, dtype=torch.int32)
    padded[: prefix + length] = ids[0]
    actual = _run(plan.getvalue(), input_ids=padded)["text_hidden"]
    assert actual.shape == (text_slots, 64)
    _assert_close(actual[:length], expected, precision)


@both_compute_types
def test_vae_decoder_matches_the_reference_decoder(tmp_path: Path, precision: str) -> None:
    diffusers = pytest.importorskip("diffusers")
    from families.qwen_image2_1.vae_builder import build_vae_decoder_plan, load_vae

    z_dim = 4
    torch.manual_seed(2)
    generator = torch.Generator().manual_seed(2)
    mean = [0.3, -0.2, 0.1, 0.5]
    std = [1.5, 2.0, 0.8, 1.2]
    reference = diffusers.AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=8, z_dim=z_dim, latents_mean=mean, latents_std=std
    ).eval()
    _perturb(reference, generator)
    reference.save_pretrained(tmp_path)

    height, width = 3, 5
    config, weights = load_vae(tmp_path)
    assert dataclasses.asdict(config)["spatial_scale"] == 16
    plan = io.BytesIO()
    build_vae_decoder_plan(
        config,
        weights,
        plan,
        latent_height=height,
        latent_width=width,
        verbose=False,
        compute=LIMITS[precision][0],
    )

    latents = torch.randn((1, z_dim, height, width), generator=generator)
    raw = latents * torch.tensor(std).view(1, -1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1)
    with torch.no_grad():
        expected = reference.decode(raw[:, :, None], return_dict=False)[0][:, :, 0]
    actual = _run(plan.getvalue(), latents=latents)["image"]
    assert actual.shape == (1, 4, height * 16, width * 16)
    # The decoder clamps to [-1, 1]; compare in pixel units as well.
    worst = float(np.abs(actual.numpy() - expected.numpy()).max())
    print(f"{precision} max_abs_pixel_error={worst:.3e}")
    assert worst <= LIMITS[precision][2]
    _assert_close(actual, expected, precision)

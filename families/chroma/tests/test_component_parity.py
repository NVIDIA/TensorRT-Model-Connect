# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-component TensorRT graph parity on tiny random Chroma components.

Each graph is built from a random checkpoint saved by the official classes
(transformers ``T5EncoderModel``, diffusers ``ChromaTransformer2DModel`` and
``AutoencoderKL``) and compared with that module's forward pass. FP32 is the
correctness gate: both sides run true FP32 (TensorRT without TF32, PyTorch
with TF32 disabled), so the graphs must agree to float rounding. BF16 builds
are compared with the reference modules loaded in BF16.

Requires a CUDA GPU, TensorRT, torch, transformers and diffusers; skipped
otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

if not torch.cuda.is_available():
    pytest.skip("component parity needs a CUDA GPU", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

from families.chroma.checkpoint import SafetensorsDirectory, load_checkpoint  # noqa: E402
from families.chroma.graph_ops import timestep_embedding  # noqa: E402
from families.chroma.model import _plan  # noqa: E402
from families.chroma.text_encoder_builder import build_text_encoder_network  # noqa: E402
from families.chroma.transformer_builder import (  # noqa: E402
    APPROXIMATOR_EPS,
    QK_NORM_EPS,
    build_transformer_network,
)
from families.chroma.vae_builder import build_vae_decoder_network  # noqa: E402

TEXT_LENGTH = 24
GRID = (4, 6)  # packed latent grid: an 8x12 VAE latent, a 64x96 image
# FP32 bounds: relative max error |trt - ref| / max|ref|. Measured on a GB200
# and a B300 well below 1e-5; the bound leaves room for GEMM order only.
FP32_BOUND = 2e-5
BF16_BOUND = 4e-2


@pytest.fixture(autouse=True)
def _true_fp32():
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.fixture(scope="module")
def tiny_pipeline(tmp_path_factory) -> Path:
    """A random, tiny ChromaPipeline layout written by the official classes."""
    from diffusers import AutoencoderKL, ChromaTransformer2DModel
    from transformers import T5Config, T5EncoderModel

    root = tmp_path_factory.mktemp("tiny_chroma")
    generator = torch.Generator().manual_seed(246)

    def randomize(module):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.08)
        return module

    text = T5EncoderModel(T5Config(
        vocab_size=128, d_model=64, d_kv=16, d_ff=96, num_layers=2, num_heads=4,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu", dense_act_fn="gelu_new", is_encoder_decoder=False,
    ))
    randomize(text).save_pretrained(root / "text_encoder")
    transformer = ChromaTransformer2DModel(
        patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=32,
        num_attention_heads=2, joint_attention_dim=64, axes_dims_rope=(8, 12, 12),
        approximator_num_channels=64, approximator_hidden_dim=96, approximator_layers=2,
    )
    randomize(transformer).save_pretrained(root / "transformer")
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
        layers_per_block=1, norm_num_groups=8, use_quant_conv=False, use_post_quant_conv=False,
        scaling_factor=0.3611, shift_factor=0.1159,
    )
    randomize(vae).save_pretrained(root / "vae")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0,
        "use_dynamic_shifting": False,
    }))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "ChromaPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "transformer": ["diffusers", "ChromaTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "image_encoder": [None, None],
        "feature_extractor": [None, None],
    }))
    return root


def _run(plan, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    outputs = {}
    keep = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = inputs[name].contiguous().cuda()
            assert tuple(tensor.shape) == shape, name
        else:
            tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
            outputs[name] = tensor
        keep.append(tensor)
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: value.cpu() for name, value in outputs.items()}


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float().cpu()
    return float((actual.float() - expected).abs().max() / expected.abs().max())


def _text_inputs():
    ids = torch.randint(3, 128, (1, TEXT_LENGTH), generator=torch.Generator().manual_seed(1))
    mask = torch.zeros(1, TEXT_LENGTH, dtype=torch.int32)
    mask[:, :9] = 1
    ids[:, 9:] = 0
    return ids.to(torch.int32), mask


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_t5_encoder(tiny_pipeline: Path, precision: str) -> None:
    from transformers import T5EncoderModel

    checkpoint = load_checkpoint(tiny_pipeline)
    weights = SafetensorsDirectory(tiny_pipeline / "text_encoder")
    plan = _plan("text encoder", precision,
                 lambda graph: build_text_encoder_network(graph, weights, checkpoint.text_encoder, TEXT_LENGTH),
                 False)
    ids, mask = _text_inputs()
    actual = _run(plan, {"input_ids": ids, "attention_mask": mask})["text_embeddings"]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    model = T5EncoderModel.from_pretrained(tiny_pipeline / "text_encoder", torch_dtype=dtype).cuda().eval()
    with torch.no_grad():
        expected = model(ids.long().cuda(), attention_mask=mask.long().cuda())[0]
    error = _relative(actual, expected)
    print(f"T5 {precision}: relative max error {error:.3e}")
    assert error <= (FP32_BOUND if precision == "fp32" else BF16_BOUND)


def _denoiser_reference(tiny_pipeline: Path, dtype, hidden, text, timestep, mask):
    from diffusers import ChromaTransformer2DModel

    model = ChromaTransformer2DModel.from_pretrained(tiny_pipeline / "transformer", torch_dtype=dtype).cuda().eval()
    rows, columns = torch.meshgrid(torch.arange(GRID[0]), torch.arange(GRID[1]), indexing="ij")
    img_ids = torch.stack([torch.zeros_like(rows), rows, columns], dim=-1).reshape(-1, 3)
    full_mask = torch.cat([mask, torch.ones(1, GRID[0] * GRID[1])], dim=1)
    with torch.no_grad():
        return model(
            hidden_states=hidden.to("cuda", dtype),
            encoder_hidden_states=text.to("cuda", dtype),
            timestep=(timestep.to("cuda", dtype) / 1000),
            img_ids=img_ids.to("cuda", dtype),
            txt_ids=torch.zeros(TEXT_LENGTH, 3, device="cuda", dtype=dtype),
            attention_mask=full_mask.to("cuda", dtype),
            return_dict=False,
        )[0]


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("timestep", [1000.0, 637.25, 41.0])
def test_transformer(tiny_pipeline: Path, precision: str, timestep: float) -> None:
    checkpoint = load_checkpoint(tiny_pipeline)
    weights = SafetensorsDirectory(tiny_pipeline / "transformer")
    plan = _plan("transformer", precision,
                 lambda graph: build_transformer_network(
                     graph, weights, checkpoint.transformer, TEXT_LENGTH, *GRID), False)
    generator = torch.Generator().manual_seed(int(timestep))
    hidden = torch.randn(1, GRID[0] * GRID[1], 16, generator=generator)
    text = torch.randn(1, TEXT_LENGTH, 64, generator=generator)
    mask = torch.zeros(1, TEXT_LENGTH)
    mask[:, :12] = 1  # 11 tokens + </s>, then Chroma's one visible padding token
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    # The timestep the model sees, rounded as ChromaPipeline rounds it in this dtype.
    seen = (torch.tensor([timestep]).to(dtype) / 1000).to(dtype) * 1000
    features = torch.from_numpy(timestep_embedding(seen.float().numpy(), 16))
    actual = _run(plan, {
        "hidden_states": hidden, "encoder_hidden_states": text,
        "timestep_features": features, "text_mask": mask,
    })["noise_pred"]
    expected = _denoiser_reference(tiny_pipeline, dtype, hidden, text, torch.tensor([timestep]), mask)
    error = _relative(actual, expected)
    print(f"transformer {precision} t={timestep}: relative max error {error:.3e}")
    assert error <= (FP32_BOUND if precision == "fp32" else BF16_BOUND)


def test_transformer_consumes_the_text_mask(tiny_pipeline: Path) -> None:
    """A different text mask must change the prediction exactly as it changes the reference's."""
    checkpoint = load_checkpoint(tiny_pipeline)
    weights = SafetensorsDirectory(tiny_pipeline / "transformer")
    plan = _plan("transformer", "fp32",
                 lambda graph: build_transformer_network(
                     graph, weights, checkpoint.transformer, TEXT_LENGTH, *GRID), False)
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, GRID[0] * GRID[1], 16, generator=generator)
    text = torch.randn(1, TEXT_LENGTH, 64, generator=generator)
    features = torch.from_numpy(timestep_embedding(np.array([500.0], np.float32), 16))
    results = []
    for visible in (5, TEXT_LENGTH):
        mask = torch.zeros(1, TEXT_LENGTH)
        mask[:, :visible] = 1
        run = _run(plan, {"hidden_states": hidden, "encoder_hidden_states": text,
                          "timestep_features": features, "text_mask": mask})["noise_pred"]
        expected = _denoiser_reference(tiny_pipeline, torch.float32, hidden, text, torch.tensor([500.0]), mask)
        assert _relative(run, expected) <= FP32_BOUND
        results.append(run)
    assert _relative(results[0], results[1]) > 100 * FP32_BOUND


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_vae_decoder(tiny_pipeline: Path, precision: str) -> None:
    from diffusers import AutoencoderKL

    checkpoint = load_checkpoint(tiny_pipeline)
    weights = SafetensorsDirectory(tiny_pipeline / "vae")
    plan = _plan("VAE decoder", precision,
                 lambda graph: build_vae_decoder_network(graph, weights, checkpoint.vae, 2 * GRID[0], 2 * GRID[1]),
                 False)
    latents = torch.randn(1, 4, 2 * GRID[0], 2 * GRID[1], generator=torch.Generator().manual_seed(3))
    actual = _run(plan, {"latents": latents})["image"]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    model = AutoencoderKL.from_pretrained(tiny_pipeline / "vae", torch_dtype=dtype).cuda().eval()
    with torch.no_grad():
        expected = model.decode(latents.to("cuda", dtype), return_dict=False)[0]
    error = _relative(actual, expected)
    print(f"VAE decoder {precision}: relative max error {error:.3e}")
    assert error <= (FP32_BOUND if precision == "fp32" else BF16_BOUND)


def _bf16_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, scale_in_float32: bool) -> torch.Tensor:
    """The family's BF16 RMSNorm graph on its own."""
    rows, width = x.shape

    def define(graph):
        source = graph.network.add_input("x", trt.float32, (rows, width))
        normed = graph.rms_norm(graph.cast(source, trt.bfloat16), weight.float().numpy(), eps,
                                scale_in_float32=scale_in_float32)
        output = graph.cast(normed, trt.float32)
        output.name = "y"
        graph.network.mark_output(output)

    return _run(_plan("RMSNorm", "bf16", define, False), {"x": x.float()})["y"]


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float | None) -> torch.Tensor:
    norm = torch.nn.RMSNorm(x.shape[-1], eps=eps).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
        return norm(x.cuda()).float().cpu()


def test_bf16_rms_norm_rounds_once_like_torch() -> None:
    """torch.nn.RMSNorm (q/k norms) applies its weight in FP32 and rounds once on CUDA.

    Rounding the normalized value before the weight, as the T5 and diffusers
    RMSNorm classes do, changes about a quarter of the BF16 outputs.
    """
    generator = torch.Generator().manual_seed(11)
    x = (torch.randn(64, 128, generator=generator) * 3).to(torch.bfloat16)
    weight = torch.randn(128, generator=generator).to(torch.bfloat16)
    expected = _torch_rms_norm(x, weight, QK_NORM_EPS)
    once = _bf16_rms_norm(x, weight, QK_NORM_EPS, scale_in_float32=True)
    twice = _bf16_rms_norm(x, weight, QK_NORM_EPS, scale_in_float32=False)
    once_mismatch = float((once != expected).float().mean())
    twice_mismatch = float((twice != expected).float().mean())
    print(f"BF16 RMSNorm mismatching outputs: one rounding {once_mismatch:.4f}, two roundings {twice_mismatch:.4f}")
    assert once_mismatch <= 0.01
    assert twice_mismatch >= 0.1


def test_approximator_rms_norm_uses_the_float32_epsilon() -> None:
    """torch.nn.RMSNorm(eps=None) uses the float32 epsilon also for BF16 inputs, not BF16's 2^-7."""
    generator = torch.Generator().manual_seed(12)
    x = (torch.randn(64, 512, generator=generator) * 1e-3).to(torch.bfloat16)  # mean(x^2) ~ 1e-6
    weight = torch.randn(512, generator=generator).to(torch.bfloat16)
    expected = _torch_rms_norm(x, weight, None)
    family = _relative(_bf16_rms_norm(x, weight, APPROXIMATOR_EPS, scale_in_float32=True), expected)
    bf16_epsilon = _relative(_bf16_rms_norm(x, weight, 2.0**-7, scale_in_float32=True), expected)
    print(f"approximator RMSNorm relative max error: float32 eps {family:.3e}, BF16 eps {bf16_epsilon:.3e}")
    assert family <= 1e-2
    assert bf16_epsilon >= 0.5

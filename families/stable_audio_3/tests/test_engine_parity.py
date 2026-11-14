# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three graphs are exact: small random models, FP32, against the official modules.

Every parameter of the official module is re-drawn (the reference zero-initializes
many output layers), the family builder turns the state dict into an FP32 engine,
and both run on the same inputs. The official attention primitive is replaced by
FP32 math with the same flash-attention semantics, because the reference casts
FP32 attention inputs to FP16. The Fourier embedders run at most at 100 Hz here:
the official module computes its frequencies with CUDA's linspace and expf, up to
14 ulps from the correctly rounded values the family uses, which at the
reference's 1e4 Hz (angles up to 6e4 rad) moves features by up to 1e-2;
``test_fourier_features_hold_large_angles`` checks the full range against exact
math. Also checks the FP16 precision policy (FP32 accumulation, exact TF32,
rounded parameters). Needs a GPU, TensorRT, PyTorch, Transformers and the reference
checkout in ``TRTMC_REFERENCE_SOURCE_DIR``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("engine parity needs a GPU", allow_module_level=True)
if not os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"):
    pytest.skip("engine parity needs TRTMC_REFERENCE_SOURCE_DIR", allow_module_level=True)

from families.stable_audio_3 import conditioner_builder, decoder_builder, dit_builder  # noqa: E402
from families.stable_audio_3 import model as family_model  # noqa: E402
from families.stable_audio_3.tests import official_reference  # noqa: E402

TOLERANCE = 2e-5  # relative RMS
PARITY_MAX_FREQ = 100.0  # Fourier embedders, Hz (see the module docstring)


@pytest.fixture(scope="module", autouse=True)
def reference():
    official_reference.install_flash_attn()
    source = str(official_reference.source_dir())
    import sys

    if source not in sys.path:
        sys.path.insert(0, source)
    from stable_audio_3.models import transformer

    def apply_attn(self, q, k, v, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None,
                   flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None):
        dtype = q.dtype
        window = flash_attn_sliding_window if flash_attn_sliding_window is not None else (-1, -1)
        q, k, v = q.float(), k.float(), v.float()
        if varlen_metadata is not None and padding_mask is not None:
            out = torch.zeros_like(q)
            for batch in range(q.shape[0]):
                index = padding_mask[batch].nonzero().flatten()
                out[batch][:, index] = official_reference._attention_bhnd(
                    q[batch:batch + 1][:, :, index], k[batch:batch + 1][:, :, index],
                    v[batch:batch + 1][:, :, index], bool(causal), window, None)[0]
            return out.to(dtype)
        return official_reference._attention_bhnd(q, k, v, bool(causal), window, None).to(dtype)

    original = transformer.Attention.apply_attn
    transformer.Attention.apply_attn = apply_attn
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    transformer.Attention.apply_attn = original


def _randomize(module, seed: int, scale: float = 0.05):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(parameter.shape, generator=generator) * scale
            if name.endswith(("gamma", "weight_g", "scaling_factor")) or "norm" in name and name.endswith("weight"):
                values = 1.0 + values
            parameter.copy_(values)
    return module


def _numpy_state(module, prefix: str = "") -> dict:
    return {key[len(prefix):]: value.detach().float().cpu().numpy()
            for key, value in module.state_dict().items() if key.startswith(prefix)}


def _run(plan: bytes, inputs: dict) -> dict:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    keep = []
    for name, tensor in inputs.items():
        tensor = tensor.contiguous().cuda()
        keep.append(tensor)
        context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.data_ptr())
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: value.cpu() for name, value in outputs.items()}


def _relative_rms(actual, expected) -> float:
    actual, expected = actual.double(), expected.double()
    return float((actual - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt())


def _dit_cfg(memory: int = 4) -> dict:
    return {
        "text": {"max_length": 7},
        "dit": {"io_channels": 8, "embed": 128, "depth": 2, "heads": 2, "head_dim": 64, "rope_dim": 32,
                "cond_dim": 16, "local_dim": 9, "ff_inner": 512, "memory_tokens": memory, "norm_eps": 1e-5,
                "qk_norm_eps": 1e-6, "timestep_features": 256, "fourier_min_freq": 0.5,
                "fourier_max_freq": PARITY_MAX_FREQ},
    }


def _dit_module():
    from stable_audio_3.models.dit import DiffusionTransformer

    module = DiffusionTransformer(
        io_channels=8, embed_dim=128, depth=2, num_heads=2, cond_token_dim=16, global_cond_dim=16,
        local_add_cond_dim=9, global_cond_type="adaLN", timestep_features_type="expo",
        attn_kwargs={"qk_norm": "rms", "differential": True}, norm_type="rms_norm",
        norm_kwargs={"force_fp32": True}, ff_kwargs={"mult": 4.0}, num_memory_tokens=4,
        diffusion_objective="rf_denoiser")
    module.timestep_features.max_freq = PARITY_MAX_FREQ
    return _randomize(module, 1).cuda().eval()


def _network(populate):
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return builder, network, populate(network)


def _network_for(graph: str, work):
    """The family network of one graph with the bundle's weight rounding (the conditioner's T5Gemma stays BF16)."""
    if graph == "dit":
        weights = family_model.as_model_dtype(_numpy_state(_dit_module()), work)
        return _network(lambda network: dit_builder.populate(network, weights, _dit_cfg(), work, 40))
    if graph == "decoder":
        _, weights, running_std, cfg = _decoder_parts()
        weights = family_model.as_model_dtype(weights, work)
        running_std = family_model.as_model_dtype({"running_std": running_std}, work)["running_std"]
        return _network(lambda network: decoder_builder.populate(network, weights, running_std, cfg, work, 32))
    *_, cfg, text_weights, cond_weights = _conditioner_parts()
    cond_weights = family_model.as_model_dtype(cond_weights, work)
    return _network(lambda network: conditioner_builder.populate(network, text_weights, cond_weights, cfg, work))


@pytest.mark.parametrize("graph", ["dit", "decoder", "conditioner"])
@pytest.mark.parametrize("work", [trt.float16, trt.float32], ids=["fp16", "fp32"])
def test_fp16_products_accumulate_in_fp32_and_tf32_is_exact(graph, work):
    """FP16 networks keep no FP16 matrix product (FP16 TensorRT GEMMs may accumulate in FP16) and get TF32,
    which holds their FP16-valued FP32 operands exactly. FP32 networks keep full FP32 products; so does the FP32
    conditioner, whose seconds embedder multiplies FP32 values next to the BF16 T5Gemma."""
    builder, network, kept = _network_for(graph, work)
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    products = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    assert products and all(layer.get_output(0).dtype == trt.float32 for layer in products)
    assert family_model.products_exact_in_tf32(network) == (work == trt.float16)
    del kept, network, builder  # the constants' host buffers outlive the network


def test_fp16_linear_accumulates_in_fp32():
    """An FP16 linear layer as the bundles build it (K = 6144, the DiT feed-forward) equals the FP16 rounding of the
    exact product of its FP16 operands within one FP16 ulp, as an FP32-accumulating cuBLAS GEMM does (plus 1e-5 for
    FP32 summation near zero). FP16 accumulation misses this by orders of magnitude."""
    from families.stable_audio_3.graph import Graph

    rng = np.random.default_rng(11)
    rows, width, out_features = 64, 6144, 256
    x = rng.standard_normal((1, rows, width)).astype(np.float16)
    weight = (rng.standard_normal((out_features, width)) / np.sqrt(width)).astype(np.float16)
    bias = (0.1 * rng.standard_normal(out_features)).astype(np.float16)

    def populate(network):
        g = Graph(network, trt.float16)
        value = network.add_input("x", trt.float16, (1, rows, width))
        out = g.cast(g.linear(value, weight.astype(np.float32), bias.astype(np.float32)), trt.float32)
        out.name = "y"
        network.mark_output(out)
        return g

    plan = family_model._compile(populate, verbose=False)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    inputs = torch.from_numpy(x).cuda()
    output = torch.empty((1, rows, out_features), dtype=torch.float32, device="cuda")
    context.set_tensor_address("x", inputs.data_ptr())
    context.set_tensor_address("y", output.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    exact = (x.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)).astype(np.float16)
    error = np.abs(output.cpu().numpy().astype(np.float64) - exact.astype(np.float64))
    excess = error - np.spacing(np.abs(exact)).astype(np.float64)
    assert float(excess.max()) <= 1e-5, float(error.max())


def test_fp16_builds_round_parameters_like_the_reference():
    weights = {"inv_freq": np.array([1.0, 0.1, 1e-4], dtype=np.float32), "index": np.array([3], dtype=np.int64)}
    rounded = family_model.as_model_dtype(weights, trt.float16)
    np.testing.assert_array_equal(rounded["inv_freq"], weights["inv_freq"].astype(np.float16).astype(np.float32))
    assert rounded["inv_freq"].dtype == np.float32 and rounded["index"] is weights["index"]
    assert family_model.as_model_dtype(weights, trt.float32) is weights


def test_fourier_features_hold_large_angles():
    """1e4 Hz features (angles up to 6e4 rad) stay within about one ulp of the angle of exact cos / sin.

    The official FP32 module is farther from exact math (up to ~9 ulps of the angle): it computes its frequencies
    with CUDA's linspace and expf, up to 14 ulps from the correctly rounded values. That is a property of the
    reference, so the FP32 parity tests run the embedders at 100 Hz.
    """
    from families.stable_audio_3.graph import Graph, expo_fourier_features, expo_fourier_frequencies

    frequencies = expo_fourier_frequencies(256, 0.5, 1e4)

    def populate(network):
        g = Graph(network, trt.float32)
        value = network.add_input("t", trt.float32, (1, 1))
        out = expo_fourier_features(g, value, frequencies)
        out.name = "features"
        network.mark_output(out)
        return g

    plan = family_model._compile(populate, verbose=False)
    angular = 2.0 * np.pi * frequencies.astype(np.float64)
    for t in (0.27, 0.83, 1.0):
        actual = _run(plan, {"t": torch.tensor([[t]], dtype=torch.float32)})["features"][0].double().numpy()
        angle = float(np.float32(t)) * angular
        exact = np.concatenate([np.cos(angle), np.sin(angle)])
        # FP32 rounds t * f * 2 * pi about once per operation: within ~0.7 ulp of the angle here.
        ulps = np.abs(actual - exact) / (np.concatenate([angle, angle]) * 2.0 ** -23 + 1e-6)
        assert float(ulps.max()) < 1.5, float(ulps.max())


def test_dit_graph_is_exact_in_fp32():
    cfg = _dit_cfg()
    module = _dit_module()
    weights = _numpy_state(module)
    plan = family_model._compile(lambda network: dit_builder.populate(network, weights, cfg, trt.float32, 40),
                                 verbose=False,
                                 profile={dit_builder.LATENTS: ((1, 8, 2), (1, 8, 24), (1, 8, 40))})
    generator = torch.Generator().manual_seed(2)
    for length, valid, t in ((24, 20, 0.83), (40, 40, 0.27), (9, 3, 0.5)):
        x = torch.randn(1, 8, length, generator=generator)
        context = torch.randn(1, 8, 16, generator=generator)
        global_cond = torch.randn(1, 16, generator=generator)
        time = torch.tensor([t], dtype=torch.float32)
        actual = _run(plan, {"x": x, "t": time, "cross_attn_cond": context, "global_cond": global_cond,
                             "valid_latents": torch.tensor([valid], dtype=torch.int32)})["v"]
        padding = torch.arange(length)[None, :] < valid
        with torch.inference_mode():
            expected = module(x.cuda(), time.cuda(), cross_attn_cond=context.cuda(), global_embed=global_cond.cuda(),
                              local_add_cond=torch.zeros(1, 9, length, device="cuda"), padding_mask=padding.cuda(),
                              cfg_scale=1.0).float().cpu()
        keep = padding[:, None, :].expand_as(expected)
        assert _relative_rms(actual[keep], expected[keep]) < TOLERANCE


def _decoder_parts():
    from stable_audio_3.models.autoencoders import SAMEDecoder

    module = SAMEDecoder(out_channels=8, channels=64, c_mults=[2], strides=[4], latent_dim=6, transformer_depths=[2],
                         sinusoidal_blocks=[2], differential=True, dim_heads=64, sliding_window=[1, 1],
                         variable_stride=True)
    module = _randomize(module, 3).cuda().eval()
    cfg = {"decoder": {"latent_dim": 6, "width": 128, "stride": 4, "depth": 2, "heads": 2, "head_dim": 64,
                       "rope_dim": 32, "ff_inner": 384, "sinusoidal": [False, True], "group": 5, "window": 5,
                       "out_channels": 8, "patch_size": 4, "audio_channels": 2}}
    return module, _numpy_state(module), np.array([1.7], dtype=np.float32), cfg


def test_decoder_graph_is_exact_in_fp32():
    module, weights, running_std, cfg = _decoder_parts()
    plan = family_model._compile(
        lambda network: decoder_builder.populate(network, weights, running_std, cfg, trt.float32, 32),
        verbose=False, profile={decoder_builder.LATENTS: ((1, 6, 1), (1, 6, 12), (1, 6, 32))})
    generator = torch.Generator().manual_seed(4)
    for length in (1, 2, 13, 32):
        latents = torch.randn(1, 6, length, generator=generator)
        actual = _run(plan, {"latents": latents})["audio"]
        with torch.inference_mode():
            decoded = module(latents.cuda() * float(running_std[0])).float()  # [1, 8, 4 * length]
            expected = decoded.reshape(1, 2, 4, -1).transpose(2, 3).reshape(1, 2, -1).cpu()  # unpatch
        assert actual.shape == expected.shape
        assert _relative_rms(actual, expected) < TOLERANCE


def _conditioner_parts():
    from stable_audio_3.models.conditioners import NumberConditioner
    from transformers import T5GemmaConfig, T5GemmaEncoderModel

    module_config = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 300, "query_pre_attn_scalar": 32,
                     "sliding_window": 4096, "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
                     "layer_types": ["sliding_attention", "full_attention"], "attn_logit_softcapping": 50.0}
    config = T5GemmaConfig(encoder=module_config, decoder=module_config, is_encoder_decoder=False, pad_token_id=0)
    encoder = T5GemmaEncoderModel(config)
    encoder = _randomize(encoder, 5).cuda().eval()
    numbers = _randomize(NumberConditioner(output_dim=64, min_val=0, max_val=384, fourier_features_type="expo"), 6)
    numbers.embedder.embedding[0].max_freq = PARITY_MAX_FREQ
    numbers = numbers.cuda().eval()
    padding = (torch.randn(64, generator=torch.Generator().manual_seed(7)) * 0.1).numpy()
    tokens = 12
    cfg = {"text": {"max_length": tokens, "hidden": 64, "layers": 2, "heads": 2, "head_dim": 32, "intermediate": 128,
                    "eps": 1e-6, "rope_theta": 10000.0, "query_scale": 32 ** -0.5, "vocab": 300, "pad_token_id": 0},
           "seconds": {"min": 0.0, "max": 384.0, "features": 256, "min_freq": 0.5, "max_freq": PARITY_MAX_FREQ},
           "dit": {"cond_dim": 64}}
    text_weights = _numpy_state(encoder, "encoder.")
    cond_weights = {"prompt.padding_embedding": padding,
                    "seconds_total.embedder.embedding.1.weight": _numpy_state(numbers)["embedder.embedding.1.weight"],
                    "seconds_total.embedder.embedding.1.bias": _numpy_state(numbers)["embedder.embedding.1.bias"]}
    return encoder, numbers, padding, cfg, text_weights, cond_weights


def test_conditioner_graph_is_exact_in_fp32():
    encoder, numbers, padding, cfg, text_weights, cond_weights = _conditioner_parts()
    tokens = cfg["text"]["max_length"]
    plan = family_model._compile(lambda network: conditioner_builder.populate(
        network, text_weights, cond_weights, cfg, trt.float32, trt.float32), verbose=False)
    ids = torch.tensor([[5, 77, 140, 2, 299, 31, 8] + [0] * 5], dtype=torch.int32)
    mask = (ids != 0).to(torch.int32)
    seconds = 47.3
    actual = _run(plan, {"input_ids": ids, "attention_mask": mask,
                         "seconds_total": torch.tensor([[seconds]], dtype=torch.float32)})
    with torch.inference_mode():
        hidden = encoder(input_ids=ids.long().cuda(), attention_mask=mask.bool().cuda())["last_hidden_state"]
        keep = mask.bool().cuda()[..., None]
        prompt = torch.where(keep, hidden, torch.from_numpy(padding).cuda()).float().cpu()
        number = numbers([seconds], device="cuda")[0][:, 0].float().cpu()
    assert encoder.config._attn_implementation == "sdpa"
    assert _relative_rms(actual["cross_attn_cond"][0, :tokens], prompt[0]) < TOLERANCE
    assert _relative_rms(actual["cross_attn_cond"][0, tokens], number[0]) < TOLERANCE
    assert _relative_rms(actual["global_cond"], number) < TOLERANCE

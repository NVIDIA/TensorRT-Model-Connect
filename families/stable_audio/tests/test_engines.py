# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-level parity of the three family networks against the diffusers modules.

Runs on a GPU with TensorRT and diffusers (skipped otherwise) over the tiny random
checkpoint: the conditioning network against ``encode_prompt`` + ``encode_duration``,
the denoiser against ``StableAudioDiTModel`` (both guidance batch sizes) and the
decoder against ``AutoencoderOobleck.decode``. FP32 builds must agree closely (TF32 is
off on both sides); FP16 builds of the denoiser and decoder are bounded loosely. Another
check keeps the FP32 projections of TF32 (FP16) engines exact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"


def _relative(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("engine parity needs a CUDA device")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download("google-t5/t5-base", revision=T5_REVISION, local_files_only=True,
                          allow_patterns=["tokenizer.json", "spiece.model", "*.json"])
    except Exception:
        pytest.skip("engine parity needs the cached google-t5/t5-base tokenizer")
    from diffusers import StableAudioPipeline

    from families.stable_audio import config as config_module
    from families.stable_audio.tests.random_checkpoint import checkpoint_name, is_current, write_random_checkpoint
    from families.stable_audio.tests.reference_t5 import restore_t5_layer_norm

    root = Path(os.environ.get("TRTMC_STABLE_AUDIO_RANDOM_ROOT") or tmp_path_factory.mktemp("stable-audio"))
    model_dir = root / checkpoint_name("tiny", 0)
    if not is_current(model_dir, size="tiny", seed=0, t5_revision=T5_REVISION):
        write_random_checkpoint(model_dir, size="tiny", seed=0, t5_revision=T5_REVISION)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = StableAudioPipeline.from_pretrained(model_dir, torch_dtype=torch.float32).to("cuda")
    restore_t5_layer_norm(pipeline.text_encoder)
    return torch, pipeline, model_dir, config_module.resolve(model_dir)


def _run(plan: bytes, inputs: dict, outputs: list[str], torch) -> dict:
    from families.stable_audio.tests.trt_runner import EngineRunner

    del torch
    return EngineRunner(plan)(inputs, outputs)


def test_fp32_projections_stay_fp32_in_tf32_engines():
    """FP16 engines enable TF32; their FP32-input projections must not be TF32-rounded.

    The denoiser's 1x1 pre / post convolutions are such projections, at this shape. A graph
    that does not know about TF32 (``tf32_engine=False``) is the negative control: on a
    device with TF32 tensor cores its matrix multiply is off by about 3e-4.
    """
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from families.stable_audio.graph_ops import Graph
    from families.stable_audio.tests.trt_runner import EngineRunner

    generator = np.random.default_rng(0)
    weight = (generator.standard_normal((64, 64)) / 8.0).astype(np.float32)
    sample = generator.standard_normal((2, 1024, 64)).astype(np.float32)
    expected = sample.astype(np.float64) @ weight.T.astype(np.float64)

    def error(tf32_engine: bool) -> float:
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        graph = Graph(network, tf32_engine=tf32_engine)
        out = graph.linear(network.add_input("x", trt.float32, sample.shape), weight)
        out.name = "y"
        network.mark_output(out)
        builder_config = builder.create_builder_config()
        builder_config.set_flag(trt.BuilderFlag.TF32)
        plan = builder.build_serialized_network(network, builder_config)
        assert plan is not None
        return _relative(EngineRunner(bytes(plan))({"x": sample}, ["y"])["y"], expected)

    exact, rounded = error(True), error(False)
    assert exact < 1e-6, (exact, rounded)
    if torch.cuda.get_device_capability() >= (8, 0):
        assert rounded > 1e-5, ("the negative control was not TF32-rounded", exact, rounded)


def test_conditioner_matches_encode_prompt_and_duration(stack):
    torch, pipeline, model_dir, cfg = stack
    from families.stable_audio import model

    plan = model._conditioner_plan(model_dir, cfg, False)
    length = cfg["tokenizer"]["max_length"]
    # The long prompt fills all 128 positions, so every relative-position bucket is used.
    long_prompt = " ".join(["Rain on a tin roof, thunder, a wind chime and footsteps on gravel."] * 12)
    for prompt, start, end in (("A dog barks twice in a large hall.", 0.0, 0.004), ("", 0.001, 0.003),
                               (long_prompt, 0.002, 0.004)):
        encoded = pipeline.tokenizer(prompt, padding="max_length", max_length=length, truncation=True,
                                     return_tensors="pt")
        native = _run(plan, {
            "input_ids": encoded.input_ids.to(torch.int32),
            "attention_mask": encoded.attention_mask.to(torch.float32),
            "seconds_start": torch.tensor([start], dtype=torch.float32),
            "seconds_end": torch.tensor([end], dtype=torch.float32),
        }, ["encoder_hidden_states", "global_hidden_states"], torch)
        with torch.inference_mode():
            prompt_embeds = pipeline.encode_prompt(prompt, "cuda", False)
            begin, finish = pipeline.encode_duration(start, end, "cuda", False, 1)
        expected_encoder = torch.cat([prompt_embeds, begin, finish], dim=1).cpu().numpy()
        expected_global = torch.cat([begin, finish], dim=2).cpu().numpy()
        assert _relative(native["encoder_hidden_states"], expected_encoder) < 1e-5
        assert _relative(native["global_hidden_states"], expected_global) < 1e-5


@pytest.mark.parametrize("precision,bound", [("fp32", 1e-5), ("fp16", 5e-3)])
def test_denoiser_matches_the_dit(stack, precision, bound):
    torch, pipeline, model_dir, cfg = stack
    import tensorrt as trt
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    from families.stable_audio import model

    work = {"fp32": trt.float32, "fp16": trt.float16}[precision]
    plan = model._denoiser_plan(model_dir, cfg, work, False)
    dit = cfg["transformer"]
    generator = torch.Generator().manual_seed(3)
    for batch in (1, 2):
        sample = torch.randn(batch, dit["in_channels"], dit["sample_size"], generator=generator)
        context = torch.randn(batch, cfg["tokenizer"]["max_length"] + 2, dit["cross_attention_input_dim"],
                              generator=generator)
        global_states = torch.randn(batch, 1, dit["global_states_input_dim"], generator=generator)
        timestep = torch.tensor([0.83], dtype=torch.float32)
        native = _run(plan, {"hidden_states": sample, "timestep": timestep, "encoder_hidden_states": context,
                             "global_hidden_states": global_states}, ["sample"], torch)["sample"]
        rotary = get_1d_rotary_pos_embed(dit["rotary_dim"], dit["sample_size"] + 1, use_real=True,
                                         repeat_interleave_real=False)
        rotary = tuple(value.cuda() for value in rotary)
        with torch.inference_mode():
            expected = pipeline.transformer(sample.cuda(), timestep.cuda(), encoder_hidden_states=context.cuda(),
                                            global_hidden_states=global_states.cuda(), rotary_embedding=rotary,
                                            return_dict=False)[0].cpu().numpy()
        assert _relative(native, expected) < bound, (batch, _relative(native, expected))


@pytest.mark.parametrize("precision,bound", [("fp32", 1e-5), ("fp16", 5e-3)])
def test_decoder_matches_oobleck_decode(stack, precision, bound):
    torch, pipeline, model_dir, cfg = stack
    import tensorrt as trt

    from families.stable_audio import model

    work = {"fp32": trt.float32, "fp16": trt.float16}[precision]
    plan = model._decoder_plan(model_dir, cfg, work, False)
    latents = torch.randn(1, cfg["vae"]["latent_channels"], cfg["transformer"]["sample_size"],
                          generator=torch.Generator().manual_seed(4))
    native = _run(plan, {"latents": latents}, ["audio"], torch)["audio"]
    with torch.inference_mode():
        expected = pipeline.vae.decode(latents.cuda()).sample.cpu().numpy()
    assert native.shape == expected.shape
    assert _relative(native, expected) < bound


def test_runtime_config_matches_the_checkpoint(stack):
    _, pipeline, _, cfg = stack
    from families.stable_audio import model

    runtime = model._runtime_config(cfg, "fp32")
    assert runtime["audio_samples"] == cfg["transformer"]["sample_size"] * pipeline.vae.hop_length
    assert runtime["sample_rate"] == pipeline.vae.config.sampling_rate
    assert runtime["text_length"] == pipeline.tokenizer.model_max_length
    assert json.loads(json.dumps(runtime)) == runtime

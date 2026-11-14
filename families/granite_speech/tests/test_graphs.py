# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both engines against the Transformers modules, on a small random Granite Speech (GPU).

The random model keeps the real front end and projector windows but shrinks every width. Its
encoder context of 8 frames and the 1.3 s clip give nine attention blocks with one real frame in
the last one and a partial last projector window, so the masking and padding paths are exercised.
FP32 engines must match the FP32 modules to float accuracy (TF32 off on both sides); BF16 engines
are checked for direction only, because their rounding is not the reference's.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from families.granite_speech import config as config_module  # noqa: E402
from families.granite_speech.audio import build_audio_network  # noqa: E402
from families.granite_speech.checkpoint import Checkpoint  # noqa: E402
from families.granite_speech.constants import reflect_pad  # noqa: E402
from families.granite_speech.decoder import build_decoder_network  # noqa: E402
from families.granite_speech.graph import Graph  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

_PREPROCESSOR = {
    "feature_extractor_type": "GraniteSpeechFeatureExtractor", "projector_downsample_rate": 5,
    "projector_window_size": 15, "sampling_rate": 16000,
}
_CAPACITY = 64
_DTYPES = {"fp32": (trt.float32, torch.float32), "bf16": (trt.bfloat16, torch.bfloat16)}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = transformers.GraniteSpeechConfig(
        encoder_config=dict(input_dim=160, hidden_dim=64, num_layers=4, num_heads=2, dim_head=32, context_size=8,
                            max_pos_emb=16, conv_kernel_size=15, conv_expansion_factor=2, feedforward_mult=4,
                            output_dim=11),
        projector_config=dict(model_type="blip_2_qformer", hidden_size=64, encoder_hidden_size=64,
                              num_attention_heads=4, num_hidden_layers=2, intermediate_size=128,
                              cross_attention_frequency=1, hidden_act="gelu", layer_norm_eps=1e-12,
                              use_qformer_text_input=False),
        text_config=dict(model_type="granite", hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=128, vocab_size=256, max_position_embeddings=512,
                         rms_norm_eps=1e-5, embedding_multiplier=12.0, attention_multiplier=0.0625,
                         residual_multiplier=0.22, logits_scaling=8.0, eos_token_id=254, pad_token_id=253,
                         bos_token_id=254, tie_word_embeddings=False),
        audio_token_index=255, window_size=15, downsample_rate=5, has_lora_adapter=False)
    torch.manual_seed(0)
    model = transformers.GraniteSpeechForConditionalGeneration(cfg).eval()
    with torch.no_grad():  # BatchNorm statistics away from the identity, so their folding is tested
        for layer in model.encoder.layers:
            layer.conv.batch_norm.running_mean.uniform_(-0.5, 0.5)
            layer.conv.batch_norm.running_var.uniform_(0.5, 2.0)
    directory = tmp_path_factory.mktemp("tiny-granite-speech")
    model.save_pretrained(str(directory))
    (directory / "preprocessor_config.json").write_text(json.dumps(_PREPROCESSOR))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return directory, model.to("cuda"), config_module.resolve(directory)


def _engine(build_network, profiles, work):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    graph = Graph(network, work)
    build_network(graph)
    for shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (low, opt, high) in shapes.items():
            profile.set_shape(name, low, opt, high)
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    return trt.Runtime(logger).deserialize_cuda_engine(plan)


def _run(engine, profile: int, inputs: dict, outputs: dict) -> None:
    """``inputs`` map a name to a tensor, or to ``(buffer, shape)`` for a view that may be empty
    (an empty tensor has no address, the runtime binds the cache buffer itself)."""
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    context.set_optimization_profile_async(profile, stream.cuda_stream)
    for name, value in inputs.items():
        buffer, shape = value if isinstance(value, tuple) else (value, tuple(value.shape))
        context.set_input_shape(name, shape)
        context.set_tensor_address(name, buffer.data_ptr())
    for name, tensor in outputs.items():
        assert tuple(context.get_tensor_shape(name)) == tuple(tensor.shape), name
        context.set_tensor_address(name, tensor.data_ptr())
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()


def _clip(samples: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    audio = (0.1 * rng.standard_normal(samples)).astype(np.float32)
    audio[: samples // 4] *= 1e-3
    return audio


def _audio_embeds(tiny, precision: str, samples: int):
    directory, model, cfg = tiny
    work, torch_dtype = _DTYPES[precision]
    limits = config_module.sample_limits(cfg, 5, _CAPACITY)
    with Checkpoint(directory, cfg) as ckpt:
        engine = _engine(lambda g: build_audio_network(g, ckpt, cfg, limits["max_frames"]),
                         [{"waveform": ((limits["min_samples"] + 512,), (samples + 512,),
                                        (limits["max_samples"] + 512,))}], work)
    padded = torch.from_numpy(reflect_pad(_clip(samples), cfg.front_end)).cuda()
    rows = cfg.audio_tokens(samples)
    out = torch.empty(rows, cfg.text.hidden_size, dtype=torch_dtype, device="cuda")
    _run(engine, 0, {"waveform": padded}, {"audio_embeds": out})
    return out


def _reference_audio(tiny, samples: int):
    _, model, _ = tiny
    extractor = transformers.GraniteSpeechFeatureExtractor()
    features = extractor(torch.from_numpy(_clip(samples)))["input_features"].cuda()
    with torch.no_grad():
        return model.get_audio_features(features).pooler_output[0]


@pytest.mark.parametrize("samples", [20800, 16000 * 3 + 37])
def test_fp32_audio_engine_matches_the_encoder_and_projector(tiny, samples):
    ours = _audio_embeds(tiny, "fp32", samples)
    expected = _reference_audio(tiny, samples)
    assert ours.shape == expected.shape
    torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)


def _decoder(tiny, precision: str):
    directory, _, cfg = tiny
    work, _ = _DTYPES[precision]
    text = cfg.text
    rows = {"input_ids": (1,), "audio_mask": (1,), "audio_embeds": (1, text.hidden_size)}
    with Checkpoint(directory, cfg) as ckpt:
        profiles = []
        for new, past in (((1, 24, _CAPACITY), (0, 0, 0)), ((1, 1, 1), (1, 24, _CAPACITY - 1))):
            shapes = {name: tuple((value,) + shape[1:] for value in new) for name, shape in rows.items()}
            for index in range(text.num_layers):
                for name in (f"past_key_{index}", f"past_value_{index}"):
                    shapes[name] = tuple((value, text.kv_width) for value in past)
            profiles.append(shapes)
        return _engine(lambda g: build_decoder_network(g, ckpt, cfg, _CAPACITY), profiles, work)


def _step(engine, profile, cfg, ids, mask, audio, cache_k, cache_v, cached):
    text = cfg.text
    new = ids.shape[0]
    inputs = {"input_ids": ids, "audio_mask": mask, "audio_embeds": audio}
    outputs = {"logits": torch.empty(1, text.vocab_size, dtype=torch.float32, device="cuda")}
    for index in range(text.num_layers):
        inputs[f"past_key_{index}"] = (cache_k[index], (cached, text.kv_width))
        inputs[f"past_value_{index}"] = (cache_v[index], (cached, text.kv_width))
        outputs[f"present_key_{index}"] = cache_k[index][cached:cached + new]
        outputs[f"present_value_{index}"] = cache_v[index][cached:cached + new]
    _run(engine, profile, inputs, outputs)
    return outputs["logits"][0]


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_decoder_prefill_and_step_match_the_language_model(tiny, precision):
    _, model, cfg = tiny
    work, torch_dtype = _DTYPES[precision]
    samples = 20800
    audio = _reference_audio(tiny, samples)  # [A, hidden], the reference projector output
    count = audio.shape[0]
    prefix, suffix = [11, 12, 13], [21, 22]
    ids = torch.tensor(prefix + [255] * count + suffix, dtype=torch.int32, device="cuda")
    mask = torch.zeros_like(ids)
    mask[len(prefix):len(prefix) + count] = 1
    rows = torch.zeros(ids.shape[0], cfg.text.hidden_size, dtype=torch_dtype, device="cuda")
    rows[len(prefix):len(prefix) + count] = audio.to(torch_dtype)
    cache_k = [torch.zeros(_CAPACITY, cfg.text.kv_width, dtype=torch_dtype, device="cuda")
               for _ in range(cfg.text.num_layers)]
    cache_v = [torch.zeros_like(tensor) for tensor in cache_k]
    engine = _decoder(tiny, precision)
    prefill = _step(engine, 0, cfg, ids, mask, rows, cache_k, cache_v, 0)
    token = int(torch.argmax(prefill))
    step = _step(engine, 1, cfg, torch.tensor([token], dtype=torch.int32, device="cuda"),
                 torch.zeros(1, dtype=torch.int32, device="cuda"),
                 torch.zeros(1, cfg.text.hidden_size, dtype=torch_dtype, device="cuda"),
                 cache_k, cache_v, ids.shape[0])

    with torch.no_grad():
        embeds = model.language_model.get_input_embeddings()(torch.where(mask.bool(), 0, ids).long())
        embeds[len(prefix):len(prefix) + count] = audio
        full = torch.cat([embeds, model.language_model.get_input_embeddings()(torch.tensor([token], device="cuda"))])
        logits = model.language_model(inputs_embeds=full[None]).logits[0]
    expected_prefill, expected_step = logits[ids.shape[0] - 1], logits[ids.shape[0]]
    if precision == "fp32":
        torch.testing.assert_close(prefill, expected_prefill, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(step, expected_step, rtol=1e-4, atol=1e-4)
    else:
        for ours, expected in ((prefill, expected_prefill), (step, expected_step)):
            cosine = torch.nn.functional.cosine_similarity(ours[None], expected[None]).item()
            assert cosine > 0.999, cosine


def test_bf16_audio_engine_follows_the_fp32_modules(tiny):
    ours = _audio_embeds(tiny, "bf16", 20800).float()
    expected = _reference_audio(tiny, 20800)
    cosine = torch.nn.functional.cosine_similarity(ours.flatten()[None], expected.flatten()[None]).item()
    assert cosine > 0.999, cosine

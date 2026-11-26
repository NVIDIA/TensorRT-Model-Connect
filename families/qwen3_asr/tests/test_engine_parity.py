# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine parity on a small randomly initialised Qwen3-ASR, without a checkpoint download.

A tiny ``Qwen3ASRForConditionalGeneration`` is saved in the checkpoint layout; the family builds
its encoder and decoder engines from that directory and each engine is compared with the official
module in FP32 on the CPU.

Every graph is built twice. With FP32 as its compute type the engine has no BF16 rounding and must
reproduce the reference to FP32 accuracy; this pins the graph itself (chunk layout, window and key
masks, weight mapping, rotary tables, KV-cache rows). With BF16, the type bundles default to, the
same comparison bounds the rounding error.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

# Compute type -> largest relative L2 error against the FP32 reference. BF16 carries 8 significant
# bits; a wrong weight mapping, mask, position or cache row is far outside both limits.
LIMITS = {"bf16": (trt.bfloat16, 0.03), "fp32": (trt.float32, 2.0e-5)}
both_compute_types = pytest.mark.parametrize("precision", sorted(LIMITS))

AUDIO_TOKEN = 60
VOCAB = 64
CAPACITY = 48
# Fewer than the 11 prompt tokens, so the prompt is prefilled in two launches (8 + 3).
PREFILL_TOKENS = 8


def _tiny_checkpoint(directory: Path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3ASRForConditionalGeneration"):
        pytest.skip("Qwen3-ASR needs transformers >= 5.13 (the family requirements pin 5.22.2)")
    config = transformers.Qwen3ASRConfig(
        audio_config={
            "num_mel_bins": 128, "encoder_layers": 2, "encoder_attention_heads": 4,
            "num_key_value_heads": 4, "encoder_ffn_dim": 48, "d_model": 32, "n_window": 50,
            "n_window_infer": 800, "downsample_hidden_size": 8, "max_position_embeddings": 13,
            "output_dim": 32,
        },
        text_config={
            "vocab_size": VOCAB, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 512, "tie_word_embeddings": True, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1.0e6, "rope_type": "default"},
        },
        audio_token_id=AUDIO_TOKEN,
        eos_token_id=[61, 62],
        pad_token_id=62,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ASRForConditionalGeneration(config).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # Move every parameter off its initial value so that no weight can be mapped silently.
        for parameter in model.parameters():
            scale = 0.1 if parameter.ndim == 1 else 0.05
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    model.save_pretrained(directory)
    processor = {"feature_extractor": {
        "feature_extractor_type": "Qwen3ASRFeatureExtractor", "dither": 0.0, "padding_value": 0.0,
        "sampling_rate": 16000, "n_fft": 400, "hop_length": 160, "feature_size": 128,
        "min_length": 8000, "n_window": 50,
    }}
    (directory / "processor_config.json").write_text(json.dumps(processor))
    return model


# One logger and runtime for the module: TensorRT requires them to outlive the engines they create.
_RUNTIME: list = []


def _engine(plan: bytes):
    if not _RUNTIME:
        logger = trt.Logger(trt.Logger.WARNING)
        _RUNTIME.extend((logger, trt.Runtime(logger)))
    engine = _RUNTIME[1].deserialize_cuda_engine(plan)
    assert engine is not None
    return engine


_DTYPES = {trt.float32: torch.float32, trt.int32: torch.int32, trt.bfloat16: torch.bfloat16}


def _execute(context, engine, inputs: dict, outputs: dict) -> None:
    """Bind CUDA tensors by name and run one launch on the current stream."""
    for name, tensor in inputs.items():
        if engine.is_shape_inference_io(name):
            raise AssertionError(f"unexpected shape tensor {name}")
        assert context.set_input_shape(name, tuple(tensor.shape)), name
        assert context.set_tensor_address(name, tensor.data_ptr()), name
    for name, tensor in outputs.items():
        assert context.set_tensor_address(name, tensor.data_ptr()), name
    assert context.infer_shapes() == []
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()


def _relative_error(actual, expected) -> float:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    return float((actual - expected).norm() / expected.norm())


@both_compute_types
def test_audio_encoder_matches_the_reference(tmp_path: Path, precision: str) -> None:
    from families.qwen3_asr.config import load_config
    from families.qwen3_asr.encoder_builder import build_encoder_plan
    from families.qwen3_asr.frontend import chunk_token_counts
    from families.qwen3_asr.weights import read_safetensors

    model = _tiny_checkpoint(tmp_path)
    config = load_config(tmp_path)
    plan = io.BytesIO()
    build_encoder_plan(config.encoder, read_safetensors(tmp_path), plan, max_windows=3,
                       verbose=False, compute=LIMITS[precision][0])
    engine = _engine(plan.getvalue())

    generator = torch.Generator().manual_seed(1)
    # 937 frames: 9 full chunks and a 37-frame chunk, so two attention windows, the second partial.
    for frames in (937, 64):
        chunks = -(-frames // 100)
        features = torch.zeros((1, 128, chunks * 100))
        features[:, :, :frames] = torch.randn((1, 128, frames), generator=generator)
        mask = torch.zeros((1, chunks * 100), dtype=torch.long)
        mask[:, :frames] = 1
        with torch.no_grad():
            expected = model.model.get_audio_features(features, mask, return_dict=True).pooler_output

        windows = -(-chunks // 8)
        grid = torch.zeros((windows * 8, 128, 100))
        grid[:chunks] = features[0].reshape(128, chunks, 100).permute(1, 0, 2)
        counts = chunk_token_counts(frames, chunk_frames=100)
        tokens = torch.zeros((windows * 8,), dtype=torch.int32)
        tokens[:chunks] = torch.tensor(counts, dtype=torch.int32)
        output = torch.empty((windows, 104, 32), device="cuda")
        context = engine.create_execution_context()
        _execute(
            context, engine,
            {"input_features": grid.reshape(windows, 8, 128, 100).cuda().contiguous(),
             "chunk_tokens": tokens.reshape(windows, 8).cuda().contiguous()},
            {"audio_embeds": output},
        )
        actual = output.reshape(-1, 32)[: sum(counts)].cpu()
        assert actual.shape == expected.shape
        error = _relative_error(actual, expected)
        print(f"encoder {precision} frames={frames} relative_error={error:.3e}")
        assert error <= LIMITS[precision][1], error


@both_compute_types
def test_decoder_prefill_and_decode_match_the_reference(tmp_path: Path, precision: str) -> None:
    from families.qwen3_asr.config import load_config
    from families.qwen3_asr.decoder_builder import build_decoder_plan
    from families.qwen3_asr.weights import read_safetensors

    model = _tiny_checkpoint(tmp_path)
    config = load_config(tmp_path)
    compute = LIMITS[precision][0]
    plan = io.BytesIO()
    build_decoder_plan(config.decoder, read_safetensors(tmp_path), plan, capacity=CAPACITY,
                       prefill_tokens=PREFILL_TOKENS, verbose=False, compute=compute)
    engine = _engine(plan.getvalue())
    assert engine.num_optimization_profiles == 2

    generator = torch.Generator().manual_seed(2)
    prompt = [5, 7, 9] + [AUDIO_TOKEN] * 6 + [11, 13]
    audio = torch.randn((6, 32), generator=generator)
    follow = [17, 23, 29]

    def reference_logits(ids: list[int]) -> torch.Tensor:
        with torch.no_grad():
            embeds = model.model.language_model.embed_tokens(torch.tensor([ids]))
            embeds[0, 3:9] = audio
            hidden = model.model.language_model(inputs_embeds=embeds).last_hidden_state
            return model.lm_head(hidden)[0, -1]

    cache_dtype = _DTYPES[compute]
    caches = {}
    for layer in range(config.decoder.layers):
        for kind in ("k", "v"):
            # Stale rows must not leak into attention: start from NaN, not zeros.
            caches[f"cache_{kind}_{layer}"] = torch.full(
                (1, config.decoder.kv_heads, CAPACITY, config.decoder.head_dim), float("nan"),
                dtype=cache_dtype, device="cuda",
            )
    aliased = {name.replace("cache_", "present_"): tensor for name, tensor in caches.items()}
    logits = torch.empty((1, VOCAB), device="cuda")
    key_positions = torch.arange(CAPACITY, dtype=torch.int32, device="cuda")

    prefill = engine.create_execution_context()
    decode = engine.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream
    assert decode.set_optimization_profile_async(1, stream)

    length = len(prompt)
    rows = torch.zeros((1, length, 32))
    rows[0, 3:9] = audio
    mask = torch.zeros((1, length), dtype=torch.int32)
    mask[0, 3:9] = 1
    for start in range(0, length, PREFILL_TOKENS):
        end = min(length, start + PREFILL_TOKENS)
        _execute(
            prefill, engine,
            {"input_ids": torch.tensor([prompt[start:end]], dtype=torch.int32).cuda(),
             "audio_embeds": rows[:, start:end].cuda().contiguous(),
             "audio_mask": mask[:, start:end].cuda().contiguous(),
             "cache_write_index": torch.tensor([start], dtype=torch.int32).cuda(),
             # More key rows than valid ones: the rows past the chunk hold NaN.
             "key_positions": key_positions[: end + 5].contiguous(), **caches},
            {"logits": logits, **aliased},
        )
    errors = [_relative_error(logits[0].cpu(), reference_logits(prompt))]

    ids = list(prompt)
    for step, token in enumerate(follow):
        ids.append(token)
        position = length + step
        _execute(
            decode, engine,
            {"input_ids": torch.tensor([[token]], dtype=torch.int32).cuda(),
             "audio_embeds": torch.zeros((1, 1, 32), device="cuda"),
             "audio_mask": torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
             "cache_write_index": torch.tensor([position], dtype=torch.int32).cuda(),
             "key_positions": key_positions[: min(CAPACITY, position + 1 + 7)].contiguous(), **caches},
            {"logits": logits, **aliased},
        )
        errors.append(_relative_error(logits[0].cpu(), reference_logits(ids)))
    print(f"decoder {precision} relative_errors={[f'{error:.3e}' for error in errors]}")
    assert max(errors) <= LIMITS[precision][1], errors

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-by-engine parity with the Transformers modules on a small random checkpoint.

Each engine is built in float32 (TF32 off on both sides) from a random checkpoint
in the original layout and compared with the corresponding module of
``VibeVoiceForConditionalGeneration`` loaded from the same tensors. These tests
need a CUDA GPU with TensorRT; they are skipped elsewhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
pytest.importorskip("transformers")

from families.vibevoice import config as config_module  # noqa: E402
from families.vibevoice.codec_engines import (  # noqa: E402
    build_feedback_engine,
    build_voice_encoder_engine,
    stream_state_size,
)
from families.vibevoice.head_engine import build_head_engine  # noqa: E402
from families.vibevoice.lm_engine import (  # noqa: E402
    DECODE_PROFILE,
    PREFILL_PROFILE,
    build_embed_engine,
    build_lm_engine,
)
from families.vibevoice.model import rope_inverse_frequencies  # noqa: E402
from families.vibevoice.tests import reference  # noqa: E402
from families.vibevoice.weights import Checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="engine parity needs a CUDA GPU"
)

CONTROL_IDS = [151643, 151652, 151653, 151654]
_TORCH = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.int32: torch.int32}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = reference.write_tiny_checkpoint(tmp_path_factory.mktemp("vibevoice_tiny"), seed=57)
    cfg = config_module.load(root)
    model = reference.load_reference(root, torch.float32, "cuda")
    return root, cfg, Checkpoint(root), model


class Engine:
    """Run a serialized engine on torch CUDA tensors."""

    def __init__(self, plan: bytes, profile: int = 0):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        self.context.set_optimization_profile_async(profile, self.stream.cuda_stream)
        # TensorRT needs a non-null address even for a zero-volume input.
        self.placeholder = torch.zeros(16, device="cuda")

    def __call__(self, **inputs):
        keep = []
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            keep.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            address = tensor.data_ptr() if tensor.numel() else self.placeholder.data_ptr()
            self.context.set_tensor_address(name, address)
        outputs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            outputs[name] = torch.empty(
                shape, dtype=_TORCH[self.engine.get_tensor_dtype(name)], device="cuda"
            )
            self.context.set_tensor_address(name, outputs[name].data_ptr())
        assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


def _rope(cfg, positions):
    inverse = torch.from_numpy(rope_inverse_frequencies(cfg))
    angles = torch.tensor(positions, dtype=torch.float32)[:, None] * inverse[None, :]
    angles = torch.cat([angles, angles], dim=-1).double()
    return angles.cos().float().cuda(), angles.sin().float().cuda()


def _close(actual, expected, tolerance):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    scale = expected.abs().max().clamp_min(1e-6)
    error = float(((actual - expected).abs().max() / scale))
    assert error <= tolerance, f"relative max error {error:.3e} > {tolerance:.1e}"
    return error


def test_rope_frequencies_match_the_reference(tiny):
    _, cfg, _, model = tiny
    expected = model.model.language_model.rotary_emb.inv_freq.float().cpu().numpy()
    np.testing.assert_array_equal(rope_inverse_frequencies(cfg), expected)


def test_language_model_prefill_and_decode(tiny):
    _, cfg, weights, model = tiny
    plan = build_lm_engine(
        weights,
        cfg,
        dtype=trt.float32,
        capacity=256,
        prefill_chunk=64,
        control_token_ids=CONTROL_IDS,
    )
    prefill, decode = Engine(plan, PREFILL_PROFILE), Engine(plan, DECODE_PROFILE)
    generator = torch.Generator().manual_seed(1)
    embeds = torch.randn(1, 23, cfg.hidden_size, generator=generator).cuda()
    step = torch.randn(1, 1, cfg.hidden_size, generator=generator).cuda()
    empty = torch.empty(1, 0, cfg.num_kv_heads, cfg.head_dim, device="cuda")
    pasts = {
        f"past_{kind}_{layer}": empty
        for layer in range(cfg.num_layers)
        for kind in ("key", "value")
    }
    cos, sin = _rope(cfg, range(23))
    first = prefill(inputs_embeds=embeds, rope_cos=cos, rope_sin=sin, **pasts)
    lm = model.model.language_model
    with torch.no_grad():
        out = lm(inputs_embeds=embeds, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1])
    _close(first["hidden"], out.last_hidden_state[:, -1], 2e-5)
    _close(first["control_logits"], logits[:, CONTROL_IDS], 2e-5)
    _close(first["log_normalizer"], torch.logsumexp(logits.double(), -1), 2e-5)
    for layer in range(cfg.num_layers):
        _close(
            first[f"present_key_{layer}"],
            out.past_key_values.layers[layer].keys.transpose(1, 2),
            2e-5,
        )
        _close(
            first[f"present_value_{layer}"],
            out.past_key_values.layers[layer].values.transpose(1, 2),
            2e-5,
        )
    pasts = {
        f"past_{kind}_{layer}": first[f"present_{kind}_{layer}"]
        for layer in range(cfg.num_layers)
        for kind in ("key", "value")
    }
    cos, sin = _rope(cfg, [23])
    second = decode(inputs_embeds=step, rope_cos=cos, rope_sin=sin, **pasts)
    with torch.no_grad():
        nxt = lm(inputs_embeds=step, past_key_values=out.past_key_values, use_cache=True)
        logits = model.lm_head(nxt.last_hidden_state[:, -1])
    _close(second["hidden"], nxt.last_hidden_state[:, -1], 2e-5)
    _close(second["control_logits"], logits[:, CONTROL_IDS], 2e-5)


def test_chunked_prefill_continues_the_cache(tiny):
    # A prompt longer than the prefill chunk arrives as several T > 1 calls
    # with a non-empty past: the causal mask and rotary positions must start
    # after the cached rows.
    _, cfg, weights, model = tiny
    plan = build_lm_engine(
        weights,
        cfg,
        dtype=trt.float32,
        capacity=256,
        prefill_chunk=16,
        control_token_ids=CONTROL_IDS,
    )
    prefill = Engine(plan, PREFILL_PROFILE)
    generator = torch.Generator().manual_seed(5)
    embeds = torch.randn(1, 37, cfg.hidden_size, generator=generator).cuda()
    empty = torch.empty(1, 0, cfg.num_kv_heads, cfg.head_dim, device="cuda")
    keys = [empty] * cfg.num_layers
    values = [empty] * cfg.num_layers
    for start, stop in ((0, 16), (16, 32), (32, 37)):
        cos, sin = _rope(cfg, range(start, stop))
        pasts = {f"past_key_{layer}": keys[layer] for layer in range(cfg.num_layers)}
        pasts.update({f"past_value_{layer}": values[layer] for layer in range(cfg.num_layers)})
        out = prefill(inputs_embeds=embeds[:, start:stop], rope_cos=cos, rope_sin=sin, **pasts)
        keys = [torch.cat([keys[i], out[f"present_key_{i}"]], 1) for i in range(cfg.num_layers)]
        values = [
            torch.cat([values[i], out[f"present_value_{i}"]], 1) for i in range(cfg.num_layers)
        ]
    with torch.no_grad():
        full = model.model.language_model(inputs_embeds=embeds, use_cache=True)
        logits = model.lm_head(full.last_hidden_state[:, -1])
    _close(out["hidden"], full.last_hidden_state[:, -1], 2e-5)
    _close(out["control_logits"], logits[:, CONTROL_IDS], 2e-5)
    for layer in range(cfg.num_layers):
        _close(keys[layer], full.past_key_values.layers[layer].keys.transpose(1, 2), 2e-5)


def test_embedding_gather(tiny):
    _, cfg, weights, model = tiny
    engine = Engine(build_embed_engine(weights, cfg, dtype=trt.float32, max_tokens=64))
    ids = torch.tensor([0, 151643, 151652, 151654, 42], dtype=torch.int32, device="cuda")
    rows = engine(token_ids=ids)["embeds"]
    expected = model.model.language_model.embed_tokens(ids.long())
    assert torch.equal(rows, expected)


def test_diffusion_head(tiny):
    _, cfg, weights, model = tiny
    engine = Engine(build_head_engine(weights, cfg, dtype=trt.float32))
    generator = torch.Generator().manual_seed(2)
    noisy = torch.randn(2, cfg.latent_size, generator=generator).cuda()
    condition = torch.randn(2, cfg.hidden_size, generator=generator).cuda()
    for value in (999.0, 500.0, 100.0):
        timestep = torch.full((2,), value, device="cuda")
        out = engine(noisy_latent=noisy, timestep=timestep, condition=condition)["velocity"]
        with torch.no_grad():
            expected = model.model.diffusion_head(noisy, timestep, condition=condition)
        _close(out, expected, 2e-5)


def test_feedback_step_streams_the_whole_sequence(tiny):
    _, cfg, weights, model = tiny
    engine = Engine(build_feedback_engine(weights, cfg, dtype=trt.float32))
    frames = 11
    generator = torch.Generator().manual_seed(3)
    latents = torch.randn(frames, cfg.latent_size, generator=generator).cuda()
    state = torch.zeros(stream_state_size(cfg), device="cuda")
    audio, embeds = [], []
    for frame in range(frames):
        out = engine(latent=latents[frame : frame + 1], state_in=state)
        state = out["state_out"].clone()
        audio.append(out["audio"])
        embeds.append(out["next_embed"])
    audio = torch.cat(audio, dim=1)
    vv = model.model
    with torch.no_grad():
        scaled = latents / vv.latent_scaling_factor - vv.latent_bias_factor
        expected_audio = vv.audio_tower.decode(scaled[None]).audio
        semantic = vv.semantic_tokenizer_encoder(expected_audio).latents[0]
        expected_embeds = vv.multi_modal_projector(latents) + vv.semantic_connector(semantic)
    _close(audio, expected_audio.reshape(1, -1), 2e-5)
    _close(torch.cat(embeds, dim=0), expected_embeds, 2e-5)


def test_voice_encoder(tiny):
    _, cfg, weights, model = tiny
    hop = cfg.acoustic.hop_length
    engine = Engine(
        build_voice_encoder_engine(weights, cfg, dtype=trt.float32, max_samples=64 * hop)
    )
    generator = torch.Generator().manual_seed(4)
    frames = 37
    audio = (0.1 * torch.randn(frames * hop, generator=generator)).cuda()
    noise = torch.randn(1, frames, cfg.acoustic.vae_dim, generator=generator).cuda()
    scale = torch.tensor([0.7], device="cuda")
    out = engine(audio=audio.reshape(1, 1, 1, -1), noise=noise, noise_scale=scale)["embeds"]
    vv = model.model
    with torch.no_grad():
        latents = vv.audio_tower.encode(audio.reshape(1, 1, -1), sample=False).latents
        latents = latents + (cfg.acoustic_vae_std * scale) * noise
        expected = vv.multi_modal_projector(
            (latents + vv.latent_bias_factor) * vv.latent_scaling_factor
        )
    _close(out, expected, 2e-5)


def test_tiny_checkpoint_matches_the_original_layout_listing():
    config, _ = reference.tiny_config()
    shapes = reference.original_tensor_shapes(config)
    assert json.loads(json.dumps(config))["model_type"] == "vibevoice"
    assert "model.prediction_head.final_layer.linear.weight" in shapes

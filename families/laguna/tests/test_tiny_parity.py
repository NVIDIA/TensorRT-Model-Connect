# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 graph parity of the Laguna engine against transformers on a tiny random model.

A random ``LagunaForCausalLM`` with every Laguna feature (per-layer head
counts, sliding and full attention, partial YaRN RoPE, softplus head gate,
dense first layer, sigmoid-routed MoE with a selection bias, routed scaling
and a shared expert) is written in the published per-expert checkpoint layout,
built into an FP32 engine by the family builder and run through the engine's
prefill (several chunks, past the sliding window) and decode profiles with the
runtime's KV-cache contract. Every compared logit row must match the
transformers forward pass. Requires CUDA, TensorRT and transformers >= 5.7.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")
laguna_hf = pytest.importorskip("transformers.models.laguna")

if not torch.cuda.is_available():
    pytest.skip("Laguna engine parity needs CUDA", allow_module_level=True)

from families.laguna.checkpoint import LagunaWeights  # noqa: E402
from families.laguna.config import load_config  # noqa: E402
from families.laguna.model import _build_plan  # noqa: E402


MAX_CACHE = 64
TINY = {
    "vocab_size": 96,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_attention_heads_per_layer": [4, 8, 8, 4],
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 24,
    "moe_routed_scaling_factor": 2.5,
    "sliding_window": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "bos_token_id": 2,
    "eos_token_id": [2],
    "pad_token_id": 0,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 8.0,
            "beta_slow": 1.0,
            "attention_factor": 1.1,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    },
}


def _hub_layout(model) -> dict[str, np.ndarray]:
    """The published Laguna checkpoint layout of a transformers state dict."""
    out = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().float().cpu().numpy()
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            inner = value.shape[1] // 2
            for expert in range(value.shape[0]):
                out[f"{prefix}{expert}.gate_proj.weight"] = value[expert, :inner]
                out[f"{prefix}{expert}.up_proj.weight"] = value[expert, inner:]
        elif name.endswith("mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for expert in range(value.shape[0]):
                out[f"{prefix}{expert}.down_proj.weight"] = value[expert]
        elif name.endswith("mlp.gate.e_score_correction_bias"):
            out[name.replace("mlp.gate.", "mlp.experts.")] = value
        else:
            out[name.replace(".mlp.shared_experts.", ".mlp.shared_expert.")] = value
    return {name: np.ascontiguousarray(value) for name, value in out.items()}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield _tiny_model(tmp_path_factory)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _tiny_model(tmp_path_factory):
    torch.manual_seed(0)
    config = laguna_hf.LagunaConfig(**TINY)
    model = laguna_hf.LagunaForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.copy_(1.0 + 0.2 * torch.randn(parameter.shape, generator=generator))
            elif name.endswith("e_score_correction_bias"):
                parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))
            else:
                fan_in = parameter.shape[-1]
                parameter.copy_(torch.randn(parameter.shape, generator=generator) / fan_in**0.5)
    directory = tmp_path_factory.mktemp("laguna-tiny")
    safetensors_numpy.save_file(_hub_layout(model), str(directory / "model.safetensors"))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "laguna", "architectures": ["LagunaForCausalLM"], **TINY}),
        encoding="utf-8",
    )
    model = model.to("cuda", dtype=torch.float32)
    model.config._attn_implementation = "eager"
    model.config._experts_implementation = "eager"
    return model, directory


class _Engine:
    """Drives the engine with the runtime's KV-cache and mask contract."""

    def __init__(self, plan, cfg) -> None:
        self.cfg = cfg
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        self.stream = torch.cuda.Stream()
        self.contexts = []
        for profile in (0, 1):
            context = self.engine.create_execution_context()
            context.set_optimization_profile_async(profile, self.stream.cuda_stream)
            self.contexts.append(context)
        width = cfg.kv_width
        self.cache = [
            (
                torch.zeros(MAX_CACHE, width, device="cuda"),
                torch.zeros(MAX_CACHE, width, device="cuda"),
            )
            for _ in range(cfg.num_hidden_layers)
        ]
        self.position = 0

    def run(self, tokens: list[int], profile: int) -> torch.Tensor:
        context = self.contexts[profile]
        rows = len(tokens)
        width = MAX_CACHE + rows
        mask = torch.full((rows, width), -1.0e4, device="cuda")
        mask[:, : self.position] = 0.0
        for row in range(rows):
            mask[row, MAX_CACHE : MAX_CACHE + row + 1] = 0.0
        tensors = {
            "token_id": torch.tensor(tokens, dtype=torch.int32, device="cuda"),
            "position_id": torch.arange(
                self.position, self.position + rows, dtype=torch.int32, device="cuda"
            ),
            "attention_mask": mask,
            "logits": torch.empty(1, self.cfg.vocab_size, device="cuda"),
        }
        for layer, (k, v) in enumerate(self.cache):
            tensors[f"cache_k_{layer}"] = k
            tensors[f"cache_v_{layer}"] = v
            tensors[f"present_k_{layer}"] = torch.empty(rows, self.cfg.kv_width, device="cuda")
            tensors[f"present_v_{layer}"] = torch.empty(rows, self.cfg.kv_width, device="cuda")
        for name in ("token_id", "position_id", "attention_mask"):
            assert context.set_input_shape(name, tuple(tensors[name].shape))
        for name, tensor in tensors.items():
            context.set_tensor_address(name, tensor.data_ptr())
        torch.cuda.synchronize()
        assert context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        for layer, (k, v) in enumerate(self.cache):
            k[self.position : self.position + rows] = tensors[f"present_k_{layer}"]
            v[self.position : self.position + rows] = tensors[f"present_v_{layer}"]
        self.position += rows
        return tensors["logits"][0].clone()


@pytest.fixture(scope="module")
def engine(tiny):
    _, directory = tiny
    cfg = load_config(directory)
    weights = LagunaWeights(directory, cfg, np.dtype(np.float32))
    plan, chunk = _build_plan(cfg, weights, MAX_CACHE, verbose=False)
    assert chunk == MAX_CACHE
    return lambda: _Engine(plan, cfg)


def _reference_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        ids = torch.tensor([tokens], device="cuda")
        return model(input_ids=ids).logits[0].float()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    scale = expected.abs().max().item()
    error = (actual - expected).abs().max().item()
    assert error <= 2e-4 * max(scale, 1.0), f"{what}: max |diff| {error:.3e}, scale {scale:.3f}"


def test_hub_layout_round_trips_through_transformers(tiny) -> None:
    model, directory = tiny
    reloaded = laguna_hf.LagunaForCausalLM.from_pretrained(directory, dtype=torch.float32)
    reloaded = reloaded.to("cuda").eval()
    reloaded.config._attn_implementation = "eager"
    reloaded.config._experts_implementation = "eager"
    tokens = list(range(3, 30))
    torch.testing.assert_close(
        _reference_logits(reloaded, tokens), _reference_logits(model, tokens), rtol=0, atol=0
    )


def test_chunked_prefill_then_decode_matches_transformers(tiny, engine) -> None:
    model, _ = tiny
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(3, TINY["vocab_size"], (48,), generator=generator).tolist()
    expected = _reference_logits(model, tokens)
    runner = engine()
    # Prefill 40 tokens in chunks of 16 / 16 / 8 (past the 8-token window).
    for start, stop in ((0, 16), (16, 32), (32, 40)):
        logits = runner.run(tokens[start:stop], profile=0)
        _assert_close(logits, expected[stop - 1], f"prefill chunk ending at {stop - 1}")
    for position in range(40, 48):
        logits = runner.run([tokens[position]], profile=1)
        _assert_close(logits, expected[position], f"decode position {position}")


def test_decode_only_matches_transformers(tiny, engine) -> None:
    model, _ = tiny
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(3, TINY["vocab_size"], (24,), generator=generator).tolist()
    expected = _reference_logits(model, tokens)
    runner = engine()
    for position, token in enumerate(tokens):
        _assert_close(runner.run([token], profile=1), expected[position], f"position {position}")

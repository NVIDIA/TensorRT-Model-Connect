# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CosyVoice3 Qwen2 speech-token decoder, with compact GQA KV cache.

Offline B=1 only. Checkpoint: llm.pt (not the distinct llm.rl.pt).
Equations and packing follow CosyVoice revision 074ca6dc / CosyVoice3LM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .components import ComponentEngine, Graph

SPEECH_VOCAB = 6561
SPEECH_CLASSES = 6761
SOS, EOS, TASK = 6561, 6562, 6563
END_OF_PROMPT = 151646


@dataclass(frozen=True)
class LLMConfig:
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in ("rms_norm_eps", "rope_theta") and (type(value) is not int or value <= 0):
                raise ValueError(f"Invalid {name}")
        if (self.hidden_size % self.num_attention_heads or self.head_dim % 2
                or self.num_attention_heads % self.num_key_value_heads
                or not np.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0
                or not np.isfinite(self.rope_theta) or self.rope_theta <= 0):
            raise ValueError("Invalid Qwen2 attention/norm configuration")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


def weight_shapes(cfg):
    h, d, kv, f = cfg.hidden_size, cfg.head_dim, cfg.num_key_value_heads, cfg.intermediate_size
    result = {"text_embedding": (cfg.vocab_size, h), "speech_embedding": (SPEECH_CLASSES, h),
              "decoder": (SPEECH_CLASSES, h), "norm": (h,)}
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        for name, out, inp in (("q_proj", h, h), ("k_proj", kv * d, h), ("v_proj", kv * d, h),
                               ("o_proj", h, h), ("gate_proj", f, h), ("up_proj", f, h), ("down_proj", h, f)):
            result[p + name + ".weight"] = (out, inp)
            if name in ("q_proj", "k_proj", "v_proj"):
                result[p + name + ".bias"] = (out,)
        result[p + "input_layernorm"] = result[p + "post_attention_layernorm"] = (h,)
    return result


def validate_weights(weights, cfg):
    shapes = weight_shapes(cfg)
    if set(weights) != set(shapes):
        raise ValueError("Unexpected CosyVoice3 LLM weight keys")
    for key, shape in shapes.items():
        a = weights[key]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Invalid LLM weight: {key}, expected FP32 {shape}")


def load_weights(model_dir):
    import torch
    from .config import read_config

    read_config(model_dir)
    raw = json.loads((Path(model_dir) / "CosyVoice-BlankEN/config.json").read_text())
    cfg = LLMConfig()
    if (any(raw.get(k) != v for k, v in asdict(cfg).items()) or raw.get("model_type") != "qwen2"
            or raw.get("hidden_act") != "silu" or raw.get("use_sliding_window") is not False
            or raw.get("rope_scaling") is not None or raw.get("tie_word_embeddings") is not True):
        raise ValueError("Only the published CosyVoice3 Qwen2 architecture is supported")
    state = torch.load(Path(model_dir) / "llm.pt", weights_only=True, mmap=True, map_location="cpu")
    mapping = {"text_embedding": "llm.model.model.embed_tokens.weight",
               "speech_embedding": "speech_embedding.weight", "decoder": "llm_decoder.weight",
               "norm": "llm.model.model.norm.weight"}
    for key in weight_shapes(cfg):
        if key in mapping:
            continue
        layer, index, name, *suffix = key.split(".")
        prefix = f"llm.model.model.{layer}.{index}."
        if name.endswith("layernorm"):
            mapping[key] = prefix + name + ".weight"
        else:
            mapping[key] = prefix + ("self_attn." if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp.") + name + "." + suffix[0]
    if set(state) != set(mapping.values()) | {"llm.model.lm_head.weight"}:
        raise ValueError("Unexpected LLM checkpoint parameters")
    if not torch.equal(state["llm.model.lm_head.weight"], state[mapping["text_embedding"]]):
        raise ValueError("Expected tied Qwen2 text head; it is not used for speech decoding")
    weights = {k: state[v].float().numpy() for k, v in mapping.items()}
    validate_weights(weights, cfg)
    return cfg, weights


def build_engine(weights, cfg=LLMConfig(), *, max_context=1024, opt_tokens=64, workspace_mib=256):
    validate_weights(weights, cfg)
    if (type(max_context) is not int or type(opt_tokens) is not int
            or not 1 <= opt_tokens <= max_context <= 32768):
        raise ValueError("Require 1 <= opt_tokens <= max_context <= 32768")
    g = Graph()
    t, net = g.trt, g.net
    h, d, heads, kv = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    ids = net.add_input("ids", t.int32, (1, -1))
    positions = net.add_input("positions", t.int32, (-1,))
    mask = net.add_input("mask", t.bool, (1, 1, -1, -1))
    keys = net.add_input("keys", t.float32, (cfg.num_hidden_layers, kv, -1, d))
    values = net.add_input("values", t.float32, (cfg.num_hidden_layers, kv, -1, d))
    n = g.dim(ids, 1)
    # One native gather; packing IDs chooses text or speech embedding explicitly.
    embedded = np.concatenate((weights["text_embedding"], weights["speech_embedding"]), axis=0)
    x = g.gather(g.const(embedded), ids, 0)
    cast = net.add_cast(positions, t.float32).get_output(0)
    freq = (1 / cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float32) / d)).astype(np.float32)
    angle = g.ew(g.reshape(cast, (1, 1, -1, 1)), g.const(freq[None, None, None]), "PROD")
    cos, sin = (g.cat([g.unary(angle, op)] * 2, 3) for op in ("COS", "SIN"))
    rotate_indices = g.const(np.r_[np.arange(d // 2, d), np.arange(d // 2)].astype(np.int32))
    signs = g.const(np.r_[-np.ones(d // 2), np.ones(d // 2)].astype(np.float32)[None, None, None])

    def norm(a, weight):
        square = g.ew(a, a, "PROD")
        mean = net.add_reduce(square, t.ReduceOperation.AVG, 4, True).get_output(0)
        inv = g.unary(g.ew(mean, g.scalar(cfg.rms_norm_eps, 3), "SUM"), "SQRT")
        return g.ew(g.ew(a, inv, "DIV"), g.const(weight[None, None]), "PROD")

    def rope(a):
        rotated = g.ew(g.gather(a, rotate_indices, 3), signs, "PROD")
        return g.ew(g.ew(a, cos, "PROD"), g.ew(rotated, sin, "PROD"), "SUM")

    present_k, present_v = [], []
    for i in range(cfg.num_hidden_layers):
        prefix = f"layers.{i}."

        def linear(a, name):
            return g.linear(a, weights[prefix + name + ".weight"], weights.get(prefix + name + ".bias"))

        a = norm(x, weights[prefix + "input_layernorm"])
        q, k, v = [g.transpose(g.reshape(linear(a, name + "_proj"), g.shape(1, n, nh, d)), (0, 2, 1, 3))
                   for name, nh in (("q", heads), ("k", kv), ("v", kv))]
        q, k = rope(q), rope(k)
        idx = g.const([i], np.int32)
        k = g.cat([g.gather(keys, idx, 0), k], 2)
        v = g.cat([g.gather(values, idx, 0), v], 2)
        present_k.append(k)
        present_v.append(v)
        total = g.dim(k, 2)
        # Compact GQA storage; broadcast K/V over the query-group axis only.
        q = g.reshape(q, g.shape(1, kv, heads // kv, n, d))
        kg = g.reshape(k, g.shape(1, kv, 1, total, d))
        vg = g.reshape(v, g.shape(1, kv, 1, total, d))
        q = g.ew(q, g.scalar(d ** -.5, 5), "PROD")
        scores = net.add_matrix_multiply(q, t.MatrixOperation.NONE, kg, t.MatrixOperation.TRANSPOSE).get_output(0)
        selected = net.add_select(g.reshape(mask, g.shape(1, 1, 1, n, total)), scores,
                                  g.scalar(-float("inf"), 5)).get_output(0)
        softmax = net.add_softmax(selected)
        softmax.axes = 1 << 4
        a = net.add_matrix_multiply(softmax.get_output(0), t.MatrixOperation.NONE, vg, t.MatrixOperation.NONE).get_output(0)
        a = g.transpose(g.reshape(a, g.shape(1, heads, n, d)), (0, 2, 1, 3))
        x = g.ew(x, linear(g.reshape(a, g.shape(1, n, h)), "o_proj"), "SUM")
        a = norm(x, weights[prefix + "post_attention_layernorm"])
        gate = linear(a, "gate_proj")
        gate = g.ew(gate, g.activation(gate, "SIGMOID"), "PROD")
        x = g.ew(x, linear(g.ew(gate, linear(a, "up_proj"), "PROD"), "down_proj"), "SUM")
    last = g.ew(n, g.const([1], np.int32), "SUB")
    hidden = norm(g.gather(x, last, 1), weights["norm"])
    g.mark(g.linear(hidden, weights["decoder"]), "logits")
    g.mark(g.cat(present_k, 0), "present_keys")
    g.mark(g.cat(present_v, 0), "present_values")
    profiles = {"ids": [(1, n) for n in (1, opt_tokens, max_context)],
                "positions": [(n,) for n in (1, opt_tokens, max_context)],
                "mask": [(1, 1, 1, 1), (1, 1, opt_tokens, opt_tokens * 2), (1, 1, max_context, 2 * max_context - 1)]}
    for name in ("keys", "values"):
        profiles[name] = [(cfg.num_hidden_layers, kv, n, d) for n in (0, min(opt_tokens, max_context - 1), max_context - 1)]
    # Make the optimization mask consistent even for a tiny max-context profile.
    profiles["mask"][1] = (1, 1, opt_tokens, opt_tokens + min(opt_tokens, max_context - 1))
    return g.build(profiles, workspace_mib)


def pack_prompt(text_ids, prompt_text_ids=(), prompt_speech_ids=(), *, vocab_size=151936):
    """[speech SOS, prompt text + target text, speech TASK, prompt speech]."""
    text, prompt_text, speech = (list(x) for x in (text_ids, prompt_text_ids, prompt_speech_ids))
    if not text or any(type(x) is not int or not 0 <= x < vocab_size for x in prompt_text + text):
        raise ValueError("Expected nonempty target text and valid text token IDs")
    if END_OF_PROMPT not in prompt_text + text:
        raise ValueError("CosyVoice3 requires <|endofprompt|> (151646)")
    if any(type(x) is not int or not 0 <= x < SPEECH_VOCAB for x in speech):
        raise ValueError("Prompt speech tokens must be in [0, 6561)")
    return [vocab_size + SOS, *prompt_text, *text, vocab_size + TASK, *(vocab_size + x for x in speech)]


def sample_token(logits, history, rng, *, min_tokens=0, greedy=False):
    """CPU RAS: top-p=.8/top-k=25; repeat in last 10 => full-distribution redraw.

    Match the pinned native CosyVoice3 sampling_ids implementation exactly:
    ignore_eos masks only 6561, NOT all 200 stop classes. Report early stops;
    do not pretend the upstream min-token option guarantees a minimum length.
    """
    scores = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if scores.size != SPEECH_CLASSES or not np.isfinite(scores).all():
        raise ValueError("Expected 6761 finite speech logits")
    if len(history) < min_tokens:
        scores[SOS] = -np.inf
    if greedy:
        return int(np.argmax(scores))
    p = np.exp(scores - np.max(scores))
    p /= p.sum()
    order = np.argsort(-p, kind="stable")[:25]
    count = min(len(order), int(np.searchsorted(np.cumsum(p[order]), .8)) + 1)
    candidates = order[:count]
    token = int(rng.choice(candidates, p=p[candidates] / p[candidates].sum()))
    if token in history[-10:]:
        p[token] = 0
        p /= p.sum()
        token = int(rng.choice(len(p), p=p))
    return token


class LLMEngine(ComponentEngine):
    def __init__(self, directory, *, device=0):
        super().__init__(directory, "llm", {"ids": "int32", "positions": "int32", "mask": "bool",
                                           "keys": "float32", "values": "float32"},
                         {"logits": "float32", "present_keys": "float32", "present_values": "float32"}, device=device)
        self.cfg = LLMConfig(**self.manifest["architecture"])
        self.max_context = self.manifest["max_context"]

    def step(self, ids, cache=None):
        torch, cfg = self.torch, self.cfg
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int32 or ids.device != self.device:
            raise ValueError("ids must be INT32 [1, N] on the engine device")
        if ids.numel() == 0 or (ids < 0).any().item() or (ids >= cfg.vocab_size + SPEECH_CLASSES).any().item():
            raise ValueError("Invalid joint embedding IDs")
        if cache is None:
            shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, 0, cfg.head_dim)
            cache = tuple(torch.empty(shape, dtype=torch.float32, device=self.device) for _ in range(2))
        if (len(cache) != 2 or any(x.ndim != 4 for x in cache) or cache[0].shape != cache[1].shape
                or cache[0].shape[:2] != (cfg.num_hidden_layers, cfg.num_key_value_heads)
                or cache[0].shape[-1] != cfg.head_dim):
            raise ValueError("Malformed compact GQA cache")
        past, n = cache[0].shape[2], ids.shape[1]
        if past + n > self.max_context:
            raise ValueError("LLM context limit exceeded; no silent truncation")
        positions = torch.arange(past, past + n, dtype=torch.int32, device=self.device)
        all_positions = torch.arange(past + n, dtype=torch.int32, device=self.device)
        result = self.run(ids=ids, positions=positions, mask=(all_positions[None] <= positions[:, None])[None, None],
                          keys=cache[0], values=cache[1])
        return result["logits"][0, 0], (result["present_keys"], result["present_values"])

    def generate(self, packed_ids, *, max_tokens, min_tokens=0, seed=0, greedy=False):
        if type(max_tokens) is not int or type(min_tokens) is not int or not 0 <= min_tokens <= max_tokens or max_tokens < 1:
            raise ValueError("Require 0 <= min_tokens <= max_tokens, max_tokens >= 1")
        if len(packed_ids) + max_tokens - 1 > self.max_context:
            raise ValueError("Requested generation exceeds LLM context profile")
        torch = self.torch
        ids = torch.tensor([packed_ids], dtype=torch.int32, device=self.device)
        rng, history, cache = np.random.default_rng(seed), [], None
        for _ in range(max_tokens):
            logits, cache = self.step(ids, cache)
            token = sample_token(logits.cpu().numpy(), history, rng, min_tokens=min_tokens, greedy=greedy)
            if token >= SPEECH_VOCAB:
                return {"tokens": history, "finish_reason": "stop_token", "stop_token": token}
            history.append(token)
            ids = torch.tensor([[self.cfg.vocab_size + token]], dtype=torch.int32, device=self.device)
        return {"tokens": history, "finish_reason": "length", "stop_token": None}

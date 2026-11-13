# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The padded PyTorch reference for one HSTU ranking request.

A dense, batch-1 port of NVIDIA recsys-examples (commit in
``families.hstu.weights.REFERENCE``), ``InferenceRankingGR.forward_nokvcache``:

- sparse lookup (``InferenceEmbedding``) of the contextual, item and action
  tables;
- ``hstu_preprocess_embeddings``: contextual rows, then the history's item and
  action embeddings interleaved, then the candidates' item embeddings;
- ``HSTUPositionalEncoder`` (inference, no time encoding):
  ``x * sqrt(d) + position[min(n, min(N - C, K - 1))]``;
- per layer, ``PagedHSTUInferLayer.forward_naive``: input LayerNorm, the
  uvqk projection with SiLU, ``pytorch_hstu_mha`` (``SiLU(alpha QK^T) /
  scaling_seqlen`` under ``_get_valid_attn_mask``: causal, target group 1, no
  contextual, no rab), ``u * LayerNorm(attention)``, the output projection
  and the residual;
- ``HSTUBlockPostprocessor``: the candidates' rows, L2-normalized;
- the ranking ``MLP``.

``precision="fp32"`` computes everything in FP32 from the BF16 weights (the
golden). ``precision="bf16"`` keeps every op's input and output in BF16 and
computes inside each op in FP32, rounding once per op output: the arithmetic
of recsys-examples' fused addmm-SiLU, layer-norm-mul and CUTLASS attention
kernels (its inference layer takes the fused Triton addmm-SiLU only on SM 8.x,
the unfused ``torch.addmm`` + SiLU on SM 9, 10 and 12, and refuses SM 11).
The reference's own BF16 spread is what the BF16 tolerance is declared from.
"""

from __future__ import annotations

import math

import torch

from ..config import HstuConfig


def valid_attention_mask(tokens: int, num_candidates: int) -> torch.Tensor:
    """``_get_valid_attn_mask`` for one unpadded sequence, causal, with
    ``num_targets = num_candidates``, ``target_group_size = 1`` and no
    contextual prefix: a token sees itself and every earlier token, except
    that a candidate is seen only by itself."""
    rows = torch.arange(tokens).view(tokens, 1)
    cols = torch.arange(tokens).view(1, tokens)
    first_candidate = tokens - num_candidates
    return (cols <= rows) & ((cols < first_candidate) | (cols == rows))


class Reference:
    def __init__(self, config: HstuConfig, tensors: dict[str, torch.Tensor], *, precision: str):
        if precision not in ("fp32", "bf16"):
            raise ValueError("reference precision is fp32 or bf16")
        self.config = config
        self.precision = precision
        self.tensors = tensors  # BF16, upcast where used (the tables stay BF16)

    def w(self, name: str) -> torch.Tensor:
        return self.tensors[name].float()

    def _round(self, value: torch.Tensor) -> torch.Tensor:
        """One op's output: BF16-rounded in bf16 mode, kept in fp32 otherwise."""
        return value.bfloat16().float() if self.precision == "bf16" else value

    def _layer_norm(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        weight, bias = self.w(f"{prefix}.weight"), self.w(f"{prefix}.bias")
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), weight, bias, self.config.layernorm_epsilon
        )

    def embed(self, record: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        contextual = torch.stack(
            [
                self.tensors[f"embedding.{feature.name}.weight"][record["user"][feature.name]]
                for feature in cfg.contextual_features
            ]
        ).float()
        item_ids = torch.tensor(
            record["history"][cfg.item_feature.name] + record["candidates"][cfg.item_feature.name]
        )
        items = self.tensors[f"embedding.{cfg.item_feature.name}.weight"][item_ids].float()
        actions = self.tensors[f"embedding.{cfg.action_feature.name}.weight"][
            torch.tensor(record["history"][cfg.action_feature.name])
        ].float()
        return contextual, items, actions

    def sequence(self, contextual, items, actions) -> torch.Tensor:
        cfg = self.config
        history = items[: cfg.history_length]
        candidates = items[cfg.history_length :]
        interleaved = torch.stack([history, actions], dim=1).reshape(2 * cfg.history_length, -1)
        x = torch.cat([contextual, interleaved, candidates], dim=0)
        position = self.w("position.weight")[torch.tensor(cfg.position_ids())]
        return self._round(x * math.sqrt(cfg.hidden_size) + position)

    def attention(self, q, k, v) -> torch.Tensor:
        cfg = self.config
        alpha = 1.0 / math.sqrt(cfg.head_dim)
        mask = valid_attention_mask(cfg.tokens, cfg.num_candidates)
        heads = []
        for head in range(cfg.num_attention_heads):
            scores = (q[:, head] @ k[:, head].T) * alpha
            probs = torch.nn.functional.silu(scores) / cfg.scaling_seqlen
            probs = self._round(torch.where(mask, probs, torch.zeros(())))
            heads.append(probs @ v[:, head])
            del scores, probs
        return self._round(torch.stack(heads, dim=1).reshape(cfg.tokens, cfg.attention_width))

    def layer(self, x: torch.Tensor, index: int) -> torch.Tensor:
        cfg = self.config
        prefix = f"layers.{index}"
        normed = self._round(self._layer_norm(x, f"{prefix}.input_layernorm"))
        uvqk = torch.nn.functional.linear(
            normed, self.w(f"{prefix}.uvqk.weight"), self.w(f"{prefix}.uvqk.bias")
        )
        uvqk = self._round(torch.nn.functional.silu(uvqk))
        width = cfg.attention_width
        user, value, query, key = torch.split(uvqk, [width] * 4, dim=-1)
        shape = (cfg.tokens, cfg.num_attention_heads, cfg.head_dim)
        attended = self.attention(query.reshape(shape), key.reshape(shape), value.reshape(shape))
        parallel = self._round(user * self._layer_norm(attended, f"{prefix}.output_layernorm"))
        return self._round(
            torch.nn.functional.linear(parallel, self.w(f"{prefix}.proj.weight")) + x
        )

    def candidates(self, x: torch.Tensor) -> torch.Tensor:
        rows = x[self.config.candidate_start :]
        return self._round(
            rows / torch.linalg.vector_norm(rows, dim=-1, keepdim=True).clamp(min=1e-6)
        )

    def head(self, x: torch.Tensor) -> torch.Tensor:
        layers = len(self.config.prediction_head_arch)
        for index in range(layers):
            x = torch.nn.functional.linear(
                x, self.w(f"head.{index}.weight"), self.w(f"head.{index}.bias")
            )
            if index < layers - 1:
                x = torch.relu(x)
            x = self._round(x)
        return x

    @torch.inference_mode()
    def forward(self, record: dict) -> dict[str, torch.Tensor]:
        contextual, items, actions = self.embed(record)
        x = self.sequence(contextual, items, actions)
        for index in range(self.config.num_layers):
            x = self.layer(x, index)
        candidate_embeddings = self.candidates(x)
        return {
            "candidate_embeddings": candidate_embeddings,
            "logits": self.head(candidate_embeddings),
        }

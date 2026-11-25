# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schema, with no dependency on a customer checkpoint or feature set."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

UPSTREAM = "https://github.com/meta-recsys/generative-recommenders"
REVISION = "474afe5cfdb3d16d603d04f943f4dbe7eb6866fd"


@dataclass(frozen=True)
class Config:
    model_type: str = "hstu_public_ranking"
    hidden_size: int = 256
    num_heads: int = 8
    history_length: int = 128
    decode_steps: int = 50
    candidates: int = 50
    vocab_size: int = 1024
    epsilon: float = 1e-6
    execution_method: str = "staged"
    bf16_policy: str = "strict"
    cuda_graphs: bool = True
    batching: str = "dynamic"
    bf16_attention_epilogue: bool = False

    def __post_init__(self):
        if self.model_type != "hstu_public_ranking":
            raise ValueError("expected hstu_public_ranking checkpoint")
        if (self.hidden_size, self.num_heads, self.history_length, self.decode_steps) != (
            256,
            8,
            128,
            50,
        ):
            raise ValueError("supported profile is one block, D256/H8, history128, decode50")
        if self.candidates not in (50, 64) or not 2 <= self.vocab_size <= 1_000_000:
            raise ValueError("invalid candidates or vocabulary size")
        if self.epsilon != 1e-6 or self.execution_method != "staged":
            raise ValueError("only upstream epsilon=1e-6 and staged runtime are supported")
        if self.bf16_policy not in ("strict", "fast"):
            raise ValueError("bf16_policy must be strict or fast")
        if not isinstance(self.cuda_graphs, bool):
            raise ValueError("cuda_graphs must be a boolean")
        if self.batching not in ("dynamic", "fixed"):
            raise ValueError("batching must be dynamic or fixed")
        if not isinstance(self.bf16_attention_epilogue, bool):
            raise ValueError("bf16_attention_epilogue must be a boolean")

    @classmethod
    def load(cls, path):
        return cls(**json.loads((Path(path) / "config.json").read_text()))

    def save(self, path):
        (Path(path) / "config.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    def runtime(self, batch, precision):
        def spec(name, dtype, *shape, **bounds):
            return dict(name=name, dtype=dtype, shape=[-1, *shape], **bounds)

        d, h, t, c, r = (
            self.hidden_size,
            self.num_heads,
            self.history_length,
            self.candidates,
            self.decode_steps,
        )
        history = [
            spec("history_ids", "int64", t, column_vocab_sizes=[self.vocab_size] * t),
            spec("history_lengths", "int64", 1, column_vocab_sizes=[t + 1]),
        ]
        request = [
            spec("candidate_ids", "int64", c, column_vocab_sizes=[self.vocab_size] * c),
            spec("num_candidates", "int64", 1, column_vocab_sizes=[c + 1]),
        ]
        context = [
            spec("history_k_b", "float32", h, t, d // h),
            spec("history_v_b", "float32", h, t, d // h),
            spec("history_mask", "int32", t),
            spec("candidate_emb", "float32", c, d),
            spec("valid_mask", "int32", c),
            spec("candidate_ids_context", "int64", c),
        ]
        state = [
            spec("scores", "float32", c),
            spec("placed_mask", "int32", c),
            spec("frozen_raw", "float32", r, d),
            spec("placed_count", "int64"),
        ]
        state_out = [dict(s, name=s["name"] + "_out") for s in state]
        return dict(
            family="hstu",
            task="structured_ranking",
            schema_version=1,
            execution="staged",
            execution_method="staged",
            history_cache_schema=1,
            candidates=c,
            decode_steps=r,
            max_batch_size=batch,
            min_batch_size=batch if self.batching == "fixed" else 1,
            batching=self.batching,
            precision=precision,
            bf16_policy=self.bf16_policy,
            cuda_graphs=self.cuda_graphs,
            bf16_attention_epilogue=self.bf16_attention_epilogue,
            inputs=history + request,
            outputs=[spec("scores", "float32", c), spec("picked_indices", "int64", r)],
            stages=[
                dict(name="history_prepare", inputs=history, outputs=context[:3]),
                dict(name="request_prepare", inputs=request, outputs=context[3:]),
                dict(
                    name="step",
                    inputs=context + state + [dict(name="step_index", dtype="int64", shape=[1])],
                    outputs=state_out + [spec("picked", "int64")],
                ),
            ],
        )

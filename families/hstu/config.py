# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schema, with no dependency on a customer checkpoint or feature set."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

UPSTREAM = "https://github.com/meta-recsys/generative-recommenders"
REVISION = "474afe5cfdb3d16d603d04f943f4dbe7eb6866fd"


@dataclass(frozen=True)
class Architecture:
    hidden_size: int
    num_heads: int
    vocab_size: int
    max_history_length: int
    max_decode_steps: int
    normalization_length: int
    epsilon: float


@dataclass(frozen=True)
class ServingProfile:
    history_length: int
    candidates: int
    decode_steps: int
    batching: str
    min_batch_size: int
    opt_batch_size: int | None


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
    max_history_length: int | None = None
    max_decode_steps: int | None = None
    normalization_length: int | None = None
    min_batch_size: int = 1
    opt_batch_size: int | None = None
    refinement_candidates: int = 4
    selection_margin: float = 0.02

    def __post_init__(self):
        if self.model_type != "hstu_public_ranking":
            raise ValueError("expected hstu_public_ranking checkpoint")
        self._resolve_dimensions()
        self._validate_architecture()
        self._validate_batching()
        self._validate_selection()
        self._validate_policy()

    def _resolve_dimensions(self):
        for key in (
            "hidden_size",
            "num_heads",
            "history_length",
            "decode_steps",
            "candidates",
            "vocab_size",
            "min_batch_size",
            "refinement_candidates",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key, fallback in (
            ("max_history_length", self.history_length),
            ("max_decode_steps", self.decode_steps),
            ("normalization_length", self.history_length + self.decode_steps + self.candidates),
        ):
            if getattr(self, key) is None:
                object.__setattr__(self, key, fallback)
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")

    def _validate_architecture(self):
        if (
            self.hidden_size % self.num_heads
            or self.hidden_size > 4096
            or self.num_heads > 64
            or self.candidates > 4096
            or self.decode_steps > 4096
            or self.history_length > self.max_history_length
            or self.decode_steps > self.max_decode_steps
            or not 2 <= self.vocab_size <= 1_000_000
        ):
            raise ValueError("invalid architecture or serving profile exceeds checkpoint capacity")

    def _validate_batching(self):
        if self.opt_batch_size is not None and (
            type(self.opt_batch_size) is not int or self.opt_batch_size < self.min_batch_size
        ):
            raise ValueError("opt_batch_size must be at least min_batch_size")
        if self.batching not in ("dynamic", "fixed"):
            raise ValueError("batching must be dynamic or fixed")

    def _validate_selection(self):
        import math

        if (
            isinstance(self.selection_margin, bool)
            or not isinstance(self.selection_margin, (int, float))
            or not math.isfinite(self.selection_margin)
            or self.selection_margin < 0
        ):
            raise ValueError("selection_margin must be finite and nonnegative")

    def _validate_policy(self):
        if self.epsilon != 1e-6 or self.execution_method != "staged":
            raise ValueError("only upstream epsilon=1e-6 and staged runtime are supported")
        if self.bf16_policy not in ("strict", "fast"):
            raise ValueError("bf16_policy must be strict or fast")
        for name in ("cuda_graphs", "bf16_attention_epilogue"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @classmethod
    def load(cls, path):
        data = json.loads((Path(path) / "config.json").read_text())
        if "architecture" in data:
            flat = {"model_type": data.pop("model_type")}
            for section in ("architecture", "serving", "precision"):
                values = data.pop(section)
                if set(flat) & set(values):
                    raise ValueError("duplicate checkpoint configuration field")
                flat.update(values)
            if data:
                raise ValueError("unknown checkpoint configuration section")
            data = flat
        return cls(**data)

    @property
    def architecture(self):
        return Architecture(
            **{name: getattr(self, name) for name in Architecture.__dataclass_fields__}
        )

    @property
    def serving(self):
        return ServingProfile(
            **{name: getattr(self, name) for name in ServingProfile.__dataclass_fields__}
        )

    def save(self, path):
        rest = asdict(self)
        architecture, serving = asdict(self.architecture), asdict(self.serving)
        for name in (*architecture, *serving, "model_type"):
            rest.pop(name)
        data = dict(
            model_type=self.model_type, architecture=architecture, serving=serving, precision=rest
        )
        (Path(path) / "config.json").write_text(json.dumps(data, indent=2) + "\n")

    def runtime(self, batch, precision):
        minimum = batch if self.batching == "fixed" else self.min_batch_size
        optimum = batch if self.batching == "fixed" else (self.opt_batch_size or batch)
        if type(batch) is not int or not 1 <= minimum <= optimum <= batch:
            raise ValueError("batch profile must satisfy 1 <= min <= opt <= max")

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
            min_batch_size=minimum,
            opt_batch_size=optimum,
            architecture=asdict(self.architecture),
            serving=asdict(self.serving),
            batching=self.batching,
            precision=precision,
            bf16_policy=self.bf16_policy,
            refinement_candidates=self.refinement_candidates,
            selection_margin=self.selection_margin,
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

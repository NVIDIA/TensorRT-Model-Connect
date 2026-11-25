# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture comes from a complete trained checkpoint; profiles bound serving IO."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class Architecture:
    hidden_size: int = 256
    num_heads: int = 8
    attention_dim: int = 32
    value_dim: int = 32
    num_layers: int = 16
    vocab_size: int = 131263
    max_sequence_length: int = 211
    max_history_length: int = 200
    time_buckets: int = 128
    epsilon: float = 1e-6
    l2_epsilon: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ("epsilon", "l2_epsilon"):
                if value != 1e-6:
                    raise ValueError("checkpoint normalization epsilon must be 1e-6")
            elif type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_capacity()

    def _validate_capacity(self):
        if (
            self.max_history_length >= self.max_sequence_length
            or self.num_layers > 64
            or self.num_heads > 64
            or self.hidden_size > 4096
            or self.vocab_size > 2_000_000
        ):
            raise ValueError("invalid checkpoint architecture")


@dataclass(frozen=True)
class ServingProfile:
    history_length: int = 200
    candidates: int = 64
    batching: str = "dynamic"
    min_batch_size: int = 1
    opt_batch_size: int | None = None

    def __post_init__(self):
        self._validate_dimensions()
        if self.batching not in ("fixed", "dynamic"):
            raise ValueError("batching must be fixed or dynamic")
        if self.opt_batch_size is not None and (
            type(self.opt_batch_size) is not int or self.opt_batch_size < self.min_batch_size
        ):
            raise ValueError("invalid optimum batch")

    def _validate_dimensions(self):
        if (
            type(self.history_length) is not int
            or self.history_length < 1
            or type(self.candidates) is not int
            or not 1 <= self.candidates <= 4096
            or type(self.min_batch_size) is not int
            or self.min_batch_size < 1
        ):
            raise ValueError("invalid serving profile")


@dataclass(frozen=True)
class Config:
    model_type: str = "hstu_meta_research"
    architecture: Architecture = field(default_factory=Architecture)
    serving: ServingProfile = field(default_factory=ServingProfile)
    cuda_graphs: bool = True
    bf16_attention_epilogue: bool = False
    bf16_projections: str = "none"
    fp32_block_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if self.model_type != "hstu_meta_research":
            raise ValueError("expected a converted complete HSTU research checkpoint")
        if self.serving.history_length > self.architecture.max_history_length:
            raise ValueError("serving history exceeds the trained checkpoint limit")
        if self.bf16_projections not in ("none", "output", "all"):
            raise ValueError("bf16_projections must be none, output, or all")
        indices = tuple(self.fp32_block_indices)
        if any(type(i) is not int or not 0 <= i < self.architecture.num_layers for i in indices):
            raise ValueError("FP32 block indices must refer to existing layers")
        if len(set(indices)) != len(indices):
            raise ValueError("FP32 block indices must be unique")
        object.__setattr__(self, "fp32_block_indices", indices)
        for name in ("cuda_graphs", "bf16_attention_epilogue"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("precision/graph options must be booleans")

    @classmethod
    def load(cls, directory):
        data = json.loads((Path(directory) / "config.json").read_text())
        data["architecture"] = Architecture(**data["architecture"])
        data["serving"] = ServingProfile(**data["serving"])
        return cls(**data)

    def save(self, directory):
        (Path(directory) / "config.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    def runtime(self, batch, precision):
        a, s = self.architecture, self.serving
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if precision == "bf16" and len(self.fp32_block_indices) == a.num_layers:
            raise ValueError("BF16 requires at least one reduced-precision block")
        minimum = batch if s.batching == "fixed" else s.min_batch_size
        optimum = batch if s.batching == "fixed" else (s.opt_batch_size or batch)
        if type(batch) is not int or not 1 <= minimum <= optimum <= batch:
            raise ValueError("batch profile must satisfy 1 <= min <= opt <= max")

        def spec(name, dtype, *shape, **kwargs):
            return dict(name=name, dtype=dtype, shape=[-1, *shape], **kwargs)

        t, c, d = s.history_length, s.candidates, a.hidden_size
        history = [
            spec("history_ids", "int64", t, column_vocab_sizes=[a.vocab_size] * t),
            spec("history_lengths", "int64", 1, column_vocab_sizes=[t + 1], integer_min=1),
            spec("history_timestamps", "int64", t + 1, integer_min=0, integer_max=2**40),
        ]
        request = [
            spec("candidate_ids", "int64", c, column_vocab_sizes=[a.vocab_size] * c),
            spec("num_candidates", "int64", 1, column_vocab_sizes=[c + 1]),
        ]
        context = [
            spec("history_k_b", "float32", a.num_layers, a.num_heads, t, a.attention_dim),
            spec("history_v_b", "float32", a.num_layers, a.num_heads, t, a.value_dim),
            spec("history_mask", "int32", t),
            spec("query_x", "float32", 1, d),
            spec("query_bias", "float32", a.num_layers, 1, t + 1),
        ]
        candidate = [spec("candidate_emb", "float32", c, d), spec("valid_mask", "int32", c)]
        outputs = [spec("scores", "float32", c), spec("picked_indices", "int64", 1)]
        return dict(
            family="hstu_research",
            task="structured_ranking",
            schema_version=1,
            execution="staged",
            history_cache_schema=1,
            decode_steps=1,
            candidates=c,
            precision=precision,
            bf16_projections=self.bf16_projections if precision == "bf16" else "none",
            fp32_block_indices=list(self.fp32_block_indices),
            bf16_attention_epilogue=self.bf16_attention_epilogue if precision == "bf16" else False,
            min_batch_size=minimum,
            opt_batch_size=optimum,
            max_batch_size=batch,
            batching=s.batching,
            cuda_graphs=self.cuda_graphs,
            architecture=asdict(a),
            serving=asdict(s),
            inputs=history + request,
            outputs=outputs,
            stages=[
                dict(name="history_prepare", inputs=history, outputs=context),
                dict(name="request_prepare", inputs=request, outputs=candidate),
                dict(name="step", inputs=context + candidate, outputs=outputs),
            ],
        )

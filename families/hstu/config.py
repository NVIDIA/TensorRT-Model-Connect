# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official MLPerf DLRMv3 architecture and bounded dense serving profiles."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class ServingProfile:
    history_length: int = 512
    candidates: int = 128
    batching: str = "dynamic"
    min_batch_size: int = 1
    opt_batch_size: int = 1

    def __post_init__(self):
        if (
            type(self.history_length) is not int
            or not 1 <= self.history_length <= 16384
            or type(self.candidates) is not int
            or not 1 <= self.candidates <= 2048
            or self.history_length + self.candidates + 1 > 18433
        ):
            raise ValueError("invalid DLRMv3 sequence profile")
        if self.batching not in ("fixed", "dynamic"):
            raise ValueError("batching must be fixed or dynamic")
        if any(type(x) is not int or x < 1 for x in (self.min_batch_size, self.opt_batch_size)):
            raise ValueError("invalid batch profile")


@dataclass(frozen=True)
class Config:
    model_type: str = "dlrmv3_hstu"
    serving: ServingProfile = field(default_factory=ServingProfile)
    cuda_graphs: bool = True
    fp32_block_indices: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        if self.model_type != "dlrmv3_hstu" or type(self.cuda_graphs) is not bool:
            raise ValueError("expected the official DLRMv3 dense checkpoint")

        indices = tuple(self.fp32_block_indices)
        if len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < 5 for i in indices
        ):
            raise ValueError("FP32 block indices must be unique values in 0..4")
        object.__setattr__(self, "fp32_block_indices", indices)

    @classmethod
    def load(cls, directory):
        data = json.loads((Path(directory) / "config.json").read_text())
        data["serving"] = ServingProfile(**data["serving"])
        return cls(**data)

    def save(self, directory):
        (Path(directory) / "config.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    def runtime(self, batch, precision):
        s = self.serving
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if precision == "bf16" and len(self.fp32_block_indices) == 5:
            raise ValueError("BF16 requires a reduced-precision HSTU block")
        minimum = batch if s.batching == "fixed" else s.min_batch_size
        optimum = batch if s.batching == "fixed" else s.opt_batch_size
        if type(batch) is not int or not 1 <= minimum <= optimum <= batch <= 64:
            raise ValueError("batch profile must satisfy 1 <= min <= opt <= max <= 64")

        def spec(name, dtype, *shape, **kwargs):
            return dict(name=name, dtype=dtype, shape=[-1, *shape], **kwargs)

        t, c, d = s.history_length, s.candidates, 512
        history = [
            spec("history_item_embeddings", "float32", t, d),
            spec("history_category_embeddings", "float32", t, d),
            spec("history_lengths", "int32", 1, integer_min=0, integer_max=t),
            spec("history_timestamps", "int64", t, integer_min=0, integer_max=2**40),
            spec("history_actions", "int32", t, integer_min=0, integer_max=31),
            spec("user_embeddings", "float32", 1, d),
            spec("query_timestamp", "int64", 1, integer_min=0, integer_max=2**40),
            spec("num_candidates", "int32", 1, integer_min=0, integer_max=c),
        ]
        request = [
            spec("candidate_item_embeddings", "float32", c, d),
            spec("candidate_category_embeddings", "float32", c, d),
        ]
        context = [
            spec("history_k_b", "float32", 5, 4, t + 1, 128),
            spec("history_v_b", "float32", 5, 4, t + 1, 128),
            spec("history_mask", "int32", t + 1),
            spec("candidate_mask", "int32", c),
            spec("query_timestamp_b", "int64", 1),
        ]
        candidates = [spec("candidate_x", "float32", c, d), spec("candidate_emb", "float32", c, d)]
        outputs = [spec("scores", "float32", c)]
        return dict(
            family="hstu",
            task="structured_ranking",
            schema_version=1,
            execution="staged",
            history_cache_schema=1,
            decode_steps=1,
            candidates=c,
            precision=precision,
            fp32_block_indices=list(self.fp32_block_indices)
            if precision == "bf16"
            else list(range(5)),
            min_batch_size=minimum,
            opt_batch_size=optimum,
            max_batch_size=batch,
            batching=s.batching,
            cuda_graphs=self.cuda_graphs,
            serving=asdict(s),
            inputs=history + request,
            outputs=outputs,
            stages=[
                dict(name="history_prepare", inputs=history, outputs=context),
                dict(name="request_prepare", inputs=request, outputs=candidates),
                dict(name="step", inputs=context + candidates, outputs=outputs),
            ],
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned engine ABI. No dependency on TensorRT or a decoding algorithm."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    """Phase and MIN/OPT/MAX row counts; index is its position in the manifest."""

    phase: str
    query: tuple[int, int, int]
    logits: tuple[int, int, int]

    def __post_init__(self):
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("unknown execution phase")
        for bounds in (self.query, self.logits):
            if (len(bounds) != 3 or any(type(x) is not int for x in bounds)
                    or not 1 == bounds[0] <= bounds[1] <= bounds[2]):
                raise ValueError("profile requires positive ordered MIN/OPT/MAX rows with MIN=1")
        if any(logits > query for logits, query in zip(self.logits, self.query)):
            raise ValueError("selected logits exceed query rows")


@dataclass(frozen=True)
class EngineContract:
    """Fixed-capacity linear state with explicit query visibility and positions.

    Tensor semantics, including aliasing and row alignment, are documented in
    ../SPECULATIVE_DECODING.md. A lowering may change without changing this ABI.
    """

    role: str
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    capacity: int
    max_query: int = 64
    feature_width: int = 0
    version: int = 1
    precision: str = "fp16"
    cache_layout: str = "batch_heads_capacity_dim"
    cache_update: str = "aliased_contiguous_append"
    alias_contract: str = "engine_required_alias"
    execution_profiles: tuple[ExecutionProfile, ...] = ()
    greedy_selection: str = "host"

    def __post_init__(self):
        if self.version != 1 or self.role not in {"target", "draft"}:
            raise ValueError("unsupported speculative engine contract")
        if self.precision != "fp16":
            raise ValueError("the first speculative ABI supports FP16 only")
        if self.greedy_selection not in {"host", "device_v1"}:
            raise ValueError("unsupported greedy selection contract")
        if self.greedy_selection == "device_v1" and self.role == "draft" and self.vocab_size < 2:
            raise ValueError("draft selection requires at least two vocabulary entries")
        linear = (self.cache_layout == "batch_heads_capacity_dim"
                  and self.cache_update == "aliased_contiguous_append"
                  and self.alias_contract == "engine_required_alias")
        if not linear:
            raise ValueError("unsupported speculative state layout or effects")
        for name in ("layers", "hidden_size", "kv_heads", "head_dim", "vocab_size",
                     "capacity", "max_query", "feature_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_query > self.capacity:
            raise ValueError("max_query exceeds cache capacity")
        if self.execution_profiles:
            if tuple(p.phase for p in self.execution_profiles) != ("prefill", "decode"):
                raise ValueError("expected profile 0=prefill and profile 1=decode")
            if max(p.query[2] for p in self.execution_profiles) != self.max_query:
                raise ValueError("max_query must cover exactly the declared profiles")
            if self.execution_profiles[0].logits != (1, 1, 1):
                raise ValueError("prefill selects one output row")
            if self.role == "draft" and self.execution_profiles[1].logits != (1, 1, 1):
                raise ValueError("draft selects one output row")

    def profile_shapes(self, name, shape):
        """Return each profile's MIN/OPT/MAX tensor shapes, independent of TRT."""
        if not self.execution_profiles:
            bounds = [(1, min(16, self.max_query), self.max_query)]
        else:
            bounds = [p.logits if name == "logits_indices" else p.query
                      for p in self.execution_profiles]
        return [tuple(tuple(rows if dim == -1 else dim for dim in shape) for rows in bound)
                for bound in bounds]

    def to_dict(self):
        return asdict(self)


def tree_visibility(parents: list[int]) -> list[list[int]]:
    """Inclusive ancestors in topological row order; root has parent -1."""
    if not parents or parents[0] != -1:
        raise ValueError("a candidate tree must start with one root")
    result = [[0] * len(parents) for _ in parents]
    for row, parent in enumerate(parents):
        if row and not 0 <= parent < row:
            raise ValueError("parents must precede their children")
        if row:
            result[row] = result[parent].copy()
        result[row][row] = 1
    return result

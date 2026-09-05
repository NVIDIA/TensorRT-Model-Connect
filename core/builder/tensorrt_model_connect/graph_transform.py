# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one optional user transform immediately before TensorRT serialization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from threading import Lock
from typing import Any, Callable, Iterator


GraphTransform = Callable[[Any, int], None]
"""An in-place ``transform(network, engine_index)`` callback."""


_PATCH_LOCK = Lock()
_ACTIVE: ContextVar[tuple[GraphTransform, Iterator[int]] | None] = ContextVar(
    "trtmc_graph_transform", default=None
)


class _TransformingBuilder:
    def __init__(self, builder: Any, transform: GraphTransform, indexes: Iterator[int]) -> None:
        self._builder = builder
        self._transform = transform
        self._indexes = indexes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)

    def build_serialized_network(self, network: Any, config: Any) -> Any:
        self._transform(network, next(self._indexes))
        return self._builder.build_serialized_network(network, config)


class _BuilderFactory:
    def __init__(self, builder: Any) -> None:
        self._builder = builder

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        builder = self._builder(*args, **kwargs)
        active = _ACTIVE.get()
        if active is None:
            return builder
        transform, indexes = active
        return _TransformingBuilder(builder, transform, indexes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)


@contextmanager
def graph_transform(transform: GraphTransform | None) -> Iterator[None]:
    """Scope a transform to TensorRT builders created by one build request."""

    if transform is None:
        yield
        return

    import tensorrt as trt

    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("graph transforms cannot run concurrently in one process")

    try:
        original_builder = trt.Builder
        trt.Builder = _BuilderFactory(original_builder)
        token = _ACTIVE.set((transform, count()))
        try:
            yield
        finally:
            _ACTIVE.reset(token)
            trt.Builder = original_builder
    finally:
        _PATCH_LOCK.release()

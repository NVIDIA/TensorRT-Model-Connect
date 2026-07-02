# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT IProfiler-based per-layer timing.

LayerProfiler is a tensorrt.IProfiler implementation that accumulates per-layer
timing across multiple execute_async_v3 calls and aggregates it into a JSON-
serializable dict.

Usage (standalone):
    from layer_profiler import LayerProfiler
    profiler = LayerProfiler()
    runner = TrtRunner(..., profiler=profiler)
    # run warmup, then:
    profiler.reset()
    # run timed iterations
    data = profiler.to_dict(metadata={"model": "...", "gpu": "..."})

TRT calls report_layer_time() synchronously after each layer during execution.
Times are host-wall-clock measurements of GPU kernel completion for that layer,
accurate for sequential (non-CUDA-graph) execution — which is how TrtRunner runs.
"""
from __future__ import annotations

import statistics

try:
    import tensorrt as _trt
    _IProfilerBase = _trt.IProfiler
except ImportError:
    _IProfilerBase = object  # type: ignore[assignment,misc]


class LayerProfiler(_IProfilerBase):
    """Accumulates per-layer TRT timing across multiple execute_async_v3 calls.

    Implements the tensorrt.IProfiler interface. Attach to a TRT execution
    context via ``context.profiler = profiler`` before the first execute call.
    """

    def __init__(self) -> None:
        try:
            _IProfilerBase.__init__(self)
        except TypeError:
            pass
        self._records: dict[str, list[float]] = {}

    # Called by TRT engine after each layer completes
    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self._records.setdefault(layer_name, []).append(ms)

    def reset(self) -> None:
        """Clear all accumulated records (call between warmup and measurement)."""
        self._records.clear()

    def to_dict(self, metadata: dict | None = None) -> dict:
        """Aggregate recorded times into a JSON-serializable dict.

        Layers are sorted by mean_ms descending (slowest first).
        """
        if not self._records:
            return {"metadata": metadata or {}, "layers": [], "total_ms": 0.0}

        total = sum(statistics.mean(v) for v in self._records.values())
        layers = []
        for name, times in self._records.items():
            mean_ms = statistics.mean(times)
            std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
            layers.append({
                "name": name,
                "mean_ms": round(mean_ms, 4),
                "std_ms": round(std_ms, 4),
                "pct": round(100.0 * mean_ms / total, 2) if total > 0 else 0.0,
                "calls": len(times),
            })
        layers.sort(key=lambda x: x["mean_ms"], reverse=True)
        return {
            "metadata": metadata or {},
            "layers": layers,
            "total_ms": round(total, 4),
        }

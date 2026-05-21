"""Schema for the ``runtime`` namespace.

Process-level decode-runtime knobs that used to live behind
``TRTMC_DISABLE_CUDA_GRAPH`` and ``TRTMC_GPU_ARGMAX``. Session/platform
layers only — these affect the in-process runtime, not bundled graphs.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


SCHEMA = Schema(
    namespace="runtime",
    fields=(
        ConfigField(
            name="disable_cuda_graph",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="prefer_gpu_greedy",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
    ),
)


register_schema(SCHEMA)

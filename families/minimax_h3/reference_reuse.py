# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an existing original image-reference bundle and stream its plans into a new one.

An incremental build names the plans it reuses; the caller selects compatible
compiled artifacts and owns writer finish/abort. This neither adapts plans to
another platform nor qualifies trained video output.
"""

import json
from pathlib import Path
import re
import struct

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC

from . import reference_build as owner

_CHUNK = 8 << 20


def _checkpoint_identity(model_dir):
    """Recognize the snapshot identity used by this family's source-bundle gate."""
    if model_dir is None:
        return None
    try:
        path = Path(model_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    repository = path.parent.parent.name
    if (
        path.parent.name != "snapshots"
        or re.fullmatch("[0-9a-f]{40}", path.name) is None
        or not repository.startswith("models--")
    ):
        return None
    return {
        "hf_id": repository.removeprefix("models--").replace("--", "/"),
        "revision": path.name,
    }


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class _ImageBundle:
    """Validate the source layout and metadata; copy selected plans with bounded host buffers."""

    def __init__(
        self,
        path,
        expected_runtime,
        checkpoint,
        tokenizer,
        *,
        reused_plans,
        allow_reuse_receipt=False,
    ):
        self.path = Path(path)
        self.identity = _identity(self.path.stat())
        self.reused_plans = frozenset(reused_plans)
        if not self.reused_plans or not self.reused_plans <= set(owner.PLAN_NAMES):
            raise ValueError("MiniMax-H3 source reuse requires explicit owned plan names")
        if type(allow_reuse_receipt) is not bool:
            raise ValueError("MiniMax-H3 reuse receipt admission must be explicit")
        with self.path.open("rb") as source:
            prefix = source.read(16)
            if len(prefix) != 16 or prefix[:8] != BUNDLE_MAGIC:
                raise ValueError("MiniMax-H3 source is not a format-v1 bundle")
            size = struct.unpack("<Q", prefix[8:])[0]
            if not 0 < size <= 100 << 20 or size > self.identity[2] - 16:
                raise ValueError("MiniMax-H3 source bundle has an invalid header length")
            raw_header = source.read(size)
            header = json.loads(raw_header)
            expected_header = dict(format=1, family="minimax_h3", task=owner.TASK, backend="trt")
            if any(header.get(k) != v for k, v in expected_header.items()):
                raise ValueError("MiniMax-H3 source bundle has a different family/task/backend")
            self.base = 16 + size
            self.sections = header["sections"]
            required = {
                *owner.PLAN_NAMES,
                "runtime.json",
                "tokenizer.json",
                "trtmc.provenance.json",
            }
            if set(self.sections) != required and not (
                allow_reuse_receipt
                and set(self.sections) == required | {"minimax_h3.reused_plans.json"}
            ):
                raise ValueError(
                    "MiniMax-H3 source requires the complete original eight-plan bundle"
                )
            position = 0
            for name, section in sorted(self.sections.items(), key=lambda x: x[1]["offset"]):
                offset, length = section["offset"], section["length"]
                if (
                    type(offset) is not int
                    or type(length) is not int
                    or offset != position
                    or length <= 0
                ):
                    raise ValueError(
                        "MiniMax-H3 source section ranges must be disjoint and complete"
                    )
                position += length
                if position > self.identity[2] - self.base:
                    raise ValueError("MiniMax-H3 source section exceeds its bundle")
            if position != self.identity[2] - self.base:
                raise ValueError("MiniMax-H3 source bundle has unaccounted trailing bytes")

            def metadata(name):
                section = self.sections[name]
                if section["length"] > 16 << 20:
                    raise ValueError("MiniMax-H3 source metadata is unexpectedly large")
                source.seek(self.base + section["offset"])
                return json.loads(source.read(section["length"]))

            section = self.sections["tokenizer.json"]
            source.seek(self.base + section["offset"])
            if section["length"] != len(tokenizer) or source.read(len(tokenizer)) != tokenizer:
                raise ValueError("MiniMax-H3 source tokenizer differs from the new checkpoint")

            if metadata("runtime.json") != expected_runtime:
                raise ValueError(
                    "MiniMax-H3 source profile/config/statistics differ from the requested base"
                )
            record = metadata("trtmc.provenance.json")
            if not checkpoint or record.get("checkpoint") != checkpoint:
                raise ValueError(
                    "MiniMax-H3 source and new build require the same pinned HF checkpoint"
                )
            build = record.get("build", {})
            if build.get("precision") != "bf16" or build.get("quantization") not in (None, "none"):
                raise ValueError("MiniMax-H3 source requires the original unquantized BF16 build")
            owner._require(
                build,
                dict(
                    tensor_parallel_size=1,
                    context_parallel_size=1,
                    max_batch_size=1,
                    dynamic_kv_cache=False,
                    fp32_layers=[],
                    max_sequence_length=None,
                    mtp_seq_len=None,
                    graph_transform=False,
                ),
                "source bundle build overrides",
            )
        self._unchanged()

    def _unchanged(self):
        if _identity(self.path.stat()) != self.identity:
            raise ValueError("MiniMax-H3 source bundle changed during reuse")

    def copy_to(self, name, writer):
        if name not in self.reused_plans:
            raise ValueError("MiniMax-H3 must rebuild every plan outside its explicit reuse set")
        self._unchanged()
        section = self.sections[name]
        with self.path.open("rb") as source, writer.open_section(name) as output:
            source.seek(self.base + section["offset"])
            remaining = section["length"]
            while remaining:
                data = source.read(min(_CHUNK, remaining))
                if not data:
                    raise ValueError("MiniMax-H3 source plan was truncated during copy")
                output.write(data)
                remaining -= len(data)
            self._unchanged()

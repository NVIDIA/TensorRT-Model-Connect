# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf 0.13 compatibility: grade by conversation, across timing phases.

The dataset sampler survives warmup while the phase credit counter restarts.
Use the dataset notification's conversation identity to select the gold and task,
keeping AIPerf's grader, output extraction, and exported request identity intact.
"""

from __future__ import annotations

from importlib.metadata import version


def install() -> None:
    if version("aiperf") != "0.13.0":
        return
    from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor

    cls = AccuracyRecordProcessor
    if getattr(cls, "_trtmc_conversation_alignment", False):
        return
    configure = cls.on_dataset_configured
    process = cls.process_record

    def on_dataset_configured(self, metadata):
        configure(self, metadata)
        graded = [item for item in metadata.conversations if item.accuracy_ground_truth is not None]
        self._trtmc_problem_indices = {item.conversation_id: index for index, item in enumerate(graded)}
        if len(self._trtmc_problem_indices) != len(graded):
            raise ValueError("accuracy dataset contains duplicate conversation IDs")

    async def process_record(self, record, metadata):
        indices = getattr(self, "_trtmc_problem_indices", {})
        if metadata.conversation_id not in indices:
            raise RuntimeError(f"accuracy conversation {metadata.conversation_id!r} has no configured gold answer")
        # The upstream method uses this index for both gold and task. A copy keeps
        # phase/session metadata unchanged, including for concurrent processors.
        selected = metadata.model_copy(update={"session_num": indices[metadata.conversation_id]})
        result = await process(self, record, selected)
        return result.model_copy(update={"session_num": metadata.session_num})

    cls.on_dataset_configured = on_dataset_configured
    cls.process_record = process_record
    cls._trtmc_conversation_alignment = True


install()

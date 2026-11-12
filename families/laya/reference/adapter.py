# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laya's released package (``laya==0.3.20``) for ``decide``, as tests/test_e2e.py runs it: the router picks
the checkpoint for a ``router`` bundle, the agent encodes the state with every question, one forward pass, and
each question's option probabilities are the softmax of its logits at the package's temperature."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from huggingface_hub import snapshot_download

        self.spec, self.host = spec, host
        self.checkpoint = Path(snapshot_download(spec.model, revision=spec.revision))
        self.variant = str(spec.options["variant"])  # the manifest's variant
        self.max_len = spec.options.get("max_sequence_length")
        self.agents: dict[str, Any] = {}

    def _agent(self, variant: str) -> Any:
        if variant not in self.agents:
            from laya import Agent

            from families.laya.cli import VARIANTS

            agent = Agent(str(self.checkpoint / VARIANTS[variant]), device=self.spec.device, fast=False, compile=False)
            if self.max_len is not None:
                agent.cfg["max_len"] = int(self.max_len)
            self.agents[variant] = agent
        return self.agents[variant]

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import json

        import numpy as np
        import torch
        from laya.common import collate_items, temp_bucket

        record = json.loads(str(self.host.required(request, "document")))
        variant = self.variant
        if variant == "router":
            from laya import Router

            variant = Router().route(record["state"], record["questions"], model=record.get("model")).model
        agent = self._agent(variant)
        ids = list(record["questions"])
        internal = {qid: agent._to_internal(record["questions"][qid]) for qid in ids}
        items = agent._encode_state(record["state"], ids, internal) if ids else []
        with torch.inference_mode():
            logits, model_ms = self.host.timed(
                lambda: agent._forward(collate_items([items], agent.tok.pad_token_id))[0] if items else [])
        scores = []
        for qid, item, raw in zip(ids, items, logits):
            count = len(item["markers"])
            scale = agent.temperature_by_options.get(temp_bucket(item["qtype"], count), agent.temperature[item["qtype"]])
            values = np.asarray(raw[:count].float().cpu(), dtype=np.float64) / scale
            probabilities = np.exp(values - values.max())
            probabilities /= probabilities.sum()
            scores.append({"question_id": qid, "logits": values.tolist(), "probabilities": probabilities.tolist()})
        return self.host.invocation({"questions": len(scores), "scores": scores}, model_ms)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion's native pipeline for the trtmc-perf-serve reference backend (``generate``): the
checkpoint's own remote code (``AutoModel``) generating autoregressively with ``ar_generate``, as the family's
benchmark reference does. Requests carry the catalog's ``text_generation_mode: ar`` (the serving base request),
which TRTMC honours too; another mode is refused rather than compared against a different decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from trtmc_perf_serving.backends.base import BackendError, Invocation
from trtmc_perf_serving.backends.reference.common import ReferenceSpec, invocation, required, timed

AUTOREGRESSIVE = ("ar", "autoregressive")


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        options = {"trust_remote_code": True, **({"revision": spec.revision} if spec.revision else {})}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **options)
        self.model = AutoModel.from_pretrained(spec.model, torch_dtype=spec.dtype, **options).eval().to(spec.device)

    def _input_ids(self, request: Mapping[str, Any]) -> Any:
        prompt = str(required(request, "prompt"))
        if request.get("use_chat_template"):
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                        add_generation_prompt=True,
                                                        enable_thinking=bool(request.get("enable_thinking", False)))
            return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"]

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        mode = str(request.get("text_generation_mode") or "auto").lower().replace("-", "_")
        if mode not in AUTOREGRESSIVE:
            raise BackendError(f"the native reference generates autoregressively only (requested {mode!r})")
        input_ids = self._input_ids(request).to(self.spec.device)
        steps = int(request.get("max_new_tokens", 128))
        generated, model_ms = timed(lambda: self.model.ar_generate(
            input_ids, max_new_tokens=steps, temperature=float(request.get("temperature") or 0.0),
            eos_token_id=self.tokenizer.eos_token_id))
        generated = generated[0] if isinstance(generated, tuple) else generated
        token_ids = [int(token) for token in generated[0, input_ids.shape[-1]:][:steps].cpu().tolist()]
        return invocation({"output_tokens": len(token_ids), "token_ids": token_ids,
                           "text": self.tokenizer.decode(token_ids, skip_special_tokens=True)}, model_ms)

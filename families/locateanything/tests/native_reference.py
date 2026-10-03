# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything's native pipeline for the trtmc-perf-serve reference backend (``generate``): the checkpoint's
own remote code at fp32 with the family's tokenizer, configuration, and rotary-buffer repair, the fixed 448x448
patchified image contract (which reads the image file itself), the manual chat prompt, and greedy "slow"
generation, as the family's benchmark reference runs it. The remote code imports video and dataset readers
(decord, lmdb, OpenCV) it does not use for images; where they are not installed (decord has no aarch64 wheel),
empty stand-ins satisfy Transformers' import check."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Mapping

from families.locateanything.tests.hf_reference import (_LocalTokenizer, _load_config, _repair_rotary_buffers,
                                                         manual_chat_prompt)
from families.locateanything.tests.vision_oracle import preprocess_image_inputs_for_trt
from trtmc_perf_serving.backends.base import BackendError, Invocation
from trtmc_perf_serving.backends.reference.common import ReferenceSpec, invocation, required, timed

UNUSED_IMPORTS = ("cv2", "decord", "lmdb")


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        if spec.precision != "fp32":
            raise BackendError("the official LocateAnything reference runs at fp32")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        for name in UNUSED_IMPORTS:
            if importlib.util.find_spec(name) is None:
                sys.modules.setdefault(name, types.ModuleType(name))
        self.spec, self.torch = spec, torch
        directory = Path(snapshot_download(spec.model, revision=spec.revision))
        self.tokenizer = _LocalTokenizer(directory)
        self.model = AutoModel.from_pretrained(directory, config=_load_config(directory), trust_remote_code=True,
                                               torch_dtype=torch.float32).to(spec.device).eval()
        _repair_rotary_buffers(self.model)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        torch, device = self.torch, self.spec.device
        prompt, steps = str(request.get("prompt", "")), int(request.get("max_new_tokens", 32))
        image = Path(str(required(request, "image_path")))

        def run() -> Any:
            inputs = preprocess_image_inputs_for_trt(image, fixed_image_size=448, patch_size=14,
                                                     image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                                                     interpolation="bicubic")
            encoded = self.tokenizer(manual_chat_prompt(prompt), return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            with torch.inference_mode():
                output = self.model.generate(
                    pixel_values=torch.from_numpy(inputs["pixel_values"]).to(device),
                    image_grid_hws=torch.from_numpy(inputs["image_grid_hws"]).to(device=device, dtype=torch.int32),
                    input_ids=input_ids, attention_mask=encoded["attention_mask"].to(device),
                    tokenizer=self.tokenizer, max_new_tokens=steps, use_cache=True, generation_mode="slow",
                    do_sample=False)
            return output, int(input_ids.shape[-1])

        (output, prompt_tokens), model_ms = timed(run)
        if isinstance(output, str):
            text = output
        elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
            text = output[0]
        else:
            ids = output[0] if output.ndim > 1 else output
            text = self.tokenizer.decode(ids[prompt_tokens:] if ids.numel() > prompt_tokens else ids,
                                         skip_special_tokens=True)
        text = str(text).strip()
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)

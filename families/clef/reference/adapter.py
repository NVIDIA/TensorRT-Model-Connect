# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clef's release implementation (the checkpoint's ``joint_schema_model.py``) for ``decide``, as
tests/reference.py runs it: the record encoded with the release processor, the backbone and the joint schema
head in one forward pass, each question's option probabilities the softmax of its logits."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import sys

        from huggingface_hub import snapshot_download

        self.spec, self.host = spec, host
        checkpoint = snapshot_download(spec.model, revision=spec.revision)
        sys.path.insert(0, checkpoint)
        import joint_schema_model as release

        self.release = release
        model, self.processor = release.load_release_model(Path(checkpoint), device="cpu")
        self.model = model.to(spec.device).eval()

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import json

        import numpy as np
        import torch

        record = json.loads(str(self.host.required(request, "document")))
        if request.get("image_paths"):
            record["images"] = [self.host.load_image(str(path)) for path in request["image_paths"]]
        if request.get("video_frame_paths"):
            record["videos"] = [np.stack([np.asarray(self.host.load_image(str(path))) for path in frames])
                                for frames in request["video_frame_paths"]]
        tokenizer = self.processor.tokenizer
        encoded = self.release.encode_record(tokenizer, record, processor=self.processor)
        batch = self.release.collate_records([encoded], tokenizer.pad_token_id, torch.device(self.spec.device))
        language = self.model.language_model

        def forward() -> Any:
            backbone = language.model if batch["media"] else language.model.language_model
            hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              use_cache=False, return_dict=True, **batch["media"]).last_hidden_state
            return self.model.head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"],
                                   language.get_output_embeddings().weight)[0]

        with torch.inference_mode():
            logits, model_ms = self.host.timed(forward)
        scores = [{"question_id": question.question_id, "option_ids": list(question.option_ids),
                   "logits": values.float().tolist(), "probabilities": values.float().softmax(-1).tolist()}
                  for question, values in zip(encoded.questions, logits)]
        return self.host.invocation({"questions": len(scores), "options": sum(len(s["option_ids"]) for s in scores),
                                     "input_tokens": len(encoded.input_ids), "scores": scores}, model_ms)

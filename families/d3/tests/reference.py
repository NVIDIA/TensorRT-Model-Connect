# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unconverted checkpoint in an explicitly separate reference interpreter."""
import json
import os
from pathlib import Path
import sys


def load_model(root, model_name="d3-edge"):
    sys.path.insert(0, str(root))
    import transformers
    if transformers.__version__ != "5.17.0":
        raise RuntimeError("D3 reference requires Transformers 5.17.0")
    from d3_runtime import D3
    return D3.from_pretrained(str(root), device="cuda:0", batch_size=1, model_name=model_name)


def evaluate(model, record, images=None):
    import torch
    images = images or []
    prepared = model.prepare(record["state"], record["questions"], images=images or None)
    if prepared.errors:
        raise AssertionError(prepared.errors)
    probs, tokens = model.run(prepared)
    rows = []
    for key in prepared.keys:
        question = prepared.questions[key]
        count = len(question.keys)
        if images:
            logits = model.image_logits([prepared.texts[key]], prepared.images, [count], prepared.lengths[key])
        else:
            logits = model.logits([prepared.sequences[key]], [count])
        rows.append({"question_id": key, "option_ids": question.keys,
                     "logits": (logits[0, :count] / model.temperature).tolist(),
                     "probabilities": probs[key]})
    torch.cuda.synchronize()
    return {"response": model.respond(prepared, probs, tokens), "scores": rows}


def main():
    import numpy as np
    from PIL import Image
    root, record_path, output_path = map(Path, sys.argv[1:4])
    model = load_model(root, os.environ.get("TRTMC_D3_REFERENCE_MODEL", "d3-edge"))
    images = []
    if len(sys.argv) > 4:
        pixels = np.fromfile(sys.argv[4], dtype=np.uint8).reshape(256, 256, 3)
        images = [Image.fromarray(pixels)]
    payload = evaluate(model, json.loads(record_path.read_text()), images)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

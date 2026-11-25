# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged upstream oracle in its separately provisioned environment."""

import json
from pathlib import Path
import sys


def main():
    import torch
    import transformers

    if transformers.__version__ != "5.20.0":
        raise RuntimeError(
            "Prepare the isolated reference environment from tests/requirements.txt "
            "and set TRTMC_JINA_EMBEDDINGS_V3_REFERENCE_PYTHON to its absolute Python interpreter"
        )
    directory = Path(sys.argv[1])
    case = json.loads(Path(sys.argv[2]).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = (
        AutoModel.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
        .eval()
        .to("cuda")
    )
    for name in ("retrieval_query", "retrieval_passage"):
        model.load_adapter(str(directory / name), adapter_name=name)
    model.eval()
    expected = {}
    for index, item in enumerate(case["inputs"]):
        if item["role"] == "default":
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_adapter(
                {"query": "retrieval_query", "document": "retrieval_passage"}[item["role"]]
            )
        inputs = tokenizer(item["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            valid = inputs["attention_mask"].unsqueeze(-1)
            mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1)
            expected[f"embedding_{index}"] = (
                torch.nn.functional.normalize(mean, dim=-1)[0].cpu().numpy()
            )
        expected[f"token_ids_{index}"] = inputs["input_ids"].cpu().numpy()
    np.savez_compressed(sys.argv[3], **expected)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged upstream oracle in its separately provisioned environment."""

import json
from pathlib import Path
import sys


def main():
    import torch
    import transformers

    if transformers.__version__ != "4.57.6":
        raise RuntimeError(
            "Prepare the isolated reference environment from tests/requirements.txt "
            "and set TRTMC_BAILING_HYBRID_REFERENCE_PYTHON to its absolute Python interpreter"
        )
    directory = Path(sys.argv[1])
    case = json.loads(Path(sys.argv[2]).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    reference = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=torch.float32,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    expected = []
    for prompt in case["prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, use_cache=False, max_new_tokens=case["max_new_tokens"]
            )
        ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        expected.append({"token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True)})
    Path(sys.argv[3]).write_text(json.dumps(expected, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

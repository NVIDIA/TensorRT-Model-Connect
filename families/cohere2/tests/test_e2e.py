# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual Tiny Aya checkpoint: native tokenizer, public task, greedy HF oracle."""

import gc
import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

MANIFEST = json.loads((Path(__file__).parent / "manifests/tiny-aya-global-fp32.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}
_BUNDLE = None


def selected(case, config):
    choices = list(config.getoption("--e2e-model", default=[]) or [])
    choices += list(config.getoption("--e2e-testcase", default=[]) or [])
    selection_file = config.getoption("--e2e-models-file", default=None)
    if selection_file:
        choices += Path(selection_file).read_text().splitlines()
    names = {
        v.strip()
        for item in choices
        for v in item.split(",")
        if v.strip() and not v.startswith("#")
    }
    if not names and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real checkpoint E2E requires explicit selection or TRTMC_E2E=1")
    if names and not names.intersection({"cohere2", MANIFEST["name"], case}):
        pytest.skip("case was not selected")


@pytest.mark.parametrize("case_name", CASES)
def test_e2e(case_name, request, tmp_path_factory):
    global _BUNDLE
    selected(case_name, request.config)
    case = CASES[case_name]
    record_evidence("inputs", {"manifest": MANIFEST, "case": case})
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    binary = Path(os.environ["TRTMC_BINARY"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert binary.is_file() and (runtime / "libtrtmc_model_cohere2.so").is_file()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(
        snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    )
    record_evidence(
        "checkpoint",
        {
            "path": str(checkpoint),
            "hf_id": MANIFEST["hf_id"],
            "hf_revision": MANIFEST["hf_revision"],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = [
        case["prompt"],
        "",
        "Hello 1234567!\n",
        "你好世界 — café",
        "I'm HERE.\n\n",
        "<BOS_TOKEN> test",
    ]
    helper = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_cohere2_tokenizer"
    with evidence_stage("preprocess"):
        completed = subprocess.run(
            [str(helper), str(checkpoint / "tokenizer.json")],
            input=json.dumps(samples),
            text=True,
            capture_output=True,
            check=True,
        )
        native_tokenizer = json.loads(completed.stdout)
        for text, result in zip(samples, native_tokenizer, strict=True):
            expected = tokenizer.encode(text, add_special_tokens=False)
            assert result["ids"] == expected, (text, result["ids"], expected)
            assert result["text"] == tokenizer.decode(expected, skip_special_tokens=True)
        prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=True)
        record_evidence(
            "preprocessing",
            {"samples": samples, "native_tokenizer": native_tokenizer, "prompt_ids": prompt_ids},
        )
    with evidence_stage("build"):
        if _BUNDLE is None:
            _BUNDLE = tmp_path_factory.mktemp("cohere2-bundle") / MANIFEST["bundle"]
            build(
                BuildRequest(
                    model_dir=checkpoint,
                    output_path=_BUNDLE,
                    family="cohere2",
                    task="text_continuation",
                    precision="fp32",
                    max_sequence_length=MANIFEST["max_sequence_length"],
                    tensor_parallel_size=MANIFEST["tensor_parallel_size"],
                )
            )
    record_evidence("bundle", {"path": str(_BUNDLE), "bytes": _BUNDLE.stat().st_size})
    command = [
        str(binary),
        "run",
        str(_BUNDLE),
        "--runtime-root",
        str(runtime),
        "--prompt",
        case["prompt"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
    ]
    with evidence_stage("native"):
        completed = subprocess.run(
            command, text=True, capture_output=True, check=True, timeout=1800
        )
        native = json.loads(completed.stdout)
        record_evidence("commands", {"argv": command})
        record_evidence("native", {**native, "stderr": completed.stderr[-4000:]})
        framed = command[: command.index("--prompt")] + [
            "--token-ids",
            json.dumps(prompt_ids),
            "--max-new-tokens",
            str(case["max_new_tokens"]),
        ]
        from_ids = json.loads(
            subprocess.run(framed, text=True, capture_output=True, check=True, timeout=1800).stdout
        )
        assert from_ids["token_ids"] == native["token_ids"]
    with evidence_stage("reference"):
        model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint, dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        ids = torch.tensor([prompt_ids], device="cuda")
        with torch.inference_mode():
            output = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                use_cache=False,
                max_new_tokens=case["max_new_tokens"],
            )
        expected = output[0, len(prompt_ids) :].tolist()
        reference = {
            "token_ids": expected,
            "text": tokenizer.decode(expected, skip_special_tokens=True),
        }
        record_evidence(
            "reference",
            {
                **reference,
                "precision": "fp32",
                "implementation": "transformers.Cohere2ForCausalLM",
                "use_cache": False,
            },
        )
        del model, ids, output
        gc.collect()
        torch.cuda.empty_cache()
    with evidence_stage("compare"):
        record_evidence("thresholds", {"token_ids": "exact", "decoded_text": "exact"})
        assert native["token_ids"] == reference["token_ids"]
        assert native["text"] == reference["text"]
        record_evidence("comparison", {"exact_token_match": True, "exact_text_match": True})

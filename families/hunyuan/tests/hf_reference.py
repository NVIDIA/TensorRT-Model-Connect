# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated Hunyuan HF oracle; never installs or changes project dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _reference_source() -> Path | None:
    """Resolve the existing CI-provisioned source contract without installing it."""
    value = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    if value is None:
        return None
    root = Path(value)
    source = root / "src"
    if not root.is_absolute() or not (source / "transformers" / "__init__.py").is_file():
        raise ValueError(
            "TRTMC_REFERENCE_SOURCE_DIR must name an absolute Transformers source checkout"
        )
    return source


def run_reference(payload: dict) -> dict:
    """Run HF outside the Model Connect interpreter when a newer oracle is needed."""
    _reference_source()
    python = os.environ.get("TRTMC_HUNYUAN_REFERENCE_PYTHON", sys.executable)
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("TRTMC_HUNYUAN_REFERENCE_PYTHON must name an absolute Python executable")
    with tempfile.TemporaryDirectory(prefix="hunyuan-reference-") as directory:
        output = Path(directory) / "reference.json"
        command = [python, "-I", str(Path(__file__).resolve()), str(output)]
        environment = os.environ.copy()
        # Parallel GPU cases must not each create a host-wide CPU thread pool.
        # Honor explicit caller settings and leave the parent unchanged.
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment.setdefault(variable, "1")
        result = subprocess.run(command, input=json.dumps(payload), text=True,
                                capture_output=True, timeout=600, check=False, env=environment)
        if result.returncode:
            raise RuntimeError(
                "Hunyuan independent reference failed. The project pins Transformers 5.2; "
                "provide the pinned tests/reference-source.json checkout through "
                "TRTMC_REFERENCE_SOURCE_DIR, or set TRTMC_HUNYUAN_REFERENCE_PYTHON "
                "to an isolated Transformers >=5.6,<6 interpreter. "
                "Do not upgrade Model Connect. "
                f"Command: {command!r}\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(output.read_text())


def _translation(payload, torch, transformers):
    model_dir = payload["model_dir"]
    case = payload["case"]
    if case.get("do_sample", False):
        raise ValueError("The current Hunyuan oracle covers deterministic translation only")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False)
    prompt = payload["prompt"]
    if case.get("use_chat_template", False):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=case.get("enable_thinking", False))
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    if "expected_prompt_token_ids" in case:
        assert inputs["input_ids"][0].tolist() == case["expected_prompt_token_ids"]
    precision = payload["reference_precision"]
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager").eval().to("cuda")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False,
                                max_new_tokens=case["max_new_tokens"],
                                repetition_penalty=case.get("repetition_penalty", 1.0))
    ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    return {
        "reference_ids": ids,
        "reference_text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
        "actual_decoded": tokenizer.decode(payload["actual_ids"], skip_special_tokens=True),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }


def _tiny(payload, torch):
    from transformers.models.hunyuan_v1_dense import (
        HunYuanDenseV1Config, HunYuanDenseV1ForCausalLM,
    )
    torch.manual_seed(812)
    config = HunYuanDenseV1Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=32, rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "dynamic", "rope_theta": 10000.,
                         "alpha": 4., "factor": 1.},
        bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = HunYuanDenseV1ForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.query_layernorm.weight.uniform_(0.5, 1.5)
            layer.self_attn.key_layernorm.weight.uniform_(0.5, 1.5)
    model.save_pretrained(payload["model_dir"])
    with torch.inference_mode():
        return {
            "prefill_logits": model(torch.tensor([[7, 41, 19]])).logits[0, -1].tolist(),
            "decode_logits": model(torch.tensor([[7, 41, 19, 23]])).logits[0, -1].tolist(),
        }


def _main():
    source = _reference_source()
    if source is not None:
        # Only this isolated child imports the newer official reference code.
        sys.path.insert(0, str(source))
    import torch
    import transformers
    from packaging.version import Version

    if source is not None and not Path(transformers.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Hunyuan reference did not import the provided Transformers source")

    if not Version("5.6") <= Version(transformers.__version__) < Version("6"):
        raise RuntimeError("Hunyuan independent HF oracle requires Transformers >=5.6,<6")
    payload = json.load(sys.stdin)
    if payload["mode"] == "translation":
        result = _translation(payload, torch, transformers)
    elif payload["mode"] == "tiny":
        result = _tiny(payload, torch)
    else:
        raise ValueError("Unknown Hunyuan reference mode")
    result["transformers_version"] = transformers.__version__
    result["torch_version"] = torch.__version__
    result["transformers_source"] = transformers.__file__
    Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    _main()

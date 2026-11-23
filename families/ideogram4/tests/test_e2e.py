# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published FP8 weights, native image generation, and the original PyTorch pipeline."""

import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

MANIFEST = json.loads((Path(__file__).parent / "manifests/ideogram4-fp8-bf16.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}
_BUNDLE = None


def require_selected(case, config):
    options = list(config.getoption("--e2e-model", default=[]) or [])
    options += list(config.getoption("--e2e-testcase", default=[]) or [])
    selection_file = config.getoption("--e2e-models-file", default=None)
    if selection_file:
        options += Path(selection_file).read_text().splitlines()
    selected = {
        v.strip()
        for item in options
        for v in item.split(",")
        if v.strip() and not v.startswith("#")
    }
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real checkpoint E2E requires explicit selection or TRTMC_E2E=1")
    if selected and not selected.intersection({"ideogram4", MANIFEST["name"], case}):
        pytest.skip("case was not selected")


def caption(color):
    return json.dumps(
        {
            "high_level_description": f"A {color} cup on a white table.",
            "style_description": {
                "aesthetics": "Minimalist",
                "lighting": "Soft studio light",
                "photo": "Studio product photography",
                "medium": "photograph",
            },
            "compositional_deconstruction": {
                "background": "White studio.",
                "elements": [
                    {"type": "obj", "bbox": [250, 250, 750, 750], "desc": f"A {color} cup."}
                ],
            },
        },
        separators=(",", ":"),
    )


def reference(root, prompt, initial):
    """The official components and sampler, replaying the same explicit initial latents."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from ideogram4.modeling_ideogram4 import Ideogram4Config
    from ideogram4.pipeline_ideogram4 import (
        Ideogram4Pipeline,
        Ideogram4PipelineConfig,
        _build_transformer,
        _load_autoencoder,
    )
    from ideogram4.quantized_loading import swap_linears_to_fp8, load_fp8_state_dict
    from ideogram4.sampler_configs import PRESETS
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16
    conditional = _build_transformer(
        Ideogram4Config(),
        load_file(root / "transformer/diffusion_pytorch_model.safetensors"),
        device,
        dtype,
    )
    unconditional = _build_transformer(
        Ideogram4Config(),
        load_file(root / "unconditional_transformer/diffusion_pytorch_model.safetensors"),
        device,
        dtype,
    )
    config = AutoConfig.from_pretrained(root / "text_encoder", local_files_only=True)
    text = AutoModel.from_config(config, attn_implementation="eager")
    weights = load_file(root / "text_encoder/model.safetensors")
    swap_linears_to_fp8(text, weights, compute_dtype=dtype)
    load_fp8_state_dict(text, weights, device=device, dtype=dtype, assign=True)
    del weights
    text.eval()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    decoder = _load_autoencoder(
        str(root / "vae/diffusion_pytorch_model.safetensors"), device, dtype
    )
    pipe = Ideogram4Pipeline(
        conditional,
        unconditional,
        text,
        tokenizer,
        decoder,
        Ideogram4PipelineConfig(),
        device,
        dtype,
    )
    height, width = MANIFEST["image_height"], MANIFEST["image_width"]
    preset = PRESETS["V4_DEFAULT_20"]
    schedule = get_schedule_for_resolution((height, width), known_mean=preset.mu, std=preset.std)
    intervals = make_step_intervals(preset.num_steps).to(device)
    with torch.inference_mode():
        inputs = pipe._build_inputs([prompt], height=height, width=width)
        features = pipe._encode_text(
            inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
        )
        ntext = inputs["max_text_tokens"]
        z = torch.from_numpy(initial.copy()).to(device)
        text_zeros = torch.zeros((1, ntext, 128), device=device, dtype=torch.float32)
        negative_features = torch.zeros_like(features[:, ntext:])
        for step in range(preset.num_steps - 1, -1, -1):
            t = schedule(intervals[step + 1].unsqueeze(0))
            s = schedule(intervals[step].unsqueeze(0))
            positive = conditional(
                llm_features=features,
                x=torch.cat([text_zeros, z], dim=1),
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )[:, ntext:]
            negative = unconditional(
                llm_features=negative_features,
                x=z,
                t=t,
                position_ids=inputs["position_ids"][:, ntext:],
                segment_ids=inputs["segment_ids"][:, ntext:],
                indicator=inputs["indicator"][:, ntext:],
            )
            velocity = (
                preset.guidance_schedule[step] * positive
                + (1.0 - preset.guidance_schedule[step]) * negative
            )
            z = z + velocity * (s - t)
        image = pipe._decode(z, grid_h=inputs["grid_h"], grid_w=inputs["grid_w"])[0]
    del pipe, conditional, unconditional, text, decoder
    gc.collect()
    torch.cuda.empty_cache()
    return image


@pytest.mark.parametrize("case_name", CASES)
def test_e2e(case_name, request, tmp_path, tmp_path_factory):
    global _BUNDLE
    require_selected(case_name, request.config)
    case = CASES[case_name]
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    root = Path(
        snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    )
    prompt = caption(case["color"])
    record_evidence("inputs", {"manifest": MANIFEST, "case": case, "prompt": prompt})
    record_evidence(
        "checkpoint",
        {"path": str(root), "hf_id": MANIFEST["hf_id"], "hf_revision": MANIFEST["hf_revision"]},
    )
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = Path(os.environ["TRTMC_BINARY"])
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    framed = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    with evidence_stage("preprocess"):
        from ideogram4.caption_verifier import CaptionVerifier

        assert CaptionVerifier().verify_raw(prompt) == []
        samples = [framed, "Cafe\u0301 and 中文 1234567", "<|im_start|>user\nhello"]
        helper = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_ideogram4_tokenizer"
        actual = json.loads(
            subprocess.run(
                [str(helper), str(root / "tokenizer/tokenizer.json")],
                input=json.dumps(samples),
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
        for sample, result in zip(samples, actual, strict=True):
            assert result["ids"] == tokenizer.encode(sample, add_special_tokens=False)
        record_evidence("preprocessing", {"framed_prompt": framed, "token_ids": actual[0]["ids"]})
    with evidence_stage("build"):
        if _BUNDLE is None:
            _BUNDLE = tmp_path_factory.mktemp("ideogram4-bundle") / MANIFEST["bundle"]
            build(
                BuildRequest(
                    model_dir=root,
                    output_path=_BUNDLE,
                    family="ideogram4",
                    task="text_to_image",
                    precision=MANIFEST["precision"],
                    image_height=MANIFEST["image_height"],
                    image_width=MANIFEST["image_width"],
                    max_sequence_length=MANIFEST["max_sequence_length"],
                    tensor_parallel_size=MANIFEST["tensor_parallel_size"],
                )
            )
    count = (MANIFEST["image_height"] // 16) * (MANIFEST["image_width"] // 16)
    initial = np.random.default_rng(case["seed"]).standard_normal((1, count, 128), dtype=np.float32)
    raw = tmp_path / "initial.raw"
    initial.tofile(raw)
    native_path = tmp_path / "native.png"
    command = [
        str(binary),
        "generate-image",
        str(_BUNDLE),
        "--runtime-root",
        str(runtime),
        "--prompt",
        prompt,
        "--output",
        str(native_path),
        "--initial-latents-raw",
        str(raw),
        "--num-steps",
        "20",
    ]
    with evidence_stage("native"):
        completed = subprocess.run(
            command, text=True, capture_output=True, check=True, timeout=3600
        )
        record_evidence("commands", {"argv": command})
        record_evidence(
            "native",
            {
                "image": native_path,
                "result": json.loads(completed.stdout),
                "stderr": completed.stderr[-4000:],
            },
        )
    with evidence_stage("reference"):
        expected = reference(root, prompt, initial)
        expected_path = tmp_path / "reference.png"
        expected.save(expected_path)
        record_evidence(
            "reference",
            {
                "image": expected_path,
                "implementation": "ideogram4@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2",
                "sampler": "V4_DEFAULT_20",
                "explicit_initial_latents": True,
            },
        )
    with evidence_stage("compare"):
        actual = np.asarray(Image.open(native_path).convert("RGB"), dtype=np.float32) / 255
        expected = np.asarray(expected, dtype=np.float32) / 255
        assert actual.shape == expected.shape == (512, 512, 3)
        mse = float(np.mean((actual - expected) ** 2))
        psnr = float("inf") if mse == 0 else -10 * np.log10(mse)
        mae = float(np.mean(np.abs(actual - expected)))
        record_evidence("thresholds", {"minimum_psnr_db": 25.0, "maximum_mae": 0.03})
        record_evidence(
            "comparison",
            {"psnr_db": float(psnr) if np.isfinite(psnr) else "infinite (exact match)", "mae": mae},
        )
        assert psnr >= 25.0 and mae <= 0.03, (psnr, mae)

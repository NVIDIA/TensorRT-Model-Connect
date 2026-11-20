# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime and official-reference E2E for Chroma.

Each case builds the bundle, generates through the native CLI from replayed
initial latents, and runs the official diffusers ``ChromaPipeline`` from the
same latents in the bundle's precision and in FP32.

Oracle. A flow-matching sampler amplifies rounding: on eight calibration
cases at 512 x 512 the official pipeline's own BF16 and FP32 images were
19.6-37.8 dB PSNR apart, and swapping only the attention kernel of the official
BF16 run moved it by 28.7-48.7 dB. Gates, from those cases:

* FP32 bundle: PSNR against the official FP32 image >= 38 dB (measured 40.8-72.3).
* BF16 bundle: PSNR against the official BF16 image >= 20 dB (measured 21.6-43.4),
  and the native image may trail the official BF16 image by at most 12 dB in
  PSNR against the official FP32 image (measured -11.7 to +5.7 dB; it is closer
  than the official BF16 image in seven of the eight cases).

The BF16 gates show coarse agreement only (a BF16 build with a wrong
RMSNorm epsilon also passed them); the BF16 rounding semantics are guarded by
the RMSNorm tests of ``test_component_parity.py`` and the host test, and the
graphs are proven to float rounding in FP32 there. Each case also checks that
the bundle carries the checked tokenizer, and runs the C and C++ SDK consumers
on the built bundle; ``test_native_tokenizer_matches_reference`` compares the
native tokenizer with the reference tokenizer, also truncated and padded.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence

FAMILY = "chroma"
TASK = "text_to_image"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
LATENT_CHANNELS = 16
TOKENIZER_PROMPTS = [
    "",
    "A cat",
    "  leading and   multiple   spaces ",
    "Caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u2014 \u201cquotes\u201d and \u2018single\u2019\u2026",
    "\uff46\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48 \uff21\uff22\uff23 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4",
    "emoji \U0001f431\u200d\U0001f464 \U0001f44d\U0001f3fd",
    "e\u0301 combining, tab\tand\nnewline, nbsp\u00a0and\u3000ideographic",
    "<extra_id_0> special </s> token<pad>",
    "a\u2581b \u2581\u2581c \u2581",
    "word " * 600,
]
# Code-point ranges of the seeded random prompts: ASCII, Latin, Greek, Cyrillic, Hebrew/Arabic, Devanagari,
# general punctuation and spaces, arrows/math, CJK symbols and kana, CJK, Hangul, full-width forms, emoji,
# combining marks, and ASCII control/whitespace characters.
TOKENIZER_RANGES = [
    (0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
    (0x2000, 0x206F), (0x2190, 0x22FF), (0x3000, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00),
    (0xFF00, 0xFFEF), (0x1F300, 0x1F64F), (0x0300, 0x036F), (0x09, 0x0D),
]
TOKENIZER_WORDS = ["a", "photo", "of", "cinematic", "lighting", "4k", "isn't", "e-mail", "U.S.A.", "3.14", "Chroma"]


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        for case in manifest["testcases"]:
            assert case["name"] not in result
            result[case["name"]] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    models = set()
    for raw in config.getoption("--e2e-model") or []:
        models.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip() for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcases = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcases.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not models and not testcases:
        return sorted(CASES), False
    selected = [
        name for name, (_, manifest, _) in CASES.items()
        if (not models or FAMILY in models or name in models or manifest["name"] in models)
        and (not testcases or name in testcases)
    ]
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names if enabled else [
        pytest.param(name, marks=pytest.mark.skip(reason="direct E2E requires an explicit E2E selector"))
        for name in names
    ]
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _text_length(manifest: dict) -> int:
    """The bundle's text budget; the build, the official pipeline and the tokenizer check all use it."""
    from families.chroma.model import MAX_TEXT_LENGTH

    return int(manifest.get("max_sequence_length") or MAX_TEXT_LENGTH)


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_CHROMA_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_CHROMA_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True,
            allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                            "transformer/*", "vae/*"],
        ))
    except Exception as error:  # noqa: BLE001 - reported as the missing-checkpoint precondition
        raise AssertionError(f"selected {FAMILY} E2E requires the cached checkpoint {manifest['hf_id']}") from error


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    return binary, runtime_root, native


def _environment(runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v)
    return env


def _thresholds(case_name: str) -> dict:
    return json.loads((THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8"))["threshold_overrides"]


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    """Packed [tokens, 64] float32 noise: torch CPU normal of the VAE latent, then 2x2 packing."""
    import torch

    height, width = manifest["image_height"] // 8, manifest["image_width"] // 8
    generator = torch.Generator("cpu").manual_seed(int(case["seed"]))
    latents = torch.randn((1, LATENT_CHANNELS, height, width), generator=generator, dtype=torch.float32)
    packed = latents.view(1, LATENT_CHANNELS, height // 2, 2, width // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return packed.reshape((height // 2) * (width // 2), LATENT_CHANNELS * 4).contiguous().numpy()


def _official(model_dir: Path, manifest: dict, case: dict, latents: np.ndarray, precision: str) -> np.ndarray:
    """ChromaPipeline image as the CLI would store it: uint8(clamp(x) * 255 + 0.5)."""
    import torch
    from diffusers import ChromaPipeline

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    pipe = ChromaPipeline.from_pretrained(model_dir, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        with torch.no_grad():
            image = pipe(
                prompt=case["prompt"],
                negative_prompt=case.get("negative_prompt", ""),
                height=manifest["image_height"],
                width=manifest["image_width"],
                num_inference_steps=case["num_inference_steps"],
                guidance_scale=case["guidance_scale"],
                max_sequence_length=_text_length(manifest),
                latents=torch.from_numpy(latents)[None].to("cuda"),
                output_type="np",
            ).images[0]
    finally:
        del pipe
        torch.cuda.empty_cache()
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    error = np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2)
    return math.inf if error == 0 else 10.0 * math.log10(255.0**2 / error)


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _tokenizer_prompts(count: int) -> list[str]:
    import random

    generator = random.Random(246)
    prompts = list(TOKENIZER_PROMPTS)
    for index in range(count):
        if index % 2:
            words = [generator.choice(TOKENIZER_WORDS) for _ in range(generator.randint(1, 40))]
            prompts.append("".join(word + generator.choice([" ", "  ", ", ", "\t", "\n"]) for word in words))
            continue
        characters = []
        for _ in range(generator.randint(1, 60)):
            low, high = generator.choice(TOKENIZER_RANGES)
            code = generator.randint(low, high)
            if not 0xD800 <= code <= 0xDFFF:
                characters.append(chr(code))
        if generator.random() < 0.2:
            special = generator.choice(["</s>", "<pad>", "<extra_id_3>"])
            characters.insert(generator.randint(0, len(characters)), special)
        prompts.append("".join(characters))
    return prompts


def _bundle_section(bundle: Path, name: str) -> bytes:
    import struct

    with bundle.open("rb") as handle:
        assert handle.read(8)[:6] == b"BUNDLE"
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
        section = header["sections"][name]
        # Section offsets count from the first byte after the header.
        handle.seek(16 + size + int(section["offset"]))
        return handle.read(int(section["length"]))


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    binary, runtime_root, native = _runtime()
    bundle = tmp_path / manifest["bundle"]
    build(BuildRequest(
        model_dir=model_dir, output_path=bundle, family=FAMILY, task=TASK,
        precision=manifest["precision"], image_height=manifest["image_height"],
        image_width=manifest["image_width"], max_sequence_length=_text_length(manifest),
        tensor_parallel_size=int(manifest["tensor_parallel_size"]),
    ))
    env = _environment(runtime_root)
    from families.chroma.tokenizer_asset import tokenizer_json

    # The bundle carries the tokenizer that test_native_tokenizer_matches_reference checks.
    assert _bundle_section(bundle, "tokenizer.json") == tokenizer_json(model_dir)

    latents = _initial_latents(manifest, case)
    latents_path = tmp_path / "latents.f32"
    latents.astype("<f4").tofile(latents_path)
    output = tmp_path / "native.png"
    command = [
        str(binary), "generate-image", str(bundle), "--runtime-root", str(runtime_root),
        "--prompt", case["prompt"], "--output", str(output), "--initial-latents-raw", str(latents_path),
        "--set", f"negative_prompt={case.get('negative_prompt', '')}",
        "--set", f"num_inference_steps={case['num_inference_steps']}",
        "--set", f"guidance_scale={case['guidance_scale']}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False,
                               timeout=int(case.get("runtime_timeout_s", 3600)))
    record_evidence("native", {"argv": command, "stdout": completed.stdout, "stderr": completed.stderr[-4000:]})
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["height"] == manifest["image_height"] and payload["width"] == manifest["image_width"]
    native_image = _read_png(output)

    same = _official(model_dir, manifest, case, latents, manifest["precision"])
    thresholds = record_evidence("thresholds", _thresholds(case_name))
    measured = {"native_vs_official_same_precision_db": _psnr(native_image, same)}
    if manifest["precision"] != "fp32":
        fp32 = _official(model_dir, manifest, case, latents, "fp32")
        measured["native_vs_official_fp32_db"] = _psnr(native_image, fp32)
        measured["official_same_precision_vs_fp32_db"] = _psnr(same, fp32)
    record_evidence("comparison", measured)
    print(f"{case_name}: " + ", ".join(f"{key} {value:.2f}" for key, value in measured.items()))
    assert native_image.shape == same.shape == (manifest["image_height"], manifest["image_width"], 3)
    assert measured["native_vs_official_same_precision_db"] >= thresholds["min_psnr_db_same_precision"]
    if manifest["precision"] != "fp32":
        assert (measured["native_vs_official_fp32_db"]
                >= measured["official_same_precision_vs_fp32_db"] - thresholds["max_fp32_drift_margin_db"])

    for language in ("c", "cpp"):
        consumer = subprocess.run([str(native / f"test_chroma_sdk_{language}"), str(bundle), str(runtime_root)],
                                  capture_output=True, text=True, env=env, timeout=3600)
        record_evidence(f"sdk_{language}", {"stdout": consumer.stdout, "stderr": consumer.stderr[-2000:]})
        assert consumer.returncode == 0, consumer.stderr[-2000:]


def test_native_tokenizer_matches_reference(case_name: str, tmp_path: Path) -> None:
    """The native T5 tokenizer reproduces the reference tokenizer the pipeline uses, also truncated and padded."""
    from transformers import AutoTokenizer

    from families.chroma.tokenizer_asset import tokenizer_json

    _, manifest, _ = CASES[case_name]
    model_dir = _model_dir(manifest)
    _, _, native = _runtime()
    length = _text_length(manifest)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    prompts = _tokenizer_prompts(500)
    cases = tmp_path / "tokenizer_cases.jsonl"
    with cases.open("w", encoding="utf-8") as handle:
        for text in prompts:
            padded = tokenizer(text, padding="max_length", max_length=length, truncation=True)
            handle.write(json.dumps({
                "text": text, "ids": tokenizer(text)["input_ids"], "length": length,
                "padded_ids": padded["input_ids"], "mask": padded["attention_mask"],
            }, ensure_ascii=False) + "\n")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_json(model_dir))
    completed = subprocess.run([str(native / "test_chroma_host"), str(tokenizer_path), str(cases)],
                               capture_output=True, text=True, timeout=600)
    record_evidence("tokenizer", {"prompts": len(prompts), "stdout": completed.stdout,
                                  "stderr": completed.stderr[-4000:]})
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert f"tokenizer parity cases: {len(prompts)}" in completed.stdout

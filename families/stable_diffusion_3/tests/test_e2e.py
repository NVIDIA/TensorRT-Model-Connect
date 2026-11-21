# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and diffusers-reference E2E for stable_diffusion_3.

The native CLI and the official ``StableDiffusion3Pipeline`` start from the same
initial latents and the same sampling settings, and the reference is the
official pipeline at the bundle's precision. An FP32 bundle is held to
near-identity with the FP32 pipeline (TF32 off). A sampler amplifies BF16
rounding into drift that depends on the seed, the kernels and, for FP16-stored
weights, their rounding to BF16, which both BF16 runs share; a BF16 bundle is
held to floors set below the observed distance between the two BF16 runs.
Each case also checks that its floors reject the official pipeline run without
classifier-free guidance.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "stable_diffusion_3"
TASKS = frozenset({"text_to_image"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _thresholds(case_name: str) -> dict:
    path = TEST_ROOT / "thresholds" / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            image_height=manifest["image_height"],
            image_width=manifest["image_width"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )


def _initial_latents(manifest: dict, case: dict):
    """Seeded standard-normal latents, shared by both sides through a raw file."""
    import torch

    generator = torch.Generator().manual_seed(int(case["seed"]))
    shape = (1, 16, int(manifest["image_height"]) // 8, int(manifest["image_width"]) // 8)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, latents: Path,
            destination: Path):
    import numpy as np
    from PIL import Image

    invocation = [
        str(binary), "generate-image", str(bundle), "--runtime-root", str(runtime_root),
        "--prompt", str(case["prompt"]), "--num-steps", str(case["num_steps"]),
        "--guidance-scale", str(case["guidance_scale"]),
        "--initial-latents-raw", str(latents), "--output", str(destination),
    ]
    if case["negative_prompt"]:
        invocation += ["--negative-prompt", str(case["negative_prompt"])]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        invocation, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    assert destination.is_file(), f"native {FAMILY} wrote no image: {completed.stdout[-500:]}"
    return np.asarray(Image.open(destination).convert("RGB"), dtype=np.float32) / 255.0


def _assert_sdk_consumers(runtime_root: Path, bundle: Path, manifest: dict) -> None:
    """The family-owned C and C++ consumers drive text_to_image through the public SDK."""
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    shape = f"{manifest['image_height']}x{manifest['image_width']}"
    for language, expected in (("c", f"text_to_image: {shape} channels=3"), ("cpp", f"run 0: {shape}")):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root)],
            check=True, capture_output=True, text=True, env=env, timeout=1800,
        )
        print(completed.stdout.strip())
        assert expected in completed.stdout, f"SDK {language} consumer output: {completed.stdout}"
        assert "all checks passed" in completed.stdout


def _official_reference(model_dir: Path, manifest: dict, case: dict, latents):
    """The official pipeline at the bundle's precision (FP32 without TF32, or BF16)."""
    import numpy as np
    import torch
    from diffusers import StableDiffusion3Pipeline

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[manifest["precision"]]
    pipeline = StableDiffusion3Pipeline.from_pretrained(str(model_dir), torch_dtype=dtype)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    image = pipeline(
        str(case["prompt"]),
        negative_prompt=str(case["negative_prompt"]),
        height=int(manifest["image_height"]),
        width=int(manifest["image_width"]),
        num_inference_steps=int(case["num_steps"]),
        guidance_scale=float(case["guidance_scale"]),
        max_sequence_length=int(manifest["max_sequence_length"]),
        latents=latents.to("cuda"),
        output_type="np",
    ).images[0]
    del pipeline
    torch.cuda.empty_cache()
    return np.asarray(image, dtype=np.float32)


def _psnr(left, right) -> float:
    import numpy as np

    mse = float(((left - right) ** 2).mean())
    return 99.0 if mse == 0.0 else 10.0 * float(np.log10(1.0 / mse))


def _ssim(left, right) -> float:
    """Global SSIM of the luminance-averaged images."""
    left, right = left.mean(axis=2), right.mean(axis=2)
    mean_left, mean_right = float(left.mean()), float(right.mean())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    stabilizer_one, stabilizer_two = 0.01 ** 2, 0.03 ** 2
    return float(
        ((2 * mean_left * mean_right + stabilizer_one) * (2 * covariance + stabilizer_two))
        / ((mean_left ** 2 + mean_right ** 2 + stabilizer_one)
           * (float(left.var()) + float(right.var()) + stabilizer_two))
    )


def _assert_parity(native, reference, thresholds: dict) -> None:
    import numpy as np

    assert np.isfinite(native).all(), f"native {FAMILY} image is not finite"
    assert native.shape == reference.shape, (
        f"image shape mismatch: native={native.shape} reference={reference.shape}"
    )
    std = float(native.std())
    assert std >= float(thresholds["min_pixel_std"]), f"native image is too flat: std {std:.4f}"
    psnr = _psnr(native, reference)
    ssim = _ssim(native, reference)
    print(f"{FAMILY} parity vs the official pipeline: PSNR {psnr:.2f} dB SSIM {ssim:.5f} "
          f"std {std:.4f}")
    assert psnr >= float(thresholds["psnr"]), f"PSNR {psnr:.2f} is below {thresholds['psnr']}"
    assert ssim >= float(thresholds["ssim"]), f"SSIM {ssim:.5f} is below {thresholds['ssim']}"


# Prompt text the tokenizers must frame exactly as the checkpoint's own tokenizers:
# plain ASCII, whitespace, truncation of both encoders, special tokens in text,
# the Unicode normalization the tokenizer files declare (NFC and lowercase for
# CLIP, the Precompiled NFKC charsmap for T5, applied per grapheme cluster) and
# words whose Unigram segmentations score nearly or exactly alike.
TOKENIZER_PROMPTS = [
    "A capybara holding a sign that reads Hello World",
    "",
    "   leading and trailing spaces   ",
    "multiple    internal     spaces, tabs\tand\nnewlines",
    "UPPER case, punctuation!? (parentheses) \"quotes\" don't 3.14159 e-mail@example.com",
    "a " + "very " * 300 + "long prompt that exceeds every encoder limit",
    "café naïve Ångström, decomposed cafe\u0301 and A\u030a",
    "МОСКВА и Москва, ΑΘΗΝΑ και Αθήνα, İzmir, Vietnamese Ơ Ư phở Đà Nẵng",
    "ＦＵＬＬ－ＷＩＤＴＨ ｔｅｘｔ, １２３, ligature ﬁne, ™ ½ ①",
    "emoji 🙂🚀, 中文提示词 日本語 한국어, Arabic مرحبا, Devanagari नमस्ते, Thai สวัสดี",
    "<|endoftext|> and </s> and <pad> and <extra_id_0> in text, <|StartOfText|> in any case",
    "near ties: lllllllllllllllllll ooooooooooooooooooo hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh "
    "GUIGUIGUIGUIGUIGUIGUIGUI",
    "a cat noooooooo cute, sooooo goooooood, yessssss, hahahahaha, zzzzzzzzzz",
    "controls end graphemes: tab\t\u0301 newline\n\U0001f3fd zero width\u200b\u0308 mark\u200f\u0610, "
    "prepended\u0600e\u0301, joiner e\u200dx\u0308",
    "cluster ends: \u00b2\uff9f \u00b2\u0e33 \u00aa\u0eb3 \u00b2\u102c \u00bc\u1038, specials absorb white "
    "space: <extra_id_0>\x85x a\u3000<extra_id_5>\u2009 b",
]


def _t5_reference(path: Path, length: int, pad_id: int):
    """tokenizer.json through the tokenizers library, with exactly parsed scores.

    The library's JSON reader can move a score by one unit in the last place,
    which flips exact ties between segmentations; the scores of the
    checkpoint's SentencePiece model, which the pipeline uses, are the decimal
    values in the file.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram

    model = json.loads(path.read_text(encoding="utf-8"))["model"]
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.model = Unigram([(piece, float(score)) for piece, score in model["vocab"]],
                              model["unk_id"], model.get("byte_fallback", False))
    tokenizer.enable_truncation(length)
    tokenizer.enable_padding(length=length, pad_id=pad_id)
    return tokenizer


def _pipeline_comparable(prompt: str) -> bool:
    """Text on which transformers 5.2.0's T5Tokenizer, which drops the
    Precompiled normalizer, still tokenizes as the checkpoint's tokenizer.json:
    no NFKC change, no control or format character but tab and newline, no
    literal metaspace."""
    import unicodedata

    return unicodedata.normalize("NFKC", prompt) == prompt and "\u2581" not in prompt and all(
        c in "\t\n" or unicodedata.category(c) not in ("Cc", "Cf", "Zl", "Zp") for c in prompt)


def test_tokenizers_match_the_checkpoint(case_name: str, tmp_path: Path) -> None:
    """The runtime tokenizers frame prompts as the checkpoint's tokenizers do.

    CLIP is compared with the pipeline's CLIPTokenizer, T5 with the
    checkpoint's tokenizer.json through the tokenizers library and, where
    the two agree in kind (``_pipeline_comparable``), with the pipeline's
    tokenizer.
    """
    from transformers import CLIPTokenizer, T5TokenizerFast

    from families.stable_diffusion_3 import prompt_tokenizers

    _, manifest, _ = CASES[case_name]
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    model_dir = _model_dir(manifest)
    t5_config = json.loads((model_dir / "text_encoder_3" / "config.json").read_text(encoding="utf-8"))
    t5_length = int(manifest["max_sequence_length"])
    request = {"clip_length": 77, "t5_length": t5_length, "prompts": TOKENIZER_PROMPTS}
    for component, key, encoder in (("tokenizer", "clip_l", "text_encoder"),
                                    ("tokenizer_2", "clip_g", "text_encoder_2")):
        data, ids = prompt_tokenizers.clip_tokenizer(model_dir, component)
        path = tmp_path / f"{key}.json"
        path.write_bytes(data)
        encoder_config = json.loads((model_dir / encoder / "config.json").read_text(encoding="utf-8"))
        request[key] = {"tokenizer": str(path), **ids, "config_eos": encoder_config["eos_token_id"]}
    t5_path = tmp_path / "t5.json"
    t5_path.write_bytes(prompt_tokenizers.t5_tokenizer(
        model_dir, {"pad_token_id": t5_config["pad_token_id"], "eos_token_id": t5_config["eos_token_id"]}))
    request["t5"] = {"tokenizer": str(t5_path), "eos": t5_config["eos_token_id"],
                     "pad": t5_config["pad_token_id"]}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [str(native_build / f"test_{FAMILY}_tokenize"), str(request_path)],
        check=True, capture_output=True, text=True, timeout=600,
    )
    native = json.loads(completed.stdout)

    for component, key in (("tokenizer", "clip_l"), ("tokenizer_2", "clip_g")):
        reference = CLIPTokenizer.from_pretrained(str(model_dir / component))
        for prompt, actual in zip(TOKENIZER_PROMPTS, native[key]):
            expected = reference(prompt, padding="max_length", max_length=77, truncation=True).input_ids
            assert actual["ids"] == expected, f"{key} ids differ for {prompt!r}"
            assert actual["pool"] == expected.index(max(expected)), f"{key} pooling for {prompt!r}"
    t5_file = _t5_reference(t5_path, t5_length, t5_config["pad_token_id"])
    pipeline_t5 = T5TokenizerFast.from_pretrained(str(model_dir / "tokenizer_3"))
    for prompt, actual in zip(TOKENIZER_PROMPTS, native["t5"]):
        assert actual == t5_file.encode(prompt).ids, f"t5 ids differ for {prompt!r}"
        if _pipeline_comparable(prompt):
            expected = pipeline_t5(prompt, padding="max_length", max_length=t5_length,
                                   truncation=True).input_ids
            assert actual == expected, f"t5 ids differ from the pipeline for {prompt!r}"


@pytest.mark.parametrize("threshold_case", sorted(CASES), ids=lambda name: f"control-{name}")
def test_parity_metrics_reject_a_different_image(threshold_case: str) -> None:
    """The case's thresholds must fail an image that is not the reference."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    # A smooth reference with texture, and the same picture mirrored.
    height, width = 64, 64
    y, x = np.mgrid[0:height, 0:width] / 64.0
    reference = np.stack([0.5 + 0.4 * np.sin(6 * x + 2 * y), 0.5 + 0.4 * np.cos(5 * y),
                          0.5 + 0.2 * np.sin(9 * x * y)], axis=2).astype(np.float32)
    reference += rng.normal(0.0, 0.02, reference.shape).astype(np.float32)
    mirrored = reference[:, ::-1].copy()
    with pytest.raises(AssertionError):
        _assert_parity(mirrored, reference, _thresholds(threshold_case))


def test_e2e(case_name: str, tmp_path: Path) -> None:
    import numpy as np

    _, manifest, case = CASES[case_name]
    assert manifest["trust_remote_code"] is False
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]

    # The native pipeline seeds with the C++ standard library and the reference
    # with torch, so a shared seed is not a shared starting point. Both are
    # given the same latents instead.
    latents = _initial_latents(manifest, case)
    raw = tmp_path / "latents.f32"
    raw.write_bytes(np.ascontiguousarray(latents.numpy(), dtype=np.float32).tobytes())

    _build(model_dir, bundle, manifest)
    native = _native(binary, runtime_root, bundle, case, raw, tmp_path / "native.png")
    _assert_sdk_consumers(runtime_root, bundle, manifest)
    reference = _official_reference(model_dir, manifest, case, latents)
    thresholds = _thresholds(case_name)
    _assert_parity(native, reference, thresholds)
    # Semantic control: the official pipeline without classifier-free guidance,
    # from the same latents, is a plausible wrong pipeline the oracle must reject.
    unguided = _official_reference(model_dir, manifest, {**case, "guidance_scale": 1.0}, latents)
    with pytest.raises(AssertionError):
        _assert_parity(unguided, reference, thresholds)

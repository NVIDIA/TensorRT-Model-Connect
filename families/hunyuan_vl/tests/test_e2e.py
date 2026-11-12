# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HunyuanOCR builds and native public-task runs against the Transformers reference.

``test_host_preprocessing_matches_reference`` compares the family's C++ tokenizer and image
preprocessing with the reference processor. ``test_e2e`` builds the bundle, runs the public
``images_text_to_text`` task natively and compares greedy token ids with the FP32 reference.

An FP32 bundle must reproduce every reference token. A BF16 bundle is held to the official model's
own BF16 run, made in the same test on the same GPU. Where that run equals the FP32 output, the
bundle must equal it too, and at least ``bf16_min_stable_requests`` requests must be of this kind.
Where the official BF16 run departs from the FP32 output (BF16 rounding flips a near-tied token),
the bundle must follow the FP32 output at least as long as the official run does, less a small
slack for flipping a near-tie earlier. After such a departure greedy continuations are no longer
comparable token by token (the official BF16 run's own drift differs widely between GPUs), so no
bound is placed on them; the FP32 case checks the same requests token for token.
"""

from pathlib import Path
import gc
import json
import os
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
FAMILY = "hunyuan_vl"
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text())
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)
# The released processor resizes on the CPU; these sizes cover plain rounding, ties, upscaling
# below the minimum area and downscaling above the maximum area (5000 x 3800 = 19 MP).
PROBE_SIZES = (
    (1180, 2400),
    (700, 1000),
    (528, 1040),
    (50, 100),
    (333, 517),
    (3000, 1500),
    (5000, 3800),
)
TOKENIZER_TEXTS = (
    "Extract all text from the image.",
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略。",
    "日本語のテキスト、カタカナとひらがな。 123456789 and 3.14159",
    "  leading spaces\ttabs\nnew lines\r\n\x0bvertical\x85next-line　ideographic  ",
    "Price: $1,234.56 (approx.) -- it's 'quoted' and \"double\" isn't it ?",
    "<table><tr><td>a</td></tr></table> $$\\frac{a}{b}$$ emoji 🙂 Ünïcödé",
)


def _selection(config, name):
    return {
        part.strip()
        for value in (config.getoption(name, default=[]) or [])
        for part in value.split(",")
        if part.strip()
    }


def _selected(config, names):
    models = _selection(config, "--e2e-model")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    cases = _selection(config, "--e2e-testcase")
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real family E2E requires explicit selection")
    if models and not models.intersection({FAMILY, *names}):
        pytest.skip("model profile was not selected")
    if cases and not cases.intersection(names):
        pytest.skip("testcase was not selected")


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_HUNYUAN_VL_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "config.json").is_file() and (root / "model.safetensors").is_file()
        return root
    from huggingface_hub import snapshot_download

    # The E2E reads only the exact cached snapshot, as CI provides it offline. The repository root
    # holds HunyuanOCR-1.5; v1.0/ and the dflash/ draft are other models.
    try:
        snapshot = snapshot_download(
            manifest["hf_id"],
            revision=manifest["hf_revision"],
            allow_patterns=["*.json", "*.jinja", "model.safetensors", "assets/*"],
            local_files_only=True,
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']} at "
            f"revision {manifest['hf_revision']} (download it first or set "
            "TRTMC_HUNYUAN_VL_MODEL_DIR)"
        ) from error
    return Path(snapshot)


def _native_tool(runtime, name):
    roots = [runtime]
    configured = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if configured:
        roots.insert(0, Path(configured))
    for library in ("libtrtmc_model_hunyuan_vl.so", "trtmc_model_hunyuan_vl.dll"):
        if (runtime / library).is_file():
            roots.append((runtime / library).resolve().parent)
    suffix = ".exe" if os.name == "nt" else ""
    for root in roots:
        for configuration in ("", "Release", "Debug"):
            candidate = root / "families/hunyuan_vl" / configuration / (name + suffix)
            if candidate.is_file():
                return candidate
    raise AssertionError(f"build the owning {name} CMake target")


def _document_image():
    """A deterministic printed page: dark text on white, rendered with Pillow's bundled font."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1024, 512), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    lines = (
        "TensorRT Model Connect",
        "HunyuanOCR native check",
        "Invoice No. 4711",
        "Total: 128.50 EUR",
    )
    for index, line in enumerate(lines):
        draw.text((48, 48 + index * 100), line, fill=(20, 20, 20), font=font)
    return image


def _paragraph_image():
    """A longer printed page whose transcription is a stable greedy target at both precisions."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1152, 1024), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    lines = (
        "Each model family lives in its own directory with",
        "its graph, runtime, tests and reference checks.",
        "The vision tower encodes one image per call, and",
        "the decoder reads fused text and image embeddings",
        "with four position axes for every token.",
        "Prefill runs in chunks of up to 2048 tokens.",
        "Numbers: 3.14159, 2.71828, 1,048,576 and 42.",
        "Dates: 2026-11-11 and 11/11/2026.",
        "End of the sample page.",
    )
    for index, line in enumerate(lines):
        draw.text((48, 48 + index * 100), line, fill=(10, 10, 10), font=font)
    return image


def _receipt_image():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 896), (250, 248, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    for index, line in enumerate(
        ("COFFEE BAR", "Espresso 2.40", "Croissant 3.10", "Water 1.50", "TOTAL 7.00")
    ):
        draw.text((40, 60 + index * 150), line, fill=(0, 0, 0), font=font)
    return image


def _requests(checkpoint, directory):
    from PIL import Image

    images = {
        "document": _document_image(),
        "receipt": _receipt_image(),
        "paragraph": _paragraph_image(),
        "teaser": Image.open(checkpoint / "assets/HyOCR_1_5_teaser.png").convert("RGB"),
    }
    specs = {}
    for name, image in images.items():
        target = directory / f"{name}.rgb"
        np.asarray(image).tofile(target)
        specs[name] = {"path": str(target), "height": image.height, "width": image.width}
    parse = (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )

    def user(*parts):
        return {"role": "user", "parts": list(parts)}

    requests = [
        {
            "messages": [user({"image": "document"}, {"text": "Extract all text from the image."})],
            "max_new_tokens": 96,
        },
        {"messages": [user({"image": "teaser"}, {"text": parse})], "max_new_tokens": 160},
        {
            "messages": [
                {"role": "system", "parts": [{"text": "You are an OCR assistant."}]},
                user({"image": "receipt"}, {"text": "What is the total amount on the receipt?"}),
            ],
            "max_new_tokens": 48,
        },
        {
            "messages": [
                user(
                    {"image": "document"},
                    {"image": "receipt"},
                    {"text": "Read the text of both images."},
                )
            ],
            "max_new_tokens": 128,
        },
        {
            "messages": [
                user({"image": "document"}, {"text": "Extract all text from the image."}),
                {
                    "role": "assistant",
                    "parts": [{"text": "TensorRT Model Connect\nInvoice No. 4711"}],
                },
                user({"text": "What is the invoice number? Answer with the number only."}),
            ],
            "max_new_tokens": 24,
        },
        {
            "messages": [
                user({"image": "paragraph"}, {"text": "Extract all text from the image."})
            ],
            "max_new_tokens": 256,
        },
    ]
    native = []
    for request in requests:
        messages = [
            {
                "role": message["role"],
                "parts": [
                    {"image": specs[part["image"]]} if "image" in part else part
                    for part in message["parts"]
                ],
            }
            for message in request["messages"]
        ]
        native.append({**request, "messages": messages})
    return requests, native, images


def _reference(checkpoint, requests, images, dtype_name="fp32"):
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]

    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = None
    try:
        model, info = HunYuanVLForConditionalGeneration.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not any(
            info.get(k)
            for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
        ), info
        model = model.to("cuda").eval()
        results = []
        for request in requests:
            messages, media = [], []
            for message in request["messages"]:
                content = []
                for part in message["parts"]:
                    if "image" in part:
                        content.append({"type": "image"})
                        media.append(images[part["image"]])
                    else:
                        content.append({"type": "text", "text": part["text"]})
                if (
                    message["role"] != "user"
                ):  # The template renders these contents as plain strings.
                    content = "".join(part["text"] for part in content)
                messages.append({"role": message["role"], "content": content})
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[prompt], images=media or None, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=False, max_new_tokens=request["max_new_tokens"]
                )
            tokens = output[0, inputs["input_ids"].shape[1] :].cpu().tolist()
            results.append(
                {
                    "text": processor.decode(tokens, skip_special_tokens=True),
                    "token_ids": tokens,
                    "prompt_tokens": int(inputs["input_ids"].shape[1]),
                }
            )
            del inputs, output
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _ned(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = current
    return row[-1] / max(len(a), len(b), 1)


def _common_prefix(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b))


def test_host_preprocessing_matches_reference(request, tmp_path):
    _selected(request.config, {case["name"] for _, case in CASES.values()})
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer

    manifest = next(iter(CASES.values()))[0]
    checkpoint = _checkpoint(manifest)
    probe = _native_tool(Path(os.environ["TRTMC_RUNTIME_ROOT"]), "hunyuan_vl_preprocess_probe")
    from families.hunyuan_vl.config import ImageProcessor, ModelConfig

    config = ModelConfig.from_dir(checkpoint)
    settings = ImageProcessor.from_dir(checkpoint, config)
    metadata = tmp_path / "runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "patch_size": config.patch_size,
                "merge_size": config.merge_size,
                "max_patches": 1 << 30,
                "position_edge": config.position_edge,
                "image": {
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels,
                    "mean": list(settings.mean),
                    "std": list(settings.std),
                    "rescale_factor": settings.rescale_factor,
                },
            }
        )
    )
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    rng = np.random.default_rng(218)
    table = torch.from_numpy(rng.normal(size=(1 + config.position_edge**2, 8)).astype(np.float32))
    # The reference interpolates its position grid on the model device. CUDA evaluates source
    # indices in FP32, as the family does; the CPU kernel uses FP64, which moves the bilinear
    # weights by up to half an FP32 ulp of the source index (about 4e-6 at index 127).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    position_tolerance = 2e-6 if device == "cuda" else 1e-4
    results = []
    sources = [
        Image.open(checkpoint / "assets/HyOCR_1_5_teaser.png").convert("RGB"),
        _document_image(),
    ]
    sources += [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in PROBE_SIZES
    ]
    for index, image in enumerate(sources):
        raw = tmp_path / f"image{index}.rgb"
        np.asarray(image).tofile(raw)
        prefix = tmp_path / f"probe{index}"
        subprocess.run(
            [
                str(probe),
                "image",
                str(raw),
                str(image.height),
                str(image.width),
                str(metadata),
                str(prefix),
            ],
            check=True,
            timeout=600,
        )
        actual = json.loads(Path(f"{prefix}.json").read_text())
        pixels = np.fromfile(f"{prefix}.f32", dtype=np.float32).reshape(
            -1, 3 * config.patch_size**2
        )
        expected = processor.image_processor(images=[image], return_tensors="pt")
        _, rows, cols = expected["image_grid_thw"][0].tolist()
        assert (actual["rows"], actual["cols"]) == (rows, cols)
        # The reference flattens patches row-major; the family groups each 2x2 merge block.
        order = [
            (by + dy) * cols + bx + dx
            for by in range(0, rows, 2)
            for bx in range(0, cols, 2)
            for dy in (0, 1)
            for dx in (0, 1)
        ]
        difference = np.abs(pixels - expected["pixel_values"].numpy()[order]).max()
        # Bilinear position grid: the corner rows and weights reproduce the reference interpolation.
        grid = (
            table[1:].reshape(1, config.position_edge, config.position_edge, -1).permute(0, 3, 1, 2)
        )
        reference = (
            torch.nn.functional.interpolate(
                grid.to(device), size=(rows, cols), mode="bilinear", align_corners=False
            )[0]
            .permute(1, 2, 0)
            .reshape(-1, 8)
            .cpu()
        )
        ids = np.array(actual["position_ids"]).reshape(-1, 4)
        h0, h1, w0, w1 = (
            np.array(actual["position_lambdas"], np.float32).reshape(-1, 4).T[..., None]
        )
        corners = table.numpy()[ids]
        rebuilt = h0 * (w0 * corners[:, 0] + w1 * corners[:, 1]) + h1 * (
            w0 * corners[:, 2] + w1 * corners[:, 3]
        )
        position_error = np.abs(rebuilt - reference.numpy()[order]).max()
        results.append(
            {
                "size": [image.height, image.width],
                "grid": [rows, cols],
                "device": device,
                "pixel_max_abs": float(difference),
                "position_max_abs": float(position_error),
            }
        )
        assert difference == 0 and position_error <= position_tolerance, results[-1]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(list(TOKENIZER_TEXTS)))
    output = tmp_path / "tokens.json"
    subprocess.run(
        [str(probe), "tokens", str(checkpoint / "tokenizer.json"), str(texts), str(output)],
        check=True,
        timeout=600,
    )
    for text, actual in zip(TOKENIZER_TEXTS, json.loads(output.read_text())):
        expected = tokenizer.encode(text, add_special_tokens=False)
        assert actual["ids"] == expected, text
        assert actual["decoded"] == tokenizer.decode(expected, skip_special_tokens=True), text
    record_evidence("preprocessing", {"images": results, "tokenizer_texts": len(TOKENIZER_TEXTS)})


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path):
    manifest, case = CASES[case_name]
    _selected(request.config, {manifest["name"], case["name"]})
    assert case["reference_backend"] == "hf_transformers"
    assert case["reference_precision"] == "fp32"
    import torch

    assert torch.cuda.is_available(), "selected HunYuanVL E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    runner = _native_tool(runtime, "hunyuan_vl_inference_test")
    checkpoint = _checkpoint(manifest)
    requests, native, images = _requests(checkpoint, tmp_path)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=checkpoint,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    source, destination = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(native))
    command = [str(runner), str(bundle), str(runtime), str(source), str(destination)]
    env = dict(os.environ)
    library_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[library_path] = str(runtime) + os.pathsep + env.get(library_path, "")
    with evidence_stage("native"):
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=3600)
        record_evidence("commands", {"argv": command, "returncode": completed.returncode})
        record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
        assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(destination.read_text())
    with evidence_stage("reference"):
        expected = _reference(checkpoint, requests, images)
        record_evidence(
            "reference", {"backend": "hf_transformers", "precision": "fp32", "results": expected}
        )
        baseline = None
        if manifest["precision"] == "bf16":
            baseline = _reference(checkpoint, requests, images, "bf16")
            record_evidence(
                "reference_bf16",
                {"backend": "hf_transformers", "precision": "bf16", "results": baseline},
            )
    assert len(actual) == len(expected) == len(requests)
    sidecar = ROOT / "thresholds" / f"{case_name}.json"
    # Without a sidecar every reference token must be reproduced.
    limits = json.loads(sidecar.read_text())["threshold_overrides"] if sidecar.is_file() else {}
    rows = []
    for index, (a, b, item) in enumerate(zip(actual, expected, requests)):
        assert 0 < len(a["token_ids"]) <= item["max_new_tokens"]
        reference = b["token_ids"]
        required = len(reference)
        row = {
            "actual": a,
            "reference": b,
            "normalized_text_edit_distance": _ned(a["text"], b["text"]),
            "matching_prefix_tokens": _common_prefix(a["token_ids"], reference),
        }
        if baseline is not None:
            stable = _common_prefix(baseline[index]["token_ids"], reference)
            if baseline[index]["token_ids"] != reference:
                required = max(0, stable - limits["bf16_baseline_prefix_slack_tokens"])
            row["reference_bf16_matching_prefix_tokens"] = stable
        row["required_prefix_tokens"] = required
        rows.append(row)
    record_evidence("metrics", {"cases": rows, "limits": limits})
    if baseline is not None:
        stable = sum(
            row["required_prefix_tokens"] == len(row["reference"]["token_ids"]) for row in rows
        )
        assert stable >= limits["bf16_min_stable_requests"], rows
    for row in rows:
        if row["required_prefix_tokens"] == len(row["reference"]["token_ids"]):
            assert row["actual"]["token_ids"] == row["reference"]["token_ids"], rows
        assert row["matching_prefix_tokens"] >= row["required_prefix_tokens"], rows

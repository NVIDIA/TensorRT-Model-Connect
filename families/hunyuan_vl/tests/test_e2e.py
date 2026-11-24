# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HunyuanOCR builds and native public-task runs against the Transformers reference.

``test_host_preprocessing_matches_reference`` compares the family's C++ tokenizer and image
preprocessing with the reference processor. ``test_e2e`` builds the bundle, runs the public
``images_text_to_text`` task natively and compares greedy token ids with the FP32 reference.

An FP32 bundle must reproduce every reference token. A BF16 bundle is held to the FP32 reference
with the official model's own BF16 run, made in the same test on the same GPU, as a guide. Where
that run equals the FP32 output, the bundle must equal it too, and at least
``bf16_min_stable_requests`` requests must be of this kind. Elsewhere BF16 rounding flips a
near-tied token somewhere, and where it does depends on the GPU's BF16 kernels, so the bundle may
leave the FP32 output only at a near-tie: at its first departure, the FP32 logit of the reference
token may exceed the FP32 logit of the bundle's token by at most ``bf16_departure_max_logit_gap``.
Greedy continuations after a departure are not comparable token by token, so no bound is placed
on them; the FP32 case checks the same requests token for token.
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


def _chosen(models, cases, names):
    """Whether the model and testcase filters both choose the case named by ``names``."""
    return (not models or bool(models & {FAMILY, *names})) and (not cases or bool(cases & names))


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
    family = {FAMILY, *CASES, *(manifest["name"] for manifest, _ in CASES.values())}
    inference = [{manifest["name"], case["name"]} for manifest, case in CASES.values()]
    # Naming this family must run one of its inference cases, not only the host-side checks.
    if models & family and not any(_chosen(models, cases, case) for case in inference):
        pytest.fail(
            f"--e2e-model {sorted(models)} with --e2e-testcase {sorted(cases)} selects none of "
            f"{FAMILY}'s inference cases {sorted(CASES)}"
        )
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
            # Printed on the page: both answers must state it.
            "answer": "4711",
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
        native.append({**{k: v for k, v in request.items() if k != "answer"}, "messages": messages})
    return requests, native, images


def _reference(checkpoint, requests, images, dtype_name="fp32", candidates=None):
    """Greedy reference outputs. With ``candidates`` (one token list per request), each result also
    records where the candidate first leaves the reference and the reference's logit gap there."""
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
        for index, request in enumerate(requests):
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
                    **inputs,
                    do_sample=False,
                    max_new_tokens=request["max_new_tokens"],
                    return_dict_in_generate=True,
                    output_logits=candidates is not None,
                )
            tokens = output.sequences[0, inputs["input_ids"].shape[1] :].cpu().tolist()
            result = {
                "text": processor.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens,
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
            }
            if candidates is not None:
                result["departure"] = _departure(tokens, candidates[index], output.logits)
            results.append(result)
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


def _departure(reference, candidate, logits):
    """Where ``candidate`` first leaves the greedy ``reference``, and the reference's logit gap
    there: its chosen token's logit minus the candidate token's (``logits[i]`` chose token ``i``).
    ``None`` when the candidate equals the reference."""
    if candidate == reference:
        return None
    index = _common_prefix(reference, candidate)
    if index >= min(len(reference), len(candidate)):
        # One output stops early without a differing token: no logit compares the two.
        return {"index": index, "logit_gap": None}
    chosen, other = reference[index], candidate[index]
    step = logits[index][0].float()
    return {
        "index": index,
        "reference_token": chosen,
        "candidate_token": other,
        "logit_gap": float(step[chosen] - step[other]),
    }


def _check(name, label, actual, operator, expected, scope="independent_reference"):
    passed = {"==": actual == expected, "<=": actual <= expected, ">=": actual >= expected}
    return {
        "name": name,
        "label": label,
        "scope": scope,
        "actual": actual,
        "operator": operator,
        "expected": expected,
        "passed": passed[operator],
    }


# Calibration of thresholds/hunyuanocr-bf16.json on one B300 with the six requests below:
# - bf16_departure_max_logit_gap = 1.0. The first departures of the BF16 bundle and of the official
#   BF16 run from the FP32 output happen at FP32 logit gaps of 0.14 and 0.19. Teacher-forced along
#   the FP32 tokens, BF16 moves the FP32 top-1 / top-2 gap by 0.093 (median), 0.41 (95th
#   percentile), 0.71 (99th) and 1.96 (largest) over 313 steps; 51 of those steps have a gap of at
#   most 1.0. A bundle decoding with a wrong repetition penalty (1.3) departs at gaps of 3.42, 0.86
#   and 0.26 and fails.
# - bf16_min_stable_requests = 3: the official BF16 run equals the FP32 output on four requests.
def _verdict(actual, reference, baseline, departure, limits):
    """Judge one request (module docstring); ``checks`` use the evidence report's comparison
    schema. ``baseline`` is the official BF16 run's tokens, or ``None`` for an FP32 bundle, which
    must reproduce every reference token."""
    exact = [
        _check(
            "agreeing_prefix_tokens",
            "Tokens equal to the FP32 reference before the first difference",
            _common_prefix(actual, reference),
            "==",
            len(reference),
        ),
        _check("output_tokens", "Output length in tokens", len(actual), "==", len(reference)),
    ]
    if baseline is None:
        verdict = {"rule": "exact", "checks": exact}
    elif baseline == reference:
        verdict = {
            "rule": "exact: the official BF16 run equals FP32",
            "bf16_stable": True,
            "checks": exact,
        }
    else:
        bound = limits["bf16_departure_max_logit_gap"]
        gap = None if departure is None else departure["logit_gap"]
        if actual == reference:
            checks = exact
        elif gap is None:
            checks = [
                _check(
                    "departure_compared",
                    "A logit compares the first departure",
                    False,
                    "==",
                    True,
                    "contract",
                )
            ]
        else:
            checks = [
                _check(
                    "first_departure_logit_gap",
                    "FP32 logit gap at the first departure from the FP32 reference",
                    gap,
                    "<=",
                    bound,
                )
            ]
        verdict = {
            "rule": f"exact, or the first departure at an FP32 logit gap <= {bound}",
            "bf16_stable": False,
            "checks": checks,
        }
    verdict["passed"] = all(check["passed"] for check in verdict["checks"])
    return verdict


def _end_tokens(checkpoint):
    """The end-of-sequence ids generation stops at, from generation_config.json."""
    eos = json.loads((checkpoint / "generation_config.json").read_text())["eos_token_id"]
    return {eos} if isinstance(eos, int) else set(eos)


def _stopping_errors(actual, requests, eos):
    """Requests whose native output does not stop as greedy generation does: at its first end
    token, or after its whole token budget without one."""
    errors = []
    for index, (a, item) in enumerate(zip(actual, requests)):
        ids = a["token_ids"]
        ends = [position for position, token in enumerate(ids) if token in eos]
        if ends != [len(ids) - 1] and (ends or len(ids) != item["max_new_tokens"]):
            errors.append(index)
    return errors


def _wrong_answers(actual, reference, requests):
    """Requests with a printed answer that the native or the reference text does not state
    exactly (surrounding whitespace aside)."""
    return [
        index
        for index, (a, b, item) in enumerate(zip(actual, reference, requests))
        if "answer" in item
        and (a["text"].strip() != item["answer"] or b["text"].strip() != item["answer"])
    ]


def _native_text_mismatches(actual, decode):
    """Requests whose native text differs from the reference tokenizer's decoding of the native
    token ids (``decode``): the runtime must decode what it generated."""
    return [index for index, item in enumerate(actual) if item["text"] != decode(item["token_ids"])]


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
        # Shapes first: a short native result must not broadcast into a passing difference.
        assert pixels.shape == expected["pixel_values"].shape == (rows * cols, pixels.shape[1])
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
        assert ids.shape == h0.shape[:1] + (4,) == (rows * cols, 4)
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
    tokenized = json.loads(output.read_text())
    assert len(tokenized) == len(TOKENIZER_TEXTS), tokenized
    for text, actual in zip(TOKENIZER_TEXTS, tokenized):
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
        expected = _reference(
            checkpoint, requests, images, candidates=[a["token_ids"] for a in actual]
        )
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    mismatched_text = _native_text_mismatches(actual, decode)
    wrong_answers = _wrong_answers(actual, expected, requests)
    stopping_errors = _stopping_errors(actual, requests, _end_tokens(checkpoint))
    sidecar = ROOT / "thresholds" / f"{case_name}.json"
    # Without a sidecar every reference token must be reproduced.
    limits = json.loads(sidecar.read_text())["threshold_overrides"] if sidecar.is_file() else {}
    previews = {}
    for name, image in images.items():
        previews[f"image_{name}"] = tmp_path / f"{name}.png"
        image.save(previews[f"image_{name}"])
    prompts = [
        " | ".join(
            f"{message['role']}: "
            + " ".join(part.get("text", f"<{part.get('image')}>") for part in message["parts"])
            for message in item["messages"]
        )
        for item in requests
    ]
    record_evidence(
        "inputs", {"prompt": "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(prompts)), **previews}
    )
    record_evidence(
        "native",
        {
            "text": "\n\n".join(f"[{i + 1}] {a['text']}" for i, a in enumerate(actual)),
            "token_ids": [a["token_ids"] for a in actual],
        },
    )
    record_evidence(
        "reference",
        {
            "reference_text": "\n\n".join(f"[{i + 1}] {b['text']}" for i, b in enumerate(expected)),
            "reference_ids": [b["token_ids"] for b in expected],
        },
    )
    rows = []
    for index, (a, b, item) in enumerate(zip(actual, expected, requests)):
        assert 0 < len(a["token_ids"]) <= item["max_new_tokens"]
        reference = b["token_ids"]
        official = None if baseline is None else baseline[index]["token_ids"]
        verdict = _verdict(a["token_ids"], reference, official, b["departure"], limits)
        text = _check(
            "native_text_decodes_native_tokens",
            "Native text equals the reference tokenizer's decoding of the native tokens",
            index not in mismatched_text,
            "==",
            True,
            "contract",
        )
        native_file, reference_file = (
            tmp_path / f"native_{index}.npy",
            tmp_path / f"reference_{index}.npy",
        )
        np.save(native_file, np.asarray(a["token_ids"], dtype=np.int64))
        np.save(reference_file, np.asarray(reference, dtype=np.int64))
        stops = _check(
            "stops_at_end_token_or_budget",
            "Native output ends at its first end token, or after its whole token budget",
            index not in stopping_errors,
            "==",
            True,
            "contract",
        )
        checks = [*verdict["checks"], text, stops]
        if "answer" in item:
            checks.append(
                _check(
                    "printed_answer",
                    f"Native and reference answers state the printed {item['answer']}",
                    index not in wrong_answers,
                    "==",
                    True,
                    "contract",
                )
            )
        record_evidence(
            "reference_comparison",
            {
                "label": f"request {index + 1}: {verdict['rule']}",
                "scope": "independent_reference",
                "enforced": True,
                "native": native_file,
                "reference": reference_file,
                "checks": checks,
            },
        )
        row = {
            "actual": a,
            "reference": {key: value for key, value in b.items() if key != "departure"},
            "normalized_text_edit_distance": _ned(a["text"], b["text"]),
            "matching_prefix_tokens": _common_prefix(a["token_ids"], reference),
            "departure": b["departure"],
            **verdict,
        }
        if official is not None:
            row["reference_bf16_matching_prefix_tokens"] = _common_prefix(official, reference)
        rows.append(row)
    record_evidence(
        "metrics",
        {
            "cases": rows,
            "limits": limits,
            "native_text_mismatches": mismatched_text,
            "wrong_answers": wrong_answers,
            "stopping_errors": stopping_errors,
        },
    )
    assert not stopping_errors, [actual[i]["token_ids"] for i in stopping_errors]
    assert not wrong_answers, [(actual[i]["text"], expected[i]["text"]) for i in wrong_answers]
    assert not mismatched_text, [
        (actual[i]["text"], decode(actual[i]["token_ids"])) for i in mismatched_text
    ]
    if baseline is not None:
        stable = sum(row["bf16_stable"] for row in rows)
        assert stable >= limits["bf16_min_stable_requests"], rows
    failed = [index for index, row in enumerate(rows) if not row["passed"]]
    assert not failed, [rows[index] for index in failed]

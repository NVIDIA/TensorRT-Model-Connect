# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for GLM-OCR.

Each case builds its manifest's bundle, runs ``trtmc run --image ... --prompt ...`` and
compares the generated tokens with Transformers' ``GlmOcrForConditionalGeneration`` run on
the same GPU and the same pixels (the test decodes the image once with Pillow and hands the
same lossless PNG to both sides):

* fp32: the native tokens and text equal the official FP32 generation (TF32 off).
* bf16: teacher-forced against the official FP32 model, every native token scores within
  ``_BF16_LOGIT_TOLERANCE`` of the FP32 maximum at its step (see the measurement below).

The host-parity tests compare the native host steps, run through the host probe of the
native ``test_glm_ocr_runtime`` (the family's CTest executable), with the processor and the
tokenizers library: image preprocessing to the bit, tokenization, the pre-tokenizer on every
Unicode scalar and on random text, chat-template rendering and the multimodal rotary
positions.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "glm_ocr"
TASKS = frozenset({"images_text_to_text"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# The bf16 gate. Measured on this checkpoint (B300, TensorRT 11.4, Transformers 5.22.2), with
# the 448x924 crop of the page as a third input; "deficit" is the FP32 maximum logit minus
# the FP32 logit of the token chosen, teacher-forced along the generated sequence:
#
# ==========  =================  ===============  ==========  ==================  ==========
# input       FP32 min top-2 gap  official BF16 vs  official    native BF16         defective
#                                 FP32 max |logit|  BF16 worst  worst deficit       BF16 worst
#                                 difference        deficit     (tokens = FP32?)    deficit
# ==========  =================  ===============  ==========  ==================  ==========
# line        1.030               0.374            0           0 (yes)             0
# page crop   0.046               2.003            0           0.046 (no)          26.1
# page        0.268               1.293            0           0 (yes)             27.9
# ==========  =================  ===============  ==========  ==================  ==========
#
# A correct bf16 bundle may pick the second-best token at a near tie (the crop flips at its
# 0.046 gap) and then follows its own path; each of its tokens stays within the logit noise
# the official BF16 pipeline itself shows (up to 2.0). The defective build, whose rotary
# embedding pairs dimensions (i, i + d/2) instead of (2i, 2i + 1), loops on the page and
# falls 26-28 logits short. The single line does not expose that defect, so the page case
# is a premerge case too.
_BF16_LOGIT_TOLERANCE = 2.0


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
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names
    if not enabled:
        parameters = [
            pytest.param(name, marks=pytest.mark.skip(
                reason="direct E2E requires one of the three explicit E2E selectors"))
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
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True)
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
    return binary, runtime_root


def _probe() -> Path:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    return _required_path(str(build_dir / "test_glm_ocr_runtime"), "the test_glm_ocr_runtime target")


def _bundle(model_dir: Path, manifest: dict, directory: Path) -> Path:
    bundle = directory / manifest["bundle"]
    if not bundle.is_file():
        build(BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        ))
    return bundle


def _image(case: dict, directory: Path) -> Path:
    """The case image decoded once with Pillow and saved losslessly for both sides."""
    from PIL import Image

    source = TEST_ROOT / case["image"]
    assert source.is_file(), f"{FAMILY} E2E asset does not exist: {source}"
    image = Image.open(source).convert("RGB")
    path = directory / f"{case['name']}.png"
    image.save(path, format="PNG")
    return path


def _run_native(binary: Path, runtime_root: Path, bundle: Path, image: Path, case: dict) -> dict:
    invocation = [
        str(binary), "run", str(bundle), "--runtime-root", str(runtime_root),
        "--image", str(image), "--prompt", case["prompt"],
        "--max-new-tokens", str(int(case["max_new_tokens"])),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    completed = subprocess.run(invocation, check=True, capture_output=True, text=True, env=env,
                               timeout=1800)
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native run returned no JSON: {completed.stdout[-1000:]}"
    return payloads[-1]


@lru_cache(maxsize=2)
def _official(model_dir: str, dtype_name: str):
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_dir, dtype=getattr(torch, dtype_name)).to("cuda").eval()
    return processor, model


def _official_inputs(processor, image: Path, prompt: str):
    from PIL import Image

    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image)}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt").to("cuda")


def _official_generation(model_dir: Path, image: Path, case: dict, dtype_name: str):
    import torch

    processor, model = _official(str(model_dir), dtype_name)
    inputs = _official_inputs(processor, image, case["prompt"])
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=int(case["max_new_tokens"]),
                                do_sample=False)
    ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    return ids, processor.decode(ids, skip_special_tokens=True)


def _teacher_forced_deficits(model_dir: Path, image: Path, case: dict, tokens: list[int],
                             dtype_name: str) -> list[float]:
    """Per step, the official model's maximum logit minus the logit of ``tokens[step]``."""
    import torch

    processor, model = _official(str(model_dir), dtype_name)
    inputs = _official_inputs(processor, image, case["prompt"])
    prompt_length = inputs["input_ids"].shape[1]
    extra = torch.tensor([tokens[:-1]], dtype=inputs["input_ids"].dtype, device="cuda")
    inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
    inputs["mm_token_type_ids"] = torch.cat(
        [inputs["mm_token_type_ids"], torch.zeros_like(extra, dtype=inputs["mm_token_type_ids"].dtype)],
        dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    with torch.inference_mode():
        logits = model(**inputs).logits[0, prompt_length - 1:].float()
    chosen = logits.gather(1, torch.tensor(tokens, device="cuda")[:, None])[:, 0]
    return (logits.max(dim=1).values - chosen).tolist()


def test_official_checkpoint_e2e(case_name: str, tmp_path_factory) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    bundles = tmp_path_factory.getbasetemp() / "glm_ocr_bundles"
    bundles.mkdir(exist_ok=True)
    bundle = _bundle(model_dir, manifest, bundles)
    image = _image(case, tmp_path_factory.mktemp(case_name))
    payload = _run_native(binary, runtime_root, bundle, image, case)
    tokens = [int(token) for token in payload["token_ids"]]
    assert 0 < len(tokens) <= int(case["max_new_tokens"])

    if manifest["precision"] == "fp32":
        reference, text = _official_generation(model_dir, image, case, "float32")
        first = next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b), None)
        assert tokens == reference, (
            f"{FAMILY} fp32 tokens differ from the official FP32 run at step {first}: "
            f"native {tokens[first or 0:][:8]} official {reference[first or 0:][:8]}")
        assert payload["text"] == text
        print(f"{FAMILY} fp32 {case_name}: {len(tokens)} tokens identical to the official model")
        return

    assert manifest["precision"] == "bf16"
    deficits = _teacher_forced_deficits(model_dir, image, case, tokens, "float32")
    worst = max(deficits)
    assert worst <= _BF16_LOGIT_TOLERANCE, (
        f"{FAMILY} bf16 token at step {deficits.index(worst)} is {worst:.3f} logits below the "
        f"official FP32 maximum")
    # The bf16 tokens may differ from FP32's; the native text must be the reference decoding of
    # the native tokens.
    processor, _ = _official(str(model_dir), "float32")
    assert payload["text"] == processor.decode(tokens, skip_special_tokens=True)
    reference, text = _official_generation(model_dir, image, case, "float32")
    print(f"{FAMILY} bf16 {case_name}: {len(tokens)} tokens, largest FP32 deficit {worst:.4f}, "
          f"identical to the FP32 tokens: {tokens == reference}")


# --------------------------------------------------------------------------------------
# Host parity: the native host steps against the processor and the tokenizer (CPU only).
# --------------------------------------------------------------------------------------

_TOKENIZER_TEXTS = [
    "Text Recognition:", "Formula Recognition:", "Table Recognition:",
    "请按下列JSON格式输出图中信息:\n{\n    \"id_number\": \"\",\n    \"last_name\": \"\"\n}",
    "  leading spaces and trailing   ", "tabs\tand\r\nwindows\nnewlines\n\n\n",
    "Ünïcödé, ŝtrïng — “quotes” … ellipsis", "数学公式：$E = mc^2$，以及 x²+y²=1。",
    "日本語のテキストと한국어 텍스트", "def f(x):\n    return x ** 2  # comment\n",
    "it's we'RE they'll I'M DON'T 'll", "$12.50 (approx.) 1234567 0.001",
    "emoji 🙂👍🏽 and symbols ©®™ ±×÷", "<|image|> inside text and <|user|> too",
    "<think>reasoning</think> <|begin_of_box|>box<|end_of_box|>", "ſ's x'ſun ǅungla",
    " non-breaking em　ideographic", "",
]

# Image grids are patch rows x columns; each one holds at least the processor's minimum
# pixel count, so an image of exactly that many patches keeps its size.
_CONVERSATIONS = [
    [{"role": "user", "parts": [{"image": [4, 60]}, {"text": "Text Recognition:"}]}],
    [{"role": "user", "parts": [{"text": "Table Recognition:"}, {"image": [8, 12]}]}],
    [{"role": "user", "parts": [{"image": [6, 6]}, {"image": [4, 10]}, {"text": "Compare."}]}],
    [{"role": "system", "parts": [{"text": "You read documents."}]},
     {"role": "user", "parts": [{"image": [10, 6]}, {"text": "  Formula Recognition:\n"}]}],
    [{"role": "user", "parts": [{"image": [8, 4]}, {"text": "Text Recognition:"}]},
     {"role": "assistant", "parts": [{"text": "  ALPHA 123 \n"}]},
     {"role": "user", "parts": [{"text": "请按下列JSON格式输出图中信息:"}, {"image": [6, 8]}]}],
]


def _host_selected(request) -> None:
    names, enabled = _selected_cases(request.config)
    if not enabled or not names:
        pytest.skip("host parity requires an explicit E2E selector that selects a glm_ocr case")


def _host_model_dir() -> Path:
    return _model_dir(json.loads((MANIFEST_ROOT / "glm-ocr-bf16.json").read_text(encoding="utf-8")))


def _run_probe(*arguments: str) -> str:
    completed = subprocess.run([str(_probe()), *arguments], check=True, capture_output=True,
                               text=True, timeout=600)
    return completed.stdout


def _runtime_json(model_dir: Path, directory: Path) -> Path:
    from families.glm_ocr.checkpoint import load_config
    from families.glm_ocr.model import _runtime_contract

    config = load_config(model_dir)
    path = directory / "runtime.json"
    path.write_text(json.dumps(_runtime_contract(config, "bf16", 16384, config.max_image_patches)))
    return path


def _synthetic_images(directory: Path) -> list[Path]:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(191)
    paths = []
    for height, width in ((37, 900), (290, 400), (700, 1000), (2100, 1500), (3000, 2500),
                          (30, 30), (1000, 31)):
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        # Smooth half of each image so that the antialiasing filter sees both regimes.
        pixels[: height // 2] = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
        path = directory / f"synthetic_{height}x{width}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


def test_native_preprocessing_matches_the_processor(request, tmp_path) -> None:
    _host_selected(request)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GlmOcrConfig
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionRotaryEmbedding
    from transformers.vision_utils import get_vision_position_ids

    model_dir = _host_model_dir()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    vision = GlmOcrConfig.from_pretrained(str(model_dir)).vision_config
    rotary = GlmOcrVisionRotaryEmbedding(vision)
    runtime_json = _runtime_json(model_dir, tmp_path)
    page = {"name": "page", "image": "data/orc_test_img.jpeg"}
    line = {"name": "line", "image": "data/orc_test_img_line.png"}
    for path in [_image(page, tmp_path), _image(line, tmp_path), *_synthetic_images(tmp_path)]:
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        raw = tmp_path / f"{path.stem}.rgb8"
        raw.write_bytes(pixels.tobytes())
        prefix = tmp_path / path.stem
        grid = json.loads(_run_probe("preprocess", str(runtime_json), str(raw),
                                     str(pixels.shape[0]), str(pixels.shape[1]), str(prefix)))
        expected = processor.image_processor(images=Image.open(path), return_tensors="pt")
        thw = expected["image_grid_thw"][0].tolist()
        assert [grid["grid_h"], grid["grid_w"]] == thw[1:], path.name
        native = np.fromfile(f"{prefix}.patches.f32", dtype=np.float32)
        reference = expected["pixel_values"].numpy().reshape(-1)
        assert np.array_equal(native, reference), (
            f"{path.name}: preprocessing differs by {np.abs(native - reference).max()}")
        positions = get_vision_position_ids(expected["image_grid_thw"], vision.spatial_merge_size)
        cos, sin = rotary(torch.zeros(1), positions)
        half = vision.hidden_size // vision.num_heads // 2
        for suffix, table in (("cos", cos), ("sin", sin)):
            native_table = np.fromfile(f"{prefix}.{suffix}.f32", dtype=np.float32)
            assert np.abs(native_table - table[:, :half].numpy().reshape(-1)).max() <= 1e-6


def test_native_tokenizer_matches_tokenizers(request, tmp_path) -> None:
    _host_selected(request)
    from transformers import AutoTokenizer

    model_dir = _host_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps(_TOKENIZER_TEXTS, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "tokens.json"
    _run_probe("tokenize", str(model_dir / "tokenizer.json"), str(texts), str(out))
    results = json.loads(out.read_text(encoding="utf-8"))
    for text, native in zip(_TOKENIZER_TEXTS, results, strict=True):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert native["ids"] == ids, text
        assert native["decoded"] == tokenizer.decode(ids, skip_special_tokens=True), text
        assert native["decoded_with_special"] == tokenizer.decode(ids), text


def _split_pre_tokenizer(model_dir: Path):
    """The tokenizers library's Split pre-tokenizer of the checkpoint's tokenizer.json."""
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import Split

    definition = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    steps = definition["pre_tokenizer"]
    splits = [step for step in steps.get("pretokenizers", [steps]) if step["type"] == "Split"]
    assert len(splits) == 1
    return Split(Regex(splits[0]["pattern"]["Regex"]), behavior=splits[0]["behavior"].lower(),
                 invert=splits[0]["invert"])


def _assert_pre_tokenizer_parity(model_dir: Path, texts: list[str], directory: Path) -> None:
    split = _split_pre_tokenizer(model_dir)
    path = directory / "pretokenize.json"
    path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    out = directory / "pieces.json"
    _run_probe("pretokenize", str(path), str(out))
    native = json.loads(out.read_text(encoding="utf-8"))
    mismatches = []
    for text, lengths in zip(texts, native, strict=True):
        expected = [len(piece.encode("utf-8")) for piece, _ in split.pre_tokenize_str(text)]
        if lengths != expected:
            mismatches.append(text)
    assert not mismatches, (
        f"{len(mismatches)} pre-tokenizer mismatches, first: "
        f"{[hex(ord(char)) for char in mismatches[0]]}")


def test_native_pre_tokenizer_matches_tokenizers_on_every_scalar(request, tmp_path) -> None:
    """Letter, number and white-space classes of every Unicode scalar, as the regex sees them:
    "a" + c is one piece only for a letter c, "1" + c only for a number, and "!" + c for c
    neither a letter, a number nor white space (or for a line break, which [\\r\\n]* takes)."""
    _host_selected(request)
    scalars = [chr(code) for code in range(0x110000) if not 0xD800 <= code <= 0xDFFF]
    texts = [prefix + char for char in scalars for prefix in ("a", "1", "!")]
    _assert_pre_tokenizer_parity(_host_model_dir(), texts, tmp_path)


def test_native_pre_tokenizer_matches_tokenizers_on_random_text(request, tmp_path) -> None:
    _host_selected(request)
    import random

    alphabet = [
        *"abeZdlmrstv'", "\u2019", *"0123456789", "\u0663", "\u216b", "\u00b2", "\u00bd",
        *" \t\n\r\x0b\x0c", "\x85", "\xa0", "\u1680", "\u2003", "\u2028", "\u2029", "\u202f",
        "\u3000", "\u200b", "\u200d", "\u180e", *"!.,-($_", "\u00a9", "\u2014", "\u2026",
        "\u4e2d", "\u00e9", "\u0436", "\u017f", "\u212a", "\u0130", "\u00df", "\ufb05", "\u0301",
        "\ufe0f", "\U0001f642", "\ud55c", "\u1100", "\u0e01", "\u0e31", "\u3042", "\u30fc",
    ]
    generator = random.Random(191)
    texts = ["".join(generator.choice(alphabet) for _ in range(generator.randint(1, 16)))
             for _ in range(50000)]
    _assert_pre_tokenizer_parity(_host_model_dir(), texts, tmp_path)


def test_native_prompt_matches_the_chat_template(request, tmp_path) -> None:
    _host_selected(request)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GlmOcrConfig
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrModel

    model_dir = _host_model_dir()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    config = GlmOcrConfig.from_pretrained(str(model_dir))
    with torch.device("meta"):
        model = GlmOcrModel(config)
    conversations = tmp_path / "conversations.json"
    conversations.write_text(json.dumps([{"messages": c} for c in _CONVERSATIONS],
                                        ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "prompts.json"
    _run_probe("prompt", str(model_dir / "tokenizer.json"), str(conversations), str(out))
    results = json.loads(out.read_text(encoding="utf-8"))
    patch = config.vision_config.patch_size
    for conversation, native in zip(_CONVERSATIONS, results, strict=True):
        messages = []
        for message in conversation:
            content = []
            for part in message["parts"]:
                if "text" in part:
                    content.append({"type": "text", "text": part["text"]})
                else:
                    rows, cols = part["image"]
                    # An image of exactly rows x cols patches keeps its size in the processor.
                    content.append({"type": "image", "image": Image.new(
                        "RGB", (cols * patch, rows * patch), (200, 180, 160))})
            messages.append({"role": message["role"], "content": content})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert native["text"] == text
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        grids = [part["image"] for message in conversation for part in message["parts"]
                 if "image" in part]
        assert inputs["image_grid_thw"][:, 1:].tolist() == grids
        assert native["ids"] == inputs["input_ids"][0].tolist()
        positions, _ = model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                            image_grid_thw=inputs["image_grid_thw"])
        assert native["positions"] == positions[:, 0, :].T.tolist()
        assert native["next_position"] == int(positions.max()) + 1

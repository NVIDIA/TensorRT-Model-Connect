# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-pipeline E2E for LayoutLM document QA.

The reference is the model card's own usage: transformers'
DocumentQuestionAnsweringPipeline on the document image, which runs Tesseract
through pytesseract, tokenizes question and words into overlapping windows,
runs LayoutLMForQuestionAnswering in float32 and decodes the best span.

Each testcase names one document image and the questions asked about it. Per
question the gate is:

* the answer: the native text equals the pipeline's answer, and the native
  token ids equal the input ids of the reference's winning span. Both depend
  on the OCR words, the tokenization, the window layout and the span decoding,
  not only on the logits;
* the logits: the bundle's network, built in this process at the case
  precision, scores the reference's own window encodings within
  ``_LOGIT_TOLERANCE`` of the reference model. The same engine given all-zero
  boxes (layout ignored) must fail that tolerance by a wide margin, so the gate
  sees the 2-D position embeddings;
* the host steps on the real vocabulary: the family's C++ checker reruns with
  the checkpoint's tokenizer.json on every OCR word of the document and on
  every reference window (ids, word ids, boxes).

The public C (float32 pixels) and C++ (uint8 pixels) SDK consumers must return
the command line's answer for the first question.

Tesseract 5 must be installed with its eng language data: the reference runs
the ``tesseract`` command through pytesseract, the family runtime loads
``libtesseract.so.5``, and the two must be the same version so that both read
the same words. The runtime gives the library the page's RGB pixels where
pytesseract writes a PNG for the command; their TSVs are identical.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "layoutlm"
TASKS = frozenset({"images_text_to_text"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# Largest |native - reference| over the start and end logits of every window.
#
# Measured on the 16 windows of the fp32 cases (B300, TensorRT 11.4), with one
# rule broken at a time in an otherwise correct fp32 engine:
#
# =========================================  =========
# rule broken                                max abs
# =========================================  =========
# (correct build)                            3.9e-05
# LayerNorm eps 1e-12 (BERT's) not 1e-5      5.0e-03
# tanh GELU instead of erf                   7.9e-02
# positions start at 2 (RoBERTa's offset)    5.4e+00
# token-type row left out                    7.3e+00
# height and width tables swapped            8.2e+00
# all-zero boxes (layout ignored)            1.9e+01
# =========================================  =========
#
# 1e-3 sits 25x above the correct fp32 build and 5x below the weakest break.
# A correct fp16 engine is 1.9e-2 to 4.2e-2 off (the reference model's own
# fp16 run is up to 3.0e-2 off on the same windows), which covers the two
# weakest breaks; they are gated by the fp32 cases, whose engines come from the
# same graph code. 1e-1 still rejects every structural break by 50x or more.
_LOGIT_TOLERANCE = {"fp32": 1e-3, "fp16": 1e-1}
# A layout-blind engine must miss the reference by at least this multiple of the tolerance.
_BLIND_MARGIN = 10.0


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
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
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
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"),
                                     local_files_only=True)
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native_build


def _tesseract_version() -> str:
    """The OCR engine both sides use; the command and the library must agree."""
    import ctypes

    try:
        import pytesseract

        command = str(pytesseract.get_tesseract_version())
        library = ctypes.CDLL("libtesseract.so.5")
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires pytesseract, the tesseract command and "
            "libtesseract.so.5 (Tesseract 5 with the eng language data)") from error
    library.TessVersion.restype = ctypes.c_char_p
    version = library.TessVersion().decode()
    assert version == command, (
        f"tesseract command {command} and libtesseract.so.5 {version} differ; the reference and "
        "the native runtime would read different words")
    return version


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return env


def _last_json(stdout: str) -> dict:
    payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"no JSON in output: {stdout[-1000:]}"
    return payloads[-1]


def _native_answer(binary: Path, runtime_root: Path, bundle: Path, image: Path,
                   question: str) -> dict:
    completed = subprocess.run(
        [str(binary), "run", str(bundle), "--runtime-root", str(runtime_root), "--image", str(image),
         "--prompt", question],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=600)
    return _last_json(completed.stdout)


def _reference(model_dir: Path, image: Path, questions: list[str], window: int) -> list[dict]:
    """The official pipeline's answers plus the windows and logits behind them."""
    import torch
    from transformers import pipeline
    from transformers.pipelines.question_answering import select_starts_ends

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    qa = pipeline("document-question-answering", model=str(model_dir), tokenizer=str(model_dir),
                  device="cuda", dtype=torch.float32)
    results = []
    for question in questions:
        outputs, windows = [], []
        for model_inputs in qa.preprocess({"image": str(image), "question": question},
                                          max_seq_len=window):
            record = {
                "input_ids": model_inputs["input_ids"][0].tolist(),
                "bbox": model_inputs["bbox"][0].tolist(),
                "word_ids": [-1 if value is None else value for value in model_inputs["word_ids"]],
                "p_mask": [int(value) for value in model_inputs["p_mask"]],
            }
            output = qa.forward(dict(model_inputs))
            record["start_logits"] = output["start_logits"][0].float().tolist()
            record["end_logits"] = output["end_logits"][0].float().tolist()
            outputs.append(output)
            windows.append(record)
        answers = qa.postprocess(outputs, top_k=1)
        assert qa(image=str(image), question=question, max_seq_len=window) == answers
        # The winning span in token positions: the first best-scoring window.
        best = None
        for record, output in zip(windows, outputs):
            starts, ends, scores, _ = select_starts_ends(
                start=output["start_logits"], end=output["end_logits"], p_mask=output["p_mask"],
                attention_mask=output["attention_mask"].numpy(), top_k=1,
                handle_impossible_answer=False, max_answer_len=15)
            for start, end, score in zip(starts, ends, scores):
                if record["word_ids"][start] < 0 or record["word_ids"][end] < 0:
                    continue
                if best is None or float(score) > best[0]:
                    best = (float(score), record["input_ids"][start:end + 1])
        results.append({
            "question": question,
            "words": outputs[0]["words"] if outputs else [],
            "answer": answers[0]["answer"] if answers else "",
            "score": answers[0]["score"] if answers else None,
            "token_ids": best[1] if best else [],
            "windows": windows,
        })
        if answers:
            assert best is not None and abs(best[0] - answers[0]["score"]) < 1e-6
    return results


def _check_host_steps(native_build: Path, model_dir: Path, window: int, references: list[dict],
                      tmp_path: Path) -> None:
    """Rerun the family's C++ checker with the checkpoint's own tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    words = []
    windows = []
    for index, reference in enumerate(references):
        words.extend(reference["words"])
        words.extend(reference["question"].split())
        windows.append({
            "name": f"question-{index}",
            "question_words": reference["question"].split(),
            "document_words": reference["words"],
            "boxes": _word_boxes(reference),
            "max_sequence_length": window,
            "stride": min(window // 2, 256),
            # A document token is answer-eligible (p_mask 0) and maps to a word;
            # none of the case questions contains a literal <s>.
            "windows": [{
                "input_ids": item["input_ids"],
                "word_ids": item["word_ids"],
                "in_document": [int(not mask and word >= 0)
                                for mask, word in zip(item["p_mask"], item["word_ids"])],
                "boxes": item["bbox"],
            } for item in reference["windows"]],
        })
    word_cases = []
    for word in dict.fromkeys(words):
        encoding = tokenizer([word], is_split_into_words=True, add_special_tokens=False)
        word_cases.append({"word": word, "ids": list(encoding["input_ids"])})
    golden = tmp_path / "real_vocabulary_golden.json"
    golden.write_text(json.dumps({"tokenizer_json": str(model_dir / "tokenizer.json"),
                                  "word_cases": word_cases, "window_cases": windows},
                                 ensure_ascii=False), encoding="utf-8")
    completed = subprocess.run([str(native_build / "test_layoutlm_document_qa"), str(golden)],
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout.strip())


def _word_boxes(reference: dict) -> list[list[int]]:
    """Each OCR word's box, read back from the windows that carry it."""
    boxes: dict[int, list[int]] = {}
    for window in reference["windows"]:
        for word, mask, box in zip(window["word_ids"], window["p_mask"], window["bbox"]):
            if not mask and word >= 0:
                boxes.setdefault(word, box)
    assert sorted(boxes) == list(range(len(reference["words"])))
    return [boxes[index] for index in range(len(reference["words"]))]


_ENGINES: dict[str, object] = {}


def _engine(model_dir: Path, manifest: dict):
    """The bundle's network at the case precision, built once per manifest in this process."""
    if manifest["name"] not in _ENGINES:
        import tensorrt as trt

        from .. import checkpoint, model

        config = checkpoint.read_config(model_dir)
        plan = model.build_engine(checkpoint.read_weights(model_dir, config), config,
                                  max_sequence_length=int(manifest["max_sequence_length"]),
                                  precision=manifest["precision"])
        _ENGINES[manifest["name"]] = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    return _ENGINES[manifest["name"]]


def _engine_logits(engine, input_ids, boxes) -> tuple[list[float], list[float]]:
    import torch

    context = engine.create_execution_context()
    length = len(input_ids)
    tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int32, device="cuda"),
        "bbox": torch.tensor(boxes, dtype=torch.int32, device="cuda").reshape(length, 4),
        "start_logits": torch.empty(length, dtype=torch.float32, device="cuda"),
        "end_logits": torch.empty(length, dtype=torch.float32, device="cuda"),
    }
    context.set_input_shape("input_ids", (length,))
    context.set_input_shape("bbox", (length, 4))
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return tensors["start_logits"].tolist(), tensors["end_logits"].tolist()


def _logit_error(engine, window, boxes) -> float:
    starts, ends = _engine_logits(engine, window["input_ids"], boxes)
    return max(max(abs(a - b) for a, b in zip(starts, window["start_logits"])),
               max(abs(a - b) for a, b in zip(ends, window["end_logits"])))


def _sdk_answers(native_build: Path, runtime_root: Path, bundle: Path, image: Path,
                 question: str, tmp_path: Path) -> list[dict]:
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    height, width = pixels.shape[:2]
    as_bytes = tmp_path / "document.rgb8"
    as_floats = tmp_path / "document.rgbf32"
    pixels.tofile(as_bytes)
    (pixels.astype(np.float32) / np.float32(255.0)).tofile(as_floats)
    answers = []
    for program, path in (("test_layoutlm_sdk_cpp", as_bytes), ("test_layoutlm_sdk_c", as_floats)):
        completed = subprocess.run(
            [str(native_build / program), str(bundle), str(runtime_root), str(path), str(width),
             str(height), question],
            check=True, capture_output=True, text=True, env=_environment(runtime_root),
            timeout=600)
        answers.append(_last_json(completed.stdout))
    return answers


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    precision = manifest["precision"]
    model_dir = _model_dir(manifest)
    binary, runtime_root, native_build = _runtime()
    version = _tesseract_version()
    image = TEST_ROOT / case["image"]
    questions = [str(question) for question in case["questions"]]
    assert image.is_file() and questions

    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    window = int(manifest["max_sequence_length"])
    references = _reference(model_dir, image, questions, window)
    _check_host_steps(native_build, model_dir, window, references, tmp_path)

    native_answers = []
    for reference in references:
        native = _native_answer(binary, runtime_root, bundle, image, reference["question"])
        native_answers.append(native)
        print(f"{FAMILY} {precision} | {reference['question']} | reference "
              f"{reference['answer']!r} ({reference['score']}) | native {native['text']!r} | "
              f"windows {len(reference['windows'])} | setup {native['setup_ms']:.1f} ms, "
              f"encoder {native['prefill_ms']:.1f} ms")
        assert native["text"] == reference["answer"], reference["question"]
        assert native["token_ids"] == reference["token_ids"], reference["question"]

    engine = _engine(model_dir, manifest)
    worst = 0.0
    for reference in references:
        for window in reference["windows"]:
            worst = max(worst, _logit_error(engine, window, window["bbox"]))
    print(f"{FAMILY} {precision} engine logits: max |native - reference| {worst:.3e} over "
          f"{sum(len(r['windows']) for r in references)} windows; Tesseract {version}")
    assert worst <= _LOGIT_TOLERANCE[precision]
    # The control needs document words, whose boxes it removes; a blank page has none.
    if references[0]["words"]:
        window = references[0]["windows"][0]
        blind = _logit_error(engine, window, [[0, 0, 0, 0]] * len(window["bbox"]))
        print(f"{FAMILY} {precision} layout-blind control: {blind:.3e}")
        assert blind >= _BLIND_MARGIN * _LOGIT_TOLERANCE[precision]

    for answer in _sdk_answers(native_build, runtime_root, bundle, image, questions[0], tmp_path):
        assert answer["task"] == "images_text_to_text"
        assert answer["text"] == native_answers[0]["text"], answer["format"]
        assert answer["token_ids"] == native_answers[0]["token_ids"], answer["format"]

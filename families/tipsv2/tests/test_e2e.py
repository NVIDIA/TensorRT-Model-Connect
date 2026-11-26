# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for TIPSv2.

One bundle serves all three tasks, so the native side is exercised three times through
``--task``: the image tower alone, the text tower alone, and the scored pair. A fourth check runs
the bundle's own tokenizer against the reference SentencePiece processor, because the ids are the
one part of the text path that an embedding comparison can only see second-hand.

**The gate is the embedding, not the score and not the cosine.** Three structural errors were
built on purpose, each a realistic port mistake, and measured against the fp32 case on one B300:

======================  =========  ===========  ====================  ============
quantity                correct    tanh GELU    register positioned   q/v/k split
======================  =========  ===========  ====================  ============
image embedding maxabs   1.08e-07     1.61e-04              1.52e-02      3.66e-01
image embedding cosine   1.0000000    0.9999990             0.9861881     0.2587784
relevance logit maxabs   6.13e-06     1.30e-02              3.70e-01      1.88e+01
zero-shot top class      agrees       agrees                agrees        agrees
======================  =========  ===========  ====================  ============

Two rows of that table are the reason the thresholds below look the way they do. A **cosine**
floor cannot separate a correct build from the GELU break, and the **top class** separates
nothing at all: a tower reading its own attention projection wrong, at cosine 0.26, still prefers
`a photo of a car` to `a photo of a cat`. Both are kept as loud, early failures; the maximum
absolute difference per coordinate is the gate, set an order of magnitude below the weakest break
and two above a correct build.

**The fp32 case is the structural gate; the fp16 case is not.** A correct fp16 build already
differs from the reference by 3.70e-04 - ten times what the weakest structural break costs - and
measuring the same three breaks at fp16 confirms it: the GELU one lands at 3.71e-04 and the case
*passes* with the break in place. The fp16 tolerances therefore gate the lowering, not the graph;
the other two breaks (1.52e-02 and 3.67e-01) are still caught there.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence

FAMILY = "tipsv2"
TASKS = frozenset({"image_to_embedding", "text_to_embedding", "text_image_to_relevance"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# Measured on this case (see the table above): a correct fp32 build sits two orders of magnitude
# below each of these and the weakest deliberate structural break one order above. A case may
# raise them in the manifest, which the fp16 one does.
_EMBEDDING_TOLERANCE = 1e-5
_LOGIT_TOLERANCE = 1e-3
# Kept only to fail early and loudly on a gross error; it cannot separate a correct build from
# the structural breaks above, which is why it is not the gate.
_COSINE_FLOOR = 0.999

# Prompts the bundle's tokenizer is diffed against the reference processor on. They cover what an
# embedding comparison cannot reach on its own: the Unicode lowercase table, the NFKC charsmap,
# byte fallback, whitespace collapsing, truncation past the window, and text that spells a
# special token.
_TOKENIZER_PROMPTS = (
    "a photo of a car",
    "A Photo Of A CAR",
    "  leading   and  trailing   spaces  ",
    "café näive — em-dash",
    "ＦＵＬＬＷＩＤＴＨ text",
    "日本語のテキスト",
    "emoji 🚌 bus",
    "İstanbul and ÅNGSTRÖM",
    # The one character in Unicode whose lower case depends on its neighbours: a word-final
    # capital sigma lowercases to U+03C2 and every other one to U+03C3.
    "ΟΔΥΣΣΕΥΣ ΚΑΙ ΤΑΣ ΘΑΛΑΣΣΑΣ",
    "ΣΟΦΟΣ ΟΣ' ΟΣ\u0301",
    "<pad> </s> <unk> <s>",
    "a very long prompt that runs past the sixty-four token window " * 8,
)


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result: dict[str, tuple[Path, dict, dict]] = {}
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


def _values(raw_values) -> set[str]:
    return {value.strip() for raw in raw_values or [] for value in str(raw).split(",")
            if value.strip()}


def _testcase_filters(config) -> set[str]:
    return _values(config.getoption("--e2e-testcase", default=[]))


def _selected_cases(config) -> list[str] | None:
    """The selected cases, or ``None`` when no E2E selector is given.

    A case is selected when it matches the model filters (``--e2e-model``, ``--e2e-models-file``:
    the family, a manifest or a case name) and the testcase filters; a missing kind of filter
    matches every case. A selection that names this family, one of its manifests or one of its
    cases but matches no case is a usage error, so that a misspelt name cannot pass without
    testing anything; a selection of other families selects nothing here.
    """
    models = _values(config.getoption("--e2e-model", default=[]))
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcases = _testcase_filters(config)
    if not models and not testcases:
        return None
    own_names = {FAMILY, *CASES, *(manifest["name"] for _, manifest, _ in CASES.values())}
    if models and not (models & own_names):
        # The model filters name only other families, so this file has nothing to say - not even
        # about a testcase filter that happens to spell one of these cases.
        return []
    selected = [
        name
        for name, (_, manifest, _) in CASES.items()
        if (not models or models & {FAMILY, name, manifest["name"]})
        and (not testcases or name in testcases)
    ]
    if not selected and models:
        # The selection named this family and still matched nothing, so it is a typo scoped to
        # here. A testcase filter on its own names no family, so a case this family does not own
        # just deselects it; `conftest.py` is what refuses one that no collected family owns.
        raise pytest.UsageError(
            f"the E2E selection names {FAMILY} but matches none of its testcases "
            f"(models: {sorted(models)}, testcases: {sorted(testcases)}; "
            f"available: {sorted(CASES)})")
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selected_cases(metafunc.config)
    if selected is None:
        selected = [
            pytest.param(
                name,
                marks=pytest.mark.skip(
                    reason=f"the real {FAMILY} E2E requires explicit selection"),
                id=name,
            )
            for name in CASES
        ]
    metafunc.parametrize("case_name", selected)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    """The cached snapshot of **the revision the manifest pins**, and nothing else.

    There is deliberately no environment override here. This checkpoint ships its own modelling
    code and the reference loads it with `trust_remote_code`, so the directory this returns is a
    directory whose Python then runs: a variable pointing somewhere else would both execute code
    of unknown provenance and quietly compare the engine against a checkpoint other than the one
    the manifest names - which is the whole of what the parity claim rests on.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
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
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


@cache
def _native_build() -> Path:
    root = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    # CI builds only a family's CTest and SDK-consumer targets (tools/ci/e2e.py), so the E2E
    # builds the bundle tokenizer check it runs.
    subprocess.run(["cmake", "--build", str(root), "--parallel", "8", "--target",
                    "tipsv2_tokenizer_check"], check=True, timeout=900)
    checker = root / "tipsv2_tokenizer_check"
    assert checker.is_file(), "build the tipsv2_tokenizer_check CMake target"
    return checker


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _run_json(binary: Path, runtime_root: Path, bundle: Path, command: str,
              *arguments: str) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root),
                  *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(invocation, check=True, capture_output=True, text=True, env=env,
                               timeout=3600)
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _native_embedding(binary: Path, runtime_root: Path, bundle: Path,
                      *arguments: str) -> list[float]:
    payload = _run_json(binary, runtime_root, bundle, "embed", *arguments)
    values = [float(value) for value in payload["values"]]
    assert values, f"native {FAMILY} returned an empty embedding"
    assert payload["normalization"] == "l2"
    assert payload["embedding_space"] == FAMILY
    norm = math.sqrt(sum(value * value for value in values))
    assert abs(norm - 1.0) <= 1e-3, f"native {FAMILY} embedding is not unit length: {norm}"
    return values


def _reference(model_dir: Path, case: dict):
    """The reference's own numbers, computed the way its model card computes them.

    The card's recipe is ``Resize((448, 448))`` then ``ToTensor()``: pixels in [0, 1] with **no**
    ImageNet normalisation, which is what the runtime reproduces. transformers has no native
    ``tipsv2``, so the checkpoint's own modelling code is loaded through ``trust_remote_code``;
    that code is part of the pinned snapshot, so nothing is fetched here.
    """
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      local_files_only=True).eval()
    transform = transforms.Compose([transforms.Resize((model.config.img_size,
                                                       model.config.img_size)),
                                    transforms.ToTensor()])
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    pixel_values = transform(image).unsqueeze(0)
    texts = [str(case["query"]), str(case["negative_query"])]
    with torch.no_grad():
        class_token = model.encode_image(pixel_values).cls_token[0, 0]
        text_vectors = model.encode_text(texts)

    def unit(vector):
        return (vector / vector.norm()).tolist()

    return (unit(class_token), [unit(row) for row in text_vectors],
            float(model.config.temperature))


def _reference_token_ids(model_dir: Path) -> list[list[int]]:
    """What the checkpoint's own SentencePiece tokenizer makes of the probe prompts.

    `text_encoder.py` lower-cases and then encodes with BOS and EOS disabled; this calls the same
    library the same way, so the comparison is against SentencePiece and not against a second
    copy of the family's own rules.
    """
    import sentencepiece

    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(model_dir / "tokenizer.model"))
    return [list(processor.encode(prompt.lower(), add_bos=False, add_eos=False))
            for prompt in _TOKENIZER_PROMPTS]


def _max_abs(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right), (
        f"{FAMILY} embedding width mismatch: native={len(left)} reference={len(right)}"
    )
    return max(abs(a - b) for a, b in zip(left, right))


def _cosine(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right), (
        f"{FAMILY} embedding width mismatch: native={len(left)} reference={len(right)}"
    )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def _check(name: str, label: str, actual: float, operator: str, expected: float) -> dict:
    passed = actual <= expected if operator == "<=" else actual >= expected
    return {"name": name, "label": label, "scope": "independent_reference", "actual": actual,
            "operator": operator, "expected": expected, "passed": passed}


def reference_checks(*, image_delta: float, image_cosine: float, text_delta: float,
                     logit_delta: float, agrees: bool, embedding_tolerance: float,
                     logit_tolerance: float, cosine_floor: float) -> list[dict]:
    """The five checks the comparison record carries.

    A function rather than a literal in the test body so that `tests/test_model.py` can render
    the record this builds and assert the badge it earns: `tools.e2e_report` voids a whole
    record whose checks do not each have their own name, and a regression there would otherwise
    only show as a badge nobody re-read.
    """
    return [
        _check("image_embedding_maxabs", "Image embedding max |native - reference|", image_delta,
               "<=", embedding_tolerance),
        _check("image_embedding_cosine", "Image embedding cosine", image_cosine, ">=",
               cosine_floor),
        _check("text_embedding_maxabs", "Text embedding max |native - reference|", text_delta,
               "<=", embedding_tolerance),
        _check("relevance_logit_maxabs", "Relevance logit max |native - reference|", logit_delta,
               "<=", logit_tolerance),
        _check("zero_shot_top_class", "Zero-shot top class agrees with the reference",
               1.0 if agrees else 0.0, ">=", 1.0),
    ]


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    embedding_tolerance = float(case.get("embedding_tolerance", _EMBEDDING_TOLERANCE))
    logit_tolerance = float(case.get("logit_tolerance", _LOGIT_TOLERANCE))
    cosine_floor = float(case.get("cosine_floor", _COSINE_FLOOR))
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    checker = _native_build()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    image_path = _asset(case["test_image"])
    queries = [str(case["query"]), str(case["negative_query"])]
    record_evidence("inputs", {"prompt": " | ".join(queries), "queries": queries,
                               "image": image_path, "precision": manifest["precision"]})
    record_evidence("thresholds", {"embedding_maxabs": embedding_tolerance,
                                   "relevance_logit": logit_tolerance,
                                   "cosine_floor": cosine_floor})

    reference_image, reference_texts, temperature = _reference(model_dir, case)

    native_image = _native_embedding(binary, runtime_root, bundle, "--task",
                                     "image_to_embedding", "--image", str(image_path))
    image_delta = _max_abs(native_image, reference_image)
    image_cosine = _cosine(native_image, reference_image)

    native_texts = []
    text_deltas = []
    for text in queries:
        native = _native_embedding(binary, runtime_root, bundle, "--task", "text_to_embedding",
                                   "--text", text)
        cosine = _cosine(native, reference_texts[len(native_texts)])
        assert cosine >= cosine_floor, f"text embedding cosine {cosine} for {text!r}"
        native_texts.append(native)
        text_deltas.append(_max_abs(native, reference_texts[len(text_deltas)]))

    # TIPSv2 is trained with a softmax contrastive loss, so one pair has no probability of its
    # own: the runtime divides the cosine by the learned temperature and returns that logit,
    # which is exactly what the reference would feed its softmax.
    scores = []
    logit_deltas = []
    for index, text in enumerate(queries):
        payload = _run_json(binary, runtime_root, bundle, "rerank", "--task",
                            "text_image_to_relevance", "--query", text, "--image",
                            str(image_path))
        assert payload["score_kind"] == "logit"
        score = float(payload["score"])
        expected_logit = sum(a * b for a, b in zip(reference_image,
                                                   reference_texts[index])) / temperature
        logit_deltas.append(abs(score - expected_logit))
        scores.append(score)

    native_probabilities = _softmax(scores)
    reference_probabilities = _softmax(
        [sum(a * b for a, b in zip(reference_image, row)) / temperature
         for row in reference_texts])
    native_top = max(range(len(scores)), key=scores.__getitem__)
    reference_top = max(range(len(reference_probabilities)),
                        key=reference_probabilities.__getitem__)

    record_evidence("native", {"image_embedding": native_image, "text_embeddings": native_texts,
                               "relevance_logits": scores,
                               "zero_shot_probabilities": native_probabilities,
                               "top_class": queries[native_top]})
    record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32",
                                  "reference_image_embedding": reference_image,
                                  "reference_text_embeddings": reference_texts,
                                  "reference_probabilities": reference_probabilities,
                                  "reference_top_class": queries[reference_top]})

    # The two embeddings themselves, retained as two distinct files: what the comparison is
    # *of*, so the report shows the vectors side by side rather than the input image twice, and
    # so the record satisfies `tools.e2e_report`'s schema for an enforced reference comparison
    # (two different artifacts, and one name per check).
    native_file = tmp_path / "native_image_embedding.npy"
    reference_file = tmp_path / "reference_image_embedding.npy"
    np.save(native_file, np.asarray(native_image, dtype=np.float32))
    np.save(reference_file, np.asarray(reference_image, dtype=np.float32))

    checks = reference_checks(
        image_delta=image_delta, image_cosine=image_cosine, text_delta=max(text_deltas),
        logit_delta=max(logit_deltas), agrees=native_top == reference_top,
        embedding_tolerance=embedding_tolerance, logit_tolerance=logit_tolerance,
        cosine_floor=cosine_floor)
    record_evidence("reference_comparison", {
        "label": f"{manifest['name']} against the checkpoint's own Transformers code",
        "scope": "independent_reference", "enforced": True,
        "native": native_file, "reference": reference_file, "checks": checks})

    # The ids themselves, against SentencePiece: the one part of the text path an embedding
    # comparison can only see second-hand.
    reference_ids = _reference_token_ids(model_dir)
    (tmp_path / "tokenizer_cases.json").write_text(
        json.dumps({"cases": [{"text": prompt, "ids": ids}
                              for prompt, ids in zip(_TOKENIZER_PROMPTS, reference_ids)]}),
        encoding="utf-8")
    completed = subprocess.run([str(checker), str(bundle), str(tmp_path / "tokenizer_cases.json")],
                               capture_output=True, text=True, timeout=900)
    record_evidence("tokenizer", {"returncode": completed.returncode,
                                  "stdout": completed.stdout, "stderr": completed.stderr})
    assert completed.returncode == 0, completed.stdout + completed.stderr

    assert image_cosine >= cosine_floor, f"image embedding cosine {image_cosine}"
    assert image_delta <= embedding_tolerance, (
        f"image embedding differs from the reference by {image_delta}"
    )
    assert max(text_deltas) <= embedding_tolerance, (
        f"text embedding differs from the reference by {max(text_deltas)}"
    )
    assert max(logit_deltas) <= logit_tolerance, (
        f"relevance logit differs from the reference by {max(logit_deltas)}"
    )
    assert scores[0] > scores[1], (
        f"{FAMILY} scored the negative text higher: {scores[0]} <= {scores[1]}"
    )
    assert native_top == reference_top, (
        f"{FAMILY} picked {queries[native_top]!r}, the reference {queries[reference_top]!r}"
    )
    print(
        f"{FAMILY} parity ({manifest['precision']}): image maxabs {image_delta:.3e}, "
        f"text maxabs {max(text_deltas):.3e}, logit maxabs {max(logit_deltas):.3e}, "
        f"zero-shot {native_probabilities[0]:.6f} vs {native_probabilities[1]:.6f}, "
        f"tokenizer {len(_TOKENIZER_PROMPTS)}/{len(_TOKENIZER_PROMPTS)} ids equal"
    )


def _softmax(logits: list[float]) -> list[float]:
    largest = max(logits)
    weights = [math.exp(value - largest) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for openai_privacy_filter.

Each case builds the manifest's bundle once and checks, in order:

* the native tokenizer gives the Hugging Face tokenizer's ids for the prompt and
  for a set of tokenizer probes (case splits, contractions, digit groups,
  whitespace runs, scripts, emoji, added-token strings);
* ``trtmc encode --text`` returns raw logits of shape ``[tokens, labels]``;
  ``--token-ids`` with the reference ids and the public C and C++ consumers
  return the same values;
* the logits match ``OpenAIPrivacyFilterForTokenClassification`` run in
  float32 on the GPU (TF32 off): every value within ``_LOGIT_TOLERANCE`` and
  the predicted label (argmax) equal on every token whose reference top-2
  margin exceeds twice that tolerance.

The long case exceeds twice the receptive field of the stack
(``layers * sliding_window`` on each side), so the interior tokens exercise the
full band of every layer.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "openai_privacy_filter"
TASKS = frozenset({"text_to_head_scores"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# Max |native - reference| logit on the clinical / multilingual / long-records
# cases, measured on a B300 with TensorRT 11.4 against this checkpoint:
#
#   correct engine                                  7.5e-05 / 6.8e-05 / 1.0e-03
#   reference on the CPU instead of the GPU         4.8e-05 / 5.8e-05 / 1.8e-04
#   band |i - j| < window instead of <= window      2.5e-01 / 1.2e+00 / 1.3e+00
#   SwiGLU clamp removed                            4.3e+00 / 3.0e+00 / 2.5e+00
#   two experts per token instead of four           8.3e+00 / 5.5e+00 / 6.0e+00
#   YaRN cos/sin gain removed                       6.4e+00 / 8.9e+00 / 6.1e+00
#   key not scaled by head_dim ** -0.25             8.7e+00 / 1.1e+01 / 1.0e+01
#   gate and linear expert halves swapped           1.3e+01 / 1.3e+01 / 1.4e+01
#   attention sinks removed                         2.3e+01 / 1.9e+01 / 2.0e+01
#   rotate-half instead of interleaved pairs        3.4e+01 / 2.5e+01 / 3.2e+01
#
# The correct engine also stays within 1.3e-05 on the one-token case and
# 1.1e-03 on the near-bound case, so the tolerance sits 4.5x above the correct
# engine and 50x below the weakest break. Most of the long cases' difference
# comes from two of the 32 YaRN inverse frequencies, which the reference
# computes one float32 ulp off (torch's CPU pow); with the reference's values
# the engine is within 1.2e-04 on the long-records case. RMSNorm's epsilon is
# not gated: 1e-6 instead of 1e-5 moves no logit measurably.
_LOGIT_TOLERANCE = 5e-3
_COSINE_FLOOR = 0.99999

_TOKENIZER_PROBES = (
    "HelloWorld helloWorld HELLOworld helloWORLD camelCaseIdentifier",
    "don't I'M they'RE we'Ve she'LL he'd it's",
    "1234567 12 3 1,234.56 +49 30 1234567 2024-11-03T14:22:05Z",
    "...\n\n/// //\n !!! ?!/\r\n# $%^&*()_+-=[]{};:'\",.<>/?\\|`~",
    "a  b   c\t\td \u00a0e\u3000f  \n  \n  x  ",
    "e\u0301 cafe\u0301 \u0301x Ǆǅǆ",
    "Ελληνικά ΑΒΓ αβγ Иван ИВАН Zażółć ŁÓDŹ Příliš žluťoučký Árvíztűrő Șțăîâ",
    "中文测试 日本語 한국어 עברית العربية हिन्दी ไทย",
    "emoji 👍🏽 👨\u200d👩\u200d👧 🇩🇪 ❤\ufe0f",
    "a<|endoftext|>b <|start|>user<|message|>hi<|end|> <|endof",
    "https://example.com/path?q=1#frag user@example.org 2001:db8::1 00:1A:2B:3C:4D:5E",
)


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
        if model_match and (not testcase_filters or name in testcase_filters):
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
    explicit = os.environ.get("TRTMC_OPENAI_PRIVACY_FILTER_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_OPENAI_PRIVACY_FILTER_MODEL_DIR")
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


def _runtime(manifest: dict) -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    assert int(manifest["tensor_parallel_size"]) == 1
    return binary, runtime_root, native_build


@lru_cache(maxsize=None)
def _bundle(model_dir: Path, manifest_path: Path, output_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = output_dir / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )
    return bundle


def _case_text(case: dict) -> str:
    if "prompt" in case:
        return str(case["prompt"])
    repeat = case["prompt_repeat"]
    count = int(repeat["count"])
    assert count >= 1
    return str(repeat.get("separator", "")).join([str(repeat["text"])] * count) + str(
        repeat.get("suffix", "")
    )


def _run(invocation: list[str], runtime_root: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        invocation, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    record_evidence("commands", {"argv": invocation[:4] + ["..."]})
    return completed


def _json_line(completed: subprocess.CompletedProcess) -> dict:
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert len(payloads) == 1, f"expected one JSON result, got: {completed.stdout[-1000:]}"
    return payloads[0]


def _reference_ids(model_dir: Path, texts: list[str]) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return [list(tokenizer(text)["input_ids"]) for text in texts]


def _check_tokenizer(
    native_build: Path, runtime_root: Path, model_dir: Path, texts: list[str], tmp_path: Path
) -> None:
    probe = tmp_path / "tokenizer_probes.json"
    probe.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    # Built on demand: CI lanes build only the SDK consumers of a family by name.
    assert (native_build / "CMakeCache.txt").is_file(), native_build
    target = f"test_{FAMILY}_tokenizer_checkpoint"
    build_command = ["cmake", "--build", str(native_build), "--target", target, "--parallel", "8"]
    record_evidence("commands", {"argv": build_command})
    subprocess.run(build_command, check=True, text=True, timeout=600)
    checkers = [path for path in native_build.rglob(target) if path.is_file()]
    assert len(checkers) == 1, checkers
    completed = _run(
        [str(checkers[0]), str(model_dir / "tokenizer.json"), str(probe)], runtime_root
    )
    native = [json.loads(line) for line in completed.stdout.splitlines()]
    expected = _reference_ids(model_dir, texts)
    assert len(native) == len(expected)
    for text, got, want in zip(texts, native, expected):
        assert got.get("ids") == want, f"native tokenizer differs on {text[:80]!r}: {got}"
    record_evidence("tokenizer", {"texts": len(texts), "tokens": sum(len(ids) for ids in expected)})


def _check_payload(payload: dict, tokens: int, labels: int) -> np.ndarray:
    assert payload["task"] == "text_to_head_scores"
    assert payload["score_kind"] == "logit"
    assert payload["pooling"] == "none" and payload["normalization"] == "none"
    assert [int(value) for value in payload["shape"]] == [tokens, labels]
    values = np.asarray(payload["values"], dtype=np.float64)
    assert values.size == tokens * labels and np.isfinite(values).all()
    return values.reshape(tokens, labels)


_REFERENCE_MODELS: dict[str, object] = {}


def _reference_logits(model_dir: Path, ids: list[int]) -> np.ndarray:
    import torch
    from transformers import AutoModelForTokenClassification

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    key = str(model_dir)
    if key not in _REFERENCE_MODELS:
        _REFERENCE_MODELS[key] = (
            AutoModelForTokenClassification.from_pretrained(str(model_dir), dtype=torch.float32)
            .to("cuda")
            .eval()
        )
    with torch.no_grad():
        logits = _REFERENCE_MODELS[key](input_ids=torch.tensor([ids], device="cuda")).logits[0]
    return logits.double().cpu().numpy()


def _check_logits(native: np.ndarray, reference: np.ndarray) -> dict:
    assert native.shape == reference.shape
    difference = np.abs(native - reference)
    cosine = float(
        np.sum(native * reference) / (np.linalg.norm(native) * np.linalg.norm(reference))
    )
    ordered = np.sort(reference, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    decided = margin > 2 * _LOGIT_TOLERANCE
    agree = native.argmax(axis=1) == reference.argmax(axis=1)
    measured = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "cosine": cosine,
        "tokens": int(native.shape[0]),
        "decided_tokens": int(decided.sum()),
        "label_agreement": float(agree.mean()),
    }
    record_evidence("compare", measured)
    print(f"{FAMILY} parity: {measured}")
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    assert measured["max_abs"] <= _LOGIT_TOLERANCE, (
        f"{FAMILY} logits differ from the reference by {measured['max_abs']}"
    )
    assert agree[decided].all(), (
        f"{FAMILY} labels differ on decided tokens {np.nonzero(decided & ~agree)[0][:20]}"
    )
    return measured


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path, tmp_path_factory) -> None:
    manifest_path, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest["name"], "case": case_name})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]}
    )
    binary, runtime_root, native_build = _runtime(manifest)
    text = _case_text(case)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    labels = len(config["id2label"])
    (ids,) = _reference_ids(model_dir, [text])
    assert 0 < len(ids) <= int(manifest["max_sequence_length"])
    if case.get("exercise_window"):
        reach = int(config["num_hidden_layers"]) * int(config["sliding_window"])
        assert len(ids) > 2 * reach, "the window case must exceed twice the receptive field"

    with evidence_stage("tokenizer"):
        _check_tokenizer(
            native_build, runtime_root, model_dir, [text, *_TOKENIZER_PROBES], tmp_path
        )
    with evidence_stage("build"):
        bundle = _bundle(model_dir, manifest_path, tmp_path_factory.getbasetemp())
    with evidence_stage("native"):
        common = ["--runtime-root", str(runtime_root)]
        cli = _json_line(
            _run([str(binary), "encode", str(bundle), *common, "--text", text], runtime_root)
        )
        native = _check_payload(cli, len(ids), labels)
        by_ids = _json_line(
            _run(
                [str(binary), "encode", str(bundle), *common, "--token-ids", json.dumps(ids)],
                runtime_root,
            )
        )
        np.testing.assert_allclose(
            _check_payload(by_ids, len(ids), labels), native, rtol=0, atol=1e-6
        )
        for language in ("c", "cpp"):
            consumer = _required_path(
                str(native_build / f"test_{FAMILY}_sdk_{language}"), f"the {language} consumer"
            )
            result = _json_line(
                _run([str(consumer), str(bundle), str(runtime_root), text], runtime_root)
            )
            np.testing.assert_allclose(
                _check_payload(result, len(ids), labels), native, rtol=1e-6, atol=1e-6
            )
    with evidence_stage("reference"):
        reference = _reference_logits(model_dir, ids)
    with evidence_stage("compare"):
        _check_logits(native, reference)

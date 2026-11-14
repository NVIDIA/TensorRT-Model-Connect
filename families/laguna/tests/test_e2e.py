# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for Laguna.

Each case builds (once per manifest) the bundle, runs the native CLI through
the bundle's primary Task (``text_continuation``, Task SDK), records a logits
trace (prefill context for the last prompt token, decode context for every
fed token) and compares with transformers on the same tokens:

* the native prompt token IDs equal the transformers chat-template IDs;
* the greedy text agrees with the reference greedy text (manifest
  ``reference_precision``) and contains the case's expected continuation;
* the prefill context's next token equals the reference's;
* the decode-context logits stay as close to the reference as transformers'
  own runs at the bundle precision do on the same token sequence. Laguna's
  sigmoid-routed 256-expert MoE moves far under BF16 rounding, so a fixed FP32
  distance would reject the reference framework's own BF16 runs; the oracle
  therefore measures two of them (eager and SDPA attention), bounds the engine
  by the less favourable value of each statistic, keeps an absolute floor on the
  median row, and repeats the relative bounds on the rows past the sliding
  window, where the window changes the result.
"""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import gc
import json
import os
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_THRESHOLDS = {
    # The median trace row must agree closely with the reference.
    "logit_cosine_p50_floor": 0.99,
    # Bounds relative to transformers at the bundle precision on the same tokens.
    "logit_cosine_p50_margin": 0.005,
    "logit_cosine_p5_margin": 0.03,
    "logit_rel_l2_p95_ratio": 1.25,
    "logit_rel_l2_p95_slack": 0.02,
    "top1_agreement_margin": 0.03,
    # A trace that crosses the sliding window must do so by enough rows for
    # the per-segment percentiles to mean something.
    "min_rows_past_window": 32,
    "normalized_text_edit_distance": 0.15,
}
# Transformers runs at the bundle precision whose spread the oracle accepts.
_BASELINE_ATTENTION = ("eager", "sdpa")


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "text_continuation", path
        assert manifest["tensor_parallel_size"] == 1, path
        assert isinstance(manifest["max_sequence_length"], int), path
        for case in manifest["testcases"]:
            assert case["name"] not in cases, case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> set[str]:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        selected |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selection(config)
    if os.environ.get("TRTMC_E2E") != "1" and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _required_environment():
    binary = Path(os.environ.get("TRTMC_BINARY", ""))
    runtime_root = Path(os.environ.get("TRTMC_RUNTIME_ROOT", ""))
    build_dir = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", ""))
    assert binary.is_file() and os.access(binary, os.X_OK), "selected E2E requires TRTMC_BINARY"
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), "TRTMC_RUNTIME_ROOT"
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), "TRTMC_RUNTIME_ROOT"
    assert build_dir.is_dir(), "the Laguna logits trace requires TRTMC_NATIVE_BUILD_DIR"

    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    return binary, runtime_root, build_dir, torch


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"]))
    assert (path / "config.json").is_file(), path
    return path


@pytest.fixture(scope="session")
def _bundles(tmp_path_factory) -> dict:
    return {"root": tmp_path_factory.mktemp("laguna-bundles")}


def _bundle(bundles: dict, manifest: dict, model_dir: Path) -> Path:
    """Builds each manifest's bundle once per session; cases reuse it."""
    path = bundles["root"] / manifest["bundle"]
    if not path.is_file():
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=path,
                family=_FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest["max_sequence_length"],
            )
        )
    assert path.is_file() and path.stat().st_size > 0, path
    return path


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _run_native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    """``trtmc run`` through the bundle's primary Task; options map to its config fields."""
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        case["prompt"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--use-chat-template",
        _flag(case["use_chat_template"]),
        "--enable-thinking",
        _flag(case["enable_thinking"]),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stderr[-4000:]
    record_evidence("commands", {"argv": completed.args})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    return json.loads(completed.stdout)


@cache
def _trace_binary(build_dir: Path) -> Path:
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "8", "--target", "laguna_logits_trace"],
        check=True,
        timeout=1800,
    )
    binary = build_dir / "families" / _FAMILY / "laguna_logits_trace"
    assert binary.is_file(), binary
    return binary


def _native_trace(
    build_dir: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path
) -> tuple[list[int], np.ndarray, np.ndarray]:
    prefix = tmp_path / "laguna-trace"
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(_trace_binary(build_dir)),
            str(bundle),
            str(runtime_root),
            str(prefix),
            case["prompt"],
            str(case["max_new_tokens"]),
            "1" if case["use_chat_template"] else "0",
            "1" if case["enable_thinking"] else "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    record_evidence("commands", {"argv": completed.args})
    prompt_ids = [int(line) for line in Path(f"{prefix}.ids").read_text().split()]
    rows, columns = (int(value) for value in Path(f"{prefix}.shape").read_text().split())
    logits = np.fromfile(f"{prefix}.f32", dtype="<f4").reshape(rows, columns)
    prefill_last = np.fromfile(f"{prefix}.prefill.f32", dtype="<f4")
    assert prefill_last.shape == (columns,), prefill_last.shape
    # Row r holds the logits after token r of the fed sequence; the tokens
    # after the prompt are the greedy argmax of the preceding row.
    fed = prompt_ids + [int(np.argmax(logits[row])) for row in range(len(prompt_ids) - 1, rows - 1)]
    return fed, logits, prefill_last


def _reference(model_dir: Path, manifest: dict, case: dict, fed_ids: list[int], torch) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if case["use_chat_template"]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=case["enable_thinking"],
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tokenizer(case["prompt"])["input_ids"]
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = {
            "prompt_ids": prompt_ids,
            "tokenizer": tokenizer,
            "window": config["sliding_window"],
        }
        # The reference first (greedy text and the target logits), then the same
        # framework at the bundle precision on the same tokens, once per attention
        # implementation.
        runs = [("reference", manifest["reference_precision"], "eager")]
        runs += [(f"baseline_{name}", manifest["precision"], name) for name in _BASELINE_ATTENTION]
        for role, precision, attention in runs:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                dtype=dtypes[precision],
                device_map="cuda",
                attn_implementation=attention,
                experts_implementation="eager",
            ).eval()
            with torch.inference_mode():
                if role == "reference":
                    inputs = torch.tensor([prompt_ids], device="cuda")
                    result["ids"] = model.generate(
                        input_ids=inputs,
                        attention_mask=torch.ones_like(inputs),
                        max_new_tokens=int(case["max_new_tokens"]),
                        do_sample=False,
                    )[0, len(prompt_ids) :].tolist()
                fed = torch.tensor([fed_ids], device="cuda")
                result[f"{role}_logits"] = model(input_ids=fed).logits[0].float().cpu().numpy()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    result["text"] = tokenizer.decode(result["ids"], skip_special_tokens=True)
    return result


def _cosine(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(actual, axis=-1) * np.linalg.norm(reference, axis=-1)
    return np.sum(actual * reference, axis=-1) / np.maximum(norms, 1e-30)


def _logit_statistics(actual: np.ndarray, reference: np.ndarray) -> dict:
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    actual = actual.astype(np.float64)
    reference = reference.astype(np.float64)
    assert np.isfinite(actual).all(), "logits are not finite"
    cosine = _cosine(actual, reference)
    relative_l2 = np.linalg.norm(actual - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-30
    )
    return {
        "rows": int(actual.shape[0]),
        "cosine_p50": float(np.percentile(cosine, 50)),
        "cosine_p5": float(np.percentile(cosine, 5)),
        "cosine_min": float(np.min(cosine)),
        "relative_l2_p95": float(np.percentile(relative_l2, 95)),
        "top1_agreement": float(np.mean(np.argmax(actual, 1) == np.argmax(reference, 1))),
    }


def _envelope(baselines: list[dict]) -> dict:
    """The less favourable value of each statistic over the baseline runs."""
    return {
        "rows": baselines[0]["rows"],
        "cosine_p50": min(run["cosine_p50"] for run in baselines),
        "cosine_p5": min(run["cosine_p5"] for run in baselines),
        "cosine_min": min(run["cosine_min"] for run in baselines),
        "relative_l2_p95": max(run["relative_l2_p95"] for run in baselines),
        "top1_agreement": min(run["top1_agreement"] for run in baselines),
    }


def _assert_logits(engine: dict, baseline: dict, thresholds: dict) -> None:
    """Bounds the engine's statistics by the baseline envelope's."""
    assert engine["cosine_p50"] >= baseline["cosine_p50"] - thresholds["logit_cosine_p50_margin"], (
        engine,
        baseline,
    )
    assert engine["cosine_p5"] >= baseline["cosine_p5"] - thresholds["logit_cosine_p5_margin"], (
        engine,
        baseline,
    )
    assert engine["relative_l2_p95"] <= (
        baseline["relative_l2_p95"] * thresholds["logit_rel_l2_p95_ratio"]
        + thresholds["logit_rel_l2_p95_slack"]
    ), (engine, baseline)
    assert engine["top1_agreement"] >= (
        baseline["top1_agreement"] - thresholds["top1_agreement_margin"]
    ), (engine, baseline)


def _normalized_edit_distance(left: str, right: str) -> float:
    left = " ".join(left.casefold().split())
    right = " ".join(right.casefold().split())
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(left), len(right))


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path, _bundles) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root, build_dir, torch = _required_environment()
    model_dir = _checkpoint(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest["hf_id"],
            "hf_revision": manifest["hf_revision"],
        },
    )
    with evidence_stage("build"):
        bundle = _bundle(_bundles, manifest, model_dir)
    with evidence_stage("native"):
        native = _run_native(binary, runtime_root, bundle, case)
        fed_ids, trace, prefill_last = _native_trace(
            build_dir, runtime_root, bundle, case, tmp_path
        )
    record_evidence("native", native)
    with evidence_stage("reference"):
        reference = _reference(model_dir, manifest, case, fed_ids, torch)
    record_evidence(
        "reference",
        {"prompt_ids": reference["prompt_ids"], "ids": reference["ids"], "text": reference["text"]},
    )

    with evidence_stage("compare"):
        thresholds = record_evidence("thresholds", dict(_THRESHOLDS))
        prompt_length = len(reference["prompt_ids"])
        assert fed_ids[:prompt_length] == reference["prompt_ids"], "prompt token IDs differ"
        assert native["token_ids"] and len(native["token_ids"]) <= case["max_new_tokens"]
        target = reference["reference_logits"]
        # Row r holds position r; from row `window` on, the sliding window hides
        # the oldest keys from the sliding-attention layers.
        window = reference["window"]
        segments = {"all_rows": slice(0, trace.shape[0])}
        past_window = trace.shape[0] - window
        if past_window > 0:
            assert past_window >= thresholds["min_rows_past_window"], (past_window, window)
            segments["rows_past_window"] = slice(window, trace.shape[0])
        statistics = {}
        for segment, rows in segments.items():
            runs = {
                name: _logit_statistics(reference[f"baseline_{name}_logits"][rows], target[rows])
                for name in _BASELINE_ATTENTION
            }
            statistics[segment] = {
                "engine": _logit_statistics(trace[rows], target[rows]),
                "baselines": runs,
                "envelope": _envelope(list(runs.values())),
            }
        prefill_cosine = float(_cosine(prefill_last.astype(np.float64), target[prompt_length - 1]))
        record_evidence("diagnostics", {"logits": statistics, "prefill_cosine": prefill_cosine})
        engine = statistics["all_rows"]["engine"]
        assert engine["cosine_p50"] >= thresholds["logit_cosine_p50_floor"], engine
        for segment in statistics.values():
            _assert_logits(segment["engine"], segment["envelope"], thresholds)
        assert int(np.argmax(prefill_last)) == int(np.argmax(target[prompt_length - 1])), (
            "the prefill context predicts a different next token"
        )
        decoded = reference["tokenizer"].decode(native["token_ids"], skip_special_tokens=True)
        assert native["text"].strip() == decoded.strip(), "native detokenization differs"
        matching = next(
            (i for i, (a, b) in enumerate(zip(native["token_ids"], reference["ids"])) if a != b),
            min(len(native["token_ids"]), len(reference["ids"])),
        )
        ned = _normalized_edit_distance(native["text"], reference["text"])
        record_evidence(
            "diagnostics", {"matching_prefix_tokens": matching, "normalized_edit_distance": ned}
        )
        assert ned <= thresholds["normalized_text_edit_distance"], (
            native["text"],
            reference["text"],
        )
        expected = case["expected_continuation_text"]
        assert expected.casefold() in native["text"].casefold(), native["text"]
        assert expected.casefold() in reference["text"].casefold(), reference["text"]

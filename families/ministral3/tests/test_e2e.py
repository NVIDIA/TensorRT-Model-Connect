# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for ministral3.

The oracle is the checkpoint's documented Transformers recipe: tokenize
``prompt + text`` without padding, run ``AutoModel``, average the final hidden
states over every token and L2-normalize. It runs in FP32 with TF32 disabled
(the truth) and, for a BF16 bundle, also in BF16 to bound the precision drift
the official BF16 path itself shows. The native side runs the CLI for every
text, the token-feature Task for every query and document without prompt, the
public C and C++ consumers for one query and the pooled-feature Task for one
document. BF16 drift is compared on statistics pooled over all texts, tokens and
query-document pairs of a case, since the worst single row of one text varies by
about 2x between any two BF16 implementations.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "ministral3"
TASKS = frozenset({"text_to_embedding"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
DEFAULT_THRESHOLDS = {
    # FP32 bundle vs the FP32 reference.
    "embedding_cosine": 0.999999,
    "score_abs_diff": 1e-5,
    "token_feature_cosine": 0.99999,
    "pooled_feature_cosine": 0.999999,
    # Raw features also bound their length: |native - reference| / |reference|
    # may not exceed sqrt(2 (1 - cosine bound)), the deviation the cosine bound
    # allows at equal length (rounded down).
    "token_feature_relative_error": 4.4e-3,
    "pooled_feature_relative_error": 1.4e-3,
    # BF16 bundle: its drift from the FP32 truth (1 - cosine, |score difference|,
    # relative error of raw features) as a multiple of the official BF16 run's
    # drift on the same inputs.
    "bf16_mean_drift_ratio": 1.5,
    "bf16_max_drift_ratio": 3.0,
    # Both precisions: text_to_embedding returns the mean L2-normalized in FP64 on
    # the host, so |norm - 1| of every native embedding stays at FP32 rounding.
    "embedding_norm_deviation": 1e-5,
}


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


def _model_dir(manifest: dict) -> tuple[Path, bool]:
    """The checkpoint directory, and whether it is the pinned snapshot itself."""
    explicit = os.environ.get("TRTMC_MINISTRAL3_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_MINISTRAL3_MODEL_DIR"), False
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot), True


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native_build


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


def _run_json(program: Path, arguments: list[str], runtime_root: Path, timeout: int) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [str(program), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    record_evidence("commands", {"argv": completed.args})
    lines = [line for line in completed.stdout.splitlines() if line.lstrip().startswith("{")]
    assert lines, (
        f"{program.name} returned no JSON: {completed.stdout[-1000:]} {completed.stderr[-1000:]}"
    )
    return json.loads(lines[-1])


def _texts(case: dict) -> list[tuple[str, str]]:
    """``(role, text)`` pairs of the case: queries first, then documents."""
    texts = [("query", text) for text in case["queries"]]
    texts += [("document", text) for text in case["documents"]]
    for repeat in case.get("repeated_documents", []):
        texts.append(
            (
                "document",
                " ".join(f"[{index}] {repeat['text']}" for index in range(repeat["count"])),
            )
        )
    assert texts, f"selected {FAMILY} E2E case needs queries or documents"
    return texts


def _native(binary, runtime_root, native_build, bundle, case) -> dict:
    timeout = int(case.get("runtime_timeout_s", 1800))
    embeddings = []
    for role, text in _texts(case):
        result = _run_json(
            binary,
            [
                "embed",
                str(bundle),
                "--runtime-root",
                str(runtime_root),
                "--role",
                role,
                "--text",
                text,
            ],
            runtime_root,
            timeout,
        )
        assert result["pooling"] == "mean" and result["normalization"] == "l2"
        embeddings.append(np.asarray(result["values"], dtype=np.float32))
    role, text = _texts(case)[0]
    for language in ("c", "cpp"):
        consumer = _required_path(
            str(native_build / f"test_{FAMILY}_sdk_{language}"), f"public {language} consumer"
        )
        result = _run_json(
            consumer, [str(bundle), str(runtime_root), role, text], runtime_root, timeout
        )
        record_evidence("native", {"consumer": language, "values_head": result["values"][:8]})
        assert result["task"] == "text_to_embedding"
        np.testing.assert_allclose(result["values"], embeddings[0], rtol=1e-6, atol=1e-7)
    tokens = []
    for raw in case["queries"] + case["documents"]:
        common = [str(bundle), "--runtime-root", str(runtime_root), "--text", raw]
        result = _run_json(
            binary, ["encode", *common, "--task", "text_to_token_features"], runtime_root, timeout
        )
        rows, columns = result["shape"]
        tokens.append(
            {
                "token_ids": [item["token_id"] for item in result["tokens"]],
                "byte_offsets": [item["byte_offsets"] for item in result["tokens"]],
                "hidden": np.asarray(result["values"], dtype=np.float32).reshape(rows, columns),
            }
        )
    common = [str(bundle), "--runtime-root", str(runtime_root), "--text", case["documents"][0]]
    pooled = _run_json(
        binary, ["encode", *common, "--task", "text_to_pooled_features"], runtime_root, timeout
    )
    assert pooled["pooling"] == "mean" and pooled["normalization"] == "none"
    return {
        "embeddings": np.stack(embeddings),
        "tokens": tokens,
        "pooled": np.asarray(pooled["values"], dtype=np.float32),
    }


@contextmanager
def _tf32_disabled(torch):
    """Run the reference without TF32, then restore the two process-wide flags.

    The flags are global to the process, which may run other families' tests
    next; they are restored whether the reference succeeds or fails. Only these
    flags are touched: setting the matmul precision through the newer API as
    well would make ``torch.get_float32_matmul_precision`` raise later.
    """
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.get_float32_matmul_precision() == "highest"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _official_reference(model_dir: Path, case: dict, precision: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    with _tf32_disabled(torch):
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
        prompts = json.loads((model_dir / "config_sentence_transformers.json").read_text())[
            "prompts"
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa")
        model = model.to("cuda").eval()
        embeddings, tokens = [], []
        with torch.inference_mode():
            for role, text in _texts(case):
                encoded = tokenizer(prompts[role] + text, return_tensors="pt", truncation=False)
                hidden = model(**encoded.to("cuda")).last_hidden_state
                pooled = hidden.sum(dim=1) / hidden.shape[1]
                embeddings.append(
                    functional.normalize(pooled, p=2, dim=-1)[0].float().cpu().numpy()
                )
            for raw in case["queries"] + case["documents"]:
                encoded = tokenizer(raw, return_tensors="pt", truncation=False)
                hidden = model(**encoded.to("cuda")).last_hidden_state[0].float()
                tokens.append(
                    {"token_ids": encoded["input_ids"][0].tolist(), "hidden": hidden.cpu().numpy()}
                )
        del model
        torch.cuda.empty_cache()
        return {"embeddings": np.stack(embeddings), "tokens": tokens}


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    overrides = (
        json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]
        if path.is_file()
        else {}
    )
    return {**DEFAULT_THRESHOLDS, **overrides}


def _row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row cosines in FP64; the cosine of a zero row is NaN (undefined), not a warning."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    # Round-off can put the cosine of equal rows just above 1, which would make
    # an exact match drift below 0; NaN stays NaN.
    return np.clip(cosines, -1.0, 1.0)


def _row_relative_errors(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def _scores(embeddings: np.ndarray, queries: int) -> np.ndarray:
    values = embeddings.astype(np.float64)
    return values[:queries] @ values[queries:].T


def _token_shapes_match(mine: dict, theirs: dict) -> bool:
    """One native feature row per token ID and byte offset, shaped like the reference's."""
    hidden = np.asarray(mine["hidden"])
    return (
        hidden.ndim == 2
        and hidden.shape == np.shape(theirs["hidden"])
        and hidden.shape[0] == len(mine["token_ids"]) == len(mine["byte_offsets"])
    )


def _drifts(native: dict, reference: dict, queries: int, keep: list[bool] | None = None) -> dict:
    """Per-text embedding, per-pair score and per-token feature drift from ``reference``.

    Token features enter only for the texts that ``keep`` selects (default: all);
    a text whose features are misshaped is reported by its own check instead of
    being broadcast against the reference rows.
    """
    scores = _scores(native["embeddings"], queries) - _scores(reference["embeddings"], queries)
    pairs = [
        (mine, theirs)
        for index, (mine, theirs) in enumerate(zip(native["tokens"], reference["tokens"]))
        if keep is None or keep[index]
    ]
    nothing = [np.zeros(0)]
    return {
        "embedding": 1 - _row_cosines(native["embeddings"], reference["embeddings"]),
        "score": np.abs(scores).reshape(-1),
        "token": np.concatenate(
            nothing + [1 - _row_cosines(mine["hidden"], theirs["hidden"]) for mine, theirs in pairs]
        ),
        "token_relative": np.concatenate(
            nothing
            + [_row_relative_errors(mine["hidden"], theirs["hidden"]) for mine, theirs in pairs]
        ),
    }


def _largest(values: np.ndarray) -> float:
    """The largest value; NaN when there is none or one of them is undefined."""
    return float(np.max(values)) if np.size(values) else math.nan


def _order_violations(native_scores: np.ndarray, truth_scores: np.ndarray, tolerance: float) -> int:
    """Document pairs ranked in the wrong order for some query.

    A pair counts when its FP32 truth scores differ by more than ``tolerance``
    (the score difference the case allows) and the native scores do not keep the
    truth's order; pairs closer than that are ties the score check already bounds.
    """
    count = 0
    for mine, theirs in zip(native_scores, truth_scores):
        ahead = (theirs[:, None] - theirs[None, :]) > tolerance
        count += int(np.count_nonzero(ahead & ~(mine[:, None] > mine[None, :])))
    return count


def _pooled_cosine(pooled: np.ndarray, reference: dict, queries: int) -> float:
    document_mean = reference["tokens"][queries]["hidden"].astype(np.float64).mean(axis=0)
    return float(_row_cosines(np.asarray(pooled)[None, :], document_mean[None, :])[0])


def _pooled_relative_error(pooled: np.ndarray, reference: dict, queries: int) -> float:
    document_mean = reference["tokens"][queries]["hidden"].astype(np.float64).mean(axis=0)
    return float(_row_relative_errors(np.asarray(pooled)[None, :], document_mean[None, :])[0])


def _check(name: str, label: str, scope: str, actual, operator: str, expected) -> dict:
    """One row of the recorded comparison; values are plain Python numbers.

    A measured value or bound that is not a finite number (for example the cosine
    of a zero vector) cannot be compared. The row then records the failed contract
    that both are defined, so the recorded comparison stays complete.
    """
    if operator != "==" and not (math.isfinite(float(actual)) and math.isfinite(float(expected))):
        return {
            "name": f"{name}_defined",
            "label": f"{label}: measured value and bound are finite numbers",
            "scope": "contract",
            "actual": 0,
            "operator": "==",
            "expected": 1,
            "passed": False,
        }
    actual, expected = (
        (int(actual), int(expected))
        if operator == "=="
        else (
            float(actual),
            float(expected),
        )
    )
    passed = (
        actual == expected
        if operator == "=="
        else actual >= expected
        if operator == ">="
        else actual <= expected
    )
    return {
        "name": name,
        "label": label,
        "scope": scope,
        "actual": actual,
        "operator": operator,
        "expected": expected,
        "passed": bool(passed),
    }


def _tiles(text: str, offsets: list) -> bool:
    data = text.encode("utf-8")
    ends = [0] + [end for _, end in offsets]
    return bool(
        offsets
        and all(begin == ends[index] and begin < end for index, (begin, end) in enumerate(offsets))
        and ends[-1] == len(data)
        and b"".join(data[begin:end] for begin, end in offsets) == data
    )


def _parity_checks(
    native: dict, truth: dict, official: dict | None, case: dict, thresholds: dict
) -> tuple[dict, list[tuple[dict, str]]]:
    """The case's summary and every check with its failure message, in assertion order."""
    queries = len(case["queries"])
    texts = case["queries"] + case["documents"]
    differing_ids = [
        text
        for text, mine, theirs in zip(texts, native["tokens"], truth["tokens"])
        if mine["token_ids"] != theirs["token_ids"]
    ]
    untiled = [
        (text, mine["byte_offsets"])
        for text, mine in zip(texts, native["tokens"])
        if not _tiles(text, mine["byte_offsets"])
    ]
    checks = [
        (
            _check(
                "token_ids_differing_texts",
                "Texts whose native token IDs differ from AutoTokenizer",
                "independent_reference",
                len(differing_ids),
                "==",
                0,
            ),
            f"native token IDs differ from AutoTokenizer: {differing_ids[:1]!r}",
        ),
        (
            _check(
                "byte_offsets_untiled_texts",
                "Texts whose byte offsets do not tile the input",
                "contract",
                len(untiled),
                "==",
                0,
            ),
            f"native byte offsets do not tile the input: {untiled[:1]!r}",
        ),
    ]
    shaped = [
        _token_shapes_match(mine, theirs) for mine, theirs in zip(native["tokens"], truth["tokens"])
    ]
    misshaped = [
        {
            "text": text,
            "native_hidden": list(np.shape(mine["hidden"])),
            "token_ids": len(mine["token_ids"]),
            "byte_offsets": len(mine["byte_offsets"]),
            "reference_hidden": list(np.shape(theirs["hidden"])),
        }
        for text, mine, theirs, ok in zip(texts, native["tokens"], truth["tokens"], shaped)
        if not ok
    ]
    checks.append(
        (
            _check(
                "token_feature_shapes_mismatched_texts",
                "Texts whose token features are not one row per token, shaped like the reference",
                "contract",
                len(misshaped),
                "==",
                0,
            ),
            f"native token-feature shape differs: {misshaped[:1]!r}",
        )
    )
    norm_deviation = _largest(
        np.abs(np.linalg.norm(np.asarray(native["embeddings"], dtype=np.float64), axis=-1) - 1.0)
    )
    checks.append(
        (
            _check(
                "embedding_norm_deviation_max",
                "Largest |L2 norm - 1| of a native embedding",
                "contract",
                norm_deviation,
                "<=",
                thresholds["embedding_norm_deviation"],
            ),
            f"native embeddings are not L2-normalized: |norm - 1| up to {norm_deviation:.3g}",
        )
    )
    drift = _drifts(native, truth, queries, keep=shaped)
    reference_drift = None if official is None else _drifts(official, truth, queries)
    summary = {
        "embedding_cosine_min": 1 - _largest(drift["embedding"]),
        "score_abs_diff_max": _largest(drift["score"]) if drift["score"].size else 0.0,
        "token_feature_cosine_min": 1 - _largest(drift["token"]),
        "token_feature_relative_error_max": _largest(drift["token_relative"]),
        "pooled_feature_cosine": _pooled_cosine(native["pooled"], truth, queries),
        "pooled_feature_relative_error": _pooled_relative_error(native["pooled"], truth, queries),
        "embedding_norm_deviation_max": norm_deviation,
        "tokens": int(drift["token"].size),
    }
    if drift["score"].size:
        native_scores = _scores(native["embeddings"], queries)
        truth_scores = _scores(truth["embeddings"], queries)
        moved = int((native_scores.argmax(axis=1) != truth_scores.argmax(axis=1)).sum())
        checks.append(
            (
                _check(
                    "retrieval_top_document_changes",
                    "Queries whose top-scoring document differs from the FP32 truth",
                    "independent_reference",
                    moved,
                    "==",
                    0,
                ),
                "retrieval ranking differs",
            )
        )
        # The order of the other documents is held to the score difference the
        # case allows: FP32 its absolute bound, BF16 the bound on its largest
        # score difference. Documents closer than that in the truth are ties.
        tolerance = (
            thresholds["score_abs_diff"]
            if reference_drift is None
            else thresholds["bf16_max_drift_ratio"] * _largest(reference_drift["score"])
        )
        reversed_pairs = _order_violations(native_scores, truth_scores, tolerance)
        summary["retrieval_order_tolerance"] = tolerance
        checks.append(
            (
                _check(
                    "retrieval_order_violations",
                    "Document pairs whose FP32 truth scores differ by more than the case's score "
                    "tolerance but whose native order is not the truth's",
                    "independent_reference",
                    reversed_pairs,
                    "==",
                    0,
                ),
                f"retrieval ranking differs: {reversed_pairs} document pairs out of the FP32 "
                f"truth's order beyond the score tolerance {tolerance:.3g}",
            )
        )
    if official is None:
        rows = (
            ("embedding_cosine_min", "Embedding cosine, minimum", ">=", "embedding_cosine", None),
            (
                "score_abs_diff_max",
                "Query-document score difference, maximum",
                "<=",
                "score_abs_diff",
                None,
            ),
            (
                "token_feature_cosine_min",
                "Token-feature cosine, minimum",
                ">=",
                "token_feature_cosine",
                None,
            ),
            (
                "token_feature_relative_error_max",
                "Token-feature relative L2 error, maximum",
                "<=",
                "token_feature_relative_error",
                "token-feature relative error exceeds its bound",
            ),
            ("pooled_feature_cosine", "Pooled-feature cosine", ">=", "pooled_feature_cosine", None),
            (
                "pooled_feature_relative_error",
                "Pooled-feature relative L2 error",
                "<=",
                "pooled_feature_relative_error",
                "pooled-feature relative error exceeds its bound",
            ),
        )
        for key, label, operator, limit, prefix in rows:
            message = f"{prefix}: {summary}" if prefix else str(summary)
            checks.append(
                (
                    _check(
                        key,
                        label,
                        "independent_reference",
                        summary[key],
                        operator,
                        thresholds[limit],
                    ),
                    message,
                )
            )
        return summary, checks
    # A BF16 bundle is judged only relative to the official BF16 run's drift from
    # the FP32 truth. A mean over few tokens is dominated by single texts.
    checks.append(
        (
            _check(
                "bf16_token_count",
                "Tokens with features behind the BF16 statistics",
                "contract",
                summary["tokens"],
                ">=",
                100,
            ),
            "BF16 drift statistics need at least 100 tokens of token features",
        )
    )
    official_pooled = official["tokens"][queries]["hidden"].astype(np.float64).mean(axis=0)
    pooled = {
        "native": 1 - summary["pooled_feature_cosine"],
        "official_bf16": 1 - _pooled_cosine(official_pooled, truth, queries),
        "native_relative": summary["pooled_feature_relative_error"],
        "official_bf16_relative": _pooled_relative_error(official_pooled, truth, queries),
    }
    # Each repeated long document is one embedding averaged over thousands of
    # tokens, a stable statistic on its own; it covers the positions beyond
    # original_max_position_embeddings that only these documents reach.
    first_long = len(case["queries"]) + len(case["documents"])
    long_documents = [
        {
            "native": float(drift["embedding"][index]),
            "official_bf16": float(reference_drift["embedding"][index]),
        }
        for index in range(first_long, drift["embedding"].size)
    ]
    statistics = {}
    for kind in ("embedding", "score", "token", "token_relative"):
        values, baseline = drift[kind], reference_drift[kind]
        if values.size:
            statistics[kind] = {
                "native_mean": float(values.mean()),
                "official_bf16_mean": float(baseline.mean()),
                "native_max": float(values.max()),
                "official_bf16_max": float(baseline.max()),
            }
    summary.update(drift_statistics=statistics, pooled_drift=pooled, long_documents=long_documents)
    mean_ratio, max_ratio = thresholds["bf16_mean_drift_ratio"], thresholds["bf16_max_drift_ratio"]
    words = {
        "embedding": "embedding drift (1 - cosine)",
        "score": "score difference",
        "token": "token-feature drift (1 - cosine)",
        "token_relative": "token-feature relative L2 error",
    }
    for kind, values in statistics.items():
        for statistic, ratio in (("mean", mean_ratio), ("max", max_ratio)):
            checks.append(
                (
                    _check(
                        f"bf16_{kind}_{statistic}",
                        f"BF16 {words[kind]}, {statistic}: at most {ratio:g}x the official BF16 run's",
                        "independent_reference",
                        values[f"native_{statistic}"],
                        "<=",
                        ratio * values[f"official_bf16_{statistic}"],
                    ),
                    f"BF16 {kind} {statistic} drift exceeds the official BF16 drift: {summary}",
                )
            )
    checks.append(
        (
            _check(
                "bf16_pooled_feature_drift",
                f"BF16 pooled-feature drift (1 - cosine): at most {max_ratio:g}x the official BF16 run's",
                "independent_reference",
                pooled["native"],
                "<=",
                max_ratio * pooled["official_bf16"],
            ),
            f"BF16 pooled-feature drift exceeds the official BF16 drift: {summary}",
        )
    )
    checks.append(
        (
            _check(
                "bf16_pooled_feature_relative_error",
                f"BF16 pooled-feature relative L2 error: at most {max_ratio:g}x the official BF16 run's",
                "independent_reference",
                pooled["native_relative"],
                "<=",
                max_ratio * pooled["official_bf16_relative"],
            ),
            f"BF16 pooled-feature relative error exceeds the official BF16 one: {summary}",
        )
    )
    for index, item in enumerate(long_documents):
        checks.append(
            (
                _check(
                    f"bf16_long_document_{index + 1}_drift",
                    f"BF16 drift of long document {index + 1}: at most {max_ratio:g}x the official BF16 run's",
                    "independent_reference",
                    item["native"],
                    "<=",
                    max_ratio * item["official_bf16"],
                ),
                f"BF16 long document drift exceeds the official BF16 drift: {summary}",
            )
        )
    return summary, checks


def _assert_parity(
    native: dict,
    truth: dict,
    official: dict | None,
    case: dict,
    thresholds: dict,
    record=None,
) -> dict:
    """Evaluate every check, hand the rows to ``record`` first, then assert them in order."""
    summary, checks = _parity_checks(native, truth, official, case, thresholds)
    if record is not None:
        record([row for row, _ in checks])
    for row, message in checks:
        assert row["passed"], message
    return summary


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence(
        "inputs",
        {"manifest": manifest, "case": case, "query": "\n\n".join(case["queries"])},
    )
    model_dir, pinned = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest["hf_id"],
            "hf_revision": manifest["hf_revision"],
            "identity": (
                "pinned Hugging Face snapshot"
                if pinned
                else "TRTMC_MINISTRAL3_MODEL_DIR override, not verified against hf_revision"
            ),
        },
    )
    binary, runtime_root, native_build = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        native = _native(binary, runtime_root, native_build, bundle, case)
    with evidence_stage("reference"):
        truth = _official_reference(model_dir, case, "fp32")
        official = (
            _official_reference(model_dir, case, "bf16")
            if manifest["precision"] == "bf16"
            else None
        )
    with evidence_stage("compare"):
        first = native["tokens"][0]
        record_evidence(
            "native",
            {
                "embeddings": native["embeddings"],
                "token_ids": first["token_ids"],
                "byte_offsets": first["byte_offsets"],
                "pooled_features": native["pooled"],
            },
        )
        record_evidence(
            "reference",
            {
                "source": "Transformers AutoModel in FP32, TF32 off: the checkpoint's documented recipe",
                "embeddings": truth["embeddings"],
                "reference_ids": truth["tokens"][0]["token_ids"],
            },
        )
        native_file = tmp_path / "native_embeddings.npy"
        reference_file = tmp_path / "reference_embeddings.npy"
        np.save(native_file, native["embeddings"])
        np.save(reference_file, truth["embeddings"])
        label = f"{manifest['precision'].upper()} bundle against the FP32 Transformers recipe" + (
            " and the official BF16 run" if official is not None else ""
        )

        def record(checks: list) -> None:
            record_evidence(
                "reference_comparison",
                {
                    "label": label,
                    "scope": "independent_reference",
                    "enforced": True,
                    "native": native_file,
                    "reference": reference_file,
                    "checks": checks,
                },
            )

        summary = _assert_parity(
            native,
            truth,
            official,
            case,
            record_evidence("thresholds", _thresholds(case_name)),
            record=record,
        )
        record_evidence("compare", summary)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2-owned Accuracy executor.

The high-level qualification application treats the suite definition as opaque.
This module owns the MMLU input interpretation, Hugging Face reference, token
comparison, and gate used by GPT-2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUEST_SCHEMA = "trtmc.qualification-executor-request/v1"
RESULT_SCHEMA = "trtmc.qualification-result/v1"
REFERENCE_REQUEST_SCHEMA = "trtmc.gpt2-reference-request/v1"
REFERENCE_RESULT_SCHEMA = "trtmc.gpt2-reference-result/v1"


class Gpt2QualificationError(RuntimeError):
    """The GPT-2 Accuracy suite cannot be executed correctly."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.reference:
        return _reference_main(arguments.request, arguments.output)
    return _qualification_main(arguments.request, arguments.output)


def _qualification_main(request_path: Path, output_path: Path) -> int:
    try:
        request = _read_json(request_path, "executor request")
        item = _item(request)
        result = _execute(request, item, output_path.parent.resolve())
        _write_json(output_path, result)
        return 0
    except Exception as error:
        try:
            request = _read_json(request_path, "executor request")
            item = _item(request)
            _write_json(
                output_path,
                _error_result(item, str(error), output_path.parent.resolve()),
            )
        except Exception:
            print(
                f"GPT-2 qualification failed before result identity was available: {error}",
                file=sys.stderr,
            )
        return 1


def _execute(request: Mapping[str, Any], item: Mapping[str, Any], item_dir: Path) -> dict[str, Any]:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise Gpt2QualificationError(f"request schema_version must be {REQUEST_SCHEMA}")
    if item.get("family") != "gpt2" or item.get("kind") != "accuracy":
        raise Gpt2QualificationError("the GPT-2 executor only accepts gpt2 Accuracy items")
    definition = _mapping(item.get("definition"), "suite definition")
    if definition.get("implementation") != "mmlu_continuation_parity":
        raise Gpt2QualificationError(
            f"unsupported GPT-2 Accuracy implementation {definition.get('implementation')!r}"
        )
    case = _mapping(item.get("case"), "case")
    environment = _mapping(request.get("environment"), "environment")
    if environment.get("schema_version") != "trtmc.qualification-environment/v1":
        raise Gpt2QualificationError("unsupported qualification environment")

    manifest = _read_json(Path(str(item["manifest_path"])), "model manifest")
    if manifest.get("name") != item.get("model") or manifest.get("family") != "gpt2":
        raise Gpt2QualificationError("plan item and GPT-2 manifest do not match")
    samples, dataset_path = _load_samples(definition, case, environment)
    _write_jsonl(item_dir / "selected-samples.jsonl", samples)

    reference = _run_reference(
        manifest=manifest,
        definition=definition,
        case=case,
        samples=samples,
        environment=environment,
        item_dir=item_dir,
    )
    candidate = _run_candidate(
        request=request,
        item=item,
        case=case,
        reference=reference,
        environment=environment,
        item_dir=item_dir,
    )
    comparison = _compare(definition, case, item, reference, candidate)
    _write_jsonl(item_dir / "samples.jsonl", comparison["samples"])
    _write_jsonl(item_dir / "disagreements.jsonl", comparison["disagreements"])

    gate_policy = str(item["gate_policy"])
    verdict = comparison["verdict"] if gate_policy == "blocking" else None
    artifacts = [
        _artifact("selected samples", "selected-samples.jsonl"),
        _artifact("sample comparisons", "samples.jsonl"),
        _artifact("disagreements", "disagreements.jsonl"),
        _artifact("candidate request", "candidate-spec.json"),
        _artifact("candidate result", "candidate/result.json"),
        _artifact("candidate stdout", "candidate.stdout.log"),
        _artifact("candidate stderr", "candidate.stderr.log"),
        _artifact("reference request", "reference-request.json"),
        _artifact("reference result", "reference.json"),
        _artifact("reference stdout", "reference.stdout.log"),
        _artifact("reference stderr", "reference.stderr.log"),
    ]
    return {
        **_identity(item),
        "schema_version": RESULT_SCHEMA,
        "execution": "completed",
        "verdict": verdict,
        "details": {
            "gate_policy": gate_policy,
            "dataset": {
                "path": str(dataset_path),
                "version": _mapping(definition.get("dataset"), "suite dataset").get("version"),
            },
            "reference": {
                "model": reference.get("model"),
                "revision": reference.get("revision"),
                "precision": reference.get("precision"),
            },
            "actual_sample_count": comparison["actual_sample_count"],
            "passed_sample_count": comparison["passed_sample_count"],
            "failed_sample_count": comparison["failed_sample_count"],
            "allowed_failure_count": comparison["allowed_failure_count"],
            "metrics": comparison["metrics"],
            "gate_evaluations": comparison["gate_evaluations"],
        },
        "artifacts": artifacts,
    }


def _load_samples(
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    storage = _mapping(environment.get("storage"), "environment storage")
    data_root = _path(storage.get("data_root"), "storage.data_root")
    dataset = _mapping(definition.get("dataset"), "suite dataset")
    relative = Path(_string(dataset.get("relative_path"), "dataset.relative_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise Gpt2QualificationError("dataset.relative_path must stay below data_root")
    dataset_path = (data_root / relative).resolve()
    try:
        dataset_path.relative_to(data_root)
    except ValueError as error:
        raise Gpt2QualificationError("dataset path escapes data_root") from error
    if not dataset_path.is_file():
        raise Gpt2QualificationError(f"MMLU dataset does not exist: {dataset_path}")
    payload = _read_json(dataset_path, "MMLU dataset")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise Gpt2QualificationError("MMLU dataset must contain a requests list")
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for index, value in enumerate(requests):
        if not isinstance(value, Mapping):
            raise Gpt2QualificationError(f"MMLU request {index} must be an object")
        indexed.append((index, value))

    selection = _mapping(definition.get("selection", {}), "suite selection")
    method = selection.get("method", "first")
    if method == "seeded":
        seed = selection.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Gpt2QualificationError("seeded selection requires an integer seed")
        random.Random(seed).shuffle(indexed)
    elif method != "first":
        raise Gpt2QualificationError(f"unsupported GPT-2 sample selection {method!r}")
    sample_limit = case.get("sample_limit")
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int):
        raise Gpt2QualificationError("case sample_limit must be an integer")
    if sample_limit == 0 or sample_limit < -1:
        raise Gpt2QualificationError("case sample_limit must be -1 or positive")
    if sample_limit > 0:
        indexed = indexed[:sample_limit]
    if not indexed:
        raise Gpt2QualificationError("MMLU selection contains no samples")

    result = []
    seen_ids: set[str] = set()
    for index, value in indexed:
        sample_id = str(value.get("id") or value.get("sample_id") or f"mmlu_{index:06d}")
        if sample_id in seen_ids:
            raise Gpt2QualificationError(f"duplicate MMLU sample id {sample_id!r}")
        seen_ids.add(sample_id)
        result.append(
            {
                "sample_id": sample_id,
                "dataset_index": index,
                "subject": str(value.get("subject", "")),
                "answer": str(value.get("answer", "")),
                "prompt": _request_prompt(value),
            }
        )
    return result, dataset_path


def _request_prompt(request: Mapping[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and str(message["content"]).strip()
            ):
                return str(message["content"])
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    raise Gpt2QualificationError("MMLU request has neither a prompt nor a user message")


def _run_reference(
    *,
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    item_dir: Path,
) -> dict[str, Any]:
    tools = _mapping(environment.get("tools"), "environment tools")
    execution = _mapping(environment.get("execution", {}), "environment execution")
    python = Path(str(tools.get("reference_python") or sys.executable)).expanduser().resolve()
    if not python.is_file():
        raise Gpt2QualificationError(f"reference Python does not exist: {python}")
    reference = _mapping(case.get("reference"), "case reference")
    if reference.get("implementation") != "hf_transformers":
        raise Gpt2QualificationError("unsupported GPT-2 reference implementation")
    prompt = _mapping(case.get("prompt"), "case prompt")
    payload = {
        "schema_version": REFERENCE_REQUEST_SCHEMA,
        "model": _string(manifest.get("hf_id"), "manifest hf_id"),
        "revision": manifest.get("hf_revision") or None,
        "trust_remote_code": bool(manifest.get("trust_remote_code", False)),
        "precision": _string(reference.get("precision", "fp32"), "reference precision"),
        "prompt_token_limit": int(prompt.get("token_limit", 960)),
        "prompt_truncation_side": str(prompt.get("truncation_side", "left")),
        "local_files_only": bool(execution.get("local_files_only", False)),
        "device": str(execution.get("reference_device", "auto")),
        "generation": dict(_candidate_request(case)),
        "samples": [dict(sample) for sample in samples],
    }
    request_path = item_dir / "reference-request.json"
    result_path = item_dir / "reference.json"
    stdout_path = item_dir / "reference.stdout.log"
    stderr_path = item_dir / "reference.stderr.log"
    _write_json(request_path, payload)
    timeout = _timeout(environment)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            [
                str(python),
                str(Path(__file__).resolve()),
                "--reference",
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ],
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=timeout,
        )
    if completed.returncode != 0:
        raise Gpt2QualificationError(
            f"GPT-2 reference exited {completed.returncode}; see {stderr_path.name}"
        )
    result = _read_json(result_path, "GPT-2 reference result")
    if result.get("schema_version") != REFERENCE_RESULT_SCHEMA:
        raise Gpt2QualificationError("GPT-2 reference returned an unsupported result")
    return result


def _reference_main(request_path: Path, output_path: Path) -> int:
    try:
        request = _read_json(request_path, "GPT-2 reference request")
        if request.get("schema_version") != REFERENCE_REQUEST_SCHEMA:
            raise Gpt2QualificationError("unsupported GPT-2 reference request")
        result = _generate_reference(request)
        _write_json(output_path, result)
        return 0
    except Exception as error:
        print(f"GPT-2 reference failed: {error}", file=sys.stderr)
        return 1


def _generate_reference(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = str(request["precision"])
    dtypes = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise Gpt2QualificationError(f"unsupported GPT-2 reference precision {precision!r}")
    device_name = str(request.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    common = {
        "revision": request.get("revision"),
        "trust_remote_code": bool(request.get("trust_remote_code", False)),
        "local_files_only": bool(request.get("local_files_only", False)),
    }
    common = {name: value for name, value in common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(str(request["model"]), **common)
    model = AutoModelForCausalLM.from_pretrained(
        str(request["model"]), torch_dtype=dtypes[precision], **common
    ).eval()
    model.to(device_name)
    generation = _mapping(request.get("generation"), "reference generation")
    limit = int(request["prompt_token_limit"])
    outputs = []
    with torch.inference_mode():
        for sample in request["samples"]:
            effective_prompt = _truncate_prompt(
                tokenizer,
                str(sample["prompt"]),
                limit,
                str(request["prompt_truncation_side"]),
            )
            encoded = tokenizer(effective_prompt, return_tensors="pt")
            input_token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
            encoded = {name: value.to(device_name) for name, value in encoded.items()}
            options = {
                "max_new_tokens": int(generation.get("max_new_tokens", 64)),
                "do_sample": bool(generation.get("do_sample", False)),
                "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
                "num_beams": 1,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if options["do_sample"]:
                options.update(
                    temperature=float(generation.get("temperature", 1.0)),
                    top_k=int(generation.get("top_k", 0)),
                    top_p=float(generation.get("top_p", 1.0)),
                )
                seed = int(generation.get("seed", -1))
                if seed >= 0:
                    torch.manual_seed(seed)
            generation_result = model.generate(**encoded, **options)
            prompt_length = int(encoded["input_ids"].shape[-1])
            token_ids = generation_result.sequences[0, prompt_length:].detach().to("cpu").tolist()
            max_score_token_ids = _generated_token_max_score_ids(generation_result.scores)
            outputs.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "prompt": effective_prompt,
                    "input_token_ids": input_token_ids,
                    "token_ids": [int(value) for value in token_ids],
                    "generated_token_max_score_ids": max_score_token_ids,
                    "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            )
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return {
        "schema_version": REFERENCE_RESULT_SCHEMA,
        "model": request["model"],
        "revision": revision or request.get("revision"),
        "precision": precision,
        "samples": outputs,
    }


def _truncate_prompt(tokenizer: Any, prompt: str, limit: int, side: str) -> str:
    if limit < 1:
        raise Gpt2QualificationError("reference prompt_token_limit must be positive")
    if side not in {"left", "right"}:
        raise Gpt2QualificationError("reference prompt_truncation_side must be left or right")
    token_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
    if len(token_ids) <= limit:
        return prompt
    selected = token_ids[-limit:] if side == "left" else token_ids[:limit]
    return str(
        tokenizer.decode(
            selected,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _generated_token_max_score_ids(scores: Sequence[Any]) -> list[list[int]]:
    result = []
    for scores_at_step in scores:
        row = scores_at_step[0]
        maximum = row.max()
        result.append(
            [
                int(value)
                for value in (row == maximum)
                .nonzero(as_tuple=False)
                .reshape(-1)
                .detach()
                .to("cpu")
                .tolist()
            ]
        )
    return result


def _run_candidate(
    *,
    request: Mapping[str, Any],
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    reference: Mapping[str, Any],
    environment: Mapping[str, Any],
    item_dir: Path,
) -> dict[str, Any]:
    tools = _mapping(environment.get("tools"), "environment tools")
    storage = _mapping(environment.get("storage"), "environment storage")
    execution = _mapping(environment.get("execution", {}), "environment execution")
    executable = _path(tools.get("trtmc_bench"), "tools.trtmc_bench")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise Gpt2QualificationError(f"trtmc-bench is not executable: {executable}")
    runtime_root = _path(storage.get("runtime_root"), "storage.runtime_root")
    candidate_request = _candidate_request(case)
    testcase = _string(
        _mapping(case.get("candidate"), "case candidate").get("testcase"),
        "candidate.testcase",
    )
    reference_samples = reference.get("samples")
    if not isinstance(reference_samples, list) or not reference_samples:
        raise Gpt2QualificationError("GPT-2 reference produced no samples")
    spec = {
        "models": [
            {
                "model": item["model"],
                "cases": [
                    {
                        "name": str(sample["sample_id"]),
                        "testcase": testcase,
                        "request": {**candidate_request, "prompt": str(sample["prompt"])},
                        "measurement": {"warmup": 0, "iterations": 1},
                        "telemetry": {"gpu": "off"},
                    }
                    for sample in reference_samples
                ],
            }
        ]
    }
    spec_path = item_dir / "candidate-spec.json"
    output_dir = item_dir / "candidate"
    stdout_path = item_dir / "candidate.stdout.log"
    stderr_path = item_dir / "candidate.stderr.log"
    _write_json(spec_path, spec)
    command = [
        str(executable),
        "run",
        str(spec_path),
        "--manifest-root",
        str(request["families_root"]),
        "--runtime-root",
        str(runtime_root),
        "-o",
        str(output_dir),
    ]
    worker = tools.get("trtmc_worker")
    if worker:
        command.extend(("--worker", str(_path(worker, "tools.trtmc_worker"))))
    bundle_cache = storage.get("bundle_cache")
    if bundle_cache:
        command.extend(("--bundle-cache", str(_path(bundle_cache, "storage.bundle_cache"))))
    roots = storage.get("bundle_roots", [])
    if not isinstance(roots, list) or not all(isinstance(value, str) for value in roots):
        raise Gpt2QualificationError("storage.bundle_roots must be a path list")
    for root in roots:
        command.extend(("--bundle-root", str(_path(root, "storage.bundle_roots"))))
    if not bool(execution.get("allow_build", False)):
        command.append("--no-build")
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=_timeout(environment),
        )
    if completed.returncode != 0:
        raise Gpt2QualificationError(
            f"GPT-2 candidate exited {completed.returncode}; see {stderr_path.name}"
        )
    result = _read_json(output_dir / "result.json", "candidate result")
    if result.get("schema_version") != "trtmc.benchmark-run/v2":
        raise Gpt2QualificationError("GPT-2 candidate returned an unsupported result")
    cells = result.get("cells")
    if not isinstance(cells, list) or len(cells) != len(reference_samples):
        raise Gpt2QualificationError("candidate result count does not match selected samples")
    if any(not isinstance(cell, Mapping) or cell.get("status") != "completed" for cell in cells):
        raise Gpt2QualificationError("one or more GPT-2 candidate samples failed")
    return result


def _candidate_request(case: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _mapping(case.get("candidate"), "case candidate")
    request = _mapping(candidate.get("request"), "candidate request")
    return dict(request)


def _compare(
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    item: Mapping[str, Any],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_samples = reference["samples"]
    cells = candidate["cells"]
    rows = []
    disagreements = []
    for reference_sample, cell in zip(reference_samples, cells, strict=True):
        sample_id = str(reference_sample["sample_id"])
        if cell.get("name") != sample_id:
            raise Gpt2QualificationError(
                f"candidate sample {cell.get('name')!r} does not match {sample_id!r}"
            )
        summary = _mapping(cell.get("output_summary"), "candidate output summary")
        candidate_ids = summary.get("token_ids")
        reference_ids = reference_sample.get("token_ids")
        if not isinstance(candidate_ids, list) or not isinstance(reference_ids, list):
            raise Gpt2QualificationError("token comparison requires token_ids from both sides")
        candidate_ids = [int(value) for value in candidate_ids]
        reference_ids = [int(value) for value in reference_ids]
        exact = candidate_ids == reference_ids
        first_divergence = _first_divergence(candidate_ids, reference_ids)
        max_score_token_ids = reference_sample.get("generated_token_max_score_ids", [])
        tie_ids: list[int] = []
        if (
            not exact
            and isinstance(max_score_token_ids, list)
            and first_divergence < len(max_score_token_ids)
            and isinstance(max_score_token_ids[first_divergence], list)
        ):
            tie_ids = [int(value) for value in max_score_token_ids[first_divergence]]
        tie_equivalent = (
            not exact
            and len(tie_ids) > 1
            and first_divergence < len(candidate_ids)
            and first_divergence < len(reference_ids)
            and candidate_ids[first_divergence] in tie_ids
            and reference_ids[first_divergence] in tie_ids
        )
        passed = exact or tie_equivalent
        row = {
            "sample_id": sample_id,
            "passed": passed,
            "exact": exact,
            "reference_tie_equivalent": tie_equivalent,
            "candidate_token_ids": candidate_ids,
            "reference_token_ids": reference_ids,
            "candidate_text": str(summary.get("text", "")),
            "reference_text": str(reference_sample.get("text", "")),
            "first_divergence": first_divergence,
            "max_score_token_ids_at_first_divergence": tie_ids,
        }
        rows.append(row)
        if not exact:
            disagreements.append(row)

    scoring = _mapping(definition.get("scoring"), "suite scoring")
    if scoring.get("implementation") != "continuation_token_parity":
        raise Gpt2QualificationError("unsupported GPT-2 scoring implementation")
    acceptance = _mapping(case.get("gate"), "case gate")
    min_pass_rate = float(acceptance.get("min_pass_rate"))
    min_allowed_failures = int(acceptance.get("min_allowed_failures", 0))
    if not 0.0 <= min_pass_rate <= 1.0 or min_allowed_failures < 0:
        raise Gpt2QualificationError("invalid GPT-2 sample acceptance")
    sample_count = len(rows)
    exact_count = sum(bool(row["exact"]) for row in rows)
    tie_equivalent_count = sum(bool(row["reference_tie_equivalent"]) for row in rows)
    failed_count = sum(not bool(row["passed"]) for row in rows)
    passed_count = sample_count - failed_count
    matched_prefix_tokens = sum(int(row["first_divergence"]) for row in rows)
    compared_tokens = sum(
        max(len(row["candidate_token_ids"]), len(row["reference_token_ids"]), 1) for row in rows
    )
    allowed = max(math.floor((1.0 - min_pass_rate) * sample_count), min_allowed_failures)
    gate_passed = failed_count <= allowed
    return {
        "verdict": "pass" if gate_passed else "fail",
        "actual_sample_count": sample_count,
        "passed_sample_count": passed_count,
        "failed_sample_count": failed_count,
        "allowed_failure_count": allowed,
        "metrics": {
            "exact_token_match_rate": exact_count / sample_count,
            "tie_adjusted_exact_match_rate": passed_count / sample_count,
            "token_prefix_agreement": matched_prefix_tokens / compared_tokens,
            "divergent_count": sample_count - exact_count,
            "reference_tie_equivalent_count": tie_equivalent_count,
        },
        "gate_evaluations": [
            {
                "name": "sample_acceptance",
                "operator": "<=",
                "threshold": allowed,
                "actual": failed_count,
                "passed": gate_passed,
            }
        ],
        "samples": rows,
        "disagreements": disagreements,
        "gate_policy": item["gate_policy"],
    }


def _first_divergence(candidate: Sequence[int], reference: Sequence[int]) -> int:
    for index, (left, right) in enumerate(zip(candidate, reference, strict=False)):
        if left != right:
            return index
    return min(len(candidate), len(reference))


def _item(request: Mapping[str, Any]) -> dict[str, Any]:
    item = request.get("plan_item")
    if not isinstance(item, dict):
        raise Gpt2QualificationError("executor request requires a plan_item object")
    required = (
        "id",
        "family",
        "model",
        "kind",
        "suite_id",
        "case_id",
        "gate_policy",
        "manifest_path",
    )
    if any(not isinstance(item.get(name), str) or not item[name] for name in required):
        raise Gpt2QualificationError("plan item identity is incomplete")
    return item


def _identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_item_id": item["id"],
        "family": item["family"],
        "model": item["model"],
        "kind": item["kind"],
        "suite_id": item["suite_id"],
        "case_id": item["case_id"],
    }


def _error_result(
    item: Mapping[str, Any], message: str, item_dir: Path | None = None
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        **_identity(item),
        "execution": "error",
        "verdict": None,
        "details": {"error": message},
        "artifacts": _existing_artifacts(item_dir) if item_dir else [],
    }


def _existing_artifacts(item_dir: Path) -> list[dict[str, str]]:
    known = (
        ("selected samples", "selected-samples.jsonl"),
        ("sample comparisons", "samples.jsonl"),
        ("disagreements", "disagreements.jsonl"),
        ("candidate request", "candidate-spec.json"),
        ("candidate result", "candidate/result.json"),
        ("candidate stdout", "candidate.stdout.log"),
        ("candidate stderr", "candidate.stderr.log"),
        ("reference request", "reference-request.json"),
        ("reference result", "reference.json"),
        ("reference stdout", "reference.stdout.log"),
        ("reference stderr", "reference.stderr.log"),
    )
    return [
        _artifact(label, relative)
        for label, relative in known
        if not (item_dir / relative).is_symlink() and (item_dir / relative).is_file()
    ]


def _artifact(label: str, path: str) -> dict[str, str]:
    return {"label": label, "path": path}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Gpt2QualificationError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Gpt2QualificationError(f"{field} must be a non-empty string")
    return value


def _path(value: Any, field: str) -> Path:
    path = Path(_string(value, field)).expanduser().resolve()
    return path


def _timeout(environment: Mapping[str, Any]) -> int:
    execution = _mapping(environment.get("execution", {}), "environment execution")
    value = execution.get("timeout_seconds", 7200)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Gpt2QualificationError("execution.timeout_seconds must be positive")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gpt2QualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Gpt2QualificationError(f"{label} must contain an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

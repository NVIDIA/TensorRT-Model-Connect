# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned VBench prompt slice for MiniMax-H3 task-quality scoring.

The resulting dataset owns a deterministic 100-prompt slice and records the
exact VBench and MiniMax-H3 tokenizer inputs used to create it. The companion
SigLIP scorer is a TRTMC candidate-only proxy; it is not an official VBench
aggregate score.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


VBENCH_REPOSITORY = "https://github.com/Vchitect/VBench.git"
VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
VBENCH_INFO_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
VBENCH_LICENSE = "Apache-2.0"
VBENCH_LICENSE_SHA256 = "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
EXPECTED_SOURCE_COUNT = 946
SELECTION_DIMENSIONS = (
    "motion_smoothness",
    "dynamic_degree",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
)
PROMPTS_PER_DIMENSION = 10
EXPECTED_PROMPT_COUNT = len(SELECTION_DIMENSIONS) * PROMPTS_PER_DIMENSION

MINIMAX_H3_MODEL = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
TOKENIZER_JSON_SHA256 = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
MAX_PROMPT_TOKENS = 537


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_source_rows(source_info: Path) -> list[dict[str, Any]]:
    if _sha256(source_info) != VBENCH_INFO_SHA256:
        raise ValueError("VBench_full_info.json does not match the pinned revision")
    raw = json.loads(source_info.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_COUNT} VBench records, found "
            f"{len(raw) if isinstance(raw, list) else 'non-list input'}"
        )

    rows = []
    for source_index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ValueError(f"VBench row {source_index} must be an object")
        prompt = value.get("prompt_en")
        dimensions = value.get("dimension")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"VBench row {source_index} has no prompt_en")
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or not all(isinstance(item, str) and item for item in dimensions)
        ):
            raise ValueError(f"VBench row {source_index} has invalid dimensions")
        rows.append(
            {
                "source_index": source_index,
                "prompt": prompt.strip(),
                "source_dimensions": list(dimensions),
            }
        )
    return rows


def _select_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen_prompts: set[str] = set()
    for dimension in SELECTION_DIMENSIONS:
        dimension_rows = []
        for row in rows:
            prompt = str(row["prompt"])
            if dimension not in row["source_dimensions"] or prompt in seen_prompts:
                continue
            selected_row = dict(row)
            selected_row["selection_dimension"] = dimension
            dimension_rows.append(selected_row)
            seen_prompts.add(prompt)
            if len(dimension_rows) == PROMPTS_PER_DIMENSION:
                break
        if len(dimension_rows) != PROMPTS_PER_DIMENSION:
            raise ValueError(
                f"VBench dimension {dimension!r} yielded {len(dimension_rows)} unique "
                f"prompts; expected {PROMPTS_PER_DIMENSION}"
            )
        selected.extend(dimension_rows)
    if len(selected) != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            f"selected {len(selected)} VBench prompts; expected {EXPECTED_PROMPT_COUNT}"
        )
    return selected


def _load_tokenizer(tokenizer_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=True,
    )


def _token_count(tokenizer: Any, prompt: str) -> int:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("MiniMax-H3 tokenizer.encode must return a token sequence")
    return len(token_ids)


def _annotate_token_counts(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    annotated = []
    for source_row in rows:
        row = dict(source_row)
        count = _token_count(tokenizer, str(row["prompt"]))
        if count < 1 or count > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"VBench row {row['source_index']} token count {count} is outside "
                f"MiniMax-H3 [1, {MAX_PROMPT_TOKENS}]"
            )
        row["token_count"] = count
        annotated.append(row)
    return annotated


def _tokenizer_manifest(tokenizer_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(tokenizer_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(tokenizer_dir).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not files:
        raise ValueError(f"MiniMax-H3 tokenizer directory is empty: {tokenizer_dir}")
    return files


def prepare_vbench_siglip(
    source_info: Path,
    source_license: Path,
    tokenizer_dir: Path,
    output_root: Path,
    *,
    tokenizer_loader: Callable[[Path], Any] = _load_tokenizer,
) -> Path:
    """Create a deterministic, fail-closed VBench/SigLIP dataset."""

    source_info = source_info.resolve(strict=True)
    source_license = source_license.resolve(strict=True)
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    if tokenizer_dir.parent.name != MINIMAX_H3_REVISION:
        raise ValueError(
            "tokenizer-dir must be the tokenizer subdirectory of the pinned "
            f"MiniMax-H3 snapshot {MINIMAX_H3_REVISION}"
        )
    if _sha256(source_license) != VBENCH_LICENSE_SHA256:
        raise ValueError("VBench LICENSE does not match the pinned revision")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if _sha256(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise ValueError("MiniMax-H3 tokenizer.json does not match the pinned model revision")

    rows = _annotate_token_counts(
        _select_rows(_load_source_rows(source_info)),
        tokenizer_loader(tokenizer_dir),
    )
    tokenizer_files = _tokenizer_manifest(tokenizer_dir)
    output_root.mkdir(parents=True)
    upstream_info = output_root / "upstream" / "VBench_full_info.json"
    upstream_info.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_info, upstream_info)
    license_output = output_root / "licenses" / "VBENCH_LICENSE"
    license_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_license, license_output)

    requests = []
    for dataset_index, row in enumerate(rows):
        source_index = int(row["source_index"])
        dimension = str(row["selection_dimension"])
        prompt = str(row["prompt"])
        prompt_relative = Path("prompts") / f"{dataset_index:03d}-{dimension}.json"
        _write_json(output_root / prompt_relative, {"prompt": prompt, "seed": 0})
        requests.append(
            {
                "sample_id": f"vbench-{dataset_index:03d}-{source_index:03d}",
                "dataset_index": dataset_index,
                "testcase": "minimax-h3-768p",
                "stage": "end_to_end",
                "category": f"vbench-siglip,{dimension}",
                "prompt": prompt,
                "token_count": int(row["token_count"]),
                "selection_dimension": dimension,
                "source_dimensions": list(row["source_dimensions"]),
                "source_index": source_index,
                "inputs": {
                    "prompt_file": prompt_relative.as_posix(),
                    "validation_mode": "vbench_siglip",
                },
            }
        )

    token_counts = [int(row["token_count"]) for row in rows]
    dataset_name = "VBench MiniMax-H3 candidate task-quality proxy"
    dataset_path = output_root / "dataset.json"
    _write_json(
        dataset_path,
        {
            "schema_version": "trtmc.model-plugin-validation/v1",
            "dataset": dataset_name,
            "version": f"{VBENCH_REVISION}-minimax-h3-siglip-v1",
            "source": VBENCH_REPOSITORY,
            "source_revision": VBENCH_REVISION,
            "license": VBENCH_LICENSE,
            "model": MINIMAX_H3_MODEL,
            "model_revision": MINIMAX_H3_REVISION,
            "validation_scope": (
                "candidate-only TRTMC SigLIP/temporal proxy over a fixed VBench "
                "prompt slice; not an official VBench score or aggregate"
            ),
            "sampling": (
                "first 10 globally unique prompts in source order for each of 10 "
                "ordered VBench dimensions"
            ),
            "selection_dimensions": list(SELECTION_DIMENSIONS),
            "prompts_per_dimension": PROMPTS_PER_DIMENSION,
            "request_count": len(requests),
            "token_count": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "allowed_maximum": MAX_PROMPT_TOKENS,
            },
            "requests": requests,
        },
    )
    _write_json(
        output_root / "provenance" / "SOURCE.json",
        {
            "source_repository": VBENCH_REPOSITORY,
            "source_revision": VBENCH_REVISION,
            "source_file": {
                "path": "VBench_full_info.json",
                "sha256": VBENCH_INFO_SHA256,
            },
            "selection_dimensions": list(SELECTION_DIMENSIONS),
            "prompts_per_dimension": PROMPTS_PER_DIMENSION,
            "model": MINIMAX_H3_MODEL,
            "model_revision": MINIMAX_H3_REVISION,
            "tokenizer_file": {
                "path": "tokenizer.json",
                "sha256": TOKENIZER_JSON_SHA256,
            },
            "prompt_count": len(requests),
            "prompt_token_count": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "allowed_minimum": 1,
                "allowed_maximum": MAX_PROMPT_TOKENS,
            },
        },
    )

    generated_paths = sorted(path for path in output_root.rglob("*") if path.is_file())
    _write_json(
        output_root / "DATASET_MANIFEST.json",
        {
            "schema_version": "trtmc.dataset-manifest/v1",
            "dataset": dataset_name,
            "source": {
                "repository": VBENCH_REPOSITORY,
                "revision": VBENCH_REVISION,
                "info_sha256": VBENCH_INFO_SHA256,
                "license": VBENCH_LICENSE,
                "license_sha256": VBENCH_LICENSE_SHA256,
            },
            "tokenizer": {
                "model": MINIMAX_H3_MODEL,
                "revision": MINIMAX_H3_REVISION,
                "files": tokenizer_files,
            },
            "request_count": len(requests),
            "path_policy": "manifest_relative",
            "files": [
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in generated_paths
            ],
        },
    )
    return dataset_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--source-license", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    dataset = prepare_vbench_siglip(
        arguments.source_info,
        arguments.source_license,
        arguments.tokenizer_dir,
        arguments.output_root,
    )
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

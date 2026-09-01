# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned AVGen-Bench prompt set for MiniMax-H3 task accuracy.

The resulting dataset covers the official AVGen-Bench visual-quality component
over all 235 prompts. It intentionally excludes audio, AV-sync, lip-sync, and
the aggregate Total and Basic scores because TRTMC currently exports video only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


AVGEN_REPOSITORY = "https://github.com/NVIDIA/AVGen-Bench.git"
AVGEN_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
AVGEN_PROMPTS_TREE = "0ab7c2572f523df1db6cb0170d64be23b9747d12"
AVGEN_LICENSE = "MIT"
AVGEN_LICENSE_SHA256 = "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
MINIMAX_H3_MODEL = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
TOKENIZER_JSON_SHA256 = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
MAX_PROMPT_TOKENS = 537

CATEGORY_COUNTS = {
    "ads": 20,
    "animals": 20,
    "asmr": 20,
    "chemical_reaction": 20,
    "cooking": 20,
    "gameplays": 20,
    "movie_trailer": 20,
    "musical_instrument_tutorial": 35,
    "news": 20,
    "physical_experiment": 20,
    "sports": 20,
}

SOURCE_SHA256 = {
    "ads": "97a2ef8d5c9038f620c88e4b4e29f1397123ba6b80625094bcf05742f45c7605",
    "animals": "7cdc73742f3f9f01b9a7826fb77c56c0f7f69eef4063c928c3c57782fbd5c640",
    "asmr": "ad194ee510cf80c7444cb1e092950b9eb2814ec1fdf76ae0f8ca698a08fd73d9",
    "chemical_reaction": "9c20b2d481bae75230a89d9c3f4ab62d2c8e98307e9befb49b64d32e4850a662",
    "cooking": "62996c8ea1a1a3af13f1c380c1f326eda19fccdce335b0c92c96bc09db27ab26",
    "gameplays": "7e8b9a646e12f8f19159f497a79e5854163035eeaec81c56b5ad28fb04c9f2b4",
    "movie_trailer": "c4a3d8f6f836d5a65d688dee175f4d4b5f3a12f1ebe990845fac1362cd31036d",
    "musical_instrument_tutorial": "e138f017d6dfc5c77c8424796d70552daaba2b9f6ae125c5289d29d098f63d54",
    "news": "a2766c5ded55133ff177b36136ab79ad28ece90e27e40ea140149b9b28f42e4d",
    "physical_experiment": "e6ce7ba9018fedd8bf1dfc3afb129eb13091dcfe98cc8a1e80d0d2676fb0c1da",
    "sports": "2f04007b151085940d2734ab089c75762d19a6f11f340e267074d0d1db9f63c8",
}

# These are the exact short, median, and long representatives measured with
# the pinned MiniMax-H3 tokenizer over all 235 source prompts.
REPRESENTATIVES = (
    {
        "label": "short",
        "category": "musical_instrument_tutorial",
        "source_index": 8,
        "source_title": "Tambourine: Shake Roll",
        "token_count": 51,
    },
    {
        "label": "median",
        "category": "chemical_reaction",
        "source_index": 9,
        "source_title": "Supercooling of Water (Instant Ice)",
        "token_count": 84,
    },
    {
        "label": "long",
        "category": "movie_trailer",
        "source_index": 0,
        "source_title": "REDLINE PROTOCOL",
        "token_count": 218,
    },
)


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


def _load_source_rows(source_root: Path) -> list[dict[str, Any]]:
    prompts_root = source_root / "prompts"
    rows: list[dict[str, Any]] = []
    for category, expected_count in CATEGORY_COUNTS.items():
        path = prompts_root / f"{category}.json"
        actual_sha256 = _sha256(path)
        if actual_sha256 != SOURCE_SHA256[category]:
            raise ValueError(
                f"{path}: SHA256 {actual_sha256} does not match pinned AVGen-Bench source"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) != expected_count:
            raise ValueError(f"{path}: expected exactly {expected_count} prompt records")
        for source_index, value in enumerate(raw):
            if (
                not isinstance(value, Mapping)
                or not {"content", "prompt"} <= set(value)
                or set(value) - {"content", "prompt", "style"}
            ):
                raise ValueError(
                    f"{path}: prompt {source_index} must contain content, prompt, "
                    "and optional style"
                )
            source_title = value["content"]
            prompt = value["prompt"]
            if not isinstance(source_title, str) or not source_title.strip():
                raise ValueError(f"{path}: prompt {source_index} has no content title")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}: prompt {source_index} is empty")
            row = {
                "category": category,
                "source_index": source_index,
                "source_title": source_title.strip(),
                "prompt": prompt.strip(),
            }
            if "style" in value:
                if not isinstance(value["style"], str) or not value["style"].strip():
                    raise ValueError(f"{path}: prompt {source_index} has an invalid style")
                row["source_style"] = value["style"].strip()
            rows.append(row)
    expected_total = sum(CATEGORY_COUNTS.values())
    if len(rows) != expected_total:
        raise ValueError(f"expected {expected_total} AVGen-Bench prompts, found {len(rows)}")
    return rows


def _validate_source_revision(source_root: Path) -> None:
    checks = (
        ("revision", ["git", "-C", str(source_root), "rev-parse", "HEAD"], AVGEN_REVISION),
        (
            "prompts tree",
            [
                "git",
                "-C",
                str(source_root),
                "rev-parse",
                f"{AVGEN_REVISION}:prompts",
            ],
            AVGEN_PROMPTS_TREE,
        ),
    )
    for label, command, expected in checks:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        actual = completed.stdout.strip()
        if completed.returncode or actual != expected:
            detail = completed.stderr.strip() or actual or "unresolved"
            raise ValueError(f"AVGen-Bench {label} does not match {expected}: {detail}")


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


def _validate_and_annotate(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    by_source: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        count = _token_count(tokenizer, str(row["prompt"]))
        if count < 1 or count > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"{row['category']}[{row['source_index']}] token count {count} "
                f"is outside MiniMax-H3 [1, {MAX_PROMPT_TOKENS}]"
            )
        row["token_count"] = count
        by_source[(str(row["category"]), int(row["source_index"]))] = row

    for expected in REPRESENTATIVES:
        key = (str(expected["category"]), int(expected["source_index"]))
        row = by_source.get(key)
        if row is None:
            raise ValueError(f"missing pinned representative {key[0]}[{key[1]}]")
        for field in ("source_title", "token_count"):
            if row[field] != expected[field]:
                raise ValueError(
                    f"representative {expected['label']} {field} is {row[field]!r}; "
                    f"expected {expected[field]!r}"
                )
        row["representative"] = str(expected["label"])
    return rows


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


def prepare_avgen_bench(
    source_root: Path,
    tokenizer_dir: Path,
    output_root: Path,
    *,
    tokenizer_loader: Callable[[Path], Any] = _load_tokenizer,
    source_verifier: Callable[[Path], None] = _validate_source_revision,
) -> Path:
    """Create a deterministic, fail-closed AVGen-Bench Vis dataset."""
    source_root = source_root.resolve(strict=True)
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    if tokenizer_dir.parent.name != MINIMAX_H3_REVISION:
        raise ValueError(
            "tokenizer-dir must be the tokenizer subdirectory of the pinned "
            f"MiniMax-H3 snapshot {MINIMAX_H3_REVISION}"
        )
    license_path = source_root / "LICENSE"
    if _sha256(license_path) != AVGEN_LICENSE_SHA256:
        raise ValueError("AVGen-Bench LICENSE does not match the pinned source")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    source_verifier(source_root)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if _sha256(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise ValueError("MiniMax-H3 tokenizer.json does not match the pinned model revision")
    rows = _validate_and_annotate(_load_source_rows(source_root), tokenizer_loader(tokenizer_dir))
    tokenizer_files = _tokenizer_manifest(tokenizer_dir)
    output_root.mkdir(parents=True)
    upstream_prompts_root = output_root / "upstream" / "prompts"
    for category in CATEGORY_COUNTS:
        source = source_root / "prompts" / f"{category}.json"
        destination = upstream_prompts_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    license_output = output_root / "licenses" / "AVGEN_BENCH_LICENSE"
    license_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(license_path, license_output)
    requests = []
    for dataset_index, row in enumerate(rows):
        prompt_relative = Path("prompts") / (
            f"{row['category']}-{int(row['source_index']):03d}.json"
        )
        _write_json(
            output_root / prompt_relative,
            {"prompt": row["prompt"], "seed": 0},
        )
        categories = ["avgen-bench", str(row["category"])]
        representative = row.get("representative")
        if representative:
            categories.append(f"representative-{representative}")
        request = {
            "sample_id": f"{row['category']}-{int(row['source_index']):03d}",
            "dataset_index": dataset_index,
            "testcase": "minimax-h3-768p",
            "stage": "end_to_end",
            "category": ",".join(categories),
            "token_count": int(row["token_count"]),
            "source_title": row["source_title"],
            "source_category": row["category"],
            "source_index": int(row["source_index"]),
            "inputs": {
                "prompt_file": prompt_relative.as_posix(),
                "validation_mode": "avgen_vis",
            },
        }
        if row.get("source_style"):
            request["source_style"] = row["source_style"]
        requests.append(request)

    token_counts = [int(row["token_count"]) for row in rows]
    dataset_path = output_root / "dataset.json"
    _write_json(
        dataset_path,
        {
            "schema_version": "trtmc.model-plugin-validation/v1",
            "dataset": "AVGen-Bench MiniMax-H3 Vis task accuracy",
            "version": f"{AVGEN_REVISION}-minimax-h3-video-v1",
            "source": AVGEN_REPOSITORY,
            "source_revision": AVGEN_REVISION,
            "license": AVGEN_LICENSE,
            "model": MINIMAX_H3_MODEL,
            "model_revision": MINIMAX_H3_REVISION,
            "validation_scope": (
                "candidate-only official AVGen-Bench Vis component; excludes audio, "
                "AV-sync, lip-sync, and AVGen aggregate Total/Basic scores"
            ),
            "sampling": "all 235 source prompts in category-file and source-array order",
            "request_count": len(requests),
            "token_count": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "allowed_maximum": MAX_PROMPT_TOKENS,
            },
            "requests": requests,
        },
    )
    source_path = output_root / "provenance" / "SOURCE.json"
    _write_json(
        source_path,
        {
            "source_repository": AVGEN_REPOSITORY,
            "source_revision": AVGEN_REVISION,
            "source_prompts_tree": AVGEN_PROMPTS_TREE,
            "source_prompt_sha256": SOURCE_SHA256,
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
            "dataset": "AVGen-Bench MiniMax-H3 Vis task accuracy",
            "source": {
                "repository": AVGEN_REPOSITORY,
                "revision": AVGEN_REVISION,
                "prompts_tree": AVGEN_PROMPTS_TREE,
                "license": AVGEN_LICENSE,
                "license_sha256": AVGEN_LICENSE_SHA256,
                "prompt_sha256": SOURCE_SHA256,
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    dataset = prepare_avgen_bench(
        arguments.source_root,
        arguments.tokenizer_dir,
        arguments.output_root,
    )
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

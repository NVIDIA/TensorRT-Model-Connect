#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned MIT-licensed Beans dataset for DINOv3 k-NN Accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import urllib.request


DATASET_ID = "AI-Lab-Makerere/beans"
DATASET_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
CLASS_NAMES = ("angular_leaf_spot", "bean_rust", "healthy")
SHARD_SIZE = 16
SOURCES = {
    "train": {
        "filename": "train-00000-of-00001.parquet",
        "sha256": "7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19",
        "size": 143773054,
        "count": 1034,
    },
    "test": {
        "filename": "test-00000-of-00001.parquet",
        "sha256": "534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27",
        "size": 17706070,
        "count": 128,
    },
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f"{path}: expected {size} bytes, got {path.stat().st_size}")
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(f"{path}: expected SHA256 {sha256}, got {actual}")


def _download(split: str, directory: Path) -> Path:
    source = SOURCES[split]
    destination = directory / str(source["filename"])
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/data/{source['filename']}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "trtmc-dinov3-beans/1"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as out,
    ):
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    _verify(destination, size=int(source["size"]), sha256=str(source["sha256"]))
    return destination


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in preparation environments
        raise RuntimeError("Beans preparation requires pyarrow") from exc
    rows = parquet.read_table(path, columns=["image", "labels"]).to_pylist()
    prepared = []
    for index, row in enumerate(rows):
        image = row.get("image")
        label = row.get("labels")
        if not isinstance(image, Mapping) or not isinstance(image.get("bytes"), bytes):
            raise ValueError(f"{path}: row {index} has no embedded image bytes")
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"{path}: row {index} has an invalid label")
        prepared.append(
            {"image_bytes": image["bytes"], "label": label, "source_index": index}
        )
    return prepared


def _write_split(
    output: Path, split: str, rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    samples = []
    image_dir = output / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    for output_index, row in enumerate(rows):
        image_bytes = row.get("image_bytes")
        label = row.get("label")
        source_index = row.get("source_index", output_index)
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError(f"{split} row {output_index} has no image bytes")
        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < len(CLASS_NAMES)
        ):
            raise ValueError(f"{split} row {output_index} has an invalid label")
        relative = Path("images") / split / f"{output_index:06d}.jpg"
        (output / relative).write_bytes(image_bytes)
        samples.append(
            {
                "image": relative.as_posix(),
                "label": label,
                "class_name": CLASS_NAMES[label],
                "source_index": int(source_index),
            }
        )
    return samples


def prepare_records(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    require_official_counts: bool = True,
) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if require_official_counts:
        expected = (int(SOURCES["train"]["count"]), int(SOURCES["test"]["count"]))
        actual = (len(train_rows), len(test_rows))
        if actual != expected:
            raise ValueError(
                f"official Beans split counts must be {expected}, got {actual}"
            )

    train = _write_split(output, "train", train_rows)
    test = _write_split(output, "test", test_rows)
    _json(
        output / "bank.json",
        {
            "schema_version": "trtmc.dinov3-knn-image-manifest/v1",
            "split": "train",
            "class_names": list(CLASS_NAMES),
            "samples": train,
        },
    )
    requests = []
    for shard_index, start in enumerate(range(0, len(test), SHARD_SIZE)):
        shard = []
        for sample in test[start : start + SHARD_SIZE]:
            copied = dict(sample)
            copied["image"] = "../" + str(copied["image"])
            shard.append(copied)
        query_path = Path("queries") / f"test-{shard_index:03d}.json"
        _json(
            output / query_path,
            {
                "schema_version": "trtmc.dinov3-knn-image-manifest/v1",
                "split": "test",
                "class_names": list(CLASS_NAMES),
                "samples": shard,
            },
        )
        requests.append(
            {
                "sample_id": f"beans-test-{start:03d}-{start + len(shard) - 1:03d}",
                "category": "beans_knn_task_accuracy",
                "inputs": {
                    "bank_manifest": "bank.json",
                    "query_manifest": query_path.as_posix(),
                },
            }
        )
    _json(
        output / "dataset.json",
        {
            "schema_version": "trtmc.model-plugin-dataset/v1",
            "dataset": "Beans leaf-disease image classification",
            "version": f"{DATASET_ID}@{DATASET_REVISION}/dinov3-beans-knn-v1",
            "license": "MIT",
            "train_count": len(train),
            "test_count": len(test),
            "requests": requests,
        },
    )
    _json(
        output / "provenance.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": "MIT",
            "source_files": SOURCES,
            "class_names": list(CLASS_NAMES),
            "train_count": len(train),
            "test_count": len(test),
            "selection": "complete official train and test splits in source order",
            "knn": {"ks": [10, 20, 100, 200], "temperature": 0.07},
        },
    )
    (output / "LICENSE.md").write_text(
        "# Dataset license\n\n"
        "Beans is published by AI-Lab-Makerere under the MIT license.\n\n"
        f"Pinned source: https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}\n",
        encoding="utf-8",
    )
    entries = []
    for path in sorted(
        candidate for candidate in output.rglob("*") if candidate.is_file()
    ):
        if path.name == "manifest.sha256.json":
            continue
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    tree_digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _json(
        output / "manifest.sha256.json",
        {
            "schema_version": "trtmc.validation-dataset-manifest/v1",
            "asset_path_policy": "manifest_relative",
            "entries": entries,
            "tree_sha256": tree_digest,
        },
    )
    return output / "dataset.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="directory containing the two pinned Parquet files; downloads when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.source_root:
        source_root = arguments.source_root.resolve()
        paths = {
            split: source_root / str(info["filename"])
            for split, info in SOURCES.items()
        }
        for split, path in paths.items():
            info = SOURCES[split]
            _verify(path, size=int(info["size"]), sha256=str(info["sha256"]))
        prepare_records(
            _parquet_rows(paths["train"]),
            _parquet_rows(paths["test"]),
            arguments.output.resolve(),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="trtmc-dinov3-beans-") as temporary:
            root = Path(temporary)
            paths = {split: _download(split, root) for split in SOURCES}
            prepare_records(
                _parquet_rows(paths["train"]),
                _parquet_rows(paths["test"]),
                arguments.output.resolve(),
            )
    print(arguments.output.resolve() / "dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

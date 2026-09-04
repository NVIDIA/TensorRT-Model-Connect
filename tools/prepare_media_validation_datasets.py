# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare fixed media-generation validation datasets.

The converters consume official benchmark downloads and write only the small
runtime slice needed by validation. They do not run model inference or benchmark
judges. Relative asset paths are resolved by the validation engine.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


VBENCH_REPOSITORY = "https://github.com/Vchitect/VBench.git"
VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
VBENCH_SOURCE = (
    f"https://github.com/Vchitect/VBench/blob/{VBENCH_REVISION}/"
    "vbench/VBench_full_info.json"
)
VBENCH_INFO_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
VBENCH_LICENSE = "Apache-2.0"
VBENCH_MODEL_PLUGIN_DIR = "VBench-fd18b3d-model-plugin-v1"
GEDIT_SOURCE = "https://huggingface.co/datasets/stepfun-ai/GEdit-Bench"
GEDIT_REVISION = "50766778e2a737474c7e9bdf84cdce82c3ea3f4f"
SANA_WM_SOURCE = "https://huggingface.co/datasets/Efficient-Large-Model/SANA-WM-Bench"
SANA_WM_REVISION = "0e5279250e7a531b83ce4e9ee2870fab7003be0b"
VBENCH_DIMENSIONS = (
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
SANA_WM_SPLITS = (
    "benchmark_v2_smooth_60s",
    "benchmark_v2_hard_60s",
)
SANA_WM_ACTIONS = (
    "w-80,jw-40,w-40,lw-60,w-100",
    "w-100,d-40,w-60,a-40,w-80",
    "w-80,l-40,w-80,j-40,w-80",
    "i-40,w-100,k-40,w-100,none-40",
    "wi-60,w-80,l-40,w-80,j-60",
    "none-40,w-120,d-40,w-80,a-40",
    "w-60,j-60,w-80,l-60,w-60",
    "i-30,w-100,k-30,w-100,none-60",
    "d-40,w-80,a-80,w-80,none-40",
    "w-40,wi-80,w-80,wk-80,w-40",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def _select_vbench_requests(source_info: Path, limit: int) -> list[dict[str, Any]]:
    """Select one unique official prompt from each review dimension."""
    raw = json.loads(source_info.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{source_info}: expected a JSON list")
    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for dimension in VBENCH_DIMENSIONS:
        match = next(
            (
                (index, row, str(row.get("prompt_en", "")).strip())
                for index, row in enumerate(raw)
                if isinstance(row, dict)
                and dimension in row.get("dimension", [])
                and str(row.get("prompt_en", "")).strip()
                and str(row.get("prompt_en", "")).strip() not in used_prompts
            ),
            None,
        )
        if match is None:
            raise ValueError(f"{source_info}: no unique prompt for {dimension}")
        source_index, row, prompt = match
        used_prompts.add(prompt)
        selected.append(
            {
                "sample_id": f"vbench_{dimension}_{source_index:06d}",
                "dataset_index": source_index,
                "prompt": prompt,
                "category": dimension,
                "challenge": "official_vbench_prompt_dimension",
                "vbench_dimensions": list(row.get("dimension", [])),
            }
        )
    if limit < 1 or limit > len(selected):
        raise ValueError(f"VBench validation limit must be in [1, {len(selected)}]")
    return selected[:limit]


def prepare_vbench(source_info: Path, output_root: Path, limit: int = 10) -> Path:
    """Write the shared diffusion-runner view of the VBench prompt slice."""
    source_info = source_info.resolve(strict=True)
    if _sha256(source_info) != VBENCH_INFO_SHA256:
        raise ValueError("VBench_full_info.json does not match the pinned revision")
    selected = _select_vbench_requests(source_info, limit)
    return _write_json(
        output_root / "VBench" / "vbench_t2v_task_eval.json",
        {
            "dataset": "VBench text-to-video prompt suite (validation slice)",
            "source": VBENCH_SOURCE,
            "source_info_sha256": _sha256(source_info),
            "source_revision": VBENCH_REVISION,
            "license": VBENCH_LICENSE,
            "sampling": "first unique prompt in each fixed review dimension",
            "request_count": len(selected),
            "requests": selected,
        },
    )


def prepare_vbench_model_plugin_dataset(
    source_info: Path,
    output_root: Path,
    limit: int = 10,
) -> Path:
    """Package the pinned VBench slice for prompt-file model plugins.

    The output is a versioned, portable dataset asset intended for NAS
    publication. It contains no model outputs and runs no external evaluator.
    """
    source_info = source_info.resolve(strict=True)
    if _sha256(source_info) != VBENCH_INFO_SHA256:
        raise ValueError("VBench_full_info.json does not match the pinned revision")

    output_dir = output_root / VBENCH_MODEL_PLUGIN_DIR
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_dir}")
    selected = _select_vbench_requests(source_info, limit)
    output_dir.mkdir(parents=True)

    requests: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        prompt_relative = Path("prompts") / f"{sample_id}.json"
        _write_json(
            output_dir / prompt_relative,
            {"prompt": str(row["prompt"]), "seed": 0},
        )
        requests.append(
            {
                **row,
                "inputs": {"prompt_file": prompt_relative.as_posix()},
            }
        )

    dataset_name = "VBench text-to-video prompt suite (TRTMC model-plugin slice)"
    dataset_path = _write_json(
        output_dir / "dataset.json",
        {
            "schema_version": "trtmc.model-plugin-validation/v1",
            "dataset": dataset_name,
            "version": f"{VBENCH_REVISION}-model-plugin-v1",
            "source": VBENCH_REPOSITORY,
            "source_revision": VBENCH_REVISION,
            "source_info_sha256": VBENCH_INFO_SHA256,
            "license": VBENCH_LICENSE,
            "sampling": "first unique prompt in each fixed review dimension",
            "request_count": len(requests),
            "requests": requests,
        },
    )

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    _write_json(
        output_dir / "DATASET_MANIFEST.json",
        {
            "schema_version": "trtmc.dataset-manifest/v1",
            "dataset": dataset_name,
            "source": {
                "repository": VBENCH_REPOSITORY,
                "revision": VBENCH_REVISION,
                "info_sha256": VBENCH_INFO_SHA256,
                "license": VBENCH_LICENSE,
            },
            "request_count": len(requests),
            "path_policy": "manifest_relative",
            "files": [
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )
    return dataset_path


def _english_gedit_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    for row in rows:
        language = str(row.get("instruction_language", "")).strip().lower()
        if language and language not in {"en", "eng", "english"}:
            continue
        if str(row.get("instruction", "")).strip():
            yield row


def prepare_gedit_rows(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    limit: int = 10,
) -> Path:
    """Write a task-diverse English GEdit slice from loaded dataset rows."""
    if limit < 1:
        raise ValueError("GEdit validation limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    used_tasks: set[str] = set()
    for source_index, row in enumerate(_english_gedit_rows(rows)):
        task_type = str(row.get("task_type", "unknown")).strip() or "unknown"
        if task_type in used_tasks:
            continue
        source_image = row.get("input_image") or row.get("input_image_raw")
        if not isinstance(source_image, Image.Image):
            raise TypeError(
                f"GEdit row {source_index} input_image must be a PIL image, "
                f"got {type(source_image).__name__}"
            )
        key = _safe_name(str(row.get("key", f"{source_index:06d}")))
        image_relative = Path("images") / f"{source_index:06d}_{key}.png"
        image_path = output_dir / image_relative
        normalized = ImageOps.fit(
            source_image.convert("RGB"),
            (1024, 1024),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        normalized.save(image_path)
        selected.append(
            {
                "sample_id": f"gedit_{source_index:06d}_{key}",
                "dataset_index": source_index,
                "prompt": str(row["instruction"]).strip(),
                "image": image_relative.as_posix(),
                "category": task_type,
                "challenge": "real_world_image_edit_instruction",
                "instruction_language": str(row.get("instruction_language", "")),
                "intersection_exist": bool(row.get("Intersection_exist", False)),
                "source_key": str(row.get("key", "")),
                "condition_image_sha256": _sha256(image_path),
            }
        )
        used_tasks.add(task_type)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(
            f"GEdit source provided only {len(selected)} distinct English task types; "
            f"need {limit}"
        )
    return _write_json(
        output_dir / "gedit_bench_task_eval.json",
        {
            "dataset": "GEdit-Bench English task-diverse validation slice",
            "source": GEDIT_SOURCE,
            "source_revision": GEDIT_REVISION,
            "license": "MIT",
            "sampling": "first English row from each distinct task_type",
            "condition_image_transform": (
                "RGB center crop with LANCZOS resampling to 1024x1024 for the "
                "static Qwen-Image-Edit condition geometry"
            ),
            "request_count": len(selected),
            "requests": selected,
        },
    )


def prepare_gedit(source: str, output_root: Path, limit: int = 10) -> Path:
    """Load the official HF dataset (or a local dataset checkout) and convert it."""
    source_path = Path(source)
    if source_path.exists():
        arrow_files = sorted(source_path.glob("data-*.arrow"))
        if not arrow_files:
            arrow_files = sorted(source_path.glob("**/data-*.arrow"))
        if not arrow_files:
            raise ValueError(f"{source_path}: no GEdit data-*.arrow files found")
        try:
            from datasets import load_dataset
        except ImportError:
            rows = _local_gedit_arrow_rows(arrow_files)
        else:
            rows = load_dataset(
                "arrow",
                data_files=[str(path.resolve()) for path in arrow_files],
                split="train",
            )
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Remote GEdit preparation requires the 'datasets' package"
            ) from exc
        rows = load_dataset(
            source,
            revision=GEDIT_REVISION,
            split="train",
            streaming=True,
        )
    return prepare_gedit_rows(rows, output_root / "GEdit-Bench", limit)


def _local_gedit_arrow_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Stream rows from a Hugging Face Arrow checkout without datasets."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError(
            "Local GEdit preparation requires either 'datasets' or 'pyarrow'"
        ) from exc

    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            for batch in ipc.open_stream(source):
                for row in batch.to_pylist():
                    for field in ("input_image", "input_image_raw"):
                        encoded = row.get(field)
                        if isinstance(encoded, dict) and encoded.get("bytes"):
                            with Image.open(BytesIO(encoded["bytes"])) as image:
                                row[field] = image.copy()
                    yield row


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scene_category(scene_id: str) -> str:
    return scene_id.rsplit("_", 1)[0] if "_" in scene_id else scene_id


def _balanced_scene_rows(
    rows: list[dict[str, Any]],
    limit: int,
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = exclude_ids or set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_id = str(row.get("id", "unknown"))
        if scene_id not in excluded:
            grouped[_scene_category(scene_id)].append(row)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for category in sorted(grouped):
            category_rows = grouped[category]
            if offset >= len(category_rows):
                continue
            selected.append(category_rows[offset])
            added = True
            if len(selected) == limit:
                return selected
        if not added:
            break
        offset += 1
    raise ValueError(f"SANA-WM manifest contains only {len(selected)} usable scenes")


def _first_intrinsics(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if values.shape != (3, 3):
        raise ValueError(f"Expected SANA-WM intrinsics [F,3,3] or [3,3], got {values.shape}")
    return values


def prepare_sana_wm(source_root: Path, output_root: Path, limit: int = 10) -> Path:
    """Prepare unique balanced scenes across the official 60-second splits."""
    if limit < 1 or limit > len(SANA_WM_ACTIONS):
        raise ValueError(f"SANA-WM validation limit must be in [1, {len(SANA_WM_ACTIONS)}]")
    output_dir = output_root / "SANA-WM-Bench"
    requests: list[dict[str, Any]] = []
    simple_count = (limit + 1) // 2
    split_limits = (simple_count, limit - simple_count)
    action_index = 0
    used_scene_ids: set[str] = set()
    for split_name, split_limit in zip(SANA_WM_SPLITS, split_limits, strict=True):
        if split_limit == 0:
            continue
        manifest_path = (
            source_root / split_name / "sanawm_export_v2" / "run_manifest.jsonl"
        )
        rows = _balanced_scene_rows(
            _load_jsonl(manifest_path),
            split_limit,
            exclude_ids=used_scene_ids,
        )
        for row in rows:
            scene_id = str(row["id"])
            used_scene_ids.add(scene_id)
            image_source = source_root / str(
                row.get("image_path", Path("images") / f"{scene_id}.png")
            )
            camera_source = source_root / str(row["camera_path"])
            with np.load(camera_source) as trajectory:
                intrinsics = _first_intrinsics(trajectory["intrinsics"])

            sample_id = f"sana_wm_{split_name}_{scene_id}"
            image_relative = Path("images") / f"{sample_id}.png"
            prompt_relative = Path("prompts") / f"{sample_id}.txt"
            intrinsics_relative = Path("intrinsics") / f"{sample_id}.npy"
            for relative in (image_relative, prompt_relative, intrinsics_relative):
                (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_source, output_dir / image_relative)
            prompt = str(row["prompt"]).strip()
            (output_dir / prompt_relative).write_text(prompt + "\n", encoding="utf-8")
            np.save(output_dir / intrinsics_relative, intrinsics)
            requests.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": action_index,
                    "prompt": prompt,
                    "prompt_file": prompt_relative.as_posix(),
                    "image": image_relative.as_posix(),
                    "camera_intrinsics_file": intrinsics_relative.as_posix(),
                    "camera_intrinsics": [
                        float(intrinsics[0, 0]),
                        float(intrinsics[1, 1]),
                        float(intrinsics[0, 2]),
                        float(intrinsics[1, 2]),
                    ],
                    "action": SANA_WM_ACTIONS[action_index],
                    "category": split_name,
                    "challenge": _scene_category(scene_id),
                    "source_scene_id": scene_id,
                    "source_camera_path": str(row["camera_path"]),
                    "condition_image_sha256": _sha256(output_dir / image_relative),
                }
            )
            action_index += 1
    return _write_json(
        output_dir / "sana_wm_task_eval.json",
        {
            "dataset": "SANA-WM 80-scene benchmark validation parity slice",
            "source": SANA_WM_SOURCE,
            "source_revision": SANA_WM_REVISION,
            "source_manifest_sha256": {
                split_name: _sha256(
                    source_root
                    / split_name
                    / "sanawm_export_v2"
                    / "run_manifest.jsonl"
                )
                for split_name in SANA_WM_SPLITS
            },
            "license": "CC-BY-4.0",
            "sampling": (
                "ten unique category-balanced scenes across the smooth and hard "
                "60-second splits"
            ),
            "control_limitation": (
                "The native TRT runtime accepts action strings, not arbitrary official "
                "c2w files. This parity slice uses fixed 320-step action trajectories "
                "with official scene images, prompts, and first-frame intrinsics."
            ),
            "request_count": len(requests),
            "requests": requests,
        },
    )


def prepare_media_datasets(
    *,
    output_root: Path,
    vbench_info: Path | None = None,
    vbench_model_plugin: bool = False,
    gedit_source: str = "",
    sana_wm_root: Path | None = None,
    limit: int = 10,
) -> list[Path]:
    outputs: list[Path] = []
    if vbench_info:
        if vbench_model_plugin:
            outputs.append(
                prepare_vbench_model_plugin_dataset(
                    vbench_info,
                    output_root,
                    limit,
                )
            )
        else:
            outputs.append(prepare_vbench(vbench_info, output_root, limit))
    elif vbench_model_plugin:
        raise ValueError("VBench model-plugin preparation requires --vbench-info")
    if gedit_source:
        outputs.append(prepare_gedit(gedit_source, output_root, limit))
    if sana_wm_root:
        outputs.append(prepare_sana_wm(sana_wm_root, output_root, limit))
    if not outputs:
        raise ValueError("Pass at least one media benchmark source")
    return outputs

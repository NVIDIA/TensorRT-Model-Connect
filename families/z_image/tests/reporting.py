# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned image previews from the outputs already used for comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.e2e_evidence import evidence_enabled, record_evidence


_MAX_FRAMES = 6
_RAW_ARRAY_BUDGET = 16 * 1024 * 1024


def _indices(count: int) -> list[int]:
    samples = min(_MAX_FRAMES, count)
    return [round(index * (count - 1) / (samples - 1)) for index in range(samples)] if samples > 1 else list(range(samples))


def _files(output: Path) -> list[Path]:
    return sorted(path for path in output.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm"}) if output.is_dir() else [output]


def _native_paths(actual: dict) -> list[Path]:
    return _files(Path(actual["artifact"]))


def _reference_frames(expected: dict):
    return expected["images"]


def _thumbnail(source, path: Path) -> None:
    from PIL import Image

    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            image = image.convert("RGB")
    else:
        values = np.asarray(source)
        if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError("preview requires a finite HWC RGB frame")
        if values.dtype != np.uint8:
            values = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(values)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def record_native_preview(output: Path) -> None:
    if not evidence_enabled():
        return
    try:
        files = _files(output)
        views = []
        for index in _indices(len(files)):
            image = output.parent / "native-report-views" / f"frame-{index:04d}.png"
            _thumbnail(files[index], image)
            views.append({"title": f"Native frame {index}", "image": image,
                          "caption": f"Frame {index} of {len(files)}. Original file: {files[index]}"})
        record_evidence("views", views)
        record_evidence("native_artifacts", {"frame_count": len(files), "sampled_indices": _indices(len(files)),
                                             "original_location": f"Original files: {output}"})
    except (ImportError, AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError, OSError) as error:
        record_evidence("views", {"unavailable": str(error)})


def native_snapshot(actual: dict) -> dict:
    if not evidence_enabled():
        return actual
    result = dict(actual)
    if "artifact" in result:
        result["artifact"] = f"Original artifact: {result['artifact']}"
    if "output" in result:
        result["output"] = f"Original output: {result['output']}"
    for name in ("frames", "outputs"):
        if name in result:
            values = result[name]
            result[name + "_count"] = len(values)
            result[name + "_sampled_indices"] = _indices(len(values))
            result[name] = [f"Original frame: {values[index]}" for index in _indices(len(values))]
    return result


def reference_snapshot(expected: dict) -> dict:
    if not evidence_enabled():
        return expected
    if expected.get("_invariant_only"):
        return {"mode": "contract_only", "reference_images": "No reference frames were produced by this case."}
    try:
        frames = _reference_frames(expected)
        indices = _indices(len(frames))
        samples = []
        used = 0
        for index in indices:
            value = frames[index]
            if isinstance(value, (str, Path)):
                samples.append({"frame_index": index, "original_file": f"Original file: {value}"})
                continue
            array = np.asarray(value)
            item = {"frame_index": index, "shape": list(array.shape), "dtype": str(array.dtype)}
            if used + array.nbytes <= _RAW_ARRAY_BUDGET:
                item["values"] = array
                used += array.nbytes
            else:
                item["omitted"] = "raw preview array exceeds the family reporting budget; PNG preview remains available"
            samples.append(item)
        result = {key: value for key, value in expected.items() if key not in {"images", "frame_paths", "frames_path"}}
        result.update({"frame_count": len(frames), "sampled_indices": indices, "sampled_frames": samples,
                       "sampling_note": "At most six evenly spaced frames are archived; all original outputs still participate in the unchanged comparison."})
        if "frames_path" in expected:
            result["original_file"] = f"Original file: {expected['frames_path']}"
        return result
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError, OSError) as error:
        return {"unavailable": str(error)}


def record_report_views(actual: dict, expected: dict, directory: Path) -> None:
    if not evidence_enabled():
        return
    if expected.get("_invariant_only"):
        record_evidence("views", {"reference_unavailable": "This case runs native output invariants; no reference frames were produced."})
        return
    try:
        native = _native_paths(actual)
        reference = _reference_frames(expected)
        count = min(len(native), len(reference))
        if not count:
            raise ValueError("no paired native/reference frames are available")
        views = []
        for index in _indices(count):
            for role, source in (("native", native[index]), ("reference", reference[index])):
                image = directory / f"frame-{index:04d}-{role}.png"
                _thumbnail(source, image)
                views.append({"title": f"Frame {index} / {role}", "image": image,
                              "caption": f"Matched frame index {index}; native count {len(native)}, reference count {len(reference)}. Preview only; no resampling or clipping enters the numerical comparison."})
        record_evidence("views", views)
    except (ImportError, AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError, OSError) as error:
        record_evidence("views", {"unavailable": str(error)})

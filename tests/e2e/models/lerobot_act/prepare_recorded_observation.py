# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one immutable LeRobot v3 recorded observation for replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

_DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
_DATASET_REVISION = "6a43d500f101255823a9d2b9dc244eeb01a2cd31"
_DATA_FILE = "data/chunk-000/file-000.parquet"
_VIDEO_FILE = "videos/observation.images.top/chunk-000/file-000.mp4"
_DATA_SHA256 = "8d02ba269356ed960741c3169e8106383b960583e9cd1e5960a0495580a9d6c2"
_VIDEO_SHA256 = "e16528b7da7dd433a60cb0e17c59689fc834be43fc0f0b24586377c833fcd0de"
_IMAGE_SHA256 = "a53369bda31c6563548bd834e88f6640eb28e3233807c499a56d012de992799c"
_STATE_SHA256 = "40cd79b41ce45e9ffc35f6dc70f74a980d8760857326a9482e109b1d763f54c0"
_FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "recorded_observation"
_FIXTURE_FILES = (
    "observation.images.top.png",
    "observation.state.f32",
    "recorded_observation.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(filename: str, *, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=_DATASET_ID,
            repo_type="dataset",
            revision=_DATASET_REVISION,
            filename=filename,
            local_files_only=local_files_only,
        )
    )


def _recorded_row(data_path: Path, episode_index: int, frame_index: int) -> tuple[int, np.ndarray]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(
        data_path,
        columns=["observation.state", "episode_index", "frame_index", "index"],
    )
    episodes = table.column("episode_index").to_numpy(zero_copy_only=False)
    frames = table.column("frame_index").to_numpy(zero_copy_only=False)
    matches = np.flatnonzero((episodes == episode_index) & (frames == frame_index))
    if matches.size != 1:
        raise ValueError(
            f"recording has {matches.size} rows for episode={episode_index}, frame={frame_index}"
        )
    row = int(matches[0])
    global_index = int(table.column("index")[row].as_py())
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype="<f4")
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"recorded state has invalid shape or values: {state.shape}")
    return global_index, state


def _decode_frame(video_path: Path, global_index: int) -> np.ndarray:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{global_index})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to decode recorded frame: {completed.stderr.decode()}")
    expected = 480 * 640 * 3
    if len(completed.stdout) != expected:
        raise ValueError(
            f"recorded RGB frame has {len(completed.stdout)} bytes, expected {expected}"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(480, 640, 3).copy()


def _is_qualified_observation(directory: Path, episode_index: int, frame_index: int) -> bool:
    image_path = directory / "observation.images.top.png"
    state_path = directory / "observation.state.f32"
    metadata_path = directory / "recorded_observation.json"
    if not image_path.is_file() or not state_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("dataset_id") == _DATASET_ID
        and metadata.get("dataset_revision") == _DATASET_REVISION
        and metadata.get("data_file") == _DATA_FILE
        and metadata.get("data_sha256") == _DATA_SHA256
        and metadata.get("video_file") == _VIDEO_FILE
        and metadata.get("video_sha256") == _VIDEO_SHA256
        and metadata.get("episode_index") == episode_index
        and metadata.get("frame_index") == frame_index
        and metadata.get("image_sha256") == _IMAGE_SHA256 == _sha256(image_path)
        and metadata.get("state_sha256") == _STATE_SHA256 == _sha256(state_path)
    )


def _materialize_packaged_observation(
    output: Path, episode_index: int, frame_index: int
) -> bool:
    if not _FIXTURE_DIR.exists():
        return False
    if not _is_qualified_observation(_FIXTURE_DIR, episode_index, frame_index):
        raise ValueError("packaged LeRobot recorded observation failed its qualified contract")
    for name in _FIXTURE_FILES:
        shutil.copyfile(_FIXTURE_DIR / name, output / name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.episode_index != 0 or arguments.frame_index != 0:
        parser.error("the qualified replay contract is episode 0, frame 0")

    arguments.output.mkdir(parents=True, exist_ok=True)
    image_path = arguments.output / "observation.images.top.png"
    state_path = arguments.output / "observation.state.f32"
    metadata_path = arguments.output / "recorded_observation.json"
    if _is_qualified_observation(
        arguments.output, arguments.episode_index, arguments.frame_index
    ):
        print(metadata_path)
        return 0
    if _materialize_packaged_observation(
        arguments.output, arguments.episode_index, arguments.frame_index
    ):
        print(metadata_path)
        return 0

    data_path = _download(_DATA_FILE, local_files_only=arguments.local_files_only)
    video_path = _download(_VIDEO_FILE, local_files_only=arguments.local_files_only)
    if _sha256(data_path) != _DATA_SHA256:
        raise ValueError("recorded LeRobot parquet SHA-256 does not match the qualified artifact")
    if _sha256(video_path) != _VIDEO_SHA256:
        raise ValueError("recorded LeRobot video SHA-256 does not match the qualified artifact")
    global_index, state = _recorded_row(data_path, arguments.episode_index, arguments.frame_index)
    pixels = _decode_frame(video_path, global_index)

    from PIL import Image

    Image.fromarray(pixels, mode="RGB").save(image_path, format="PNG", optimize=False)
    state.tofile(state_path)
    image_sha256 = _sha256(image_path)
    state_sha256 = _sha256(state_path)
    if image_sha256 != _IMAGE_SHA256 or state_sha256 != _STATE_SHA256:
        raise ValueError(
            "decoded LeRobot observation does not match the qualified image/state artifacts"
        )
    metadata = {
        "dataset_id": _DATASET_ID,
        "dataset_revision": _DATASET_REVISION,
        "dataset_codebase_version": "v3.0",
        "data_file": _DATA_FILE,
        "data_sha256": _DATA_SHA256,
        "video_file": _VIDEO_FILE,
        "video_sha256": _VIDEO_SHA256,
        "episode_index": arguments.episode_index,
        "frame_index": arguments.frame_index,
        "global_index": global_index,
        "fps": 50,
        "image_shape_hwc": [480, 640, 3],
        "image_sha256": image_sha256,
        "state_shape": [14],
        "state_sha256": state_sha256,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

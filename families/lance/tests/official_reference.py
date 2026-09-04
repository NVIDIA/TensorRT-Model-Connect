# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream Lance x2t_image reference for this family."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ENTRYPOINT = "inference_lance.py"
_MODEL_DIR = "Lance_3B"
_VISION_DIR = "Qwen2.5-VL-ViT"


def _source_dir() -> Path:
    value = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert value, "selected lance E2E requires TRTMC_REFERENCE_SOURCE_DIR"
    source = Path(value).resolve()
    assert (source / _ENTRYPOINT).is_file(), f"Lance reference is missing {source / _ENTRYPOINT}"
    return source


def _link_children(source: Path, destination: Path, excluded: set[str]) -> None:
    destination.mkdir()
    for path in source.iterdir():
        if path.name not in excluded:
            target = destination / path.name
            if path.name in {"__init__.py", _ENTRYPOINT}:
                target.write_bytes(path.read_bytes())
            else:
                target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def _image_only_source(source: Path, output_root: Path) -> Path:
    """Expose the pinned upstream image path without its unused video dependency."""
    destination = output_root / "official_image_source"
    _link_children(source, destination, {"data"})
    _link_children(source / "data", destination / "data", {"datasets_custom"})
    _link_children(
        source / "data/datasets_custom",
        destination / "data/datasets_custom",
        {"validation_dataset.py"},
    )

    source_path = source / "data/datasets_custom/validation_dataset.py"
    text = source_path.read_text(encoding="utf-8")
    for line in ("import decord\n", "from decord import VideoReader\n"):
        assert text.count(line) == 1, f"Lance upstream import contract changed: {line.strip()}"
        text = text.replace(line, "")
    annotation = "video: VideoReader"
    assert text.count(annotation) == 1, "Lance upstream VideoReader annotation contract changed"
    text = text.replace(annotation, "video")
    (destination / "data/datasets_custom/validation_dataset.py").write_text(text, encoding="utf-8")
    return destination


def _result_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    text = str(payload[0].get("answer", "")).strip()
    assert text, "Lance official reference produced an empty answer"
    return text


def run_official_generation(
    model_root: Path,
    image: Path,
    prompt: str,
    output_root: Path,
    timeout_s: int,
) -> str:
    output_root = output_root.resolve()
    source = _image_only_source(_source_dir(), output_root)
    model_dir = model_root / _MODEL_DIR
    assert (model_dir / "llm_config.json").is_file(), model_dir
    assert (model_dir / "model.safetensors").is_file(), model_dir
    assert image.is_file(), image
    assert prompt

    workspace = output_root / "official_workspace"
    workspace.mkdir()
    (workspace / "downloads").symlink_to(model_root.resolve(), target_is_directory=True)
    vision_dir = model_root / _VISION_DIR
    assert (vision_dir / "config.json").is_file(), vision_dir
    assert (vision_dir / "vit.safetensors").is_file(), vision_dir
    request = output_root / "official_request.json"
    request.write_text(
        json.dumps(
            {
                "000000": {
                    "interleave_array": [str(image.resolve()), ["", prompt, ""]],
                    "element_dtype_array": ["image", "text"],
                    "istarget_in_interleave": [0, 1],
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result_dir = output_root / "official_output"
    environment = os.environ.copy()
    command = [
        os.environ.get("TRTMC_REFERENCE_PYTHON", sys.executable),
        str(source / _ENTRYPOINT),
        "--model_path",
        str(model_dir),
        "--llm_path",
        str(model_dir),
        "--vit_path",
        str(vision_dir),
        "--vit_type",
        "qwen_2_5_vl_original",
        "--llm_qk_norm",
        "true",
        "--llm_qk_norm_und",
        "true",
        "--llm_qk_norm_gen",
        "true",
        "--tie_word_embeddings",
        "false",
        "--copy_init_moe",
        "true",
        "--max_num_frames",
        "121",
        "--max_latent_size",
        "64",
        "--latent_patch_size",
        "1",
        "1",
        "1",
        "--visual_und",
        "true",
        "--visual_gen",
        "true",
        "--vae_model_type",
        "wan",
        "--apply_qwen_2_5_vl_pos_emb",
        "true",
        "--apply_chat_template",
        "false",
        "--cfg_type",
        "0",
        "--validation_data_seed",
        "42",
        "--resolution",
        "image_512res",
        "--task",
        "x2t_image",
        "--save_path_gen",
        str(result_dir),
        "--val_dataset_config_file",
        str(request),
        "--text_template",
        "true",
        "--use_KVcache",
        "true",
        "--enhance_prompt",
        "false",
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    assert completed.returncode == 0, (
        f"Lance official reference failed (rc={completed.returncode}): {completed.stderr[-4000:]}"
    )
    return _result_text(result_dir / "result.json")

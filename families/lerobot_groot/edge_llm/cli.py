# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build for Isaac LIBERO-10 and native LeRobot GR00T policies."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tensorrt_model_connect.build import build as publish
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family

_MODEL = "nvidia/GR00T-N1.7-LIBERO"
_REVISION = "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"
_LEROBOT = "nvidia/gr00t17-lerobot-libero_spatial-640"
_LEROBOT_REVISION = "32a6ec786d6509df31b40392b4e4dcdda78c0f11"


@dataclass(frozen=True)
class GrootBuildRequest:
    """Build inputs for the Isaac LIBERO-10 policy."""

    model_dir: Path
    output_path: Path
    precision: str
    verbose: bool
    edge_builder_python: Path | None = None
    quantization: str | None = None

    family: ClassVar[str] = "lerobot_groot"
    task: ClassVar[str] = "images_state_text_to_action_chunk"
    backend: ClassVar[str] = "trt"
    max_batch_size: ClassVar[int] = 1
    tensor_parallel_size: ClassVar[int] = 1
    context_parallel_size: ClassVar[int] = 1
    fp32_layers: ClassVar[tuple[int, ...]] = ()
    dynamic_kv_cache: ClassVar[bool] = False
    max_sequence_length: ClassVar[None] = None
    mtp_seq_len: ClassVar[None] = None
    image_height: ClassVar[None] = None
    image_width: ClassVar[None] = None
    video_num_frames: ClassVar[None] = None
    graph_transform: ClassVar[None] = None


def build(*, model: str, output: Path, source_revision: str | None = None,
          precision: str | None = None, verbose: bool = False, edge_builder_python: Path | None = None,
          checkpoint_subdir: str | None = None, quantization: str | None = None) -> int:
    """Resolve the named download or local policy without changing its precision or layout."""
    if checkpoint_subdir not in (None, "isaac-libero10"):
        raise ValueError("Only the staged Isaac LIBERO-10 subdirectory is supported")
    directory = Path(model)
    if not directory.is_dir():
        if model not in {_MODEL, _LEROBOT}:
            raise ValueError(f"Supply a local GR00T policy directory, {_MODEL}, or {_LEROBOT}")
        from huggingface_hub import snapshot_download

        if model == _MODEL:
            directory = Path(snapshot_download(
                _MODEL, revision=source_revision or _REVISION,
                allow_patterns=["libero_10/" + name for name in (
                    "config.json", "embodiment_id.json", "statistics.json", "processor_config.json",
                    "model.safetensors.index.json", "model-*.safetensors")],
            )) / "libero_10"
        else:
            directory = Path(snapshot_download(_LEROBOT, revision=source_revision or _LEROBOT_REVISION))
    elif checkpoint_subdir and (directory / checkpoint_subdir / "config.json").is_file():
        directory = directory / checkpoint_subdir
    elif not (directory / "config.json").is_file() and not checkpoint_subdir:
        directory = directory / "libero_10"

    metadata = load_model_metadata(directory)
    _, support = resolve_family(metadata, "lerobot_groot")
    isaac = metadata.model_type == "Gr00tN1d7"
    if not isaac and (checkpoint_subdir is not None or edge_builder_python is not None):
        raise ValueError("The staged Isaac subdirectory and Edge interpreter apply only to LIBERO-10")
    if isaac and quantization not in (None, "none"):
        raise ValueError("Isaac LIBERO-10 requires unquantized FP16")
    publish(GrootBuildRequest(model_dir=directory, output_path=output,
                              precision=precision or support.default_precision, verbose=verbose,
                              edge_builder_python=edge_builder_python, quantization=quantization))
    return 0

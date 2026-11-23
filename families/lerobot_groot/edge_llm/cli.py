# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public build for the Isaac LIBERO-10 checkpoint; LeRobot policies retain the generic build command."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tensorrt_model_connect.build import build as publish
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family

_MODEL = "nvidia/GR00T-N1.7-LIBERO"
_REVISION = "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"


@dataclass(frozen=True)
class GrootBuildRequest:
    """Build inputs for the Isaac LIBERO-10 policy."""

    model_dir: Path
    output_path: Path
    precision: str
    verbose: bool
    edge_builder_python: Path | None = None

    family: ClassVar[str] = "lerobot_groot"
    task: ClassVar[str] = "images_state_text_to_action_chunk"
    backend: ClassVar[str] = "trt"
    max_batch_size: ClassVar[int] = 1
    tensor_parallel_size: ClassVar[int] = 1
    context_parallel_size: ClassVar[int] = 1
    quantization: ClassVar[None] = None
    fp32_layers: ClassVar[tuple[int, ...]] = ()
    dynamic_kv_cache: ClassVar[bool] = False
    max_sequence_length: ClassVar[None] = None
    mtp_seq_len: ClassVar[None] = None
    image_height: ClassVar[None] = None
    image_width: ClassVar[None] = None
    video_num_frames: ClassVar[None] = None
    graph_transform: ClassVar[None] = None


def build(*, model: str, output: Path, revision: str | None = None, precision: str = "fp16",
          verbose: bool = False, edge_builder_python: Path | None = None) -> int:
    """Fetch only the documented subfolder, or accept the same already-staged policy directory."""
    directory = Path(model)
    if not directory.is_dir():
        if model != _MODEL:
            raise ValueError(f"Supply a local Isaac LIBERO-10 checkpoint directory or {_MODEL}")
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(
            _MODEL, revision=revision or _REVISION,
            allow_patterns=["libero_10/" + name for name in (
                "config.json", "embodiment_id.json", "statistics.json", "processor_config.json",
                "model.safetensors.index.json", "model-*.safetensors")],
        ))
    if not (directory / "config.json").is_file():
        directory = directory / "libero_10"
    metadata = load_model_metadata(directory)
    if metadata.model_type != "Gr00tN1d7":
        raise ValueError("The family build command requires the Isaac GR00T N1.7 LIBERO-10 layout")
    resolve_family(metadata, "lerobot_groot")
    publish(GrootBuildRequest(model_dir=directory, output_path=output, precision=precision, verbose=verbose,
                              edge_builder_python=edge_builder_python))
    return 0

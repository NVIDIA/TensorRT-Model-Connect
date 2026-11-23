# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the public openpi policy explicitly; ordinary LeRobot requests remain unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import ClassVar
from urllib.request import urlopen

from huggingface_hub.constants import HF_HUB_OFFLINE
from tensorrt_model_connect.build import BuildRequest, build as publish
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family
from ..openpi import POLICIES


_CHECKPOINTS = {
    "pi05_libero": ("lerobot/pi05_libero_base", "5fcf06fc42b6717e22ab885fd381a096574b12c7"),
    "pi05_droid": ("lerobot/pi05_droid", "72824c0a93f00ce5bb8bedb7feb58953ba1da364"),
    "pi05_aloha": ("lerobot/pi05_base", "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"),
}


@dataclass(frozen=True)
class OpenPiBuildRequest:
    """Inputs exposed by the family CLI; the generic LeRobot build keeps its own contract."""

    model_dir: Path
    output_path: Path
    precision: str
    verbose: bool
    pi05_policy_config: str
    edge_builder_python: Path | None = None

    family: ClassVar[str] = "lerobot_pi05"
    task: ClassVar[str] = "images_state_text_to_action_chunk"
    backend: ClassVar[str] = "trt"
    max_sequence_length: ClassVar[None] = None
    image_height: ClassVar[None] = None
    image_width: ClassVar[None] = None
    video_num_frames: ClassVar[None] = None
    max_batch_size: ClassVar[int] = 1
    tensor_parallel_size: ClassVar[int] = 1
    context_parallel_size: ClassVar[int] = 1
    quantization: ClassVar[None] = None
    fp32_layers: ClassVar[tuple[int, ...]] = ()
    dynamic_kv_cache: ClassVar[bool] = False
    graph_transform: ClassVar[None] = None
    mtp_seq_len: ClassVar[None] = None


def _asset(local: Path, destination: Path, url: str) -> None:
    if local.is_file():
        # The release copies the assets subtree and rejects links outside its root.
        # Stage the small metadata as a regular file; checkpoint weights remain linked.
        if destination.name == "norm_stats.json":
            shutil.copyfile(local, destination)
        else:
            destination.symlink_to(local.resolve())
    else:
        if HF_HUB_OFFLINE:
            raise FileNotFoundError(f"Offline openpi build needs {local.name} staged at {local}")
        with urlopen(url, timeout=90) as src, destination.open("xb") as out:
            shutil.copyfileobj(src, out)


def build(*, model: str, output: Path, pi05_policy_config: str | None = None, revision: str | None = None,
          precision: str | None = None, verbose: bool = False, edge_builder_python: Path | None = None,
          quantization: str | None = None) -> int:
    """Build the selected action contract without modifying the original checkpoint."""
    source = Path(model)
    if pi05_policy_config is None:
        if (edge_builder_python is not None or precision not in (None, "bf16", "fp32")
                or quantization not in (None, "none", "fp8")
                or (quantization == "fp8" and precision == "fp32")):
            raise ValueError("Native LeRobot requires BF16 or FP32, with FP8 available only for BF16")
        if not source.is_dir():
            from huggingface_hub import snapshot_download

            pinned = dict(_CHECKPOINTS.values()).get(model)
            source = Path(snapshot_download(model, revision=revision or pinned,
                                           allow_patterns=["*.json", "*.safetensors", "tokenizer.model"]))
        _, support = resolve_family(load_model_metadata(source), "lerobot_pi05")
        publish(BuildRequest(model_dir=source, output_path=output, family="lerobot_pi05",
                             task=support.default_task, precision=precision or support.default_precision,
                             verbose=verbose, quantization=quantization))
        return 0
    precision = precision or "fp16"
    if pi05_policy_config not in POLICIES or precision != "fp16" or quantization not in (None, "none"):
        raise ValueError("Select a documented openpi policy with unquantized FP16 precision")
    if not source.is_dir():
        repo, pinned = _CHECKPOINTS[pi05_policy_config]
        if model != repo:
            raise ValueError(f"{pi05_policy_config} uses {repo} or a matching local checkpoint")
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(repo, revision=revision or pinned,
                                       allow_patterns=["config.json", "model.safetensors"]))
    resolve_family(load_model_metadata(source), "lerobot_pi05")
    with tempfile.TemporaryDirectory(prefix="trtmc-openpi-") as temp:
        overlay = Path(temp)
        for name in ("config.json", "model.safetensors"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
            (overlay / name).symlink_to((source / name).resolve())
        (overlay / "assets").mkdir()
        policy = POLICIES[pi05_policy_config]
        _asset(source / "assets/norm_stats.json", overlay / "assets/norm_stats.json",
               "https://storage.googleapis.com/openpi-assets/checkpoints/"
               + policy.stats_subdir + "/norm_stats.json?generation=" + policy.stats_generation)
        _asset(source / "tokenizer.model", overlay / "tokenizer.model",
               "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model?generation=1711547605575873")
        publish(OpenPiBuildRequest(model_dir=overlay, output_path=output, precision=precision, verbose=verbose,
                                   pi05_policy_config=pi05_policy_config, edge_builder_python=edge_builder_python))
    return 0

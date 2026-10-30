# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""minimax_h3-owned command for the explicit OmniRef build; importing this module is CPU-only."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

BASE_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
PROFILE = {"memory_preset": "minimal", "reference_capacity": "six_portrait", "continuation": True}


def _snapshot(model: str, revision: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


def _request(model_dir: Path, output: Path):
    from tensorrt_model_connect.build import BuildRequest

    return BuildRequest(model_dir=model_dir, output_path=output, family="minimax_h3",
                        task="references_text_to_audio_video", precision="bf16", backend="trt",
                        image_height=1344, image_width=768, video_num_frames=226)


def _publish(output: Path, build) -> None:
    from tensorrt_model_connect.bundle_writer import BundleWriter

    writer = BundleWriter(output)
    try:
        build(writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def _drop_page_cache(path: Path) -> None:
    """Release a large bundle's clean page cache.

    On coherent-memory systems such as GB300, page cache can sit on the GPU's NUMA node and
    take the memory the next engine build needs.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def _register_kernel(bridge: Path, kernel_library: Path) -> None:
    """Load the bridge (TensorRT's TvmFfiKernel creator) and register the VSA kernel through it."""
    from .reference_vsa import KERNEL_NAME

    library = ctypes.CDLL(str(bridge.resolve()), mode=ctypes.RTLD_GLOBAL)
    load = library.trtmc_load_byok_kernel
    load.argtypes = [ctypes.c_char_p] * 3
    load.restype = ctypes.c_char_p
    error = load(os.fsencode(kernel_library.resolve()), b"run", KERNEL_NAME.encode())
    if error is not None:
        raise RuntimeError(error.decode())


def build_omniref(*, output: Path, scratch: Path, bridge: Path, kernel_library: Path,
                  denoiser_workspace_gib: int = 24, base: str = "MiniMaxAI/MiniMax-H3",
                  student: str = "FastVideo/FastVideo-FastH3-Omni-8-Step-V1") -> int:
    """Build the original two-window bundle once (kept in `scratch`), then the student."""
    for library, role in ((bridge, "the TVM-FFI bridge"), (kernel_library, "its VSA kernel")):
        if not library.is_file():
            raise FileNotFoundError(f"OmniRef needs {role}: {library}")
    from .reference_pdd import MODEL_REVISION

    base_dir, student_dir = _snapshot(base, BASE_REVISION), _snapshot(student, MODEL_REVISION)
    scratch.mkdir(parents=True, exist_ok=True)
    source = scratch / "minimax-h3-two-window-six-portrait.bundle"
    if not source.is_file():
        from .reference_build import build as build_original

        original = _request(base_dir, source)

        def build_source(writer) -> None:
            writer.record_build(original)
            build_original(original, writer, **PROFILE)

        _publish(source, build_source)
        os.sync()
        _drop_page_cache(source)
    from .reference_pdd_build import build as build_student

    _register_kernel(bridge, kernel_library)
    request = _request(student_dir, output)
    _publish(output, lambda writer: build_student(
        request, writer, base_model_dir=base_dir, source_bundle=source,
        plugin_library=bridge.resolve(), denoiser_workspace_bytes=denoiser_workspace_gib << 30,
        **PROFILE))
    return 0

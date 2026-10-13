# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the image reference's binary dependencies without requiring a GPU."""

from importlib import import_module, metadata
from pathlib import Path
import platform
import sys


def main() -> None:
    assert sys.version_info[:2] == (3, 12), "Lance reference requires CPython 3.12"
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert Path(sys.prefix) == Path("/opt/lance-reference"), "Wrong reference interpreter"
    expected = {
        "torch": "2.10.0+cu130",
        "torchvision": "0.25.0+cu130",
        # The upstream filename carries the ABI suffix; its wheel METADATA
        # retains the public version. The lock verifies the exact binary hash.
        "flash-attn": "2.8.3",
        "transformers": "4.49.0",
        "numpy": "1.26.4",
    }
    for package, version in expected.items():
        actual = metadata.version(package)
        assert actual.lower() == version, f"Unexpected {package}: {actual}"
    for module in (
        "accelerate",
        "cv2",
        "einops",
        "imageio",
        "jinja2",
        "PIL",
        "safetensors.torch",
        "scipy",
        "sentencepiece",
        "sklearn",
        "tokenizers",
        "torchvision",
        "tqdm",
        "yaml",
    ):
        import_module(module)

    import torch
    from flash_attn import flash_attn_varlen_func
    from torch.nn.attention.flex_attention import flex_attention
    from transformers import HfArgumentParser, Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

    assert torch.version.cuda == "13.0", "FlashAttention wheel requires the CUDA 13 Torch build"
    assert torch._C._GLIBCXX_USE_CXX11_ABI, "FlashAttention wheel requires the C++11 ABI"
    assert all(
        callable(value)
        for value in (
            flash_attn_varlen_func,
            flex_attention,
            HfArgumentParser,
            Qwen2VLImageProcessor,
            Qwen2_5_VLVisionConfig,
        )
    )
    print("Lance reference dependency imports passed; GPU parity has not run", flush=True)


if __name__ == "__main__":
    main()

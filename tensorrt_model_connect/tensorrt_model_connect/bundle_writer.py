"""Write .trtfb bundle files — 1:1 compatible with C++ ReadBundleFile().

Format:
  Bytes 0-7:   Magic "TRTFB\\x00\\x01\\x00"
  Bytes 8-15:  uint64_t json_header_length (LE)
  Bytes 16..N: JSON metadata header (UTF-8)
  Bytes N..EOF: Binary sections
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"

# Section names are part of the .trtfb external schema. Define as constants
# so downstream code references them by symbol — a typo or rename becomes
# an import error rather than a silent miss at bundle load.
QWEN_IMAGE_TEXT_ENGINE_SECTION = "qwen_image_text_engine_plan"
QWEN_IMAGE_DENOISER_SECTION = "qwen_image_denoiser_engine_plan"
QWEN_IMAGE_VAE_DECODER_SECTION = "qwen_image_vae_decoder_plan"
QWEN_IMAGE_VAE_ENCODER_SECTION = "qwen_image_vae_encoder_plan"
QWEN_IMAGE_VISION_ENGINE_SECTION = "qwen_image_vision_engine_plan"
QWEN_IMAGE_PREPROCESSOR_WEIGHTS_SECTION = "qwen_image_preprocessor_weights"


@dataclass
class BundleInfo:
    model_id: str = ""
    model_type: str = ""
    family: str = ""
    trt_version: str = ""
    trt_abi: str = ""
    gpu_name: str = ""
    created_at: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    max_cache_length: int = 32
    runtime_strategy: str = ""
    precision: str = "fp32"
    quantization: str = "none"
    tokenizer_add_special_tokens: bool = False
    io_map: dict | None = None  # tensor name mapping; None = TRT API defaults
    # Optional diffusion batch caps. Missing means old-bundle behavior: every
    # component is treated as batch-cap 1 by the runtime.
    max_batch_size: dict[str, int] | None = None
    # Namespaced defaults produced at build time. When non-empty, serialized
    # into the header as `defaults: {namespace: {field: value, ...}}` and
    # read back at runtime as the BUNDLE_DEFAULT layer — the lowest-priority
    # input to the config registry merge. None/empty → no block emitted, so
    # old readers continue to work untouched.
    defaults: dict | None = None


@dataclass
class BundleSection:
    name: str
    data: bytes


def write_bundle(
    path: str | Path,
    info: BundleInfo,
    sections: list[BundleSection],
) -> None:
    """Write a .trtfb bundle file."""
    # Build section offset/size list for JSON header
    section_meta: list[dict[str, Any]] = []
    offset = 0
    for s in sections:
        section_meta.append({
            "name": s.name,
            "offset": offset,
            "size": len(s.data),
        })
        offset += len(s.data)

    # Build JSON header
    header = {
        "model_id": info.model_id,
        "model_type": info.model_type,
        "family": info.family,
        "trt_version": info.trt_version,
        "trt_abi": info.trt_abi,
        "gpu_name": info.gpu_name,
        "created_at": info.created_at,
        "vocab_size": info.vocab_size,
        "hidden_size": info.hidden_size,
        "num_layers": info.num_layers,
        "num_attention_heads": info.num_attention_heads,
        "num_key_value_heads": info.num_key_value_heads,
        "max_cache_length": info.max_cache_length,
        **({"runtime_strategy": info.runtime_strategy}
           if info.runtime_strategy else {}),
        "precision": info.precision,
        **({"quantization": info.quantization}
           if info.quantization != "none" else {}),
        "tokenizer_add_special_tokens": int(info.tokenizer_add_special_tokens),
        **({"io_map": info.io_map} if info.io_map else {}),
        **({"max_batch_size": dict(info.max_batch_size)}
           if info.max_batch_size is not None else {}),
        **({"defaults": info.defaults} if info.defaults else {}),
        "sections": {
            s["name"]: {"offset": s["offset"], "size": s["size"]}
            for s in section_meta
        },
    }
    header_json = json.dumps(header, indent=2).encode("utf-8")

    with open(path, "wb") as f:
        f.write(BUNDLE_MAGIC)
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for s in sections:
            f.write(s.data)


def read_bundle_header(path: str | Path) -> dict[str, Any]:
    """Read only the JSON header from a .trtfb bundle."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != BUNDLE_MAGIC:
            raise ValueError(f"Bad bundle magic: {magic!r}")
        header_len_data = f.read(8)
        if len(header_len_data) != 8:
            raise ValueError("Bundle truncated before header length")
        header_len = struct.unpack("<Q", header_len_data)[0]
        header_data = f.read(header_len)
        if len(header_data) != header_len:
            raise ValueError("Bundle truncated before full header")
    return json.loads(header_data.decode("utf-8"))

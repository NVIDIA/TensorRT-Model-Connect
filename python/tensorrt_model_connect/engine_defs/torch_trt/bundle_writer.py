"""Write .trtfb bundle files — raw TRT engine bundles from Torch-TRT pipeline.

Uses the same format as the raw TRT pipeline (TRTFB magic, section-based layout)
so the C++ runtime can load them with the standard bundle reader.

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

# Use TRTFB magic — same as raw TRT pipeline bundles.
# The C++ runtime reads this with the standard ReadBundleFile().
BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"


@dataclass
class TtrtBundleInfo:
    model_id: str = ""
    model_type: str = ""
    family: str = ""
    torch_version: str = ""
    torchtrt_version: str = ""
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
    precision: str = "fp16"
    runtime_strategy: str = ""
    tokenizer_add_special_tokens: bool = False
    build_backend: str = ""
    io_map: dict | None = None  # tensor name mapping; None = TRT API defaults


@dataclass
class BundleSection:
    name: str
    data: bytes


def write_bundle(
    path: str | Path,
    info: TtrtBundleInfo,
    sections: list[BundleSection],
) -> None:
    """Write a .trtfb bundle file."""
    section_meta: list[dict[str, Any]] = []
    offset = 0
    for s in sections:
        section_meta.append({
            "name": s.name,
            "offset": offset,
            "size": len(s.data),
        })
        offset += len(s.data)

    header = {
        "model_id": info.model_id,
        "model_type": info.model_type,
        "family": info.family,
        "torch_version": info.torch_version,
        "torchtrt_version": info.torchtrt_version,
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
        "precision": info.precision,
        **({"runtime_strategy": info.runtime_strategy}
           if info.runtime_strategy else {}),
        **({"build_backend": info.build_backend}
           if info.build_backend else {}),
        **({"io_map": info.io_map} if info.io_map else {}),
        "tokenizer_add_special_tokens": int(info.tokenizer_add_special_tokens),
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

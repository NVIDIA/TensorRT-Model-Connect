# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict provenance and partitioning for the public H3 ``transformer_ref``.

Ref2VA is not a mode bit on the T2VA transformer.  The released repository has
a second 66.28 GB checkpoint partition with the same architecture and tensor
names but different values.  This module deliberately refuses to build from
``transformer/`` or to fall back when ``transformer_ref/`` is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


MODEL_ID = "MiniMaxAI/MiniMax-H3"
CHECKPOINT_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
COMPONENT_NAME = "transformer_ref"
TOTAL_TENSOR_BYTES = 66_280_430_080
CONFIG_BYTES = 546
CONFIG_SHA256 = "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae"
INDEX_BYTES = 64_488
INDEX_SHA256 = "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 8 << 20

SHARDS: tuple[tuple[str, int, str], ...] = (
    (
        "diffusion_pytorch_model-00001-of-00014.safetensors",
        4_825_958_704,
        "7a3fcad885f51560e550b2e84c9a8d8b35e62996cfd9076937e992bd23478df9",
    ),
    (
        "diffusion_pytorch_model-00002-of-00014.safetensors",
        4_702_158_032,
        "1638ae1dc8ae26c4ba43ad28a6d851ad8983847324bb2b468719c7c81f219706",
    ),
    (
        "diffusion_pytorch_model-00003-of-00014.safetensors",
        4_933_368_192,
        "1ef3c4954ffe5a664c2e3028e2a3241190d9c159dce6ba1136002c6af1db5353",
    ),
    (
        "diffusion_pytorch_model-00004-of-00014.safetensors",
        4_567_069_608,
        "12d92f2975cfd5c5b786126385c52e5bf64884d4b4d6e60c3ef5d857c3f7469f",
    ),
    (
        "diffusion_pytorch_model-00005-of-00014.safetensors",
        4_702_158_080,
        "304d41ce03d59ac94bceb055935bf4e034df0badf8b0df4ded327c08a288a4cc",
    ),
    (
        "diffusion_pytorch_model-00006-of-00014.safetensors",
        4_933_368_232,
        "12a134b7c76d86edbe8fa2dc315f6cdaf4e1aca1b6ea4dfe4cad92df03d42eeb",
    ),
    (
        "diffusion_pytorch_model-00007-of-00014.safetensors",
        4_567_069_608,
        "b96395261359937c00fb42f4eb29306dc59b1a3368eeba52af4fb66e3e142c69",
    ),
    (
        "diffusion_pytorch_model-00008-of-00014.safetensors",
        4_702_158_080,
        "1897a6bf3b4fc834bb82d73ca02a7afc7d38c07f50ec5382cd54cd2f91b604d1",
    ),
    (
        "diffusion_pytorch_model-00009-of-00014.safetensors",
        4_933_368_232,
        "edfb38235adc96b99f55a401849befce59075a745e99c2d8c63ff358dd36443d",
    ),
    (
        "diffusion_pytorch_model-00010-of-00014.safetensors",
        4_567_069_608,
        "f8710775cf3413670edd7e23861b650a3431a71a6cc14cb1080623ab6b052385",
    ),
    (
        "diffusion_pytorch_model-00011-of-00014.safetensors",
        4_702_158_080,
        "9e18acc09f84edb5b34df9628efa15cfcab8bb76e8e20c1c2e979a107a0f7215",
    ),
    (
        "diffusion_pytorch_model-00012-of-00014.safetensors",
        4_933_368_232,
        "ea2e18228f8bdba1a4e0f32b155e4586df055997c45356213d05b971ba13e2f4",
    ),
    (
        "diffusion_pytorch_model-00013-of-00014.safetensors",
        4_567_069_608,
        "1e12083b1875678f7414ff55b09cd8bb1c30b861243f9bb7ff1e75b6ad3f1bdc",
    ),
    (
        "diffusion_pytorch_model-00014-of-00014.safetensors",
        4_644_161_920,
        "b340f44b5690cc745d48ae399381ec15b26a4fe25d483f677ccb4960dadb50d4",
    ),
)

EXPECTED_CONFIG = {
    "_class_name": "MiniMaxH3Transformer3DModel",
    "_diffusers_version": "0.36.0.dev0",
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,
    "rope_theta": 10_000.0,
    "norm_eps": 1.0e-5,
    "qk_norm_eps": 1.0e-5,
    "final_norm_eps": 1.0e-5,
}


def _dit_checkpoint_keys() -> tuple[str, ...]:
    names = [
        "proj_in.weight",
        "proj_in.bias",
        "audio_proj_in.weight",
        "audio_proj_in.bias",
        "context_embedder.weight",
        "context_embedder.bias",
        "token_refiner.final_norm.weight",
    ]
    for index in range(2):
        prefix = f"token_refiner.refiner_blocks.{index}"
        names.extend(
            (
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                *(f"{prefix}.attn.to_{name}.weight" for name in ("q", "k", "v")),
                f"{prefix}.attn.norm_q.weight",
                f"{prefix}.attn.norm_k.weight",
                f"{prefix}.attn.to_out.0.weight",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.2.weight",
            )
        )
    for index in range(50):
        prefix = f"transformer_blocks.{index}"
        names.extend(
            (
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                *(f"{prefix}.attn.to_{name}.weight" for name in ("q", "k", "v")),
                f"{prefix}.attn.norm_q.weight",
                f"{prefix}.attn.norm_k.weight",
                f"{prefix}.attn.to_out.0.weight",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.2.weight",
            )
        )
    names.extend(
        (
            "norm_out.norm.weight",
            "proj_out.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
        )
    )
    return tuple(names)


def _adaln_checkpoint_keys() -> tuple[str, ...]:
    names = [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    ]
    for index in range(50):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        names.extend((f"{prefix}.weight", f"{prefix}.bias"))
    names.extend(("norm_out.linear.weight", "norm_out.linear.bias"))
    return tuple(names)


REF2VA_DENOISER_KEYS = _dit_checkpoint_keys()
REF2VA_ADALN_KEYS = _adaln_checkpoint_keys()
REF2VA_ALL_KEYS = REF2VA_DENOISER_KEYS + REF2VA_ADALN_KEYS


def download_command(local_dir: str = "MiniMax-H3") -> str:
    """Return the build-time-only command for the missing public partition."""

    if not isinstance(local_dir, str) or not local_dir.strip():
        raise ValueError("MiniMax-H3 checkpoint destination must be a non-empty string")
    return (
        f"hf download {MODEL_ID} --revision {CHECKPOINT_REVISION} "
        f'--include "transformer_ref/*" --local-dir "{local_dir}"'
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(
    path: Path,
    *,
    size: int,
    sha256: str,
    hash_file: bool,
) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"MiniMax-H3 Ref2VA checkpoint file is missing: {path.name}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(
            f"MiniMax-H3 Ref2VA checkpoint size mismatch for {path.name}: "
            f"expected={size}, actual={actual_size}"
        )
    if not _SHA256.fullmatch(sha256):
        raise RuntimeError("MiniMax-H3 internal Ref2VA SHA-256 manifest is invalid")
    if hash_file:
        actual_sha = _sha256_file(path)
        if actual_sha != sha256:
            raise ValueError(
                f"MiniMax-H3 Ref2VA checkpoint SHA-256 mismatch for {path.name}: "
                f"expected={sha256}, actual={actual_sha}"
            )
    return {"bytes": size, "sha256": sha256}


@dataclass(frozen=True)
class TransformerRefIdentity:
    model_id: str
    revision: str
    component: str
    tensor_bytes: int
    tensor_count: int
    inventory_sha256: str
    files: dict[str, dict[str, int | str]]

    def bundle_metadata(self) -> dict[str, object]:
        """Path-free provenance safe to persist in the native bundle."""

        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "revision": self.revision,
            "component": self.component,
            "tensor_bytes": self.tensor_bytes,
            "tensor_count": self.tensor_count,
            "inventory_sha256": self.inventory_sha256,
            "files": self.files,
            "runtime_framework": None,
        }


def _canonical_inventory_sha256(files: Mapping[str, object]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_transformer_ref_checkpoint(component_dir: str | Path) -> TransformerRefIdentity:
    """Validate the released ``transformer_ref`` layout and plan partition.

    The pinned config and index are hashed; large shards are checked against
    the exact index names and released sizes without rereading 66 GiB solely
    for an integrity pass.
    """
    root = Path(component_dir)
    if root.name != COMPONENT_NAME:
        candidate = root / COMPONENT_NAME
        if candidate.is_dir():
            root = candidate
        else:
            raise FileNotFoundError(
                "MiniMax-H3 Ref2VA requires the distinct transformer_ref checkpoint. "
                "The T2VA transformer is not a valid fallback. Download it with: "
                f"{download_command(str(root))}"
            )
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(
            "MiniMax-H3 transformer_ref is absent; no T2VA fallback is permitted. "
            f"Download it with: {download_command(str(root.parent))}"
        )

    files: dict[str, dict[str, int | str]] = {}
    config_path = root / "config.json"
    files["config.json"] = _require_file(
        config_path,
        size=CONFIG_BYTES,
        sha256=CONFIG_SHA256,
        hash_file=True,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise ValueError(
            "MiniMax-H3 transformer_ref config does not match the released architecture"
        )

    index_name = "diffusion_pytorch_model.safetensors.index.json"
    index_path = root / index_name
    files[index_name] = _require_file(
        index_path,
        size=INDEX_BYTES,
        sha256=INDEX_SHA256,
        hash_file=True,
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ValueError("MiniMax-H3 transformer_ref index has an invalid schema")
    if index["metadata"] != {"total_size": TOTAL_TENSOR_BYTES}:
        raise ValueError("MiniMax-H3 transformer_ref index total_size is invalid")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise ValueError("MiniMax-H3 transformer_ref weight_map is invalid")
    expected_keys = set(REF2VA_ALL_KEYS)
    if len(REF2VA_ALL_KEYS) != 638 or len(expected_keys) != 638:
        raise RuntimeError("MiniMax-H3 internal transformer_ref partition is not exhaustive")
    actual_keys = set(weight_map)
    if actual_keys != expected_keys:
        raise ValueError(
            "MiniMax-H3 transformer_ref tensor partition mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:8]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )
    expected_shards = {name for name, _size, _sha in SHARDS}
    if set(weight_map.values()) != expected_shards:
        raise ValueError("MiniMax-H3 transformer_ref index does not reference the exact 14 shards")

    for name, size, sha256 in SHARDS:
        files[name] = _require_file(
            root / name,
            size=size,
            sha256=sha256,
            hash_file=False,
        )
    allowed = {"config.json", index_name, *expected_shards}
    unexpected_files = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.name not in allowed
    )
    if unexpected_files:
        raise ValueError(
            f"MiniMax-H3 transformer_ref contains unexpected files: {unexpected_files}"
        )
    return TransformerRefIdentity(
        model_id=MODEL_ID,
        revision=CHECKPOINT_REVISION,
        component=COMPONENT_NAME,
        tensor_bytes=TOTAL_TENSOR_BYTES,
        tensor_count=len(weight_map),
        inventory_sha256=_canonical_inventory_sha256(files),
        files=files,
    )

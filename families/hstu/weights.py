# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU checkpoint I/O: the tensor layout, a numpy-only BF16 safetensors
reader, and the seeded synthetic checkpoint generator.

No public trained HSTU ranking checkpoint exists (recsys-examples ships
training scripts only). ``generate`` writes a deterministic seeded one for the
reference KuaiRand-1K ranking configuration: ``config.json`` (which records
the name, origin, generator, seed and initializers) and ``model.safetensors``
in BF16. The same seed writes the same bytes; the build records the name,
generator and seed in the bundle, and the family's E2E golden logits fail on
any other weights. The values set the validation's numerics, never the
kernels' cost.

Usage: ``python -m families.hstu.weights <output_dir> [--seed 0]``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import HstuConfig, parse_config
from .support import ARCHITECTURE, MODEL_TYPE

WEIGHTS_FILE = "model.safetensors"
SYNTHETIC_ORIGIN = "synthetic"
GENERATOR = "families/hstu/weights.py generate (numpy PCG64)"

#: recsys-examples c6e2bb22 examples/hstu, KuaiRand-1K ranking inference:
#: NetworkArgs/RankingArgs from inference/configs/kuairand_1k_inference_ranking.gin,
#: tables from inference/inference_gr_ranking.py, positions from its
#: PositionEncodingConfig, scaling_seqlen = (max_history + max_candidates) * 2
#: + contextual (run_ranking_gr_evaluate's total_max_seqlen).
REFERENCE = {
    "repository": "https://github.com/NVIDIA/recsys-examples",
    "commit": "c6e2bb22350e7ece96df970aadee1c56fe87059f",
    "config": "examples/hstu/inference/configs/kuairand_1k_inference_ranking.gin",
    "forward": "examples/hstu/model/inference_ranking_gr.py InferenceRankingGR.forward_nokvcache",
    "dataset": "kuairand-1k",
}
REFERENCE_CONFIG = {
    "name": "hstu-kuairand-1k-synthetic",
    "model_type": MODEL_TYPE,
    "architectures": [ARCHITECTURE],
    "reference": REFERENCE,
    "hidden_size": 512,
    "num_layers": 3,
    "num_attention_heads": 4,
    "head_dim": 128,
    "layernorm_epsilon": 1e-5,
    "max_history_seqlen": 4096,
    "max_num_candidates": 100,
    "scaling_seqlen": (4096 + 100) * 2 + 6,
    "num_position_buckets": 8192,
    "contextual_features": [
        {"name": "user_id", "vocab_size": 1000},
        {"name": "user_active_degree", "vocab_size": 8},
        {"name": "follow_user_num_range", "vocab_size": 9},
        {"name": "fans_user_num_range", "vocab_size": 9},
        {"name": "friend_user_num_range", "vocab_size": 8},
        {"name": "register_days_range", "vocab_size": 8},
    ],
    "item_feature": {"name": "video_id", "vocab_size": 1_000_000},
    "action_feature": {"name": "action_weights", "vocab_size": 233},
    "prediction_head_arch": [512, 8],
    "prediction_head_act_type": "relu",
    "prediction_head_bias": True,
    "num_tasks": 8,
}
#: How each tensor class is drawn (recorded in config.json's ``weights``).
#: Every draw is uniform from PCG64's raw stream (``_symmetric``), whose
#: output NumPy guarantees across versions; the reference's normal
#: initializers keep their standard deviation as a uniform's.
INITIALIZERS = {
    "embedding": "uniform, std 0.02 (bound 0.02*sqrt(3)): a stand-in for the trained tables",
    "position": "uniform(+-sqrt(1/num_position_buckets)): HSTUPositionalEncoder",
    "hstu_linear": "uniform, std 0.02 (bound 0.02*sqrt(3)): Megatron's init_method std",
    "head_linear": "xavier_uniform: init_mlp_weights_optional_bias",
    "bias_and_norm_affine": (
        "uniform, std 0.02, around the reference's 0 (biases, norm bias) and 1 (norm "
        "weight), so validation exercises every parameter"
    ),
}
_CHUNK_ROWS = 65536


def tensor_layout(config: HstuConfig) -> list[tuple[str, tuple[int, ...], str]]:
    """Every tensor (name, shape, initializer class), in file order. Linear
    weights use PyTorch's [out_features, in_features] layout."""
    d = config.hidden_size
    layout = [
        (f"embedding.{feature.name}.weight", (feature.vocab_size, d), "embedding")
        for feature in (*config.contextual_features, config.item_feature, config.action_feature)
    ]
    layout.append(("position.weight", (config.num_position_buckets, d), "position"))
    for layer in range(config.num_layers):
        prefix = f"layers.{layer}"
        layout += [
            (f"{prefix}.input_layernorm.weight", (d,), "norm_weight"),
            (f"{prefix}.input_layernorm.bias", (d,), "bias"),
            (f"{prefix}.uvqk.weight", (config.uvqk_width, d), "hstu_linear"),
            (f"{prefix}.uvqk.bias", (config.uvqk_width,), "bias"),
            (f"{prefix}.output_layernorm.weight", (config.attention_width,), "norm_weight"),
            (f"{prefix}.output_layernorm.bias", (config.attention_width,), "bias"),
            (f"{prefix}.proj.weight", (d, config.attention_width), "hstu_linear"),
        ]
    width = d
    for index, out_features in enumerate(config.prediction_head_arch):
        layout += [
            (f"head.{index}.weight", (out_features, width), "head_linear"),
            (f"head.{index}.bias", (out_features,), "bias"),
        ]
        width = out_features
    return layout


def fp32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (round to nearest even), as uint16 bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _symmetric(rng: np.random.Generator, size: tuple[int, ...], bound: float) -> np.ndarray:
    """Uniform on (-bound, bound) in float32 from PCG64's raw 64-bit stream:
    the top 23 bits ``k`` of each draw map to the odd integer ``2k + 1 - 2**23``
    times ``2**-23``, exact in float32 and symmetric in (-1, 1), and the one
    float32 product rounds the same way on every IEEE platform, so the same
    seed writes the same bytes whatever NumPy version runs it (``Generator``'s
    own distributions carry no such guarantee)."""
    top = (rng.bit_generator.random_raw(math.prod(size)) >> np.uint64(41)).astype(np.int64)
    odd = (2 * top + 1 - (1 << 23)).astype(np.float32)  # |odd| < 2**23: exact
    return (odd * np.float32(2.0**-23) * np.float32(bound)).reshape(size)


def _draw(
    rng: np.random.Generator, kind: str, shape: tuple[int, ...], config: HstuConfig
) -> Iterator[np.ndarray]:
    rows = shape[0]
    tail = shape[1:]
    std = 0.02 * math.sqrt(3.0)  # a uniform's bound for standard deviation 0.02
    for start in range(0, rows, _CHUNK_ROWS):
        size = (min(_CHUNK_ROWS, rows - start), *tail)
        if kind in ("embedding", "hstu_linear", "bias"):
            chunk = _symmetric(rng, size, std)
        elif kind == "position":
            chunk = _symmetric(rng, size, math.sqrt(1.0 / config.num_position_buckets))
        elif kind == "head_linear":
            fan_out, fan_in = shape
            chunk = _symmetric(rng, size, math.sqrt(6.0 / (fan_in + fan_out)))
        elif kind == "norm_weight":
            chunk = np.float32(1.0) + _symmetric(rng, size, std)
        else:
            raise ValueError(f"unknown initializer class {kind!r}")
        yield fp32_to_bf16_bits(chunk)


def _header(layout: list[tuple[str, tuple[int, ...], str]], metadata: dict) -> bytes:
    entries: dict = {"__metadata__": metadata}
    offset = 0
    for name, shape, _ in layout:
        size = 2 * math.prod(shape)
        entries[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    text = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return text + b" " * (-len(text) % 8)


#: The validation's small geometry: the reference architecture with 16
#: history items, 8 candidates and small tables, where a candidate's own row
#: is a large share of its attention, so a mask, scaling or projection-split
#: defect moves the logits far past the tolerance (the measured geometry's
#: 4096 history items drown them).
SMALL_GEOMETRY = {
    "name": "hstu-kuairand-1k-synthetic-small",
    "max_history_seqlen": 16,
    "max_num_candidates": 8,
    "scaling_seqlen": (16 + 8) * 2 + 6,
    "item_feature": {"name": "video_id", "vocab_size": 4096},
}


def generate(output_dir: str | Path, *, seed: int = 0, small: bool = False) -> Path:
    """Write config.json and model.safetensors into ``output_dir`` and return
    it. ``small`` writes the validation's small geometry (``SMALL_GEOMETRY``)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = dict(REFERENCE_CONFIG, **(SMALL_GEOMETRY if small else {}))
    config = parse_config(raw)
    layout = tensor_layout(config)
    rng = np.random.Generator(np.random.PCG64(seed))
    header = _header(layout, {"format": "pt", "origin": SYNTHETIC_ORIGIN, "seed": str(seed)})
    with (output / WEIGHTS_FILE).open("wb") as out:
        out.write(struct.pack("<Q", len(header)) + header)
        for _, shape, kind in layout:
            for chunk in _draw(rng, kind, shape, config):
                out.write(chunk.tobytes())
    raw["weights"] = {
        "origin": SYNTHETIC_ORIGIN,
        "generator": GENERATOR,
        "seed": seed,
        "file": WEIGHTS_FILE,
        "dtype": "bf16",
        "initializers": INITIALIZERS,
    }
    (output / "config.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return output


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Memory-mapped BF16 tensors, as uint16 bits."""
    path = Path(path)
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + length)
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != "BF16":
            raise ValueError(f"{path}: tensor {name} is {entry['dtype']}, not BF16")
        begin, end = entry["data_offsets"]
        tensors[name] = data[begin:end].view(np.uint16).reshape(entry["shape"])
    return tensors


def weights_record(config: HstuConfig) -> dict:
    """``config.json``'s ``weights``: a synthetic checkpoint's origin,
    generator, seed and file, as ``generate`` wrote them."""
    weights = config.raw.get("weights")
    if not isinstance(weights, dict) or weights.get("file") != WEIGHTS_FILE:
        raise ValueError("HSTU config.json must record its weights file")
    if weights.get("origin") != SYNTHETIC_ORIGIN or weights.get("generator") != GENERATOR:
        raise ValueError(
            f"HSTU weights must be {GENERATOR}'s ({SYNTHETIC_ORIGIN}): no trained "
            "HSTU ranking checkpoint is public"
        )
    if type(weights.get("seed")) is not int:
        raise ValueError("HSTU config.json weights must record the generator's integer seed")
    return weights


def load_checkpoint(model_dir: str | Path, config: HstuConfig) -> dict[str, np.ndarray]:
    """The checkpoint's tensors, held to its configured layout."""
    model_dir = Path(model_dir)
    weights_record(config)
    tensors = read_safetensors(model_dir / WEIGHTS_FILE)
    layout = tensor_layout(config)
    if set(tensors) != {name for name, _, _ in layout}:
        raise ValueError("HSTU weights do not hold exactly the configured tensors")
    for name, shape, _ in layout:
        if tuple(tensors[name].shape) != shape or tensors[name].dtype != np.uint16:
            raise ValueError(f"HSTU tensor {name} must be BF16 {shape}")
    return tensors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--small", action="store_true", help="the validation's small geometry (SMALL_GEOMETRY)"
    )
    args = parser.parse_args(argv)
    print(generate(args.output_dir, seed=args.seed, small=args.small))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

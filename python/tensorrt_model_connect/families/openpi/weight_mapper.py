# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive OpenPI Orbax-to-TensorRT weight mapping.

Destination matrices use input-major layout (``[in, out]``), matching the
TensorRT matrix-multiply builders.  Prefix and action K/V matrices remain
compact MQA tensors rather than being expanded to the query-head count.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .checkpoint_reader import CheckpointReader, tensor_sha256
from .model_config import (
    OPENPI_UPSTREAM_COMMIT,
    OPENPI_UPSTREAM_REPOSITORY,
    OpenPIProfile,
)


CONVERSION_MANIFEST_SCHEMA_VERSION = 1


class WeightMappingError(RuntimeError):
    """Raised when mapping is incomplete, ambiguous, or shape-incompatible."""


@dataclass(frozen=True)
class DestinationTensor:
    name: str
    array: np.ndarray
    transform: str


Transform = Callable[[np.ndarray], Iterable[DestinationTensor]]


@dataclass(frozen=True)
class MappingRule:
    source: str
    expected_shape: tuple[int, ...]
    transform: Transform


@dataclass(frozen=True)
class MappingResult:
    weights: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _single(
    destination: str,
    *,
    transform_name: str = "identity",
    operation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Transform:
    def apply(array: np.ndarray) -> tuple[DestinationTensor, ...]:
        output = array if operation is None else operation(array)
        return (
            DestinationTensor(
                name=destination,
                array=np.ascontiguousarray(output),
                transform=transform_name,
            ),
        )

    return apply


def _scanned(
    depth: int,
    destination: str,
    *,
    transform_name: str = "slice-axis-0",
    operation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Transform:
    def apply(array: np.ndarray) -> list[DestinationTensor]:
        outputs: list[DestinationTensor] = []
        for layer in range(depth):
            value = array[layer]
            if operation is not None:
                value = operation(value)
            outputs.append(
                DestinationTensor(
                    name=destination.format(layer=layer),
                    array=np.ascontiguousarray(value),
                    transform=f"{transform_name}[{layer}]",
                )
            )
        return outputs

    return apply


def _scanned_pair(
    depth: int,
    first_destination: str,
    second_destination: str,
    *,
    labels: tuple[str, str],
) -> Transform:
    def apply(array: np.ndarray) -> list[DestinationTensor]:
        outputs: list[DestinationTensor] = []
        for layer in range(depth):
            for index, (destination, label) in enumerate(
                zip((first_destination, second_destination), labels, strict=True)
            ):
                outputs.append(
                    DestinationTensor(
                        name=destination.format(layer=layer),
                        array=np.ascontiguousarray(array[layer, index]),
                        transform=f"slice-axis-0[{layer}]-axis-1[{index}:{label}]",
                    )
                )
        return outputs

    return apply


def _gemma_q(width: int, attention_width: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda value: value.transpose(1, 0, 2).reshape(width, attention_width)


def _gemma_kv(which: int, width: int, kv_width: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda value: value[which].transpose(1, 0, 2).reshape(width, kv_width)


def _gemma_o(attention_width: int, width: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda value: value.reshape(attention_width, width)


def _vision_projection(width: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda value: value.reshape(width, width)


def _rules_for_gemma_expert(
    *,
    source_suffix: str,
    destination_prefix: str,
    profile: OpenPIProfile,
    adaptive_norm: bool,
) -> list[MappingRule]:
    cfg = profile.action_expert if destination_prefix == "action" else profile.prefix
    depth = cfg.depth
    suffix = source_suffix
    rules = [
        MappingRule(
            f"PaliGemma/llm/layers/attn/q_einsum{suffix}/w",
            (depth, cfg.num_heads, cfg.width, cfg.head_dim),
            _scanned(
                depth,
                f"{destination_prefix}.layer.{{layer}}.attention.q.weight",
                transform_name="slice-and-q-head-flatten",
                operation=_gemma_q(cfg.width, cfg.attention_width),
            ),
        ),
        MappingRule(
            f"PaliGemma/llm/layers/attn/kv_einsum{suffix}/w",
            (depth, 2, cfg.num_kv_heads, cfg.width, cfg.head_dim),
            lambda array: [
                DestinationTensor(
                    name=f"{destination_prefix}.layer.{layer}.attention.{kind}.weight",
                    array=np.ascontiguousarray(
                        _gemma_kv(which, cfg.width, cfg.kv_width)(array[layer])
                    ),
                    transform=f"slice-axis-0[{layer}]-{kind}-compact-kv",
                )
                for layer in range(depth)
                for which, kind in ((0, "k"), (1, "v"))
            ],
        ),
        MappingRule(
            f"PaliGemma/llm/layers/attn/attn_vec_einsum{suffix}/w",
            (depth, cfg.num_heads, cfg.head_dim, cfg.width),
            _scanned(
                depth,
                f"{destination_prefix}.layer.{{layer}}.attention.o.weight",
                transform_name="slice-and-output-head-flatten",
                operation=_gemma_o(cfg.attention_width, cfg.width),
            ),
        ),
        MappingRule(
            f"PaliGemma/llm/layers/mlp{suffix}/gating_einsum",
            (depth, 2, cfg.width, cfg.mlp_dim),
            _scanned_pair(
                depth,
                f"{destination_prefix}.layer.{{layer}}.mlp.gate.weight",
                f"{destination_prefix}.layer.{{layer}}.mlp.up.weight",
                labels=("gate", "up"),
            ),
        ),
        MappingRule(
            f"PaliGemma/llm/layers/mlp{suffix}/linear",
            (depth, cfg.mlp_dim, cfg.width),
            _scanned(depth, f"{destination_prefix}.layer.{{layer}}.mlp.down.weight"),
        ),
    ]
    if adaptive_norm:
        for source_name, destination_name in (
            ("pre_attention_norm", "pre_attention_norm"),
            ("pre_ffw_norm", "pre_ffw_norm"),
        ):
            rules.extend(
                [
                    MappingRule(
                        f"PaliGemma/llm/layers/{source_name}{suffix}/Dense_0/kernel",
                        (depth, cfg.width, 3 * cfg.width),
                        _scanned(
                            depth,
                            f"{destination_prefix}.layer.{{layer}}.{destination_name}.dense.weight",
                        ),
                    ),
                    MappingRule(
                        f"PaliGemma/llm/layers/{source_name}{suffix}/Dense_0/bias",
                        (depth, 3 * cfg.width),
                        _scanned(
                            depth,
                            f"{destination_prefix}.layer.{{layer}}.{destination_name}.dense.bias",
                        ),
                    ),
                ]
            )
        rules.extend(
            [
                MappingRule(
                    f"PaliGemma/llm/final_norm{suffix}/Dense_0/kernel",
                    (cfg.width, 3 * cfg.width),
                    _single(f"{destination_prefix}.final_norm.dense.weight"),
                ),
                MappingRule(
                    f"PaliGemma/llm/final_norm{suffix}/Dense_0/bias",
                    (3 * cfg.width,),
                    _single(f"{destination_prefix}.final_norm.dense.bias"),
                ),
            ]
        )
    else:
        rules.extend(
            [
                MappingRule(
                    f"PaliGemma/llm/layers/pre_attention_norm{suffix}/scale",
                    (depth, cfg.width),
                    _scanned(
                        depth,
                        f"{destination_prefix}.layer.{{layer}}.pre_attention_norm.scale",
                    ),
                ),
                MappingRule(
                    f"PaliGemma/llm/layers/pre_ffw_norm{suffix}/scale",
                    (depth, cfg.width),
                    _scanned(
                        depth,
                        f"{destination_prefix}.layer.{{layer}}.pre_ffw_norm.scale",
                    ),
                ),
                MappingRule(
                    f"PaliGemma/llm/final_norm{suffix}/scale",
                    (cfg.width,),
                    _single(f"{destination_prefix}.final_norm.scale"),
                ),
            ]
        )
    return rules


def openpi_mapping_rules(profile: OpenPIProfile) -> tuple[MappingRule, ...]:
    """Return the complete source inventory for audited π0.5 checkpoints."""

    vision = profile.vision
    rules: list[MappingRule] = [
        MappingRule(
            "PaliGemma/img/embedding/kernel",
            (vision.patch_size, vision.patch_size, 3, vision.width),
            _single(
                "vision.patch_embedding.weight",
                transform_name="hwio-to-oihw",
                operation=lambda value: value.transpose(3, 2, 0, 1),
            ),
        ),
        MappingRule(
            "PaliGemma/img/embedding/bias",
            (vision.width,),
            _single("vision.patch_embedding.bias"),
        ),
        MappingRule(
            "PaliGemma/img/pos_embedding",
            (1, vision.tokens_per_image, vision.width),
            _single(
                "vision.position_embedding",
                transform_name="squeeze-batch-axis",
                operation=lambda value: value[0],
            ),
        ),
    ]

    for upstream, destination in (
        ("LayerNorm_0/scale", "norm1.weight"),
        ("LayerNorm_0/bias", "norm1.bias"),
        ("LayerNorm_1/scale", "norm2.weight"),
        ("LayerNorm_1/bias", "norm2.bias"),
    ):
        rules.append(
            MappingRule(
                f"PaliGemma/img/Transformer/encoderblock/{upstream}",
                (vision.depth, vision.width),
                _scanned(vision.depth, f"vision.layer.{{layer}}.{destination}"),
            )
        )

    for upstream, destination, shape in (
        ("Dense_0/kernel", "fc1.weight", (vision.depth, vision.width, vision.mlp_dim)),
        ("Dense_0/bias", "fc1.bias", (vision.depth, vision.mlp_dim)),
        ("Dense_1/kernel", "fc2.weight", (vision.depth, vision.mlp_dim, vision.width)),
        ("Dense_1/bias", "fc2.bias", (vision.depth, vision.width)),
    ):
        rules.append(
            MappingRule(
                f"PaliGemma/img/Transformer/encoderblock/MlpBlock_0/{upstream}",
                shape,
                _scanned(vision.depth, f"vision.layer.{{layer}}.mlp.{destination}"),
            )
        )

    vision_head_dim = vision.width // vision.num_heads
    for kind in ("query", "key", "value"):
        short = {"query": "q", "key": "k", "value": "v"}[kind]
        rules.extend(
            [
                MappingRule(
                    f"PaliGemma/img/Transformer/encoderblock/"
                    f"MultiHeadDotProductAttention_0/{kind}/kernel",
                    (vision.depth, vision.width, vision.num_heads, vision_head_dim),
                    _scanned(
                        vision.depth,
                        f"vision.layer.{{layer}}.attention.{short}.weight",
                        transform_name="slice-and-head-flatten",
                        operation=_vision_projection(vision.width),
                    ),
                ),
                MappingRule(
                    f"PaliGemma/img/Transformer/encoderblock/"
                    f"MultiHeadDotProductAttention_0/{kind}/bias",
                    (vision.depth, vision.num_heads, vision_head_dim),
                    _scanned(
                        vision.depth,
                        f"vision.layer.{{layer}}.attention.{short}.bias",
                        transform_name="slice-and-head-flatten",
                        operation=lambda value, width=vision.width: value.reshape(width),
                    ),
                ),
            ]
        )
    rules.extend(
        [
            MappingRule(
                "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel",
                (vision.depth, vision.num_heads, vision_head_dim, vision.width),
                _scanned(
                    vision.depth,
                    "vision.layer.{layer}.attention.o.weight",
                    transform_name="slice-and-head-flatten",
                    operation=_vision_projection(vision.width),
                ),
            ),
            MappingRule(
                "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias",
                (vision.depth, vision.width),
                _scanned(vision.depth, "vision.layer.{layer}.attention.o.bias"),
            ),
            MappingRule(
                "PaliGemma/img/Transformer/encoder_norm/scale",
                (vision.width,),
                _single("vision.post_norm.weight"),
            ),
            MappingRule(
                "PaliGemma/img/Transformer/encoder_norm/bias",
                (vision.width,),
                _single("vision.post_norm.bias"),
            ),
            MappingRule(
                "PaliGemma/img/head/kernel",
                (vision.width, vision.output_width),
                _single("vision.projector.weight"),
            ),
            MappingRule(
                "PaliGemma/img/head/bias",
                (vision.output_width,),
                _single("vision.projector.bias"),
            ),
            MappingRule(
                "PaliGemma/llm/embedder/input_embedding",
                (profile.vocab_size, profile.prefix.width),
                _single("prefix.embedding"),
            ),
        ]
    )
    rules.extend(
        _rules_for_gemma_expert(
            source_suffix="",
            destination_prefix="prefix",
            profile=profile,
            adaptive_norm=False,
        )
    )
    rules.extend(
        _rules_for_gemma_expert(
            source_suffix="_1",
            destination_prefix="action",
            profile=profile,
            adaptive_norm=True,
        )
    )

    action = profile.action_expert
    for name, shape in (
        ("action_in", (profile.action_dim, action.width)),
        ("time_mlp_in", (action.width, action.width)),
        ("time_mlp_out", (action.width, action.width)),
        ("action_out", (action.width, profile.action_dim)),
    ):
        rules.extend(
            [
                MappingRule(
                    f"{name}_proj/kernel"
                    if name in {"action_in", "action_out"}
                    else f"{name}/kernel",
                    shape,
                    _single(f"projections.{name}.weight"),
                ),
                MappingRule(
                    f"{name}_proj/bias" if name in {"action_in", "action_out"} else f"{name}/bias",
                    (shape[1],),
                    _single(f"projections.{name}.bias"),
                ),
            ]
        )
    return tuple(rules)


def _dtype_is_float(dtype: np.dtype[Any]) -> bool:
    return dtype.kind == "f" or dtype.name == "bfloat16"


def _inventory_sha256(entries: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        encoded = (
            f"{entry['name']}\0{entry['dtype']}\0{','.join(map(str, entry['shape']))}"
            f"\0{entry['sha256']}"
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def map_weights(
    reader: CheckpointReader,
    profile: OpenPIProfile,
    *,
    rules: Sequence[MappingRule] | None = None,
) -> MappingResult:
    """Map a checkpoint and reject any non-exhaustive conversion."""

    selected_rules = tuple(rules if rules is not None else openpi_mapping_rules(profile))
    if not selected_rules:
        raise WeightMappingError("weight mapping contains no rules")

    rule_sources: set[str] = set()
    duplicate_rules: list[str] = []
    for rule in selected_rules:
        if rule.source in rule_sources:
            duplicate_rules.append(rule.source)
        rule_sources.add(rule.source)
    if duplicate_rules:
        raise WeightMappingError(
            "duplicate source mapping rules: " + ", ".join(sorted(set(duplicate_rules)))
        )

    resolved: list[tuple[MappingRule, str]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    consumed: set[str] = set()
    for rule in selected_rules:
        candidates = [name for name in (rule.source, f"{rule.source}/value") if name in reader]
        if not candidates:
            missing.append(rule.source)
            continue
        if len(candidates) != 1:
            ambiguous.append(f"{rule.source}: {candidates}")
            continue
        actual = candidates[0]
        if actual in consumed:
            ambiguous.append(f"{actual}: consumed by multiple rules")
            continue
        consumed.add(actual)
        resolved.append((rule, actual))

    unexpected = sorted(set(reader) - consumed)
    if missing or ambiguous or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(sorted(missing)))
        if ambiguous:
            details.append("ambiguous=" + "; ".join(sorted(ambiguous)))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise WeightMappingError("checkpoint mapping is not exhaustive: " + " | ".join(details))

    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    for rule, actual in resolved:
        array = reader[actual]
        shape = tuple(int(dim) for dim in array.shape)
        if shape != rule.expected_shape:
            shape_errors.append(f"{actual}: expected {rule.expected_shape}, got {shape}")
        if not _dtype_is_float(array.dtype):
            dtype_errors.append(f"{actual}: expected floating point, got {array.dtype.name}")
    if shape_errors or dtype_errors:
        raise WeightMappingError(
            "checkpoint tensor contract mismatch: " + " | ".join(shape_errors + dtype_errors)
        )

    weights: dict[str, np.ndarray] = {}
    destination_entries: list[dict[str, Any]] = []
    mapping_entries: list[dict[str, Any]] = []
    for rule, actual in resolved:
        source_record = reader.record(actual)
        destinations: list[dict[str, Any]] = []
        for mapped in rule.transform(source_record.array):
            if not mapped.name:
                raise WeightMappingError(f"mapping for {actual} emitted an empty destination name")
            if mapped.name in weights:
                raise WeightMappingError(f"duplicate destination mapping: {mapped.name}")
            array = np.ascontiguousarray(mapped.array)
            if not _dtype_is_float(array.dtype):
                raise WeightMappingError(
                    f"mapping for {actual} emitted non-floating destination {mapped.name}: {array.dtype}"
                )
            weights[mapped.name] = array
            destination = {
                "name": mapped.name,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "sha256": tensor_sha256(array),
                "source": actual,
                "transform": mapped.transform,
            }
            destination_entries.append(destination)
            destinations.append(destination)
        if not destinations:
            raise WeightMappingError(f"mapping for {actual} emitted no destination tensors")
        mapping_entries.append(
            {
                "expected_source": rule.source,
                "expected_shape": list(rule.expected_shape),
                "source": source_record.manifest_entry(),
                "destinations": destinations,
            }
        )

    destination_entries.sort(key=lambda item: item["name"])
    mapping_entries.sort(key=lambda item: item["expected_source"])
    source_entries = reader.manifest_inventory()
    manifest: dict[str, Any] = {
        "schema_version": CONVERSION_MANIFEST_SCHEMA_VERSION,
        "format": "trtmc.openpi.weight-conversion",
        "profile": profile.name,
        "policy": profile.to_dict(),
        "upstream": {
            "repository": OPENPI_UPSTREAM_REPOSITORY,
            "commit": OPENPI_UPSTREAM_COMMIT,
            "checkpoint_uri": profile.checkpoint_uri,
        },
        "source_checkpoint": {
            "tensor_count": len(source_entries),
            "identity_sha256": reader.identity_sha256,
        },
        "destination_weights": {
            "tensor_count": len(destination_entries),
            "identity_sha256": _inventory_sha256(destination_entries),
        },
        "source_tensors": source_entries,
        "destination_tensors": destination_entries,
        "mappings": mapping_entries,
    }
    return MappingResult(weights=weights, manifest=manifest)

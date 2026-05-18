"""Multi-device execution plan schema and selector helpers.

This module is the builder-side contract for distributed bundle metadata.  It
does not search for an optimal plan.  It records the concrete plan that TRT-MC
should validate, serialize, and eventually execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import json
from typing import Any


DISTRIBUTED_PLAN_SECTION = "distributed_plan.json"
DISTRIBUTED_PLAN_SCHEMA_VERSION = "1.0"
MESH_AXES = ("tp", "pp", "cp", "dp", "ep")


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if out < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return out


def _default_rank_mapping(axes: dict[str, int], world_size: int) -> list[dict[str, int]]:
    mapping: list[dict[str, int]] = []
    for rank in range(world_size):
        rem = rank
        entry: dict[str, int] = {"rank": rank}
        for axis in MESH_AXES:
            size = axes[axis]
            entry[axis] = rem % size
            rem //= size
        mapping.append(entry)
    return mapping


@dataclass
class DistributedConfig:
    """Process mesh and communication defaults for a distributed bundle."""

    world_size: int = 1
    axes: dict[str, int] = field(default_factory=dict)
    rank_mapping: list[dict[str, int]] = field(default_factory=list)
    collective_backend: str = "nccl"
    allreduce_strategy: str = "nccl"

    def __post_init__(self) -> None:
        self.world_size = _positive_int(self.world_size, field_name="world_size")
        normalized = {axis: _positive_int(self.axes.get(axis, 1), field_name=f"axes.{axis}")
                      for axis in MESH_AXES}
        unknown = sorted(set(self.axes) - set(MESH_AXES))
        if unknown:
            raise ValueError(f"Unsupported mesh axes: {', '.join(unknown)}")
        self.axes = normalized
        self.validate()
        if not self.rank_mapping:
            self.rank_mapping = _default_rank_mapping(self.axes, self.world_size)
        self._validate_rank_mapping()

    @classmethod
    def from_parallel_config(cls, parallel: Any) -> "DistributedConfig":
        """Build a one-dimensional TP mesh from the legacy ``ParallelConfig``."""
        tp_size = int(getattr(parallel, "tp_size", 1))
        return cls(world_size=tp_size, axes={"tp": tp_size})

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DistributedConfig":
        return cls(
            world_size=value.get("world_size", 1),
            axes=dict(value.get("axes", {})),
            rank_mapping=[dict(item) for item in value.get("rank_mapping", [])],
            collective_backend=str(value.get("collective_backend", "nccl")),
            allreduce_strategy=str(value.get("allreduce_strategy", "nccl")),
        )

    @property
    def tp_size(self) -> int:
        return self.axes["tp"]

    def validate(self) -> None:
        product = 1
        for axis in MESH_AXES:
            product *= self.axes[axis]
        if product != self.world_size:
            raise ValueError(
                f"Mesh axis product ({product}) must equal world_size ({self.world_size})"
            )

    def _validate_rank_mapping(self) -> None:
        if len(self.rank_mapping) != self.world_size:
            raise ValueError("rank_mapping length must equal world_size")
        seen: set[int] = set()
        seen_coords: set[tuple[int, ...]] = set()
        for entry in self.rank_mapping:
            rank = int(entry.get("rank", -1))
            if rank < 0 or rank >= self.world_size:
                raise ValueError(f"rank_mapping rank {rank} is outside [0, world_size)")
            if rank in seen:
                raise ValueError(f"rank_mapping rank {rank} is duplicated")
            seen.add(rank)
            coords: list[int] = []
            for axis in MESH_AXES:
                coord = int(entry.get(axis, 0))
                if coord < 0 or coord >= self.axes[axis]:
                    raise ValueError(
                        f"rank_mapping rank {rank} has invalid {axis} coordinate {coord}"
                    )
                coords.append(coord)
            coord_key = tuple(coords)
            if coord_key in seen_coords:
                raise ValueError(
                    f"rank_mapping mesh coordinates for rank {rank} are duplicated"
                )
            seen_coords.add(coord_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "axes": dict(self.axes),
            "rank_mapping": [dict(item) for item in self.rank_mapping],
            "collective_backend": self.collective_backend,
            "allreduce_strategy": self.allreduce_strategy,
        }


@dataclass
class ComponentPlan:
    """Placement for a high-level model component such as a decoder or denoiser."""

    placement: str
    mesh_axes: list[str] = field(default_factory=list)
    section: str | None = None
    rank_section_pattern: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ComponentPlan":
        return cls(
            placement=str(value.get("placement", "replicated")),
            mesh_axes=[str(axis) for axis in value.get("mesh_axes", [])],
            section=value.get("section"),
            rank_section_pattern=value.get("rank_section_pattern"),
        )

    def validate(self) -> None:
        if self.placement not in {"replicated", "sharded"}:
            raise ValueError(f"Unsupported component placement: {self.placement!r}")
        for axis in self.mesh_axes:
            if axis not in MESH_AXES:
                raise ValueError(f"Unsupported component mesh axis: {axis!r}")
        if self.placement == "replicated" and not self.section:
            raise ValueError("Replicated components must name a bundle section")
        if self.placement == "sharded" and not self.rank_section_pattern:
            raise ValueError("Sharded components must name a rank_section_pattern")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"placement": self.placement}
        if self.mesh_axes:
            out["mesh_axes"] = list(self.mesh_axes)
        if self.section:
            out["section"] = self.section
        if self.rank_section_pattern:
            out["rank_section_pattern"] = self.rank_section_pattern
        return out


@dataclass
class RegionPlan:
    """Partial sharding decision for one or more recipe regions."""

    selector: str
    policy: str
    tp_size: int | None = None
    mesh_axes: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RegionPlan":
        known = {"selector", "policy", "tp_size", "mesh_axes"}
        options = {k: v for k, v in value.items() if k not in known}
        return cls(
            selector=str(value["selector"]),
            policy=str(value["policy"]),
            tp_size=(int(value["tp_size"]) if "tp_size" in value else None),
            mesh_axes=[str(axis) for axis in value.get("mesh_axes", [])],
            options=options,
        )

    def validate(self) -> None:
        if not self.selector:
            raise ValueError("Region selector must be non-empty")
        if self.policy not in {"replicated", "tensor_parallel", "pipeline_parallel",
                               "context_parallel", "data_parallel", "expert_parallel"}:
            raise ValueError(f"Unsupported region policy: {self.policy!r}")
        if self.tp_size is not None and self.tp_size < 1:
            raise ValueError("Region tp_size must be >= 1")
        for axis in self.mesh_axes:
            if axis not in MESH_AXES:
                raise ValueError(f"Unsupported region mesh axis: {axis!r}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "selector": self.selector,
            "policy": self.policy,
        }
        if self.tp_size is not None:
            out["tp_size"] = self.tp_size
        if self.mesh_axes:
            out["mesh_axes"] = list(self.mesh_axes)
        out.update(self.options)
        return out


@dataclass
class CollectivePlan:
    """Semantic collective required by a distributed plan."""

    id: str
    type: str
    group: str
    op: str = "sum"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CollectivePlan":
        return cls(
            id=str(value["id"]),
            type=str(value["type"]),
            group=str(value["group"]),
            op=str(value.get("op", "sum")),
        )

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Collective id must be non-empty")
        if self.type not in {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
            raise ValueError(f"Unsupported collective type: {self.type!r}")
        if self.group not in MESH_AXES:
            raise ValueError(f"Unsupported collective group: {self.group!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "group": self.group,
            "op": self.op,
        }


@dataclass
class DistributedPlan:
    """Concrete distributed execution plan serialized into a ``.trtfb`` bundle."""

    mesh: DistributedConfig
    model: dict[str, Any] = field(default_factory=dict)
    components: dict[str, ComponentPlan] = field(default_factory=dict)
    regions: list[RegionPlan] = field(default_factory=list)
    collectives: list[CollectivePlan] = field(default_factory=list)
    bundle_sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    schema_version: str = DISTRIBUTED_PLAN_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DistributedPlan":
        return cls(
            schema_version=str(value.get("schema_version", DISTRIBUTED_PLAN_SCHEMA_VERSION)),
            model=dict(value.get("model", {})),
            mesh=DistributedConfig.from_dict(value.get("mesh", {})),
            components={
                name: ComponentPlan.from_dict(component)
                for name, component in value.get("components", {}).items()
            },
            regions=[RegionPlan.from_dict(item) for item in value.get("regions", [])],
            collectives=[
                CollectivePlan.from_dict(item) for item in value.get("collectives", [])
            ],
            bundle_sections={
                str(name): dict(section)
                for name, section in value.get("bundle_sections", {}).items()
            },
            constraints=dict(value.get("constraints", {})),
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "DistributedPlan":
        return cls.from_dict(json.loads(data.decode("utf-8")))

    def validate(self) -> None:
        if self.schema_version != DISTRIBUTED_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported distributed plan schema_version={self.schema_version!r}"
            )
        self.mesh.validate()
        for component in self.components.values():
            component.validate()
        for region in self.regions:
            region.validate()
        for collective in self.collectives:
            collective.validate()
        for name, section in self.bundle_sections.items():
            if not isinstance(section, dict):
                raise ValueError(f"bundle_sections.{name} must be an object")
            if "section" not in section and "rank_section_pattern" not in section:
                raise ValueError(
                    f"bundle_sections.{name} must name section or rank_section_pattern"
                )

    def validate_region_selectors(self, recipe_regions: list[str]) -> None:
        missing = [
            region.selector for region in self.regions
            if not resolve_selector(region.selector, recipe_regions)
        ]
        if missing:
            raise ValueError(
                "Distributed plan selectors matched no recipe regions: "
                + ", ".join(missing)
            )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "model": dict(self.model),
            "mesh": self.mesh.to_dict(),
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
            "regions": [region.to_dict() for region in self.regions],
            "collectives": [collective.to_dict() for collective in self.collectives],
            "bundle_sections": {
                name: dict(section)
                for name, section in self.bundle_sections.items()
            },
        }
        if self.constraints:
            out["constraints"] = dict(self.constraints)
        return out

    def to_json_bytes(self) -> bytes:
        self.validate()
        return json.dumps(self.to_dict(), indent=2, sort_keys=True).encode("utf-8")


def selector_matches(selector: str, region_name: str) -> bool:
    """Return whether a plan selector matches a model recipe region name.

    Selectors use recipe names with optional numeric layer ranges:
    ``decoder.layers[0:12].mlp`` matches ``decoder.layers.0.mlp`` through
    ``decoder.layers.11.mlp``. ``[*]`` matches any numeric index.
    """

    selector_parts = selector.split(".")
    region_parts = region_name.split(".")
    i = 0
    j = 0
    while i < len(selector_parts) and j < len(region_parts):
        part = selector_parts[i]
        if "[" in part and part.endswith("]"):
            prefix, bracket = part[:-1].split("[", 1)
            if j + 1 >= len(region_parts) or region_parts[j] != prefix:
                return False
            try:
                index = int(region_parts[j + 1])
            except ValueError:
                return False
            if bracket != "*":
                start_text, end_text = bracket.split(":", 1)
                start = int(start_text) if start_text else 0
                end = int(end_text)
                if index < start or index >= end:
                    return False
            i += 1
            j += 2
            continue

        if not fnmatchcase(region_parts[j], part):
            return False
        i += 1
        j += 1
    return i == len(selector_parts) and j == len(region_parts)


def resolve_selector(selector: str, recipe_regions: list[str]) -> list[str]:
    """Return recipe regions matched by ``selector`` in input order."""
    return [region for region in recipe_regions if selector_matches(selector, region)]

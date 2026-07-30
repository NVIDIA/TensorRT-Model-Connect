# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, family-owned slots for declarative TVM-FFI kernels."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml


class KernelSlotError(ValueError):
    """A kernel manifest does not conform to a family-owned slot."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str, ...] | str


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    type: str


@dataclass(frozen=True)
class KernelSlot:
    """The ABI that one model family exposes at graph-construction time."""

    id: str
    description: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    workspace_bytes: int
    instances: Callable[[Any], tuple[str, ...]]
    model_arguments: tuple[ArgumentSpec, ...] = ()
    validate_build: Callable[[Any], None] | None = None


@dataclass(frozen=True)
class KernelSpec:
    """The only data supplied by an existing-slot kernel user."""

    slot: str
    instance_ids: tuple[str, ...] | None
    expect_count: int
    library: Path
    library_sha256: str
    function: str

    @property
    def global_name(self) -> str:
        identity = f"{self.slot}\0{self.library_sha256}\0{self.function}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return f"trtmc.byok.{digest}"

    @property
    def kernel_artifact(self) -> tuple[str, str, str, str]:
        return (
            self.global_name,
            str(self.library),
            self.function,
            self.library_sha256,
        )


def _mapping(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise KernelSlotError(f"{where} must be a mapping")
    if any(type(key) is not str for key in value):
        raise KernelSlotError(f"{where} keys must be strings")
    unknown = sorted(set(value) - keys)
    if unknown:
        raise KernelSlotError(f"{where} contains unknown field(s): " + ", ".join(unknown))
    return value


def _string(value: Any, where: str) -> str:
    if type(value) is not str or not value:
        raise KernelSlotError(f"{where} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise KernelSlotError(f"Cannot read kernel library {path}: {exc}") from exc
    return digest.hexdigest()


def load_kernel_spec(path: str | Path) -> KernelSpec:
    """Parse one strict YAML document and verify its DSO digest."""

    manifest_path = Path(path)
    try:
        source = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise KernelSlotError(f"Cannot read kernel YAML {manifest_path}: {exc}") from exc
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise KernelSlotError(f"Invalid kernel YAML {manifest_path}: {exc}") from exc

    root = _mapping(
        document,
        "kernel YAML",
        {"schema_version", "slot", "instances", "kernel"},
    )
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise KernelSlotError("schema_version must be 1")

    selection = _mapping(
        root.get("instances"),
        "instances",
        {"all", "expect_count", "ids"},
    )
    select_all = selection.get("all")
    raw_ids = selection.get("ids")
    if select_all is True:
        if raw_ids is not None:
            raise KernelSlotError("instances cannot contain both all and ids")
        expect_count = selection.get("expect_count")
        if type(expect_count) is not int or expect_count < 1:
            raise KernelSlotError(
                "instances.expect_count must be a positive integer with all: true"
            )
        instance_ids = None
    else:
        if select_all is not None:
            raise KernelSlotError("instances.all, when present, must be true")
        if type(raw_ids) is not list or not raw_ids:
            raise KernelSlotError(
                "instances must use all: true plus expect_count, or a non-empty ids list"
            )
        instance_ids = tuple(
            _string(value, f"instances.ids[{index}]") for index, value in enumerate(raw_ids)
        )
        if len(set(instance_ids)) != len(instance_ids):
            raise KernelSlotError("instances.ids contains a duplicate")
        if "expect_count" in selection:
            raise KernelSlotError("instances.expect_count is only valid with all: true")
        expect_count = len(instance_ids)

    kernel = _mapping(
        root.get("kernel"),
        "kernel",
        {"library", "sha256", "function"},
    )
    library = Path(_string(kernel.get("library"), "kernel.library"))
    if not library.is_absolute():
        library = manifest_path.parent / library
    library = library.resolve()
    if not library.is_file():
        raise KernelSlotError(f"kernel.library is not a file: {library}")
    expected_digest = _string(kernel.get("sha256"), "kernel.sha256")
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise KernelSlotError("kernel.sha256 must be a lowercase 64-character SHA-256")
    actual_digest = _sha256_file(library)
    if actual_digest != expected_digest:
        raise KernelSlotError(
            f"Kernel library SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )

    return KernelSpec(
        slot=_string(root.get("slot"), "slot"),
        instance_ids=instance_ids,
        expect_count=expect_count,
        library=library,
        library_sha256=expected_digest,
        function=_string(kernel.get("function"), "kernel.function"),
    )


def load_family_kernel_slots(family: str) -> tuple[KernelSlot, ...]:
    """Load an optional ``kernel_slots.py`` module owned by one family."""

    family = _string(family, "family").replace("-", "_")
    if not family.isidentifier():
        raise KernelSlotError(f"Invalid model family {family!r}")
    root_package = __package__.rsplit(".", 1)[0]
    module_name = f"{root_package}.families.{family}.kernel_slots"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return ()
        raise
    slots = getattr(module, "SLOTS", ())
    if type(slots) is not tuple or any(type(slot) is not KernelSlot for slot in slots):
        raise KernelSlotError(f"{module_name}.SLOTS must be a tuple of KernelSlot values")
    ids = [slot.id for slot in slots]
    if len(ids) != len(set(ids)):
        raise KernelSlotError(f"{module_name}.SLOTS contains a duplicate slot ID")
    return slots


class _Selection:
    def __init__(self, spec: KernelSpec, slot: KernelSlot) -> None:
        self.spec = spec
        self.slot = slot
        self.seen: set[str] = set()

    def select(self, slot_id: str, instance_id: str) -> KernelSpec | None:
        if slot_id != self.slot.id:
            return None
        selected = self.spec.instance_ids is None or instance_id in self.spec.instance_ids
        if not selected:
            return None
        if instance_id in self.seen:
            raise KernelSlotError(f"Kernel slot instance {instance_id!r} was built more than once")
        self.seen.add(instance_id)
        return self.spec

    def finish(self) -> None:
        if self.spec.instance_ids is None:
            if len(self.seen) != self.spec.expect_count:
                raise KernelSlotError(
                    f"Slot {self.slot.id!r} matched {len(self.seen)} instances; "
                    f"YAML expected {self.spec.expect_count}"
                )
            return
        missing = sorted(set(self.spec.instance_ids) - self.seen)
        if missing:
            raise KernelSlotError(
                "Requested kernel slot instance(s) were not built: " + ", ".join(missing)
            )


_ACTIVE: ContextVar[_Selection | None] = ContextVar(
    "tensorrt_model_connect_kernel_slot",
    default=None,
)


@contextmanager
def activate_kernel_slot(
    spec: KernelSpec,
    slot: KernelSlot,
) -> Iterator[None]:
    """Activate one direct slot for exactly one native bundle build."""

    if _ACTIVE.get() is not None:
        raise KernelSlotError("Kernel slot activations cannot be nested")
    selection = _Selection(spec, slot)
    token = _ACTIVE.set(selection)
    try:
        yield
        selection.finish()
    finally:
        _ACTIVE.reset(token)


def active_kernel_artifact() -> tuple[str, str, str, str] | None:
    active = _ACTIVE.get()
    return active.spec.kernel_artifact if active is not None else None


def finish_active_kernel_slot() -> None:
    """Validate that the active slot was wired before writing the bundle."""

    active = _ACTIVE.get()
    if active is not None:
        active.finish()


def select_kernel_slot(slot_id: str, instance_id: str) -> KernelSpec | None:
    """Return the active kernel when this exact family slot instance is selected."""

    active = _ACTIVE.get()
    if active is None:
        return None
    return active.select(
        _string(slot_id, "slot_id"),
        _string(instance_id, "instance_id"),
    )

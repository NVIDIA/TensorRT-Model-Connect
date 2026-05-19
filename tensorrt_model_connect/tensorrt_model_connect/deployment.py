"""Deployment specialization manifests and artifact helpers.

The deployment manifest is intentionally provider-neutral.  It describes the
implementation variants bundled with a model and the artifact sections each
variant needs.  Runtime code can then select a runtime-scope provider such as
TensorRT Edge-LLM or remap kernel/component artifacts onto the normal native
runtime path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .bundle_writer import BundleSection

DEPLOYMENT_MANIFEST_SECTION = "deployment_manifest.json"


@dataclass(frozen=True)
class DeploymentArtifact:
    name: str
    kind: str = "bundle_section"
    section: str | None = None
    section_prefix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
        }
        if self.section:
            out["section"] = self.section
        if self.section_prefix:
            out["section_prefix"] = self.section_prefix
        return out


@dataclass(frozen=True)
class DeploymentVariant:
    id: str
    scope: str
    provider: str
    runtime_strategy: str = ""
    fallback: bool = False
    compatibility: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[DeploymentArtifact, ...] = ()
    performance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "scope": self.scope,
            "provider": self.provider,
        }
        if self.runtime_strategy:
            out["runtime_strategy"] = self.runtime_strategy
        if self.fallback:
            out["fallback"] = True
        if self.compatibility:
            out["compatibility"] = self.compatibility
        if self.artifacts:
            out["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        if self.performance:
            out["performance"] = self.performance
        return out


@dataclass(frozen=True)
class DeploymentManifest:
    target: dict[str, Any]
    default_variant: str
    selected_variant: str
    variants: tuple[DeploymentVariant, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "default_variant": self.default_variant,
            "selected_variant": self.selected_variant,
            "variants": [variant.to_dict() for variant in self.variants],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False).encode("utf-8")


def manifest_section(manifest: DeploymentManifest) -> BundleSection:
    return BundleSection(DEPLOYMENT_MANIFEST_SECTION, manifest.to_json_bytes())


def portable_default_variant(runtime_strategy: str) -> DeploymentVariant:
    return DeploymentVariant(
        id="portable_default",
        scope="runtime",
        provider="native_trt",
        runtime_strategy=runtime_strategy,
        fallback=True,
        artifacts=(DeploymentArtifact(name="engine_plan", section="engine_plan"),),
    )


def _performance_record(
    performance: dict[str, Any] | None,
    *,
    target: str,
    variant_id: str,
    provider: str,
    scope: str,
) -> dict[str, Any]:
    if not performance:
        return {}
    record = dict(performance)
    record.setdefault("target_id", target or "generic")
    record.setdefault("variant_id", variant_id)
    record.setdefault("provider", provider)
    record.setdefault("scope", scope)
    return record


def ffi_attention_manifest(
    *,
    target: str,
    runtime_strategy: str,
    selected_variant: str = "ffi_attention",
    selected_engine_section: str = "deployment/variants/ffi_attention/engine_plan",
    selected_kernel_manifest_section: str = (
        "deployment/variants/ffi_attention/kernel_manifest.json"
    ),
    performance: dict[str, Any] | None = None,
) -> DeploymentManifest:
    target_id = target or "generic"
    return DeploymentManifest(
        target={"platform": target_id, "objective": "best_perf_memory"},
        default_variant="portable_default",
        selected_variant=selected_variant,
        variants=(
            portable_default_variant(runtime_strategy),
            DeploymentVariant(
                id=selected_variant,
                scope="kernel",
                provider="tvm_ffi",
                runtime_strategy=runtime_strategy,
                compatibility={"platform": [target_id]},
                artifacts=(
                    DeploymentArtifact(
                        name="engine_plan",
                        kind="bundle_section",
                        section=selected_engine_section,
                    ),
                    DeploymentArtifact(
                        name="kernel_manifest.json",
                        kind="bundle_section",
                        section=selected_kernel_manifest_section,
                    ),
                ),
                performance=_performance_record(
                    performance,
                    target=target_id,
                    variant_id=selected_variant,
                    provider="tvm_ffi",
                    scope="kernel",
                ),
            ),
        ),
    )


def edge_llm_manifest(
    *,
    target: str,
    engine_section_prefix: str,
    selected_variant: str = "edge_llm",
    performance: dict[str, Any] | None = None,
) -> DeploymentManifest:
    target_id = target or "generic"
    return DeploymentManifest(
        target={"platform": target_id, "objective": "best_perf_memory"},
        default_variant=selected_variant,
        selected_variant=selected_variant,
        variants=(
            DeploymentVariant(
                id=selected_variant,
                scope="runtime",
                provider="tensorrt-edge-llm",
                runtime_strategy="text_generation",
                fallback=True,
                compatibility={"platform": [target_id]},
                artifacts=(
                    DeploymentArtifact(
                        name="engine_dir",
                        kind="directory",
                        section_prefix=engine_section_prefix,
                    ),
                ),
                performance=_performance_record(
                    performance,
                    target=target_id,
                    variant_id=selected_variant,
                    provider="tensorrt-edge-llm",
                    scope="runtime",
                ),
            ),
        ),
    )


def parse_kernel_artifacts(spec: str | None) -> list[tuple[str, str]]:
    """Parse ``global_name=/path/to/kernel.so`` pairs.

    Multiple pairs may be separated by comma or semicolon.  The parser keeps
    paths as strings so callers can decide when to read/validate the files.
    """
    if not spec:
        return []
    artifacts: list[tuple[str, str]] = []
    for item in re.split(r"[;,]", spec):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "deployment.ffi_kernel_artifacts expects "
                "global_name=/path/to/kernel.so entries"
            )
        global_name, _, path = item.partition("=")
        global_name = global_name.strip()
        path = path.strip()
        if not global_name or not path:
            raise ValueError(
                "deployment.ffi_kernel_artifacts entries require both "
                "global name and path"
            )
        artifacts.append((global_name, path))
    return artifacts


def section_name_for_kernel(global_name: str) -> str:
    return f"kernel_{global_name.replace('.', '_')}.so"


def kernel_sections(
    artifacts: Iterable[tuple[str, str]],
    *,
    section_prefix: str = "",
) -> tuple[list[BundleSection], bytes]:
    manifest_entries = []
    sections: list[BundleSection] = []
    for global_name, so_path in artifacts:
        section_base = section_name_for_kernel(global_name)
        section_name = f"{section_prefix}{section_base}"
        so_data = Path(so_path).read_bytes()
        sections.append(BundleSection(section_name, so_data))
        manifest_entries.append({
            "global_name": global_name,
            "func_name": "run",
            "section": section_name,
        })
    manifest_json = json.dumps({"kernels": manifest_entries}, indent=2).encode("utf-8")
    return sections, manifest_json


def directory_sections(
    directory: str | Path,
    *,
    section_prefix: str,
) -> list[BundleSection]:
    """Convert a directory tree into bundle sections under ``section_prefix``."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Expected directory artifact, got: {root}")
    if not section_prefix.endswith("/"):
        section_prefix += "/"

    sections: list[BundleSection] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("../") or rel == ".." or os.path.isabs(rel):
            raise ValueError(f"Unsafe directory artifact path: {rel}")
        sections.append(BundleSection(f"{section_prefix}{rel}", path.read_bytes()))
    if not sections:
        raise ValueError(f"Directory artifact is empty: {root}")
    return sections

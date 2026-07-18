# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic build-time routing for isolated runtime-provider capsules."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provider_process import (
    _ACTIVE_CUDA_DEVICE_ENV,
    ProbeResult,
    run_build,
    run_probe,
)
from .bundle import write_optimized_bundle
from .manifest import (
    AmbiguousImplementationError,
    ImplementationManifest,
    ImplementationRequest,
    ManifestDiscoveryError,
    discover_implementations_for_model,
)
from .target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


@dataclass(frozen=True)
class DelegatedBuildSelection:
    manifest: ImplementationManifest
    request: ImplementationRequest
    probe: ProbeResult
    # Transient process-launch context. This never enters request serialization,
    # manifest matching, build binding, or bundle metadata.
    _adapter_environment: Mapping[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def family_implementation_root(family_name: str) -> Path | None:
    """Return the selected model family's builder-owned discovery root."""

    if (
        not isinstance(family_name, str)
        or not family_name
        or not family_name.replace("_", "").isalnum()
    ):
        raise ValueError(f"Invalid model family name: {family_name!r}")
    families_root = Path(__file__).resolve().parents[1] / "families"
    family_root = families_root / family_name
    if not family_root.is_dir():
        return None
    return family_root


def discover_family_implementations_for_model(
    family_name: str,
    model_id: str,
) -> tuple[ImplementationManifest, ...]:
    """Discover candidates only inside the request's owning model family."""

    family_root = family_implementation_root(family_name)
    if family_root is None:
        return ()
    manifests = discover_implementations_for_model(family_root, model_id)
    resolved_family_root = family_root.resolve()
    for manifest in manifests:
        try:
            manifest.path.relative_to(resolved_family_root)
        except ValueError as exc:  # Defensive; discovery is rooted above.
            raise ManifestDiscoveryError(
                f"Model-family adapter manifest escapes {resolved_family_root}: {manifest.path}"
            ) from exc
    return manifests


def _model_source_identity(model_ref: str) -> tuple[str, str] | None:
    """Return the model and revision encoded by a canonical HF cache snapshot."""

    candidate = Path(model_ref).expanduser()
    if not candidate.exists():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    if resolved.parent.name != "snapshots":
        return None
    repository = resolved.parent.parent.name
    if not repository.startswith("models--"):
        return None
    components = repository[len("models--") :].split("--")
    if len(components) != 2 or any(not component for component in components):
        return None
    return "/".join(components), resolved.name.lower()


def select_delegated_build(
    manifests: Iterable[ImplementationManifest],
    request: ImplementationRequest,
    *,
    _adapter_environment: Mapping[str, str] | None = None,
) -> DelegatedBuildSelection | None:
    """Probe exact candidates and enforce one authoritative production path."""

    candidates = sorted(
        (manifest for manifest in manifests if manifest.matches(request)),
        key=lambda manifest: (manifest.implementation_id, str(manifest.path)),
    )
    supported: list[DelegatedBuildSelection] = []
    for manifest in candidates:
        probe = run_probe(
            manifest,
            request,
            _adapter_environment=_adapter_environment,
        )
        if probe.supported:
            supported.append(
                DelegatedBuildSelection(
                    manifest=manifest,
                    request=request,
                    probe=probe,
                    _adapter_environment=_adapter_environment,
                )
            )
    if not supported:
        return None
    if len(supported) > 1:
        details = ", ".join(
            f"{item.manifest.implementation_id} ({item.probe.profile_id})" for item in supported
        )
        raise AmbiguousImplementationError(
            f"Multiple optimized implementations claim the same deployment profile: {details}"
        )
    return supported[0]


def build_selected_implementation(
    selection: DelegatedBuildSelection,
    output_path: str | Path,
) -> Path:
    """Run one selected capsule and atomically publish its self-contained bundle."""

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".trtmc-{selection.manifest.implementation_id}-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / "capsule-output"
        artifact = run_build(
            selection.manifest,
            selection.request,
            staging,
            probe=selection.probe,
            _adapter_environment=selection._adapter_environment,
        )
        write_optimized_bundle(
            output,
            selection.manifest,
            selection.request,
            artifact,
        )
    return output


def try_build_optimized_runtime(
    model_ref: str,
    output_path: str | Path,
    *,
    family_name: str,
    parameters: Mapping[str, Any] | None = None,
) -> DelegatedBuildSelection | None:
    """Select and build one capsule; return None when no supported profile matches."""

    identity = _model_source_identity(model_ref)
    if identity is None:
        return None
    model_id, model_revision = identity
    available = discover_family_implementations_for_model(family_name, model_id)
    if not available:
        return None
    try:
        target_facts, active_device = _probe_current_target_with_device()
    except TargetResolutionError:
        # The unchanged public API implicitly targets the active device. If it
        # cannot be described for optimized-runtime selection, no capsule is
        # selected and the caller must retain the existing native path.
        return None
    merged_parameters = dict(parameters or {})
    merged_parameters.setdefault("model_source", model_ref)
    request = ImplementationRequest(
        model_id=model_id,
        model_revision=model_revision,
        target=target_facts,
        parameters=merged_parameters,
    )
    adapter_environment = {
        _ACTIVE_CUDA_DEVICE_ENV: str(active_device),
    }
    selection = select_delegated_build(
        available,
        request,
        _adapter_environment=adapter_environment,
    )
    if selection is None:
        return None
    build_selected_implementation(selection, output_path)
    return selection

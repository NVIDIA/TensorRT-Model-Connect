# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic build-time routing for isolated optimized-runtime capsules."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .build_adapter import (
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
    TargetScalar,
    discover_implementations_for_model,
    matching_implementations,
)
from .target import (
    TargetResolutionError,
    _probe_current_target_with_device,
    probe_current_target,
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


def family_implementation_roots(family_name: str) -> tuple[Path, ...]:
    """Return the one selected model family's builder-owned discovery root."""

    if (
        not isinstance(family_name, str)
        or not family_name
        or not family_name.replace("_", "").isalnum()
    ):
        raise ValueError(f"Invalid model family name: {family_name!r}")
    families_root = Path(__file__).resolve().parents[1] / "families"
    family_root = families_root / family_name
    if not family_root.is_dir():
        return ()
    return (family_root,)


def discover_family_implementations_for_model(
    family_name: str,
    model_id: str,
) -> tuple[ImplementationManifest, ...]:
    """Discover candidates only inside the request's owning model family."""

    roots = family_implementation_roots(family_name)
    if not roots:
        return ()
    manifests = discover_implementations_for_model(roots, model_id)
    family_root = roots[0].resolve()
    for manifest in manifests:
        try:
            relative = manifest.path.relative_to(family_root)
        except ValueError as exc:  # Defensive; discovery is rooted above.
            raise ManifestDiscoveryError(
                f"Model-family adapter manifest escapes {family_root}: {manifest.path}"
            ) from exc
        if len(relative.parts) != 2 or relative.name != "IMPLEMENTATION.toml":
            raise ManifestDiscoveryError(
                "Model-family adapters must use "
                "<family>/<adapter>/IMPLEMENTATION.toml: "
                f"{manifest.path}"
            )
    return manifests


def _model_source_identity(model_ref: str) -> tuple[str, str | None]:
    """Resolve only canonical HF cache snapshots; arbitrary local paths stay native."""

    candidate = Path(model_ref).expanduser()
    if not candidate.exists():
        if candidate.is_absolute():
            raise ValueError(f"Configured local model path does not exist: {candidate}")
        return model_ref, None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Unable to resolve local model source {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError(f"Local model source is not a directory: {resolved}")
    if resolved.parent.name != "snapshots":
        return model_ref, None
    repository = resolved.parent.parent.name
    if not repository.startswith("models--"):
        return model_ref, None
    components = repository[len("models--") :].split("--")
    if len(components) != 2 or any(not component for component in components):
        return model_ref, None
    return "/".join(components), resolved.name.lower()


def _probe_model_default_revision(model_id: str) -> str | None:
    """Resolve the immutable revision behind a bare Hugging Face model ID.

    Capsule profiles are revision-pinned. Selecting one from the set of
    installed manifests would silently redirect a mutable public model ID to
    stale weights after its upstream default branch advances.  A failed or
    incomplete lookup therefore means that no optimized profile is selected;
    the established native path retains control.
    """

    try:
        from huggingface_hub import HfApi

        revision = str(HfApi().model_info(model_id).sha or "").strip().lower()
        if revision:
            return revision
    except Exception:
        pass

    # Model-proof CI is intentionally network-disabled after warming the exact
    # Hugging Face snapshot. Resolve that cached ``main`` ref through the Hub
    # client, then accept only its canonical models--org--name/snapshots/<sha>
    # layout. The caller still requires the SHA to match an installed capsule.
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id, local_files_only=True)
        cached_model_id, cached_revision = _model_source_identity(str(snapshot))
    except Exception:
        return None
    return cached_revision if cached_model_id == model_id else None


def _resolve_revision(
    manifests: Iterable[ImplementationManifest],
    model_id: str,
    snapshot_revision: str | None,
    model_revision_probe: Callable[[str], str | None],
) -> str | None:
    if snapshot_revision is not None:
        return snapshot_revision
    revisions = {
        revision
        for manifest in manifests
        if manifest.matches_model(model_id)
        for revision in manifest.model_revisions
    }
    resolved = str(model_revision_probe(model_id) or "").strip().lower()
    return resolved if resolved in revisions else None


def make_implementation_request(
    model_ref: str,
    *,
    parameters: Mapping[str, Any] | None,
    manifests: Sequence[ImplementationManifest],
    current_target_probe=None,
    model_revision_probe: Callable[[str], str | None] | None = None,
) -> ImplementationRequest | None:
    """Normalize the public build request, or return None when no capsule can own it."""

    model_id, snapshot_revision = _model_source_identity(model_ref)
    model_manifests = [manifest for manifest in manifests if manifest.matches_model(model_id)]
    if not model_manifests:
        return None
    revision = _resolve_revision(
        model_manifests,
        model_id,
        snapshot_revision,
        model_revision_probe or _probe_model_default_revision,
    )
    if revision is None:
        return None
    target_facts: Mapping[str, TargetScalar] = dict(
        (current_target_probe or probe_current_target)()
    )
    merged_parameters = dict(parameters or {})
    merged_parameters.setdefault("model_source", model_ref)
    return ImplementationRequest(
        model_id=model_id,
        model_revision=revision,
        target=target_facts,
        parameters=merged_parameters,
    )


def select_delegated_build(
    manifests: Iterable[ImplementationManifest],
    request: ImplementationRequest,
    *,
    _adapter_environment: Mapping[str, str] | None = None,
) -> DelegatedBuildSelection | None:
    """Probe exact candidates and enforce one authoritative production path."""

    candidates = matching_implementations(manifests, request)
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
    family_name: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    manifests: Sequence[ImplementationManifest] | None = None,
    current_target_probe=None,
    model_revision_probe: Callable[[str], str | None] | None = None,
) -> DelegatedBuildSelection | None:
    """Select and build one capsule; return None when no supported profile matches."""

    if manifests is None:
        if not family_name:
            return None
        model_id, _ = _model_source_identity(model_ref)
        available = discover_family_implementations_for_model(family_name, model_id)
    else:
        available = tuple(manifests)
    adapter_environment: Mapping[str, str] | None = None
    try:
        effective_current_probe = current_target_probe
        captured_active_device: list[int] = []
        if current_target_probe is None:
            def probe_with_launch_context() -> Mapping[str, TargetScalar]:
                current_facts, active_device = _probe_current_target_with_device()
                captured_active_device.append(active_device)
                return current_facts

            effective_current_probe = probe_with_launch_context
        request = make_implementation_request(
            model_ref,
            parameters=parameters,
            manifests=available,
            current_target_probe=effective_current_probe,
            model_revision_probe=model_revision_probe,
        )
        if captured_active_device:
            adapter_environment = {
                _ACTIVE_CUDA_DEVICE_ENV: str(captured_active_device[0]),
            }
    except TargetResolutionError:
        # The unchanged public API implicitly targets the active device. If it
        # cannot be described for optimized-runtime selection, no capsule is
        # selected and the caller must retain the existing native path.
        return None
    if request is None:
        return None
    selection = select_delegated_build(
        available,
        request,
        _adapter_environment=adapter_environment,
    )
    if selection is None:
        return None
    build_selected_implementation(selection, output_path)
    return selection

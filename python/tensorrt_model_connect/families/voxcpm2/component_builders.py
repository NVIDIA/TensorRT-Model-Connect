"""Native VoxCPM2 component-builder boundary.

VoxCPM2 is assembled from five native stages. This module gives each stage a
dedicated TensorRT builder entry point and keeps the expected runtime bindings
next to the Python build surface. The individual graph builders still need to
be implemented with TensorRT network construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ...config import ModelConfig


@dataclass(frozen=True)
class VoxCPM2ComponentSpec:
    name: str
    engine_section: str
    input_artifact: str
    output_artifact: str


@dataclass(frozen=True)
class VoxCPM2ComponentBuildContext:
    spec: VoxCPM2ComponentSpec
    model_dir: Path
    config: ModelConfig
    source: Any
    precision: str
    verbose: bool


VoxCPM2ComponentBuilder = Callable[[VoxCPM2ComponentBuildContext], bytes]


VOXCPM2_COMPONENT_SPECS: tuple[VoxCPM2ComponentSpec, ...] = (
    VoxCPM2ComponentSpec(
        "locenc",
        "locenc_engine_plan",
        "text_utf8",
        "local_text_features",
    ),
    VoxCPM2ComponentSpec(
        "tslm",
        "tslm_engine_plan",
        "local_text_features",
        "semantic_lm_states",
    ),
    VoxCPM2ComponentSpec(
        "ralm",
        "ralm_engine_plan",
        "semantic_lm_states",
        "acoustic_residual_states",
    ),
    VoxCPM2ComponentSpec(
        "locdit",
        "locdit_engine_plan",
        "acoustic_residual_states",
        "audio_vae_latents",
    ),
    VoxCPM2ComponentSpec(
        "audiovae",
        "audiovae_engine_plan",
        "audio_vae_latents",
        "waveform_f32",
    ),
)


def _describe_source(source: Any) -> str:
    config_keys = ", ".join(getattr(source, "config_keys", ())) or "<none>"
    weight_files = ", ".join(getattr(source, "weight_files", ())) or "<none>"
    asset_files = ", ".join(getattr(source, "asset_files", ())) or "<none>"
    return f"config: {config_keys}; weights: {weight_files}; assets: {asset_files}"


def _raise_native_builder_gap(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    raise NotImplementedError(
        "VoxCPM2 native TRT builder for component "
        f"{ctx.spec.name!r} is not implemented yet. It must build "
        f"{ctx.spec.engine_section!r} with input binding "
        f"{ctx.spec.input_artifact!r} and output binding "
        f"{ctx.spec.output_artifact!r}. Discovered source inputs: "
        f"{_describe_source(ctx.source)}."
    )


def build_locenc_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx)


def build_tslm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx)


def build_ralm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx)


def build_locdit_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx)


def build_audiovae_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx)


DEFAULT_COMPONENT_BUILDERS: dict[str, VoxCPM2ComponentBuilder] = {
    "locenc": build_locenc_engine,
    "tslm": build_tslm_engine,
    "ralm": build_ralm_engine,
    "locdit": build_locdit_engine,
    "audiovae": build_audiovae_engine,
}


def build_voxcpm2_component_plans(
    model_dir: Path,
    config: ModelConfig,
    sources: Mapping[str, Any],
    *,
    precision: str,
    verbose: bool,
    builders: Mapping[str, VoxCPM2ComponentBuilder] | None = None,
) -> dict[str, bytes]:
    """Build every VoxCPM2 component plan and return bundle sections."""
    selected_builders = builders or DEFAULT_COMPONENT_BUILDERS
    sections: dict[str, bytes] = {}
    for spec in VOXCPM2_COMPONENT_SPECS:
        if spec.name not in sources:
            raise NotImplementedError(
                "VoxCPM2 raw checkpoint is missing source inputs for "
                f"component {spec.name!r}"
            )
        if spec.name not in selected_builders:
            raise NotImplementedError(
                "VoxCPM2 native TRT builder registry is missing component "
                f"{spec.name!r}"
            )

        ctx = VoxCPM2ComponentBuildContext(
            spec=spec,
            model_dir=model_dir,
            config=config,
            source=sources[spec.name],
            precision=precision,
            verbose=verbose,
        )
        plan = selected_builders[spec.name](ctx)
        if not isinstance(plan, (bytes, bytearray, memoryview)):
            raise TypeError(
                "VoxCPM2 native TRT builder for component "
                f"{spec.name!r} returned {type(plan).__name__}, expected bytes"
            )
        plan_bytes = bytes(plan)
        if not plan_bytes:
            raise ValueError(
                "VoxCPM2 native TRT builder for component "
                f"{spec.name!r} returned an empty plan"
            )
        sections[spec.engine_section] = plan_bytes
    return sections

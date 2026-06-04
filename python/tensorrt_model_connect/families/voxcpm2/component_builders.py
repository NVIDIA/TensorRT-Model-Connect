"""Native VoxCPM2 component-builder boundary.

VoxCPM2 is assembled from five native stages. This module gives each stage a
dedicated TensorRT builder entry point and keeps the expected runtime bindings
next to the Python build surface. The individual graph builders still need to
be implemented with TensorRT network construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...config import ModelConfig


@dataclass(frozen=True)
class VoxCPM2TensorSpec:
    name: str
    dtype_contract: tuple[str, ...]
    rank: int
    symbolic_shape: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"{self.name}:"
            f"{'|'.join(self.dtype_contract)}"
            f"[{', '.join(self.symbolic_shape)}]"
        )


@dataclass(frozen=True)
class VoxCPM2ComponentSpec:
    name: str
    engine_section: str
    input_artifact: str
    output_artifact: str
    input_tensor: VoxCPM2TensorSpec
    output_tensor: VoxCPM2TensorSpec


@dataclass(frozen=True)
class VoxCPM2ComponentBuildContext:
    spec: VoxCPM2ComponentSpec
    model_dir: Path
    config: ModelConfig
    source: Any
    precision: str
    verbose: bool

    @property
    def weight_paths(self) -> tuple[Path, ...]:
        return _resolve_model_files(self.model_dir, getattr(self.source, "weight_files", ()))

    @property
    def asset_paths(self) -> tuple[Path, ...]:
        return _resolve_model_files(self.model_dir, getattr(self.source, "asset_files", ()))

    def load_safetensor(self, tensor_name: str) -> Any:
        """Load one tensor from the component's safetensors checkpoint."""
        if not self.weight_paths:
            raise FileNotFoundError(
                "VoxCPM2 component "
                f"{self.spec.name!r} has no safetensors weight files"
            )
        from ...checkpoint_mapper import _load_tensor, _open_safetensors

        return _load_tensor(_open_safetensors(self.model_dir), tensor_name)

    def safetensor_keys(self) -> tuple[str, ...]:
        """Return tensor names visible from this component's safetensors input."""
        if not self.weight_paths:
            raise FileNotFoundError(
                "VoxCPM2 component "
                f"{self.spec.name!r} has no safetensors weight files"
            )
        from ...checkpoint_mapper import _open_safetensors

        readers = _open_safetensors(self.model_dir)
        tensor_map = getattr(readers, "tensor_map", None)
        if tensor_map is not None:
            return tuple(sorted(str(key) for key in tensor_map))
        keys: set[str] = set()
        for reader in readers:
            keys.update(str(key) for key in reader.keys())
        return tuple(sorted(keys))

    def load_safetensor_group(
        self, prefixes: Sequence[str] | None = None
    ) -> Mapping[str, Any]:
        """Load all safetensors entries for the component state-dict prefixes."""
        selected_prefixes = tuple(
            prefixes
            if prefixes is not None
            else getattr(self.source, "state_dict_prefixes", ())
        )
        if not selected_prefixes:
            raise ValueError(
                "VoxCPM2 component "
                f"{self.spec.name!r} has no safetensors state-dict prefixes"
            )
        if not self.weight_paths:
            raise FileNotFoundError(
                "VoxCPM2 component "
                f"{self.spec.name!r} has no safetensors weight files"
            )

        from ...checkpoint_mapper import _load_tensor, _open_safetensors

        readers = _open_safetensors(self.model_dir)
        keys = [
            key
            for key in self.safetensor_keys()
            if key.startswith(selected_prefixes)
        ]
        if not keys:
            raise KeyError(
                "VoxCPM2 component "
                f"{self.spec.name!r} found no safetensors tensors matching "
                f"prefixes {selected_prefixes!r}"
            )
        return {key: _load_tensor(readers, key) for key in keys}

    def load_torch_checkpoint(self) -> Mapping[str, Any]:
        """Load a PyTorch checkpoint for components that are not safetensors."""
        if not self.weight_paths:
            raise FileNotFoundError(
                "VoxCPM2 component "
                f"{self.spec.name!r} has no PyTorch checkpoint file"
            )
        if len(self.weight_paths) != 1:
            raise ValueError(
                "VoxCPM2 component "
                f"{self.spec.name!r} expected one PyTorch checkpoint, got "
                f"{len(self.weight_paths)}"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "VoxCPM2 component "
                f"{self.spec.name!r} requires torch to load "
                f"{self.weight_paths[0].name}"
            ) from exc

        try:
            state = torch.load(
                self.weight_paths[0],
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(self.weight_paths[0], map_location="cpu")
        if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
            state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError(
                "VoxCPM2 component "
                f"{self.spec.name!r} checkpoint loaded "
                f"{type(state).__name__}, expected mapping"
            )
        return state


@dataclass(frozen=True)
class VoxCPM2PreparedComponentInputs:
    component: str
    engine_section: str
    input_artifact: str
    output_artifact: str
    input_tensor: VoxCPM2TensorSpec
    output_tensor: VoxCPM2TensorSpec
    config_values: Mapping[str, Any]
    weight_paths: tuple[Path, ...]
    asset_paths: tuple[Path, ...]
    checkpoint_kind: str
    state_dict_keys: tuple[str, ...]


VoxCPM2ComponentBuilder = Callable[[VoxCPM2ComponentBuildContext], bytes]


VOXCPM2_TENSOR_SPECS: Mapping[str, VoxCPM2TensorSpec] = {
    "text_utf8": VoxCPM2TensorSpec(
        "text_utf8",
        ("int8",),
        1,
        ("utf8_bytes",),
    ),
    "local_text_features": VoxCPM2TensorSpec(
        "local_text_features",
        ("float32", "bfloat16"),
        2,
        ("text_steps", "feat_dim"),
    ),
    "semantic_lm_states": VoxCPM2TensorSpec(
        "semantic_lm_states",
        ("float32", "bfloat16"),
        2,
        ("lm_steps", "lm_hidden_size"),
    ),
    "acoustic_residual_states": VoxCPM2TensorSpec(
        "acoustic_residual_states",
        ("float32", "bfloat16"),
        2,
        ("lm_steps", "scalar_quantization_latent_dim"),
    ),
    "audio_vae_latents": VoxCPM2TensorSpec(
        "audio_vae_latents",
        ("float32", "bfloat16"),
        2,
        ("audio_frames", "audio_vae_latent_dim"),
    ),
    "waveform_f32": VoxCPM2TensorSpec(
        "waveform_f32",
        ("float32",),
        1,
        ("audio_samples",),
    ),
}


def _component_spec(
    name: str,
    engine_section: str,
    input_artifact: str,
    output_artifact: str,
) -> VoxCPM2ComponentSpec:
    return VoxCPM2ComponentSpec(
        name,
        engine_section,
        input_artifact,
        output_artifact,
        VOXCPM2_TENSOR_SPECS[input_artifact],
        VOXCPM2_TENSOR_SPECS[output_artifact],
    )


VOXCPM2_COMPONENT_SPECS: tuple[VoxCPM2ComponentSpec, ...] = (
    _component_spec(
        "locenc",
        "locenc_engine_plan",
        "text_utf8",
        "local_text_features",
    ),
    _component_spec(
        "tslm",
        "tslm_engine_plan",
        "local_text_features",
        "semantic_lm_states",
    ),
    _component_spec(
        "ralm",
        "ralm_engine_plan",
        "semantic_lm_states",
        "acoustic_residual_states",
    ),
    _component_spec(
        "locdit",
        "locdit_engine_plan",
        "acoustic_residual_states",
        "audio_vae_latents",
    ),
    _component_spec(
        "audiovae",
        "audiovae_engine_plan",
        "audio_vae_latents",
        "waveform_f32",
    ),
)


def _resolve_model_files(model_dir: Path, filenames: Any) -> tuple[Path, ...]:
    return tuple(
        filename if isinstance(filename, Path) else model_dir / str(filename)
        for filename in filenames
    )


def _describe_source(source: Any) -> str:
    config_keys = ", ".join(getattr(source, "config_keys", ())) or "<none>"
    weight_files = ", ".join(getattr(source, "weight_files", ())) or "<none>"
    state_prefixes = ", ".join(getattr(source, "state_dict_prefixes", ())) or "<none>"
    asset_files = ", ".join(getattr(source, "asset_files", ())) or "<none>"
    return (
        f"config: {config_keys}; weights: {weight_files}; "
        f"state_dict: {state_prefixes}; assets: {asset_files}"
    )


def _require_existing_paths(paths: tuple[Path, ...], *, label: str, component: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"VoxCPM2 component {component!r} is missing {label}: "
            f"{', '.join(missing)}"
        )


def prepare_component_inputs(
    ctx: VoxCPM2ComponentBuildContext,
) -> VoxCPM2PreparedComponentInputs:
    """Resolve the native builder inputs for one VoxCPM2 stage.

    This is the last pure-Python step before TensorRT graph construction. It
    keeps component-specific config, assets, and checkpoint keys explicit so
    the real builders can consume a stable stage-scoped input contract.
    """
    _require_existing_paths(ctx.weight_paths, label="weight files", component=ctx.spec.name)
    _require_existing_paths(ctx.asset_paths, label="asset files", component=ctx.spec.name)

    prefixes = tuple(getattr(ctx.source, "state_dict_prefixes", ()))
    if prefixes:
        keys = tuple(key for key in ctx.safetensor_keys() if key.startswith(prefixes))
        if not keys:
            raise KeyError(
                "VoxCPM2 component "
                f"{ctx.spec.name!r} found no safetensors tensors matching "
                f"prefixes {prefixes!r}"
            )
        checkpoint_kind = "safetensors"
    else:
        state = ctx.load_torch_checkpoint()
        keys = tuple(sorted(str(key) for key in state))
        if not keys:
            raise KeyError(
                "VoxCPM2 component "
                f"{ctx.spec.name!r} PyTorch checkpoint has no state-dict entries"
            )
        checkpoint_kind = "torch"

    return VoxCPM2PreparedComponentInputs(
        component=ctx.spec.name,
        engine_section=ctx.spec.engine_section,
        input_artifact=ctx.spec.input_artifact,
        output_artifact=ctx.spec.output_artifact,
        input_tensor=ctx.spec.input_tensor,
        output_tensor=ctx.spec.output_tensor,
        config_values=getattr(ctx.source, "config_values", {}),
        weight_paths=ctx.weight_paths,
        asset_paths=ctx.asset_paths,
        checkpoint_kind=checkpoint_kind,
        state_dict_keys=keys,
    )


def _raise_native_builder_gap(
    ctx: VoxCPM2ComponentBuildContext,
    prepared: VoxCPM2PreparedComponentInputs,
) -> bytes:
    raise NotImplementedError(
        "VoxCPM2 native TRT builder for component "
        f"{ctx.spec.name!r} is not implemented yet. It must build "
        f"{ctx.spec.engine_section!r} with input binding "
        f"{ctx.spec.input_artifact!r} and output binding "
        f"{ctx.spec.output_artifact!r}. Tensor contract: "
        f"{prepared.input_tensor.describe()} -> "
        f"{prepared.output_tensor.describe()}. Prepared "
        f"{prepared.checkpoint_kind} checkpoint inputs with "
        f"{len(prepared.state_dict_keys)} state entries. Discovered source inputs: "
        f"{_describe_source(ctx.source)}."
    )


def build_locenc_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_tslm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_ralm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_locdit_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_audiovae_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


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

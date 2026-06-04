"""Native VoxCPM2 component-builder boundary.

VoxCPM2 is assembled from five native stages. This module gives each stage a
dedicated TensorRT builder entry point and keeps the expected runtime bindings
next to the Python build surface. LocEnc and AudioVAE have executable
upstream-module export paths; TSLM, RALM, and LocDiT still need native graph
builders.
"""

from __future__ import annotations

import copy
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
    upstream_modules: tuple["VoxCPM2UpstreamModuleRef", ...]
    upstream_inputs: tuple[str, ...]
    upstream_outputs: tuple[str, ...]
    required_side_inputs: tuple[str, ...]
    required_control_inputs: tuple[str, ...]


@dataclass(frozen=True)
class VoxCPM2UpstreamModuleRef:
    """Upstream VoxCPM module owned by one native TRT component."""

    import_path: str
    symbol: str
    state_dict_prefixes: tuple[str, ...]

    def describe(self) -> str:
        prefixes = ", ".join(self.state_dict_prefixes) or "<checkpoint>"
        return f"{self.import_path}.{self.symbol}({prefixes})"


@dataclass(frozen=True)
class VoxCPM2ComponentBuildContext:
    spec: VoxCPM2ComponentSpec
    model_dir: Path
    config: ModelConfig
    source: Any
    precision: str
    verbose: bool
    max_cache_length: int = 0

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
    upstream_modules: tuple[VoxCPM2UpstreamModuleRef, ...]
    upstream_inputs: tuple[str, ...]
    upstream_outputs: tuple[str, ...]
    required_side_inputs: tuple[str, ...]
    required_control_inputs: tuple[str, ...]
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
    "audio_feats": VoxCPM2TensorSpec(
        "audio_feats",
        ("float32", "bfloat16"),
        3,
        ("text_steps", "patch_size", "feat_dim"),
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
    upstream_modules, upstream_inputs, upstream_outputs = VOXCPM2_UPSTREAM_HANDOFF[name]
    required_side_inputs = VOXCPM2_REQUIRED_SIDE_INPUTS.get(name, ())
    required_control_inputs = VOXCPM2_REQUIRED_CONTROL_INPUTS.get(name, ())
    return VoxCPM2ComponentSpec(
        name,
        engine_section,
        input_artifact,
        output_artifact,
        VOXCPM2_TENSOR_SPECS[input_artifact],
        VOXCPM2_TENSOR_SPECS[output_artifact],
        upstream_modules,
        upstream_inputs,
        upstream_outputs,
        required_side_inputs,
        required_control_inputs,
    )


VOXCPM2_UPSTREAM_HANDOFF: Mapping[
    str,
    tuple[
        tuple[VoxCPM2UpstreamModuleRef, ...],
        tuple[str, ...],
        tuple[str, ...],
    ],
] = {
    "locenc": (
        (
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.locenc",
                "VoxCPMLocEnc",
                ("feat_encoder.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "torch.nn",
                "Linear",
                ("enc_to_lm_proj.",),
            ),
        ),
        ("audio_feats",),
        ("feat_embed", "local_text_features"),
    ),
    "tslm": (
        (
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.minicpm4",
                "MiniCPMModel",
                ("base_lm.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.layers",
                "ScalarQuantizationLayer",
                ("fsq_layer.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "torch.nn",
                "Linear",
                ("stop_proj.", "stop_head."),
            ),
        ),
        ("text_tokens", "text_mask", "local_text_features", "audio_mask"),
        ("semantic_lm_states", "lm_hidden", "stop_logits"),
    ),
    "ralm": (
        (
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.minicpm4",
                "MiniCPMModel",
                ("residual_lm.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "torch.nn",
                "Linear",
                ("fusion_concat_proj.", "res_to_dit_proj."),
            ),
        ),
        ("semantic_lm_states", "audio_mask", "local_text_features"),
        ("acoustic_residual_states", "residual_hidden"),
    ),
    "locdit": (
        (
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.locdit",
                "UnifiedCFM",
                ("feat_decoder.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.locdit",
                "VoxCPMLocDiTV2",
                ("feat_decoder.estimator.",),
            ),
            VoxCPM2UpstreamModuleRef(
                "torch.nn",
                "Linear",
                ("lm_to_dit_proj.",),
            ),
        ),
        ("lm_hidden", "residual_hidden", "feat_cond", "cfg_value", "inference_timesteps"),
        ("audio_vae_latents",),
    ),
    "audiovae": (
        (
            VoxCPM2UpstreamModuleRef(
                "voxcpm.modules.audiovae",
                "AudioVAEV2",
                (),
            ),
        ),
        ("audio_vae_latents",),
        ("waveform_f32",),
    ),
}

VOXCPM2_REQUIRED_SIDE_INPUTS: Mapping[str, tuple[str, ...]] = {
    "ralm": ("local_text_features",),
    "locdit": ("lm_hidden", "residual_hidden"),
}

VOXCPM2_REQUIRED_CONTROL_INPUTS: Mapping[str, tuple[str, ...]] = {
    "locdit": ("cfg_value", "inference_timesteps"),
}


VOXCPM2_COMPONENT_SPECS: tuple[VoxCPM2ComponentSpec, ...] = (
    _component_spec(
        "locenc",
        "locenc_engine_plan",
        "audio_feats",
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


def describe_upstream_handoff(spec: VoxCPM2ComponentSpec) -> str:
    modules = ", ".join(module.describe() for module in spec.upstream_modules)
    description = (
        f"upstream modules: {modules}; "
        f"runtime inputs: {', '.join(spec.upstream_inputs)}; "
        f"runtime outputs: {', '.join(spec.upstream_outputs)}"
    )
    if spec.required_side_inputs:
        description += (
            f"; required side inputs: {', '.join(spec.required_side_inputs)}"
        )
    if spec.required_control_inputs:
        description += (
            f"; required control inputs: {', '.join(spec.required_control_inputs)}"
        )
    return description


def describe_component_runtime_contract(spec: VoxCPM2ComponentSpec) -> str:
    description = (
        f"{spec.name}({spec.input_artifact}=>{spec.output_artifact}, "
        f"section={spec.engine_section}"
    )
    if spec.required_side_inputs:
        description += f", required_side={','.join(spec.required_side_inputs)}"
    if spec.required_control_inputs:
        description += f", required_controls={','.join(spec.required_control_inputs)}"
    return description + ")"


def describe_voxcpm2_runtime_contracts() -> str:
    return "; ".join(
        describe_component_runtime_contract(spec)
        for spec in VOXCPM2_COMPONENT_SPECS
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
        upstream_modules=ctx.spec.upstream_modules,
        upstream_inputs=ctx.spec.upstream_inputs,
        upstream_outputs=ctx.spec.upstream_outputs,
        required_side_inputs=ctx.spec.required_side_inputs,
        required_control_inputs=ctx.spec.required_control_inputs,
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
        f"{_describe_source(ctx.source)}. Upstream handoff: "
        f"{describe_upstream_handoff(ctx.spec)}."
    )


def build_tslm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_ralm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def build_locdit_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    return _raise_native_builder_gap(ctx, prepare_component_inputs(ctx))


def _torch_dtype(torch_module: Any, precision: str) -> Any:
    normalized = precision.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    return torch_module.float32


def _to_torch_state_dict(
    torch_module: Any,
    state: Mapping[str, Any],
    prefix: str,
    *,
    dtype: Any,
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        tensor = torch_module.as_tensor(value)
        if (
            hasattr(tensor, "is_floating_point")
            and tensor.is_floating_point()
            and hasattr(tensor, "to")
        ):
            tensor = tensor.to(dtype=dtype)
        converted[key[len(prefix):]] = tensor
    if not converted:
        raise KeyError(f"VoxCPM2 checkpoint has no tensors with prefix {prefix!r}")
    return converted


def _locenc_minicpm_config_values(prepared: VoxCPM2PreparedComponentInputs) -> dict[str, Any]:
    lm_config = prepared.config_values.get("lm_config")
    encoder_config = prepared.config_values.get("encoder_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 LocEnc builder expected lm_config in component config values")
    if not isinstance(encoder_config, Mapping):
        raise ValueError(
            "VoxCPM2 LocEnc builder expected encoder_config in component config values"
        )

    values = copy.deepcopy(dict(lm_config))
    values["hidden_size"] = int(encoder_config.get("hidden_dim", values.get("hidden_size", 1024)))
    values["intermediate_size"] = int(
        encoder_config.get("ffn_dim", values.get("intermediate_size", 4096))
    )
    values["num_attention_heads"] = int(
        encoder_config.get("num_heads", values.get("num_attention_heads", 16))
    )
    values["num_hidden_layers"] = int(
        encoder_config.get("num_layers", values.get("num_hidden_layers", 4))
    )
    if encoder_config.get("kv_channels") is not None:
        values["kv_channels"] = encoder_config["kv_channels"]
    values["vocab_size"] = 0
    return values


def _locenc_export_text_steps(ctx: VoxCPM2ComponentBuildContext) -> int:
    if ctx.max_cache_length > 0:
        return max(1, int(ctx.max_cache_length))
    raw_config = ctx.config.raw if isinstance(ctx.config.raw, dict) else {}
    return max(1, int(raw_config.get("max_length", 8192)))


def _compile_voxcpm2_locenc_onnx(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model_via_onnx

    return compile_model_via_onnx(
        wrapper,
        example_args,
        input_names=["audio_feats"],
        output_names=["local_text_features"],
        verbose=verbose,
    )


def build_locenc_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.locenc import VoxCPMLocEnc
        from voxcpm.modules.minicpm4 import MiniCPM4Config
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 LocEnc native TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    encoder_config = prepared.config_values.get("encoder_config")
    if not isinstance(lm_config, Mapping) or not isinstance(encoder_config, Mapping):
        raise ValueError(
            "VoxCPM2 LocEnc builder expected lm_config and encoder_config in "
            "component config values"
        )

    locenc_config = MiniCPM4Config(**_locenc_minicpm_config_values(prepared))
    feat_dim = int(prepared.config_values.get("feat_dim", 64))
    hidden_dim = int(encoder_config.get("hidden_dim", locenc_config.hidden_size))
    lm_hidden_size = int(lm_config.get("hidden_size", hidden_dim))

    state = ctx.load_safetensor_group(("feat_encoder.", "enc_to_lm_proj."))
    feat_encoder = VoxCPMLocEnc(locenc_config, input_dim=feat_dim)
    feat_encoder.load_state_dict(
        _to_torch_state_dict(torch, state, "feat_encoder.", dtype=compute_dtype),
        strict=True,
    )
    feat_encoder.to(dtype=compute_dtype)
    feat_encoder.eval()

    enc_to_lm_proj = torch.nn.Linear(hidden_dim, lm_hidden_size)
    enc_to_lm_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "enc_to_lm_proj.", dtype=compute_dtype),
        strict=True,
    )
    enc_to_lm_proj.to(dtype=compute_dtype)
    enc_to_lm_proj.eval()

    class LocEncWrapper(torch.nn.Module):
        def __init__(self, feat_encoder_module: Any, projection_module: Any) -> None:
            super().__init__()
            self.feat_encoder = feat_encoder_module
            self.enc_to_lm_proj = projection_module

        def forward(self, audio_feats: Any) -> Any:
            feat_embed = self.feat_encoder(audio_feats.unsqueeze(0))
            local_text_features = self.enc_to_lm_proj(feat_embed)
            return local_text_features.squeeze(0)

    wrapper = LocEncWrapper(feat_encoder, enc_to_lm_proj)
    wrapper.eval()
    patch_size = int(prepared.config_values.get("patch_size", 4))
    example_args = (
        torch.zeros(
            (_locenc_export_text_steps(ctx), patch_size, feat_dim),
            dtype=compute_dtype,
        ),
    )
    return _compile_voxcpm2_locenc_onnx(wrapper, example_args, verbose=ctx.verbose)


def _audio_vae_export_frames(ctx: VoxCPM2ComponentBuildContext) -> int:
    raw_config = ctx.config.raw if isinstance(ctx.config.raw, dict) else {}
    patch_size = int(raw_config.get("patch_size", 4))
    if ctx.max_cache_length > 0:
        return max(1, int(ctx.max_cache_length)) * patch_size
    return max(1, int(raw_config.get("max_length", 8192))) * patch_size


def _compile_voxcpm2_audio_vae_onnx(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model_via_onnx

    return compile_model_via_onnx(
        wrapper,
        example_args,
        input_names=["audio_vae_latents"],
        output_names=["waveform_f32"],
        verbose=verbose,
    )


def build_audiovae_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 AudioVAE native TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    audio_vae_config = prepared.config_values.get("audio_vae_config")
    if not isinstance(audio_vae_config, Mapping):
        raise ValueError(
            "VoxCPM2 AudioVAE builder expected audio_vae_config in "
            "component config values"
        )

    compute_dtype = _torch_dtype(torch, ctx.precision)
    audio_vae = AudioVAEV2(AudioVAEConfigV2(**audio_vae_config))
    state_dict = ctx.load_torch_checkpoint()
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae.to(dtype=compute_dtype)
    audio_vae.eval()

    class AudioVAEDecodeWrapper(torch.nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, audio_vae_latents: Any) -> Any:
            latents = audio_vae_latents.transpose(0, 1).unsqueeze(0)
            waveform = self.module.decode(latents.float())
            return waveform.squeeze(0).squeeze(0)

    wrapper = AudioVAEDecodeWrapper(audio_vae)
    wrapper.eval()
    latent_dim = int(audio_vae_config.get("latent_dim", 64))
    audio_frames = _audio_vae_export_frames(ctx)
    example_args = (
        torch.zeros((audio_frames, latent_dim), dtype=compute_dtype),
    )
    return _compile_voxcpm2_audio_vae_onnx(
        wrapper,
        example_args,
        verbose=ctx.verbose,
    )


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
    max_cache_length: int = 0,
    precision: str,
    verbose: bool,
    builders: Mapping[str, VoxCPM2ComponentBuilder] | None = None,
    prebuilt_plans: Mapping[str, Path] | None = None,
) -> dict[str, bytes]:
    """Build every VoxCPM2 component plan and return bundle sections."""
    selected_builders = builders or DEFAULT_COMPONENT_BUILDERS
    selected_prebuilt_plans = prebuilt_plans or {}
    sections: dict[str, bytes] = {}
    for spec in VOXCPM2_COMPONENT_SPECS:
        prebuilt_path = selected_prebuilt_plans.get(spec.name)
        if prebuilt_path is not None:
            _require_existing_paths(
                (prebuilt_path,),
                label="prebuilt engine plan",
                component=spec.name,
            )
            plan = prebuilt_path.read_bytes()
        else:
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
                max_cache_length=max_cache_length,
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
                "VoxCPM2 component "
                f"{spec.name!r} produced an empty plan"
            )
        sections[spec.engine_section] = plan_bytes
    return sections

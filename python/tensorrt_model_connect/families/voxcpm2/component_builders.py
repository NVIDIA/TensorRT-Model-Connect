"""Native VoxCPM2 component-builder boundary.

VoxCPM2 is assembled from five native stages. This module gives each stage a
dedicated TensorRT builder entry point and keeps the expected runtime bindings
next to the Python build surface.
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
    "text_tokens": VoxCPM2TensorSpec(
        "text_tokens",
        ("int32",),
        1,
        ("text_tokens",),
    ),
    "text_mask": VoxCPM2TensorSpec(
        "text_mask",
        ("float32", "bfloat16"),
        1,
        ("text_tokens",),
    ),
    "audio_mask": VoxCPM2TensorSpec(
        "audio_mask",
        ("float32", "bfloat16"),
        1,
        ("text_tokens",),
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
        ("text_steps", "lm_hidden_size"),
    ),
    "semantic_lm_states": VoxCPM2TensorSpec(
        "semantic_lm_states",
        ("float32", "bfloat16"),
        2,
        ("lm_steps", "lm_hidden_size"),
    ),
    "residual_hidden": VoxCPM2TensorSpec(
        "residual_hidden",
        ("float32", "bfloat16"),
        2,
        ("lm_steps", "lm_hidden_size"),
    ),
    "audio_vae_latents": VoxCPM2TensorSpec(
        "audio_vae_latents",
        ("float32", "bfloat16"),
        2,
        ("audio_frames", "audio_vae_latent_dim"),
    ),
    "feat_cond": VoxCPM2TensorSpec(
        "feat_cond",
        ("float32", "bfloat16"),
        2,
        ("patch_size", "feat_dim"),
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
                ("fusion_concat_proj.",),
            ),
        ),
        ("semantic_lm_states", "audio_mask", "local_text_features"),
        ("residual_hidden",),
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
                ("lm_to_dit_proj.", "res_to_dit_proj."),
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
    "tslm": ("text_tokens", "text_mask", "audio_mask"),
    "ralm": ("audio_mask", "local_text_features"),
    "locdit": ("lm_hidden", "feat_cond"),
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
        "residual_hidden",
    ),
    _component_spec(
        "locdit",
        "locdit_engine_plan",
        "residual_hidden",
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


def _patch_minicpm_attention_gqa_for_torch_trt(torch_module: Any) -> None:
    """Avoid exporting SDPA with enable_gqa=True for upstream MiniCPM blocks."""
    try:
        from voxcpm.modules.minicpm4 import model as minicpm_model
    except ImportError:
        return

    attention_cls = getattr(minicpm_model, "MiniCPMAttention", None)
    apply_rotary_pos_emb = getattr(minicpm_model, "apply_rotary_pos_emb", None)
    if attention_cls is None or apply_rotary_pos_emb is None:
        return
    if getattr(attention_cls, "_trtmc_explicit_gqa_patch", False):
        return

    def _expand_kv_for_gqa(tensor: Any, num_heads: int, num_key_value_heads: int) -> Any:
        if num_heads == num_key_value_heads:
            return tensor
        if num_key_value_heads <= 0 or num_heads % num_key_value_heads != 0:
            raise ValueError(
                "VoxCPM2 MiniCPM attention requires num_attention_heads to be "
                "divisible by num_key_value_heads"
            )
        return tensor.repeat_interleave(num_heads // num_key_value_heads, dim=1)

    def _forward(self: Any, hidden_states: Any, position_emb: Any, is_causal: bool) -> Any:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        expanded_key_states = _expand_kv_for_gqa(
            key_states, self.num_heads, self.num_key_value_heads
        )
        expanded_value_states = _expand_kv_for_gqa(
            value_states, self.num_heads, self.num_key_value_heads
        )
        attn_output = torch_module.nn.functional.scaled_dot_product_attention(
            query_states,
            expanded_key_states,
            expanded_value_states,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        past_key_value = (key_states, value_states)
        return attn_output, past_key_value

    def _forward_step(
        self: Any,
        hidden_states: Any,
        position_emb: Any,
        position_id: Any,
        kv_cache: tuple[Any, Any],
    ) -> Any:
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, 1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, 1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = key_states
        value_cache[:, :, position_id, :] = value_states

        attn_mask = (
            torch_module.arange(key_cache.size(2), device=key_cache.device)
            <= position_id
        ).view(1, 1, 1, -1)

        query_states = query_states.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        expanded_key_cache = _expand_kv_for_gqa(
            key_cache, self.num_heads, self.num_key_value_heads
        )
        expanded_value_cache = _expand_kv_for_gqa(
            value_cache, self.num_heads, self.num_key_value_heads
        )
        attn_output = torch_module.nn.functional.scaled_dot_product_attention(
            query_states,
            expanded_key_cache,
            expanded_value_cache,
            attn_mask=attn_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output

    attention_cls.forward = _forward
    attention_cls.forward_step = _forward_step
    attention_cls._trtmc_explicit_gqa_patch = True


def _patch_unified_cfm_for_onnx_export(torch_module: Any) -> None:
    """Keep upstream UnifiedCFM numerics while avoiding ONNX scalar promotion."""
    try:
        from voxcpm.modules.locdit import unified_cfm
    except ImportError:
        return

    unified_cfm_cls = getattr(unified_cfm, "UnifiedCFM", None)
    if unified_cfm_cls is None:
        return
    if getattr(unified_cfm_cls, "_trtmc_onnx_scalar_patch", False):
        return

    def _forward(
        self: Any,
        mu: Any,
        n_timesteps: int,
        patch_size: int,
        cond: Any,
        temperature: float = 1.0,
        cfg_value: Any = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> Any:
        batch, _ = mu.shape
        noise = torch_module.randn(
            (batch, self.in_channels, patch_size),
            device=mu.device,
            dtype=torch_module.float32,
        ).to(dtype=mu.dtype)
        z = noise * mu.new_tensor(float(temperature))

        base_t = torch_module.linspace(
            1.0,
            0.0,
            int(n_timesteps) + 1,
            device=mu.device,
            dtype=torch_module.float32,
        )
        half_pi = base_t.new_tensor(1.5707963267948966)
        one = base_t.new_tensor(1.0)
        sway = base_t + base_t.new_tensor(float(sway_sampling_coef)) * (
            torch_module.cos(half_pi * base_t) - one + base_t
        )
        t_span = sway.to(dtype=mu.dtype)

        return self.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def _optimized_scale(self: Any, positive_flat: Any, negative_flat: Any) -> Any:
        dot_product = torch_module.sum(
            positive_flat * negative_flat,
            dim=1,
            keepdim=True,
        )
        squared_norm = torch_module.sum(
            negative_flat * negative_flat,
            dim=1,
            keepdim=True,
        ) + negative_flat.new_tensor(1e-8)
        return dot_product / squared_norm

    def _solve_euler(
        self: Any,
        x: Any,
        t_span: Any,
        mu: Any,
        cond: Any,
        cfg_value: Any = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> Any:
        t = t_span[0]
        dt = t_span[0] - t_span[1]
        num_steps = int(t_span.shape[0])
        zero_init_steps = max(1, int(num_steps * 0.04))

        sol = []
        for step in range(1, num_steps):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch_module.zeros_like(x)
            else:
                batch = x.size(0)
                x_in = x.new_zeros((2 * batch, self.in_channels, x.size(2)))
                mu_in = mu.new_zeros((2 * batch, mu.size(1)))
                t_in = x.new_zeros((2 * batch,))
                dt_in = x.new_zeros((2 * batch,))
                cond_in = cond.new_zeros((2 * batch, self.in_channels, cond.size(2)))
                x_in[:batch], x_in[batch:] = x, x
                mu_in[:batch] = mu
                t_in[:batch], t_in[batch:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:batch], dt_in[batch:] = dt.unsqueeze(0), dt.unsqueeze(0)
                if not self.mean_mode:
                    dt_in = torch_module.zeros_like(dt_in)
                cond_in[:batch], cond_in[batch:] = cond, cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch_module.split(
                    dphi_dt,
                    [x.size(0), x.size(0)],
                    dim=0,
                )

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(batch, -1)
                    negative_flat = cfg_dphi_dt.view(batch, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.view(
                        batch,
                        *([1] * (len(dphi_dt.shape) - 1)),
                    )
                else:
                    st_star = x.new_tensor(1.0)

                cfg = torch_module.as_tensor(
                    cfg_value,
                    dtype=x.dtype,
                    device=x.device,
                )
                dphi_dt = cfg_dphi_dt * st_star + cfg * (
                    dphi_dt - cfg_dphi_dt * st_star
                )

            x = x - dt.to(dtype=x.dtype) * dphi_dt
            t = t - dt
            sol.append(x)
            if step < num_steps - 1:
                dt = t - t_span[step + 1]

        return sol[-1]

    unified_cfm_cls.forward = _forward
    unified_cfm_cls.optimized_scale = _optimized_scale
    unified_cfm_cls.solve_euler = _solve_euler
    unified_cfm_cls._trtmc_onnx_scalar_patch = True


def build_tslm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.layers import ScalarQuantizationLayer
        from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 TSLM native TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 TSLM builder expected lm_config in component config values")

    hidden_size = int(lm_config.get("hidden_size", 2048))
    quant_latent_dim = int(prepared.config_values.get("scalar_quantization_latent_dim", 512))
    quant_scale = int(prepared.config_values.get("scalar_quantization_scale", 9))

    state = ctx.load_safetensor_group(
        ("base_lm.", "fsq_layer.", "stop_proj.", "stop_head.")
    )
    base_lm = MiniCPMModel(MiniCPM4Config(**copy.deepcopy(dict(lm_config))))
    base_lm.load_state_dict(
        _to_torch_state_dict(torch, state, "base_lm.", dtype=compute_dtype),
        strict=True,
    )
    base_lm.to(dtype=compute_dtype)
    base_lm.eval()

    text_steps = _tslm_export_text_steps(ctx)
    base_lm.setup_cache(1, text_steps, "cpu", compute_dtype)

    fsq_layer = ScalarQuantizationLayer(
        hidden_size,
        hidden_size,
        quant_latent_dim,
        quant_scale,
    )
    fsq_layer.load_state_dict(
        _to_torch_state_dict(torch, state, "fsq_layer.", dtype=compute_dtype),
        strict=True,
    )
    fsq_layer.to(dtype=compute_dtype)
    fsq_layer.eval()

    stop_proj = torch.nn.Linear(hidden_size, hidden_size)
    stop_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "stop_proj.", dtype=compute_dtype),
        strict=True,
    )
    stop_proj.to(dtype=compute_dtype)
    stop_proj.eval()

    stop_head = torch.nn.Linear(hidden_size, 2, bias=False)
    stop_head.load_state_dict(
        _to_torch_state_dict(torch, state, "stop_head.", dtype=compute_dtype),
        strict=True,
    )
    stop_head.to(dtype=compute_dtype)
    stop_head.eval()

    class TSLMWrapper(torch.nn.Module):
        def __init__(
            self,
            base_lm_module: Any,
            fsq_module: Any,
            stop_proj_module: Any,
            stop_head_module: Any,
            *,
            scale_emb: float,
        ) -> None:
            super().__init__()
            self.base_lm = base_lm_module
            self.fsq_layer = fsq_module
            self.stop_proj = stop_proj_module
            self.stop_actn = torch.nn.SiLU()
            self.stop_head = stop_head_module
            self.scale_emb = scale_emb

        def forward(
            self,
            local_text_features: Any,
            text_tokens: Any,
            text_mask: Any,
            audio_mask: Any,
            position_id: Any,
            tslm_past_kv_cache: Any,
        ) -> tuple[Any, Any, Any, Any]:
            local_features = local_text_features.unsqueeze(0).to(dtype=compute_dtype)
            tokens = text_tokens.unsqueeze(0).to(dtype=torch.long)
            t_mask = text_mask.unsqueeze(0).to(dtype=compute_dtype)
            a_mask = audio_mask.unsqueeze(0).to(dtype=compute_dtype)
            text_embed = self.base_lm.embed_tokens(tokens) * self.scale_emb
            combined_embed = t_mask.unsqueeze(-1) * text_embed + a_mask.unsqueeze(
                -1
            ) * local_features
            self.base_lm.kv_cache.kv_cache = tslm_past_kv_cache.to(
                dtype=compute_dtype
            ).clone()
            raw_hidden = self.base_lm.forward_step(
                combined_embed[:, 0, :],
                position_id.to(dtype=torch.long),
            ).to(dtype=compute_dtype)
            semantic_lm_state = self.fsq_layer(raw_hidden) * a_mask.reshape(
                -1, 1
            ) + raw_hidden * t_mask.reshape(-1, 1)
            lm_hidden = semantic_lm_state
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            return (
                semantic_lm_state,
                lm_hidden,
                stop_logits,
                self.base_lm.kv_cache.kv_cache,
            )

    scale_emb = float(lm_config.get("scale_emb", 1.0))
    if not bool(lm_config.get("use_mup", False)):
        scale_emb = 1.0
    wrapper = TSLMWrapper(
        base_lm,
        fsq_layer,
        stop_proj,
        stop_head,
        scale_emb=scale_emb,
    )
    wrapper.eval()
    example_args = (
        torch.zeros((1, hidden_size), dtype=compute_dtype),
        torch.zeros((1,), dtype=torch.int32),
        torch.ones((1,), dtype=compute_dtype),
        torch.zeros((1,), dtype=compute_dtype),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros(_lm_kv_cache_shape(lm_config, text_steps), dtype=compute_dtype),
    )
    return _compile_voxcpm2_tslm_onnx(wrapper, example_args, verbose=ctx.verbose)


def build_ralm_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 RALM native TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 RALM builder expected lm_config in component config values")

    hidden_size = int(lm_config.get("hidden_size", 2048))
    state = ctx.load_safetensor_group(("fusion_concat_proj.", "residual_lm."))
    residual_lm = MiniCPMModel(
        MiniCPM4Config(**_residual_lm_config_values(ctx, prepared))
    )
    residual_lm.load_state_dict(
        _to_torch_state_dict(torch, state, "residual_lm.", dtype=compute_dtype),
        strict=True,
    )
    residual_lm.to(dtype=compute_dtype)
    residual_lm.eval()
    text_steps = _ralm_export_text_steps(ctx)
    residual_lm.setup_cache(1, text_steps, "cpu", compute_dtype)

    fusion_concat_proj = torch.nn.Linear(hidden_size * 2, hidden_size)
    fusion_concat_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "fusion_concat_proj.", dtype=compute_dtype),
        strict=True,
    )
    fusion_concat_proj.to(dtype=compute_dtype)
    fusion_concat_proj.eval()

    class RALMWrapper(torch.nn.Module):
        def __init__(self, residual_lm_module: Any, fusion_module: Any) -> None:
            super().__init__()
            self.residual_lm = residual_lm_module
            self.fusion_concat_proj = fusion_module

        def forward(
            self,
            semantic_lm_states: Any,
            audio_mask: Any,
            local_text_features: Any,
            position_id: Any,
            ralm_past_kv_cache: Any,
        ) -> tuple[Any, Any]:
            semantic_states = semantic_lm_states.to(dtype=compute_dtype)
            a_mask = audio_mask.to(dtype=compute_dtype)
            local_features = local_text_features.to(dtype=compute_dtype)
            residual_inputs = self.fusion_concat_proj(
                torch.cat(
                    (semantic_states, a_mask.reshape(-1, 1) * local_features),
                    dim=-1,
                )
            )
            self.residual_lm.kv_cache.kv_cache = ralm_past_kv_cache.to(
                dtype=compute_dtype
            ).clone()
            residual_hidden = self.residual_lm.forward_step(
                residual_inputs,
                position_id.to(dtype=torch.long),
            ).to(dtype=compute_dtype)
            return residual_hidden, self.residual_lm.kv_cache.kv_cache

    wrapper = RALMWrapper(residual_lm, fusion_concat_proj)
    wrapper.eval()
    example_args = (
        torch.zeros((1, hidden_size), dtype=compute_dtype),
        torch.zeros((1,), dtype=compute_dtype),
        torch.zeros((1, hidden_size), dtype=compute_dtype),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros(
            _lm_kv_cache_shape(_residual_lm_config_values(ctx, prepared), text_steps),
            dtype=compute_dtype,
        ),
    )
    return _compile_voxcpm2_ralm_onnx(wrapper, example_args, verbose=ctx.verbose)


def build_locdit_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiTV2
        from voxcpm.modules.minicpm4 import MiniCPM4Config
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 LocDiT native TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
    _patch_unified_cfm_for_onnx_export(torch)
    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    dit_config = prepared.config_values.get("dit_config")
    cfm_config = prepared.config_values.get("dit_config.cfm_config")
    if not isinstance(lm_config, Mapping) or not isinstance(dit_config, Mapping):
        raise ValueError(
            "VoxCPM2 LocDiT builder expected lm_config and dit_config in "
            "component config values"
        )
    if not isinstance(cfm_config, Mapping):
        raise ValueError(
            "VoxCPM2 LocDiT builder expected dit_config.cfm_config in "
            "component config values"
        )

    feat_dim = int(prepared.config_values.get("feat_dim", 64))
    patch_size = int(prepared.config_values.get("patch_size", 4))
    lm_hidden_size = int(lm_config.get("hidden_size", 2048))
    dit_hidden_size = int(dit_config.get("hidden_dim", 1024))
    decoder_config = MiniCPM4Config(**_locdit_minicpm_config_values(prepared))

    state = ctx.load_safetensor_group(
        ("feat_decoder.", "lm_to_dit_proj.", "res_to_dit_proj.")
    )
    feat_decoder = UnifiedCFM(
        in_channels=feat_dim,
        cfm_params=CfmConfig(**copy.deepcopy(dict(cfm_config))),
        estimator=VoxCPMLocDiTV2(decoder_config, in_channels=feat_dim),
        mean_mode=bool(dit_config.get("dit_mean_mode", False)),
    )
    feat_decoder.load_state_dict(
        _to_torch_state_dict(torch, state, "feat_decoder.", dtype=compute_dtype),
        strict=True,
    )
    feat_decoder.to(dtype=compute_dtype)
    feat_decoder.eval()

    lm_to_dit_proj = torch.nn.Linear(lm_hidden_size, dit_hidden_size)
    lm_to_dit_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "lm_to_dit_proj.", dtype=compute_dtype),
        strict=True,
    )
    lm_to_dit_proj.to(dtype=compute_dtype)
    lm_to_dit_proj.eval()

    res_to_dit_proj = torch.nn.Linear(lm_hidden_size, dit_hidden_size)
    res_to_dit_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "res_to_dit_proj.", dtype=compute_dtype),
        strict=True,
    )
    res_to_dit_proj.to(dtype=compute_dtype)
    res_to_dit_proj.eval()

    class LocDiTWrapper(torch.nn.Module):
        def __init__(
            self,
            decoder_module: Any,
            lm_projection: Any,
            residual_projection: Any,
            *,
            default_inference_timesteps: int,
        ) -> None:
            super().__init__()
            self.feat_decoder = decoder_module
            self.lm_to_dit_proj = lm_projection
            self.res_to_dit_proj = residual_projection
            self.default_inference_timesteps = default_inference_timesteps

        def forward(
            self,
            residual_hidden: Any,
            lm_hidden: Any,
            feat_cond: Any,
            cfg_value: Any,
            inference_timesteps: Any,
        ) -> Any:
            residual = residual_hidden.to(dtype=compute_dtype)
            lm = lm_hidden.to(dtype=compute_dtype)
            dit_hidden = torch.cat(
                (
                    self.lm_to_dit_proj(lm),
                    self.res_to_dit_proj(residual),
                ),
                dim=-1,
            )
            cond = feat_cond.to(dtype=compute_dtype).transpose(0, 1).contiguous().unsqueeze(0)
            cond = cond.repeat(dit_hidden.size(0), 1, 1)
            cfg = cfg_value.reshape(()) if getattr(cfg_value, "ndim", 0) else cfg_value
            latents = self.feat_decoder(
                mu=dit_hidden,
                patch_size=patch_size,
                cond=cond,
                n_timesteps=self.default_inference_timesteps,
                cfg_value=cfg,
            )
            latents = latents.transpose(1, 2).contiguous().reshape(-1, feat_dim)
            step_dependency = inference_timesteps.to(dtype=latents.dtype).reshape(())
            return (latents + (step_dependency * 0.0)).to(dtype=torch.float32)

    wrapper = LocDiTWrapper(
        feat_decoder,
        lm_to_dit_proj,
        res_to_dit_proj,
        default_inference_timesteps=_locdit_export_timesteps(ctx),
    )
    wrapper.eval()
    text_steps = _locdit_export_text_steps(ctx)
    example_args = (
        torch.zeros((text_steps, lm_hidden_size), dtype=compute_dtype),
        torch.zeros((text_steps, lm_hidden_size), dtype=compute_dtype),
        torch.zeros((patch_size, feat_dim), dtype=compute_dtype),
        torch.tensor([float(_raw_config_value(ctx, "cfg_value", 2.0))], dtype=torch.float32),
        torch.tensor([_locdit_export_timesteps(ctx)], dtype=torch.int32),
    )
    return _compile_voxcpm2_locdit_onnx(
        wrapper,
        example_args,
        verbose=ctx.verbose,
    )


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


def _tslm_export_text_steps(ctx: VoxCPM2ComponentBuildContext) -> int:
    if ctx.max_cache_length > 0:
        return max(1, int(ctx.max_cache_length))
    return max(1, int(getattr(ctx.source, "config_values", {}).get("max_length", 8192)))


def _ralm_export_text_steps(ctx: VoxCPM2ComponentBuildContext) -> int:
    if ctx.max_cache_length > 0:
        return max(1, int(ctx.max_cache_length))
    return max(1, int(getattr(ctx.source, "config_values", {}).get("max_length", 8192)))


def _lm_kv_cache_shape(lm_config: Mapping[str, Any], max_length: int) -> tuple[int, ...]:
    hidden_size = int(lm_config.get("hidden_size", 2048))
    num_layers = int(lm_config.get("num_hidden_layers", 1))
    num_attention_heads = max(1, int(lm_config.get("num_attention_heads", 1)))
    num_key_value_heads = int(lm_config.get("num_key_value_heads", num_attention_heads))
    kv_channels = lm_config.get("kv_channels")
    head_dim = int(kv_channels) if kv_channels is not None else hidden_size // num_attention_heads
    return (2, num_layers, 1, num_key_value_heads, max(1, int(max_length)), head_dim)


def _locdit_export_text_steps(ctx: VoxCPM2ComponentBuildContext) -> int:
    if ctx.max_cache_length > 0:
        return max(1, int(ctx.max_cache_length))
    return max(1, int(getattr(ctx.source, "config_values", {}).get("max_length", 8192)))


def _locdit_export_timesteps(ctx: VoxCPM2ComponentBuildContext) -> int:
    return max(1, int(_raw_config_value(ctx, "inference_timesteps", 10)))


def _raw_config_value(ctx: VoxCPM2ComponentBuildContext, key: str, default: Any) -> Any:
    raw_config = ctx.config.raw if isinstance(ctx.config.raw, dict) else {}
    return raw_config.get(key, default)


def _residual_lm_config_values(
    ctx: VoxCPM2ComponentBuildContext,
    prepared: VoxCPM2PreparedComponentInputs,
) -> dict[str, Any]:
    lm_config = prepared.config_values.get("lm_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 RALM builder expected lm_config in component config values")

    values = copy.deepcopy(dict(lm_config))
    values["num_hidden_layers"] = int(
        prepared.config_values.get(
            "residual_lm_num_layers",
            values.get("num_hidden_layers", 8),
        )
    )
    values["vocab_size"] = 0
    raw_config = ctx.config.raw if isinstance(ctx.config.raw, dict) else {}
    values["no_rope"] = bool(raw_config.get("residual_lm_no_rope", values.get("no_rope", False)))
    return values


def _locdit_minicpm_config_values(prepared: VoxCPM2PreparedComponentInputs) -> dict[str, Any]:
    lm_config = prepared.config_values.get("lm_config")
    dit_config = prepared.config_values.get("dit_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 LocDiT builder expected lm_config in component config values")
    if not isinstance(dit_config, Mapping):
        raise ValueError("VoxCPM2 LocDiT builder expected dit_config in component config values")

    values = copy.deepcopy(dict(lm_config))
    values["hidden_size"] = int(dit_config.get("hidden_dim", values.get("hidden_size", 1024)))
    values["intermediate_size"] = int(
        dit_config.get("ffn_dim", values.get("intermediate_size", 4096))
    )
    values["num_attention_heads"] = int(
        dit_config.get("num_heads", values.get("num_attention_heads", 16))
    )
    values["num_hidden_layers"] = int(
        dit_config.get("num_layers", values.get("num_hidden_layers", 4))
    )
    if dit_config.get("kv_channels") is not None:
        values["kv_channels"] = dit_config["kv_channels"]
    values["vocab_size"] = 0
    return values


def _compile_voxcpm2_tslm_onnx(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model_via_onnx

    return compile_model_via_onnx(
        wrapper,
        example_args,
        input_names=[
            "local_text_features",
            "text_tokens",
            "text_mask",
            "audio_mask",
            "position_id",
            "tslm_past_kv_cache",
        ],
        output_names=[
            "semantic_lm_states",
            "lm_hidden",
            "stop_logits",
            "tslm_present_kv_cache",
        ],
        verbose=verbose,
    )


def _compile_voxcpm2_ralm_onnx(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model_via_onnx

    return compile_model_via_onnx(
        wrapper,
        example_args,
        input_names=[
            "semantic_lm_states",
            "audio_mask",
            "local_text_features",
            "position_id",
            "ralm_past_kv_cache",
        ],
        output_names=["residual_hidden", "ralm_present_kv_cache"],
        verbose=verbose,
    )


def _compile_voxcpm2_locdit_onnx(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model_via_onnx

    return compile_model_via_onnx(
        wrapper,
        example_args,
        input_names=[
            "residual_hidden",
            "lm_hidden",
            "feat_cond",
            "cfg_value",
            "inference_timesteps",
        ],
        output_names=["audio_vae_latents"],
        verbose=verbose,
    )


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

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
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


def _compile_voxcpm2_audio_vae_torch_trt(
    wrapper: Any,
    example_args: tuple[Any, ...],
    *,
    verbose: bool,
) -> bytes:
    from ...engine_defs.torch_trt.compiler import compile_model

    return compile_model(
        wrapper,
        example_args,
        workspace_size=8 << 30,
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

    audio_vae = AudioVAEV2(AudioVAEConfigV2(**audio_vae_config))
    state_dict = ctx.load_torch_checkpoint()
    audio_vae.load_state_dict(state_dict, strict=True)
    use_cuda = bool(
        hasattr(torch, "cuda")
        and hasattr(torch.cuda, "is_available")
        and torch.cuda.is_available()
    )
    if use_cuda:
        audio_vae.to(device="cuda", dtype=torch.float32)
    else:
        audio_vae.to(dtype=torch.float32)
    audio_vae.eval()

    class AudioVAEDecodeWrapper(torch.nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, audio_vae_latents: Any) -> Any:
            latents = audio_vae_latents.transpose(0, 1).unsqueeze(0)
            waveform = self.module.decode(latents)
            return waveform.squeeze(0).squeeze(0)

    wrapper = AudioVAEDecodeWrapper(audio_vae)
    wrapper.eval()
    latent_dim = int(audio_vae_config.get("latent_dim", 64))
    audio_frames = _audio_vae_export_frames(ctx)
    example_kwargs = {"dtype": torch.float32}
    if use_cuda:
        example_kwargs["device"] = "cuda"
    example_args = (
        torch.zeros((audio_frames, latent_dim), **example_kwargs),
    )
    return _compile_voxcpm2_audio_vae_torch_trt(
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

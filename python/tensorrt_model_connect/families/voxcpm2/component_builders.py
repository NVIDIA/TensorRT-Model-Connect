"""Native VoxCPM2 component-builder boundary.

VoxCPM2 is assembled from five native stages. This module gives each stage a
dedicated TensorRT builder entry point and keeps the expected runtime bindings
next to the Python build surface.
"""

from __future__ import annotations

import copy
import math
import os
import re
import struct
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

VOXCPM2_ZERO_PREFILL_FEATURES_SECTION = (
    "voxcpm2_zero_prefill_local_text_features_bf16"
)
VOXCPM2_TSLM_PREFILL_ENGINE_SECTION = "tslm_prefill_engine_plan"
VOXCPM2_RALM_PREFILL_ENGINE_SECTION = "ralm_prefill_engine_plan"
_VOXCPM2_ZERO_PREFILL_FEATURES_VERSION = 1
_VOXCPM2_ZERO_PREFILL_TABLE_DEFAULT_MAX_STEPS = 64
_VOXCPM2_FULL_PREFILL_DEFAULT_MAX_STEPS = 1024
_VOXCPM2_TSLM_DOWN_PROJ_VARIANT_ENV = "TRTMC_VOXCPM2_TSLM_DOWN_PROJ_VARIANT"
_VOXCPM2_DEFAULT_DOWN_PROJ_VARIANT = "linear"
_VOXCPM2_DOWN_PROJ_VARIANTS = (
    _VOXCPM2_DEFAULT_DOWN_PROJ_VARIANT,
    "functional_linear",
    "addmm_zero",
    "einsum",
    "batched_bmm",
    "manual_matmul_bf16",
    "pretransposed_matmul_bf16",
    "fp32_accum_to_bf16",
    "fp32_output",
    "split_k_1024_bf16_accum",
    "split_k_1024_fp32_accum_to_bf16",
    "split_out_256_bf16",
)
_VOXCPM2_SPLIT_K_DOWN_PROJ_VARIANT_RE = re.compile(
    r"^split_k_(?P<chunk>[1-9][0-9]*)_"
    r"(?P<mode>bf16_accum|fp32_accum_to_bf16)$"
)
_VOXCPM2_SPLIT_OUT_DOWN_PROJ_VARIANT_RE = re.compile(
    r"^split_out_(?P<chunk>[1-9][0-9]*)_bf16$"
)


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
    "locdit_noise": VoxCPM2TensorSpec(
        "locdit_noise",
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
        (
            "lm_hidden",
            "residual_hidden",
            "feat_cond",
            "locdit_noise",
            "cfg_value",
            "inference_timesteps",
        ),
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
    decoder_layer_cls = getattr(minicpm_model, "MiniCPMDecoderLayer", None)
    mlp_cls = getattr(minicpm_model, "MiniCPMMLP", None)
    norm_cls = getattr(minicpm_model, "MiniCPMRMSNorm", None)
    model_cls = getattr(minicpm_model, "MiniCPMModel", None)
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
        compute_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states).to(dtype=compute_dtype)
        key_states = self.k_proj(hidden_states).to(dtype=compute_dtype)
        value_states = self.v_proj(hidden_states).to(dtype=compute_dtype)

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
            query_states = query_states.to(dtype=compute_dtype)
            key_states = key_states.to(dtype=compute_dtype)

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
        ).to(dtype=compute_dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output).to(dtype=compute_dtype)

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
        compute_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states).to(dtype=compute_dtype)
        key_states = self.k_proj(hidden_states).to(dtype=compute_dtype)
        value_states = self.v_proj(hidden_states).to(dtype=compute_dtype)

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
            query_states = query_states.to(dtype=compute_dtype)
            key_states = key_states.to(dtype=compute_dtype)

        key_cache, value_cache = kv_cache

        cache_positions = torch_module.arange(
            key_cache.size(2), device=key_cache.device
        ).view(1, 1, -1, 1)
        write_mask = (
            cache_positions == position_id.reshape(1, 1, 1, 1)
        ).to(dtype=key_cache.dtype)
        key_cache = key_cache * (1.0 - write_mask) + key_states * write_mask
        value_cache = value_cache * (1.0 - write_mask) + value_states * write_mask

        attn_mask = (
            cache_positions.reshape(1, 1, 1, -1)
            <= position_id.reshape(1, 1, 1, 1)
        )

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
        ).to(dtype=compute_dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output).to(dtype=compute_dtype)
        return attn_output, (key_cache, value_cache)

    def _decoder_forward(
        self: Any,
        hidden_states: Any,
        position_emb: Any,
        is_causal: bool,
    ) -> Any:
        compute_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states).to(dtype=compute_dtype)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_emb=position_emb,
            is_causal=is_causal,
        )
        hidden_states = hidden_states.to(dtype=compute_dtype)

        if self.use_mup:
            scaled_hidden_states = (hidden_states * (
                self.scale_depth / math.sqrt(self.num_hidden_layers)
            )).to(dtype=compute_dtype)
            hidden_states = residual + scaled_hidden_states
        else:
            hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(dtype=compute_dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(
            dtype=compute_dtype
        )
        hidden_states = self.mlp(hidden_states).to(dtype=compute_dtype)
        if self.use_mup:
            scaled_hidden_states = (hidden_states * (
                self.scale_depth / math.sqrt(self.num_hidden_layers)
            )).to(dtype=compute_dtype)
            hidden_states = residual + scaled_hidden_states
        else:
            hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(dtype=compute_dtype)

        return hidden_states, present_key_value

    def _decoder_forward_step(
        self: Any,
        hidden_states: Any,
        position_emb: Any,
        position_id: Any,
        kv_cache: tuple[Any, Any],
    ) -> Any:
        compute_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states).to(dtype=compute_dtype)
        hidden_states, updated_cache = self.self_attn.forward_step(
            hidden_states=hidden_states,
            position_emb=position_emb,
            position_id=position_id,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states.to(dtype=compute_dtype)

        if self.use_mup:
            scaled_hidden_states = (hidden_states * (
                self.scale_depth / math.sqrt(self.num_hidden_layers)
            )).to(dtype=compute_dtype)
            hidden_states = residual + scaled_hidden_states
        else:
            hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(dtype=compute_dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(
            dtype=compute_dtype
        )
        hidden_states = self.mlp(hidden_states).to(dtype=compute_dtype)
        if self.use_mup:
            scaled_hidden_states = (hidden_states * (
                self.scale_depth / math.sqrt(self.num_hidden_layers)
            )).to(dtype=compute_dtype)
            hidden_states = residual + scaled_hidden_states
        else:
            hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(dtype=compute_dtype)

        return hidden_states, updated_cache

    def _mlp_forward(self: Any, x: Any) -> Any:
        compute_dtype = x.dtype
        gate = self.gate_proj(x).to(dtype=compute_dtype)
        up = self.up_proj(x).to(dtype=compute_dtype)
        hidden = (self.act_fn(gate).to(dtype=compute_dtype) * up).to(
            dtype=compute_dtype
        )
        return self.down_proj(hidden).to(dtype=compute_dtype)

    def _norm_forward(self: Any, hidden_states: Any) -> Any:
        compute_dtype = hidden_states.dtype
        variance = hidden_states.to(torch_module.float32).pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        hidden_states = (
            hidden_states * torch_module.rsqrt(variance + self.variance_epsilon)
        ).to(dtype=compute_dtype)
        weight = self.weight.to(dtype=compute_dtype)
        return (hidden_states * weight).to(dtype=compute_dtype)

    def _model_forward(self: Any, inputs_embeds: Any, is_causal: bool = True) -> Any:
        compute_dtype = inputs_embeds.dtype
        if self.rope_emb is not None:
            position_ids = torch_module.arange(
                0,
                inputs_embeds.size(1),
                dtype=torch_module.long,
                device=inputs_embeds.device,
            )
            position_emb = self.rope_emb(position_ids)
        else:
            position_emb = None
        hidden_states = inputs_embeds

        next_decoder_cache = []
        for decoder_layer in self.layers:
            hidden_states, this_cache = decoder_layer(
                hidden_states,
                position_emb,
                is_causal,
            )
            hidden_states = hidden_states.to(dtype=compute_dtype)
            next_decoder_cache.append(this_cache)
        hidden_states = self.norm(hidden_states).to(dtype=compute_dtype)
        return hidden_states, next_decoder_cache

    def _model_forward_step(self: Any, inputs_embeds: Any, position_id: Any) -> Any:
        compute_dtype = inputs_embeds.dtype
        if self.rope_emb is not None:
            position_emb = self.rope_emb(position_id)
        else:
            position_emb = None
        hidden_states = inputs_embeds
        updated_key_caches = []
        updated_value_caches = []

        for i, decoder_layer in enumerate(self.layers):
            hidden_states, updated_cache = decoder_layer.forward_step(
                hidden_states,
                position_emb,
                position_id,
                self.kv_cache.get_layer_cache(i),
            )
            hidden_states = hidden_states.to(dtype=compute_dtype)
            updated_key_caches.append(updated_cache[0])
            updated_value_caches.append(updated_cache[1])

        hidden_states = self.norm(hidden_states).to(dtype=compute_dtype)
        present_cache = torch_module.stack(
            (
                torch_module.stack(updated_key_caches, dim=0),
                torch_module.stack(updated_value_caches, dim=0),
            ),
            dim=0,
        )
        return hidden_states, present_cache

    attention_cls.forward = _forward
    attention_cls.forward_step = _forward_step
    if mlp_cls is not None:
        mlp_cls.forward = _mlp_forward
    if norm_cls is not None:
        norm_cls.forward = _norm_forward
    if decoder_layer_cls is not None and model_cls is not None:
        decoder_layer_cls.forward = _decoder_forward
        decoder_layer_cls.forward_step = _decoder_forward_step
        model_cls.forward = _model_forward
        model_cls.forward_step = _model_forward_step
    attention_cls._trtmc_explicit_gqa_patch = True


def _is_supported_down_proj_variant(variant: str) -> bool:
    return (
        variant in _VOXCPM2_DOWN_PROJ_VARIANTS
        or _VOXCPM2_SPLIT_K_DOWN_PROJ_VARIANT_RE.match(variant) is not None
        or _VOXCPM2_SPLIT_OUT_DOWN_PROJ_VARIANT_RE.match(variant) is not None
    )


def _split_k_down_proj_variant_config(variant: str) -> tuple[int, str] | None:
    match = _VOXCPM2_SPLIT_K_DOWN_PROJ_VARIANT_RE.match(variant)
    if match is None:
        return None
    return int(match.group("chunk")), match.group("mode")


def _split_out_down_proj_variant_config(variant: str) -> int | None:
    match = _VOXCPM2_SPLIT_OUT_DOWN_PROJ_VARIANT_RE.match(variant)
    if match is None:
        return None
    return int(match.group("chunk"))


def _validate_down_proj_variant(variant: str) -> str:
    normalized = variant.strip().lower()
    if _is_supported_down_proj_variant(normalized):
        return normalized
    valid = ", ".join(_VOXCPM2_DOWN_PROJ_VARIANTS)
    raise ValueError(
        f"Unsupported VoxCPM2 TSLM down-proj export variant {variant!r}; "
        f"valid values: {valid}, split_k_<chunk>_bf16_accum, "
        "split_k_<chunk>_fp32_accum_to_bf16, split_out_<chunk>_bf16"
    )


def _selected_tslm_down_proj_variant() -> str:
    raw = os.environ.get(_VOXCPM2_TSLM_DOWN_PROJ_VARIANT_ENV, "")
    if not raw.strip():
        return _VOXCPM2_DEFAULT_DOWN_PROJ_VARIANT
    return _validate_down_proj_variant(raw)


def _make_down_proj_variant_module(
    torch_module: Any,
    linear_module: Any,
    variant: str,
) -> Any:
    variant = _validate_down_proj_variant(variant)
    if variant == _VOXCPM2_DEFAULT_DOWN_PROJ_VARIANT:
        return linear_module

    split_k_config = _split_k_down_proj_variant_config(variant)
    split_out_chunk_size = _split_out_down_proj_variant_config(variant)

    class DownProjVariant(torch_module.nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.variant = variant
            self.split_k_config = split_k_config
            self.split_out_chunk_size = split_out_chunk_size
            weight = module.weight.detach()
            bias = getattr(module, "bias", None)
            if variant in {"fp32_accum_to_bf16", "fp32_output"}:
                weight = weight.float()
                bias = None if bias is None else bias.detach().float()
            elif bias is not None:
                bias = bias.detach()

            if variant == "pretransposed_matmul_bf16":
                self.register_buffer(
                    "weight_t",
                    weight.transpose(0, 1).contiguous().clone(),
                )
            else:
                self.register_buffer("weight", weight.clone())
            self.register_buffer("bias", None if bias is None else bias.clone())

        def _add_bias(self, output: Any, bias: Any | None = None) -> Any:
            selected_bias = self.bias if bias is None else bias
            if selected_bias is not None:
                output = output + selected_bias
            return output

        def _matmul_with_weight(self, down_proj_input: Any) -> Any:
            return torch_module.matmul(
                down_proj_input,
                self.weight.transpose(0, 1),
            )

        def forward(self, down_proj_input: Any) -> Any:
            if self.variant == "functional_linear":
                return torch_module.nn.functional.linear(
                    down_proj_input,
                    self.weight,
                    self.bias,
                )
            if self.variant == "pretransposed_matmul_bf16":
                return self._add_bias(
                    torch_module.matmul(down_proj_input, self.weight_t)
                )
            if self.variant == "addmm_zero":
                original_shape = down_proj_input.shape[:-1]
                flat_input = down_proj_input.reshape(-1, down_proj_input.shape[-1])
                if self.bias is None:
                    base = flat_input.new_zeros(
                        (flat_input.shape[0], self.weight.shape[0])
                    )
                else:
                    base = self.bias.unsqueeze(0).expand(
                        flat_input.shape[0],
                        self.bias.shape[0],
                    )
                output = torch_module.addmm(
                    base,
                    flat_input,
                    self.weight.transpose(0, 1),
                )
                return output.reshape(*original_shape, self.weight.shape[0])
            if self.variant == "einsum":
                return self._add_bias(
                    torch_module.einsum("...i,oi->...o", down_proj_input, self.weight)
                )
            if self.variant == "batched_bmm":
                squeezed = down_proj_input.ndim == 2
                batched_input = (
                    down_proj_input.unsqueeze(0) if squeezed else down_proj_input
                )
                transposed = self.weight.transpose(0, 1)
                batched_weight = transposed.unsqueeze(0).expand(
                    batched_input.shape[0],
                    transposed.shape[0],
                    transposed.shape[1],
                )
                output = torch_module.bmm(batched_input, batched_weight)
                if squeezed:
                    output = output.squeeze(0)
                return self._add_bias(output)
            if self.variant in {"fp32_accum_to_bf16", "fp32_output"}:
                output = torch_module.matmul(
                    down_proj_input.float(),
                    self.weight.transpose(0, 1),
                )
                output = self._add_bias(output)
                if self.variant == "fp32_accum_to_bf16":
                    output = output.to(dtype=torch_module.bfloat16)
                return output
            if self.split_k_config is not None:
                chunk_size, mode = self.split_k_config
                partials = []
                in_features = self.weight.shape[1]
                for start in range(0, in_features, chunk_size):
                    end = min(start + chunk_size, in_features)
                    partials.append(
                        torch_module.matmul(
                            down_proj_input[..., start:end],
                            self.weight[:, start:end].transpose(0, 1),
                        )
                    )
                if mode == "fp32_accum_to_bf16":
                    output = partials[0].float()
                    for partial in partials[1:]:
                        output = output + partial.float()
                    bias = None if self.bias is None else self.bias.float()
                    return self._add_bias(output, bias).to(
                        dtype=torch_module.bfloat16
                    )

                output = partials[0]
                for partial in partials[1:]:
                    output = (output + partial).to(dtype=torch_module.bfloat16)
                return self._add_bias(output).to(dtype=torch_module.bfloat16)
            if self.split_out_chunk_size is not None:
                partials = []
                out_features = self.weight.shape[0]
                for start in range(0, out_features, self.split_out_chunk_size):
                    end = min(start + self.split_out_chunk_size, out_features)
                    output = torch_module.matmul(
                        down_proj_input,
                        self.weight[start:end].transpose(0, 1),
                    )
                    bias = None if self.bias is None else self.bias[start:end]
                    partials.append(self._add_bias(output, bias))
                return torch_module.cat(partials, dim=-1)
            return self._add_bias(self._matmul_with_weight(down_proj_input))

    return DownProjVariant(linear_module)


def _patch_tslm_down_proj_modules_for_export(
    torch_module: Any,
    base_lm: Any,
    variant: str | None = None,
) -> int:
    selected = _selected_tslm_down_proj_variant() if variant is None else variant
    selected = _validate_down_proj_variant(selected)
    if selected == _VOXCPM2_DEFAULT_DOWN_PROJ_VARIANT:
        return 0

    replaced = 0
    for layer in getattr(base_lm, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is None:
            continue
        setattr(
            mlp,
            "down_proj",
            _make_down_proj_variant_module(torch_module, down_proj, selected),
        )
        replaced += 1
    return replaced


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
        noise: Any | None = None,
        temperature: float = 1.0,
        cfg_value: Any = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> Any:
        batch, _ = mu.shape
        if noise is None:
            noise = torch_module.randn(
                (batch, self.in_channels, patch_size),
                device=mu.device,
                dtype=torch_module.float32,
            )
        else:
            noise = noise.to(device=mu.device, dtype=torch_module.float32)
        noise = noise.to(dtype=mu.dtype)
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
    _patch_tslm_down_proj_modules_for_export(torch, base_lm)

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
            step_output = self.base_lm.forward_step(
                combined_embed[:, 0, :],
                position_id.to(dtype=torch.long),
            )
            if isinstance(step_output, tuple):
                raw_hidden, present_cache = step_output
            else:
                raw_hidden = step_output
                present_cache = self.base_lm.kv_cache.kv_cache
            raw_hidden = raw_hidden.to(dtype=compute_dtype)
            semantic_lm_state = self.fsq_layer(raw_hidden) * a_mask.reshape(
                -1, 1
            ) + raw_hidden * t_mask.reshape(-1, 1)
            lm_hidden = semantic_lm_state
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            return (
                semantic_lm_state,
                lm_hidden,
                stop_logits,
                present_cache,
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


def build_tslm_prefill_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.layers import ScalarQuantizationLayer
        from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 TSLM prefill TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 TSLM prefill builder expected lm_config")

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
    _patch_tslm_down_proj_modules_for_export(torch, base_lm)

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

    class TSLMPrefillWrapper(torch.nn.Module):
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
        ) -> tuple[Any, Any, Any, Any]:
            local_features = local_text_features.unsqueeze(0).to(dtype=compute_dtype)
            tokens = text_tokens.unsqueeze(0).to(dtype=torch.long)
            t_mask = text_mask.unsqueeze(0).to(dtype=compute_dtype)
            a_mask = audio_mask.unsqueeze(0).to(dtype=compute_dtype)
            text_embed = self.base_lm.embed_tokens(tokens) * self.scale_emb
            combined_embed = t_mask.unsqueeze(-1) * text_embed + a_mask.unsqueeze(
                -1
            ) * local_features
            raw_hidden, next_cache = self.base_lm(
                inputs_embeds=combined_embed,
                is_causal=True,
            )
            raw_hidden = raw_hidden.to(dtype=compute_dtype)
            semantic_lm_states = self.fsq_layer(raw_hidden) * a_mask.unsqueeze(
                -1
            ) + raw_hidden * t_mask.unsqueeze(-1)
            semantic_lm_states = semantic_lm_states.squeeze(0)
            lm_hidden = semantic_lm_states
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            present_cache = _stack_minicpm_present_cache(torch, next_cache).to(
                dtype=compute_dtype
            )
            return semantic_lm_states, lm_hidden, stop_logits, present_cache

    scale_emb = float(lm_config.get("scale_emb", 1.0))
    if not bool(lm_config.get("use_mup", False)):
        scale_emb = 1.0
    wrapper = TSLMPrefillWrapper(
        base_lm,
        fsq_layer,
        stop_proj,
        stop_head,
        scale_emb=scale_emb,
    )
    wrapper.eval()
    text_steps = _tslm_export_text_steps(ctx)
    example_args = (
        torch.zeros((text_steps, hidden_size), dtype=compute_dtype),
        torch.zeros((text_steps,), dtype=torch.int32),
        torch.ones((text_steps,), dtype=compute_dtype),
        torch.zeros((text_steps,), dtype=compute_dtype),
    )
    return _compile_voxcpm2_tslm_prefill_onnx(
        wrapper, example_args, verbose=ctx.verbose
    )


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
            step_output = self.residual_lm.forward_step(
                residual_inputs,
                position_id.to(dtype=torch.long),
            )
            if isinstance(step_output, tuple):
                residual_hidden, present_cache = step_output
            else:
                residual_hidden = step_output
                present_cache = self.residual_lm.kv_cache.kv_cache
            residual_hidden = residual_hidden.to(dtype=compute_dtype)
            return residual_hidden, present_cache

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


def build_ralm_prefill_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    prepared = prepare_component_inputs(ctx)
    try:
        import torch
        from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 RALM prefill TRT builder requires torch and the "
            "upstream voxcpm package"
        ) from exc

    _patch_minicpm_attention_gqa_for_torch_trt(torch)
    compute_dtype = _torch_dtype(torch, ctx.precision)
    lm_config = prepared.config_values.get("lm_config")
    if not isinstance(lm_config, Mapping):
        raise ValueError("VoxCPM2 RALM prefill builder expected lm_config")

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

    fusion_concat_proj = torch.nn.Linear(hidden_size * 2, hidden_size)
    fusion_concat_proj.load_state_dict(
        _to_torch_state_dict(torch, state, "fusion_concat_proj.", dtype=compute_dtype),
        strict=True,
    )
    fusion_concat_proj.to(dtype=compute_dtype)
    fusion_concat_proj.eval()

    class RALMPrefillWrapper(torch.nn.Module):
        def __init__(self, residual_lm_module: Any, fusion_module: Any) -> None:
            super().__init__()
            self.residual_lm = residual_lm_module
            self.fusion_concat_proj = fusion_module

        def forward(
            self,
            semantic_lm_states: Any,
            audio_mask: Any,
            local_text_features: Any,
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
            residual_hidden, next_cache = self.residual_lm(
                inputs_embeds=residual_inputs.unsqueeze(0),
                is_causal=True,
            )
            residual_hidden = residual_hidden.squeeze(0).to(dtype=compute_dtype)
            present_cache = _stack_minicpm_present_cache(torch, next_cache).to(
                dtype=compute_dtype
            )
            return residual_hidden, present_cache

    wrapper = RALMPrefillWrapper(residual_lm, fusion_concat_proj)
    wrapper.eval()
    text_steps = _ralm_export_text_steps(ctx)
    example_args = (
        torch.zeros((text_steps, hidden_size), dtype=compute_dtype),
        torch.zeros((text_steps,), dtype=compute_dtype),
        torch.zeros((text_steps, hidden_size), dtype=compute_dtype),
    )
    return _compile_voxcpm2_ralm_prefill_onnx(
        wrapper, example_args, verbose=ctx.verbose
    )


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
            locdit_noise: Any,
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
            noise = (
                locdit_noise.to(dtype=compute_dtype)
                .transpose(0, 1)
                .contiguous()
                .unsqueeze(0)
            )
            noise = noise.repeat(dit_hidden.size(0), 1, 1)
            cfg = cfg_value.reshape(()) if getattr(cfg_value, "ndim", 0) else cfg_value
            latents = self.feat_decoder(
                mu=dit_hidden,
                patch_size=patch_size,
                cond=cond,
                noise=noise,
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


def _stack_minicpm_present_cache(torch_module: Any, next_cache: Any) -> Any:
    key_caches = []
    value_caches = []
    for key_cache, value_cache in next_cache:
        key_caches.append(key_cache)
        value_caches.append(value_cache)
    return torch_module.stack(
        (
            torch_module.stack(key_caches, dim=0),
            torch_module.stack(value_caches, dim=0),
        ),
        dim=0,
    )


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


def _locdit_export_text_steps(_ctx: VoxCPM2ComponentBuildContext) -> int:
    # LocDiT is invoked for one autoregressive patch at a time; TSLM/RALM own cache length.
    return 1


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
        diagnostic_label="voxcpm2_tslm_decode",
    )


def _compile_voxcpm2_tslm_prefill_onnx(
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
        ],
        output_names=[
            "semantic_lm_states",
            "lm_hidden",
            "stop_logits",
            "tslm_present_kv_cache",
        ],
        verbose=verbose,
        diagnostic_label="voxcpm2_tslm_prefill",
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


def _compile_voxcpm2_ralm_prefill_onnx(
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
            "locdit_noise",
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


def _locenc_export_text_steps(_ctx: VoxCPM2ComponentBuildContext) -> int:
    # Generated patches call LocEnc one at a time. Zero-audio prefill rows are
    # repeated from the optional CUDA-computed prefill table when available,
    # because upstream BF16 numerics depend on the active text-step batch size.
    return 1


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


@dataclass(frozen=True)
class _LocEncBuildArtifacts:
    torch: Any
    wrapper: Any
    compute_dtype: Any
    patch_size: int
    feat_dim: int
    hidden_size: int


def _make_locenc_build_artifacts(
    ctx: VoxCPM2ComponentBuildContext,
    *,
    patch_for_export: bool = True,
) -> _LocEncBuildArtifacts:
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

    if patch_for_export:
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
    return _LocEncBuildArtifacts(
        torch=torch,
        wrapper=wrapper,
        compute_dtype=compute_dtype,
        patch_size=int(prepared.config_values.get("patch_size", 4)),
        feat_dim=feat_dim,
        hidden_size=lm_hidden_size,
    )


def build_locenc_engine(ctx: VoxCPM2ComponentBuildContext) -> bytes:
    artifacts = _make_locenc_build_artifacts(ctx)
    example_args = (
        artifacts.torch.zeros(
            (_locenc_export_text_steps(ctx), artifacts.patch_size, artifacts.feat_dim),
            dtype=artifacts.compute_dtype,
        ),
    )
    return _compile_voxcpm2_locenc_onnx(
        artifacts.wrapper, example_args, verbose=ctx.verbose
    )


def _zero_prefill_table_max_steps() -> int:
    raw = os.environ.get("TRTMC_VOXCPM2_ZERO_PREFILL_TABLE_MAX_STEPS")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return _VOXCPM2_ZERO_PREFILL_TABLE_DEFAULT_MAX_STEPS
    return _VOXCPM2_ZERO_PREFILL_TABLE_DEFAULT_MAX_STEPS


def _full_prefill_max_steps() -> int:
    raw = os.environ.get("TRTMC_VOXCPM2_FULL_PREFILL_MAX_STEPS")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return _VOXCPM2_FULL_PREFILL_DEFAULT_MAX_STEPS
    return _VOXCPM2_FULL_PREFILL_DEFAULT_MAX_STEPS


def _should_package_full_prefill(ctx: VoxCPM2ComponentBuildContext) -> bool:
    max_steps = _full_prefill_max_steps()
    text_steps = _tslm_export_text_steps(ctx)
    if max_steps <= 0:
        raise ValueError(
            "VoxCPM2 full-sequence LM prefill is required for Hugging Face "
            "parity; TRTMC_VOXCPM2_FULL_PREFILL_MAX_STEPS must be positive."
        )
    if text_steps > max_steps:
        raise ValueError(
            "VoxCPM2 full-sequence LM prefill is required for Hugging Face "
            f"parity, but export text steps {text_steps} exceed "
            f"TRTMC_VOXCPM2_FULL_PREFILL_MAX_STEPS={max_steps}."
        )
    return True


def build_locenc_zero_prefill_feature_table(
    ctx: VoxCPM2ComponentBuildContext,
) -> bytes:
    # The table is a serialized HF-reference tensor, not an export graph. Keep
    # it on the upstream eager LocEnc path because the TensorRT export patch can
    # shift BF16 rows by one ULP before the first TSLM token is generated.
    artifacts = _make_locenc_build_artifacts(ctx, patch_for_export=False)
    torch = artifacts.torch
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return b""
    if not cuda.is_available():
        return b""

    max_steps = _zero_prefill_table_max_steps()
    if max_steps <= 0:
        return b""

    device = torch.device("cuda")
    wrapper = artifacts.wrapper.to(device=device)
    rows: list[tuple[int, bytes]] = []
    with torch.no_grad():
        for text_steps in range(1, max_steps + 1):
            audio_feats = torch.zeros(
                (text_steps, artifacts.patch_size, artifacts.feat_dim),
                dtype=artifacts.compute_dtype,
                device=device,
            )
            local_text_features = wrapper(audio_feats)
            row = (
                local_text_features[0]
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            rows.append((text_steps, row.view(torch.uint8).numpy().tobytes()))

    header = struct.pack(
        "<III",
        _VOXCPM2_ZERO_PREFILL_FEATURES_VERSION,
        len(rows),
        artifacts.hidden_size,
    )
    body = bytearray()
    expected_row_bytes = artifacts.hidden_size * 2
    for text_steps, row_bytes in rows:
        if len(row_bytes) != expected_row_bytes:
            raise ValueError(
                "VoxCPM2 LocEnc zero-prefill row for "
                f"{text_steps} text steps has {len(row_bytes)} bytes, "
                f"expected {expected_row_bytes}"
            )
        body.extend(struct.pack("<I", text_steps))
        body.extend(row_bytes)
    return header + bytes(body)


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
        component_ctx: VoxCPM2ComponentBuildContext | None = None
        builder: VoxCPM2ComponentBuilder | None = None
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
            builder = selected_builders[spec.name]

            component_ctx = VoxCPM2ComponentBuildContext(
                spec=spec,
                model_dir=model_dir,
                config=config,
                source=sources[spec.name],
                precision=precision,
                verbose=verbose,
                max_cache_length=max_cache_length,
            )
            prefill_table = b""
            if spec.name == "locenc" and builder is build_locenc_engine:
                prefill_table = build_locenc_zero_prefill_feature_table(component_ctx)
            plan = builder(component_ctx)
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
        if (
            spec.name == "tslm"
            and component_ctx is not None
            and builder is build_tslm_engine
            and _should_package_full_prefill(component_ctx)
        ):
            sections[VOXCPM2_TSLM_PREFILL_ENGINE_SECTION] = build_tslm_prefill_engine(
                component_ctx
            )
        if (
            spec.name == "ralm"
            and component_ctx is not None
            and builder is build_ralm_engine
            and _should_package_full_prefill(component_ctx)
        ):
            sections[VOXCPM2_RALM_PREFILL_ENGINE_SECTION] = build_ralm_prefill_engine(
                component_ctx
            )
        if (
            spec.name == "locenc"
            and component_ctx is not None
            and builder is build_locenc_engine
            and prefill_table
        ):
            sections[VOXCPM2_ZERO_PREFILL_FEATURES_SECTION] = prefill_table
    return sections

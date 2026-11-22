# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact documented component precision contracts; upstream owns packing/math."""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path


def object_file(path: Path) -> dict:
    """Read a required JSON object, rejecting malformed metadata."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def expected_mixed_layers() -> dict[str, dict]:
    """Return the documented precision-defining module coverage, not a model graph."""
    result = {}
    for layer in range(40):
        prefix = f"model.language_model.layers.{layer}."
        projections = (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
            if (layer + 1) % 4 == 0
            else ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
        )
        for projection in projections:
            result[prefix + projection] = {"quant_algo": "FP8"}
        for module in (
            "mlp.experts",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
        ):
            result[prefix + module] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    result["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    return result


def validate_config(raw: dict) -> None:
    """Require a documented hybrid, shared-expert and visual topology."""
    text, visual = raw.get("text_config"), raw.get("vision_config")
    if not isinstance(text, dict) or not isinstance(visual, dict):
        raise ValueError("Full Qwen3.5-MoE text and visual configs are required")
    if raw.get("model_type") != "qwen3_5_moe" or raw.get("architectures") != [
        "Qwen3_5MoeForConditionalGeneration"
    ]:
        raise ValueError("Unsupported Qwen3.5-MoE architecture")
    expected = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "attn_output_gate": True,
        "eos_token_id": 248044,
        "hidden_act": "silu",
        "attention_bias": False,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "mamba_ssm_dtype": "float32",
    }
    if text.get("num_hidden_layers") == 48:
        # The 122B-A10B has the same layer order and expert count, not the same widths.
        expected.update(num_hidden_layers=48, hidden_size=3072, num_attention_heads=32,
                        moe_intermediate_size=1024, shared_expert_intermediate_size=1024,
                        linear_num_value_heads=64)
    if any(
        text.get(key) != value or type(text.get(key)) is not type(value)
        for key, value in expected.items()
    ):
        raise ValueError("Unsupported hybrid/shared-expert text configuration")
    layers = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(expected["num_hidden_layers"])]
    if (
        text.get("layer_types") != layers
        or text.get("full_attention_interval", 4) != 4
        or text.get("mlp_only_layers", []) != []
    ):
        raise ValueError("Unsupported hybrid attention layer order")
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if (
        not isinstance(rope, dict)
        or rope.get("partial_rotary_factor") != 0.25
        or rope.get("mrope_section") != [11, 11, 10]
        or rope.get("rope_theta") != 10000000
        or rope.get("mrope_interleaved") is not True
        or rope.get("rope_type") != "default"
    ):
        raise ValueError("Unsupported hybrid rotary configuration")
    expected_visual = {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": expected["hidden_size"],
        "in_channels": 3,
    }
    if (
        any(
            visual.get(k) != v or type(visual.get(k)) is not int for k, v in expected_visual.items()
        )
        or visual.get("deepstack_visual_indexes", []) != []
    ):
        raise ValueError("Unsupported hybrid-MoE visual configuration")
    if visual.get("model_type") not in {"qwen3_5_moe", "qwen3_5_moe_vision"}:
        raise ValueError("Unsupported visual model type")
    if any(
        raw.get(k) != v
        for k, v in {
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        }.items()
    ):
        raise ValueError("Unsupported visual token IDs")
    for component in (raw, text, visual):
        if any(
            component.get(k)
            for k in (
                "draft_config",
                "dflash_config",
                "dspark_config",
                "jetspec_config",
                "eagle_config",
                "speculative_config",
                "audio_config",
            )
        ):
            raise ValueError("Draft/audio configurations are outside ordinary family support")
    capacity = text.get("max_position_embeddings")
    if type(capacity) is not int or capacity <= 1:
        raise ValueError("Invalid source context capacity")


def _mixed_layers(value: object) -> dict:
    """Validate raw spelling/dominance and uniform group sizes used by pinned parser."""
    expected = expected_mixed_layers()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Mixed precision must cover exactly the documented modules")
    for name, required in expected.items():
        entry = value[name]
        if (
            not isinstance(entry, dict)
            or entry.get("quant_algo") != required["quant_algo"]
            or entry.get("group_size", 1) != required.get("group_size", 1)
        ):
            raise ValueError("Mixed precision algorithm/group conflict")
        if any(k in entry for k in ("bits", "num_bits", "quant_method", "zero_point", "format")):
            raise ValueError("Unqualified precision-defining mixed layer fields")
    counts = Counter(entry["quant_algo"] for entry in value.values())
    if counts != {"W4A16_NVFP4": 161, "FP8": 130}:
        raise ValueError("Pinned expert kernel must resolve NVFP4 as dominant format")
    return expected


def _precision_modifiers(description: dict, *, bits: int) -> None:
    """Do not let optional precision-defining fields contradict the packed recipe."""
    if "symmetric" in description and description["symmetric"] is not True:
        raise ValueError("Only symmetric float precision is supported")
    if "strategy" in description and description["strategy"] != (
        "group" if bits == 4 else "tensor"
    ):
        raise ValueError("Conflicting precision strategy")
    if any(
        key in description
        for key in (
            "zero_point",
            "block_structure",
            "actorder",
            "format",
            "quant_method",
            "quant_algo",
        )
    ):
        raise ValueError("Unqualified precision-defining group modifier")


def _fp8_kv(description: dict) -> None:
    """Absent embedded KV is allowed; explicit embedded/sidecar KV must agree."""
    if "kv_cache_quant_algo" in description and description["kv_cache_quant_algo"] != "FP8":
        raise ValueError("Conflicting embedded FP8 KV algorithm")
    if "kv_cache_scheme" in description:
        scheme = description["kv_cache_scheme"]
        if (
            not isinstance(scheme, dict)
            or scheme.get("num_bits") != 8
            or scheme.get("type") != "float"
            or scheme.get("dynamic", False) is not False
            or scheme.get("group_size", 1) != 1
        ):
            raise ValueError("Conflicting embedded FP8 KV scheme")
        _precision_modifiers(scheme, bits=8)


def source_policy(request, raw: dict) -> dict:
    """Preserve exact known GPTQ or mixed ModelOpt source; never calibrate/dequantize."""
    root = Path(request.model_dir)
    if any((root / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        raise ValueError("Unknown quantization sidecar")
    for component in (raw["text_config"], raw["vision_config"]):
        if component.get("quantization_config") not in (None, {}):
            raise ValueError("Conflicting component-specific quantization metadata")
    quant = raw.get("quantization_config")
    if quant is None:
        if (root / "hf_quant_config.json").exists() or raw["text_config"].get("dtype") != "bfloat16":
            raise ValueError("Original expert weights require BF16 source metadata without a quantizer")
        policy = {"weights": "none", "llm": "original_bf16_experts", "visual": "fp16",
                  "kv": "fp16", "group_size": 0}
    elif not isinstance(quant, dict):
        raise ValueError("Invalid quantization metadata")
    elif quant.get("quant_method") == "gptq":
        if raw["text_config"]["num_hidden_layers"] != 40:
            raise ValueError("This packed source recipe requires the 35B topology")
        if (root / "hf_quant_config.json").exists():
            raise ValueError("GPTQ cannot be overridden by a ModelOpt sidecar")
        if any(
            quant.get(k) != v
            for k, v in {"bits": 4, "group_size": 128, "desc_act": False, "sym": True}.items()
        ):
            raise ValueError("Only symmetric group128 non-act-order GPTQ is qualified")
        if any(
            k in quant
            for k in (
                "quant_algo",
                "format",
                "checkpoint_format",
                "zero_point",
                "kv_cache_scheme",
                "kv_cache_quant_algo",
            )
        ):
            raise ValueError("Conflicting GPTQ layout or KV metadata")
        expected_dynamic = {
            "lm_head": {},
            "model.language_model.embed_tokens": {},
            "-:.*attn.*": {},
            "-:.*shared_expert.*": {},
            "-:.*mtp.*": {},
            "-:.*visual.*": {},
        }
        if (
            quant.get("dynamic") != expected_dynamic
            or quant.get("modules_to_not_convert", []) != []
        ):
            raise ValueError("GPTQ component exclusion layout differs")
        policy = {
            "weights": "int4",
            "llm": "gptq_experts_original_shared_attention",
            "visual": "fp16",
            "kv": "fp16",
            "group_size": 128,
        }
    elif quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "NVFP4":
        text = raw["text_config"]
        if text["num_hidden_layers"] != 48:
            raise ValueError("Routed-only NVFP4 requires the documented 122B topology")
        side = object_file(root / "hf_quant_config.json").get("quantization")
        excluded = {"lm_head", "model.visual*", "mtp.layers.0*", "mtp*"}
        for layer, kind in enumerate(text["layer_types"]):
            prefix = f"model.language_model.layers.{layer}."
            excluded.update(prefix + key for key in ("mlp.shared_expert*", "mlp.shared_expert_gate",
                            "self_attn*" if kind == "full_attention" else "linear_attn*"))
        if not isinstance(side, dict) or side.get("quant_algo") != "NVFP4" or side.get("group_size") != 16:
            raise ValueError("Routed-only NVFP4 requires a matching group16 ModelOpt sidecar")
        if set(quant.get("ignore", [])) != excluded or set(side.get("exclude_modules", [])) != excluded:
            raise ValueError("Routed-only NVFP4 exclusions differ from the documented checkpoint")
        group = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
        if quant.get("config_groups") != {"group_0": {"targets": ["Linear"], "weights": group,
                                                      "input_activations": group}}:
            raise ValueError("Routed-only experts require calibrated W4A4 group16")
        if (side.get("kv_cache_quant_algo") != "FP8"
                or quant.get("kv_cache_scheme") != {"dynamic": False, "num_bits": 8, "type": "float"}):
            raise ValueError("Unexpected declared ModelOpt KV metadata")
        # Edge resolves ModelOpt's global KV claim from actual LLM K-scale keys.
        # This checkpoint has none; validate that fact against the original headers.
        policy = {"weights": "nvfp4", "llm": "nvfp4_routed_experts_original_shared_head_attention",
                  "visual": "fp16", "kv": "fp16", "group_size": 16}
    elif quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "MIXED_PRECISION":
        if raw["text_config"]["num_hidden_layers"] != 40:
            raise ValueError("This packed source recipe requires the 35B topology")
        side = object_file(root / "hf_quant_config.json").get("quantization")
        if (
            not isinstance(side, dict)
            or side.get("quant_algo") != "MIXED_PRECISION"
            or side.get("kv_cache_quant_algo") != "FP8"
        ):
            raise ValueError("Mixed ModelOpt requires its exact FP8-KV sidecar")
        _fp8_kv(quant)
        _fp8_kv(side)
        expected = _mixed_layers(side.get("quantized_layers"))
        _mixed_layers(quant.get("quantized_layers"))
        exclusions = {"mtp.layers.0*", "mtp*"}
        if (
            set(quant.get("ignore", [])) != exclusions
            or set(side.get("exclude_modules", [])) != exclusions
        ):
            raise ValueError("Conflicting ModelOpt exclusions")
        groups = quant.get("config_groups")
        if not isinstance(groups, dict) or set(groups) != {"group_0", "group_1"}:
            raise ValueError("Unqualified mixed ModelOpt groups")
        for name, algo, bits in (("group_0", "FP8", 8), ("group_1", "W4A16_NVFP4", 4)):
            group = groups[name]
            targets = {k for k, v in expected.items() if v["quant_algo"] == algo}
            if set(group.get("targets", [])) != targets:
                raise ValueError("ModelOpt group targets conflict with layer policy")
            for field in ("weights", "input_activations"):
                description = group.get(field, {})
                if not isinstance(description, dict):
                    raise ValueError("Invalid ModelOpt precision group")
                _precision_modifiers(description, bits=bits)
                if (
                    description.get("num_bits") != bits
                    or description.get("type") != "float"
                    or description.get("dynamic") is not False
                    or description.get("group_size", 1) != (16 if bits == 4 else 1)
                ):
                    raise ValueError("ModelOpt group precision conflicts with layer policy")
        policy = {
            "weights": "nvfp4",
            "llm": "mixed_nvfp4_experts_shared_head_fp8_attention_gdn",
            "visual": "fp16",
            "kv": "fp8",
            "group_size": 16,
            "layers": expected,
        }
    else:
        raise ValueError("Only documented GPTQ or mixed NVFP4 sources are supported")
    if request.quantization is not None and request.quantization != policy["weights"]:
        raise ValueError("Explicit requested quantization must match source weights exactly")
    return policy

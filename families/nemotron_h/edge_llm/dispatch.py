# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Nemotron-H complete-network map and native preparation retry."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

from ..config import normalize_target_layers
from . import builder as edge_llm
from .edge_quantization import source_quantization
from .edge_config import topology_matches

_LOG = logging.getLogger(__name__)
# Native platforms with recorded passing family profiles; admission is not qualification.
PLATFORMS = {"x86_64": (80, 120)}
EDGE_DISPATCH = {
    ("linux", arch, sm, precision): edge_llm.prepare
    for arch, sms in PLATFORMS.items()
    for sm in sms
    for precision in ("fp16", "fp8", "nvfp4")
    if precision == "fp16" or sm >= 100
}


def request_matches(request) -> bool:
    """Preserve native early rejection order for unmapped request controls."""
    return (
        request.family == "nemotron_h" and request.backend == "trt" and request.task == "text_generation"
        and request.precision.lower() == "fp16" and request.quantization in {None, "none", "fp8", "nvfp4"}
        and request.max_batch_size == request.tensor_parallel_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
    )


def candidate(request, raw: dict) -> bool:
    """Preserve native tokenizer contract and require exact source hybrid policy."""
    if not request_matches(request):
        return False
    precision = source_quantization(request, raw)
    if precision is None or not topology_matches(raw, precision):
        return False
    try:
        from .edge_tokenizer import source_chat_template
        source_chat_template(Path(request.model_dir))
        return True
    except (OSError, ValueError):
        return False


def platform_matches(raw: dict, target: dict) -> bool:
    """Admit the native runtime, not just its optional optimized SSD kernel."""
    # MambaPlugin falls back to scalar prefill when SSD cannot implement a shape.
    # Topology, source precision and the installed package are checked separately.
    return target["sm"] >= 80


def _adapter_for(raw: dict, target: dict, precision: str):
    """Keep the legacy map; let the 0.12 native SDK determine operator availability."""
    if not platform_matches(raw, target):
        return None
    key = (target["os"], target["arch"], target["sm"], precision)
    # The documented compressed FP8 MoE lowering is new in the 0.12 direct builder.
    if ((raw.get("quantization_config") or {}).get("quant_method") == "compressed-tensors"
            and edge_llm.selected_release() != "0.12.0"):
        return None
    if key in EDGE_DISPATCH:
        return EDGE_DISPATCH[key]
    if (target["os"] == "linux" and target["arch"] in {"aarch64", "x86_64"}
            and edge_llm.selected_release() == "0.12.0"):
        # Admission is not qualification. prepare checks the exact installed SM,
        # CUDA and TRT; an unsupported operator retries the unchanged native request.
        return edge_llm.prepare
    return None


def _compiler_profile(raw: dict, precision: str | None, variant: str) -> tuple[str, int] | None:
    """Identify a measured target graph, not a general restriction on quantized MoE."""
    raw = normalize_target_layers(raw)
    shape = (raw.get("num_hidden_layers"), raw.get("hidden_size"), raw.get("n_routed_experts"))
    policy = raw.get("quantization_config") or {}
    if (raw.get("model_type") != "nemotron_h" or not isinstance(policy, dict)
            or (raw.get("num_attention_heads"), raw.get("num_key_value_heads")) != (32, 2)):
        return None
    if shape == (88, 4096, 512) and variant == "ordinary" and precision == "nvfp4":
        if raw.get("moe_latent_size") == 1024 and policy.get("quant_algo") == "MIXED_PRECISION":
            return "Super 120B NVFP4", 103
    if shape != (52, 2688, 128):
        return None
    if (variant == "ordinary" and precision == "nvfp4" and not policy
            and raw.get("num_nextn_predict_layers") is None):
        return "Nano 30B NVFP4", 103
    if raw.get("num_nextn_predict_layers") == 1:
        if (variant in {"ordinary", "dflash", "dspark", "dspark_tree"} and precision == "nvfp4"
                and policy.get("quant_algo") == "MIXED_PRECISION"):
            return "Lightning 30B NVFP4", 103
        if variant == "ordinary" and precision == "fp8" and policy.get("quant_method") == "compressed-tensors":
            return "Lightning 30B compressed FP8", 100
    return None


def _measured_native_stack(target: dict, sm: int, arch: str = "aarch64") -> bool:
    return (
        (target.get("os"), target.get("arch"), target.get("sm")) == ("linux", arch, sm)
        and (target.get("cuda_version"), target.get("tensorrt_version")) == ("13.4", "11.4.0.106")
    )


def _native_gpu():
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if int(status) != 0:
        return None
    status, gpu = runtime.cudaGetDeviceProperties(device)
    return gpu if int(status) == 0 else None


def _check_native_route(request, raw: dict, target: dict | None, variant: str) -> None:
    # Unrelated native graphs and devices do not require Edge or its toolkit probe.
    os_name, arch = ((sys.platform, platform.machine()) if target is None
                     else (target.get("os"), target.get("arch")))
    if os_name != "linux" or arch not in {"aarch64", "x86_64"}:
        return
    if arch == "x86_64" and (variant != "ordinary" or raw.get("hidden_size") != 2688
                             or raw.get("num_nextn_predict_layers") is not None):
        return
    if raw.get("hidden_size") not in (2688, 4096) or raw.get("model_type") != "nemotron_h":
        return
    if raw.get("max_position_embeddings") is None or edge_llm.sequence_length(request, raw) != 256:
        return
    precision = source_quantization(request, raw)
    profile = _compiler_profile(raw, precision, variant)
    if profile is None or not topology_matches(raw, precision):
        return
    name, sm = profile
    device_label = f"ARM64 SM{sm}"
    if arch == "x86_64":
        if name != "Nano 30B NVFP4" or (target is not None and not _measured_native_stack(target, 120, arch)):
            return
        gpu = _native_gpu()
        if (gpu is None or gpu.major * 10 + gpu.minor != 120
                or bytes(gpu.name).split(b"\0", 1)[0] != b"NVIDIA RTX PRO 4500 Blackwell"):
            return
        device_label, sm = "x86-64 RTX PRO 4500 Blackwell", 120
    if target is None:
        try:
            target = edge_llm.local_target()
        except RuntimeError:
            # A different GPU cannot be the failing stack. On the matching
            # device, propagate the original discovery error rather than assume
            # an unknown toolkit is safe or report an unproved compiler crash.
            if arch == "aarch64":
                gpu = _native_gpu()
                if gpu is not None and gpu.major * 10 + gpu.minor != sm:
                    return
            raise
    if _measured_native_stack(target, sm, arch):
        raise ValueError(
            f"Nemotron {name} ({variant}, capacity 256): native TensorRT serialization crashed "
            f"on {device_label}, CUDA 13.4 and TensorRT 11.4.0.106. "
            "Not retrying this native build."
        )


def _no_ordinary_build_route(request, raw: dict, target: dict, precision: str) -> bool:
    """This one ordinary profile also failed both official Edge 0.12 builders."""
    return (
        _compiler_profile(raw, precision, "ordinary") == ("Super 120B NVFP4", 103)
        and edge_llm.sequence_length(request, raw) == 256 and _measured_native_stack(target, 103)
        and edge_llm.selected_release() == "0.12.0"
    )


def build(request, writer, native) -> None:
    """Prepare Edge privately or warn and retry unchanged native once.

    Publication and cancellation errors propagate. Runtime fallback is not part
    of this builder adapter; an Edge bundle is owned entirely by its runtime.
    """
    if not request_matches(request):
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    if not candidate(request, raw) or not edge_llm.package_present():
        _check_native_route(request, raw, None, "ordinary")
        native(request, writer)
        return
    if edge_llm.sequence_length(request, raw) > raw["max_position_embeddings"]:
        raise ValueError("Nemotron-H max_sequence_length exceeds checkpoint context capacity")
    target = None
    failure = None
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-", suffix=".log",
                                        dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = edge_llm.local_target()
            precision = source_quantization(request, raw)
            adapter = _adapter_for(raw, target, precision)
            unavailable = adapter is not None and _no_ordinary_build_route(request, raw, target, precision)
            if adapter is not None and not unavailable:
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("nemotron_h Edge build failed: %s. Diagnostics: %s. "
                         "Retrying native once with the unchanged request if eligible.", error, log_path, exc_info=True)
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            if unavailable:
                log_path.unlink()
                raise ValueError(
                    "Nemotron Super 120B NVFP4 at capacity 256 has no usable automatic build route "
                    "with Edge 0.12.0 on ARM64 SM103, CUDA 13.4 and TensorRT 11.4.0.106: "
                    "both Edge builds failed in tactic selection and native serialization crashed."
                )
            if adapter is not None:
                edge_llm.publish(request, writer, files, marker)
                log_path.unlink()
                return
            log_path.unlink()
    try:
        _check_native_route(request, raw, target, "ordinary")
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise


def build_speculative(request, writer, draft: Path, variant: str) -> None:
    """Offload the complete pair, or warn and retry the same native pair once."""
    from ..paired import build as native_paired

    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not candidate(request, raw) or source_quantization(request, raw) != "nvfp4":
        raise ValueError("Nemotron-H paired execution requires a compatible NVFP4 target and TP1 text request")
    if edge_llm.sequence_length(request, raw) > raw["max_position_embeddings"]:
        raise ValueError("Nemotron-H max_sequence_length exceeds checkpoint context capacity")
    if not edge_llm.package_present():
        _check_native_route(request, raw, None, variant)
        native_paired(request, writer, draft, variant)
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-onnx-", suffix=".log",
                                        dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    target = None
    failure = None
    with tempfile.TemporaryDirectory(prefix=f"trtmc-nemotron-h-{variant}-", dir=request.output_path.parent) as directory:
        try:
            target = edge_llm.local_target()
            use_edge = _adapter_for(raw, target, "nvfp4") is not None
            if use_edge:
                files, marker = edge_llm.prepare_paired(request, raw, target, Path(directory), log_path, draft, variant)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("Nemotron-H paired Edge ONNX build failed: %s. Diagnostics: %s. "
                         "Retrying the unchanged native pair once if eligible.", error, log_path, exc_info=True)
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            if use_edge:
                # Publication failure is not a preparation failure; never retry
                # after writing any part of an Edge bundle.
                edge_llm.publish(request, writer, files, marker)
                log_path.unlink()
                return
            log_path.unlink()
    try:
        _check_native_route(request, raw, target, variant)
        native_paired(request, writer, draft, variant)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise


def build_paired(request, writer, execution) -> None:
    """Keep the complete speculative pair and backend selection family-owned."""
    execution.validate_local()
    if execution.variant not in {"dflash", "dspark", "dspark_tree"} or tuple(
        x.role for x in execution.checkpoints
    ) != ("draft",):
        raise ValueError("Nemotron-H paired execution requires a supported variant and one draft")

    build_speculative(request, writer, execution.checkpoints[0].model_dir, execution.variant)

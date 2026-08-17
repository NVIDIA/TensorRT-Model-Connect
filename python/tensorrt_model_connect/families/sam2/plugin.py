# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only SAM2 family adapter for the native multi-plan bundle builder."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

from ..base import CompleteBundleBuildRequest
from . import archive_contract
from .model_config import require_reference_archive
from .native_builder import Sam2NativeBuilderError, locate_native_builder


_DEFAULT_PATH = "/usr/bin:/bin"
_PASSTHROUGH_ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
_PRECISIONS = frozenset({"bf16", "mixed_bf16_fp32"})
_FAMILY_OPTION_NAMESPACE = "sam2"
_FAMILY_OPTION_KEYS = frozenset({"created_at", "gpu_device", "workspace_bytes"})
_UTC_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


def _reject_nondefault(name: str, value: object, expected: object) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(
            f"SAM2 native bundle build does not support {name}={value!r}; "
            f"the only accepted value is {expected!r}"
        )


def _require_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"SAM2 build option {name} must be an integer in [{minimum}, {maximum}]")
    return value


def _require_created_at(value: object) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("SAM2 build option created_at must use canonical YYYY-MM-DDTHH:MM:SSZ UTC")
    month = int(value[5:7])
    day = int(value[8:10])
    hour = int(value[11:13])
    minute = int(value[14:16])
    second = int(value[17:19])
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59 and second <= 60):
        raise ValueError("SAM2 build option created_at must use canonical YYYY-MM-DDTHH:MM:SSZ UTC")
    return value


def _family_options(raw: Mapping[str, object]) -> dict[str, object]:
    unexpected_namespaces = sorted(set(raw) - {_FAMILY_OPTION_NAMESPACE})
    if unexpected_namespaces:
        raise ValueError(
            "SAM2 complete-bundle builds reject unrelated family_build_options "
            f"namespaces: {', '.join(unexpected_namespaces)}"
        )
    options = raw.get(_FAMILY_OPTION_NAMESPACE, {})
    if not isinstance(options, dict):
        raise ValueError("SAM2 family_build_options.sam2 must be an object")
    unexpected = sorted(set(options) - _FAMILY_OPTION_KEYS)
    if unexpected:
        raise ValueError("Unsupported SAM2 family build option(s): " + ", ".join(unexpected))
    return dict(options)


def _validate_request(request: CompleteBundleBuildRequest) -> dict[str, object]:
    if request.config.model_type != "sam2":
        raise ValueError(
            "SAM2 complete-bundle hook received an unrelated model type: "
            f"{request.config.model_type!r}"
        )
    if request.precision is not None:
        if not isinstance(request.precision, str) or request.precision.lower() not in _PRECISIONS:
            raise ValueError("SAM2 native Attention requires precision 'bf16' or 'mixed_bf16_fp32'")

    _reject_nondefault("max_cache_length", request.max_cache_length, None)
    _reject_nondefault("decoder_engine_layout", request.decoder_engine_layout, "split")
    _reject_nondefault("dynamic_kv_cache", request.dynamic_kv_cache, False)
    _reject_nondefault(
        "dynamic_kv_profile_rows_override",
        request.dynamic_kv_profile_rows_override,
        None,
    )
    _reject_nondefault("fp32_layers", request.fp32_layers, ())
    _reject_nondefault("quantize", request.quantize, None)
    _reject_nondefault("quant_scales", request.quant_scales, None)
    _reject_nondefault("quant_calibration_samples", request.quant_calibration_samples, 512)
    _reject_nondefault("verbose", request.verbose, False)
    _reject_nondefault("kernel_artifacts", request.kernel_artifacts, ())
    _reject_nondefault("rtx", request.rtx, False)
    _reject_nondefault("fp8_scales", request.fp8_scales, None)
    _reject_nondefault("save_fp8_scales", request.save_fp8_scales, None)
    _reject_nondefault("triattention_stats_path", request.triattention_stats_path, None)
    _reject_nondefault("triattention_kv_budget", request.triattention_kv_budget, None)
    _reject_nondefault("triattention_divide_length", request.triattention_divide_length, 128)
    _reject_nondefault("triattention_recent_window", request.triattention_recent_window, 128)
    _reject_nondefault(
        "triattention_score_aggregation",
        request.triattention_score_aggregation,
        "mean",
    )
    _reject_nondefault(
        "triattention_count_prompt_tokens",
        request.triattention_count_prompt_tokens,
        True,
    )
    _reject_nondefault(
        "triattention_protect_prefill",
        request.triattention_protect_prefill,
        True,
    )
    _reject_nondefault("triattention_disable_mlr", request.triattention_disable_mlr, False)
    _reject_nondefault("triattention_disable_trig", request.triattention_disable_trig, False)
    _reject_nondefault("diffusion_overrides", request.diffusion_overrides, {})
    _reject_nondefault("build_timing_path", request.build_timing_path, None)
    _reject_nondefault("max_batch_size", request.max_batch_size, 1)
    _reject_nondefault("source_revision", request.source_revision, None)

    parallel = request.parallel_config
    if (
        parallel.mode != "single"
        or parallel.tp_size != 1
        or parallel.cp_size != 1
        or parallel.rank != -1
        or parallel.require_mpirun is not True
    ):
        raise ValueError("SAM2 native bundle build does not support parallel execution")

    # _build_native_impl carries the original local model path in this field
    # for tokenizer packaging. It is a no-op only when it identifies the same
    # directory; a remote/different source or any revision is rejected.
    source = request.source_model_id_or_path
    if source is not None:
        try:
            same_source = Path(source).resolve(strict=True) == request.model_dir.resolve(
                strict=True
            )
        except OSError:
            same_source = False
        if not same_source:
            raise ValueError("SAM2 native bundle build does not support a tokenizer source model")

    return _family_options(request.family_build_options)


def _builder_environment(environ: Mapping[str, str]) -> dict[str, str]:
    environment = {"LANG": "C", "LC_ALL": "C", "PATH": _DEFAULT_PATH}
    for name in _PASSTHROUGH_ENVIRONMENT:
        value = environ.get(name)
        if value:
            environment[name] = value
    return environment


def _builder_argv(
    builder: Path,
    *,
    checkpoint: Path,
    config: Path,
    output: Path,
    options: Mapping[str, object],
) -> list[str]:
    argv = [
        str(builder),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if "workspace_bytes" in options:
        workspace_bytes = _require_int(
            "workspace_bytes",
            options["workspace_bytes"],
            minimum=1,
            maximum=(1 << 64) - 1,
        )
        argv.extend(("--workspace-bytes", str(workspace_bytes)))
    if "gpu_device" in options:
        gpu_device = _require_int(
            "gpu_device",
            options["gpu_device"],
            minimum=0,
            maximum=(1 << 31) - 1,
        )
        argv.extend(("--gpu-device", str(gpu_device)))
    if "created_at" in options:
        argv.extend(("--created-at", _require_created_at(options["created_at"])))
    return argv


class Sam2Plugin:
    """Native Attention SAM2 builder for the qualification-gated runtime."""

    name = "sam2"
    runtime_strategy = "sam2_bbox_video_tracking"
    default_build_precision = "mixed_bf16_fp32"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        normalized = model_type.lower().replace("-", "_").replace(".", "_")
        return normalized in {
            "sam2",
            "sam2_bbox_video_tracking",
            "sam2_video_tracking",
        }

    def load_weights(self, *_args, **_kwargs):
        raise NotImplementedError(
            "SAM2 owns a native complete-bundle build and does not expose Python weights"
        )

    def build_engine(self, *_args, **_kwargs):
        raise NotImplementedError(
            "SAM2 produces six authenticated native plans, not one Python engine"
        )

    def build_complete_bundle(self, request: CompleteBundleBuildRequest) -> None:
        options = _validate_request(request)

        # Re-authenticate immediately before handing paths to the native
        # process. The native reader independently authenticates its owned
        # checkpoint snapshot and source config again before graph building.
        description = require_reference_archive(request.model_dir)
        root = description.root.resolve(strict=True)
        checkpoint = (root / archive_contract.CHECKPOINT_RELATIVE_PATH).resolve(strict=True)
        config = (root / archive_contract.CONFIG_RELATIVE_PATH).resolve(strict=True)
        output = request.output_path.absolute()
        builder = locate_native_builder().resolve(strict=True)
        argv = _builder_argv(
            builder,
            checkpoint=checkpoint,
            config=config,
            output=output,
            options=options,
        )
        environment = _builder_environment(os.environ)
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
        except OSError as exc:
            raise Sam2NativeBuilderError(
                "failed to execute SAM2 native builder with "
                f"argv={argv!r}, environment={environment!r}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise Sam2NativeBuilderError(
                f"SAM2 native builder exited with status {result.returncode}; "
                f"argv={argv!r}; environment={environment!r}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


plugin = Sam2Plugin()


__all__ = ["Sam2Plugin", "plugin"]

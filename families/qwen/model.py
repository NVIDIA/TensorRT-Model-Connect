# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen family plugin — Qwen, Qwen2, Qwen3, QwQ (text-only, not VL).

Unquantized Qwen3 uses only the family-owned TensorRT native KV path. Qualified
FP8 builds retain the family-owned quantized graph; all other unsupported
Qwen3 modes fail closed. Other Qwen variants retain their explicit graph routes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .parallel import normalize_parallel_config
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
)
from .native_kv_contract import validate_native_kv_weights
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


def _quant_format_name(quant_ctx) -> str | None:
    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    return getattr(quant_format, "name", None)


def _uses_qualified_qwen3_fp8_path(
    config: ModelConfig,
    max_cache_length: int,
    *,
    precision: str,
    quant_ctx,
    parallel: ParallelConfig,
    debug_layer_outputs: bool,
) -> bool:
    """Recognize the explicit, previously qualified Qwen3 FP8 graph route."""

    raw = config.raw
    precision_name = str(precision).lower()
    qualified_route = (
        not parallel.enabled
        and precision_name == "fp16"
        and raw.get("_decoder_engine_layout") == "dual_profile"
    ) or (parallel.enabled and parallel.tp_size == 4 and precision_name == "bf16")
    if (
        not native_kv_architecture_capability(config).eligible
        or _quant_format_name(quant_ctx) != "fp8"
        or not qualified_route
        or debug_layer_outputs
        or raw.get("_fp32_layers")
    ):
        return False
    try:
        native_kv_cache_geometry(config, max_cache_length)
    except ValueError:
        return False
    return True


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _QwenModel:
    def default_build_precision(self, config: ModelConfig) -> str:
        capability = native_kv_architecture_capability(config)
        return "bf16" if capability.eligible else "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use the model's complete context for native Qwen3."""
        capability = native_kv_architecture_capability(config)
        return int(config.max_position_embeddings) if capability.eligible else 256

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        config.raw.pop("_native_kv_cache_metadata", None)
        capability = native_kv_build_capability(
            config,
            precision=precision,
            max_cache_length=max_cache_length,
            parallel_enabled=parallel.enabled,
            quantized=quant_ctx is not None,
            debug_layer_outputs=debug_layer_outputs,
        )
        qualified_fp8 = _uses_qualified_qwen3_fp8_path(
            config,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            parallel=parallel,
            debug_layer_outputs=debug_layer_outputs,
        )
        if capability.eligible:
            validate_native_kv_weights(config, weights)
            config.raw["_native_kv_cache_metadata"] = {
                "native_kv_contract_version": 1,
                "native_kv_cache": True,
            }
            role = str(config.raw.get("_decoder_engine_role", ""))
            if role not in ("prefill", "decode"):
                raise ValueError(
                    "native Qwen3 requires explicit split engine role "
                    f"'prefill' or 'decode', got {role!r}"
                )
            return build_dual_profile_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=None,
                verbose=verbose,
                profile_mode=role,
                native_kv_cache=True,
            )

        if capability.applicable and not qualified_fp8:
            raise ValueError(
                f"Qwen3 requires the fixed-capacity native KV path: {capability.reason}"
            )

        if parallel.enabled:
            if debug_layer_outputs:
                raise NotImplementedError(
                    "Qwen tensor-parallel debug layer outputs are not supported"
                )
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def get_bundle_config_overrides(
        self,
        config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that use the native KV runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, model: _QwenModel, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    runtime.update(model.get_bundle_config_overrides(config) or {})
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def _current_cuda_device() -> int:
    from cuda.bindings import runtime as cudart

    status, device = cudart.cudaGetDevice()
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot identify the CUDA device for Qwen build workers: {status}")
    return int(device)


def _stop_build_workers(workers) -> None:
    for process in workers.values():
        if process.poll() is None:
            process.terminate()
    for process in workers.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()


# Start Python in isolated mode before importing from either a checkout or a wheel.
# In particular, a local checkpoint can also be the caller's working directory.
_WORKER_MAIN = """\
import json
import runpy
import sys
sys.path[:0] = json.loads(sys.argv.pop(1))
runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)
"""


def _worker_command() -> list[str]:
    roots = [str(Path(__file__).resolve().parents[2])]
    # A --user install can supply dependencies. Add its known package directory
    # after isolated startup, without executing usercustomize or .pth files.
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE and user_site in sys.path and user_site not in roots:
        roots.append(user_site)
    return [sys.executable, "-I", "-c", _WORKER_MAIN, json.dumps(roots), "families.qwen.build_worker"]


def _build_weightless_native_engines(
    model_dir: Path, config: ModelConfig, max_sequence_length: int, *,
    output_dir: Path, verbose: bool,
) -> tuple[bytes, bytes]:
    """Build the selected Qwen topology's two unchanged roles in isolated workers."""
    device = _current_cuda_device()
    interpreter = _worker_command()
    roles = ("prefill", "decode")
    with tempfile.TemporaryDirectory(prefix="trtmc-qwen-build-", dir=output_dir) as temporary, ExitStack() as resources:
        directory = Path(temporary)
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
        workers = {}
        try:
            for role in roles:
                command = [
                    *interpreter,
                    "--model-dir", str(model_dir.resolve()), "--config", str(config_path),
                    "--role", role, "--max-sequence-length", str(max_sequence_length),
                    "--cuda-device", str(device), "--plan", str(directory / f"{role}.plan"),
                    "--metadata", str(directory / f"{role}.json"), "--watch-parent",
                ]
                if verbose:
                    command.append("--verbose")
                log = resources.enter_context((directory / f"{role}.log").open("wb"))
                workers[role] = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            pending = set(roles)
            while pending:
                for role in roles:
                    if role not in pending:
                        continue
                    status = workers[role].poll()
                    if status is None:
                        continue
                    if status != 0:
                        raise RuntimeError(f"Qwen {role} build worker exited with status {status}")
                    pending.remove(role)
                    # Forward the compiler diagnostics without adding build-mode output.
                    sys.stderr.write((directory / f"{role}.log").read_text(encoding="utf-8", errors="replace"))
                if pending:
                    time.sleep(0.05)
            metadata = []
            for role in roles:
                receipt = json.loads((directory / f"{role}.json").read_text(encoding="utf-8"))
                size = (directory / f"{role}.plan").stat().st_size
                if receipt.get("role") != role or receipt.get("plan_bytes") != size or size <= 0:
                    raise RuntimeError(f"Qwen {role} build worker returned an incomplete plan")
                native = receipt.get("native_kv_cache_metadata")
                if (not isinstance(native, dict) or native.get("native_kv_cache") is not True
                        or type(native.get("native_kv_contract_version")) is not int
                        or native["native_kv_contract_version"] != 1):
                    raise RuntimeError(f"Qwen {role} build worker returned invalid native KV metadata")
                metadata.append(native)
            if metadata[0] != metadata[1]:
                raise RuntimeError("Qwen prefill and decode build workers returned different native KV metadata")
            plans = tuple((directory / f"{role}.plan").read_bytes() for role in roles)
            config.raw["_native_kv_cache_metadata"] = metadata[0]
            return plans
        except BaseException as error:
            _stop_build_workers(workers)
            logs = "\n".join(
                f"--- Qwen {role} worker ---\n" + (directory / f"{role}.log").read_text(
                    encoding="utf-8", errors="replace")
                for role in roles if (directory / f"{role}.log").is_file()
            )
            if not isinstance(error, Exception):
                error.add_note(logs)
                raise
            raise RuntimeError(f"Qwen parallel build failed: {error}\n{logs}") from error
        finally:
            _stop_build_workers(workers)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Qwen bundle through family-owned code."""
    if request.task == "embedding":
        from .embedding import build_embedding

        return build_embedding(request, writer)

    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("qwen does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("qwen does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("qwen does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("qwen does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_continuation":
        raise ValueError("qwen supports only task=text_continuation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    model_type = str(config.model_type).lower()
    unsupported_variant = (
        "vl" in model_type
        or "moe" in model_type
        or "omni" in model_type
        or "image" in model_type
        or model_type in {"qwen3_5", "qwen3.5"}
    )
    if unsupported_variant or not (model_type.startswith("qwen") or model_type.startswith("qwq")):
        raise ValueError(f"Qwen does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Qwen precision must be fp32, fp16, or bf16")
    model = _QwenModel()
    default_length = model.default_max_cache_length(config)
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length, "max_sequence_length"
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Qwen max_sequence_length exceeds checkpoint context capacity")
    quantized = request.quantization == "fp8"
    if request.quantization not in {None, "none", "fp8"}:
        raise NotImplementedError(f"Qwen does not support quantization={request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("Qwen does not expose mixed-precision layer selection")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("Qwen does not support MTP speculative decoding")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    if quantized:
        qualified_route = model_type == "qwen3" and (
            (not parallel.enabled and precision == "fp16")
            or (parallel.tp_size == 4 and precision == "bf16")
        )
        if not qualified_route:
            raise ValueError("Qwen FP8 supports only Qwen3 FP16 single-device or BF16 TP4 builds")
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    config.raw["_quantized_build_requested"] = quantized
    use_weightless_workers = (
        request.backend == "trt"
        and model_type == "qwen3"
        and precision == "bf16"
        and (config.hidden_size, config.intermediate_size, config.num_hidden_layers,
             config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.vocab_size)
        == (1024, 3072, 28, 16, 8, 128, 151936)
        and not parallel.enabled
        and not quantized
        and request.graph_transform is None
        and native_kv_build_capability(
            config, precision=precision, max_cache_length=max_sequence_length,
            parallel_enabled=False, quantized=False, debug_layer_outputs=False,
        ).eligible
    )
    quant_ctx = None
    if quantized:
        from . import graph_ops
        from .quantization import calibrate_qwen_fp8

        config.raw["_decoder_engine_layout"] = "dual_profile"
        quant_ctx = calibrate_qwen_fp8(model_dir, config, graph_ops)
    weights = None if use_weightless_workers else model.load_weights(str(model_dir), config, precision=precision)
    writer.set_header(family="qwen", task=request.task, backend=request.backend)
    if use_weightless_workers:
        prefill, decode = _build_weightless_native_engines(
            model_dir, config, max_sequence_length,
            output_dir=Path(request.output_path).parent, verbose=bool(request.verbose))
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"
    elif quantized and parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    elif quantized:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
        layout = "dual_profile"
    elif parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

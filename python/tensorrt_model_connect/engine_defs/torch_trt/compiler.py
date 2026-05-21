"""Core compilation: torch.export + torch_tensorrt raw TRT engine.

This module orchestrates the full build pipeline:
  1. Load HF model via family plugin
  2. Select build strategy based on plugin's runtime_strategy
  3. Wrap model with strategy-specific I/O adapter
  4. Export with torch.export (strict=False)
  5. Convert to raw TRT engine via torch_tensorrt
  6. Package into a .trtfb bundle (compatible with C++ runtime)

Strategies handle different model architectures:
  - "decoder": CausalLM with StatelessCacheWrapper (KV cache I/O)
  - "encoder_only": Encoder-only models (BERT) with EncoderOnlyWrapper

No LibTorch dependency at runtime — the engine is a pure TRT .plan file.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import logging
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress known harmless third-party warnings emitted during torch_tensorrt import:
#
# 1. "TRTLLM_PLUGIN_PATH is not set" — torch_tensorrt._utils logs this when
#    USE_TRTLLM_PLUGINS env isn't set. We don't use TRT-LLM plugins.
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
#
# 2. "transformers version X is not tested with nvidia-modelopt" — modelopt
#    hasn't updated its version check for transformers 5.x yet.
warnings.filterwarnings("ignore", message="transformers version.*nvidia-modelopt")
#
# 3. "The logger passed into createInferBuilder differs from one already
#    registered" — TRT allows only one global ILogger. torch_tensorrt's
#    _TRTLogger is registered on first Builder creation; the tensorrt.plugin
#    module (loaded during import) registers its own logger first, so all
#    subsequent Builder() calls emit this warning. Harmless — TRT correctly
#    uses the first logger. This is an upstream torch_tensorrt issue.
logging.getLogger("torch_tensorrt [TensorRT Conversion Context]").setLevel(
    logging.ERROR
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .config import ModelConfig  # noqa: E402
from ... import trt_compat  # noqa: E402

PRECISION_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def precision_to_dtype(precision: str) -> torch.dtype:
    """Convert a precision string to a torch dtype."""
    if precision not in PRECISION_DTYPE_MAP:
        valid = ", ".join(sorted(PRECISION_DTYPE_MAP))
        raise ValueError(f"Unknown precision {precision!r}. Valid: {valid}")
    return PRECISION_DTYPE_MAP[precision]
from .families import find_plugin, ALL_PLUGINS  # noqa: E402
from .bundle_writer import TtrtBundleInfo, BundleSection, write_bundle  # noqa: E402
from .strategies import get_strategy  # noqa: E402

# Backward-compat aliases — tests and external code may import these from compiler.
from .strategies.decoder import StatelessCacheWrapper, patch_static_cache_scatter  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Strategy normalization: torch-trt internal strategies -> standard C++ runtime
# strategies so the resulting bundle is indistinguishable at runtime.
# ---------------------------------------------------------------------------

_NORMALIZE_STRATEGY: dict[str, str] = {
    "torchtrt_decoder": "decoder_kv_cache",
    "torchtrt_encoder": "encoder_only",
    "torchtrt_diffusion": "diffusion_pixart_torchtrt",
    "decoder": "decoder_kv_cache",
    "encoder_only": "encoder_only",
    "diffusion": "diffusion_pixart",
}


def _normalize_runtime_strategy(raw_strategy: str) -> str:
    """Map torch-trt internal strategy names to standard C++ runtime names."""
    return _NORMALIZE_STRATEGY.get(raw_strategy, raw_strategy)


# ---------------------------------------------------------------------------
# Engine compilation
# ---------------------------------------------------------------------------

def _get_torch_version() -> str:
    return torch.__version__


def _get_torchtrt_version() -> str:
    try:
        import torch_tensorrt
        return torch_tensorrt.__version__
    except (ImportError, OSError):
        return "not installed"


def _get_trt_version() -> str:
    return trt_compat.tensorrt_version()


def _trt_abi_from_version(version: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", version or "")
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def _get_gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def _detect_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect whether the HF tokenizer adds special tokens by default."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        ids_default = tok.encode("hello")
        ids_without = tok.encode("hello", add_special_tokens=False)
        return ids_default != ids_without
    except Exception:
        pass

    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            tok_cfg = json.load(open(tok_config_path))
            if bool(tok_cfg.get("add_bos_token", False)):
                return True
            if bool(tok_cfg.get("add_eos_token", False)):
                return True
        except Exception:
            pass
    return False


def _detect_diffusion_tokenizer_add_special_tokens(model_dir: Path) -> bool:
    """Detect add-special behavior from the tokenizer embedded in diffusion bundles."""
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tok_dir = model_dir / tok_subdir
        if tok_dir.is_dir():
            return _detect_tokenizer_add_special_tokens(tok_dir)
    return _detect_tokenizer_add_special_tokens(model_dir)


# ---------------------------------------------------------------------------
# IO map: declares tensor names for torch-trt decoder bundles so the C++
# runtime knows which engine tensors correspond to which semantic roles.
# ---------------------------------------------------------------------------

def _decoder_io_map(num_layers: int) -> dict:
    """Build the io_map for a decoder KV-cache model.

    The torch-trt path produces engines where cache/present tensors follow
    a numbered pattern (cache_kv_0..cache_kv_{2L-1} for inputs,
    output0..output{2L} for outputs). This map lets the C++ runtime resolve
    them without hard-coding names.
    """
    return {
        "token_id": "token_id",
        "position_id": "position_id",
        "attention_mask": "attention_mask",
        "logits": "output0",
        "cache_k": "cache_kv_{2i}",
        "cache_v": "cache_kv_{2i+1}",
        "present_k": "output{2i+1}",
        "present_v": "output{2i+2}",
        "num_layers": num_layers,
    }


def compile_model(
    wrapper: nn.Module,
    example_args: tuple,
    *,
    trt_inputs: tuple | None = None,
    verbose: bool = False,
    workspace_size: int = 1 << 30,
) -> bytes:
    """Export a wrapped model via torch.export and convert to raw TRT engine.

    Args:
        wrapper: Wrapped model (StatelessCacheWrapper, EncoderOnlyWrapper, etc.)
                 in eval mode. Can be on CPU for CPU-side export.
        example_args: Tuple of example tensors matching forward() signature.
                      Used for torch.export tracing. Must be on same device as
                      the wrapper (CPU or CUDA).
        trt_inputs: Optional separate inputs for torch_tensorrt conversion.
                    When provided, these are used as the ``inputs`` argument to
                    convert_exported_program_to_serialized_trt_engine() instead
                    of example_args. This enables CPU-side export: trace on CPU
                    with example_args, then convert with CUDA trt_inputs so the
                    TRT engine targets GPU. If None, example_args is used for
                    both steps.
        verbose: Enable detailed logging.
        workspace_size: TRT workspace size in bytes (default 1GB).

    Returns:
        Raw TRT engine bytes (.plan format).
    """
    import torch_tensorrt

    # 1. Export to ExportedProgram
    if verbose:
        print("[torch-trt] Running torch.export ...", file=sys.stderr)

    t0 = time.monotonic()
    with torch.no_grad(), _math_sdpa_only():
        exported = torch.export.export(
            wrapper,
            args=example_args,
            strict=False,
        )
    t1 = time.monotonic()

    if verbose:
        inputs_nodes = [n for n in exported.graph.nodes if n.op == 'placeholder']
        user_inputs = [n for n in inputs_nodes if not n.name.startswith('p_')]
        weight_params = [n for n in inputs_nodes if n.name.startswith('p_')]
        print(f"[torch-trt] torch.export complete [{t1-t0:.1f}s] "
              f"({len(user_inputs)} user inputs, {len(weight_params)} weights)",
              file=sys.stderr)

    # 2. Convert to raw TRT engine
    if verbose:
        print("[torch-trt] Converting to raw TRT engine ...", file=sys.stderr)

    conversion_inputs = list(trt_inputs) if trt_inputs is not None else list(example_args)

    t2 = time.monotonic()
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        inputs=conversion_inputs,
        use_explicit_typing=True,
        disable_tf32=True,
        workspace_size=workspace_size,
        min_block_size=1,
        truncate_double=True,
    )
    t3 = time.monotonic()

    # Free the ExportedProgram to release GPU memory
    del exported

    if verbose:
        print(f"[torch-trt] Raw TRT engine: {len(engine_bytes)/(1024*1024):.1f} MB "
              f"[{t3-t2:.1f}s]", file=sys.stderr)

    return engine_bytes


@contextlib.contextmanager
def _math_sdpa_only():
    """Force PyTorch export paths to use math SDPA kernels.

    Torch-TRT cannot currently lower several fused SDPA variants selected by
    PyTorch 2.11 on GB300, including cuDNN attention and CPU flash attention.
    Constraining export to the math backend keeps the traced graph in terms of
    TRT-lowerable matmul/softmax ops.
    """
    attention_mod = getattr(torch.nn, "attention", None)
    if attention_mod is not None:
        sdpa_kernel = getattr(attention_mod, "sdpa_kernel", None)
        sdp_backend = getattr(attention_mod, "SDPBackend", None)
        math_backend = getattr(sdp_backend, "MATH", None) if sdp_backend else None
        if sdpa_kernel is not None and math_backend is not None:
            with sdpa_kernel([math_backend]):
                yield
            return

    yield


def _build_engine_from_onnx_bytes(
    onnx_bytes: bytes,
    *,
    verbose: bool = False,
    workspace_size: int = 1 << 30,
) -> bytes:
    """Build a TRT engine from serialized ONNX bytes with TF32 disabled."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            explicit_batch=True,
            strongly_typed=True,
        )
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if hasattr(config, "clear_flag") and hasattr(trt, "BuilderFlag"):
        config.clear_flag(trt.BuilderFlag.TF32)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build from ONNX failed")
    return bytes(plan)


def compile_model_via_onnx(
    wrapper: nn.Module,
    example_args: tuple,
    *,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 18,
    verbose: bool = False,
) -> bytes:
    """Export a wrapped model to ONNX, then build a TRT engine via ONNX parser."""
    if verbose:
        print("[onnx-trt] Exporting to ONNX ...", file=sys.stderr)

    onnx_buffer = io.BytesIO()
    with torch.no_grad(), _math_sdpa_only():
        torch.onnx.export(
            wrapper,
            example_args,
            onnx_buffer,
            opset_version=opset_version,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
        )
    onnx_bytes = onnx_buffer.getvalue()
    if verbose:
        print(f"[onnx-trt] ONNX export complete "
              f"({len(onnx_bytes) / (1024 * 1024):.1f} MB)", file=sys.stderr)

    return _build_engine_from_onnx_bytes(onnx_bytes, verbose=verbose)


def _inspect_engine(engine_bytes: bytes) -> dict:
    """Inspect a TRT engine and return I/O tensor name mapping."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(engine_bytes)

    io_map = {"inputs": {}, "outputs": {}}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype = str(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name)
        entry = {"shape": shape, "dtype": dtype}
        if mode == trt.TensorIOMode.INPUT:
            io_map["inputs"][name] = entry
        else:
            io_map["outputs"][name] = entry

    return io_map


def _parse_model_config(model_dir_path: Path) -> ModelConfig:
    """Parse model config, falling back to model_index.json for diffusers models."""
    config_path = model_dir_path / "config.json"
    if config_path.exists():
        return ModelConfig.from_dir(model_dir_path)

    # Diffusers-format models use model_index.json at the top level
    index_path = model_dir_path / "model_index.json"
    if index_path.exists():
        raw = json.loads(index_path.read_text())
        # Use pipeline class name as model_type (e.g. "PixArtSigmaPipeline")
        model_type = raw.get("_class_name", "")
        return ModelConfig(model_type=model_type, raw=raw)

    raise FileNotFoundError(
        f"No config.json or model_index.json found in {model_dir_path}")


def _build_multi_engine_bundle(
    model_dir_path: Path,
    plugin,
    config: ModelConfig,
    strategy,
    output_path: str,
    *,
    precision: str = "fp16",
    verbose: bool = False,
    t0: float = 0.0,
) -> None:
    """Build a multi-engine bundle (diffusion models).

    The family plugin's build_components() loads each component, wraps it,
    and calls compile_model() for each. Results are packaged into a single
    .trtfb bundle with multiple engine plan sections.
    """
    print(f"[torch-trt] Multi-engine build (precision={precision}) ...",
          file=sys.stderr)

    result = plugin.build_components(
        str(model_dir_path), config, compile_model,
        precision=precision, verbose=verbose,
    )

    component_sections = result["sections"]
    raw_runtime_strategy = result["runtime_strategy"]
    normalized_strategy = _normalize_runtime_strategy(raw_runtime_strategy)
    diffusion_config = result.get("diffusion_config", {})
    trt_version = _get_trt_version()
    trt_abi = _trt_abi_from_version(trt_version)
    tokenizer_add_special_tokens = _detect_diffusion_tokenizer_add_special_tokens(
        model_dir_path)

    # Build config.json for the bundle (using normalized strategy)
    bundle_config = {
        "runtime_strategy": normalized_strategy,
        "engine_backend": "trt",
        "build_backend": "torch_trt",
        "trt_version": trt_version,
        "tokenizer_add_special_tokens": int(tokenizer_add_special_tokens),
        **diffusion_config,
    }
    if trt_abi:
        bundle_config["trt_abi"] = trt_abi
    config_data = json.dumps(bundle_config, indent=2).encode("utf-8")

    # Assemble all sections: engine plans + config
    sections = list(component_sections)
    sections.append(BundleSection("config.json", config_data))

    # Embed tokenizer files (T5 tokenizer for diffusion models)
    tokenizer_dir = model_dir_path / "tokenizer"
    tokenizer_search_dirs = [tokenizer_dir, model_dir_path]
    for search_dir in tokenizer_search_dirs:
        if not search_dir.exists():
            continue
        for filename in ("tokenizer.json", "tokenizer_config.json",
                         "spiece.model", "special_tokens_map.json"):
            file_path = search_dir / filename
            if file_path.exists():
                sections.append(BundleSection(filename, file_path.read_bytes()))

    info = TtrtBundleInfo(
        model_id=model_dir_path.name,
        model_type=config.model_type,
        family=plugin.name,
        torch_version=_get_torch_version(),
        torchtrt_version=_get_torchtrt_version(),
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=_get_gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        precision=precision,
        runtime_strategy=normalized_strategy,
        tokenizer_add_special_tokens=tokenizer_add_special_tokens,
        build_backend="torch_trt",
    )

    write_bundle(output_path, info, sections)
    t_end = time.monotonic()
    total_engine_mb = sum(len(s.data) for s in component_sections) / (1024 * 1024)
    print(f"[torch-trt] Bundle saved: {output_path} "
          f"({len(component_sections)} engines, {total_engine_mb:.1f} MB total) "
          f"[{t_end - t0:.1f}s]", file=sys.stderr)


def build_bundle(
    model_dir: str,
    output_path: str,
    max_cache_length: int = 256,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> None:
    """Full pipeline: load HF model -> compile to raw TRT engine -> write .trtfb bundle.

    The build strategy is selected based on the family plugin's runtime_strategy
    attribute (defaults to "decoder" if absent). Each strategy handles model
    wrapping, export arg construction, and pre-export setup.

    Supports two build paths:
      - Single-engine (decoder, encoder-only): standard wrap -> export -> compile
      - Multi-engine (diffusion): plugin.build_components() compiles each
        component separately and returns multiple engine sections

    Args:
        model_dir: Path to HF model directory with config.json + safetensors,
                   or diffusers-format directory with model_index.json.
        output_path: Where to write the .trtfb bundle.
        max_cache_length: KV cache / max sequence length.
        precision: Compute precision (model loaded in this precision).
        verbose: Print detailed logs.
    """
    model_dir_path = Path(model_dir)
    compute_dtype = precision_to_dtype(precision)
    t0 = time.monotonic()
    print(
        f"[torch-trt] Builder TensorRT resolved: {trt_compat.resolved_summary()}",
        file=sys.stderr,
    )

    # 1. Parse config (supports both config.json and model_index.json)
    config = _parse_model_config(model_dir_path)
    print(f"[torch-trt] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size})", file=sys.stderr)

    # 2. Find family plugin
    plugin = find_plugin(config)
    if plugin is None:
        supported = ", ".join(p.name for p in ALL_PLUGINS)
        raise ValueError(
            f"No Torch-TRT family plugin for model_type={config.model_type!r}. "
            f"Supported: {supported}")

    print(f"[torch-trt] Family: {plugin.name}", file=sys.stderr)

    # 3. Select build strategy from plugin (defaults to "decoder")
    strategy_name = getattr(plugin, 'runtime_strategy', 'decoder')
    strategy = get_strategy(strategy_name)
    print(f"[torch-trt] Strategy: {strategy.name} "
          f"(runtime_strategy={strategy.runtime_strategy})", file=sys.stderr)

    # Multi-engine path: diffusion and other multi-component models.
    # The family plugin's build_components() handles loading, wrapping,
    # and compiling each component, calling compile_model() for each.
    if hasattr(plugin, 'build_components'):
        _build_multi_engine_bundle(
            model_dir_path, plugin, config, strategy, output_path,
            precision=precision, verbose=verbose, t0=t0)
        return

    # Single-engine path: standard decoder, encoder-only, etc.
    model = None
    wrapper = None
    try:
        # 4. Load HF model in the requested precision
        t1 = time.monotonic()
        print(f"[torch-trt] Loading model (dtype={compute_dtype}) ...",
              file=sys.stderr)
        model = plugin.load_model(
            str(model_dir_path), config, max_cache_length,
            dtype=compute_dtype)
        t2 = time.monotonic()
        print(f"[torch-trt] Model loaded [{t2 - t1:.1f}s]", file=sys.stderr)

        # 5. Pre-export setup (e.g. patch StaticCache for decoder strategy)
        strategy.pre_export_setup()

        # 6. Wrap model with strategy-specific I/O adapter
        hf_config = model.config  # HF PretrainedConfig
        wrapper = strategy.wrap_model(
            model, hf_config, max_cache_length,
            compute_dtype=compute_dtype)
        wrapper.eval()

        # 7. Build example inputs for torch.export
        export_args = strategy.make_export_args(
            hf_config, max_cache_length, precision=precision)

        # 8. Compile to raw TRT engine
        print(f"[torch-trt] Compiling (precision={precision}, "
              f"cache={max_cache_length}) ...", file=sys.stderr)
        if strategy.name == "timesfm":
            engine_bytes = compile_model_via_onnx(
                wrapper,
                export_args,
                input_names=["past_values", "past_values_padding", "freq"],
                output_names=["mean_predictions", "full_predictions"],
                verbose=verbose,
            )
        else:
            engine_bytes = compile_model(
                wrapper, export_args,
                verbose=verbose,
            )
        t3 = time.monotonic()
        print(f"[torch-trt] Compiled [{t3 - t2:.1f}s] "
              f"({len(engine_bytes) / (1024 * 1024):.1f} MB)", file=sys.stderr)

        # 9. Inspect engine I/O for bundle metadata
        io_map = _inspect_engine(engine_bytes)
        if verbose:
            print(f"[torch-trt] Engine I/O: {len(io_map['inputs'])} inputs, "
                  f"{len(io_map['outputs'])} outputs", file=sys.stderr)

        # 10. Detect tokenizer behavior
        tokenizer_add_special_tokens = _detect_tokenizer_add_special_tokens(
            model_dir_path)

        # 11. Write .trtfb bundle (normalize strategy to standard C++ name)
        normalized_strategy = _normalize_runtime_strategy(strategy.runtime_strategy)
        trt_version = _get_trt_version()
        trt_abi = _trt_abi_from_version(trt_version)
        info = TtrtBundleInfo(
            model_id=model_dir_path.name,
            model_type=config.model_type,
            family=plugin.name,
            torch_version=_get_torch_version(),
            torchtrt_version=_get_torchtrt_version(),
            trt_version=trt_version,
            trt_abi=trt_abi,
            gpu_name=_get_gpu_name(),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_cache_length=max_cache_length,
            precision=precision,
            runtime_strategy=normalized_strategy,
            tokenizer_add_special_tokens=tokenizer_add_special_tokens,
            build_backend="torch_trt",
            io_map=_decoder_io_map(config.num_hidden_layers)
                if normalized_strategy == "decoder_kv_cache" else None,
        )

        # Use engine_plan as section name (C++ bundle reader looks for this)
        sections = [BundleSection("engine_plan", engine_bytes)]

        # Embed config + tokenizer files
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt", "special_tokens_map.json",
                         "tokenizer.model"):
            file_path = model_dir_path / filename
            if file_path.exists():
                data = file_path.read_bytes()
                if filename == "config.json":
                    cfg_dict = json.loads(data)
                    cfg_dict["runtime_strategy"] = normalized_strategy
                    cfg_dict["engine_backend"] = "trt"
                    cfg_dict["build_backend"] = "torch_trt"
                    cfg_dict["trt_version"] = trt_version
                    if trt_abi:
                        cfg_dict["trt_abi"] = trt_abi
                    if normalized_strategy == "decoder_kv_cache":
                        cfg_dict["io_map"] = _decoder_io_map(config.num_hidden_layers)
                    cfg_dict["torchtrt_io_map"] = io_map
                    data = json.dumps(cfg_dict, indent=2).encode("utf-8")
                sections.append(BundleSection(filename, data))

        write_bundle(output_path, info, sections)
        t4 = time.monotonic()
        print(f"[torch-trt] Bundle saved: {output_path} [{t4 - t0:.1f}s total]",
              file=sys.stderr)

    finally:
        # Explicit GPU memory cleanup — prevents OOM when building multiple
        # bundles in the same process (e.g. multi-agent or batch builds).
        del wrapper
        del model
        gc.collect()
        torch.cuda.empty_cache()

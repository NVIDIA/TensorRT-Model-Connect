"""Torch-TRT build backend -- optional performance optimization.

Compiles HuggingFace models into .trtfb bundles via torch.export +
torch_tensorrt. Produces standard bundles with standard runtime_strategy
values (decoder_kv_cache, diffusion_pixart, etc.).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _is_diffusers_repo(repo_id: str) -> bool:
    """Check if a HuggingFace repo is a diffusers-format model."""
    try:
        from huggingface_hub import repo_info
    except ImportError:
        return False
    try:
        info = repo_info(repo_id)
        siblings = [s.rfilename for s in (info.siblings or [])]
        return "model_index.json" in siblings
    except Exception:
        return False


# Download patterns for standard transformer models
_TRANSFORMER_PATTERNS = [
    "config.json", "generation_config.json",
    "model.safetensors", "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json",
    "*.model",
]

# Download patterns for diffusers-format models (multi-component)
_DIFFUSERS_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "transformer/**",
    "vae/**",
    "tokenizer/**",
    "tokenizer_2/**",
]


def _resolve_model(model_id_or_path: str) -> str:
    """Resolve a HuggingFace repo ID or local path to a local directory."""
    local = Path(model_id_or_path)
    if local.is_dir() and (
        (local / "config.json").exists()
        or (local / "model_index.json").exists()
    ):
        return str(local)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    is_diffusers = _is_diffusers_repo(model_id_or_path)
    patterns = _DIFFUSERS_PATTERNS if is_diffusers else _TRANSFORMER_PATTERNS
    fmt = "diffusers" if is_diffusers else "transformers"

    print(f"[torch-trt] Downloading {model_id_or_path} ({fmt} format) ...",
          file=sys.stderr)
    local_dir = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=patterns,
    )
    print(f"[torch-trt] Downloaded to {local_dir}", file=sys.stderr)
    return local_dir


class TorchTrtBackend:
    """BuildBackend implementation for torch-trt."""

    name = "torch_trt"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import torch_tensorrt  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    def build(
        self,
        model_dir: str,
        output_path: str,
        max_cache_length: int = 256,
        *,
        precision: str = "fp16",
        verbose: bool = False,
        parallel_config=None,
    ) -> None:
        from .compiler import build_bundle

        resolved_dir = _resolve_model(model_dir)
        t0 = time.monotonic()
        build_bundle(
            resolved_dir, output_path, max_cache_length,
            precision=precision, verbose=verbose,
            parallel_config=parallel_config,
        )
        elapsed = time.monotonic() - t0
        print(f"[torch-trt] Done [{elapsed:.1f}s total]", file=sys.stderr)


# Module-level attribute for auto-discovery by engine_defs registry
backend = TorchTrtBackend()

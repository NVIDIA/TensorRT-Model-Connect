"""Configuration helpers for the GitHub ELF model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model_config import ModelConfig


ELF_VARIANTS: dict[str, tuple[int, int, int]] = {
    "ELF-B": (12, 768, 12),
    "ELF-M": (24, 1056, 16),
    "ELF-L": (32, 1280, 16),
}


def resolve_elf_config(config: "ModelConfig", max_seq_length: int | None = None) -> dict:
    """Resolve ELF architecture fields from config.json plus upstream defaults."""
    raw = config.raw or {}
    variant = str(raw.get("elf_variant") or raw.get("model") or "ELF-B")
    variant_defaults = ELF_VARIANTS.get(variant.upper(), ELF_VARIANTS["ELF-B"])
    default_depth, default_hidden, default_heads = variant_defaults

    text_encoder_dim = int(
        raw.get("text_encoder_dim")
        or raw.get("encoder_d_model")
        or raw.get("d_model")
        or raw.get("text_hidden_size", 0)
        or 512
    )
    hidden_size = int(
        config.hidden_size
        or raw.get("hidden_size")
        or raw.get("elf_hidden_size")
        or default_hidden
    )
    depth = int(
        config.num_hidden_layers
        or raw.get("depth")
        or raw.get("num_hidden_layers")
        or default_depth
    )
    num_heads = int(
        config.num_attention_heads
        or raw.get("num_heads")
        or raw.get("num_attention_heads")
        or default_heads
    )
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"ELF hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

    max_length = int(
        raw.get("_elf_engine_max_length")
        or max_seq_length
        or raw.get("max_length")
        or raw.get("max_position_embeddings")
        or config.max_position_embeddings
        or 128
    )
    max_input_length = int(raw.get("max_input_length") or raw.get("elf_max_input_length") or 0)
    self_cond_prob = float(raw.get("self_cond_prob", 0.5))
    input_dim = int(raw.get("elf_input_dim") or (
        2 * text_encoder_dim if self_cond_prob > 0.0 else text_encoder_dim))

    num_time_tokens = int(raw.get("num_time_tokens", 4))
    if num_time_tokens <= 0:
        raise ValueError("ELF num_time_tokens must be positive")

    return {
        "variant": variant,
        "text_encoder_dim": text_encoder_dim,
        "input_dim": input_dim,
        "max_length": max_length,
        "max_input_length": max_input_length,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "head_dim": hidden_size // num_heads,
        "mlp_ratio": float(raw.get("mlp_ratio", raw.get("elf_mlp_ratio", 4.0))),
        "bottleneck_dim": int(raw.get("bottleneck_dim", 128)),
        "num_time_tokens": num_time_tokens,
        "num_self_cond_cfg_tokens": int(raw.get("num_self_cond_cfg_tokens", 4)),
        "num_model_mode_tokens": int(raw.get("num_model_mode_tokens", 4)),
        "vocab_size": int(config.vocab_size or raw.get("vocab_size", 0)),
        "rope_theta": float(raw.get("rope_theta", 10000.0)),
        "rms_norm_eps": float(raw.get("rms_norm_eps", 1e-6)),
        "self_cond_prob": self_cond_prob,
        "denoiser_noise_scale": float(raw.get("denoiser_noise_scale", 1.0)),
        "denoiser_p_mean": float(raw.get("denoiser_p_mean", -1.5)),
        "denoiser_p_std": float(raw.get("denoiser_p_std", 0.8)),
        "t_eps": float(raw.get("t_eps", 5e-2)),
    }


def make_elf_rope_cache(
    *,
    max_length: int,
    head_dim: int,
    prefix_tokens: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ELF TextRotaryEmbeddingFast caches as [1, S, head_dim / 2]."""
    head_dim = int(head_dim)
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError(f"ELF RoPE head_dim must be an even value >= 2; got {head_dim}")
    half = head_dim // 2
    total_seq = int(prefix_tokens) + int(max_length)
    cos = np.ones((total_seq, half), dtype=np.float32)
    sin = np.zeros((total_seq, half), dtype=np.float32)
    if max_length <= 0:
        return cos.reshape(1, total_seq, half), sin.reshape(1, total_seq, half)

    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_length, dtype=np.float32)
    angles = positions[:, None] * freqs[None, :]
    cos[prefix_tokens:, :] = np.cos(angles).astype(np.float32)
    sin[prefix_tokens:, :] = np.sin(angles).astype(np.float32)
    return cos.reshape(1, total_seq, half), sin.reshape(1, total_seq, half)

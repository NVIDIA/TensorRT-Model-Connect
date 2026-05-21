"""Export argument construction for StatelessCacheWrapper.

Builds example input tensors for torch.export matching the raw TRT C++ runtime
I/O format (DeviceKvCache):
  - token_id:       int32 [1]
  - position_id:    int32 [1]
  - attention_mask:  float32 [1, max_cache_length + 1]
  - cache_kv_0..N:  float32 [max_cache_length, kv_dim]
                    where kv_dim = num_key_value_heads * head_dim

Key exports:
    make_export_args() — builds the example_args tuple for torch.export
    make_cache_tensors() — creates zeroed KV cache tensors
    build_attention_mask() — builds a 1D attention mask for a given step
"""

from __future__ import annotations

import torch


def build_attention_mask(
    step: int,
    max_cache_length: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Build 1D attention mask: [1, max_cache_length + 1] float32.

    Matches the C++ DeviceKvCache mask format:
      0.0 for valid positions (0..step), -1e4 for masked positions.
    The extra position (max_cache_length) is for the current token.
    """
    attention_window = max_cache_length + 1
    mask = torch.full((1, attention_window), -1.0e4, dtype=torch.float32, device=device)
    mask[0, :step + 1] = 0.0
    return mask


def make_cache_tensors(
    config,
    max_cache_length: int,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Create zeroed KV cache tensors in raw TRT format.

    Returns a flat list: [cache_k_0, cache_v_0, cache_k_1, cache_v_1, ...]
    Each tensor has shape [max_cache_length, kv_dim] float32, where
    kv_dim = num_key_value_heads * head_dim.

    Args:
        config: HF PretrainedConfig or ModelConfig with num_hidden_layers,
                num_attention_heads, and head_dim.
    """
    num_layers = config.num_hidden_layers
    num_heads = getattr(config, 'num_attention_heads', 1)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = getattr(config, 'head_dim',
                       getattr(config, 'hidden_size', 64) //
                       max(num_heads, 1))
    kv_dim = num_kv_heads * head_dim

    cache_shape = (max_cache_length, kv_dim)
    tensors = []
    for _ in range(num_layers):
        tensors.append(torch.zeros(cache_shape, dtype=torch.float32, device=device))  # k
        tensors.append(torch.zeros(cache_shape, dtype=torch.float32, device=device))  # v
    return tensors


def make_export_args(
    config,
    max_cache_length: int,
    seq_len: int = 1,
    precision: str = "fp16",
    device: str = "cuda",
) -> tuple:
    """Build example input tuple for torch.export with StatelessCacheWrapper.

    Returns a tuple of tensors matching the wrapper's forward signature:
      (token_id, position_id, attention_mask, cache_kv_0, cache_kv_1, ...)

    All tensors use raw TRT format:
      - token_id:      int32 [1]
      - position_id:   int32 [1]
      - attention_mask: float32 [1, max_cache_length + 1]
      - cache_kv_N:    float32 [max_cache_length, kv_dim]

    Args:
        config: HF PretrainedConfig or ModelConfig.
        max_cache_length: Maximum KV cache length.
        seq_len: Sequence length for example input (1 for decode step).
        precision: Not used for export args (I/O is always float32 for C++ runtime
            compatibility). The model's internal compute dtype is set separately.
        device: Device for tensors.
    """
    # Core inputs — int32 to match C++ runtime
    token_id = torch.zeros(1, dtype=torch.int32, device=device)
    position_id = torch.zeros(1, dtype=torch.int32, device=device)
    attention_mask = build_attention_mask(0, max_cache_length, device=device)

    # Cache tensors — float32, compact GQA/MQA KV heads.
    cache_tensors = make_cache_tensors(config, max_cache_length, device=device)

    return (token_id, position_id, attention_mask, *cache_tensors)

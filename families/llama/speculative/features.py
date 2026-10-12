# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EAGLE3 target feature selection, expressed as decoder layer inputs."""


def target_feature_indices(num_layers: int, draft_config: dict) -> tuple[int, ...]:
    """Select the three feature blocks in the draft projection's training order.

    The original EAGLE3 Llama target captures inputs to layers 2, L//2,
    and L-3. In a full Hugging Face hidden-state tuple (L+1 entries),
    the final tap is hidden_states[-4], not layer L-4.
    """
    indices = draft_config.get("eagle_aux_hidden_state_layer_ids")
    if indices is None:
        indices = (2, num_layers // 2, num_layers - 3)
    if (not isinstance(indices, (list, tuple)) or len(indices) != 3
            or any(type(index) is not int or not 0 <= index < num_layers for index in indices)
            or len(set(indices)) != 3):
        raise ValueError("EAGLE3 requires three distinct target layer input indices in range")
    return tuple(indices)

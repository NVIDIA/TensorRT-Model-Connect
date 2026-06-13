"""Cosmos 3 action embedding head.

The action embedding head projects per-embodiment action trajectories (9D-57D
depending on the robot/agent embodiment) into the DM generator's hidden space
(5120-dim for Super, 2048-dim for Nano). From ``transformer/config.json``:

  - ``action_dim``: 64 (the *padded* per-step action dimension)
  - ``max_action_dim``: 64
  - ``num_embodiment_domains``: 32 (number of distinct embodiments supported,
    each potentially having a different native action dim)

The embedding is a linear projection from the padded 64-D action vector into
the backbone hidden space, plus a learned embodiment-domain embedding added
to bias the projection per embodiment.

This module is currently a stand-alone definition. The actual TRT graph
construction lives inside the DM generator builder (Phase 4), since action
embeddings are mixed into the DM token stream alongside latent patches.
"""

from __future__ import annotations

# Cosmos3-Super action encoder constants (from transformer/config.json).
COSMOS3_SUPER_ACTION_DIM = 64
COSMOS3_SUPER_MAX_ACTION_DIM = 64
COSMOS3_SUPER_NUM_EMBODIMENT_DOMAINS = 32

# Documented embodiment range per the HF model card:
# action trajectories of 16-400 frames, with embodiment-specific dimensions
# in [9, 57]. The 64-D padded action vector accommodates the largest
# embodiment plus headroom.
COSMOS3_ACTION_FRAMES_MIN = 16
COSMOS3_ACTION_FRAMES_MAX = 400
COSMOS3_NATIVE_ACTION_DIM_MIN = 9
COSMOS3_NATIVE_ACTION_DIM_MAX = 57


def cosmos3_action_embed_shape(hidden_size: int) -> tuple:
    """Return the (action_proj, domain_embed) weight shapes for Cosmos 3.

    Args:
      hidden_size: backbone hidden size (5120 for Super, 2048 for Nano).

    Returns:
      Tuple of ``(action_proj_shape, domain_embed_shape)``:
        - ``action_proj_shape``: ``(64, hidden_size)`` — Linear(64 → hidden)
        - ``domain_embed_shape``: ``(32, hidden_size)`` — embedding table
    """
    return (
        (COSMOS3_SUPER_MAX_ACTION_DIM, hidden_size),
        (COSMOS3_SUPER_NUM_EMBODIMENT_DOMAINS, hidden_size),
    )

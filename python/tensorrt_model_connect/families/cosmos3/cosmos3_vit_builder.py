"""Cosmos 3 ViT visual encoder TRT builder.

Cosmos 3 uses Qwen3-VL Vision verbatim (see ARCH.md and the
``Qwen3VLVisionModel`` class name in ``model_index.json``). This builder is a
thin wrapper around ``families/qwen_vl/qwen_vl_vision_builder.py`` with the
Cosmos 3 Super vision_config baked in as default values.

Layer feature pyramid: Cosmos 3 inherits Qwen3-VL's DeepStack mechanism,
exporting features from ViT layers 8, 16, 24 alongside the final layer. These
feed into the AR reasoner via DeepStack injection (handled by the AR builder
in Phase 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..qwen_vl.qwen_vl_vision_builder import build_qwen3_vl_vision_engine

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Cosmos3-Super vision constants (locked from vision_encoder/config.json).
COSMOS3_SUPER_VIT_DEPTH = 27
COSMOS3_SUPER_VIT_HIDDEN_SIZE = 1152
COSMOS3_SUPER_VIT_NUM_HEADS = 16
COSMOS3_SUPER_VIT_INTERMEDIATE_SIZE = 4304
COSMOS3_SUPER_VIT_OUT_HIDDEN_SIZE = 5120  # matches reasoner hidden_size
COSMOS3_SUPER_VIT_PATCH_SIZE = 16
COSMOS3_SUPER_VIT_TEMPORAL_PATCH_SIZE = 2
COSMOS3_SUPER_VIT_SPATIAL_MERGE_SIZE = 2
COSMOS3_SUPER_VIT_NUM_POSITION_EMBEDDINGS = 2304
COSMOS3_SUPER_VIT_DEEPSTACK_INDEXES = (8, 16, 24)
COSMOS3_SUPER_VIT_HIDDEN_ACT = "gelu_pytorch_tanh"


def build_cosmos3_vit_engine(
    weights: "WeightDict",
    *,
    fixed_image_size: int = 448,
    vision_config: Optional[dict] = None,
    verbose: bool = False,
) -> bytes:
    """Build the Cosmos 3 ViT visual encoder TRT engine.

    Args:
      weights: weights produced by the cosmos3 plugin's load_weights, with
        the vision encoder keys under ``vision_encoder.*``.
      fixed_image_size: image size to bake into the TRT engine (256/480/720
        resolutions are documented as supported by the model card).
      vision_config: optional explicit vision_config dict. If omitted, the
        Cosmos3-Super defaults documented in this module are used.
      verbose: verbose engine build logging.

    Returns:
      Serialized TRT engine bytes for the ViT encoder.
    """
    if vision_config is None:
        vision_config = {
            "depth": COSMOS3_SUPER_VIT_DEPTH,
            "hidden_size": COSMOS3_SUPER_VIT_HIDDEN_SIZE,
            "num_heads": COSMOS3_SUPER_VIT_NUM_HEADS,
            "intermediate_size": COSMOS3_SUPER_VIT_INTERMEDIATE_SIZE,
            "out_hidden_size": COSMOS3_SUPER_VIT_OUT_HIDDEN_SIZE,
            "patch_size": COSMOS3_SUPER_VIT_PATCH_SIZE,
            "temporal_patch_size": COSMOS3_SUPER_VIT_TEMPORAL_PATCH_SIZE,
            "spatial_merge_size": COSMOS3_SUPER_VIT_SPATIAL_MERGE_SIZE,
            "num_position_embeddings": COSMOS3_SUPER_VIT_NUM_POSITION_EMBEDDINGS,
            "deepstack_visual_indexes": list(COSMOS3_SUPER_VIT_DEEPSTACK_INDEXES),
            "hidden_act": COSMOS3_SUPER_VIT_HIDDEN_ACT,
            "in_channels": 3,
        }

    return build_qwen3_vl_vision_engine(
        vision_config, weights,
        fixed_image_size=fixed_image_size,
        verbose=verbose)

"""Tests for the distributed execution plan schema."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.distributed_plan import (
    ComponentPlan,
    DISTRIBUTED_PLAN_SECTION,
    DistributedConfig,
    DistributedPlan,
    RegionPlan,
    resolve_selector,
    selector_matches,
)
from tensorrt_model_connect.parallel_config import ParallelConfig


def test_distributed_config_from_tensor_parallel_generates_rank_mapping() -> None:
    cfg = DistributedConfig.from_parallel_config(
        ParallelConfig(mode="tensor_parallel", tp_size=4)
    )

    assert cfg.world_size == 4
    assert cfg.axes == {"tp": 4, "pp": 1, "cp": 1, "dp": 1, "ep": 1}
    assert cfg.rank_mapping == [
        {"rank": 0, "tp": 0, "pp": 0, "cp": 0, "dp": 0, "ep": 0},
        {"rank": 1, "tp": 1, "pp": 0, "cp": 0, "dp": 0, "ep": 0},
        {"rank": 2, "tp": 2, "pp": 0, "cp": 0, "dp": 0, "ep": 0},
        {"rank": 3, "tp": 3, "pp": 0, "cp": 0, "dp": 0, "ep": 0},
    ]


def test_distributed_config_requires_axis_product_to_match_world_size() -> None:
    with pytest.raises(ValueError, match="Mesh axis product"):
        DistributedConfig(world_size=8, axes={"tp": 2, "dp": 2})


def test_distributed_config_rejects_duplicate_mesh_coordinates() -> None:
    with pytest.raises(ValueError, match="mesh coordinates"):
        DistributedConfig(
            world_size=2,
            axes={"tp": 2},
            rank_mapping=[
                {"rank": 0, "tp": 0},
                {"rank": 1, "tp": 0},
            ],
        )


def test_tensor_parallel_plan_roundtrips_as_json_section() -> None:
    plan = DistributedPlan(
        model={
            "family": "flux",
            "model_type": "flux",
            "model_id": "black-forest-labs/FLUX.1-schnell",
        },
        mesh=DistributedConfig.from_parallel_config(
            ParallelConfig(mode="tensor_parallel", tp_size=2)
        ),
        components={
            "denoiser": ComponentPlan(
                placement="sharded",
                mesh_axes=["tp"],
                rank_section_pattern="denoiser_rank{rank}_plan",
            ),
            "text_encoder_0": ComponentPlan(
                placement="replicated",
                section="text_encoder_0_plan",
            ),
            "vae_decoder": ComponentPlan(
                placement="replicated",
                section="vae_decoder_plan",
            ),
        },
        bundle_sections={
            "denoiser": {"rank_section_pattern": "denoiser_rank{rank}_plan"},
            "text_encoder_0": {"section": "text_encoder_0_plan"},
            "vae_decoder": {"section": "vae_decoder_plan"},
        },
    )

    encoded = plan.to_json_bytes()
    decoded = DistributedPlan.from_json_bytes(encoded)

    assert DISTRIBUTED_PLAN_SECTION == "distributed_plan.json"
    assert decoded.mesh.axes["tp"] == 2
    assert decoded.components["denoiser"].placement == "sharded"
    assert decoded.components["text_encoder_0"].placement == "replicated"
    assert decoded.bundle_sections["denoiser"] == {
        "rank_section_pattern": "denoiser_rank{rank}_plan"
    }


def test_selector_matches_layer_ranges_and_wildcards() -> None:
    assert selector_matches(
        "denoiser.transformer_blocks[0:18].ffn",
        "denoiser.transformer_blocks.17.ffn",
    )
    assert not selector_matches(
        "denoiser.transformer_blocks[0:18].ffn",
        "denoiser.transformer_blocks.18.ffn",
    )
    assert selector_matches(
        "decoder.layers[*].self_attn",
        "decoder.layers.35.self_attn",
    )
    assert selector_matches(
        "denoiser.transformer_blocks[18:36].*",
        "denoiser.transformer_blocks.21.attention",
    )


def test_resolve_selector_preserves_recipe_region_order() -> None:
    regions = [
        "decoder.layers.0.mlp",
        "decoder.layers.0.self_attn",
        "decoder.layers.1.mlp",
        "decoder.layers.1.self_attn",
    ]

    assert resolve_selector("decoder.layers[*].mlp", regions) == [
        "decoder.layers.0.mlp",
        "decoder.layers.1.mlp",
    ]


def test_plan_selector_validation_reports_unmatched_regions() -> None:
    plan = DistributedPlan(
        mesh=DistributedConfig(),
        regions=[
            RegionPlan(
                selector="decoder.layers[0:4].mlp",
                policy="tensor_parallel",
                tp_size=2,
            )
        ],
    )

    with pytest.raises(ValueError, match="matched no recipe regions"):
        plan.validate_region_selectors(["decoder.layers.5.mlp"])

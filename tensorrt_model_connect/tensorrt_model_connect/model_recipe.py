"""Model recipe objects used by distributed plan compilation.

A recipe is the builder-side description of the model structure that distributed
policies may place, shard, or replicate. It intentionally does not decide the
parallel strategy; it names the regions a plan can target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class RecipeRegion:
    """Named model region that a distributed plan can select."""

    name: str
    kind: str
    layer: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
        }
        if self.layer is not None:
            out["layer"] = self.layer
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class ModelRecipe:
    """Model-family-owned structure consumed by distributed policies."""

    family: str
    component: str
    model_type: str | None = None
    regions: tuple[RecipeRegion, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def region_names(self) -> list[str]:
        return [region.name for region in self.regions]

    def regions_by_kind(self, kind: str) -> list[RecipeRegion]:
        return [region for region in self.regions if region.kind == kind]

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "family": self.family,
            "component": self.component,
            "regions": [region.to_dict() for region in self.regions],
        }
        if self.model_type:
            out["model_type"] = self.model_type
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


def standard_decoder_recipe(
    config: "ModelConfig",
    *,
    family: str,
    component: str = "decoder",
) -> ModelRecipe:
    """Create a recipe for standard decoder-only transformer families."""

    regions: list[RecipeRegion] = []
    for layer in range(int(config.num_hidden_layers)):
        regions.append(
            RecipeRegion(
                name=f"{component}.layers.{layer}.self_attn",
                kind="self_attn",
                layer=layer,
            )
        )
        regions.append(
            RecipeRegion(
                name=f"{component}.layers.{layer}.mlp",
                kind="mlp",
                layer=layer,
            )
        )

    regions.append(RecipeRegion(name=f"{component}.lm_head", kind="lm_head"))
    return ModelRecipe(
        family=family,
        component=component,
        model_type=getattr(config, "model_type", None),
        regions=tuple(regions),
        metadata={
            "num_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
        },
    )

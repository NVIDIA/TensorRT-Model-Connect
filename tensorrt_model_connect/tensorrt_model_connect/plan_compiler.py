"""Distributed plan compiler for rank-local TensorRT bundle sections."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Callable

from .bundle_writer import BundleSection
from .distributed_plan import (
    DISTRIBUTED_PLAN_SECTION,
    ComponentPlan,
    DistributedConfig,
    DistributedPlan,
)
from .model_recipe import ModelRecipe, standard_decoder_recipe
from .parallel_config import ParallelConfig
from .sharding_policy import standard_decoder_sharding_policy


@dataclass(frozen=True)
class CompiledPlanArtifacts:
    """Bundle sections and metadata emitted by a plan compiler."""

    primary_engine_plan: bytes
    sections: list[BundleSection]
    distributed_plan: DistributedPlan | None = None
    recipe: ModelRecipe | None = None


class PlanCompiler:
    """Compile a model recipe and distributed plan into rank-local sections."""

    def __init__(
        self,
        *,
        family: str,
        component: str,
        model_id: str | None,
        model_type: str | None,
        parallel: ParallelConfig,
    ) -> None:
        self.family = family
        self.component = component
        self.model_id = model_id
        self.model_type = model_type
        self.parallel = parallel

    @property
    def rank_section_pattern(self) -> str:
        return f"{self.component}_rank{{rank}}_plan"

    def rank_section_name(self, rank: int) -> str:
        return self.rank_section_pattern.replace("{rank}", str(int(rank)))

    def compile_decoder(
        self,
        build_engine: Callable[..., bytes],
        config: Any,
        weights: Any,
        max_cache_length: int,
        *,
        build_kwargs: dict[str, Any],
        verbose: bool,
    ) -> CompiledPlanArtifacts:
        if not self.parallel.enabled:
            engine_plan = build_engine(
                config, weights, max_cache_length, verbose=verbose, **build_kwargs)
            return CompiledPlanArtifacts(
                primary_engine_plan=engine_plan,
                sections=[BundleSection("engine_plan", engine_plan)],
            )

        recipe = standard_decoder_recipe(
            config, family=self.family, component=self.component)
        rank_engine_plans: dict[int, bytes] = {}
        for rank in range(self.parallel.tp_size):
            rank_parallel = self.parallel.for_rank(rank)
            standard_decoder_sharding_policy(
                config, weights, rank_parallel, recipe=recipe)
            rank_kwargs = dict(build_kwargs)
            rank_kwargs["parallel_config"] = rank_parallel
            print(f"[trtmc-build]   rank {rank}/{self.parallel.tp_size} ...",
                  file=sys.stderr)
            rank_engine_plans[rank] = build_engine(
                config, weights, max_cache_length, verbose=verbose, **rank_kwargs)

        distributed_plan = self._build_distributed_plan(config, weights, recipe)
        sections = [
            BundleSection(self.rank_section_name(rank), plan)
            for rank, plan in sorted(rank_engine_plans.items())
        ]
        sections.append(
            BundleSection(
                DISTRIBUTED_PLAN_SECTION,
                distributed_plan.to_json_bytes(),
            )
        )
        return CompiledPlanArtifacts(
            primary_engine_plan=rank_engine_plans[0],
            sections=sections,
            distributed_plan=distributed_plan,
            recipe=recipe,
        )

    def _build_distributed_plan(
        self,
        config: Any,
        weights: Any,
        recipe: ModelRecipe,
    ) -> DistributedPlan:
        mesh = DistributedConfig.from_parallel_config(self.parallel)
        policy = standard_decoder_sharding_policy(
            config, weights, self.parallel.for_rank(0), recipe=recipe)

        model: dict[str, Any] = {
            "family": self.family,
            "recipe": recipe.to_dict(),
        }
        if self.model_id:
            model["model_id"] = self.model_id
        if self.model_type:
            model["model_type"] = self.model_type

        plan = DistributedPlan(
            model=model,
            mesh=mesh,
            components={
                self.component: ComponentPlan(
                    placement="sharded",
                    mesh_axes=["tp"],
                    rank_section_pattern=self.rank_section_pattern,
                )
            },
            regions=policy.region_plans(),
            collectives=policy.collective_plans(),
            bundle_sections={
                self.component: {
                    "rank_section_pattern": self.rank_section_pattern,
                }
            },
            constraints={
                "parallel_mode": self.parallel.mode,
                "tensor_parallel_size": int(self.parallel.tp_size),
                "tensor_parallel_require_mpirun": bool(self.parallel.require_mpirun),
            },
        )
        plan.validate_region_selectors(recipe.region_names())
        return plan

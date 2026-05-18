---
title: Multi-Device Implementation Map
---

This page maps the current multi-device implementation from the user request
down to the runtime. Read it in order: each section explains what is created,
who creates it, where it is implemented, and what the next layer consumes.

## Current Flow

For a TP=2 build, the ownership chain is:

```text
user build request
  --set parallel.mode=tensor_parallel
  --set parallel.tp_size=2
        |
        v
runtime config schema and CLI resolution
        |
        v
ParallelConfig(mode="tensor_parallel", tp_size=2)
        |
        v
DistributedConfig(world_size=2, axes={"tp": 2, "pp": 1, "cp": 1, "dp": 1, "ep": 1})
        |
        v
ModelRecipe + selector helpers + ShardingPolicy
        |
        v
PlanCompiler
        |
        v
rank-local bundle sections + distributed_plan.json
        |
        v
C++ runtime reads distributed_plan.json, selects the rank section, and creates the TP group
```

`DistributedConfig` is not something a normal user hand-writes today. The user
asks for distributed build behavior through the normal config surface. The
builder converts that request into `DistributedConfig`, serializes it into
`distributed_plan.json`, and the runtime reads that generated plan.

## Status Table

| Order | Area | Current status |
| --- | --- | --- |
| 1 | User config surface | Implemented through `--config` and repeatable `--set`. |
| 2 | `ParallelConfig` | Implemented in Python as build request metadata. |
| 3 | `DistributedConfig` | Implemented in Python as the process mesh stored in the plan. |
| 4 | `ModelRecipe` | Implemented for standard decoder structure. |
| 5 | Selector helpers | Implemented in Python for recipe-region matching. |
| 6 | `ShardingPolicy` | Implemented for decoder TP sharding and collectives. |
| 7 | `PlanCompiler` | Implemented for rank-local decoder engine emission. |
| 8 | `DistributedPlan` | Implemented in Python and serialized as `distributed_plan.json`. |
| 9 | Bundle emission | Implemented in Python builder code. |
| 10 | Runtime consumption | Implemented for decoder TP only. |
| 11 | Mesh runtime | Plan-driven entry point implemented for the current TP group. |
| 12 | CP / DP / PP / EP execution | Schema-shaped, with runtime execution still to add. |

## 1. User Config Surface

In short: this is what the user types or puts in a config file to request
distributed build behavior.

The user asks for distributed behavior at build time. For the current TP path,
that means setting `parallel.mode` and `parallel.tp_size`.

Example build request:

```bash
trtmc build Qwen/Qwen3-0.6B \
  --output qwen3-0.6b-tp2.trtfb \
  --set parallel.mode=tensor_parallel \
  --set parallel.tp_size=2
```

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/runtime_config/schemas/parallel.py
tensorrt_model_connect/tensorrt_model_connect/cli.py
tensorrt_model_connect/tensorrt_model_connect/runtime_config/cli_support.py
tests/builder/test_parallel_config.py
```

The schema defines the user-facing fields:

```python
SCHEMA = Schema(
    namespace="parallel",
    fields=(
        ConfigField(name="mode", default="single", ...),
        ConfigField(name="tp_size", default=1, ...),
        ConfigField(name="rank", default=-1, ...),
        ConfigField(name="require_mpirun", default=True, ...),
    ),
)
```

The E2E manifest path uses the same config surface. Manifest `build_args`
values become repeated `--set` tokens in
`tests/e2e_harness/orchestrator.py`:

```python
parallel = build_args.get("parallel", {})
for key, value in parallel.items():
    tokens.append(f"parallel.{key}={value}")
```

## 2. ParallelConfig

In short: `ParallelConfig` is the small Python object that carries the user's
parallel build request into the builder.

`ParallelConfig` is the builder-side request object. It is still close to the
user request: mode, size, rank, and whether MPI launch is required.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/parallel_config.py
tensorrt_model_connect/tensorrt_model_connect/cli.py
tests/builder/test_parallel_config.py
```

Code shape:

```python
@dataclass(frozen=True)
class ParallelConfig:
    mode: str = "single"
    tp_size: int = 1
    rank: int = -1
    require_mpirun: bool = True
```

The CLI creates it after resolving `--config` and `--set`:

```python
resolved_bundle = resolve_cli_config(config_path=cli_cfg, set_tokens=cli_sets)
parallel_config = parallel_config_from_bundle(resolved_bundle)
```

`ParallelConfig` is not the runtime plan. It is the compact build request that
the builder turns into a real mesh and plan.

### Why ParallelConfig And DistributedConfig Both Exist

They are different layers:

| Object | Who owns it | What it means | Current TP=2 value |
| --- | --- | --- | --- |
| `ParallelConfig` | CLI/build request layer | "The user asked for tensor parallel mode with size 2." | `mode="tensor_parallel", tp_size=2` |
| `DistributedConfig` | Plan/bundle contract layer | "The executable mesh has 2 ranks arranged on the TP axis, with other axes inactive." | `world_size=2`, `axes={"tp": 2, "pp": 1, "cp": 1, "dp": 1, "ep": 1}` |

`ParallelConfig` is intentionally small because it mirrors today's supported
user request. It is convenient for CLI parsing, validation, tests, and build
options.

`DistributedConfig` is needed because the bundle contract must be more general
than today's CLI knobs. It can represent:

- `world_size` independently from one mode-specific size field,
- multiple mesh axes in one plan, such as TP x DP or TP x PP,
- rank-to-axis coordinate mapping,
- communication defaults such as collective backend and all-reduce strategy.

This separation keeps the user-facing request simple while letting
`distributed_plan.json` become the stable runtime contract for future CP, DP,
PP, EP, and mixed-axis plans.

## 3. DistributedConfig

In short: `DistributedConfig` is the normalized process mesh that goes into the
bundle plan.

`DistributedConfig` is the process mesh. It answers how many ranks exist and
how those ranks are arranged across TP, PP, CP, DP, and EP axes.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/distributed_plan.py
tests/builder/test_distributed_plan.py
```

Code shape:

```python
MESH_AXES = ("tp", "pp", "cp", "dp", "ep")

@dataclass
class DistributedConfig:
    world_size: int = 1
    axes: dict[str, int] = field(default_factory=dict)
    rank_mapping: list[dict[str, int]] = field(default_factory=list)
    collective_backend: str = "nccl"
    allreduce_strategy: str = "nccl"
```

For the current TP path, the compiler derives it from `ParallelConfig`:

```python
@classmethod
def from_parallel_config(cls, parallel: Any) -> "DistributedConfig":
    tp_size = int(getattr(parallel, "tp_size", 1))
    return cls(world_size=tp_size, axes={"tp": tp_size})
```

So this request:

```python
ParallelConfig(mode="tensor_parallel", tp_size=2)
```

becomes this mesh in `distributed_plan.json`:

```json
{
  "world_size": 2,
  "axes": {
    "tp": 2,
    "pp": 1,
    "cp": 1,
    "dp": 1,
    "ep": 1
  }
}
```

If no rank mapping is supplied, `DistributedConfig` generates the default
mapping and validates that the axis product equals `world_size`.

## 4. ModelRecipe

In short: `ModelRecipe` gives the model stable region names that plans and
policies can target.

`ModelRecipe` is the builder-side model structure. It names regions that a
distributed plan or policy can select. It does not decide the parallel mode.

The reason this layer exists is to give distributed planning a stable model
vocabulary. Without a recipe, a plan would have to refer to incidental builder
details such as Python local variables, weight key suffixes, or model-family
specific loop structure. That would make every distributed mode tightly coupled
to each builder implementation.

With a recipe, the rest of the MD stack can talk about stable regions:

```text
decoder.layers.0.self_attn
decoder.layers.0.mlp
decoder.lm_head
```

Then selectors, policies, and plans can target those regions without knowing
how the Qwen builder, Flux builder, Wan builder, or another family internally
names variables.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/model_recipe.py
tests/builder/test_model_recipe_sharding_policy.py
```

Current standard decoder recipe:

```python
for layer in range(int(config.num_hidden_layers)):
    RecipeRegion(name=f"decoder.layers.{layer}.self_attn", kind="self_attn")
    RecipeRegion(name=f"decoder.layers.{layer}.mlp", kind="mlp")

RecipeRegion(name="decoder.lm_head", kind="lm_head")
```

This gives stable names such as:

```text
decoder.layers.0.self_attn
decoder.layers.0.mlp
decoder.lm_head
```

Those names are the bridge between model-family code and distributed placement.

When a new model family adds MD support, it should add or reuse a recipe that
names the family-owned components and shardable regions before adding mode
specific sharding behavior. For example:

| Family type | Recipe approach |
| --- | --- |
| Standard decoder LLMs such as Qwen, Llama-style, Phi | Reuse `standard_decoder_recipe()`. |
| Encoder-only models | Add or reuse a future `standard_encoder_recipe()`. |
| Encoder-decoder models | Add or reuse a future `standard_encoder_decoder_recipe()`. |
| Diffusion models such as Flux, PixArt, Wan | Add family-specific or shared diffusion/DiT recipes. |
| MoE models | Extend the decoder recipe with expert and router regions. |

The rule is: if a family wants distributed planning, it needs a recipe that
names its shardable regions. Single-device-only families do not need to care
until they add MD support.

After the recipe exists, the family adds MD support by extending the lower
layers:

| Step | What changes |
| --- | --- |
| 1 | Add recipe regions that describe the model structure. |
| 2 | Add selectors in the plan or policy that target those recipe regions. |
| 3 | Add `ShardingPolicy` rules for local shapes, local weights, and collectives. |
| 4 | Teach `PlanCompiler` how to emit the needed rank-local or component-local sections. |
| 5 | Extend runtime consumption only if the family needs a new section layout or new runtime scheduling behavior. |

So `ModelRecipe` is not another copy of the model graph. It is the stable
interface between model-family knowledge and distributed placement.

## 5. Selector Helpers

In short: selector helpers turn plan patterns into the concrete recipe regions
they match.

Selector helpers match plan selectors to recipe region names. They allow one
plan entry to target many layers without putting mode-specific branches in each
family builder.

A recipe region name is a stable name produced by `ModelRecipe`, for example:

```text
decoder.layers.0.self_attn
decoder.layers.0.mlp
decoder.layers.1.self_attn
decoder.layers.1.mlp
decoder.lm_head
```

A plan selector is a pattern used by `DistributedPlan` / `RegionPlan` to target
one or more recipe regions, for example:

```text
decoder.layers[*].mlp
decoder.layers[0:12].self_attn
decoder.lm_head
```

Matching means this selector:

```text
decoder.layers[*].mlp
```

resolves to recipe regions such as:

```text
decoder.layers.0.mlp
decoder.layers.1.mlp
decoder.layers.2.mlp
```

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/distributed_plan.py
tests/builder/test_distributed_plan.py
```

Code shape:

```python
def selector_matches(selector: str, region_name: str) -> bool:
    ...

def resolve_selector(selector: str, recipe_regions: list[str]) -> list[str]:
    return [region for region in recipe_regions if selector_matches(selector, region)]
```

Examples:

```text
decoder.layers[*].mlp
decoder.layers[0:12].self_attn
denoiser.transformer_blocks[18:36].*
```

The current TP plan uses selectors emitted from the standard decoder policy.
Future partial-sharding plans can use the same selector format to shard only a
layer range or only a region type.

Selector helpers are not one-to-one with `ModelRecipe`. They are generic
matching utilities. A new model family usually only needs new recipe region
names, not new selector helpers, as long as its names follow the same stable
numeric hierarchy. For example, a diffusion recipe could expose:

```text
denoiser.blocks.0.self_attn
denoiser.blocks.0.cross_attn
denoiser.blocks.0.ffn
denoiser.blocks.1.self_attn
```

and use existing selectors:

```text
denoiser.blocks[*].ffn
denoiser.blocks[0:12].cross_attn
```

Add new selector-helper behavior only when the selector language needs to grow,
such as metadata matching, non-numeric block IDs, boolean selectors, or
component aliases like `all_denoiser_attention`.

## 6. ShardingPolicy

In short: `ShardingPolicy` decides what one rank locally builds for the plan.

`ShardingPolicy` converts model structure plus distributed placement into
rank-local build decisions. For the current decoder TP path, it owns weight
slicing, local head counts, local MLP size, and all-reduce joins.

Plainly, it turns:

```text
this plan wants TP / CP / PP / DP / EP on these model regions
```

into:

```text
for this rank, use these local weights, these local tensor shapes, and insert these collectives
```

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/sharding_policy.py
tensorrt_model_connect/tensorrt_model_connect/dual_profile_decoder_builder.py
tests/builder/test_model_recipe_sharding_policy.py
tests/builder/test_family_plugins.py
```

This is where sharding happens today. It is build-time weight slicing, not a
runtime cache and not hidden global state.

Policy creation:

```python
policy = standard_decoder_sharding_policy(
    config,
    weights,
    parallel,
    recipe=recipe,
)
```

Inputs:

| Input | Meaning |
| --- | --- |
| `config` | Model dimensions such as heads, hidden size, and layer count. |
| `weights` | Full model weights before rank-local slicing. |
| `parallel` | Current mode, TP size, and concrete rank. |
| `recipe` | Stable model regions such as `decoder.layers.N.mlp`. |

Outputs and answers:

| Policy method | Meaning |
| --- | --- |
| `shard_weights()` | Returns rank-local weights. |
| `local_num_attention_heads()` | Returns local attention head count for this rank. |
| `local_num_key_value_heads()` | Returns local KV head count for this rank. |
| `join_row_parallel(network, tensor)` | Inserts the needed all-reduce for row-parallel outputs. |
| `region_plans()` | Emits plan metadata for which regions are sharded. |
| `collective_plans()` | Emits plan metadata for required collectives. |

Current decoder TP slicing rules:

```python
if key.endswith((".w_q", ".w_k", ".w_v", ".q_bias", ".k_bias", ".v_bias")):
    out[key] = _slice_last_dim(value, self.rank, self.tp_size)
elif key.endswith((".w_o", ".w_down")):
    out[key] = _slice_first_dim(value, self.rank, self.tp_size)
elif key.endswith((".w_gate", ".w_up", ".w_fc1")):
    out[key] = _slice_last_dim(value, self.rank, self.tp_size)
```

Current behavior by tensor group:

| Tensor group | Current TP behavior |
| --- | --- |
| Q / K / V projections | Slice output dimension. |
| Gate / up projections | Slice output dimension. |
| O / down projections | Slice input dimension. |
| Embeddings, norms, LM head | Replicated. |

The decoder builder consumes policy answers:

```python
weights = policy.shard_weights()
num_heads = policy.local_num_attention_heads()
num_kv_heads = policy.local_num_key_value_heads()
```

It also inserts TensorRT distributed joins through the policy:

```python
attn_out = policy.join_row_parallel(network, attn_out)
mlp_out = policy.join_row_parallel(network, mlp_out)
```

That keeps the decoder builder focused on graph construction. The policy owns
the rank-local distributed decisions.

A new model family does not automatically require a new `ShardingPolicy`.
Reuse or extend the closest architecture policy when possible:

| Case | Policy approach |
| --- | --- |
| Qwen / Llama-style decoder | Reuse the standard decoder policy if weight roles match. |
| Decoder with small naming differences | Reuse the policy with adapter or mapping changes. |
| MoE decoder | Extend the decoder policy with expert and router rules. |
| Diffusion DiT model | Add or reuse a shared DiT/diffusion policy. |
| Very different architecture | Add an architecture-specific policy. |

The rule is: new model family does not always mean new policy; new sharding
behavior or a new architecture pattern usually does.

## 7. PlanCompiler

In short: `PlanCompiler` runs the build for each local rank or stage and writes
the resulting sections into the bundle.

`PlanCompiler` owns rank-local engine emission. It creates the recipe, builds
one engine per rank, asks the policy for plan metadata, and emits bundle
sections.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/plan_compiler.py
tensorrt_model_connect/tensorrt_model_connect/engine_builder.py
tests/builder/test_engine_builder_extended.py
```

The engine builder delegates decoder engine emission to the compiler:

```python
plan_compiler = PlanCompiler(
    family=plugin.name,
    component="decoder",
    model_id=model_dir_path.name,
    model_type=config.model_type,
    parallel=parallel,
)
compiled_plan_artifacts = plan_compiler.compile_decoder(
    plugin.build_engine,
    config,
    weights,
    max_cache_length,
    build_kwargs=extra_kwargs,
    verbose=verbose,
)
```

Inside `compile_decoder`, single-device still emits the normal `engine_plan`.
TP emits rank-local plans:

```python
for rank in range(self.parallel.tp_size):
    rank_parallel = self.parallel.for_rank(rank)
    rank_kwargs["parallel_config"] = rank_parallel
    rank_engine_plans[rank] = build_engine(...)
```

Then it appends `distributed_plan.json`:

```python
sections = [
    BundleSection(self.rank_section_name(rank), plan)
    for rank, plan in sorted(rank_engine_plans.items())
]
sections.append(BundleSection("distributed_plan.json", distributed_plan.to_json_bytes()))
```

For TP=2, the rank-local engine sections are:

```text
decoder_rank0_plan
decoder_rank1_plan
```

## 8. DistributedPlan

In short: `DistributedPlan` is the JSON contract that connects the builder and
runtime for distributed bundles.

`DistributedPlan` is the bundle-level execution contract. It contains the
mesh, model metadata, component placement, region policies, collectives,
rank-local section names, and constraints.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/distributed_plan.py
tensorrt_model_connect/tensorrt_model_connect/plan_compiler.py
tests/builder/test_distributed_plan.py
tests/builder/test_engine_builder_extended.py
```

Code shape:

```python
@dataclass
class DistributedPlan:
    mesh: DistributedConfig
    model: dict[str, Any] = field(default_factory=dict)
    components: dict[str, ComponentPlan] = field(default_factory=dict)
    regions: list[RegionPlan] = field(default_factory=list)
    collectives: list[CollectivePlan] = field(default_factory=list)
    bundle_sections: dict[str, dict[str, Any]] = field(default_factory=dict)
```

The current TP plan is built in `PlanCompiler._build_distributed_plan`:

```python
mesh = DistributedConfig.from_parallel_config(self.parallel)
policy = standard_decoder_sharding_policy(
    config, weights, self.parallel.for_rank(0), recipe=recipe)

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
)
```

For runtime, the most important TP field is:

```json
{
  "bundle_sections": {
    "decoder": {
      "rank_section_pattern": "decoder_rank{rank}_plan"
    }
  }
}
```

That tells rank 0 to load `decoder_rank0_plan` and rank 1 to load
`decoder_rank1_plan`.

## 9. Bundle Emission

In short: bundle emission writes the normal model artifacts plus any
distributed plan and rank-local engine sections.

The bundle is the build/runtime boundary. For single-device builds, the bundle
keeps the normal `engine_plan`. For TP builds, the bundle carries rank-local
engine sections plus `distributed_plan.json`.

Implemented in:

```text
tensorrt_model_connect/tensorrt_model_connect/engine_builder.py
tensorrt_model_connect/tensorrt_model_connect/parallel_config.py
tensorrt_model_connect/tensorrt_model_connect/bundle_writer.py
tests/builder/test_engine_builder_extended.py
```

Current TP=2 bundle sections:

```text
decoder_rank0_plan
decoder_rank1_plan
distributed_plan.json
config.json
```

The bundle config points runtime code at the plan:

```python
def to_bundle_config_fields(self) -> dict[str, object]:
    if not self.enabled:
        return {}
    return {
        "parallelism": self.to_config_dict(),
        "distributed_plan_section": "distributed_plan.json",
    }
```

That means runtime does not rely on legacy `tensor_parallel_*` config fields
for decoder TP.

## 10. Runtime Consumption

In short: runtime consumption is the plugin-specific code that reads the plan
and uses it to load the right local model sections.

Runtime consumption means the C++ decoder runtime reads
`distributed_plan.json`, selects the correct local engine section for the
launched rank, initializes the process group, and passes the communicator to
the TensorRT backend.

This layer is plugin or task specific. The decoder plugin consumes the plan by
loading decoder rank sections and running token generation. A future diffusion
plugin would consume the plan by loading text encoder, denoiser, and VAE
sections and running the denoise loop.

Implemented in:

```text
include/trtmc/runtime/distributed_runtime.h
src/runtime/core/distributed_runtime.cpp
src/runtime/plugins/decoder_plugin.cpp
src/runtime/backend/trt_module_impl.cpp
tests/cpp/test_distributed_runtime_plan.cpp
```

The decoder plugin reads the plan section:

```cpp
runtime.plan = parse_distributed_plan_runtime_config(plan_json, "decoder");
runtime.group = initialize_mesh_runtime_group(runtime.plan);
runtime.engine_section_name =
    distributed_rank_section_name(runtime.plan.rank_section_pattern,
                                  runtime.group.rank);
```

Then it loads the selected section and passes the communicator through module
creation options:

```cpp
auto* plan = find_section(ctx.bundle, section_name);

ModuleCreateOptions opts;
opts.distributed_communicator = mesh_runtime.group.communicator;
opts.distributed_owner = mesh_runtime.group.owner;
```

The TensorRT backend attaches that communicator when TensorRT 11 APIs are
available:

```cpp
#if NV_TENSORRT_MAJOR >= 11
if (ctx_->setCommunicator(distributed_communicator_))
    return true;
#endif
```

## 11. Mesh Runtime

In short: mesh runtime is the shared rank, device, group, and communicator
manager.

Mesh runtime is the shared C++ runtime layer for launched rank detection,
device binding, communicator creation, and communicator lifetime.

This layer should not be different for every family. If the code is about MPI
rank, local CUDA device, world size, TP / CP / DP / PP / EP coordinates, or
NCCL communicators, it belongs in mesh runtime. If the code is about which
model sections to load or how the task runs, it belongs in runtime
consumption.

Implemented in:

```text
include/trtmc/runtime/distributed_runtime.h
src/runtime/core/distributed_runtime.cpp
tests/cpp/test_distributed_runtime_plan.cpp
```

Current entry point:

```cpp
MeshRuntimeGroup initialize_mesh_runtime_group(const MeshRuntimeConfig& config) {
    if (!config.enabled)
        return MeshRuntimeGroup{};
    return initialize_tensor_parallel_group(config.tp_size);
}
```

The decoder plugin calls this layer instead of implementing NCCL rendezvous,
rank detection, or local device binding itself.

## 12. CP / DP / PP / EP Execution

In short: the schema can describe these axes now, but executable runtime
semantics still need to be implemented.

CP, DP, PP, and EP are represented in the schema today, but they are not
executable runtime modes yet.

Implemented today:

```text
tensorrt_model_connect/tensorrt_model_connect/distributed_plan.py
src/runtime/core/distributed_runtime.cpp
tests/builder/test_distributed_plan.py
tests/cpp/test_distributed_runtime_plan.cpp
```

The Python schema already has the axes:

```python
MESH_AXES = ("tp", "pp", "cp", "dp", "ep")
```

The runtime parser reads them:

```cpp
cfg.tp_size = extract_json_int(axes, "tp", 1);
cfg.pp_size = extract_json_int(axes, "pp", 1);
cfg.cp_size = extract_json_int(axes, "cp", 1);
cfg.dp_size = extract_json_int(axes, "dp", 1);
cfg.ep_size = extract_json_int(axes, "ep", 1);
```

But runtime execution currently rejects non-TP axes:

```cpp
if (cfg.pp_size != 1 || cfg.cp_size != 1 ||
    cfg.dp_size != 1 || cfg.ep_size != 1) {
    throw std::runtime_error(
        "This runtime currently supports only tensor-parallel distributed plans");
}
```

A future mixed plan can use the existing schema shape:

```json
{
  "mesh": {
    "world_size": 8,
    "axes": {
      "tp": 2,
      "pp": 2,
      "cp": 1,
      "dp": 2,
      "ep": 1
    }
  },
  "regions": [
    {
      "selector": "decoder.layers[0:12].self_attn",
      "policy": "tensor_parallel",
      "mesh_axes": ["tp"]
    },
    {
      "selector": "decoder.layers[12:24].*",
      "policy": "pipeline_parallel",
      "mesh_axes": ["pp"]
    }
  ]
}
```

The missing work is runtime execution semantics: process-group creation for
each axis, scheduling rules, policy implementations, and launched E2E coverage.

## Where To Extend

Add new parallel modes by extending the owner of each decision:

| Need | Modify here |
| --- | --- |
| Add user-facing build knobs | `runtime_config/schemas/parallel.py`, or a new schema namespace if the knob is not generic parallelism. |
| Convert request metadata to a mesh | `parallel_config.py` and `distributed_plan.py`. |
| Describe model regions that can be sharded or staged | `model_recipe.py` or family-specific recipe helpers. |
| Convert plan placement into local tensors and collectives | `sharding_policy.py` or a new policy module. |
| Emit rank-local or stage-local sections | `plan_compiler.py`. |
| Write/read the plan contract | `distributed_plan.py`, `engine_builder.py`, and the relevant runtime plugin. |
| Build more process groups | `src/runtime/core/distributed_runtime.cpp`. |
| Attach communicators to TensorRT modules | Reuse `ModuleCreateOptions.distributed_communicator` in `include/trtmc/runtime/trt_backend.h`. |
| Prove behavior | Add builder tests, C++ runtime-plan tests, and launched E2E tests. |

Expected mode-specific work:

| Mode | High-level work needed |
| --- | --- |
| CP | Add attention/KV-cache region policy, CP groups, sequence partition rules, and runtime attention communication. |
| DP | Add replicated model groups, request partitioning or reduction semantics, and DP launch validation. |
| PP | Add layer or component placement, stage-local engine sections, activation send/recv, and pipeline scheduling. |
| EP | Add expert placement, routing metadata, expert collectives, and expert-local engine sections. |

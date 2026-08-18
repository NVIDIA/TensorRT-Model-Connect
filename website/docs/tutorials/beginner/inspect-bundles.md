---
title: Beginner Tutorial - Inspect Bundles
---

import Diagram from '@site/src/components/Diagram';

A `.bundle` bundle is the portable handoff between the Python builder and C++ runtime. Inspecting it is the first debugging step.

## Learning objectives

By the end of this lab, you should be able to classify a bundle as native or
platform-specialized, identify its family/runtime ownership, and route a load
failure to the artifact, model DSO, backend, or provider boundary.

The source-ownership checks near the end require a repository checkout; bundle
inspection itself does not.

<div className="trtmc-handout-meta">
  <div>
    <strong>Level</strong>
    <span>Beginner</span>
  </div>
  <div>
    <strong>Artifact</strong>
    <span>`model.bundle`</span>
  </div>
  <div>
    <strong>Skill</strong>
    <span>Separate builder, artifact, and runtime failures.</span>
  </div>
  <div>
    <strong>Proof</strong>
    <span>Identify header metadata and the bundle section inventory.</span>
  </div>
</div>

<Diagram
  src="/img/diagrams/getting-started/bundle-contents.svg"
  alt=".bundle artifact contents split between identity metadata, native sections, and optimized runtime artifacts"
  caption="The section inventory tells you whether the bundle will use native model/backend dispatch or an embedded optimized implementation."
/>

## Why inspect first

Most runtime failures become easier once you know what is actually inside the
artifact. Before debugging C++ code, answer these questions:

1. What model family built this bundle?
2. Does the section table contain `optimized_runtime.json`?
3. If native, what `runtime_strategy` will the C++ runtime dispatch?
4. Which engine or artifact sections and assets are present?
5. Which TensorRT version or ABI metadata was recorded?

:::danger Required task
Do not skip inspection after building a bundle. Record the answers above before
running the C++ runtime. For an optimized bundle, the current inspector proves
that the descriptor and artifact sections exist, but it does not decode the
descriptor's implementation or profile identity.
:::

:::warning Common trap
A bundle can exist on disk and still be unusable for the runtime you are
testing. For native bundles, the strategy, engine sections, assets, and
TensorRT ABI are part of the contract. For optimized bundles, the
implementation/profile identities and integrity-bound embedded artifact tree
are part of the contract.
:::

## Use the inspector

```bash
trtmc inspect ./qwen3-0.6b.bundle
trtmc inspect ./qwen3-0.6b.bundle --list-engines
```

The second command is a native-bundle check. `--list-engines` recognizes native
plan sections such as `engine_plan` and `*_plan`; optimized artifacts such as
`optimized_runtime_artifacts/.../llm.engine` are not reported as engine
sections, so an optimized bundle can legitimately produce
`No engine sections found.`. Use the regular inspector to see its complete
section-name inventory.

The inspector reads the bundle header through the same unified `trtmc` binary that runs the artifact. It is useful immediately after a build, before loading engines for inference.

Example shape:

```text
Model ID:            Qwen/Qwen3-0.6B
Model type:          qwen3
Family:              qwen
Runtime strategy:    qwen_decoder_kv_cache
TRT version:         <version recorded at build time>
TRT ABI:             <ABI key>
Precision:           bf16
Sections:
  config.json: 0.0 MB
  prefill_engine_plan: <size> MB
  engine_plan: <size> MB
  tokenizer.json: <size> MB
```

The exact sizes and TensorRT metadata depend on your build environment. This
example is a native bundle; its family, runtime strategy, precision, and
required sections are visible before inference. An optimized bundle instead
lists section names such as `optimized_runtime.json`, implementation metadata,
and `optimized_runtime_artifacts/...`; `runtime_strategy` may be empty. Neither
the C++ nor Python inspector currently decodes `optimized_runtime.json`, so
implementation/profile values are not part of this output.

:::tip Progress check
You are ready for inference when inspection confirms the expected family and
bundle shape, plus either the native strategy/engine sections or the optimized
descriptor/artifact sections. Exact optimized implementation/profile identity
is established by provider qualification and bundle-contract tests, then
enforced again by the runtime loader.
:::

:::danger Required task
Run inspection for the bundle and record its header and section names before
starting runtime debugging. Record the native dispatch identity when the
inspector prints one; do not infer an optimized implementation/profile identity
from section names alone.
:::

## Fields to check

| Field | Why it matters |
| --- | --- |
| `model_id` | Confirms the source model. |
| `family` | Confirms the Python family model module that built the artifact. |
| `precision` | Confirms build precision. |
| `runtime_strategy` | Selects the native C++ model DSO and pipeline plugin; it may be empty for an optimized bundle. |
| `optimized_runtime.json` section | Its presence selects the optimized path. The descriptor payload binds the implementation/profile, but the current inspector does not print those values. |
| `max_cache_length` | Controls default KV cache capacity for decoder bundles. |
| Engine sections | Confirm the bundle contains the plans expected by the strategy. |
| Tokenizer sections | Required for text, speech-token, and multimodal prompt flows. |

## Read native strategy like a runtime engineer

For a native bundle, `runtime_strategy` is the bridge from artifact to C++
implementation:

<Diagram
  src="/img/diagrams/tutorials/beginner/native-runtime-dispatch.svg"
  alt="Native runtime strategy resolution through the generated plugin index, model DSO, registry, and task API"
  caption="runtime_strategy connects native bundle metadata to a registered model plugin and its concrete IPipeline implementation."
/>

Do not confuse it with `family`. `family` identifies the Python `model.py`
recipe that created the native bundle. `runtime_strategy` explains its C++
runtime shape.
In the current model-encapsulated layout, every runtime strategy has exactly
one model-manifest owner and selects that owner's DSO before registry lookup.

An optimized bundle takes a different branch before this lookup. Its
`optimized_runtime.json` identifies an exact implementation and qualified
profile; the host integrity-checks and materializes the embedded artifact tree,
loads its exact `libtrtmc_impl_*.so`, and calls the private factory. It does not
consult the native strategy index, model-plugin search paths, or backend-DSO
search paths.

Examples:

| Family | Possible runtime strategy | Runtime meaning |
| --- | --- | --- |
| `qwen` | `qwen_decoder_kv_cache` | Qwen-owned decoder text generation with attention cache. |
| `llama` | `llama_decoder_kv_cache` | LLaMA-owned decoder text generation in its own model DSO. |
| `whisper` | `whisper_speech_to_text` | Whisper audio features to transcript. |
| `qwen_vl` | `qwen_vl_vision_language` | Qwen-VL image encoder plus text decoder. |
| `internvl` | `internvl_vision_language` | InternVL image encoder plus text decoder. |
| `flux` | `diffusion_flux` | Flux text-to-image denoising pipeline. |
| `wan_t2v` | `diffusion_wan` | Wan text-to-video denoising pipeline; runtime owner directory is `wan`. |
| `pixart` | `diffusion_pixart` | Image diffusion generation. |

:::tip Progress check
You understand this section when you can explain why Qwen and LLaMA have
different strategy keys even though both implement decoder text generation,
and how an E2E `task_strategy` groups them under the same user-visible task.
:::

## Common mismatch

If `runtime_strategy` is present but runtime creation fails with "No plugin registered", check:

- The owning `src/runtime/models/<owner>/MODEL.toml` declares the strategy,
  runtime library, and `plugin.cpp|registrar` pair.
- The plugin uses `REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST` with the same
  strategy and registrar.
- The owning model DSO is present in a configured model-plugin search
  directory.
- The built binary is from the same source tree as the bundle strategy you are testing.

## Debugging checklist

| Check | Command or source |
| --- | --- |
| Bundle header parses | `trtmc inspect model.bundle` |
| Qwen strategy is declared | `rg -n 'qwen_decoder_kv_cache' src/runtime/models/qwen/MODEL.toml` |
| Qwen registrar is implemented | `rg -n 'REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST' src/runtime/models/qwen/plugin.cpp` |
| Runtime DSO was built | `find build/models/qwen -name 'libtrtmc_model_qwen.so' -print` |
| Native engine sections exist | `trtmc inspect model.bundle --list-engines` |
| Optimized descriptor/artifact section names exist | `trtmc inspect model.bundle` |
| Exact optimized descriptor identity is valid | Family-owned provider qualification and bundle-contract tests; the current inspector is only a section-presence check. |
| E2E manifest matches expected contract | `tests/e2e/models/<family>/manifests/<model>.json` |

Inspecting the bundle should become muscle memory. It tells you whether you are debugging the builder, the artifact, the runtime loader, or request execution.

## Learning Log Prompts

Before leaving the tutorial, write short answers to these prompts:

1. Which field selects native dispatch, and which section claims optimized
   dispatch?
2. Which fields or sections prove the tokenizer is packaged?
3. Which metadata would you inspect for TensorRT compatibility?
4. Which optimized descriptor values are not exposed by the current inspector?
5. If inspection passes but runtime loading fails, where is the likely boundary?

<details>
<summary>Check your answers</summary>

1. Native dispatch uses `runtime_strategy`; `optimized_runtime.json` claims the
   platform-specialized path.
2. The section inventory should contain the tokenizer/processor assets required
   by that model runtime.
3. Check the recorded TensorRT compatibility/ABI metadata and the section
   layout expected by the selected runtime.
4. The current inspector does not decode the optimized implementation ID,
   profile ID, or full provider qualification tuple.
5. For native bundles, start with the model DSO/plugin and backend search path.
   For optimized bundles, start with the embedded implementation descriptor,
   artifact integrity, and provider dependencies.

</details>

{/* Collaborative review anchor: batch 2. */}

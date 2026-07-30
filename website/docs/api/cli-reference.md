---
title: CLI Reference
---

## `trtmc build`

`trtmc build` builds `.trtfb` bundles through the Python builder package.

```bash
trtmc build <hf-repo-or-local-dir> [-o <output.trtfb>] [options]
```

The C++ bridge runs `python -m tensorrt_model_connect build ...`. When
installed from the release wheel, `trtmc` is the native executable installed in
the environment's `bin/` directory and it uses the sibling `python3` or
`python` from that same environment. A source-built `./build/trtmc` falls back
to `python3` from the user's shell.

Source builds use the same subcommands through `./build/trtmc`.

Direct module execution is still available for debugging:

```bash
python -m tensorrt_model_connect build <hf-repo-or-local-dir> [-o <output.trtfb>]
```

When `-o`/`--output` is omitted, the CLI derives
`<model-name>.trtfb` from the Hugging Face ID or local-directory basename and
replaces unsafe filename characters with `-`.

### Build options

| Option | Purpose |
| --- | --- |
| `-o`, `--output PATH` | Output bundle path. Defaults to the sanitized model basename plus `.trtfb`. |
| `--kernel FILE.yaml` | Replace a family-owned kernel slot with the trusted TVM-FFI DSO declared by this YAML manifest. Requires the native TensorRT backend. |
| `--recipe RECIPE_ID INSTANCE_ID` | Build a load-time TVM-FFI slot from one exact family-owned graph Recipe. Internally uses the ordinary graph capture, selection, and patch paths and writes `<output-basename>.selection.json`. |
| `--graph-patch REGION.json` | Replace one explicitly selected TensorRT region with a load-time TVM-FFI slot. Requires the native TensorRT backend. |
| `--model-revision REV` | Build a Hugging Face commit, tag, or branch instead of its default revision. |
| `--trust-remote-code` | Accepted for E2E-command compatibility. The current build dispatcher does not forward this flag as a universal remote-code gate; family/model loaders own their loading behavior. Review the checkpoint and family implementation, and do not assume omitting this flag prevents every remote-code path. |
| `--decoder-engine-layout split|dual_profile` | Select separate prefill/decode engines or one multi-profile decoder engine. |
| `--dynamic-kv-cache` | Enable runtime-resizable KV cache support. |
| `--tensor-parallel-size N`, `--tp-size N` | Build a supported decoder for TP size `1`, `2`, `4`, or `8`. |
| `--dynamic-kv-profile-rows A,B,C` | Override dynamic-KV optimization profiles. |
| `--image-height`, `--image-width` | Diffusion image shape overrides. |
| `--video-height`, `--video-width`, `--video-num-frames` | Diffusion video shape overrides. |
| `--num-inference-steps N` | Diffusion denoising step override. |
| `--max-batch-size N` | Build supported diffusion engines for a maximum per-call batch. |
| `--precision fp32|fp16|bf16` | Override the family-selected build precision. Wan2.2 defaults to BF16. |
| `--fp32-layers I,J` | Keep selected model-local layer indices in FP32. |
| `--quantize fp8|int8|int8_sq|int4|int4_awq|nvfp4|w4a8` | Quantization format. |
| `--quant-scales PATH` | Load precomputed quantization scales. |
| `--quant-calibration-samples N` | PTQ calibration sample count. |
| `--fp8` | Enable FP8 using family-provided scales when available, otherwise auto-calibrate. |
| `--fp8-scales PATH` | Load precomputed FP8 scales from a readable UTF-8 JSON object. Missing, unreadable, malformed, or non-object input fails before the native build starts. |
| `--save-fp8-scales PATH` | Save calibrated FP8 scales. |
| `--rtx` | Build for TensorRT-RTX backend. |
| `--config FILE` | Load a schema-driven JSON or YAML profile. YAML requires PyYAML. |
| `--set NS.FIELD=VALUE` | Override a config field; repeatable. |
| `--build-timing-json PATH` | Write structured build timing. |
| `--verbose` | Enable verbose TensorRT builder output. |

TriAttention options are also exposed for experimental KV compaction: `--triattention-stats`, `--triattention-kv-budget`, `--triattention-divide-length`, `--triattention-recent-window`, score aggregation, prompt-token accounting, prefill protection, and MLR/trig disable flags.

The compatibility option `--max-cache-length N` remains accepted but is hidden
from `build --help`. Omitting it lets the selected family choose the capacity:
eligible dense Qwen3 and Llama builds use the checkpoint's full
`max_position_embeddings`; other native or legacy paths normally use 256.
For those Qwen3/Llama models, an explicit value preserves native KV only when it
equals the full model context and the other native-KV constraints are also met.

Eligible dense Qwen3 and Llama checkpoints declare a model-owned native default
route. A model-only build skips the optimized-provider probe and selects BF16,
full-context fixed KV, and split prefill/decode engines. Other families probe
their exact qualified optimized profiles before falling back to their native
builder.

TensorRT is the build backend; there is no public build-method selector. Older
`--method trt` and `--method auto` spellings remain accepted for compatibility.

## `trtmc kernel slots`

List the external-kernel contracts and exact instance IDs published for a
model without downloading its weights:

```bash
trtmc kernel slots <hf-repo-or-local-dir> [--model-revision REV]
```

Supplying `--kernel` validates one strict YAML manifest, the referenced trusted
DSO, its SHA-256 digest, and the selected slot instances. It bypasses
optimized-provider selection and uses the owning family's native TensorRT build
path; it cannot be combined with `--rtx`. See
[Bring Your Own Kernel](../tutorials/beginner/bring-your-own-kernel.md) for the
end-to-end workflow.

## `trtmc graph`

Capture a raw TensorRT graph, list its explicit node IDs, and select one region:

```bash
trtmc graph inspect \
  --snapshot graph.json \
  [--engine-role prefill|decode|dual_profile] \
  <hf-repo-or-local-dir> [build options...]

trtmc graph list graph.json [--match GLOB]

trtmc graph recipes graph.json

trtmc graph select graph.json \
  --nodes NODE_ID [NODE_ID ...] \
  --binding-id ID \
  [--workspace-bytes N] [--output-shape-like-input INPUT_INDEX] \
  [--extra-arg JSON]... \
  -o region.json
```

`inspect` passes the model and all following options verbatim to `trtmc build`,
captures immediately before TensorRT serialization, and does not compile a
bundle. Put its own `--snapshot` and `--engine-role` options before the model.
`list` prints node IDs, operation and layer names, and tensor edges; `--match`
only filters displayed IDs, operations, or names.

`recipes` shows exact, versioned region instances recorded by the owning model
family while it constructed this graph. The recommended shortcut is:

```bash
trtmc build MODEL [build options...] \
  --recipe RECIPE_ID INSTANCE_ID \
  -o model-slot.trtfb
```

That one command orchestrates the existing graph capture, exact Recipe
resolution, `select_region()` validation, and ordinary `--graph-patch` build.
It writes `model-slot.selection.json` as the ABI receipt. Recipes add no
runtime schema or weaker validation. Zero matches, duplicate matches, and
invalid regions fail.

`select` is the advanced path and accepts only explicit node IDs. Recipe and
manual selection both print the ordered boundary tensor IDs, names, dtypes,
shapes, and ABI hash. Each manual `--extra-arg` is one strict JSON object whose
type is `none`, `int`, `float`, or `ptr`. A dynamic output additionally requires
`--output-shape-like-input`; fixed outputs reject that option. Manual
`--binding-id` accepts only ASCII letters, digits, `_`, `.`, `@`, and `-`.
`--workspace-bytes` accepts values from 0 through 2147483647.

Build the same model revision and options with `--graph-patch region.json` to
produce a slot-ready native bundle. The selection must describe one connected,
convex region and must still match the live graph fingerprint. See
[Bring Your Own Kernel](../tutorials/beginner/bring-your-own-kernel.md) for the
load-time binding workflow and current limitations.

## Runtime commands

`trtmc` also inspects and runs bundles from C++.

```bash
trtmc run <bundle.trtfb> --prompt "text" [--image PATH] [--greedy] \
  [--kernel-bindings kernel-bindings.json]
trtmc encode <bundle.trtfb> --prompt "text"
trtmc segment <bundle.trtfb> --image PATH --output PATH
trtmc segment-prompted <bundle.trtfb> --image PATH --output DIR [--point-x F --point-y F]
trtmc segment-prompted <bundle.trtfb> --image PATH --output DIR --prompt "object"
trtmc classify <bundle.trtfb> --image PATH [--benchmark N --warmup N]
trtmc detect <bundle.trtfb> --image PATH [--output-json PATH]
trtmc generate-audio <bundle.trtfb> --prompt "text" --output PATH
trtmc serve-audio <bundle.trtfb>
trtmc generate-video <bundle.trtfb> --prompt "text" --output DIR
trtmc embed <bundle.trtfb> --prompt "text"
trtmc rerank <bundle.trtfb> --prompt "query" --document "text"
trtmc solve <bundle.trtfb> --field-input CSV
trtmc solve <bundle.trtfb> --branch-input CSV [--trunk-input CSV]
trtmc transcribe <bundle.trtfb> --audio FILE.wav [--stream]
trtmc speak <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav
trtmc inspect <bundle.trtfb>
trtmc inspect <bundle.trtfb> --list-engines
trtmc version
```

Regular `trtmc inspect` prints bundle-header fields and section names. The
presence of `optimized_runtime.json` identifies an optimized bundle, but
inspection does not decode that descriptor or print its implementation/profile
identity. `trtmc inspect --list-engines` recognizes only the native
`engine_plan` and `*_plan` section naming convention. Optimized artifacts use
capsule-owned names such as `optimized_runtime_artifacts/.../llm.engine`, so
`--list-engines` can legitimately report `No engine sections found.` and exit
nonzero for an otherwise valid optimized bundle.

Depending on the command, shared load/run options include `--hf-python`,
`--backend-dir`, repeatable `--model-plugin-dir`, `--runtime-cache`,
`--kernel-bindings`, `--cuda-graphs`, `--benchmark`, `--warmup`, `--config`,
and repeatable `--set`. `trtmc --help` prints one combined synopsis for all
commands; it is not separate per-command help. Read the relevant command
section in that combined output and this reference for the accepted options.

These shared options have route-specific contracts:

- On native TensorRT-RTX bundles, `--runtime-cache` names a JIT kernel cache
  file. On an optimized-runtime bundle, it names the root directory where the
  host materializes the integrity-bound artifact cache.
- `--kernel-bindings` is required for a native bundle containing
  `kernel_slots.json` and rejected for bundles without slots. Its strict JSON
  manifest binds every slot exactly once to a relative TVM-FFI DSO path,
  exported function, and matching ABI SHA-256.
- For Python builds, `--config` accepts `.json`, `.yaml`, and `.yml` profiles;
  YAML requires PyYAML. The C++ load/run `--config` surface accepts `.json`
  only and rejects YAML with a conversion error. The current Qwen
  optimized-runtime route rejects runtime `--config` and `--set` altogether.

Text-generation options include `--max-new-tokens`, `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, and `--no-thinking`.

### Complete native long-option index

The native parser accepts the following canonical long options. An option is
valid only on the command whose synopsis or section describes it; this table is
an inventory, not a claim that every option is accepted by every command.

| Area | Canonical options |
| --- | --- |
| Help and version | `--help`, `--version` |
| Primary inputs | `--prompt`, `--prompts-file`, `--image`, `--audio`, `--audio-in`, `--document`, `--field-input`, `--branch-input`, `--trunk-input` |
| Output selection | `--output`, `--output-json`, `--audio-out`, `--list-engines` |
| Runtime loading and config | `--hf-python`, `--backend-dir`, `--model-plugin-dir`, `--runtime-cache`, `--kernel-bindings`, `--kv-cache-size`, `--cuda-graphs`, `--config`, `--set` |
| Text generation | `--max-new-tokens`, `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, `--no-thinking`, `--generation-mode`, `--block-length`, `--threshold`, `--num-samples`, `--tail-frames` |
| Diffusion and raw-state generation | `--num-steps`, `--num-inference-steps`, `--guidance-scale`, `--cfg-scale`, `--sde-gamma`, `--initial-latents-raw`, `--condition-latents-raw`, `--condition-mask-raw`, `--sampling-steps-raw`, `--sde-noise-raw`, `--negative-prompt`, `--height`, `--width`, `--num-images` |
| Dynamic adapters | `--lora-adapter`, `--lora-adapter-id` |
| Transcription | `--beam-size`, `--language`, `--source-language`, `--target-language`, `--task`, `--punctuation`, `--no-punctuation`, `--timestamps`, `--no-timestamps`, `--max-input-seconds`, `--segment-length-seconds`, `--stream`, `--chunk-ms`, `--att-context-size`, `--pad-and-drop-preencoded` |
| Audio streaming | `--chunk-frames` |
| Segmentation and detection | `--point-x`, `--point-y`, `--background`, `--score-threshold` |
| Measurement | `--benchmark`, `--warmup` |

`--threshold` supplies the generation confidence threshold, while
`--score-threshold` supplies object-detection confidence. `--background`
marks a prompted-segmentation point as background instead of foreground.
`--chunk-frames` controls generated-audio stream chunks;
`--chunk-ms` controls transcription input chunks. The legacy
`--kv_cache_size` spelling remains accepted for compatibility, but new scripts
must use `--kv-cache-size`.

### Qwen-VL dynamic LoRA

Dynamic LoRA must be enabled when building the base engine. It currently
supports Qwen2.5-VL only and is incompatible with tensor-parallel Qwen-VL
builds. `qwen_vl_lora.max_rank` must be between 1 and 256 when enabled:

```bash
trtmc build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen-vl-lora.trtfb \
  --set qwen_vl_lora.enabled=true \
  --set qwen_vl_lora.max_rank=64 \
  --set qwen_vl_lora.target_modules=q_proj,k_proj,v_proj,o_proj
```

Load one standard PEFT adapter directory and select it for the request:

```bash
trtmc run /tmp/qwen-vl-lora.trtfb \
  --prompt "Describe the image." \
  --image /tmp/example.png \
  --lora-adapter /tmp/my-peft-adapter \
  --lora-adapter-id product-style
```

The directory must contain `adapter_config.json` and
`adapter_model.safetensors`. The runtime rejects non-LoRA PEFT modes, DoRA,
rsLoRA, QALoRA, adapted bias, `modules_to_save`, per-module rank/alpha
patterns, unsupported target modules, incomplete A/B tensor pairs, and ranks
or shapes that exceed the engine contract. `--lora-adapter-id` must not be
empty; when omitted while `--lora-adapter` is present, the CLI uses
`default`. Supplying an adapter to an engine built without dynamic LoRA inputs
fails during loading. The one-shot CLI loads the adapter before generation,
sets `GenerateConfig::lora_adapter_id`, and exits after that request; use the
C++ lifecycle API for a long-lived adapter registry.

Object detection is exposed through `trtmc detect` for a pipeline that
implements `IPipeline::detect`. The current model manifests and E2E catalog do
not provide an object-detection model, so command availability alone is not
support evidence.

### Canary transcription options

`trtmc transcribe` accepts repeated `--audio` inputs. These options apply to
every input in that CLI batch. Canary executes up to 16 inputs per encoder
batch and automatically chunks additional inputs:

| Option | Purpose |
| --- | --- |
| `--beam-size N` | Greedy at `1`; Canary beam search at `2` through `16`. |
| `--source-language TAG` | Language code for the input audio. |
| `--target-language TAG` | Language code for the decoded text. |
| `--task transcribe|translate` | Validate and select ASR versus translation prompting. |
| `--punctuation`, `--no-punctuation` | Enable or remove punctuation in decoded text. |
| `--timestamps` | Print segment start/end seconds with each transcript. |
| `--max-new-tokens N` | Per-segment decoder output limit. |
| `--max-input-seconds F` | Reject inputs longer than this duration. |
| `--segment-length-seconds F` | Decode independent audio windows of this duration. |

See [Configurable Canary Decoding](/tutorials/intermediate/canary-decoding)
for bounds, batch output, and local checkpoint examples.

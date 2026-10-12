# InternVL: optional native Edge-LLM execution

The family owns configuration admission, builder argument mapping, bundle
composition, runtime orchestration, and validation. Edge owns complete visual
and language networks. This route uses the official GitHub Edge-LLM **0.10.1**
snapshot **e8b29522938901f6df19ebeedd4b69bc8edbcd97**, provisioned through the
optional native CMake package. It does not use another checkout or cross-compile.

## Scope

Only original-source InternVL3 HF checkpoints with the recorded dense Qwen2
text and 448-pixel vision configurations enter this route. Compute is FP16,
batch size and tensor/context parallelism are one. The 1B/2B/8B configurations
route on native x86 SM80; 14B routes on native x86 SM120. Other platforms,
quantized sources, InternVL3.5, multiple-image public requests, and speculative
decoding are not qualified by this change.

The existing native builder remains the default for requests outside this
map. If Edge preparation fails, the family logs a warning and retries the
unchanged native request once. Publication errors never retry into a different
backend. Runtime loading dispatches on the family-owned Edge bundle marker.

## Build and runtime

Enable the generic optional SDK with `TRTMC_ENABLE_EDGELLM=ON`, build it on the
executing GPU, and expose its installation through `CMAKE_PREFIX_PATH`.
The installed package must match the pin, SM, CUDA, and TensorRT identity.
The family-owned `trtmc internvl build` command uses the existing CLI protocol.

`edge_llm/builder.py` maps the request into the pinned Python direct builder with
`--components llm,visual`. It preserves both engines, processor/tokenizer
metadata, chat template, and the checkpoint required for external weights.
The family C++ adapter uses the installed Edge runtime API; it does not
implement either model graph.

One public RGB image uses a 448-pixel tile and 256 visual tokens. The visual
profile retains 256 tokens per image and an aggregate token budget derived
from the requested context. That also permits the original Edge two-image
reference workload; it does **not** expand the Model Connect single-image API.
Edge owns preprocessing, normalization, visual inference, and token expansion.

Requests retain the existing public greedy-generation controls and context
limits. Unsupported controls are rejected rather than silently ignored.
Text-only generation and image-bearing generation share a persistent Edge
runtime. Quantization is never inferred from a filename or applied implicitly.

## Recorded full-model qualification

The following are historical local build/inference receipts, not fresh-head
CI claims. All four passed public Model Connect versus direct Edge comparisons
and independent HF image/text comparisons with NED **0** and exact tokens.
Original Edge `vlm_basic` used its two-image fixture and unchanged gates.

| Checkpoint (OpenGVLab) | Source revision | Native SM | Context | Edge ROUGE-1 / ROUGE-L |
| --- | --- | --- | --- | --- |
| InternVL3-1B-hf | `014c0583a0d4bedf29fbe2dbff4f865eb998e171` | 80 | 1024 | 0.5662 / 0.3193 |
| InternVL3-2B-hf | `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` | 80 | 1024 | 0.6275 / 0.3958 |
| InternVL3-8B-hf | `259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | 80 | 1024 | 0.5704 / 0.3654 |
| InternVL3-14B-hf | `e22931943e5336f85e06f4e2b38f3e5e6ee4de3b` | 120 | 1024 | 0.5330 / 0.3296 |

The existing 2B and 8B owning E2Es also passed at their manifest context **384**,
including actual visual-feature health. The 8B golden-reference and 2B HF
comparison criteria were not changed. The 1B and 14B profiles have no owning
registered E2E manifest. TP2/TP4 manifests remain native and are not additional
Edge qualification.

Successful full bundles and source checkpoints were retired after preserving
compact receipts. Replaying full inference requires restoring the exact source
and rebuilding locally. Current source/build checks are not substitutes for
that replay, nor proof of every model in the upstream catalog.

## Existing image-health test

Edge 0.10.1 has no visual-feature dump CLI. The test-only
`internvl_edge_vision_features` executable therefore calls the pinned
`MultimodalRunner` API and copies its actual FP16 output to a temporary file.
It contains no generation driver and is built only when both
`TRTMC_BUILD_TESTS` and the optional Edge package are enabled.

The existing `tests/vision_oracle.py` selects this helper for Edge bundles.
It streams visual/checkpoint sections into temporary storage instead of loading
checkpoint weights into Python memory. Native bundles continue to use their
existing `vision.plan` path. `TRTMC_RUNTIME_ROOT` must point at the build tree
containing the helper and matching plugin.

The owning E2E still requires nonempty, finite, nonzero features and then runs
its existing generation-quality comparison. No thresholds were weakened and no
new test framework was introduced. The standalone cosine contract remains
unchanged; a health assertion alone is not a claim of HF feature parity.

Ordinary builds without an installed optional Edge SDK select native without a
warning. Malformed or incomplete installed packages still retain diagnostics and
warn before native fallback. Temporary checkpoint/engine staging uses the bundle
output directory filesystem (choose a scratch-backed output), not system /tmp.

## Declared build command

This family uses the existing cli.json protocol introduced in #1310. The family
owns its declaration, typed inputs and Python handler. The handler adapts those
inputs to the unchanged builder API, preserving native/Edge dispatch and bundle
publication. The legacy flat build command remains available for its existing
ordinary options; new family options use `trtmc internvl build`.
Help is offline and does not need a local checkpoint. No shared parser hook or
family registry entry is added.

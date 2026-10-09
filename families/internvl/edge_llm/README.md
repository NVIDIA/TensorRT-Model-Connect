# InternVL: optional native Edge-LLM execution

The family owns configuration admission, builder argument mapping, bundle
composition, runtime orchestration, and validation. Edge owns complete visual
and language networks. This route uses the official GitHub Edge-LLM **0.11.0**
snapshot **95515c2f87fba8982db5a519f9022277667b3cc9**, provisioned through the
optional native CMake package. It does not use another checkout or cross-compile.

## Scope

Original-source InternVL3 HF checkpoints and documented group128 AWQ decoders
with the recorded dense Qwen2 text
and 448-pixel vision configurations enter this route. The admitted InternVL3.5
HF Qwen3-backed configurations use the ONNX flow described below; this does not
add standalone Qwen3 support or qualify untested InternVL3.5 checkpoints. Compute is FP16,
batch size and tensor/context parallelism are one. The 1B/2B/8B configurations
route on native x86 SM80; 14B routes on native x86 SM120. The recorded InternVL3-2B AWQ control also runs on SM120. The SM86
InternVL3.5-14B attempt is capacity-blocked, not qualified. Other quantization
formats, multiple-image public requests and speculative decoding are not
qualified by this change.

The existing native builder remains the default for requests outside this
map. If Edge preparation fails, the family logs a warning and retries the
unchanged native request once. Publication errors never retry into a different
backend. Runtime loading dispatches on the family-owned Edge bundle marker.

## Build and runtime

Enable the generic optional SDK with `TRTMC_ENABLE_EDGELLM=ON`, build it on the
executing GPU, and expose its installation through `CMAKE_PREFIX_PATH`.
The installed package must match the pin, SM, CUDA, and TensorRT identity.
The family-owned `trtmc internvl build` command uses the existing CLI protocol.
The builder prefers the active Python interpreter when its installed Edge0.11.0
wheel and TensorRT match the native SDK; otherwise it uses the SDK source-built
Python environment. It never installs dependencies during model builds. The
wheel does not replace the separately provisioned native headers and archives.

`edge_llm/builder.py` keeps flow selection inside the family. Qwen2-backed
InternVL3 uses the pinned direct builder with `--components llm,visual`.
Admitted Qwen3-backed InternVL3.5 uses the official ONNX exporter, `llm_build`,
and `visual_build`: the pinned experimental graph omits its Q/K normalization.
This requires `TRTMC_EDGELLM_ONNX=ON` during SDK provisioning. Missing tools or
export/build errors follow the same warned native fallback, not silent offload.
The selected interpreter must import the complete chosen upstream builder.
Both flows preserve visual/language engines, processor/tokenizer metadata and
chat templates. The experimental flow also preserves the original checkpoint
needed by its external-weight API. ONNX outputs contain their own weight
sidecars; original checkpoint weights are neither staged nor bundled again.
ONNX output that unexpectedly declares checkpoint-backed bindings is rejected.
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

## Recorded 0.11.0 qualification

Nine exact profiles passed installed Model Connect single-image checks in the
local 0.11 rollout, using the unchanged owning semantic/NED criteria:

| Checkpoint (OpenGVLab) | Source weights | Native SM | Recorded result |
| --- | --- | --- | --- |
| InternVL3-1B-hf | Original | 80 | Pass |
| InternVL3-2B-hf | Original | 80 | Pass |
| InternVL3-8B-hf | Original | 80 | Pass |
| InternVL3-14B-hf | Original | 120 | Pass |
| InternVL3_5-1B-HF | Original | 80 | Pass |
| InternVL3_5-2B-HF | Original | 80 | Pass |
| InternVL3_5-4B-HF | Original | 80 | Pass |
| InternVL3_5-8B-HF | Original | 80 | Pass |
| InternVL3-8B-AWQ | Group128 INT4 AWQ decoder / original vision | 80 | Pass |

These prior native Linux x86_64 results use TensorRT 11.1.0.106 / CUDA13.3,
FP16 compute and TP1/batch1. They are not new-PR-head GPU/CI results or proof
of all contexts, images, prompts or devices. Direct Edge and independent
references were compared during the rollout; no 0.10 receipt is promoted to
0.11 evidence. The original 14B and AWQ8 canonical image cases achieved NED0.

All thirteen catalog checkpoint IDs were attempted. AWQ1B/2B/14B execute but
fail image quality; InternVL3.5-14B remains capacity-blocked as detailed below.
Those failures do not invalidate the nine passing profiles or become passes
because the underlying commands exit successfully. Quality-only failures remain
runnable. Native TP2/TP4 manifests are not additional Edge qualification.

Successful heavyweight bundles/checkpoints were retired after preserving compact
receipts. Fresh end-to-end replay requires restoring the exact inputs and
rebuilding; source or syntax checks are not substitutes for that replay.

## Existing image-health test

Edge has no visual-feature dump CLI. The test-only
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

AWQ checkpoints additionally require `TRTMC_EDGELLM_ALL_KERNELS=ON` in the
native SDK configuration. The default ONNX INT4 plugin needs the native
INT4 kernel group; enabling ONNX tools alone does not provide those kernels.

## Failure triage for 0.11.0

Execution availability is not a quality qualification. The recorded AWQ 1B,
2B, and 14B image-answer quality failures remain executable through the ONNX
adapter. Their existing numerical and semantic gates are unchanged; allowing
execution does not mark those tests as passing.

The original InternVL3.5-14B FP16/context384 profile has a recorded ONNX build
allocation failure with TensorRT 11.1.0.106. The family rejects that exact
profile when the device has less total memory than the observed single
49,913,047,296-byte allocation, before launching a known-failing build or its
also-failing native fallback. This is a necessary lower bound, not a sufficient
memory estimate. Larger devices are not rejected simply because a smaller
device of the same architecture failed. The completed SM86 retry also failed,
as recorded below; it is not a qualified profile.

Other unexpected preparation failures retain the existing warned native
fallback. Missing interfaces are not implemented by silently dropping inputs.

The native SM86 48 GiB retry also failed: TensorRT requested a single
52,849,060,096-byte allocation after successful ONNX export. The recorded
FP16/context384 profile is rejected on devices below its observed
SM-specific allocation bound; larger devices are not declared qualified.

### Recorded INT4 backends

The owning `trtmc internvl build` command accepts
`--int4-gemm-plugin-version {1,2}`. Omission retains Edge 0.11's version2
default. Version1 is mapped for the recorded InternVL3-2B AWQ SM80 control;
the same checkpoint's version2 control is also executable on SM120.
These controls failed the unchanged image-quality gate, not build/inference.
Selecting a backend never waives that gate or qualifies a new profile.
The option is not silently discarded into a native/non-AWQ fallback.

The same original 3.5-14B FP16/context384 request is not retried natively on
the available 40/48 GiB SM80/SM86 targets when the SDK is absent. Its native
TP1 factory keeps separate roughly 29.5 GB prefill/decode plans resident;
the recorded 40 GiB run failed loading the second engine. 48 GiB is still below
this pair's estimated footprint, not a claimed native 48 GiB execution result.
Larger devices remain unqualified, not blacklisted.

# Muse-Glimmer with Edge-LLM

The `muse_glimmer` family owns complete-network offload, argument mapping,
engine assets, runtime orchestration, and validation. Edge 0.11 remains the
default SDK during migration. To select 0.12, configure Model Connect with
`-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.12.0`.

## Edge 0.12

Muse needs both `-DTRTMC_EDGELLM_ONNX=ON` and
`-DTRTMC_EDGELLM_ALL_KERNELS=ON`. Follow the
[SDK provisioning guide](../../../cmake/edge_llm/README.md) to build on the
target GPU. If CMake finds an external SDK, point `TRTMC_EDGELLM_SOURCE_DIR`
at the populated, pinned Edge checkout: the upstream binary SDK does not
attest which kernel groups it contains.

A packaged runner uses its own SDK. For a standalone runner, set
`CMAKE_PREFIX_PATH` to the Model Connect SDK installation containing
`share/trtmc/edge-llm.json`. Use an isolated Python environment
with the matching 0.12 exporter (wheel or source installation), Transformers
5.17.0 and the same TensorRT version as the SDK. Pass its absolute interpreter
path to the family; the wheel alone does not provide the native runtime. For example:

```sh
trtmc muse_glimmer build /models/muse-nvfp4 \
  --precision fp16 --quantization nvfp4 --max-sequence-length 1024 \
  --edge-builder-python /opt/edge-exporter/bin/python -o muse.bundle
```

The optional 0.12 route uses the released ONNX exporter and C++ builders on the
target device. Its architecture and CUDA/TensorRT versions must match the
installed SDK. The 0.11 SM120 offload restriction does not apply to 0.12.
A builder failure emits a warning and retries the original request with the
family's native TensorRT implementation. Neither a change in precision nor
dropping the draft is a valid fallback.

The prepared NVFP4 checkpoint is **text-only**. Ordinary decoding and both
documented DFlash pairs have run through the 0.12 Edge backend. This is not
a lossless speculative-decoding or accuracy claim. In particular the released
runner can return different greedy outputs with and without speculation on
the same V1 or V2 paired engine. Thinking mode also affects the output boundary:
with thinking disabled, the chat template supplies the public-channel header
in the prompt, not the generated tokens. Both routes omit checkpoint-declared
special tokens from display text while retaining the actual generated IDs.
The scientific terms and generation budget are the same in both modes.

For original floating checkpoints, request `--precision fp16 --quantization none`;
the adapter uses original weights. Building a typed image/video task requires the
original processor and visual weights. For example:

```sh
trtmc muse_glimmer build /models/Muse-Glimmer-30B \
  --task images_text_to_text --precision fp16 --quantization none \
  --max-sequence-length 1024 --edge-builder-python /opt/edge-exporter/bin/python \
  -o muse-media.bundle
```

The six documented combinations below built and ran through both Model Connect
routes at batch 1 and TP1. The results are for the tested checkpoints, prompts
and profile, not for every GPU, prompt, or sampling mode.

| Checkpoint | Decoding | Quality observed |
| --- | --- | --- |
| Original FP16 | Autoregressive | Both routes gave factual answers in both thinking modes, distinguished opposed images and both video orders, and matched token IDs on all six requests. |
| Original FP16 | DFlash V1 and V2, each with its own companion | Both routes passed the same four media checks and both factual text checks for each pair. On each backend and pair, the six greedy outputs matched target-only execution of that same engine token for token. |
| Prepared NVFP4 (text only) | Autoregressive | Both routes returned factual answers in both thinking modes. No media weights are present. |
| Prepared NVFP4 (text only) | DFlash V1 | Native matched its own target in both thinking modes. In the released Edge runner, speculative and target-only outputs differed; with thinking enabled both reached the token cap. The Model Connect Edge run returned a factual answer, so the capped result is specific to the direct control. |
| Prepared NVFP4 (text only) | DFlash V2 | Native matched its own target in both modes. The released Edge runner returned different greedy answers with and without speculation. Model Connect Edge matched the released speculative answers in both modes. |

The scientific-answer test runs both modes on one bundle. With thinking off,
it checks a direct answer without a generated private turn. With thinking on,
it checks the answer after the generated public-channel delimiter. Terms found
only in reasoning do not count. Separately, both the native and the released FP16 visual encoders exceeded the pointwise
tolerance against the eager reference on identical inputs. Correct media answers
do not establish encoder tensor parity. These quality limitations do not disable
the corresponding build or inference commands.

### DFlash2 tree decoding

Use the original FP16 or prepared NVFP4 target with its matched
`incoai/Muse-Glimmer-30B-DFlash2` companion. For example:

```sh
trtmc muse_glimmer build /models/muse-nvfp4 \
  --precision fp16 --quantization nvfp4 --max-sequence-length 1024 \
  --execution-variant dflash_tree --companion /models/muse-dflash2 \
  --edge-builder-python /opt/edge-exporter/bin/python -o muse-tree.bundle
```

The 0.12 Edge builder sets a 32-node verification profile on the target
and keeps the draft's block size at 16. At runtime, the tree retains up to four
children per node. Native TensorRT uses the same conditional selector and
verifies tokens against the target along the accepted path; it does not batch
all nodes into one target call. Both routes reject stochastic generation for an
explicit tree. Use `dflash` for the chain. The chain results above do not
establish tree accuracy.

On the original FP16 target, the tree built and ran on both routes. Both
passed the thinking-enabled science check and distinguished the two images
and both video orders. With thinking disabled, both gave a factual answer but
did not generate the header that the earlier test expected. Native matched target-only token IDs on all
six requests; Edge matched on five. On the red-then-blue video, the Edge tree
separated the two correct colors with a comma, while target-only used a
newline. The direct Edge runner also produced different text with speculation
enabled and disabled on the same engines for the two science requests; both
settings reached the 512-token cap. These observations do not establish the
cause of the difference.

For the prepared NVFP4 target, both tree routes built and generated answers
with thinking disabled and enabled. Native returned the same 38 and 171
token IDs as its own target-only engine, respectively. Both routes missed
the earlier test's generated-header check with thinking disabled; the template
already supplies that header. With thinking
enabled, native passed; the Edge answer said “solid water” rather than either
term accepted by the test (“ice” or “crystal”). The released Edge runner also
produced different text with tree verification enabled versus disabled on the
same engine and prompt. These are accuracy limitations, not command failures.

## Edge 0.11 dependency and execution

Provision the native Edge SDK using Model Connect's optional CMake dependency
with the ONNX tools and complete native kernels. Expose its installation
through `CMAKE_PREFIX_PATH`. The family prefers a compatible installed
0.11.0 Python wheel exporter and otherwise uses the SDK's provisioned Python.
The wheel does not replace the native C++ SDK.

Build on the target GPU. The Edge offload for the admitted mixed NVFP4/MXFP8
checkpoint requires SM120 and a matching CUDA/TensorRT stack. This restriction
belongs to that offload, not the native mathematical graph. Native execution
has no SM allowlist: TensorRT must support its packed-format operators and the
device must have sufficient memory. Neither route cross-compiles, recalibrates,
or replaces the source precision recipe.

## Documented prepared text configurations

The base is `RadixArk/Muse-Glimmer-NVFP4`, revision
`1416629950afdfb276618fd4810103681dd10f4a`.

| Execution | Companion |
| --- | --- |
| Autoregressive | None |
| DFlash V1 | `meta-models/Muse-Glimmer-30B-assistant` |
| DFlash V2 | `incoai/Muse-Glimmer-30B-DFlash2` |

Both companions use the documented public `dflash` mode and block size 16;
their checkpoint architecture selects the version. Family-owned arguments are
declared through the existing CLI protocol:

```sh
trtmc muse_glimmer build /models/muse-nvfp4 \
  -o muse.bundle --precision fp16 --quantization nvfp4 \
  --max-sequence-length 1024

trtmc muse_glimmer build /models/muse-nvfp4 \
  -o muse-paired.bundle --precision fp16 --quantization nvfp4 \
  --max-sequence-length 1024 --execution-variant dflash \
  --companion /models/matched-muse-draft
```

The documented text examples use a profile of TP1/batch1/input512/KV1024, with a full block size 16
verification profile for paired execution. Both base and draft must build;
a failed pair is never replaced by an autoregressive bundle. DFlash V1 requires
greedy generation. DFlash V2 does not add the V1 greedy-only restriction; local paired
qualification uses greedy generation only. Unsupported controls, such as a
seed override, remain explicit errors in the Edge route. The native runtime
supports its seeded sampler and rejects unsupported controls before execution.
The 0.11 Edge offload does not support DDTree.

## Legacy 0.11 quality and limitations

The existing owning E2E retains its scientific prompt, low-reasoning system
message, 512-token generation budget, and final-answer semantic checks. Muse's
thinking-enabled ATEM output must reach `assistant to=user`; a response
that stops in the reasoning channel is not a passing answer.

The 0.11 validation did not run the original FP16 checkpoint or its paired
modes: the compatible test device had insufficient memory. This describes
that validation environment, not a general GPU requirement. The documented NVFP4
checkpoint is text-only: its results do not qualify image or video execution.

Selection and numerical qualification are separate. Accuracy-only failures
must remain executable. Preparation failures retain diagnostic logs and warn
before attempting the unchanged native fallback. Ordinary native execution uses
this family's TensorRT Python graph and Model Connect C++ task; its full-model
qualification is separate from the recorded Edge results. The graph preserves
mixed NVFP4/MXFP8 weights and activations rather than substituting dense precision.
Native DFlash V1/V2 build real block-16 draft engines, inherit the original
packed target embedding/head, and collect the declared target feature layers.
V2 additionally runs its conditional candidate selector. The family C++ runtime
verifies the proposal prefix with the target one token at a time, applies exact
p/q rejection correction for sampled V2 requests, and commits no rejected draft
state. This is a correctness-first implementation, not a speculative speedup
claim. V1 retains its greedy-only contract.

### Legacy native validation scope

All three admitted text modes have completed full-checkpoint native build and
inference. The existing owning scientific-answer check passes for ordinary
execution and DFlash V2. The V1 output fails that unchanged semantic check,
although its output is token-identical to ordinary execution of the exact same
compiled target engine. This is a recorded target-output quality limitation,
not evidence of a draft-verification discrepancy. V1 remains runnable; command
success is not advertised as an accuracy pass.

These results cover the recorded request and profile, not every prompt or
platform. Small paired-runtime checks additionally cover seeded V2 sampling;
full-checkpoint sampled quality has not been qualified. Native graph construction
uses TensorRT's packed-format operators: operator support and sufficient memory
are still required even though the native family has no GPU allowlist.
Insufficient device memory is a runtime failure, not permission to silently
change precision or remove a companion. Inference failures are returned to the
caller and are never silently retried.

For raw media requests, pass typed image or video parts rather than inserting
special tokens into prompt text. The Edge adapter formats those parts with the
checkpoint's placeholders. If the processor's resize fallback exceeds the
compiled native visual profile, the runtime reports the limit instead of
silently shrinking the image to a single token.

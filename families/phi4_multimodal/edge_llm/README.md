# Phi4 Multimodal with Edge-LLM

When the installed Model Connect wheel includes a native runner and Edge SDK,
the family uses that SDK. A different SDK on `CMAKE_PREFIX_PATH` cannot
override the wheel's native runner. To select another release, install a
Model Connect wheel built for it, or use a matching standalone Model Connect
builder and runner in an environment without the wheel. For standalone
installations, `CMAKE_PREFIX_PATH` selects the SDK as usual.

## Edge 0.12: original checkpoint

Use an MC installation built with `TRTMC_ENABLE_EDGELLM=ON` and
`TRTMC_EDGELLM_RELEASE=0.12.0`. 0.12.0 is the default when Edge is enabled.
The native C++ SDK must match the worker's GPU architecture, CUDA and TensorRT.
The Python builder can come from the official wheel or a source installation.
When it lives outside MC's environment, pass its interpreter explicitly:

```bash
trtmc phi4_multimodal build "$MODEL_DIR" -o phi4.bundle --precision fp16 \
  --max-sequence-length 10368 --edge-builder-python "$EDGE_PYTHON"
trtmc run phi4.bundle --image "$IMAGE" --prompt "Describe this image." \
  --max-new-tokens 30 --temperature 1 --top-k 1
```

For the original Microsoft checkpoint the family builds the LLM and visual
components together. It preserves the vision LoRA and external weight bindings.
The owner image case at batch one, TP1, context 10368 passed on SM80: the
existing finite/nonzero feature check passed and the generated answer had
normalized edit distance 0.1241 against the independent FP32 reference
(limit 0.15). The warned native fallback passed the same feature check and
matched all 30 FP32 reference tokens (NED 0). This does not qualify the old 768-token profile on 0.12.

Text-only inference on the same vision-language bundle also executed. In a
separate eight-token raw completion, both Edge and HF FP32 began with the
correct city, but the continuations differed (NED 0.6757, above 0.15). That
request remains executable; the image result does not qualify standalone text.
Native also missed the language-only reference (NED 0.6486). Forcing the
vision LoRA in HF reproduced all eight native tokens but did not reproduce
Edge's continuation. Neither comparison changes the image gate.

The release catalog also lists audio, but the 0.12 direct builder rejects
`--components audio` for this checkpoint: only `llm` and `visual` are
available. Phi4 does not implement an MC audio-text task or a native audio
encoder. Neither audio inference nor audio quality is claimed. The prepared
checkpoints below describe the legacy route, not new 0.12 qualifications.

## Edge 0.11: historical profiles

This family owns Edge selection, builder arguments, engine assets, runtime
orchestration, and validation. Shared model-selection code is unchanged.
The integration targets the official public Edge revision
`95515c2f87fba8982db5a519f9022277667b3cc9`.

## Installation

For these 0.11 profiles, configure a separate Model Connect build directory with
`-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.11.0` and build there.
Expose the matching SDK install prefix through `CMAKE_PREFIX_PATH`. A compatible
`tensorrt-edgellm==0.11.0` Python wheel in the active interpreter is preferred
for Python builder/export commands. Otherwise, the source-provisioned SDK
interpreter is used. The wheel does not replace the native SDK.

Build engines natively on the target GPU; the adapter does not cross-compile.
The SDK and runtime must match the bundle's CUDA, TensorRT and GPU architecture.

## Documented model and routes

The checkpoint is `microsoft/Phi-4-multimodal-instruct`. Edge's documented
Phi4 recipe merges the vision LoRA before export. The original-checkpoint
experimental builder applies that vision adapter; the ONNX route consumes an
already merged and quantized checkpoint. Neither route claims to preserve the
separate unmerged Hugging Face LANGUAGE-mode weights.

| Checkpoint weights | Edge route | Native target | Build request |
| --- | --- | --- | --- |
| Original FP16 with vision LoRA | Experimental builder | SM80 or SM120 | FP16, TP1, batch 1 |
| ModelOpt FP8 | ONNX | SM120 | FP16 execution, TP1, batch 2, context 8192 |
| ModelOpt NVFP4, with or without NVFP4 LM head | ONNX | SM120 | FP16 execution, TP1, batch 2, context 8192 |
| ModelOpt W4A16-AWQ | ONNX, ALL_KERNELS SDK | SM80 | FP16 execution, TP1, batch 2, context 8192 |

Use the existing `--precision fp16`, `--max-sequence-length`,
`--max-batch-size`, and `--tensor-parallel-size` build arguments.
Packed checkpoints must retain matching ModelOpt metadata; the adapter neither
calibrates nor changes packed weights. The quantized LM-head route avoids
unsupported externalized LM-head export.

Original-checkpoint image capacity is bounded by context, with a maximum of
1280 tokens per image and a 256-token reserve for separators and prompt tokens.
Packed routes use input 7168 / KV 8192 and visual 256–6400, per image 1280 profiles.
The runtime also validates actual expanded input plus the original generation
budget; arbitrary prompts and images are not guaranteed to fit.

## Runtime and quality boundaries

The family implements image-to-text and no-image text requests through the
same vision-adapted engine. Image input uses checkpoint chat framing; no-image
input preserves the family's raw-prompt behavior. Both checkpoint stop tokens
199999 and 200020 are retained. Audio is listed in Edge's model catalog, but
the pinned builder lacks a Phi4 audio component and this Model Connect family
has no audio task adapter. Audio is not exposed by this integration.

The existing image validation retains its fixture, prompt, 30-token budget,
FP32 reference, actual-feature health check, and NED limit of 0.15. Recorded
FP16 context 8192 text output meets that limit. The context 768 and context 2048
runs and the four packed-weight configurations execute but fail that accuracy
gate. They remain runnable for investigation; accuracy outcomes never control
dispatch. A larger-context pass does not qualify the unchanged context 768 test.

If Edge preparation fails, diagnostics are retained and the family warns before
retrying its native builder with the unchanged request. An incomplete Edge
bundle is never published. Runtime inference errors are returned to the caller
and never silently retried.

Contexts below 768 tokens (including the default 256) use the native route
without invoking Edge. Each Edge export/build subprocess has a 12-hour limit;
a timeout retains diagnostics and follows the same native fallback path.
Successful publication removes the temporary diagnostic log; failures retain it.


## Native fallback for prepared checkpoints

The family native builder reads the original vision-LoRA checkpoint and the
already merged ModelOpt FP8, NVFP4, and W4A16-AWQ checkpoints. It retains the
source activation and weight scales, AWQ input smoothing, fused QKV/gate-up row
boundaries, and a packed LM head when present. It does not calibrate, reinterpret
packed bytes as floating-point weights, or silently use an unquantized target.

The fallback constructs its graph with TensorRT Python APIs and executes with
this family's native C++ runtime; it does not use ONNX or Edge kernels. FP8 and
NVFP4 use native TensorRT Q/DQ. AWQ uses INT4 dequantization with FP16 source values
and FP32 matrix-product accumulation. The family does not select native graph
semantics by GPU. TensorRT still must support the requested operators on the
target, and an unsupported operator is a build failure, not a model ban.

Both task adapters expose one request per call. The prepared Edge recipe's
`--max-batch-size 2` is an engine-profile ceiling, not a public batched-generation
API. Native fallback executes a single request within that ceiling and records
both `max_batch_size` and `execution_batch_size` in its runtime metadata; it warns
when these differ. It does not claim two-request vectorized execution.
Checkpoint generation metadata supplies every native EOS token. An explicit
native request EOS override replaces the checkpoint defaults.

The full-checkpoint measurements below predate the later Dynamic-HD and
LongRoPE changes now integrated from `main`. They are historical validation
evidence, not a fresh qualification of that combined implementation.

The original FP16 vision-LoRA checkpoint also passes native build, MC C++
image inference, and actual-feature health checks on SM80 with TensorRT 11.1,
TP1, batch ceiling 1, and the existing 768-token owning profile. Its 30-token
image answer matches the retained independent FP32 reference exactly (NED 0).
This native result does not change the separate Edge accuracy outcomes above.

Full prepared native builds and MC C++ image inference were exercised on SM120
with TensorRT 11.1, FP16 compute, TP1, context 8192, and the batch ceiling of 2
described above. Actual native visual features pass the existing finite/nonzero
health check. The existing image prompt, 30-token budget, independent FP32
reference, and NED <=0.15 gate are unchanged.

| Prepared source policy | Build / inference | Visual health | Text NED | Outcome |
| --- | --- | --- | --- | --- |
| FP8 | Pass | Pass | 0.192593 | Accuracy failure; runnable |
| NVFP4, FP16 LM head | Pass | Pass | 0.392593 | Accuracy failure; runnable |
| NVFP4, NVFP4 LM head | Pass | Pass | 0.355072 | Accuracy failure; runnable |
| W4A16-AWQ | Pass | Pass | 0.355556 | Accuracy failure; runnable |

These are actual full-model results, not extrapolations from the packed-weight,
projection, or small-decoder checks. No prepared-model image-quality pass is claimed.
FP8 ends at a checkpoint EOS after 25 tokens, while the FP32 reference reaches
the 30-token budget; the original text gate still fails. No output-budget, EOS,
or acceptance-criteria change is used to turn a failure into a pass. Other
capacities and platforms are not qualified by these results, and the recorded
Edge accuracy failures remain separate from these native outcomes.

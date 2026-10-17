# Qwen3.5: optional Edge-LLM 0.11 execution

This family owns Edge selection, checkpoint admission, build arguments, bundles,
runtime orchestration and validation. Shared Model Connect code is unchanged.
The native SDK is pinned to the official public Edge-LLM **0.11.0** revision
`95515c2f87fba8982db5a519f9022277667b3cc9`. Build on the executing GPU; there is
no cross-compilation or alternate vendor checkout.

## Installation and ownership

Enable `TRTMC_ENABLE_EDGELLM=ON` during native CMake provisioning and expose the
installed SDK through `CMAKE_PREFIX_PATH`. The adapter verifies the release,
architecture, GPU target, CUDA and TensorRT identities before offloading.

Compatible installed Python tools, including `tensorrt-edgellm==0.11.0` from
PyPI, are discovered automatically. Otherwise the adapter uses the SDK's private
source-installed Python. The wheel does not replace the native C++ SDK headers
and archives. There is no model-time installation or multi-version provider.

`edge_llm/builder.py` calls the official Python builder
(`experimental.builder.cli`); `runtime/edge_llm/` calls the native Edge runtime.
Edge owns the complete network, quantized operations and speculative execution.
The family does not implement replacement kernels or modify vendor sources.

## Execution scope

All Edge routes require FP16 compute, batch one, CP one and a matching native
Linux x86 SDK. Original sources use SM80 or SM120; admitted ModelOpt formats
use SM120. Checkpoint-driven weight precision is separate from compute precision.

| Configuration | Owning route |
| --- | --- |
| Dense 0.8B, 2B, 4B and 9B, Instruct and Base | Original-weight text and image generation, TP1 |
| Dense 27B | Original-weight text/image, TP2 LLM plus TP1 visual component |
| Embedded MTP, original 0.8B/2B/4B/9B topologies | Text/image, TP1; three draft steps |
| Embedded MTP, ModelOpt NVFP4 0.8B/2B/4B/9B topologies | Text/image, TP1; local model execution evidence is 4B only |
| Listed 4B/9B DFlash companions | Original matched target/draft; linear or DDTree, block16 |
| Listed 4B NVFP4 DFlash pair | Matched official target/draft exports, linear or DDTree |
| Dense standalone ModelOpt targets | NVFP4, FP8, MXFP8, INT4-AWQ group128, INT8 SmoothQuant; TP1 |
| 2B FP8 weights with calibrated FP8 KV | Standalone text only, TP1 |
| 2B NVFP4 weights with NVFP4 LM head | Standalone text, checkpoint-driven component precision |

This table describes execution admission, **not universal quality qualification**.
NVFP4 MTP uses the same four topology admissions as original-weight MTP. Only
the 4B NVFP4 profile has local model-engine evidence; the other NVFP4 sizes are
command mappings, not locally executed or accuracy-qualified results.
Results belong to individual checkpoints, profiles and requests. Accuracy-only
failures remain executable; they do not become passing results.

Qwen3 and older, MoE, Qwen3.6/3.8, unknown quantization metadata,
unsupported component combinations, and speculative TP2 are outside this owner.
The separate Qwen owners must establish their own support.

## Build and run

The existing family CLI protocol owns these arguments. No shared parser hook,
global variant option or core model dispatch is added.

```bash
trtmc qwen3_5 build /models/Qwen3.5-2B \
  --precision fp16 --max-sequence-length 256 -o qwen35.bundle
trtmc run qwen35.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

Image bundles use `--task vision_language_generation`. The native request
accepts `--image /path/to/image` or an ordered JSON `--images` list, not both.
A vision tower and source processor metadata are required. The family preserves
the provider chat template and delegates image preprocessing to Edge.

```bash
trtmc qwen3_5 build /models/Qwen3.5-4B \
  --precision fp16 --task vision_language_generation \
  --max-sequence-length 8192 -o qwen35-image.bundle
trtmc run qwen35-image.bundle --image /images/dog.jpg \
  --prompt "What animal is shown?" --max-new-tokens 16 \
  --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

The source image-token profile is 128–1280 per image, bounded by the aggregate
visual capacity. Image examples do not qualify arbitrary resolutions.

### Typed video requests

Visual bundles also expose the existing native `VideoTextToText` and
`ImageVideoTextToText` APIs. Pass decoded, ordered RGB frames in `VideoView`;
frames must share dimensions. Optional timestamps must be finite, nondecreasing
and match the frame count. Without timestamps, the owning Edge mapping uses
one frame per second. The adapter preserves video identity and mixed-content
order rather than converting a video into independent image requests.

Build the same `vision_language_generation` bundle shown above. Use the typed
C++/C API for video input; the generic `trtmc run --image` option is not a video
interface, and no shared CLI option is added. Text-only bundles do not expose
the video tasks. Edge retains responsibility for preprocessing, temporal patches
and the provider template; existing visual and context capacity limits apply.

Local evidence: original Qwen3.5-2B, FP16, TP1, context8192, two identical frames
from the existing car fixture. Model Connect and direct Edge returned identical
text, `white`; independent FP32 HF comparison passed the unchanged NED0.15
gate with NED0.0. This validates video transport and appearance, not motion
understanding, every checkpoint, or speculative video accuracy. Native owning
regressions additionally check distinct-frame ordering, mixed image/video
messages, timestamp preservation and malformed-input rejection.

### Speculative execution

```bash
trtmc qwen3_5 build /models/Qwen3.5-4B --precision fp16 \
  --execution-variant dflash --companion draft=/models/Qwen3.5-4B-DFlash \
  --max-sequence-length 2048 -o qwen35-dflash.bundle

trtmc qwen3_5 build /models/Qwen3.5-4B --precision fp16 \
  --execution-variant dflash_ddtree --companion draft=/models/Qwen3.5-4B-DFlash \
  --max-sequence-length 2048 -o qwen35-ddtree.bundle

trtmc qwen3_5 build /models/Qwen3.5-2B --precision fp16 \
  --execution-variant mtp --max-sequence-length 2048 -o qwen35-mtp.bundle
```

MTP reads the embedded draft layer and rejects `--companion`. DFlash checks the
draft architecture, vocabulary/hidden widths, target layers, mask token, context
and matching weight format. Linear uses top-k one; DDTree uses top-k four.
Text speculative builds cap input at min(context, 1024) while retaining the
requested KV capacity. Runtime sampling qualification remains greedy.

Python callers can use the existing family extension
`with_execution(request, BuildExecutionInputs(...))` from
`families.qwen3_5.edge_llm.config`. CLI help works offline.

### Tensor parallelism

```bash
CUDA_VISIBLE_DEVICES=0,1 trtmc qwen3_5 build /models/Qwen3.5-27B \
  --precision fp16 --tensor-parallel-size 2 --max-sequence-length 4096 \
  -o qwen35-27b.bundle
CUDA_VISIBLE_DEVICES=0,1 trtmc run qwen35-27b.bundle \
  --prompt "The capital of France is" --max-new-tokens 16 \
  --temperature 0 --top-k 1 --top-p 1 --use-chat-template false
```

Both devices must match the bundle target. A compatible NCCL runtime must be
discoverable through loader paths or `EDGELLM_NCCL_SO_PATH`. The family delegates
rank orchestration to Edge's `ParallelExecutionConfig`; it does not launch a
second Model Connect rank coordinator.

TP2 visual builds invoke the LLM and visual components separately because the
pinned combined builder does not accept that composition. Runtime receives both.
Speculation plus TP2 is not admitted. The 27B single-device DFlash pair exceeded
the compatible capacity used for validation; ordinary TP2 results do not qualify it.

### Quantized checkpoints

Use official Edge 0.11 tools and their declared dependencies, including
Transformers 5.14.1, to prepare quantized sources. The adapter consumes the
resulting checkpoint; it does not calibrate, repack or silently dequantize it.
Embedded and sidecar algorithms must agree. The builder uses `--dense auto`
for quantized sources, preserving projection exclusions and component metadata.

For FP8 KV, both the sidecar FP8 algorithm and the embedded static eight-bit
float KV scheme are required. Only the verified 2B standalone text combination
is admitted. The official draft exporter may declare NVFP4 only in its ModelOpt
sidecar; this specific draft schema is also supported.

## Validation boundaries and known quality issues

Release-specific local validation covers the named original dense checkpoints,
recorded image requests, MTP/DFlash modes and tracked standalone quantization
profiles. It includes actual Model Connect builds/inference, direct Edge controls,
existing owning comparisons and selected installed-package replays. Older
0.10 results are not evidence for this release.

The existing short-reference normalized edit-distance gate remains **0.15**.
The public 128-token basic fixture retains ROUGE-1 **0.25** and ROUGE-L **0.20**.
Generated token IDs must be present, valid and consistent with returned text.
Numeric scores alone do not certify factual correctness.

- Original small-model text profiles and original 4B/9B speculative text
  profiles pass their recorded short comparisons. Some termination-token
  suffixes differ from Hugging Face; decoded agreement is not exact EOS parity.
- Original 27B TP2 raw text and short single-image requests pass. Its longer
  two-image request remains unqualified: Edge misses one image's subject and
  the required distributed FP32 reference also produced invalid repetitive text.
  A different-precision diagnostic is not a substitute for that reference.
- Quantized and some Base multi-image profiles fail independent-reference NED
  despite matching direct Edge. Their execution remains available.
- Several small-model quantized long answers contain incorrect biographies,
  founders or dates despite passing numeric gates. The 2B FP8-KV direct output
  also invents NVIDIA's founders. An original unquantized 2B HF control has
  related factual errors, so this category is not established as Edge-only.
- The 0.8B INT4-AWQ short answer fails NED in both experimental and ONNX direct
  controls. Some 4B INT4-AWQ/INT8-SQ long outputs also fail semantic review.
  ONNX controls are diagnostics, not an integrated Model Connect ONNX route.

These observations are not an exhaustive benchmark, stochastic-generation,
full-context or cross-platform qualification. Existing tests and thresholds
are unchanged; the family does not skip a configuration because its quality
comparison fails.

## Artifacts and failure behavior

The bundle includes the required engines, tokenizer/template metadata and
external-weight dependencies. TP2 retains the original shards named by Edge's
runtime identity, not another copy of every baked projection. Keep sources
immutable during building and budget for bundle staging and runtime extraction.

Ordinary requests without an optional SDK remain native without a warning.
Incompatible SDKs and Edge preparation failures preserve diagnostics, warn and
retry the unchanged native request once. A requested speculative mode must
never silently become a base-only bundle. Runtime failures do not switch backends.
Use scratch-backed output and `TMPDIR` paths for large artifacts.

Run the existing owning checks with:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/qwen3_5/tests -q
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/qwen3_5/tests/test_e2e.py --e2e-testcase qwen35-2b -q
```

Selected E2Es require a checkpoint and `TRTMC_BINARY`/`TRTMC_RUNTIME_ROOT`
pointing to the native runtime built for the executing GPU. Unselected E2E
skips are not validation evidence.

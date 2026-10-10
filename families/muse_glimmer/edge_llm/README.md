# Muse-Glimmer with Edge-LLM 0.11

The `muse_glimmer` family owns complete-network offload, argument mapping,
engine assets, runtime orchestration, and validation. It uses the official
Edge 0.11.0 snapshot `95515c2f87fba8982db5a519f9022277667b3cc9`; no shared-core
model selection is added.

## Dependency and execution

Provision the native Edge SDK using Model Connect's optional CMake dependency
with the ONNX tools and complete native kernels. Expose its installation
through `CMAKE_PREFIX_PATH`. The family prefers a compatible installed
0.11.0 Python wheel exporter and otherwise uses the SDK's provisioned Python.
The wheel does not replace the native C++ SDK.

Build on the target GPU. The admitted mixed NVFP4/MXFP8 checkpoint requires
SM120 and a matching CUDA/TensorRT stack; the adapter does not cross-compile,
recalibrate, or reinterpret packed source weights.

## Documented text configurations

The base is `RadixArk/Muse-Glimmer-NVFP4`, revision
`1416629950afdfb276618fd4810103681dd10f4a`.

| Execution | Companion |
| --- | --- |
| Autoregressive | None |
| DFlash V1 | `meta-models/Muse-Glimmer-30B-assistant` |
| DFlash V2 | `incoai/Muse-Glimmer-30B-DFlash2` |

Both companions use the documented public `dflash` mode and block size16;
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

The recorded profile is TP1/batch1/input512/KV1024, with a full block16
verification profile for paired execution. Both base and draft must build;
a failed pair is never replaced by an autoregressive bundle. DFlash V1 requires
greedy generation. DFlash V2 does not add the V1 greedy-only restriction; local paired
qualification uses greedy generation only. Unsupported controls, such as a
seed override, remain explicit errors. DDTree is not exposed.

## Quality and limitations

The existing owning E2E retains its scientific prompt, low-reasoning system
message, 512-token generation budget, and final-answer semantic checks. Muse's
ATEM output must reach `assistant to=user`; a response that stops in the
reasoning channel is not a passing answer.

The original `meta-models/Muse-Glimmer-30B` FP16 text/image/video checkpoint
exceeds the available compatible single-device capacity. It and its FP16
paired modes have no executed qualification or adapter route here. No
unqualified tensor-parallel alternative is implied. The documented NVFP4
checkpoint is text-only: its results do not qualify image or video execution.

Selection and numerical qualification are separate. Accuracy-only failures
must remain executable. Preparation failures retain diagnostic logs and warn
before attempting the unchanged native fallback; because this owner currently
has no native graph, that fallback reports an explicit unsupported error.
Inference failures are returned to the caller and are never silently retried.

# Qwen3.8 DSpark Edge-LLM adapter

The Qwen3.8 family owns its configuration admission, ONNX command mapping,
bundle assets and C++ runtime orchestration. It does not reuse another Qwen
family. Standalone builds retain the original native path; this change admits
Edge only for an explicit mixed-NVFP4 target plus DSpark companion.

## Build and inference

Provision the optional native SDK with `TRTMC_EDGELLM_ALL_KERNELS=ON` and
`TRTMC_EDGELLM_ONNX=ON` using the
[pinned package instructions](../../../cmake/edge_llm/README.md). The source is
GitHub Edge-LLM 0.10.1 at `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.
Configure `CMAKE_PREFIX_PATH` for the installed package and compile the runtime
with `TRTMC_ENABLE_EDGELLM=ON`. Cross compilation is unsupported.

The Python build API accepts `BuildExecutionInputs(variant="dspark",
checkpoints=(NamedCheckpoint("draft", draft_path),))` on a family-owned typed request. The CLI equivalent adds `--execution-variant dspark` and
`--companion draft=/path/to/draft` to an ordinary build invocation.
The family invokes the original Edge Python ONNX exporter and native
`edgellm-onnx-build`; it does not alter source tensors or pad safetensors headers.
The exporter resolves the draft LM head from the target checkpoint. Both
speculative engines, embedding/head sidecars and tokenizer assets are bundled;
checkpoint weights are not duplicated in the bundle.

The runtime calls the original Edge speculative inference constructor with
proposal block 7, verify 8, drafting topK 1/step 1 and DSpark scheduling disabled.
It preserves supported sampling controls. Engine preparation errors warn and
try native once with the same requested execution variant; native currently
rejects the unmapped DSpark variant explicitly. No failure substitutes an
ordinary base-only decoder. Inference errors propagate without fallback.

## Validated exact profile

- Target: `RadixArk/Qwen3.8-27B-NVFP4`, revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Draft: `RadixArk/Qwen3.8-27B-DSpark`, revision
  `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
- Native SM120, CUDA 13.3, TensorRT 11.1.0.106, FP16 execution with source mixed
  NVFP4/FP8 metadata retained, TP 1/batch 1, input/KV capacity 1024.
- Actual Model Connect paired build and public CLI inference passed.
- Independent greedy oracle: exact token match, NED 0.0 against 0.15.
  The saved CPU FP32 reference was reused after checkpoint-byte and reference
  function verification; it was not regenerated during this run.
- Original Edge `llm_basic` prompt and 128-token sampled profile
  (temperature 1/topK 50/topP 1): ROUGE-1 **0.4246**, ROUGE-L **0.2458**, above
  unchanged **0.25/0.20** gates. Chat enabled and thinking disabled.
- Publication regressions: 11 existing family Python tests plus 106 existing
  builder/architecture tests pass; both existing native C++ tests pass.

These results qualify only the exact text profile, not standalone NVFP4,
multimodal inputs, other checkpoints/platforms or statistical sampling parity.
The local run reused existing owning E2E helpers with an explicit companion;
this pair is not yet a registered pytest manifest case.

A first inference attempt exposed incompatible development JSON headers sharing
Edge’s 3.12.0 version label. Matching the exact pinned headers fixed the crash
without changing Edge or the engines. The generic SDK now checks header content
rather than relying only on the version label.

## Family-owned build options

The existing family CLI reads this owner's cli.json and invokes cli.py.
Edge-specific inputs and selection remain in edge_llm/; the shared parser,
CLI protocol and build API gain no new options or hooks.
All variant validation and builder selection remain in this family.

```sh
trtmc qwen3_8 build /path/to/target --precision fp16 \
  --execution-variant dspark --companion draft=/path/to/draft \
  -o model.bundle
```

Options may precede or follow MODEL. `trtmc qwen3_8 build /path/to/target --help`
shows these family options using local metadata; remote-ID help does not download
a checkpoint. For Python callers, use this family's request extension:

```python
from tensorrt_model_connect import build
from families.qwen3_8.edge_llm.config import (
    BuildExecutionInputs, NamedCheckpoint, with_execution,
)

# request is an ordinary BuildRequest owned by this family; draft_path is a Path.
build(with_execution(request, BuildExecutionInputs(
    "dspark", (NamedCheckpoint("draft", draft_path),),
)))
```

A failed explicit pair is never replaced by a base-only bundle. Previously
recorded full-model results above are historical, not fresh refactor-head E2Es.

Request controls and the existing 9–1024 capacity range are checked before any
Edge preparation. The native precision default is not changed: this paired
profile explicitly requires FP16. Temporary staging uses the output filesystem.

## Declared build command

This family uses the existing cli.json protocol introduced in #1310. The family
owns its declaration, typed inputs and Python handler. The handler adapts those
inputs to the unchanged builder API, preserving native/Edge dispatch and bundle
publication. The legacy flat build command remains available for its existing
ordinary options; new family options use `trtmc qwen3_8 build`.
Help is offline and does not need a local checkpoint. No shared parser hook or
family registry entry is added.

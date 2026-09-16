# Qwen3.8 DSpark Edge-LLM adapter

The Qwen3.8 family owns its configuration admission, ONNX command mapping,
bundle assets and C++ runtime orchestration. It does not reuse another Qwen
family. Standalone builds retain the original native path; this change admits
Edge only for an explicit mixed-NVFP4 target plus DSpark companion.

## Build and inference

Provision the optional native SDK with `TRTMC_EDGELLM_ALL_KERNELS=ON` and
`TRTMC_EDGELLM_ONNX=ON` using the
[pinned package instructions](../../cmake/edgellm/README.md). The source is
GitHub Edge-LLM0.10.1 at `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.
Configure `CMAKE_PREFIX_PATH` for the installed package and compile the runtime
with `TRTMC_ENABLE_EDGELLM=ON`. Cross compilation is unsupported.

The Python build API accepts `BuildExecutionInputs(variant="dspark",
checkpoints=(NamedCheckpoint("draft", draft_path),))` through its `execution`
argument. The CLI equivalent adds `--execution-variant dspark` and
`--companion draft=/path/to/draft` to an ordinary build invocation.
The family invokes the original Edge Python ONNX exporter and native
`edgellm-onnx-build`; it does not alter source tensors or pad safetensors headers.
The exporter resolves the draft LM head from the target checkpoint. Both
speculative engines, embedding/head sidecars and tokenizer assets are bundled;
checkpoint weights are not duplicated in the bundle.

The runtime calls the original Edge speculative inference constructor with
proposal block7, verify8, drafting topK1/step1 and DSpark scheduling disabled.
It preserves supported sampling controls. Engine preparation errors warn and
try native once with the same requested execution variant; native currently
rejects the unmapped DSpark variant explicitly. No failure substitutes an
ordinary base-only decoder. Inference errors propagate without fallback.

## Validated exact profile

- Target: `RadixArk/Qwen3.8-27B-NVFP4`, revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Draft: `RadixArk/Qwen3.8-27B-DSpark`, revision
  `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
- Native SM120, CUDA13.3, TensorRT11.1.0.106, FP16 execution with source mixed
  NVFP4/FP8 metadata retained, TP1/batch1, input/KV capacity1024.
- Actual Model Connect paired build and public CLI inference passed.
- Independent greedy oracle: exact token match, NED0.0 against0.15.
  The saved CPU FP32 reference was reused after checkpoint-byte and reference
  function verification; it was not regenerated during this run.
- Original Edge `llm_basic` prompt and128-token sampled profile
  (temperature1/topK50/topP1): ROUGE-1 **0.4246**, ROUGE-L **0.2458**, above
  unchanged **0.25/0.20** gates. Chat enabled and thinking disabled.
- Publication regressions:11 existing family Python tests plus106 existing
  builder/architecture tests pass; both existing native C++ tests pass.

These results qualify only the exact text profile, not standalone NVFP4,
multimodal inputs, other checkpoints/platforms or statistical sampling parity.
The local run reused existing owning E2E helpers with an explicit companion;
this pair is not yet a registered pytest manifest case.

A first inference attempt exposed incompatible development JSON headers sharing
Edge’s3.12.0 version label. Matching the exact pinned headers fixed the crash
without changing Edge or the engines. The generic SDK now checks header content
rather than relying only on the version label.

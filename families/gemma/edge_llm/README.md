# Gemma4 MTP Edge-LLM adapter

This family owns the explicit Gemma4 unified target/assistant pair, ONNX
command mapping, bundle assets and C++ runtime adapter. Standalone Gemma1/2
builds remain native. Selecting a Gemma4 checkpoint alone does not enable this
paired path or claim native Gemma4 support.

## Build and inference

Use the [pinned native SDK provisioning](../../../cmake/edge_llm/README.md) with
`TRTMC_EDGELLM_ALL_KERNELS=ON` and `TRTMC_EDGELLM_ONNX=ON`, then configure the
Model Connect runtime with `TRTMC_ENABLE_EDGELLM=ON`. Set `CMAKE_PREFIX_PATH`
to the SDK installation. The source is official GitHub Edge-LLM 0.10.1,
revision `e8b29522938901f6df19ebeedd4b69bc8edbcd97`; cross compilation is not used.

Use the existing family CLI protocol with family-owned options (no checkpoint edits):

```sh
trtmc gemma build /path/to/target --precision fp16 \
  -o /path/to/pair.bundle --execution-variant mtp \
  --companion draft=/path/to/assistant
```

`trtmc gemma build /path/to/target --help` displays Gemma's options. Only Gemma
declares these flags in cli.json; core does not interpret them or select Edge execution.
Python callers use `GemmaBuildRequest` and the family-owned
`BuildExecutionInputs`/`NamedCheckpoint` types from `families.gemma.edge_llm.config`,
then call the unchanged `tensorrt_model_connect.build(request)` API.
The existing ordinary Gemma `build(request, writer)` entrypoint chooses paired
Edge execution only for an explicit family request. Without it, native behavior
and unsupported-model rejection remain unchanged. Companion paths must name
existing local directories and are never inferred or downloaded.

The family forwards both unmodified checkpoints to the original Python ONNX
exporter with `--mtp --mtp-draft-dir`, excluding image/audio branches for this
text-only profile. The original native ONNX builder creates both speculative
engines. Those engines, embedding and tokenizer assets are bundled; source
checkpoint weights and ONNX intermediates are not duplicated in the bundle.

The C++ adapter owns a persistent original Edge inference runtime with drafting
topK 1, steps 3 and verify size 4. This upstream MTP runtime uses greedy decoding;
the adapter rejects a requested non-greedy configuration instead of silently
changing it. Unmapped generation controls and capacity overflow are rejected.

An Edge preparation failure emits a warning and retains diagnostics. Native
Gemma currently cannot implement this MTP request, so the build fails explicitly
rather than substituting an ordinary base-only engine. Runtime errors propagate
without fallback.

## Qualification scope

The completed MTP qualification uses:

- Target `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Assistant `google/gemma-4-12B-it-assistant`, revision
  `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`.
- FP16 text execution, native SM80, CUDA 13.3, TensorRT 11.1.0.106,
  TP1/batch1, input 512 and KV capacity 1024, greedy chat with thinking disabled.
- A fresh independent BF16 Hugging Face reference and unchanged Model Connect
  normalized edit distance gate of 0.15.
- Original Edge `llm_basic` prompt and 128-token budget, with unchanged ROUGE-1
  / ROUGE-L gates of 0.25 / 0.20. Explicit greedy controls match upstream MTP's
  effective behavior; they do not claim parity with a sampling request.

The actual MTP Model Connect build and public CLI inference pass. Fresh HF
reference tokens match exactly (NED **0.0** against **0.15**). The 128-token
fixture passes ROUGE-1 **0.4343** / ROUGE-L **0.2286** against **0.25 / 0.20**.
Both checks were repeated successfully on this combined MTP/DSpark runtime,
using the same MTP bundle and a source-verified reused reference. Native
compilation, both C++ tests and all 10 existing family Python checks pass.
This recipe does not qualify other models, platforms, multimodal inputs or
sampling. It reuses existing family E2E helpers; the pair is not yet a registered
pytest manifest case.

## Qualified DSpark block7 extension

The same owning family also admits `--execution-variant dspark` with target
`google/gemma-4-12B-it` at the revision above and draft
`deepseek-ai/dspark_gemma4_12b_block7` at
`2fa72e765eec2965fc4d86a8663ce6769eba6218`. It forwards original
`--dspark-base` and `--dspark-draft` exports, packages both speculative engines
and confidence/Markov sidecars, and uses drafting topK 1, step 1, verify 8,
block7, scheduler off. Supported sampling controls remain enabled for DSpark;
the MTP-only greedy restriction is not applied to this variant.

Actual DSpark Model Connect ONNX export/build and public CLI inference pass on
SM80 with the same capacities above. Fresh independent BF16 HF reference tokens
match exactly (NED **0.0** <= **0.15**). The original 128-token Edge fixture uses
temperature 1, topK 50 and topP 1: ROUGE-1 **0.4114** / ROUGE-L **0.2286** pass
unchanged **0.25 / 0.20** gates. Both existing C++ tests and all 10 existing
family Python tests pass. This proves this exact sampled run, not statistical
sampling equivalence or other model/platform combinations.


## Checkpoint chat-template correction

The official Edge 0.10.1 static Gemma template omits the checkpoint's closed
thought channel when thinking is disabled, and differs in enabled-thinking
system-prefix handling. A first uncorrected MTP run produced `thought
Paris`
instead of the independent reference `Paris`, failing NED 0.6154 against 0.15.
The family now validates the source single-user template during build and
renders it faithfully before calling Edge with raw text. Unicode whitespace
trimming follows the checkpoint Jinja filter; raw-text requests are unchanged.
No generated-output filtering, engine change or relaxed quality gate is used.
The original Edge fixture is scored with this source-faithful prompt mapping,
not claimed as byte-for-byte parity with the faulty upstream static template.
MTP passes both unchanged quality gates with the same engine bundle after the
request-only fix, including on the combined runtime. DSpark also passes both
unchanged quality gates with the source-faithful prompt mapping. Initial failure evidence and the tokenizer audit are retained.

## Family-owned CLI refactor validation

The model qualification results above predate the CLI ownership refactor.
The refactor preserves exporter commands, engine composition, C++ inference
logic, reference outputs and quality thresholds. New coverage in the existing
family tests checks both CLI variants through ordinary core dispatch into the
family builder, malformed companions, typed request preservation and failed-build
bundle atomicity. Native adapter compilation and the sampler/pipeline C++ tests
were rerun successfully. Full checkpoint export/build/inference was not rerun
for this refactor; the successful qualification bundles were retired under the
approved artifact cleanup, so replay requires rebuilding those exact profiles.

## Declared build command

This family uses the existing cli.json protocol introduced in #1310. The family
owns its declaration, typed inputs and Python handler. The handler adapts those
inputs to the unchanged builder API, preserving native/Edge dispatch and bundle
publication. The legacy flat build command remains available for its existing
ordinary options; new family options use `trtmc gemma build`.
Help is offline and does not need a local checkpoint. No shared parser hook or
family registry entry is added.

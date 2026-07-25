---
title: CLI Reference
---

## `trtmc build`

`trtmc build` builds `.trtfb` bundles through the Python builder package.

```bash
trtmc build <hf-repo-or-local-dir> -o <output.trtfb> [options]
```

The C++ bridge runs `python -m tensorrt_model_connect build ...`. When
installed from the release wheel, `trtmc` is the native executable installed in
the environment's `bin/` directory and it uses the sibling `python3` or
`python` from that same environment. A source-built `./build/trtmc` falls back
to `python3` from the user's shell.

Source builds use the same subcommands through `./build/trtmc`.

Direct module execution is still available for debugging:

```bash
python -m tensorrt_model_connect build <hf-repo-or-local-dir> -o <output.trtfb>
```

### Build options

| Option | Purpose |
| --- | --- |
| `--model-revision REV` | Build a Hugging Face commit, tag, or branch instead of its default revision. |
| `--trust-remote-code` | Allow model repositories that require Hugging Face remote code. |
| `--max-cache-length N` | KV cache length, default `256`. |
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
| `--fp8-scales PATH` | Load precomputed FP8 scales. |
| `--save-fp8-scales PATH` | Save calibrated FP8 scales. |
| `--rtx` | Build for TensorRT-RTX backend. |
| `--config FILE` | Load schema-driven config profile. |
| `--set NS.FIELD=VALUE` | Override a config field; repeatable. |
| `--build-timing-json PATH` | Write structured build timing. |
| `--verbose` | Enable verbose TensorRT builder output. |

TriAttention options are also exposed for experimental KV compaction: `--triattention-stats`, `--triattention-kv-budget`, `--triattention-divide-length`, `--triattention-recent-window`, score aggregation, prompt-token accounting, prefill protection, and MLR/trig disable flags.

TensorRT is the build backend; there is no public build-method selector. Older
`--method trt` and `--method auto` spellings remain accepted for compatibility.

## Runtime commands

`trtmc` also inspects and runs bundles from C++.

```bash
trtmc run <bundle.trtfb> --prompt "text" [--image PATH] [--greedy]
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
`--cuda-graphs`, `--benchmark`, `--warmup`, `--config`, and repeatable
`--set`. `trtmc --help` prints one combined synopsis for all commands; it is
not separate per-command help. Read the relevant command section in that
combined output and this reference for the accepted options.

Text-generation options include `--max-new-tokens`, `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, and `--no-thinking`.

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

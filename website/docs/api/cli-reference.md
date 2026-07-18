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
| `--max-cache-length N` | KV cache length, default `256`. |
| `--dynamic-kv-cache` | Enable runtime-resizable KV cache support. |
| `--dynamic-kv-profile-rows A,B,C` | Override dynamic-KV optimization profiles. |
| `--image-height`, `--image-width` | Diffusion image shape overrides. |
| `--video-height`, `--video-width`, `--video-num-frames` | Diffusion video shape overrides. |
| `--num-inference-steps N` | Diffusion denoising step override. |
| `--precision fp32|fp16|bf16` | Override the family-selected build precision. Wan2.2 defaults to BF16. |
| `--quantize fp8|int8|int8_sq|int4|int4_awq|nvfp4|w4a8` | Quantization format. |
| `--quant-scales PATH` | Load precomputed quantization scales. |
| `--quant-calibration-samples N` | PTQ calibration sample count. |
| `--fp8` | Enable legacy FP8 auto-calibration path. |
| `--fp8-scales PATH` | Load precomputed FP8 scales. |
| `--save-fp8-scales PATH` | Save calibrated FP8 scales. |
| `--rtx` | Build for TensorRT-RTX backend. |
| `--config FILE` | Load schema-driven config profile. |
| `--set NS.FIELD=VALUE` | Override a config field; repeatable. |
| `--build-timing-json PATH` | Write structured build timing. |

TriAttention options are also exposed for experimental KV compaction: `--triattention-stats`, `--triattention-kv-budget`, `--triattention-divide-length`, `--triattention-recent-window`, score aggregation, prompt-token accounting, prefill protection, and MLR/trig disable flags.

TensorRT is the build backend; there is no public build-method selector. Older
`--method trt` and `--method auto` spellings are accepted as hidden
compatibility no-ops.

When `<hf-repo-or-local-dir>` is a Hugging Face model ID, the builder downloads
it through the standard Hugging Face cache. New bundles record the original ID
as `source_model_id` and the immutable cache commit as `source_revision`.
Wan builds also embed the AOT plugin image selected for the active TensorRT
major/minor; this is automatic and has no public plugin-path option.

## Runtime commands

`trtmc` also inspects and runs bundles from C++.

```bash
trtmc run <bundle.trtfb> --prompt "text" [--image PATH] [--greedy]
trtmc encode <bundle.trtfb> --prompt "text"
trtmc segment <bundle.trtfb> --image PATH --output PATH
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

Common options include `--hf-python`, `--backend-dir`, `--runtime-cache`, `--cuda-graphs`, `--benchmark`, `--warmup`, `--config`, and repeatable `--set`.

Text-generation options include `--max-new-tokens`, `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, and `--no-thinking`.

Object detection is available through `trtmc detect` for pipelines that implement `IPipeline::detect`.

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

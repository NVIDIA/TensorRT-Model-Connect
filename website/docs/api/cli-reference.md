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
| `--precision fp32|fp16|bf16` | Build precision. |
| `--quantize fp8|int8|int8_sq|int4|int4_awq|nvfp4|w4a8` | Quantization format. |
| `--quant-scales PATH` | Load precomputed quantization scales. |
| `--quant-calibration-samples N` | PTQ calibration sample count. |
| `--method auto|trt` | Engine definition method. `auto` selects the native TRT builder when a family plugin matches. |
| `--fp8` | Enable legacy FP8 auto-calibration path. |
| `--fp8-scales PATH` | Load precomputed FP8 scales. |
| `--save-fp8-scales PATH` | Save calibrated FP8 scales. |
| `--rtx` | Build for TensorRT-RTX backend. |
| `--config FILE` | Load schema-driven config profile. |
| `--set NS.FIELD=VALUE` | Override a config field; repeatable. |
| `--build-timing-json PATH` | Write structured build timing. |

TriAttention options are also exposed for experimental KV compaction: `--triattention-stats`, `--triattention-kv-budget`, `--triattention-divide-length`, `--triattention-recent-window`, score aggregation, prompt-token accounting, prefill protection, and MLR/trig disable flags.

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

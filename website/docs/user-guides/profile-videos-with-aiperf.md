---
title: Profile Video Generation with AIPerf
description: Benchmark an already-built text-to-video model and download complete clips.
---

First follow [Build from Source](../getting-started/source-build.md) and
[Build a Bundle](./build-a-bundle.md). This example assumes the existing
[Wan2.1 full profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/wan_t2v/tests/manifests/wan21-t2v-1.3b.json)
has been built as `wan21-t2v-1.3b.bundle` and passed its family's correctness checks:
**832×480, 17 frames, FP16 with the profile's existing FP32 layer selection**.
Run from the repository root in your prepared GPU environment. Install
**FFmpeg with `libx264rgb`** and put `ffmpeg`, `ffprobe`, and `trtmc-server` on `PATH`.

## 1. Install AIPerf

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
```

## 2. Start the built model (terminal 1)

```bash
trtmc-server ./wan21-t2v-1.3b.bundle \
  --model-name wan21-t2v-1.3b \
  --startup-timeout 1800 --request-timeout 14400 \
  --records artifacts/aiperf-video-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Add `--runtime-root`
when native libraries are in a separate directory.

## 3. Benchmark it (terminal 2)

```bash
printf '%s\n' '{"text":"A cat sitting on a beach watching the sunset"}' > /tmp/video-prompts.jsonl
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model wan21-t2v-1.3b \
  --endpoint-type video_generation --download-video-content \
  --tokenizer builtin --input-file /tmp/video-prompts.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs size:832x480 num_frames:17 num_inference_steps:30 seed:42 fps:24 \
  --request-timeout-seconds 14400 \
  --concurrency 1 --warmup-request-count 1 --request-count 2 \
  --export-level raw --artifact-dir artifacts/aiperf-videos
```

AIPerf submits multipart POST `/v1/videos`, polls GET `/v1/videos/{id}`, then
downloads GET `/v1/videos/{id}/content`. Keep `--download-video-content` to
include the complete MP4 transfer. The server encodes every native RGB frame
losslessly; `fps` controls playback. Frame count is fixed when the bundle is
built. Video serving requires explicit size and frame count in the bundle's
core build provenance; rebuild older video bundles that lack this metadata.
Existing still-image bundles retain their image endpoint, including bundles
with missing provenance or family-default dimensions.
A mismatched size/frame count, `seconds`, image conditioning, streaming,
or batching is rejected.

Read client latency in `profile_export_aiperf.json`. It includes submission,
polling, encoding, and download. The separate server `video_completion` record
reports `model_call_ms` (`public_task_call_wall`), the original request ID,
job ID, complete file hash and ordered decoded-frame hash. Task timing excludes
build, loading, warmup and external media encoding. Two requests demonstrate
the workflow; they do not establish serving capacity.

AIPerf 0.13 exports job JSON and intentionally omits downloaded video bytes.
To save a clip, take its job ID from the raw export and run:

```bash
curl --fail http://127.0.0.1:8000/v1/videos/VIDEO_ID/content -o clip.mp4
```

Completed clips expire after ten minutes; retained jobs and reserved media
storage are bounded. Use `--max-video-jobs`, `--max-video-bytes`, and
`--video-ttl-seconds` to set those limits. Shutdown waits for active native calls and encoding
before removing temporary files.

## Use another built video model

Replace the bundle, API name, prompt and settings with those of its existing
full text-to-video profile. The current full profiles are:

| Profile | Size | Frames | Steps | Seed | Explicit guidance |
| --- | --- | ---: | ---: | ---: | --- |
| `wan21-t2v-1.3b` | 832×480 | 17 | 30 | 42 | Omit to retain the public Task default |
| `wan22-ti2v-5b` | 1280×704 | 121 | 50 | 42 | `guidance_scale:5` |
| `minimax-h3-768p` | 1344×768 | 124 | 50 | 0 | Omit to retain the public Task default |

MiniMax-H3's existing prompt is the `prompt` string in
`families/minimax_h3/tests/prompts/t2va-example-1.json`; pass that whole string
unchanged. This endpoint returns its video frames. Audio output and
image-to-video are separate workloads.

Cosmos3-Nano and LTX-Video currently have only partial/L0 profiles and are not
full-model qualifications. Inventory all variants, including these gaps, with
`python -m trtmc_aiperf_qual endpoint-models --repo . --group videos` after
installing the [qualification tools](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md).

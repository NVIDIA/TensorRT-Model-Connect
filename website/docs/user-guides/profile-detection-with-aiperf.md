---
title: Profile Object Detection with AIPerf
description: Benchmark one image at a time with an already-built TRTMC detection model.
---

This example uses **YOLO11n**. Prepare the native runtime using
[Build from Source](../getting-started/source-build.md), including the
`trtmc_backend_trt` and `trtmc_model_yolo11` targets, then follow
[Build a Bundle](./build-a-bundle.md) with the existing
[YOLO11n FP16 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/e3eeb98f2e5ba1f0ceaf40126d07a3ed02257a2e/families/yolo11/tests/benchmark/yolo11n.yaml).
Run the family correctness checks first. These commands assume `yolo11n.bundle`
is already built and `trtmc-server` is on `PATH`.

Run from the TRTMC checkout in your prepared GPU environment. In Docker, use
two terminals in the same container.

## 1. Install AIPerf and prepare the images

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0
.venv-aiperf/bin/python apps/aiperf_qual/examples/prepare_detection_inputs.py \
  --output-dir /tmp/aiperf-detection-inputs
```

The helper saves 100 original images from the pinned COCO validation dataset,
without resizing or recompressing them. Its JSONL uses original-byte data URLs
to avoid AIPerf recompressing local JPEG paths. `selection.json` records image IDs,
dimensions, hashes, annotations, and skipped or failed inputs. Use a new output
directory for each preparation. The AIPerf client comes from PyPI.

## 2. Start the model (terminal 1)

```bash
trtmc-server ./yolo11n.bundle --model-name Ultralytics/YOLO11 \
  --records artifacts/aiperf-detection-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Use the full binary path
and `--runtime-root` if your runtime libraries are in a separate directory.

## 3. Run the benchmark (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model Ultralytics/YOLO11 \
  --endpoint-type image_retrieval --endpoint /v1/infer \
  --input-file /tmp/aiperf-detection-inputs/inputs.jsonl \
  --custom-dataset-type single_turn \
  --concurrency 1 --warmup-request-count 5 --request-count 50 \
  --random-seed 0 --ui-type simple --export-level raw \
  --artifact-dir artifacts/aiperf-detection
```

Each request contains one image and returns all detected boxes. AIPerf calls
this endpoint type `image_retrieval`; the workload here is **object detection**.
It does not qualify document layout or table detection.

The server accepts embedded RGB PNG/JPEG images, up to 16 million pixels
(`--max-image-pixels`). The prepared JSONL supplies data URLs directly. The server does
not fetch URLs or open client-provided paths. Raise `--max-body-bytes` if your
images exceed the default 1 MiB request limit, including base64 overhead.
Multiple images, streaming, prompts, and per-request threshold controls are
rejected. YOLO11's built-in filtering and suppression use its family configuration.

Responses use NIM's `data[].bounding_boxes` shape. Group keys preserve native
numeric class IDs as strings; YOLO11 uses contiguous COCO IDs 0–79. Coordinates
are divided by the original width and height. The server preserves scores,
empty detections, and family-produced boxes without clipping or re-filtering.
A family may produce boxes extending beyond an image; their normalized
coordinates can therefore extend beyond 0–1.

Inspect request latency and image throughput in `profile_export_aiperf.json`.
Server records retain request IDs, actual dimensions, image bytes, box counts,
and `model_call_ms` with `timing_scope=public_task_call_wall`. Task timing
includes family preprocessing and postprocessing; transport image decoding is
outside it. Keep client latency and native Task time separate. Build, load,
startup, verification, and warmup are excluded from measured performance.
This short run is an integration example, not a capacity or performance gate.

For another already-built detector, replace the bundle and API model name and
validate that family's output and label space first. The bundle must declare
legacy `object_detection` or SDK `image_to_boxes`. If serving multiple compatible
detectors, select one with `--extra-inputs model:YOUR_MODEL_NAME`; otherwise the
server uses the sole compatible detector. See the
[model-group inventory](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#native-endpoint-group-coverage).

For Deformable DETR, build the
[Deformable DETR FP16 manifest](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/deformable_detr/tests/manifests/deformable-detr-r50.json)
and use `deformable-detr-r50.bundle` with `--model-name SenseTime/deformable-detr`
on the server and `--model SenseTime/deformable-detr` in AIPerf. Rebuild older
Deformable DETR bundles: the family now preserves image aspect ratio using the
checkpoint's shortest/longest-edge bounds and a dynamic TensorRT engine. The
manifest's 800x800 build settings select the 800-pixel resize target; they do
not stretch every image to a square.

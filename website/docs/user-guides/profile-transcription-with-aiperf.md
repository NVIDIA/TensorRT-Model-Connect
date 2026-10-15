---
title: Profile Transcription with AIPerf
description: Benchmark complete WAV clips with an already-built TRTMC speech model.
---

This example uses **openai/whisper-tiny**. Prepare the native runtime using
[Build from Source](../getting-started/source-build.md), including the
`trtmc_backend_trt` and `trtmc_model_whisper` targets, then follow
[Build a Bundle](./build-a-bundle.md) with the existing
[Whisper FP16 profile](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/e3eeb98f2e5ba1f0ceaf40126d07a3ed02257a2e/families/whisper/tests/benchmark/whisper-tiny-fp16.yaml).
Rebuild older Whisper bundles from current source for the corrected GELU
activation and FP32 encoder/logit arithmetic, and run the family correctness checks first. The commands below assume
`whisper-tiny-fp16.bundle` is already built and `trtmc-server` is on `PATH`.

Run from the TRTMC checkout in your prepared GPU environment. In Docker, use
two terminals in the same container.

## 1. Install AIPerf and prepare the clips

```bash
python -m pip install -e '.[serve]' -C py-only=true
python -m venv .venv-aiperf
.venv-aiperf/bin/pip install aiperf==0.13.0 soundfile==0.13.1
.venv-aiperf/bin/python apps/aiperf_qual/examples/prepare_asr_inputs.py \
  --output-dir /tmp/aiperf-asr-inputs
```

The helper downloads a pinned LibriSpeech test-clean shard and saves the first
ten eligible complete clips as mono 16-bit PCM WAV files. It preserves their
samples and sample rates. `selection.json` records the original IDs, gold text,
durations, hashes, and any skipped or failed inputs. Use a new output directory
for each preparation. The AIPerf client comes from PyPI.

## 2. Start the model (terminal 1)

```bash
trtmc-server ./whisper-tiny-fp16.bundle \
  --model-name openai/whisper-tiny \
  --records artifacts/aiperf-asr-server.jsonl
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Use the full binary path
and `--runtime-root` if your runtime libraries are in a separate directory.

## 3. Run the benchmark (terminal 2)

```bash
.venv-aiperf/bin/aiperf profile \
  --url http://127.0.0.1:8000 --model openai/whisper-tiny \
  --endpoint-type audio_transcription --endpoint /v1/audio/transcriptions \
  --input-file /tmp/aiperf-asr-inputs/inputs.jsonl \
  --custom-dataset-type single_turn --extra-inputs max_output_tokens:120 \
  --concurrency 1 --warmup-request-count 5 --request-count 50 \
  --random-seed 0 --ui-type simple --export-level raw \
  --artifact-dir artifacts/aiperf-asr
```

This sends complete recordings and returns whole transcripts: five warmups,
then 50 measured requests. `max_output_tokens` is a TRTMC extension; here it
matches the selected family profile. JSON and plain-text responses are supported.
This initial endpoint accepts mono 16-bit PCM WAV clips up to 30 seconds
(`--max-audio-seconds` sets the server limit). Streaming, translation, language,
prompt, sampling, and timestamp controls are rejected in this stage.

Inspect successful requests and request latency in `profile_export_aiperf.json`.
Server records preserve request IDs, audio duration/rate/sample count, and
`model_call_ms` with `timing_scope=public_task_call_wall`. Task timing includes
the family's preprocessing and decoding; WAV transport decoding is outside it.
For a joined run, audio seconds/sec is the sum of successful measured audio
durations divided by the measured request span. Per-request real-time factor is
latency in seconds divided by clip duration; report client and Task factors
separately. Build/load/startup and warmup are excluded. This small integration
example does not establish a performance regression gate or streaming TTFT.

For another built ASR model, replace the bundle and API model name, and use only its declared Task controls. Validate that family's actual
bundle and transcripts first. See [transcription Tasks](./multimodal-speech.md)
and the [coverage matrix](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/apps/aiperf_qual/README.md#native-endpoint-pilot-coverage).

---
title: Configurable Canary Decoding
---

# Configurable Canary decoding

This tutorial builds Canary from a local NeMo checkpoint and exercises the
current offline transcription controls.

## Build and inspect

```bash
python -m tensorrt_model_connect build /models/canary-1b-v2.nemo \
  -o /tmp/canary-1b-v2.bundle \
  --precision fp16

trtmc inspect /tmp/canary-1b-v2.bundle
```

The Canary family owns checkpoint parsing, graph construction, packaged prompt
metadata, runtime orchestration, and validation.

## Transcribe or translate

```bash
trtmc transcribe /tmp/canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input /data/input.wav \
  --max-output-tokens 80 \
  --source-language en \
  --target-language en \
  --translate false \
  --beam-size 1 \
  --punctuation true
```

For translation, use different supported source and target languages and set
`--translate true`. Beam size, length penalty, punctuation, timestamps, and
segmentation are request options:

```bash
trtmc transcribe /tmp/canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input /data/english.wav \
  --source-language en \
  --target-language fr \
  --translate true \
  --beam-size 2 \
  --length-penalty 1.0 \
  --timestamps true \
  --segment-length-seconds 20
```

Use `transcribe-batch` with repeated `--input` options when the selected family
implements batch transcription. The request-level decoding controls apply to
the batch; results preserve input order.

Exact supported languages, limits, segmentation behavior, and expected output
belong to `families/canary/tests/`. Validate them against the checkpoint and a
family-owned E2E case rather than assuming that accepting a CLI flag proves
model support.

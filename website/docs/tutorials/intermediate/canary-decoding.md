---
title: Configurable Canary Decoding
---

This lab exercises the family-owned Canary transcription implementation through
the public `ITranscription` and optional `IBatchTranscription` interfaces.

## Learning objectives

- build the exact Canary family from a Hugging Face or local checkpoint;
- run greedy transcription, translation, and beam search controls;
- locate the family-owned manifest, audio fixtures, runtime, and tests;
- call the abstract C++ Task API without depending on Canary concrete types.

## Build from a checkpoint

```bash
python -m pip install -r families/canary/requirements.txt
python -m tensorrt_model_connect build nvidia/canary-1b-v2 \
  --output canary-1b-v2.bundle \
  --precision fp16 \
  --max-sequence-length 128
```

The resolver selects `families/canary/support.py`, then imports only
`families/canary/model.py`. That plain function owns graph construction,
weights, runtime sections, and rejection of unsupported build values.

## Greedy transcription

Use the checked-in audio fixture:

```bash
trtmc transcribe canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input families/canary/tests/data/Recording.wav \
  --max-output-tokens 50 \
  --beam-size 1
```

The CLI decodes the WAV file, then calls `ITranscription::transcribe()`. Image
and audio file I/O belongs to `apps/cli/`, not the Task interface or family.

## Translation and beam search

```bash
trtmc transcribe canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input families/canary/tests/data/Recording.wav \
  --source-language en \
  --target-language de \
  --translate true \
  --beam-size 4 \
  --length-penalty 1.0
```

These are Task-request values. Canary validates and implements them in its own
runtime. Their presence in the CLI does not establish support for another
transcription family.

## Duration, segmentation, and timestamps

```bash
trtmc transcribe canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input families/canary/tests/data/Recording.wav \
  --max-input-seconds 120 \
  --segment-length-seconds 30 \
  --segment-min-seconds 0.5 \
  --segment-overlap-seconds 1 \
  --lcs-merge true \
  --timestamps true
```

Use segmentation only for a workload that needs it, and compare the exact
request with a suitable reference. A command completing does not by itself
prove transcript quality.

## Batch CLI behavior

The separate batch Task accepts repeated inputs:

```bash
trtmc transcribe-batch canary-1b-v2.bundle \
  --runtime-root /opt/trtmc/lib \
  --input families/canary/tests/data/Recording.wav \
  --input families/canary/tests/data/asr_probes/probe_02_clean_16k_mono_no_resample.wav \
  --beam-size 1
```

The returned result count must match the request count. The family decides how
to batch encoder and decoder work; no shared scheduler owns that policy.

## C++ API

Applications load the abstract Task and cast to the interface they need:

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("canary-1b-v2.bundle", "/opt/trtmc/lib");
auto* can_transcribe = dynamic_cast<trtmc::ITranscription*>(task.get());
if (can_transcribe == nullptr) {
    throw std::runtime_error("bundle is not a transcription task");
}

trtmc::TranscriptionConfig config;
config.beam_size = 1;
auto result = can_transcribe->transcribe(
    samples.data(), static_cast<std::int32_t>(samples.size()), config);
```

The application depends on `ITranscription`; it never includes a Canary header.
The Canary runtime implements that interface and depends on the abstract Engine
API supplied by the backend.

## Validation

The exact contract lives in:

```text
families/canary/tests/manifests/canary-1b-v2.json
families/canary/tests/test_e2e.py
families/canary/tests/thresholds/
```

Run the selected E2E only with the required GPU and runtime:

```bash
TRTMC_E2E=1 \
TRTMC_BINARY=/path/to/trtmc \
TRTMC_RUNTIME_ROOT=/path/to/runtime \
PYTHONPATH=core/builder:. \
python3 -m pytest families/canary/tests/test_e2e.py \
  --e2e-testcase canary-1b-v2 -v
```

## Self-check

1. Which component owns beam-search behavior?
2. Why does the C++ example cast to an abstract interface?
3. What evidence is needed beyond a successful build?
4. Why do CLI WAV helpers stay outside the core Task API?

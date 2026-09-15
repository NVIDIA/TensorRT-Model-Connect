---
title: Runtime Lifecycle
---

The native runtime performs one exact load and returns an abstract Task:

```cpp
auto task = trtmc::load_task("model.bundle", "/opt/trtmc/lib");
```

## Load sequence

1. `BundleReader` validates the format-1 header and all section bounds.
2. The loader validates the `family` and `backend` names as safe DSO tokens.
3. It loads `libtrtmc_backend_<backend>.so` from the explicit runtime root.
4. It loads `libtrtmc_model_<family>.so` from that same root.
5. It resolves the single `trtmc_create_family` factory and passes a
   `FamilyContext` containing the read-only bundle reader and abstract backend.
6. It verifies that the returned `ITask::task()` matches the bundle header.

There is no current-directory search, environment fallback, registry lookup,
strategy switch, sibling-family probe, or load retry.

## Ownership after transfer

The selected family implements one or more interfaces in
`core/runtime/include/trtmc/task.h`. It owns preprocessing, postprocessing,
request orchestration, tokenizer/sampler state, family sections, and engine
binding. It creates engines only through the abstract `IBackend`/`IEngine`
contract. The backend owns TensorRT runtime objects, not model policy.

`FamilyContext.reader` is read-only. A factory may consume its sections before
returning or copy the lightweight `BundleReader` into the pipeline for deferred
reads; it must not retain a reference to the temporary factory context.

## Multichannel streaming audio

Families can opt into `IMultichannelStreamingAudioGeneration` without changing
the existing mono `IStreamingAudioGeneration` interface. A family implementing
both is dispatched through the multichannel capability by the CLI.

Each `AudioChunkView` borrows interleaved float PCM (`L0, R0, L1, R1, ...` for
stereo), with an explicit channel count and sample rate. `num_samples` counts
scalar samples, not frames per channel. Chunks must be nonempty whole frames;
the sample rate and channel count stay constant within a call. Callbacks are
synchronous, ordered, and non-concurrent. Their pointers are valid only during
the callback. Normal return ends the stream and reports the sum of delivered
scalar samples; callback exceptions must stop generation and propagate.

`trtmc generate-audio ... --stream true --output audio.raw` writes interleaved
float32 samples and reports `format`, `sample_rate`, `num_channels`, and
`num_samples` in its success JSON. Playback duration is
`num_samples / num_channels / sample_rate`. This is raw PCM, not a WAV file.
Invalid chunks, format changes, inconsistent totals, and file-write errors fail
the command without success JSON. A failed stream can leave a partial output
file; callers must not treat file existence alone as success.

This capability does not add HTTP transport, encoded formats, or streaming
support to models that do not already produce incremental audio. Existing mono
families remain unchanged and report `num_channels: 1`.

## Optional load settings

Runtime-sized KV capacity is passed directly to compatible families.
TensorRT-RTX runtime cache and CUDA graph settings are accepted only when the
bundle selects `trt_rtx`; the standard backend rejects them. TVM-FFI BYOK is an
explicit extension DSO and three-part binding, not a general plugin registry.

## Teardown

Applications destroy Task objects before the loaded family and backend
libraries leave scope. Families release their streams, buffers, communicators,
engines, and family-local state; the loader owns the dynamic-library handles.

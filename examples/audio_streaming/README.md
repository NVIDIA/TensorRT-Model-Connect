# Persistent audio streaming

This example loads one TensorRT-Model-Connect bundle once, reads text prompts
from standard input, and calls the header-only C++ `StreamingTextToSpeech` SDK
over the C ABI for every non-empty line. It opens no network port and adds no
behavior to the runtime or model family.

## Build

Install TensorRT-Model-Connect first, then point CMake at that installation:

```bash
cmake -S examples/audio_streaming -B /tmp/trtmc-audio-streaming \
  -DCMAKE_PREFIX_PATH=/opt/trtmc
cmake --build /tmp/trtmc-audio-streaming --target trtmc_audio_streaming -j
```

The installation must contain the C SDK (`libtrtmc_c.so.1`), `libtrtmc_runtime`,
the TensorRT backend, and the family DSO named by the bundle. The example links
the public SDK and existing runtime loader; it does not link a model family.

## Run

```bash
printf '%s\n' 'First prompt' 'Second prompt' | \
  /tmp/trtmc-audio-streaming/trtmc_audio_streaming model.bundle \
  > utterances.f32
```

Arguments are strict: one positional bundle is required. For SDK bundles,
`--runtime-root` is optional and uses the SDK library's directory by default;
pass it when backend/family DSOs live elsewhere. `--chunk-frames` and
`--max-new-tokens` accept positive integers and default to 16 and 750.
Unknown, duplicate, missing, and extra arguments fail
before the bundle is loaded.

Standard output contains native-endian FP32 PCM chunks in callback order. One
additional FP32 zero value marks the end of each completed utterance. Empty
input lines emit nothing. Standard error reports load, readiness, utterance,
sample-count, sample-rate, and EOF status; it never contains PCM bytes.

The loaded SDK model must implement `StreamingTextToSpeech` and declare the
`chunk_frames` and `max_new_tokens` options. This example keeps its intentional
16/750 defaults and passes them to the family; unsupported options are not
silently ignored. Output must be mono, and the callback sample rate is reported
on standard error; the raw stream has no WAV header. Failed or incomplete
utterances do not emit a successful end delimiter.

During family migration, an existing `audio_generation` bundle explicitly
selects the existing `IStreamingAudioGeneration` application path and still
requires `--runtime-root`. SDK bundles select the SDK path before execution;
an unsupported SDK Task or failed call never retries the existing path.

For an existing bundle, keep the explicit runtime directory:

```bash
printf '%s\n' 'First prompt' 'Second prompt' | \
  /tmp/trtmc-audio-streaming/trtmc_audio_streaming model.bundle \
    --runtime-root /opt/trtmc/lib > utterances.f32
```

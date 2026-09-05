# Persistent audio streaming

This example loads one TensorRT-Model-Connect bundle once, reads text prompts
from standard input, and calls the public `IStreamingAudioGeneration` interface
for every non-empty line. It opens no network port and adds no behavior to the
runtime or model family.

## Build

Install TensorRT-Model-Connect first, then point CMake at that installation:

```bash
cmake -S examples/audio_streaming -B /tmp/trtmc-audio-streaming \
  -DCMAKE_PREFIX_PATH=/opt/trtmc
cmake --build /tmp/trtmc-audio-streaming --target trtmc_audio_streaming -j
```

The installation must contain `libtrtmc_runtime`, the TensorRT backend, and the
family DSO named by the bundle. The example links only the public runtime
loader; it does not link a model family.

## Run

```bash
printf '%s\n' 'First prompt' 'Second prompt' | \
  /tmp/trtmc-audio-streaming/trtmc_audio_streaming model.bundle \
    --runtime-root /opt/trtmc/lib \
    --chunk-frames 16 \
    --max-new-tokens 750 \
  > utterances.f32
```

Arguments are strict: one positional bundle and one `--runtime-root` are
required; `--chunk-frames` and `--max-new-tokens` accept positive integers and
default to 16 and 750. Unknown, duplicate, missing, and extra arguments fail
before the bundle is loaded.

Standard output contains native-endian FP32 PCM chunks in callback order. One
additional FP32 zero value marks the end of each completed utterance. Empty
input lines emit nothing. Standard error reports load, readiness, utterance,
sample-count, sample-rate, and EOF status; it never contains PCM bytes.

The bundle must implement both the `audio_generation` task and the optional
`IStreamingAudioGeneration` capability. A bundle without that capability exits
with an error. The callback sample rate is reported on standard error; the raw
stream has no WAV header.

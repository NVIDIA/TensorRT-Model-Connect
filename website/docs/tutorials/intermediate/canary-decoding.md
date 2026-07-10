---
title: Configurable Canary Decoding
---

This tutorial builds a Canary bundle from a local NeMo checkpoint and uses the
offline decoding controls exposed by the CLI and C++ API.

## Build from a local checkpoint

The input may be a `.nemo` archive or a directory containing one compatible
Canary archive. No remote model identifier is required.

```bash
CANARY_NEMO=/models/canary-1b-v2.nemo

./build/trtmc build "$CANARY_NEMO" \
  -o /tmp/canary-1b-v2.trtfb \
  --precision fp16 \
  --max-cache-length 128

./build/trtmc inspect /tmp/canary-1b-v2.trtfb --list-engines
```

The builder reads `model_config.yaml`, the checkpoint weights, prompt defaults,
and the SentencePiece vocabulary from the local archive. The generated bundle
records the exact decoder prompt and control token IDs from that checkpoint.
An archive missing a required prompt or language token fails during the build.

## Greedy transcription

Omitting the new controls preserves English greedy transcription with
punctuation and no timestamps:

```bash
./build/trtmc transcribe /tmp/canary-1b-v2.trtfb \
  --audio /data/input.wav \
  --max-new-tokens 80
```

`--language fr` remains a shorthand for an equal source and target language.
The explicit form is:

```bash
./build/trtmc transcribe /tmp/canary-1b-v2.trtfb \
  --audio /data/french.wav \
  --source-language fr \
  --target-language fr \
  --task transcribe
```

## Translation and beam search

Canary 1B v2 translates between English and each other supported language.
The source and target must differ for `translate`; one of them must be `en`.

```bash
./build/trtmc transcribe /tmp/canary-1b-v2.trtfb \
  --audio /data/english.wav \
  --source-language en \
  --target-language fr \
  --task translate \
  --beam-size 4 \
  --max-new-tokens 120
```

`--beam-size 1` is the backward-compatible greedy path. Values from 2 through
16 use deterministic beam search. The current engine stores one decoder
hypothesis, so beam search replays candidate prefixes and trades throughput for
search quality.

Use `--no-punctuation` to request the checkpoint's no-punctuation prompt and
remove remaining punctuation from decoded text. `--punctuation` is the default.

## Duration, segmentation, and timestamps

```bash
./build/trtmc transcribe /tmp/canary-1b-v2.trtfb \
  --audio /data/long.wav \
  --segment-length-seconds 20 \
  --max-input-seconds 300 \
  --timestamps
```

Timestamp output is tab-separated:

```text
0.000   20.000  First decoded segment...
20.000  40.000  Second decoded segment...
```

Each interval is an independently decoded audio window. These are segment-level
input boundaries, not word alignment from Canary's auxiliary CTC timestamp
model. Segment decoding has no overlap or cross-segment decoder context.

| Option | Units and bounds | Behavior |
| --- | --- | --- |
| `--max-new-tokens` | Tokens; at least 1 and no more than the bundle target limit minus prompt tokens | Applied independently to every segment. |
| `--beam-size` | Integer in `[1, 16]` | `1` is greedy; larger values use beam search. |
| `--max-input-seconds` | Seconds; finite and greater than 0 | Rejects an input whose total duration exceeds the value. |
| `--segment-length-seconds` | Seconds; finite, greater than 0, and no greater than the bundle audio window | Splits long input into ordered independent requests. |

Without `--segment-length-seconds`, an input longer than the bundle's audio
window is rejected instead of being silently truncated.

## Batch CLI behavior

Repeat `--audio` to process more than one input. Decoding flags are batch-wide,
results retain input order, and every output line starts with its source path.

```bash
./build/trtmc transcribe /tmp/canary-1b-v2.trtfb \
  --audio /data/one.wav \
  --audio /data/two.wav \
  --source-language en \
  --target-language en \
  --beam-size 2
```

## C++ API

The C++ batch API carries a complete configuration per request:

```cpp
#include <trtmc/pipeline.h>

auto pipeline = trtmc::load("/tmp/canary-1b-v2.trtfb");

trtmc::TranscriptionRequest english;
english.audio_samples = english_pcm;
english.config.input_sample_rate = 16000;
english.config.max_output_tokens = 80;
english.config.beam_size = 1;
english.config.source_language = "en";
english.config.target_language = "en";

trtmc::TranscriptionRequest translation = english;
translation.audio_samples = second_pcm;
translation.config.beam_size = 4;
translation.config.target_language = "fr";
translation.config.task = trtmc::TranscriptionTask::kTranslate;
translation.config.timestamps = true;

std::vector<trtmc::TextResult> results =
    pipeline->transcribe_batch({english, translation});
```

The default batch implementation executes requests sequentially and preserves
their order and individual configs. The legacy four-argument `transcribe`
overload remains available and maps to the default greedy configuration.

Unsupported languages, mismatched task/language combinations, invalid beam
sizes, excessive output lengths, and invalid duration values throw
`std::invalid_argument` with the failing option named in the message.

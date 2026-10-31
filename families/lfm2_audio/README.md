# LFM2.5 audio

This family implements `LiquidAI/LFM2.5-Audio-1.5B`, revision
`c362a0625dfe45aa588dce5f0ada28a7e5707628`. It owns the FastConformer
encoder, multimodal LFM backbone, eight-codebook depth transformer, custom
LFM audio detokenizer, tokenizer, waveform processing, and generation state.
Inference uses native C++ and TensorRT. Python is used to build bundles and
run the original-model comparisons.

## Build

```python
from huggingface_hub import snapshot_download
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.lfm2_audio.model import BuildRequest, build

checkpoint = snapshot_download(
    "LiquidAI/LFM2.5-Audio-1.5B",
    revision="c362a0625dfe45aa588dce5f0ada28a7e5707628",
)
writer = BundleWriter("lfm25-audio.bundle")
try:
    build(
        BuildRequest(
            model_dir=checkpoint,
            task="offline_speech_dialogue",
            precision="bf16",
            max_sequence_length=32768,
        ),
        writer,
    )
    writer.finish()
except BaseException:
    writer.abort()
    raise
```

The equivalent family command is:

```bash
trtmc lfm2_audio build LiquidAI/LFM2.5-Audio-1.5B \
  --revision c362a0625dfe45aa588dce5f0ada28a7e5707628 \
  --max-sequence-length 32768 -o lfm25-audio.bundle
```

The bundle exposes ASR, text-to-speech, streaming text-to-speech,
speech-to-speech, ordered audio/text-to-text messages, and speech dialogue
sessions. `TextConversation` handles text-only message histories. The main
model uses BF16 and the custom detokenizer uses FP32.
Input is mono PCM at the supplied sample rate; preprocessing resamples it
to 16 kHz. Output is mono 24 kHz PCM. The checkpoint is intended for English.

## Native tasks

```cpp
#include <trtmc/audio.hpp>
#include <trtmc/speech.hpp>

trtmc::LoadOptions options;
options.runtime_root = "path/to/runtime";
auto model = trtmc::Model::load("lfm25-audio.bundle", options);
auto speech = model.task<trtmc::TextToSpeech>().run(
    {"Hello there. How can I help you today?", "en"},
    {{"voice", "uk_male"}, {"max_new_tokens", std::int64_t{512}},
     {"audio_temperature", 0.8}, {"audio_top_k", std::int64_t{64}}});
// Read speech.samples(), speech.sample_rate(), and speech.channels().
```

The four voice values are `us_male`, `us_female`, `uk_male`, and `uk_female`.
They select the checkpoint's corresponding TTS system prompts. Only the two
TTS tasks accept the voice setting.

`StreamingTextToSpeech::run` delivers successive stable waveform chunks.
Concatenating all chunks produces the same waveform as the non-streaming
task for the same generation settings and seed. The callback may stop
delivery by returning `false`.

For conversation, create an `OfflineSpeechDialogue` session, append audio,
and call `session.realtime().commit_input_turn()`. Consume its text and audio
events, then call `session.append_text(...)` and commit the next turn.
Generated audio-code history stays in the session, so a text follow-up does
not reencode generated waveforms. `finish_input()` completes input for the
session; `reset()` starts a new conversation.

`AudioTextToText` accepts ordered system, user, and assistant text parts and
user audio parts; requests must include both audio and text. Use
`TextConversation` for text-only input. Its messages accept text parts;
structured reasoning, tool calls, and tool definitions are unsupported.
Use the dialogue session when assistant speech must remain
in history, because the request interface does not carry generated audio codes.

## Generation settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_new_tokens` | 512 | Combined number of generated text tokens and audio frames |
| `text_temperature` | 0 | Text sampling temperature; nonpositive selects greedy decoding |
| `text_top_k` | 1 | Text candidates; 1 selects greedy decoding, 0 disables filtering |
| `audio_temperature` | 0 | Audio codebook sampling temperature; nonpositive selects greedy decoding |
| `audio_top_k` | 1 | Candidates for each audio codebook; 0 disables filtering |
| `seed` | 0 | Reproducible native sampling seed |
| `voice` | `us_male` | TTS voice |

The original examples use audio temperature 0.8 and top-k 64 for TTS, and
temperature 1.0 and top-k 4 for interleaved dialogue. Native sampling uses
the original BF16 probability calculation, with an independent random-number
generator; a native seed does not reproduce PyTorch's random draws.

The prompt, retained history, and requested generation budget must fit the
bundle's context capacity. Text results omit control-token framing. Validation
also checks untrimmed token IDs, including control tokens and terminal audio
frames. A token-budget truncation retains every ordinary audio frame already
generated; a completed response removes its terminal end-of-audio frame before
waveform synthesis.

## Validation

The family tests pin the original `Liquid4All/liquid-audio` source at commit
`19e65845923a7f136442c95137884ec61eb386aa`. The reference dependency is the
official `liquid-audio==1.3.0` wheel, whose package sources match that commit.
Wheel-only installation avoids requiring its source-build backend during
dependency preparation. The real-model gate checks task
semantics: ASR content, the requested TTS words, woodworking slogans followed by
chair-specific dialogue, the red-light answer, audio/text ordering, retained
dialogue history, and completion within the documented budget. Saved native and
original waveforms are transcribed by the pinned original ASR model. Transcript
receipts preserve intended text, recognized text, tokens, an exact copy of the ASR
input waveform, and stop
state. Punctuation and the same-spoken `your`/`you're` and `their`/`they're`
spellings are equivalent. An `or` immediately before a quoted slogan may be
omitted between complete list alternatives; other words, names, numbers, and
negation remain required.

All four original voice conditions are exercised with their exact prompt IDs and
intended spoken content. This does not provide an independent perceptual timbre
or accent score. Native streamed/offline waveforms must agree within 1e-6, and
public/internal session outputs must agree. Token, code, and original/native
waveform differences remain diagnostics rather than semantic acceptance
thresholds. Component and sampling-probability tests remain separate.

Selected manifest cases produce named `test_e2e[case_name]` results. One complete
13-case matrix runs per pytest invocation, even when a single case is selected;
every selected result requires the full matrix to pass. The first selected case
archives the shared execution, and each case records its own result and media.

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/lfm2_audio/tests --e2e-model lfm2_audio -q
```

Set `TRTMC_RUNTIME_ROOT` to the built native runtime directory. The optional
requirements in this family are needed for the original-model tests.
For offline validation, stage the manifest's `external_files` (`asr.wav` and
`question.wav`) in a directory and set `TRTMC_LFM2_AUDIO_MODEL_DIR` to that
directory. Without this setting, validation downloads the declared URLs and
requires external network access. `HF_HUB_OFFLINE=1` selects cached checkpoint
files; it does not disable these external downloads. Fully offline validation
requires both staged WAVs and cached checkpoint files. Evidence records the
declared URL and whether each input was staged or downloaded; the test does not
authenticate caller-supplied staged bytes.
The model, processor, detokenizer, and spoken-content ASR all use the
same pinned checkpoint from the Hugging Face cache.

For sampled card controls, `tests/cpp/sampling_draws.cpp` records native
uniform draws and checks their consumption against the actual sampler.
`tests/reference_coupled_generation.py` feeds those draws into the original
categorical selection while retaining the original logits, filtering, and
BF16 probabilities. It saves every probability vector and emitted token.
`tests/compare_sampled_card.py --evidence-dir DIRECTORY` checks the saved
tokens, completion, and waveforms, writes its comparison receipt, and exits
nonzero on failed parity. Successful generation alone is not a parity pass.

Saved semantic evidence can be reproduced without rerunning synthesis:

```bash
python families/lfm2_audio/tests/speech_reference.py \
  --checkpoint CHECKPOINT --jobs speech-jobs.json --output speech-transcripts.json
python families/lfm2_audio/tests/compare_semantics.py \
  --cases cases.json --native native.jsonl --reference reference.json \
  --public public-api.jsonl --transcripts speech-transcripts.json \
  --output semantic-comparison.json
```

The semantic comparator writes every case result and exits nonzero for a failed
content, ordering, history, completion, or API contract. A transcript containing
the expected words does not turn a budget-truncated speech response into a
completed response. The comparator checks that the generated PCM exactly equals
the saved ASR input, so a transcript from a different waveform cannot qualify it.
When reusing older results, external evidence orchestration must verify their
provenance before creating the corresponding input snapshots.

The comparator defaults to the complete model-card scope and fails if required
task, voice, sampling, or context cases are missing. `--scope partial` explicitly
checks only supplied cases and never reports a full-suite pass. A streaming case
still requires an offline counterpart with the same input and generation settings.

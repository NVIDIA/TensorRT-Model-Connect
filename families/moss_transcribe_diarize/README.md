# MOSS-Transcribe-Diarize

`moss_transcribe_diarize` builds
[`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
(`MossTranscribeDiarizeForConditionalGeneration`) for the `speech_transcription`
Task: 16 kHz audio in, one transcript out in which every span carries its start
time, an anonymous speaker label and its end time,
`[0.48][S01]Welcome everyone[1.66][12.26][S02]...`.

## Model and bundle

The checkpoint is a Whisper-medium encoder, a 4x frame merge with a small
adaptor, and a Qwen3 0.6B decoder. The bundle holds two engines:

- `audio.plan`: log-mel features of one 30 s window -> Whisper encoder -> merge
  -> adaptor, 375 audio rows per window;
- `text.plan`: the Qwen3 decoder with a linear KV cache. The prompt's audio
  placeholders take the audio rows inside the engine; profile 0 prefills up to
  256 prompt tokens per call, profile 1 decodes one token.

The native runtime reproduces the checkpoint's processor: the audio is split
into 30 s windows (each yields `(samples - 1) // 1280 + 1` audio tokens), the
Whisper log-mel front end runs per window, and the audio span of the prompt gets
the processor's numeric time markers (the decimal second every 5 s). The prompt
is the official transcription instruction rendered with the checkpoint's chat
template; it is tokenized at build time. Generation is greedy and stops at the
end-of-turn token or at `max_new_tokens`.

## Build and run

```bash
python -m pip install -r families/moss_transcribe_diarize/requirements.txt
python -m tensorrt_model_connect build OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision 10304d926740cf5a5cee05a268086e3265197910 \
  --max-sequence-length 8192 -o moss.bundle          # precision bf16 by default
trtmc transcribe moss.bundle --input meeting.wav --set max_new_tokens=2048
```

- Input: 16 kHz PCM; several channels are averaged. Other sample rates are
  refused, as are `--source-language` values (the model identifies the language
  itself).
- `max_sequence_length` (default 8192) is the KV-cache capacity: the prompt
  (about 86 tokens plus 12.5 tokens per second of audio) and the generated
  transcript must fit in it. A longer audio is refused by name.
- `max_new_tokens` defaults to the checkpoint's generation limit (5120),
  bounded by the capacity left after the prompt.
- Result: `text` is the transcript as the checkpoint's tokenizer decodes it,
  `token_ids` the generated ids (including the end-of-turn token), and
  `segments` the `[start][Sxx]text[end]` spans found by a port of the official
  transcript parser; each segment's text keeps its speaker label, e.g.
  `[S01]Welcome everyone`. Segments carry no token ids.
- `setup_ms` covers the log-mel front end and the audio engine, `prefill_ms`
  the prompt prefill, `decode_ms` the remaining generation.

Precision: `bf16` (default, the checkpoint's own precision) or `fp32`. Norms,
GELU, SiLU and the decoder's attention scores and softmax are evaluated in FP32
and rounded where the reference model rounds them; the encoder uses TensorRT's
attention layer at the build precision. A BF16 bundle may pick a runner-up token
where the FP32 reference's top two logits are within one BF16 spacing, as the
official BF16 pipeline also does.

## Not covered

Custom instructions and hotword hints, sampling, batching, streaming, and the
backbone-only `MossTranscribeDiarizeModel` head.

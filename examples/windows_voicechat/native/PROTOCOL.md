# Native VoiceChat protocol

Run `trtmc_voicechat_bridge.exe --bundle PATH --runtime-root DIR`. Optional arguments
are `--system-prompt TEXT`, `--seed N`, and `--runtime-cache PATH`. `--help` does not
load a model. The bridge validates family `nemotron_voicechat` and backend `trt_rtx`
before loading runtime DLLs. It never selects the ordinary TensorRT backend.

Use anonymous pipes for stdin and stdout. Each message is one UTF-8 JSON object
terminated by a newline; runtime diagnostics are isolated on stderr. Windows
command-line paths and prompts are decoded from Unicode into UTF-8.

## Desktop commands

| Command | Required payload | Behavior |
| --- | --- | --- |
| `audio` | `sampleRate: 16000` and `samples: number[]` or `audio: string` | Enqueue mono microphone PCM. |
| `reset` | None | Flush playback and reset conversation context. |
| `interrupt` | None | Stop playback and clear dialogue/generation state through Nemotron's live control, keeping continuous microphone encoding and input open. Earlier history is cleared without replaying it or restarting the acoustic frontend. Safe while already idle. |
| `finish` | None | Finish input and allow the model's final events to drain. |
| `stop` | None | Cancel and close the session. |
| `ping` | None | Emit `pong`. |

`audio` strings use standard padded base64 of IEEE 754 float32 samples in
little-endian order, optionally declaring `encoding: "f32le"`. A command contains
1–16000 finite samples in [-1, 1]. The microphone client must resample to 16 kHz;
other rates are rejected. Typical blocks contain 320 samples (20 ms). The command
line limit is 1 MiB. The parent must apply bounded pipe backpressure.

## Bridge messages

- `loading`: `backend`, `family`.
- `ready`: `backend`, `family`, `inputSampleRate`, `outputSampleRate`, `loadTimeMs`,
  `protocolVersion: 1`. Start transmitting microphone audio after this message.
- `event`: `kind`, `epoch`, `sequence`, `text`, `isFinal`, `sampleRate`,
  `mediaStartSample`, `mediaEndSample`, `frameIndex`. These preserve the native
  speech-session event fields. All timestamps in media fields are sample indices.
- `event` with `kind: "agent_audio"` additionally contains `audio` (base64 float32
  little endian), `encoding: "f32le"`, and `sampleCount`; playback is mono 48 kHz.
- `flush`: `reason: "reset" | "interrupt"`. Discard queued and scheduled playback.
  An explicit interrupt acknowledgment also contains `interruptStatus`:
  `"context_reset"`. The native session reset has completed before this
  acknowledgment; it does not use the generic API's historical response replay.
  The desktop immediately
  flushes on a stop click and ignores in-flight assistant audio/text until this
  acknowledgment, while microphone capture continues. The subsequent native
  `reset` event ends old partial transcript rows and displays a refresh notice.
- `error`: `message`, `fatal`. Invalid commands produce nonfatal errors; model and
  transport failures are fatal and result in a nonzero process exit.
- `pong`: response to `ping`.
- `stopped`: emitted after a normal session shutdown.

Event kinds are `agent_audio`, `agent_text`, `user_transcript`, `turn_started`,
`turn_finished`, `yielded`, `cancelled`, `reset`, `error`, `input_finished`,
`user_speech_started`, `user_speech_stopped`, `function_call`,
`function_call_started`, `function_response_finished`, `input_cleared`, and
`context_rolled`. Clear playback on `yielded`, `cancelled`, `reset`, `error`, and
`flush`; keep already published playback on `context_rolled`. Accumulate text
deltas by epoch, replacing with the final text when `isFinal` is true.

EOF cancels the session. Disconnect should send `stop`, close stdin, and terminate
the child after a bounded grace period if GPU work or initial model loading has
not returned. No speech inference is simulated in this executable.

## Device-free transport test

Configure this directory with `-DTRTMC_VOICECHAT_PROTOCOL_TEST_ONLY=ON`, then build
and run CTest. This checks wire-format fixtures, base64 validation, amplitude and
memory limits without CUDA, audio devices, or model files.

For a real model check, install `numpy` and `soundfile` in a test environment and
run `verify_voice_session.py --bridge PATH --bundle PATH --input FILE.flac
--output receipt.json`. Use a mono 16 kHz spoken recording, such as the family's
`tests/assets/sample_general_input.flac`. The check streams real audio at microphone
pace and requires a user transcript, response text, and audible model-generated
speech. It writes a JSON receipt, a WAV of generated audio, and stderr diagnostics.
Use `--system-prompt TEXT` and `--runtime-cache PATH` to match an application
configuration. `--lead-silence-seconds 10 --tail-silence-seconds 12` keeps silence
flowing before and after the recorded speech, as a live microphone does. The
receipt records these durations; the audio, transcript, and error criteria are
unchanged. Model response quality must be assessed separately from transport success.

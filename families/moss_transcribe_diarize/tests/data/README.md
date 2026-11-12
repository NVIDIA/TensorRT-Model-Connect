# Test audio

Both clips are derived from the LibriSpeech `test-clean` split (OpenSLR SLR12,
https://www.openslr.org/12/), distributed under the Creative Commons Attribution
4.0 International license (https://creativecommons.org/licenses/by/4.0/).
LibriSpeech was prepared by Vassil Panayotov with the assistance of Daniel Povey
and is derived from LibriVox public-domain audiobooks.

Changes: the listed utterances were decoded from FLAC, joined in the order given
with 0.8 s of silence between them, and written as 16 kHz mono 16-bit PCM WAV.

| File | Length | Utterances (start in the clip) |
|---|---|---|
| `two_speakers_22s.wav` | 21.90 s | `1089-134686-0002` (0.000 s), `121-121726-0000` (7.425 s), `1089-134686-0004` (16.685 s) |
| `continuation_20s.wav` | 19.71 s | `260-123286-0002` (0.000 s), `237-126133-0002` (10.785 s) |

The four-speaker cases join `two_speakers_22s.wav`, 0.8 s of silence and
`continuation_20s.wav` at test time (42.41 s). That input spans two 30 s encoder
windows, so it also exercises chunking and the time markers past 30 s, while
each committed clip stays small.

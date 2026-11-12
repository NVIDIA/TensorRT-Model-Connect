# HuBERT E2E inputs

Both inputs are 16 kHz, mono, 16-bit PCM, the checkpoint's own rate, so the
reference path resamples nothing.

`speech_16k_mono.wav` (4.16 s) is a byte-identical copy of the ASR probe
`probe_02_clean_16k_mono_no_resample.wav`, a deterministic transformation of
the project maintainer's voice recording, distributed under the Apache License
2.0 (see `ASSET_LICENSES.md`, "Maintainer voice recording and derived ASR
probes").

`librispeech-test-clean-6930-75918-0003.wav` (23.315 s) is utterance
`6930-75918-0003` from the LibriSpeech `test-clean` split (OpenSLR SLR12),
byte-identical to the copy recorded in `ASSET_LICENSES.md` ("LibriSpeech
accuracy fixture"). LibriSpeech is distributed under the Creative Commons
Attribution 4.0 International license. LibriSpeech was prepared by Vassil
Panayotov with the assistance of Daniel Povey and is derived from LibriVox
public-domain audiobooks.

- Source and attribution: https://www.openslr.org/12/
- License: https://creativecommons.org/licenses/by/4.0/

The utterance is from the evaluation split, not from the 960 h of LibriSpeech
training audio the checkpoint was fine-tuned on.

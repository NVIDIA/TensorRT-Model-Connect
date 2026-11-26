# VibeVoice-ASR E2E audio

`librispeech-test-clean-6930-75918-0003.wav` is utterance `6930-75918-0003`
from the LibriSpeech `test-clean` split (OpenSLR SLR12), 23.315 seconds of
16 kHz, mono, 16-bit PCM. LibriSpeech was prepared by Vassil Panayotov with the
assistance of Daniel Povey, is derived from LibriVox public-domain audiobooks,
and is distributed under the Creative Commons Attribution 4.0 International
license:

- Source and attribution: https://www.openslr.org/12/
- License: https://creativecommons.org/licenses/by/4.0/

`test_e2e.py` reads it as is for the 16 kHz case and derives the 24 kHz and
repeated inputs of the other cases from it at run time.

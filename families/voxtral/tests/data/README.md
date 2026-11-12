# Voxtral E2E audio fixtures

Both files are 16 kHz, mono, 16-bit PCM and are byte-identical copies of fixtures that already ship in this
repository (see `ASSET_LICENSES.md`).

- `probe_02_clean_16k_mono_no_resample.wav` (4.16 s): the project maintainer's voice recording resampled to 16 kHz
  mono, distributed under the Apache License 2.0.
- `librispeech-test-clean-6930-75918-0003.wav` (23.32 s): utterance `6930-75918-0003` of the LibriSpeech
  `test-clean` split (OpenSLR SLR12), distributed under the Creative Commons Attribution 4.0 International license.
  LibriSpeech was prepared by Vassil Panayotov with the assistance of Daniel Povey and is derived from LibriVox
  public-domain audiobooks.
  - Source and attribution: https://www.openslr.org/12/
  - License: https://creativecommons.org/licenses/by/4.0/

The two-chunk E2E case joins the LibriSpeech utterance, the recording and the LibriSpeech utterance again with 1.5 s
of silence between them (53.8 s, two 30 s encoder chunks); the test writes that input itself.

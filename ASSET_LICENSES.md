# Asset Licenses

This file records explicit provenance and redistribution terms for
third-party, maintainer-supplied, and binary or media assets. Repository-authored
source, configuration, benchmark metadata, and generated golden data are
distributed under the project license unless stated otherwise. Third-party or
externally sourced assets not listed here require separate review.

## Shared vehicle test photograph

The following paths contain byte-identical copies of an original photograph
provided by the project maintainer who took the photograph and authorized its
inclusion and redistribution in this repository under the Apache License 2.0:

- `tests/e2e/models/internvl/data/test_img.jpeg`
- `tests/e2e/models/lance/data/test_img.jpeg`
- `tests/e2e/models/locateanything/data/test_img.jpeg`
- `tests/e2e/models/phi4_multimodal/data/test_img.jpeg`
- `tests/e2e/models/qwen_image/data/test_img.jpeg`
- `tests/e2e/models/qwen_vl/data/test_img.jpeg`
- `tests/e2e/models/sam/data/test_img.jpeg`
- `tests/e2e/models/sam3/data/test_img.jpeg`
- `tests/e2e/models/segformer/data/test_img.jpeg`
- `tests/e2e/models/timm_vit/data/test_img.jpeg`

SHA-256: `d68cb42a55f79e51f71b78cf7d726f01c80a0e2dab8674da6f68361cce004cbc`

## Project-created image fixtures

The following images were created for TensorRT Model Connect testing and are
distributed under the Apache License 2.0 with the rest of the project:

- `tests/assets/test_image.jpg` — synthetic two-color image fixture;
  SHA-256 `d321e0168fc3678faca52c73dffafc80e729f125432cbcaf51f24c5044931c98`
- `tests/assets/test_scene.jpg` — synthetic geometric scene fixture;
  SHA-256 `e1d5893d77507c36b5c7703a3ad1532abc7c2f226f1bb121fa44c08797756363`
- `tests/e2e/models/deepseek_ocr/data/orc_test_img.jpeg` — screenshot of
  project source text created for OCR regression testing; SHA-256
  `d27d4e33afb8e820916b19bffc4c94f1f626536cc3375b5fafeee684b0a3b9b3`

## Maintainer voice recording and derived ASR probes

The project maintainer recorded and supplied the original human-voice fixture
and authorized its inclusion and redistribution under the Apache License 2.0.
Byte-identical copies are stored at:

- `tests/e2e/models/canary/data/Recording.wav`
- `tests/e2e/models/nemotron_speech_streaming/data/Recording.wav`
- `tests/e2e/models/whisper/data/Recording.wav`

SHA-256: `fe14352b6b83009d4e344613fc05b17c0a89a94b0c8502c8422c637928263ca4`

The WAV files below each corresponding `data/asr_probes/` directory are
deterministic transformations of that recording produced by the checked-in
`generate_asr_probe_inputs.py` script. The same relative probe has the same
content in all three model families:

| Probe | SHA-256 |
|---|---|
| `probe_01_clean_48k_stereo_baseline.wav` | `fe14352b6b83009d4e344613fc05b17c0a89a94b0c8502c8422c637928263ca4` |
| `probe_02_clean_16k_mono_no_resample.wav` | `794b1d96d9c28cf39746b37e806b1f1e012b6904526536d1f5b1dec8477e430f` |
| `probe_03_clean_48k_mono_resample.wav` | `dc544108da2b91863a7503695c8af023097601436af4d95cbb948f466e313add` |
| `probe_04_clean_48k_stereo_gain_skew.wav` | `fc2cb56ce769d4b92441a4f8483a65aa65767f8b89d4c06be8c9fb15421e03d7` |
| `probe_05_low_volume_48k_stereo.wav` | `fae8e228584dc68f7f406c98c2a7e04a35fcc5919713fc2e6dd1ff71b1fb3d4c` |
| `probe_06_leading_trailing_silence_48k_stereo.wav` | `7d10e274ccc476f581ed6767d4bed075fdf599e4f4d9e42ca25391415cca4545` |
| `probe_08_noisy_48k_stereo_snr20.wav` | `2e10d12b6d35c871f29d3559fa0975fac81912f0f04a3d6519de315766b9be28` |

The PersonaPlex fixture contains the same maintainer-owned spoken recording,
converted to mono 24 kHz float32 for official-reference regression tests:

- `tests/e2e/models/personaplex/data/Recording.wav`; SHA-256
  `6d5dc6d3b696db0d97d1e45679c81198cb1c9187d5056bb6673ef205dbd4d2e7`

The following NumPy array is project-generated golden model output for that
fixture and is distributed under the Apache License 2.0:

- `tests/e2e/models/personaplex/data/personaplex_recording_official_tokens_greedy.npy`;
  SHA-256 `1f9cbce7a20d09a65069eaa521c3bc5c492f00b27f2d32a4b5c12d1de5a9618c`

## LibriSpeech accuracy fixture

`tests/e2e/models/whisper/data/librispeech-test-clean-6930-75918-0003.wav`
is utterance `6930-75918-0003` from the LibriSpeech `test-clean` split,
distributed by OpenSLR as SLR12 under the Creative Commons Attribution 4.0
International license. LibriSpeech was prepared by Vassil Panayotov with the
assistance of Daniel Povey and is derived from LibriVox public-domain
audiobooks.

- Source and attribution: https://www.openslr.org/12/
- License: https://creativecommons.org/licenses/by/4.0/
- SHA-256: `166d138dc95c706e4eedbebb48f4ac4c8cb1b77ea796c0bc650da518308657e2`

## SANA world-model fixtures

The following files are unmodified copies from NVlabs/Sana revision
`59629fdf790850797cb657bad014fce432bd713d`, which is distributed under the
Apache License 2.0:

- Upstream: https://github.com/NVlabs/Sana/tree/59629fdf790850797cb657bad014fce432bd713d
- `tests/e2e/models/sana_wm/assets/demo_0.png`; SHA-256
  `632754d1cb85bb5d04dc0f81709065892f80fba133d065ad4edd14c1f141d626`
- `tests/e2e/models/sana_wm/assets/demo_0.txt`; SHA-256
  `e6e573dac5002554b0be2bc444b41f77e842f9b35da39984166861273d975901`
- `tests/e2e/models/sana_wm/assets/demo_0_intrinsics.npy`; SHA-256
  `ae21429541b5a61a386322f2e3dd71ab1d1b104aaf44f04d8d20a2c47d97ea1f`

## ELF numerical replay fixtures

The `.f32` files below `tests/e2e/models/elf_flow/data/` are numerical replay
tensors exported for this project from the official ELF evaluator at revision
`1f38c80457d33c95020efdaaf9463823c569c786`. They are distributed as project
test data under the Apache License 2.0. The upstream ELF implementation is MIT
licensed and is attributed in `NOTICE`.

| Relative path below `tests/e2e/models/elf_flow/data/` | SHA-256 |
|---|---|
| `elf-b-de-en-replay/condition_latents.f32` | `9f6083ed8fd0084d0e16ba8616d82fe2412bb4092919a24f661056629113885d` |
| `elf-b-de-en-replay/condition_mask.f32` | `45c048edf5f982926c32922ce9c0c55f3118cb572ea2434cdb0d0816b746ff5b` |
| `elf-b-de-en-replay/initial_latents.f32` | `3840d5e8eabe062c866b67cbe72524ee2b43a162c8ee83da6cf66de471faa18a` |
| `elf-b-de-en-replay/sampling_steps.f32` | `995a97d5841b0f032c17da2eab4ffd1669ee19bf7598abaa5e27a27f0b164779` |
| `elf-b-owt-replay/initial_latents.f32` | `1d8142e76339ecb1463237a9c84d395c274b63cabe53fde78e01f7da30cd3a38` |
| `elf-b-owt-replay/sampling_steps.f32` | `fc47f743709eb080e0c060422c1a39fb55e44d5e7a2f5f9fa3437ca7b269be12` |
| `elf-b-xsum-replay/initial_latents.f32` | `5153428af20ad73e2ba40e72a7d7cbd5dd10bf4cc7026bbdaad91fb12ae4ddd0` |
| `elf-b-xsum-replay/sampling_steps.f32` | `79c790d0590ebf5d1838da98ed968b0e7c6b4309d56d50890ae3b0e025cc982e` |

<!-- Collaborative review anchor: batch 2. -->

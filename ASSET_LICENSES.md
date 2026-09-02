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

- `tests/e2e/models/dinov3/data/test_img.jpeg`
- `tests/e2e/models/internvl/data/test_img.jpeg`
- `tests/e2e/models/lance/data/test_img.jpeg`
- `tests/e2e/models/locateanything/data/test_img.jpeg`
- `tests/e2e/models/moge/data/test_img.jpeg`
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
- `tests/e2e/models/deepseek_ocr/data/orc_test_img.jpeg` — screenshot of
  project source text created for OCR regression testing; SHA-256
  `d27d4e33afb8e820916b19bffc4c94f1f626536cc3375b5fafeee684b0a3b9b3`

## Project overview media

The following project overview images and animation were supplied by the
project maintainer for inclusion and redistribution under the Apache License
2.0:

- `website/static/img/readme/model-connect-overview.png`; SHA-256
  `15e97bc498406629d32eb043e474f835ee88ea6ba4d78cd5ea0cd91c0954b2f4`
- `website/static/img/readme/tensorrt-stack.png`; SHA-256
  `de08eb9b63d6d7808bf11c8893005514d26282b4377f1988b2bdc30366ccad16`
- `TRTMCHERO-small.gif`; SHA-256
  `2dc2b3ac0526d469748a15543f17ca0fd94e2a4caf3729690fa70e6fe35ec43a`

## AI-native development blog artwork

The following artwork was created for the TensorRT-Model-Connect
"AI-Native by Design" engineering blog post and is distributed under the
Apache License 2.0 with the rest of the project:

- `website/static/img/blog/ai-native-by-design/ai-native-by-design-hero.png`;
  1672 x 941 editorial hero generated for this blog with OpenAI image
  generation; SHA-256
  `836697868295331e957a8c53adcce8ad89f3f2cc8e4b9195f1afa56db218b5ce`
- `website/static/img/blog/ai-native-by-design/software-factory.svg`;
  SHA-256 `a08d1353deb8c9a8e843f01ccd8e430983a4f5fca836d3eabb29d3138ac7aca0`
- `website/static/img/blog/ai-native-by-design/isolation-architecture.svg`;
  SHA-256 `f82f6601f3affb80171d3df8422bc409cacf41726235bb71b078f03138a4dae7`

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

## Nemotron VoiceChat report audio

`tests/e2e/models/nemotron_voicechat/assets/sample_general_input.flac` is a
lossless FLAC conversion of the public
[`sample_general.wav`](https://github.com/NVIDIA%2DNeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_general.wav)
fixture at Speech revision `097dfe9e2f55baf653b83035868bdc89849f1b47`,
distributed under the Apache License 2.0. The source WAV SHA-256 is
`481f422a961fb160ddeba9824d55cb7c190c57acb7dc1730a2d595fd078dcb04`;
the FLAC SHA-256 is
`60e1177b7687db259679546ab0a703db4a28157f21bce784fd0b0400559e5a20`.

`tests/e2e/models/nemotron_voicechat/assets/sample_general_reference.flac` is a
lossless FLAC conversion of project-generated, seed-0 reference audio produced
from that input by the pinned public Speech implementation and
`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` checkpoint at revision
`359ada7b1c60851e40ff08065f9b0340244f27e0`. It is standalone-report evidence,
not a waveform-equality acceptance gate. The checkpoint license imposes no
restrictions or obligations on sharing its outputs, and this fixture is
distributed under the Apache License 2.0 with the other project-generated
golden data. The generated PCM16 WAV SHA-256 is
`08605f5205999d02980518939b60442cc8b33f22787fa82fda9eacad222dceab`;
the FLAC SHA-256 is
`6966ddf14fe98fc2375a4caf956001b3231630fdc0e38e108c511e7a18ea6be8`.

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

## LeRobot ACT recorded-observation fixture

The files below `tests/e2e/models/lerobot_act/data/recorded_observation/` are a
lossless PNG decoding and an exact little-endian float32 state row from episode
0, frame 0 of `lerobot/aloha_sim_transfer_cube_human` revision
`6a43d500f101255823a9d2b9dc244eeb01a2cd31`. The source dataset is distributed
under the MIT License. `recorded_observation.json` records the pinned source
parquet and video paths and their SHA-256 digests.

- Dataset: https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/tree/6a43d500f101255823a9d2b9dc244eeb01a2cd31
- `observation.images.top.png`; SHA-256
  `a53369bda31c6563548bd834e88f6640eb28e3233807c499a56d012de992799c`
- `observation.state.f32`; SHA-256
  `40cd79b41ce45e9ffc35f6dc70f74a980d8760857326a9482e109b1d763f54c0`
- `recorded_observation.json`; SHA-256
  `aecd4e0c5123d2e7fb632b32772c60c4175fa37e6425393f182b53e76bd1278f`

## FoundationPose external model artifacts

FoundationPose qualification downloads no model artifact into this repository.
The model-proof cache fetches the official `refine_model.onnx` and
`score_model.onnx` files from NVIDIA NGC model
`nvidia/isaac/foundationpose:1.0.1_onnx`, verifies their exact sizes and
SHA-256 digests, and exposes only private ephemeral copies to the offline proof
container. Users remain responsible for the license terms presented by NGC.
The deterministic RGB+XYZ qualification crops are generated locally by
project-owned Apache-2.0 code and are not derived from the upstream demo data.

<!-- Collaborative review anchor: batch 2. -->

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

- `families/dinov3/tests/data/test_img.jpeg`
- `families/internvl/tests/data/test_img.jpeg`
- `families/lance/tests/data/test_img.jpeg`
- `families/locateanything/tests/data/test_img.jpeg`
- `families/moge/tests/data/test_img.jpeg`
- `families/phi4_multimodal/tests/data/test_img.jpeg`
- `families/qwen_image/tests/data/test_img.jpeg`
- `families/qwen_vl/tests/data/test_img.jpeg`
- `families/sam/tests/data/test_img.jpeg`
- `families/sam3/tests/data/test_img.jpeg`
- `families/segformer/tests/data/test_img.jpeg`
- `families/timm_densenet/tests/data/test_img.jpeg`
- `families/timm_efficientnet/tests/data/test_img.jpeg`
- `families/timm_inception/tests/data/test_img.jpeg`
- `families/timm_mnasnet/tests/data/test_img.jpeg`
- `families/timm_mobilenetv3/tests/data/test_img.jpeg`
- `families/timm_repvgg/tests/data/test_img.jpeg`
- `families/timm_resnet/tests/data/test_img.jpeg`
- `families/timm_vgg/tests/data/test_img.jpeg`
- `families/timm_vit/tests/data/test_img.jpeg`

## Project-created image fixtures

The following images were created for TensorRT Model Connect testing and are
distributed under the Apache License 2.0 with the rest of the project:

- `families/deepseek_ocr/tests/data/orc_test_img.jpeg` — screenshot of
  project source text created for OCR regression testing

## Project overview media

The following project overview images and animation were supplied by the
project maintainer for inclusion and redistribution under the Apache License
2.0:

- `website/static/img/readme/model-connect-overview.png`
- `website/static/img/readme/tensorrt-stack.png`
- `TRTMCHERO-small.gif`

## AI-native development blog artwork

The following artwork was created for the TensorRT-Model-Connect
"AI-Native by Design" engineering blog post and is distributed under the
Apache License 2.0 with the rest of the project:

- `website/static/img/blog/ai-native-by-design/ai-native-by-design-hero.png`;
  1672 x 941 editorial hero generated for this blog with OpenAI image
  generation
- `website/static/img/blog/ai-native-by-design/software-factory.svg`
- `website/static/img/blog/ai-native-by-design/isolation-architecture.svg`

## Maintainer voice recording and derived ASR probes

The project maintainer recorded and supplied the original human-voice fixture
and authorized its inclusion and redistribution under the Apache License 2.0.
Byte-identical copies are stored at:

- `families/canary/tests/data/Recording.wav`
- `families/nemotron_speech_streaming/tests/data/Recording.wav`
- `families/whisper/tests/data/Recording.wav`

The WAV files below each corresponding `data/asr_probes/` directory are
deterministic transformations of that recording produced by the checked-in
`generate_asr_probe_inputs.py` script. The same relative probe has the same
content in all three model families:

- `probe_01_clean_48k_stereo_baseline.wav`
- `probe_02_clean_16k_mono_no_resample.wav`
- `probe_03_clean_48k_mono_resample.wav`
- `probe_04_clean_48k_stereo_gain_skew.wav`
- `probe_05_low_volume_48k_stereo.wav`
- `probe_06_leading_trailing_silence_48k_stereo.wav`
- `probe_08_noisy_48k_stereo_snr20.wav`

The PersonaPlex fixture contains the same maintainer-owned spoken recording,
converted to mono 24 kHz float32 for official-reference regression tests:

- `families/personaplex/tests/data/Recording.wav`

The following NumPy array is project-generated golden model output for that
fixture and is distributed under the Apache License 2.0:

- `families/personaplex/tests/data/personaplex_recording_official_tokens_greedy.npy`

## Nemotron VoiceChat report audio

`families/nemotron_voicechat/tests/assets/sample_general_input.flac` is a
lossless FLAC conversion of the public
[`sample_general.wav`](https://github.com/NVIDIA%2DNeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_general.wav)
fixture at Speech revision `097dfe9e2f55baf653b83035868bdc89849f1b47`,
distributed under the Apache License 2.0.

`families/nemotron_voicechat/tests/assets/sample_general_reference.flac` is a
lossless FLAC conversion of project-generated, seed-0 reference audio produced
from that input by the public Speech implementation and
`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` checkpoint at revision
`359ada7b1c60851e40ff08065f9b0340244f27e0`. It is standalone-report evidence,
not a waveform-equality acceptance gate. The checkpoint license imposes no
restrictions or obligations on sharing its outputs, and this fixture is
distributed under the Apache License 2.0 with the other project-generated
golden data.

## LibriSpeech accuracy fixture

`families/whisper/tests/data/librispeech-test-clean-6930-75918-0003.wav`
is utterance `6930-75918-0003` from the LibriSpeech `test-clean` split,
distributed by OpenSLR as SLR12 under the Creative Commons Attribution 4.0
International license. LibriSpeech was prepared by Vassil Panayotov with the
assistance of Daniel Povey and is derived from LibriVox public-domain
audiobooks.

- Source and attribution: https://www.openslr.org/12/
- License: https://creativecommons.org/licenses/by/4.0/

## SANA world-model fixtures

The following files are unmodified copies from NVlabs/Sana revision
`59629fdf790850797cb657bad014fce432bd713d`, which is distributed under the
Apache License 2.0:

- Upstream: https://github.com/NVlabs/Sana/tree/59629fdf790850797cb657bad014fce432bd713d
- `families/sana_wm/tests/assets/demo_0.png`
- `families/sana_wm/tests/assets/demo_0.txt`
- `families/sana_wm/tests/assets/demo_0_intrinsics.npy`

## ELF numerical replay fixtures

The `.f32` files below `families/elf_flow/tests/data/` are numerical replay
tensors exported for this project from the official ELF evaluator at revision
`1f38c80457d33c95020efdaaf9463823c569c786`. They are distributed as project
test data under the Apache License 2.0. The upstream ELF implementation is MIT
licensed and is attributed in `NOTICE`.

- `elf-b-de-en-replay/condition_latents.f32`
- `elf-b-de-en-replay/condition_mask.f32`
- `elf-b-de-en-replay/initial_latents.f32`
- `elf-b-de-en-replay/sampling_steps.f32`
- `elf-b-owt-replay/initial_latents.f32`
- `elf-b-owt-replay/sampling_steps.f32`
- `elf-b-xsum-replay/initial_latents.f32`
- `elf-b-xsum-replay/sampling_steps.f32`

## LeRobot ACT recorded-observation fixture

The files below `families/lerobot_act/tests/data/recorded_observation/` are a
lossless PNG decoding and an exact little-endian float32 state row from episode
0, frame 0 of `lerobot/aloha_sim_transfer_cube_human` revision
`6a43d500f101255823a9d2b9dc244eeb01a2cd31`. The source dataset is distributed
under the MIT License. `recorded_observation.json` records the exact source
revision, episode, frame, and tensor shapes.

- Dataset: https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/tree/6a43d500f101255823a9d2b9dc244eeb01a2cd31
- `observation.images.top.png`: decoded RGB frame
- `observation.state.f32`: little-endian float32 joint state
- `recorded_observation.json`: source and shape metadata

## FoundationPose external model artifacts

FoundationPose qualification downloads no model artifact into this repository.
The family-owned E2E manifest declares `refine_model.onnx` and
`score_model.onnx` from NVIDIA NGC model
`nvidia/isaac/foundationpose:1.0.1_onnx`. Trusted CI materializes those files
before starting the offline family proof, and the test receives only that
explicit model directory. Users remain responsible for the license terms
presented by NGC.
The deterministic RGB+XYZ qualification crops are generated locally by
project-owned Apache-2.0 code and are not derived from the upstream demo data.

<!-- Collaborative review anchor: batch 2. -->

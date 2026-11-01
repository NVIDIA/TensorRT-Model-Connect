# LeRobot Pi 0.5

Native TensorRT bundles for LeRobot's standard `pi05` flow-matching policy
(PaliGemma Gemma-2B prefix plus a Gemma-300M action expert). A checkpoint
directory written by `lerobot-train`, or the public `lerobot/pi05_libero_base`,
builds straight into a `.bundle` with no manual conversion. Inference needs no
Python and no PyTorch.

```bash
trtmc build lerobot/pi05_libero_base -o pi05.bundle --precision bf16
```

The family is recognised from `type: "pi05"` in `config.json`, never from a
checkpoint name.

## Supported boundary

Anything outside this list fails at build time with the offending field named.

| Area | Supported |
| --- | --- |
| Policy | `type=pi05`, `paligemma_variant=gemma_2b`, `action_expert_variant=gemma_300m` |
| Observation history | `n_obs_steps=1`; no visual or proprioceptive memory |
| Cameras | one or more real cameras sharing one square `3xHxW` shape, plus `empty_cameras` masked placeholders; `image_resolution=[224, 224]` |
| Batch | one |
| Real-time chunking, relative actions | not supported (`rtc_config.enabled`, `use_relative_actions` must be off) |
| Normalization | `IDENTITY`, `MEAN_STD`, `MIN_MAX`, `QUANTILES` (LeRobot's pi05 training default) and `QUANTILE10`, applied to exactly the features LeRobot's processors select |
| Processor steps | the steps LeRobot saves for pi05; any other step, an enabled relative or absolute action step, or a state normalized after the prompt step is rejected |
| Precision | `bf16` (default) and `fp32` |
| Prompt | right-padded to `tokenizer_max_length`; a longer instruction plus state is rejected, not truncated |
| Not claimed | Pi 0, Pi 0-FAST, Pi 0.7, Jetson targets, quantization, multi-GPU |

Statistics are read from the processor state files the checkpoint names
(`policy_preprocessor_step_N_normalizer_processor.safetensors`,
`policy_postprocessor_step_N_unnormalizer_processor.safetensors`), with LeRobot's
rules: the state is normalized only when the normalizer step lists it and
`normalize_observation_keys` keeps it, the action is always unnormalized by the
`ACTION` mode, and a feature with no statistics passes through. A checkpoint with
none, such as `pi05_libero_base`, is therefore unchanged: its "robot units" are the
model's output.

### Tokenizer

LeRobot's pi05 processor loads its tokenizer from `google/paligemma-3b-pt-224`,
and no checkpoint ships it. The builder takes `tokenizer.model` from the
checkpoint directory if present, otherwise that file from the repository at
revision `35e4f46485b4d07967e7e9935bc3786aad50687c` through the Hugging Face
cache, so after the first download builds also work offline. The repository is
gated, as it is for LeRobot itself: downloading needs a Hugging Face login with
access to it. Without access, copy the public
`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model` (same size
and SHA-256 as the pinned file) into the checkpoint directory as
`tokenizer.model`. A file of any other size is refused, and the converted
vocabulary must have 257,152 pieces with pad, eos, bos and unk at ids 0 to 3.
Converting it needs `sentencepiece` and `protobuf` (see `requirements.txt`).

## Bundle

| Section | Content |
| --- | --- |
| `prefix.plan` | cameras (resize, SigLIP, projector), prompt embedding and Gemma-2B; returns each layer's keys and values |
| `denoise.plan` | one flow-matching step of the action expert, including the Euler update |
| `tokenizer.bpe` | PaliGemma SentencePiece BPE in the runtime's flat format |
| `runtime.json` | camera names and order, dimensions, step count, state discretization, normalization |

The task is `images_state_text_to_action_chunk`. The prefix runs once per call
and the denoise step `num_inference_steps` times.

## Runtime contract

```cpp
auto model = trtmc::Model::load("pi05.bundle", options);
auto task = model.task<trtmc::ImagesStateTextToActionChunk>();
auto chunk = task.run({{images, state, "pick up the bowl"}});   // chunk.actions(): [chunk_size, action_dim]
```

- **Cameras** are passed by name, in any order. Names are the checkpoint's
  `observation.images.*` keys, for example `observation.images.image` and
  `observation.images.image2` for `pi05_libero_base`. Every configured camera is
  required; unknown names are rejected. Placeholder cameras are filled by the
  runtime.
- **Images** are HWC, `uint8` (0-255) or `float32` (0-1), RGB, and must match the
  checkpoint's camera size (256x256 for `pi05_libero_base`). Resizing to 224 is
  inside the engine.
- **State** has `state_dim` values in robot units; the runtime normalizes it.
- **Text** is the non-empty task instruction (UTF-8). The runtime builds
  `Task: <text>, State: <256-bin state>;\nAction: ` as LeRobot does.
- **Output** is `[chunk_size, action_dim]` float32 in robot units, already
  unnormalized. `within_training_bounds` is true when every action lies inside the
  dataset's per-dimension min and max, which `lerobot-train` outputs carry; without
  them, as for `pi05_libero_base`, it is false. LeRobot's `n_action_steps` queueing
  stays in the caller.
- **Config** (both optional): `seed` makes the initial flow noise reproducible;
  `noise` supplies it explicitly (`chunk_size * max_action_dim` values, row-major)
  and overrides `seed`. Without either, every call draws fresh noise.
- A bad request (unknown or missing camera, wrong image size, wrong state size,
  empty text) fails with `INVALID_ARGUMENT` and a message naming the input.

## Validation

`tests/` holds observations from three different LIBERO tasks
(`physical-intelligence/libero`, CC-BY-4.0) with the LeRobot `lerobot[pi]==0.6.1`
float32 and bfloat16 outputs for each and the reference's flow noise, plus LeRobot's
own prompt token ids for 22 instructions chosen to stress the tokenizer (whitespace,
digits, punctuation, Unicode, byte fallback, special-looking text, truncation).
The manifests declare the generation-pinned public tokenizer copy in `external_files`.
Internal CI downloads it into the directory `TRTMC_LEROBOT_PI05_MODEL_DIR` names, and
the Community GPU image (`ci/Dockerfile.dependencies`) carries it under `/opt/trtmc-ci`.
The E2E test overlays it on the unchanged checkpoint directory: the Hugging Face cache
at the manifest's revision, or `TRTMC_LEROBOT_PI05_CHECKPOINT_DIR` if set.
`test_e2e.py` first checks that both are staged locally, then
builds the pinned checkpoint, runs each observation through the SDK
and compares all 50x7 actions, checks the bundle's tokenizer against the 22 cases,
and checks that the native runtime links no Python or PyTorch library. Thresholds are
in `thresholds/` and are anchored to LeRobot's own bfloat16-versus-float32 noise.

```bash
python -m pytest families/lerobot_pi05/tests --e2e-model lerobot_pi05 -q
```

`prepare_fixtures.py` regenerates the data; `official_reference.py` is the oracle,
always eager PyTorch.

Measured on one NVIDIA GB300, batch one, `lerobot/pi05_libero_base` revision
`5fcf06fc42b6717e22ab885fd381a096574b12c7`, the three observations above:

| | TensorRT 11.1.0.106, CUDA 13.3 | TensorRT 11.4.0.106, CUDA 13.4 |
| --- | --- | --- |
| bf16 max abs error vs LeRobot fp32 | 4.7e-3 / 1.0e-2 / 6.5e-3 | 5.1e-3 / 1.0e-2 / 5.0e-3 |
| fp32 max abs error vs LeRobot fp32 | 6.6e-7 / 1.2e-6 / 1.3e-6 | 7.5e-7 / 9.8e-7 / 1.1e-6 |
| LeRobot bf16 vs LeRobot fp32 (reference noise) | 3.3e-3 / 1.6e-2 / 5.1e-3 | same |
| bf16 action chunk, p50 / p95 | 33.4-33.6 / 33.7-33.8 ms | 33.2-33.7 / 33.6-34.1 ms |
| fp32 action chunk, p50 / p95 | 119.7-119.9 / 120.5-120.7 ms | 120.2-120.3 / 121.0-121.1 ms |
| Build, bf16 / fp32 | 2 min 25 s / 1 min 28 s | 1 min 53 s / 1 min 22 s |
| Model load, bf16 / fp32 | 1.6-1.7 s / 2.3-2.4 s | 1.5-1.6 s / 2.2 s |
| Peak GPU memory, bf16 / fp32 | 7.9 / 13.3 GiB | 7.9 / 13.2 GiB |
| Bundle size, bf16 / fp32 | 7.6 / 13.0 GB | 7.6 / 13.0 GB |

For comparison, LeRobot's eager PyTorch policy takes about 340 ms per chunk on the same
GPU. The bf16 error is not fixed for a given TensorRT version: each build picks kernels
by timing, and on the first observation four builds gave 3.2e-3, 4.5e-3, 4.7e-3 and
5.1e-3 against a test bound of 6.6e-3 (twice LeRobot's own bf16 error there). The fp32
bundle stays at about 1e-6.

### Trained checkpoints and un-normalization

`pi05_libero_base` carries no statistics, so CI cannot test un-normalization with it.
It was checked outside CI on two published `lerobot-train` outputs with statistics,
building float32 bundles straight from their directories and comparing robot-unit
actions on the same observations and noise against the eager LeRobot policy
(TensorRT 11.4):

| Checkpoint | Normalization | Max abs error |
| --- | --- | --- |
| `lerobot/pi05-libero` @ `9ba5431256f4136a01c224ab0307b0d3035cc92c` | `MEAN_STD`, bfloat16 weights | 1.2e-6 / 8.1e-7 / 1.1e-6 |
| `lerobot/pi05_libero_finetuned_quantiles_v044` @ `b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd` | `QUANTILES` (LeRobot's pi05 default) | 8.3e-7 / 8.3e-7 / 1.1e-6 |

These outputs store weights in bfloat16 under a `model.` name prefix and may keep an
untied `embed_tokens` table; the loader handles all three. Both enable `torch.compile`
in their config, which changes LeRobot's own numerics at about 1e-3, which is why the
oracle is always eager.

## Known limits

- Qualified on one NVIDIA GB300 with TensorRT 11.1 (CUDA 13.3) and 11.4 (CUDA 13.4). Other
  GPUs, TensorRT versions and Jetson are not.
- Batch one, one observation step, square cameras; Pi 0, Pi 0-FAST, Pi 0.7, memory and
  real-time chunking are rejected at build time.
- An instruction plus state longer than `tokenizer_max_length` is rejected, where LeRobot
  would truncate it silently.
- `n_action_steps` queueing stays in the caller.
- Between the two engines the prefix keys and values pass through host memory, and the
  vision tower also runs on placeholder camera slots whose tokens are masked. The latency
  above includes both; neither affects results.
- Building needs `sentencepiece`, `protobuf` and the PaliGemma tokenizer: in the
  checkpoint directory, or in the Hugging Face cache, which the first download from the
  gated repository fills.
- The LeRobot side is not part of this repository. `lerobot-rollout` ships with LeRobot
  0.6.1, but `lerobot-export` does not, so the bundle has not been driven through either;
  only the runtime contract above is provided.

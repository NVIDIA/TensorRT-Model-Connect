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
| Normalization | `IDENTITY` and `MEAN_STD`; `QUANTILES` is rejected when the checkpoint carries statistics |
| Precision | `bf16` (default) and `fp32` |
| Prompt | right-padded to `tokenizer_max_length`; a longer instruction plus state is rejected, not truncated |
| Not claimed | Pi 0, Pi 0-FAST, Pi 0.7, Jetson targets, quantization, multi-GPU |

Statistics are read from the processor state files the checkpoint names
(`policy_preprocessor_step_N_normalizer_processor.safetensors`,
`policy_postprocessor_step_N_unnormalizer_processor.safetensors`). A checkpoint
with none, such as `pi05_libero_base`, is passed through unchanged, exactly as
LeRobot does: its "robot units" are the model's output.

### Tokenizer

`google/paligemma-3b-pt-224` is gated and no checkpoint ships it. The builder
takes `tokenizer.model` from the checkpoint directory if present, otherwise the
public `big_vision/paligemma_tokenizer.model`, which was checked byte for byte
against the gated repository's file when this family was added. A file of any
other size is refused, and the converted vocabulary must have 257,152 pieces with
pad, eos, bos and unk at ids 0 to 3. Converting it needs `sentencepiece` and
`protobuf` (see `requirements.txt`).

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
  unnormalized. `within_training_bounds` is `false`: the checkpoint carries no
  training range to test against. LeRobot's `n_action_steps` queueing stays in the
  caller.
- **Config** (both optional): `seed` makes the initial flow noise reproducible;
  `noise` supplies it explicitly (`chunk_size * max_action_dim` values, row-major)
  and overrides `seed`. Without either, every call draws fresh noise.
- A bad request (unknown or missing camera, wrong image size, wrong state size,
  empty text) fails with `INVALID_ARGUMENT` and a message naming the input.

## Validation

`tests/` holds three real LIBERO frames (`physical-intelligence/libero`, CC-BY-4.0)
and the LeRobot `lerobot[pi]==0.6.1` float32 and bfloat16 outputs for each, with
the reference's flow noise. `test_e2e.py` builds the pinned checkpoint, runs each
frame through the SDK and compares all 50x7 actions; thresholds are in
`thresholds/` and are anchored to LeRobot's own bfloat16-versus-float32 noise.

```bash
python -m pytest families/lerobot_pi05/tests --e2e-model lerobot_pi05 -q
```

`prepare_fixtures.py` regenerates the data; `official_reference.py` is the oracle.

Measured on one NVIDIA GB300, TensorRT 11.1, CUDA 13, batch one,
`lerobot/pi05_libero_base` revision `5fcf06fc42b6717e22ab885fd381a096574b12c7`:

| | bf16 | fp32 |
| --- | --- | --- |
| Max abs error vs LeRobot fp32, 3 frames | 3.2e-3 / 6.4e-3 / 4.3e-3 | 6.3e-7 / 1.1e-6 / 4.8e-7 |
| LeRobot bf16 vs LeRobot fp32 (reference noise) | 3.3e-3 / 1.5e-2 / 4.6e-3 | - |
| Action chunk, p50 / p95 | 34.6 / 34.7 ms | 119.9 / 120.7 ms |
| Build (checkpoint to bundle) | 2 min 35 s | 1 min 30 s |
| Bundle size | 7.6 GB | 13.0 GB |
| Model load | 1.7 s | 2.3 s |
| Peak GPU memory | 8.1 GiB | 13.3 GiB |

For comparison, LeRobot's eager PyTorch policy takes about 340 ms per chunk on the
same GPU. Other GPUs and TensorRT versions are not qualified.

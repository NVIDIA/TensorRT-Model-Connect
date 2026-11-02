# LeRobot GR00T N1.7

Native TensorRT bundles for LeRobot's `groot` policy, NVIDIA Isaac GR00T N1.7: a
Cosmos-Reason2-2B (Qwen3-VL) backbone truncated to its first 16 text layers, and a
flow-matching action head (vision-language self-attention, a 32-block DiT that
alternates cross-attention over text and image tokens, and embodiment-specific
state/action encoders and decoder). A checkpoint directory written by
`lerobot-train`, such as the public `nvidia/gr00t17-lerobot-libero_spatial-640`,
builds straight into a `.bundle` with no manual conversion. Inference needs no
Python and no PyTorch.

```bash
trtmc build nvidia/gr00t17-lerobot-libero_spatial-640 -o groot.bundle --precision bf16
```

The family is recognised from `type: "groot"` in `config.json`, never from a
checkpoint name. LeRobot 0.6 saves only GR00T N1.7 policies under that type.

## Supported boundary

Anything outside this list fails at build time with the offending field named.

| Area | Supported |
| --- | --- |
| Policy | `type=groot` (GR00T N1.7), `max_state_dim=max_action_dim=132`, the native 40-step action horizon; `chunk_size`/`n_action_steps` choose how many steps a call returns |
| Observation history | `n_obs_steps=1`, `state_horizon=1`, no video history |
| Embodiment | the checkpoint's `embodiment_tag`, which must be in its `embodiment_mapping`; the builder keeps only that embodiment's weights |
| Cameras | square cameras already at the processor's `image_target_size`, fed in LeRobot's order (sorted names, or `video_modality_keys`) |
| Validated | the LIBERO configuration: embodiment `libero_sim`, two 256x256 cameras, 8-D state, 7-D actions, 16-step chunks |
| Image preprocessing | `use_albumentations` evaluation crop (OpenCV `INTER_AREA`, floored centre crop); no letter boxing |
| Backbone | `nvidia/Cosmos-Reason2-2B` |
| Normalization | min/max state and action statistics from the processor state files, with LeRobot's clipping |
| Action decoding | `action_decode_transform` `libero` (binary gripper) or none |
| Batch | one |
| Real-time chunking, relative actions, PEFT adapters | not supported (`use_relative_actions`, `use_peft` must be off; grouped `raw_stats`/`modality_config` are rejected) |
| Processor steps | exactly the steps LeRobot 0.6.1 saves for an N1.7 policy; any other step is rejected |
| Precision | `bf16` (default) and `fp32` |
| Instruction | ASCII, up to 256 tokens and 4096 bytes; formalised as LeRobot does (lower case, keep letters, digits, whitespace and `_`), or used verbatim when `formalize_language` is off |
| Not claimed | GR00T N1.5, other backbones, Jetson targets, quantization, multi-GPU |

### Tokenizer

LeRobot's groot processor loads its tokenizer from `nvidia/Cosmos-Reason2-2B`, and
no checkpoint ships it. The builder takes `tokenizer.json` from the checkpoint
directory if present, otherwise that file from the repository at revision
`9ce19a195e423419c349abfc86fd07178b230561` through the Hugging Face cache, so after
the first download builds also work offline. The repository is gated: downloading
needs a Hugging Face login that accepted its terms. Without one, copy
`tokenizer.json` of the public `Qwen/Qwen3-VL-2B-Instruct` at revision
`89644892e4d85e24eaac8bacfd4f463576704203` into the checkpoint directory: at these
revisions the two repositories serve identical tokenizer and processor files. A file
of any other size, or with other chat-template token ids, is refused.

## Bundle

| Section | Content |
| --- | --- |
| `condition.plan` | vision tower (float32 in either precision), DeepStack, the 16 kept text layers, the action head's vision-language layers, the state token, and the keys/values of every DiT cross-attention block; the prompt length is dynamic |
| `denoise.plan` | one flow-matching step: action encoder, DiT, action decoder and the Euler update |
| `tokenizer.json` | the Qwen3-VL byte-level BPE |
| `runtime.json` | camera names and order, crop, dimensions, step count, instruction rules, normalization |

The task is `images_state_text_to_action_chunk`. The condition engine runs once per
call and the denoise step `num_inference_steps` times; the keys/values it hands over
stay on the device.

## Runtime contract

```cpp
auto model = trtmc::Model::load("groot.bundle", options);
auto task = model.task<trtmc::ImagesStateTextToActionChunk>();
auto chunk = task.run({{images, state, "pick up the black bowl and place it on the plate"}});
// chunk.actions(): [chunk_size, action_dim]
```

- **Cameras** are passed by name, in any order. Names are the checkpoint's
  `observation.images.*` keys, `observation.images.image` and
  `observation.images.wrist_image` for the LIBERO checkpoints. Every configured camera
  is required; unknown names are rejected.
- **Images** are HWC, `uint8` (0-255) or `float32` (0-1, truncated to `uint8` as
  LeRobot's float image path does), RGB, at the checkpoint's camera size (256x256).
  The crop, resize, normalization and patching match LeRobot bit for bit.
- **State** has `state_dim` values in robot units; the runtime normalizes and pads it.
- **Text** is the task instruction. Empty text becomes LeRobot's default, `Perform the
  task.`; the runtime then builds the Qwen3-VL chat prompt with the image tokens as
  LeRobot does.
- **Output** is `[chunk_size, action_dim]` float32 in robot units: the first
  `chunk_size` steps of the 40-step prediction, clipped and min/max unnormalized, with
  a LIBERO gripper as `-1`/`+1`. `within_training_bounds` is true when no component
  needed the clip; the binary LIBERO gripper, whose prediction routinely overshoots,
  is exempt. LeRobot's `n_action_steps` queueing stays in the caller; the Task keeps
  no state between calls.
- **Config** (both optional): `seed` makes the initial flow noise reproducible;
  `noise` supplies it explicitly (`40 * 132` values, row-major) and overrides `seed`.
  Without either, every call draws fresh noise.
- A bad request (unknown or missing camera, wrong image size, wrong state size,
  non-ASCII instruction, instruction over 256 tokens or 4096 bytes) fails with
  `INVALID_ARGUMENT` and a message naming the input.

## Validation

`tests/` holds observations from three different LIBERO-spatial tasks
(`physical-intelligence/libero`, CC-BY-4.0) with the LeRobot `lerobot[groot]==0.6.1`
float32 and bfloat16 outputs for each and the reference's flow noise. One case also
runs its images and state with another task's instruction, in between two runs with
its own, to show that the bundle follows the text and keeps nothing between calls.
`tokenizer_golden.json` holds LeRobot's prompt ids for 17 instructions chosen to
stress the formalisation and the tokenizer, and for 5 that a checkpoint with
`formalize_language` off passes verbatim; each camera also has LeRobot's
cropped and resized image, so the runtime's preprocessing is checked bit for bit.
The manifests declare the commit-pinned public tokenizer copy in `external_files`.
Internal CI downloads it into the directory `TRTMC_LEROBOT_GROOT_MODEL_DIR` names, and
the Community GPU image (`ci/Dockerfile.dependencies`) carries it under `/opt/trtmc-ci`.
The E2E test overlays it on the unchanged checkpoint directory: the Hugging Face cache
at the manifest's revision, or `TRTMC_LEROBOT_GROOT_CHECKPOINT_DIR` if set.
`test_e2e.py` first checks that both are staged locally, then builds the pinned
checkpoint, runs each observation through the SDK and compares all 16x7 actions,
checks the preprocessing against LeRobot's, and checks that the native runtime links
no Python or PyTorch library. Thresholds are in `thresholds/` and are anchored to
LeRobot's own bfloat16-versus-float32 noise.

```bash
python -m pytest families/lerobot_groot/tests --e2e-model lerobot_groot -q
```

`prepare_fixtures.py` regenerates the data; `official_reference.py` is the oracle,
eager PyTorch with the base model the checkpoint names (`nvidia/GR00T-N1.7-3B` at
`2fc962b973bccdd5d8ce4f67cc63b264d6886495`).

Measured on one NVIDIA RTX PRO 6000 Blackwell Max-Q, TensorRT 11.4.0.106 with CUDA
13.4, batch one, `nvidia/gr00t17-lerobot-libero_spatial-640` revision
`32a6ec786d6509df31b40392b4e4dcdda78c0f11`, the three observations above (the
alternate instruction in brackets):

| | bf16 | fp32 |
| --- | --- | --- |
| Max abs error vs LeRobot fp32 | 3.2e-3 / 3.7e-3 / 9.6e-3 (7.2e-3) | 1.9e-6 / 4.8e-7 / 5.2e-6 (6.9e-6) |
| LeRobot bf16 vs LeRobot fp32 (reference noise) | 6.3e-3 / 2.6e-3 / 1.5e-2 (3.7e-3) | |
| Action chunk, p50 / p95 | 24.3-24.6 / 24.4-24.8 ms | 49.2-50.6 / 49.4-50.9 ms |
| Build | 1 min 20 s | 51 s |
| Model load | 1.8 s | 2.7 s |
| Peak GPU memory | 6.8 GiB | 11.3 GiB |
| Bundle size | 6.5 GB | 11.3 GB |

For comparison, LeRobot's eager PyTorch policy takes about 203 ms per chunk on the same
GPU with its default bfloat16 autocast (163 ms in float32). The vision tower runs in
float32 in either bundle: in bfloat16 it is the largest error source, and keeping it in
float32 lowers the first observation's error from 1.7e-2 to 9.6e-3 for about 7 ms per
chunk. Three bf16 builds gave identical actions on this GPU.

### Other `lerobot-train` outputs

Two more published LIBERO fine-tunes with the same configuration but their own weights
and statistics were checked outside CI, building float32 bundles straight from their
directories and comparing robot-unit actions on the same observations and noise
against the eager LeRobot policy (TensorRT 11.4):

| Checkpoint | Max abs error |
| --- | --- |
| `nvidia/gr00t17-lerobot-libero_10-640` @ `5ee08ab09fac5c5ef2388a14c882ea825ac861db` | 3.2e-6 / 1.2e-6 / 6.0e-6 |
| `nvidia/gr00t17-lerobot-libero_goal-640` @ `436c57c0eb7a90be54270abf3977668b2084ad75` | 3.3e-6 / 1.7e-6 / 2.3e-6 |

## Known limits

- Qualified on one NVIDIA RTX PRO 6000 Blackwell with TensorRT 11.4 (CUDA 13.4). Other
  GPUs, TensorRT versions and Jetson are not.
- Validated for the LIBERO configuration only. Other embodiments, camera layouts, state
  and action sizes and chunk lengths build, but are not qualified.
- Batch one, one observation step, square cameras at the target size; GR00T N1.5,
  relative actions, real-time chunking and PEFT adapters are rejected at build time.
- Instructions must be ASCII. LeRobot also takes other text, which needs Python's Unicode
  `lower()` and `isalnum()` and the tokenizer's NFC normalization; this runtime rejects it
  rather than approximates them.
- `n_action_steps` queueing stays in the caller.
- The LeRobot side is not part of this repository. `lerobot-rollout` ships with LeRobot
  0.6.1, but `lerobot-export` does not, so the bundle has not been driven through either;
  only the runtime contract above is provided.

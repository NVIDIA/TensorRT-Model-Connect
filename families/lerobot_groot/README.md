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
| `condition.plan` | vision tower (in the build precision), DeepStack, the 16 kept text layers, the action head's vision-language layers, the state token, and the keys/values of every DiT cross-attention block; the prompt length is dynamic |
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
LeRobot's own bfloat16-versus-float32 noise: about twice LeRobot's bfloat16 error on
each observation, except the bf16 maximum and mean of `ep1267_f080`, where LeRobot's
bfloat16 error is the smallest of the three; those are set from the measured bundle
error with headroom (see below).

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
GPU with its default bfloat16 autocast (163 ms in float32). Three bf16 builds gave
identical actions on this GPU. The bf16 column above was measured with the vision tower
in float32.

The bf16 bundle runs the whole vision tower in bfloat16, as LeRobot's autocast does
(the fine-tune keeps the base model's bfloat16 vision weights): every matrix product
in bfloat16, with norms, softmax, gelu, the rotary embedding and the residual stream
in float32 like the rest of the graph. Measured with a Python harness that runs the
two plans as the runtime does (GPU time of the condition plan and four denoising
steps), TensorRT 11.4.0.106, on the three observations above and on 200 held-out
LIBERO-spatial observations (mean absolute error of the six arm dimensions against
LeRobot float32; LeRobot bfloat16 has 9.3e-4):

| | B200, vision float32 | B200 | B300, vision float32 | B300 |
| --- | --- | --- | --- | --- |
| Policy step | 19.1 ms | 10.3 ms | 17.8 ms | 9.9 ms |
| Vision tower alone | 10.4 ms | 1.7 ms | 9.3 ms | 1.5 ms |
| Max abs error vs LeRobot fp32 | 3.6e-3 / 2.9e-3 / 8.7e-3 | 4.8e-3 / 6.2e-3 / 1.8e-2 | 2.7e-3 / 2.3e-3 / 7.8e-3 | 4.8e-3 / 6.2e-3 / 1.8e-2 |
| Held-out mean arm error | 6.8e-4 | 9.1e-4 | 6.7e-4 | 9.1e-4 |
| Bundle size | 6.5 GB | 5.7 GB | 6.5 GB | 5.7 GB |

The held-out error stays below LeRobot's own bfloat16 policy (9.3e-4), and the median
per-observation maximum error (4.3e-3) matches LeRobot bfloat16's (4.1e-3). On
`ep1267_f080`, LeRobot bfloat16 is unusually close to float32 (2.6e-3, against a
held-out median of 4.1e-3), so twice that error was the tightest limit of the three;
the bundle's 6.2e-3 to 6.5e-3 (B200 and B300, TensorRT 11.4 and 11.1) is past it, and
that case's bf16 limits are now 9.5e-3 maximum and 1.4e-3 mean, about 1.5 times the
largest measured value. A build with one block's query/key/value bias dropped, or
with every attention output projection scaled by 1.02, still fails that case.

### Other `lerobot-train` outputs

Two more published LIBERO fine-tunes with the same configuration but their own weights
and statistics were checked outside CI, building float32 bundles straight from their
directories and comparing robot-unit actions on the same observations and noise
against the eager LeRobot policy (TensorRT 11.4):

| Checkpoint | Max abs error |
| --- | --- |
| `nvidia/gr00t17-lerobot-libero_10-640` @ `5ee08ab09fac5c5ef2388a14c882ea825ac861db` | 3.2e-6 / 1.2e-6 / 6.0e-6 |
| `nvidia/gr00t17-lerobot-libero_goal-640` @ `436c57c0eb7a90be54270abf3977668b2084ad75` | 3.3e-6 / 1.7e-6 / 2.3e-6 |

## FP8 quantization study (not shipped)

FP8 self-quantization of the flow-matching DiT was built and measured on TensorRT 11.4
(B200, B300) and 11.1 (B200) on top of the bf16 vision tower. It is not shipped: one
policy step is only 1.06x faster than the bf16 bundle, the mean action error grows to
1.8x that of the bf16 bundle, and one of the three committed bf16 test observations fails
its thresholds. This section records the recipe, the method and the numbers, so that the
study does not have to be repeated. The FP8 code was not merged; this family has no
quantized build.

### Summary and decision

Policy step = the condition engine once and the denoise engine four times (GPU time,
TensorRT 11.4; 200 held-out LIBERO-spatial observations):

| Item | bf16 bundle | FP8 bundle | FP8 vs bf16 |
| --- | --- | --- | --- |
| Policy step, B200 / B300 | 12.15 / 11.00 ms | 11.49 / 10.29 ms | 1.06x / 1.06x to 1.07x |
| Four denoising steps, B200 / B300 | 6.25 / 5.70 ms | 5.62 / 5.05 ms | 1.11x to 1.12x / 1.11x to 1.13x |
| Condition engine, B200 / B300 | 5.90 / 5.29 ms | 5.87 / 5.24 ms | unchanged |
| Bundle | 5.82 GB | 4.89 GB | -16% |
| Held-out mean arm error vs LeRobot fp32 | 8.5e-4 | 1.5e-3 | 1.78x |
| Committed bf16 E2E thresholds | 3 of 3 pass | 2 of 3 pass | `ep1267_f080` fails |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for GR00T.

- Speed: FP8 saves 0.6 to 0.7 ms of a 11 to 12 ms step. Only the DiT is quantized, and
  the DiT is half of the step; the other half, the condition engine (vision tower,
  backbone, VL blocks), stays bf16.
- The DiT runs on 41 tokens, so its GEMMs are small and FP8 makes one denoising step only
  1.11x to 1.13x faster.
- Quality: the mean arm error is 1.78x that of the bf16 bundle and 1.6x LeRobot's own
  bfloat16-versus-float32 difference. An FP8 build would need its own, looser E2E
  thresholds.
- Plan size: the denoise plan is 41% smaller, but the bundle only 16%.

### Recipe that was measured

No FP8 or NVFP4 GR00T checkpoint exists on the Hub (`nvidia/GR00T-N1.7-3B` and the
LeRobot fine-tunes are bf16 or fp32 only), and no official GR00T quantization config
exists.

| Choice | Decision |
| --- | --- |
| Quantized (192 projections) | `attn1.to_q/k/v`, `attn1.to_out.0`, `ff.net.0.proj`, `ff.net.2` of all 32 DiT blocks; in the plans: 128 FP8 GEMMs in the denoise engine and 1 in the condition engine (the 32 cross-attention key/value projections, merged by TensorRT) |
| bf16 | vision tower, the 16 backbone layers, the 4 VL self-attention blocks, attention matmuls, AdaLN modulation, timestep embedders, DiT output projections, embodiment-specific encoders and decoder |
| Weight scale | per tensor, amax / 448, from the checkpoint at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) through LeRobot's float32 policy, all four denoising steps, 128 LIBERO-spatial observations (32 episodes x 4 frames) |
| Outlier rule | a scale above 20x the median of its kind stays bf16; measured: none in the DiT (largest ratio 3.99, `ff.net.2`) |
| Dequantize type | bf16 (the build precision) |
| Build | `precision=bf16` only; fp32 with FP8 and already quantized checkpoints refused |

### Method

- Checkpoint `nvidia/gr00t17-lerobot-libero_spatial-640` at the manifest revision; plans
  from `model.build` (bf16, builder optimization level 3).
- A Python harness runs the two plans as the runtime does (prompt, image patches, state
  normalization, masks, four Euler steps, unnormalization, LIBERO gripper).
- Speed: GPU time of one policy step (CUDA events, inputs on the device), median of 20
  interleaved rounds of 10 steps (15 on 11.1). On 11.4 each variant was built twice and
  all four plans were timed in the same job, one job per GPU type.
- Quality: 200 held-out LIBERO-spatial observations (50 episodes x 4 frames, none used
  for calibration or by the tests) and the 3 committed test observations, fixed noise per
  observation, 16 x 7 actions in robot units. Arm = the 6 continuous dimensions, gripper =
  the binary LIBERO command (flip = sign change). Reference: LeRobot `lerobot[groot]==0.6.1`
  in float32.
- Non-finite scan: every condition output and every denoise output of all four steps on
  the 203 observations, every build.

### Results

Speed and size (x is bf16 time over FP8 time; two builds of each variant on 11.4):

| GPU | TensorRT | bf16 step | FP8 step | x step | bf16 / FP8 denoise x4 | x denoise | denoise plan bf16 to FP8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B200 | 11.4 | 12.15 / 12.17 ms | 11.49 / 11.47 ms | 1.06 | 6.25-6.28 / 5.61-5.62 ms | 1.11 to 1.12 | 2004 to 1178 MB |
| B300 | 11.4 | 11.00 / 10.94 ms | 10.29 / 10.35 ms | 1.06 to 1.07 | 5.67-5.70 / 5.05-5.09 ms | 1.11 to 1.13 | 2004 to 1178 MB |
| B200 | 11.1 | 11.81 ms | 11.12 ms | 1.06 | 5.97 / 5.29 ms | 1.13 | 2004 to 1179 MB |

The condition plan goes from 3809 to 3708 MB, the bundle from 5.82 to 4.89 GB. Build time
is 148 to 248 s for bf16 and 252 to 337 s for FP8.

Quality on the 200 held-out observations (B200 and B300 give the same numbers on 11.4):

| Engine | Arm mean abs vs LeRobot fp32 | Ratio to bf16 bundle | p99 | Median per-observation max | Gripper flips | Arm mean abs vs bf16 bundle |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 bundle, 11.4 | 8.46e-4 | 1.00 | 5.77e-3 | 4.03e-3 | 0.031% | |
| FP8 bundle, 11.4 | 1.50e-3 | 1.78 | 8.53e-3 | 7.82e-3 | 0.031% | 1.27e-3 |
| bf16 bundle, 11.1 | 8.39e-4 | 1.00 | 5.59e-3 | 4.04e-3 | 0.094% | |
| FP8 bundle, 11.1 | 1.46e-3 | 1.74 | 8.33e-3 | 7.27e-3 | 0.031% | 1.41e-3 |
| LeRobot bf16 policy | 9.27e-4 | 1.10 | 6.79e-3 | 4.06e-3 | 0.031% | |

The committed test observations against the bf16 E2E thresholds (TensorRT 11.4; max abs /
mean abs vs LeRobot fp32, max abs vs LeRobot bf16; cosine is at least 0.99998 everywhere):

| Observation | Threshold | bf16 bundle | FP8 bundle |
| --- | --- | --- | --- |
| `ep1262_f040` | 1.26e-2 / 1.8e-3 / 1.9e-2 | 3.64e-3 / 7.08e-4 / 6.41e-3 | 9.94e-3 / 1.49e-3 / 9.16e-3 |
| `ep1267_f080` | 5.1e-3 / 9.4e-4 / 7.7e-3 | 3.27e-3 / 5.83e-4 / 5.49e-3 | 8.16e-3 / 1.22e-3 / 9.16e-3 (fails all three) |
| `ep1272_f000` | 3.0e-2 / 2.6e-3 / 4.5e-2 | 7.38e-3 / 8.07e-4 / 8.07e-3 | 7.12e-3 / 1.10e-3 / 1.44e-2 |

TensorRT 11.1 gives the same picture (`ep1267_f080` FP8: 7.24e-3 / 1.03e-3 / 8.24e-3).

Other recipes, measured earlier with the vision tower in float32 (TensorRT 11.4; B200 /
B300; the bf16 step was then 19.2 / 17.8 ms):

| Variant | x step | x denoise x4 | Bundle | Arm mean abs vs LeRobot fp32 |
| --- | --- | --- | --- | --- |
| DiT, all 6 projections (the recipe above) | 1.03 to 1.04 / 1.03 to 1.04 | 1.11 to 1.13 / 1.10 to 1.12 | 5.55 GB | 1.37e-3 / 1.38e-3 |
| DiT without `to_out.0` | 1.03 / 1.03 | 1.10 / 1.10 | 5.63 GB | 1.26e-3 / 1.29e-3 |
| Backbone only (16 layers) | 1.01 (B200) | 1.01 | 5.69 GB | 2.36e-3 |
| Backbone MLP + DiT | 1.04 / 1.05 | 1.08 / 1.10 | 4.85 GB | 2.32e-3 / 2.49e-3 |
| Backbone, VL blocks and DiT | 1.05 / 1.06 | 1.11 to 1.12 / 1.12 | 4.56 GB | 2.93e-3 / 3.05e-3 |

FP8 of the backbone and the VL blocks adds 1% to 2% for twice the action error.

### NVFP4

A harness prototype with the DiT feed-forward projections in NVFP4 (per-16 block scales,
dynamic activation quantize; q/k/v in FP8), measured with the float32 vision tower on
TensorRT 11.4, is slower than bf16: 0.97x (B200) and 0.98x (B300) per step, 0.92x and
0.95x per denoise call, with a mean arm error of 2.75e-3. At 41 tokens the dynamic block
quantize costs more than the 4-bit weight reads save.

### What was not tried

| Item | Note |
| --- | --- |
| FP8 of the bf16 vision tower | not measured |
| FP8 attention matmuls, percentile or entropy calibration | not measured |
| Closed-loop LIBERO success rate, the native runtime with an FP8 bundle, H100 | not measured |

### When to revisit

- A deployment with more denoising steps per call: the DiT share of the step, and so the
  FP8 gain, grows with the step count.
- A quality target that accepts a mean arm error of about 1.5e-3 (1.6x LeRobot's own
  bfloat16 noise), with FP8-specific E2E thresholds.

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

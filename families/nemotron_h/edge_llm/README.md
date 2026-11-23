# Nemotron-H Edge-LLM execution

The family owns source-policy admission, command mapping, tokenizer/EOS semantics,
engine composition and runtime orchestration. TensorRT Edge-LLM owns model graphs,
lowering and execution. Provision the [optional native SDK](../../../cmake/edge_llm/README.md)
from the official release selected at configure time. The default is 0.11.0.
To select 0.12.0 (revision `513dcd18291e587c80928911a480761352f3c2d8`),
configure with `-DTRTMC_EDGELLM_RELEASE=0.12.0`. Build on the machine
that will run the model. Do not mix an SDK from one release with another
release's builder or runtime.

For 0.12.0, pass `--edge-builder-python /absolute/path/to/python` when the
builder is installed in a separate environment. A PyPI installation and a
source installation are discovered the same way: the interpreter must import
the selected Edge modules, Transformers 5.17.0 and the matching TensorRT
version. If omitted, the adapter probes the current interpreter. It does not
install Python packages or replace the native C++ SDK. With 0.11.0, the
provisioned SDK interpreter remains the fallback for an incompatible ambient
interpreter.

## Ordinary and paired builds

Ordinary compatible text builds use the release’s checkpoint-direct Python builder
with `--components llm` and the original checkpoint. Plain source weights use
FP16 compute. For the RedHatAI Lightning 30B FP8 checkpoint, the
[Edge 0.12.0 catalog](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/513dcd18291e587c80928911a480761352f3c2d8/docs/source/user_guide/getting_started/supported-models.md)
specifies checkpoint-direct execution with **FP16 routed experts**. The
builder applies the stored FP8 weight scales when materializing those experts;
the FP8 checkpoint label does not mean they execute in FP8. This checkpoint
has no K/V calibration scales, so the Edge plan uses FP16 KV storage. For packed
sources, only supported external-weight kinds are requested; FP16 bias tensors
are baked into the engine rather than externalized through a missing recipe.

The explicit Lightning DFlash and DSpark pairs use the original ONNX exporter and
native ONNX builder. Enable `TRTMC_EDGELLM_ALL_KERNELS=ON` and
`TRTMC_EDGELLM_ONNX=ON`, set `CMAKE_PREFIX_PATH` to the SDK installation, and
configure the runtime with `TRTMC_ENABLE_EDGELLM=ON`. Add
`--execution-variant dflash --companion draft=/path/to/draft` to the build CLI,
or wrap the request with this family's `with_execution(request, inputs)`
before calling `build`, as the Python example below shows.

Both paired engines are mandatory. The ONNX graph supplies the intermediate
Mamba/replay states required by this adapter to commit accepted draft tokens.
For 0.12.0 DSpark, the exporter declares the Markov W2 format in
`dspark_heads_info.json`. A packed W2 must travel in
`dspark_w2_nvfp4.safetensors` alongside the regular head sidecar.
No base-only engine is substituted. ONNX projection weights are baked into plans and are not
duplicated in the bundle. Successfully consumed ONNX intermediates are removed.

Ordinary preparation failure warns, retains a diagnostic log and attempts the
unchanged native request once. The native path consumes prepared ModelOpt FP8,
NVFP4 and mixed FP8/NVFP4 or FP8/W4A16-NVFP4 projections through family-owned
TensorRT Q/DQ.
It preserves source scales and keeps weight-only activations unquantized. Native
projection accumulation follows the same FP32 boundary as the original dense
graph. It does not export ONNX or calibrate a replacement checkpoint.

Use `--precision fp16 --quantization fp8` or `--quantization nvfp4` to require a
matching prepared source. Omitting the quantization option still preserves an
already prepared checkpoint. These are family-owned arguments; the shared CLI
and builder contracts are unchanged.

Native MoE graphs include sigmoid/grouped top-k routing, correction bias,
non-gated routed/shared experts, and optional latent projections. Operator checks
are not full-model qualification. Exported attention scale tensors enable native
E4M3 KV storage: the family owns checkpoint scale loading, TensorRT Q/DQ and cache
allocation; the shared runtime only provides the generic one-byte dtype mapping.
The recurrent graph uses single-token attention for both prompt and generation.
It keeps Q in compute precision and quantizes K/V with source scales, matching
the Edge decode path rather than its quantized block-prefill kernel.

Native paired preparation preserves the complete DFlash or DSpark request when
Edge is absent, unmapped or fails preparation. It builds a target feature-output
plan, the actual prepared draft, and (for DSpark) its Markov correction head.
The native C++ runtime reconstructs draft context from accepted target features,
then verifies proposals sequentially before advancing recurrent state. Rejected
tokens never enter Mamba state. This is a correctness fallback, not a claim of
speculative speedup. DFlash and DDTree are greedy-only; DSpark chain retains
filtered sampling with rejection correction and a request-local seed.

The native DDTree uses top-four depth candidates from Markov-corrected greedy
chain logits, matching the pinned source, with a sixteen-node verification
budget. Proposal siblings do not silently recompute a different Markov policy.
The target's actual untied LM head and companion-owned embeddings are retained.

In the historical 0.11.0 results below, the native DFlash and greedy DDTree
profiles passed full-checkpoint checks, including token-identical output to the same native target. DSpark chain
runs but retains a long-response accuracy failure. A stricter synthetic draft-logit
comparison remains under investigation; it is not silently waived by those
end-to-end results. Prepared/MoE tensor parallelism is not implemented; Edge
admission here remains TP1. Cancellation, publication and runtime errors propagate
without retry.

## Request and runtime contracts

Admission requires compatible Nemotron-H text topology and source quantization,
FP16 compute, TP 1/batch 1/context-parallel 1, and no unmapped build controls.
For 0.11.0, the Edge map admits x86_64 SM80 for plain sources and SM120 for
plain/FP8/NVFP4 sources. For 0.12.0, Linux aarch64 and x86_64 workers on SM80
or newer can try offload when the SDK receipt matches the local GPU, CUDA and
TensorRT versions. Quantized or paired requests also require the full native
kernel set. Admission is not qualification: the preparation step may still
fail, and the unchanged native request may itself fail to build.

Do not gate head80 on optimized CuTe SSD availability. The pinned Mamba plugin
has a scalar-prefill fallback, which the recorded SM80/SM120 profiles exercise.
A previous optimized-kernel-only restriction was incorrect and is removed.

The family renders and tokenizes using its native prompt semantics, then supplies
actual pretokenized IDs to the public Edge API. Full native EOS lists are
preserved with validated derivative tokenizer metadata; original source assets
are unchanged. Separate Jinja templates follow source-file precedence, and the
modern Nemotron ChatML empty-system/thinking suffix is retained. Empty prompts,
submitted counts, total capacity and generation completion are checked.

The runtime is persistent and serialized, with scoped plugin, stream and artifact
ownership. A seeded full-vocabulary request at temperature 1 (`top_k=0`, `top_p=1`)
is rejected on Edge 0.12: the released SDK would turn it greedy.
Supported ordinary sampling controls are forwarded; unmapped controls
are rejected. The current DFlash adapter uses block16/verify16 and remains
greedy-only, as does DSpark DDTree. DSpark chain forwards filtered sampling.
Exact native paired outcomes are listed below. Runtime errors are not converted
to native inference.

## Validated ordinary profiles (0.12.0)

These results use TP1, batch 1 and a 256-token context. A pass requires
the independent short-answer gate (normalized edit distance ≤ 0.15) and
both released 128-token fixtures, greedy and sampled (ROUGE-1 ≥ 0.25 and
ROUGE-L ≥ 0.20 on each). The standalone Edge runner passed both long fixtures for the MC Edge
passes below except unquantized 9B v2: standalone passed sampled generation
but missed the greedy ROUGE-L gate. A compiler failure produces no engine on
that route; it is not an inference or accuracy failure.

| Checkpoint | Tested GPU | MC Edge | MC native TensorRT |
| --- | --- | --- | --- |
| Nemotron-3 Nano 4B BF16 | A30 | Pass | Pass |
| Nemotron-3 Nano 4B FP8 | GB300 | Pass | Pass |
| Nemotron Nano 9B v2 (unquantized) | GB300 | MC passes; standalone greedy misses ROUGE-L (0.1954) | Pass |
| Nemotron Nano 9B v2 FP8 | GB300 | Pass | Pass |
| Nemotron Nano 9B v2 NVFP4 | GB300 | Pass | Pass |
| Nemotron-3 Nano 30B A3B NVFP4 | GB300 | Runs; fails long accuracy gates | TensorRT 11.4 compiler crash |
| Nemotron-3.5 Lightning 30B A3B NVFP4 | GB300 | Pass | TensorRT 11.4 compiler crash |
| RedHatAI Lightning 30B A3B FP8 | GB200 | Pass, with FP16 routed experts | TensorRT 11.4 compiler crash |

For Nano 4B BF16 on A30, an installed MC wheel also built a fresh bundle
with an isolated PyPI Edge 0.12.0 builder. The MC Python environment did not
contain PyTorch. The installed C++ runtime consumed the bundle without the
source checkout, checkpoint or separate Edge SDK; it matched all ten reference
token IDs and passed both released long-answer gates.

On x86-64 SM120, the Nano 30B NVFP4 native build also crashed in TensorRT
11.4 before producing an engine. This additional result is native-only; it
does not establish a common cause with GB300. On GB300, the Super 120B
NVFP4 checkpoint-direct Edge build failed during TensorRT tactic selection.
The full 88-layer ONNX graph exported, but the released ONNX builder also
failed the shader-workspace query and found no implementation for its fused
node. The native fallback segfaulted during serialization; no route produced
an engine and there was no GPU OOM. With the optional 0.12.0 SDK selected,
MC rejects this ordinary capacity-256 profile before retrying a known crash
on ARM64 SM103, CUDA 13.4 and TensorRT 11.4.0.106. This is not a restriction
on all GPUs or a change to the generic native graph.

The family also checks before invoking native serialization, even if no Edge
SDK is present. On Linux ARM64 with CUDA 13.4 and TensorRT 11.4.0.106, it
does not retry the measured batch-one, TP1, FP16-compute, capacity-256
compiler crashes: Super 120B NVFP4, Nano 30B NVFP4 and Lightning 30B NVFP4
on SM103, or the compressed Lightning 30B FP8 checkpoint on SM100. For
Lightning NVFP4 this includes the DFlash, DSpark and DSpark tree target builds.
Working Edge builds for the 30B profiles are still tried first, including
those that missed an accuracy gate. The x86 crash above is a separate result,
not evidence for disabling native build on every SM120 GPU. On an RTX PRO
4500 Blackwell, the same Nano 30B native build also segfaulted before
producing an engine. MC rejects that native retry on x86-64 for this model,
capacity and compiler stack; other named SM120 GPUs are not excluded by it.
If toolkit discovery fails, the family warns and leaves native available.
A missing compiler does not identify the measured failing stack. On an
affected device, set `CUDACXX` to the actual toolkit compiler so the family
can recognize the measured profile before invoking TensorRT.

Nano 30B's short Edge result matches the independent reference. Its long
greedy and sampled results miss the gates through both MC and the standalone
Edge runner. The original BF16 checkpoint also misses a gate in each mode;
these failures do not establish an MC-only regression. The Edge route remains
runnable. None of the ordinary results qualify a companion or embedded MTP mode.

For 9B v2 unquantized and FP8, the two runners did not receive identical greedy prompts.
MC's explicit no-thinking setting closes the thinking block, as the original
checkpoint template does with its `/no_think` instruction. The unmodified
standalone fixture instead opens that block. The two runners received
different rendered prompts. On the unquantized checkpoint, MC Edge passed greedy
and sampled generation (ROUGE-L 0.317 and 0.309), while standalone Edge
missed the original greedy gate and passed sampled generation (0.300).
Using the checkpoint’s `/no_think` instruction, standalone Edge got the same
rendered prompt as MC and passed greedy quality (ROUGE-L 0.296). Its output
text still differed from MC’s. The original standalone failure is unchanged.

## Speculative results (0.12.0)

On GB300, Lightning 30B NVFP4 plus its DFlash companion passed the MC Edge
short-answer and 128-token greedy gates at TP1, batch 1, context 256. The
standalone runner produced the same greedy text with 68 verification rounds
and a mean acceptance length of 1.88. The native build crashed in target
serialization before loading the companion. This result does not qualify
DSpark, tree decoding, or stochastic DFlash.

With the DSpark companion in chain mode on the same GB300 profile, the MC
Edge bundle contained both engines and the packed Markov W2 weights. Its
short answer matched the independent reference token for token. The greedy
128-token response passed (ROUGE-L 0.298); the sampled response ran but
missed the 0.20 threshold (ROUGE-L 0.149). Standalone Edge on the same
engines matched the greedy text and also missed the sampled gate (ROUGE-L
0.059). With the same prompt and MC’s sampling seed (42), standalone Edge
reproduced the MC sampled text exactly. This diagnostic does not replace
either original score. The chain remains
runnable but is not accuracy-qualified. The native chain build exited on a segmentation
fault before inference; its traceback did not identify the faulting frame.
DSpark tree was tested separately with draft top-k 4 and a 16-token
verification profile. Its MC Edge short answer matched the reference. MC and
standalone Edge produced the same greedy long answer (ROUGE-L 0.294), with
36 verification rounds and mean acceptance length 3.56. This profile is
greedy-only. Its independent native build crashed before inference.

## Historical native fallback results (0.11.0 rollout)

These results use family-owned TensorRT graphs and the MC C++ runtime, with
FP16 compute, declared source quantization, TP1/batch1 and capacity 256. They do
not qualify other capacities, platforms or speculative sampling modes. Short
checks use the retained matching independent HF oracle and unchanged NED <=0.15;
long checks retain the existing ROUGE-1 >=0.25 / ROUGE-L >=0.20 gates.

| Checkpoint / execution | Tested SM | Short NED | Long ROUGE-1 / ROUGE-L | Outcome |
| --- | --- | --- | --- | --- |
| 4B BF16, ordinary | 86 | 0.0 | 0.596685 / 0.364641 | Pass |
| 4B FP8, ordinary | 120 | 0.0 | 0.551724 / 0.310345 | Pass |
| 9B v2 BF16, ordinary | 86 | 0.0 | 0.540230 / 0.321839 | Pass |
| 9B v2 FP8, ordinary | 89 | 0.0 | 0.435294 / 0.282353 | Pass |
| 9B v2 NVFP4, ordinary | 120 | 0.0 | 0.520231 / 0.335260 | Pass |
| Nano30B NVFP4, ordinary | 120 | 0.0 | 0.285714 / 0.165414 | Accuracy failure; runnable |
| Lightning NVFP4, ordinary stochastic | 120 | 0.0 | 0.100000 / 0.060000 | Accuracy failure; runnable |
| Lightning + DFlash, greedy | 120 | 0.0 | 0.472727 / 0.278788 | Pass |
| Lightning + DSpark chain, stochastic | 120 | 0.0 | 0.079208 / 0.059406 | Accuracy failure; runnable |
| Lightning + DSpark DDTree, greedy | 120 | 0.0 | 0.472727 / 0.278788 | Pass |

The paired checks reuse real native target-feature, companion and Markov plans
in a private validation container; they do not substitute a target-only runtime
for speculative execution. DFlash and DDTree match the same feature-output target
on both short and 128-token greedy requests. Their long runs record 61 rounds /
915 proposed / 67 accepted and 39 rounds / 585 proposed / 88 accepted, respectively.
DSpark chain records real proposals and exact short greedy parity, but its long
stochastic quality failure remains open. These are not speculative-speedup claims.
No failed quality mode is blacklisted or silently downgraded.

Super120B native checkpoint qualification is blocked by a TensorRT 11.1 compiler
crash on the tested native ARM configuration: the default 8 MiB CPU stack is
exhausted inside the compiler. A 64 MiB-stack diagnostic was stopped after more
than 14 hours without an engine; that intervention is not proof of a compiler
hang, a GPU-memory failure, or an Edge-LLM failure. No native inference result
is claimed for this checkpoint. Native ARM C++ runtime contract tests passing
on SM103 do not qualify it. The tested-SM column records evidence, not native
family admission restrictions.

## Documented Edge 0.11.0 scope and results

The following results exercise the Edge-backed adapter, not the pure TensorRT
native fallback. Native qualification is tracked separately; the presence of a
source format or graph path is not a catalog-wide native support claim.

This scope follows the pinned public [supported-model catalog](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/getting_started/supported-models.md):
eight standalone Nemotron-H checkpoints and the listed Lightning DFlash,
DSpark chain, and greedy DSpark DDTree modes. Every listed case has a passing,
accuracy-failing, or resource-blocked disposition; this is not an all-model pass.
Nemotron Omni and ASR use distinct interfaces and are not this language family.

- **Nine passing profiles:** the table below.
- **Nano30B NVFP4:** build and inference work, but the unchanged output-128
  accuracy gate fails. The adapter remains available.
- **Super120B NVFP4:** not executed in the recorded Edge-backed validation,
  which lacked sufficient single-device capacity. There is no qualified family
  tensor-parallel route. This is not an observed Edge command failure.

The following exact profiles were rebuilt and run locally on native SM120 with
CUDA 13.3 and TensorRT 11.1.0.106: FP16 compute, original declared source
quantization, capacity 256, TP 1/batch 1. These are not family-wide or long-context
qualifications. The source revisions for passing profiles are listed below. No quality threshold was changed.

| Exact source model | Independent NED | MC ROUGE-1 / ROUGE-L | Direct Edge ROUGE-1 / ROUGE-L |
| --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | 0.0 | 0.5263 / 0.2690 | 0.5263 / 0.2690 |
| NVIDIA-Nemotron-3-Nano-4B-FP8 | 0.0 | 0.5600 / 0.2971 | 0.5600 / 0.2971 |
| NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | 0.0 | 0.6061 / 0.3152 | 0.3842 / 0.2599 |
| NVIDIA-Nemotron-Nano-9B-v2 | 0.0 | 0.5153 / 0.3190 | 0.3864 / 0.2614 |
| NVIDIA-Nemotron-Nano-9B-v2-FP8 | 0.0 | 0.5125 / 0.3000 | 0.4270 / 0.2697 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 0.0 | 0.4819 / 0.2530 | 0.4819 / 0.2530 |
| Same Lightning target + DFlash companion, greedy | 0.0 | 0.4719 / 0.2809 | 0.4719 / 0.2809 |
| Same Lightning target + DSpark companion, chain | 0.0 | 0.4835 / 0.2967 | 0.4835 / 0.2967 |
| Same Lightning target + DSpark companion, greedy DDTree | 0.0 | 0.4938 / 0.2963 | 0.4938 / 0.2963 |

Independent checks reuse the owning ten-token, non-thinking greedy testcase and
its NED <=0.15 gate. The 4B/9B FP8 oracles use their corresponding original BF16 models
with byte-identical tokenizer/template. The 9B and Lightning NVFP4 oracles use the existing
strict ModelOpt weight decode into BF16 HF. The Lightning target oracle is reused
for the DFlash and DSpark pairs after immutable source and prompt identity checks. Neither emulates activation/KV
quantization rounding. Official Transformers 5.14.1 provides the builtin
Nemotron-H reference; the older builder environment is kept separate.

Both MC and direct Edge also run the 128-token `llm_basic.json` fixture
with its unchanged ROUGE-1 >=0.25 and ROUGE-L >=0.20 gates. Ordinary profiles
preserve the original stochastic controls. DFlash V1 and DSpark DDTree require greedy `top_k=1`;
its prompt, reference, output budget and thresholds remain unchanged. This is
not a qualification of stochastic DFlash or DSpark DDTree. DFlash recorded 69
speculative verifications with mean acceptance length 1.855; DSpark chain 54 /
2.370 and DDTree 35 / 3.657. Chain uses the original stochastic fixture. The direct control uses
the MC-built engine, not a separately built engine. Sampling text need not be
identical. These local recipes use existing owning helpers; they are not additional
registered CI cases. Other capacities and platforms are not covered by these results.

Nano30B builds and runs, but fails the original output-128 ROUGE-L gate in both
MC (0.1594) and direct Edge (0.1618). A separately sampled greedy HF diagnostic
also fails (0.1231), with verbose text truncated before the CEO answer. Actual
MC, Edge and HF prompt token IDs match. This suggests a response-budget/fixture
interaction, not an established TensorRT or Edge defect; the failure remains open.
Super120B remains unexecuted in these recorded Edge-backed results; their
single-device capacity limitation is not a permanent native-platform exclusion.

The initial Lightning standalone build requested external-weight kinds forbidden
by Edge for mixed W4A16 NVFP4. The family now follows the upstream externalization
policy; the successful rebuild and inference above validate the fix.

## Source revisions

All models are from the public `nvidia` namespace. Immutable revisions, in table
order, are:

- 4B-BF16: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`.
- 4B-FP8: `3fe6dab75665a93884214ad4b1b95cf02717d081`.
- 9B-v2: `6533e8de2c68e4536bf7c411d7a3ce5734111476`.
- 9B-v2-FP8: `8bc5eece2eb5514c4bca7f2ec655b91eb554f4c0`.
- 9B-v2-NVFP4: `8556c9164ddb43fe1f4f4ad730593b3c5e3f7328`.
- Lightning target: `bee7596271d1495f6992ae224aefde4410e816b8`.
- Lightning DFlash companion: `8abcc4db8f34a5080c31eef05d4467afc06c6b9e`.
- Lightning DSpark companion (0.11.0 only): `8a0177116d138011e63103110f136ec0ca09ebbf`.

## Family-owned build options

The existing family CLI reads this owner's cli.json and invokes cli.py.
Edge-specific inputs and selection remain in edge_llm/; the shared parser,
CLI protocol and build API gain no new options or hooks.
All variant validation and builder selection remain in this family.

```sh
trtmc nemotron_h build /path/to/target --precision fp16 \
  --execution-variant dflash --companion draft=/path/to/draft \
  -o model.bundle
```

Options may precede or follow MODEL. `trtmc nemotron_h build /path/to/target --help`
shows these family options using local metadata; remote-ID help does not download
a checkpoint. For Python callers, use this family's request extension:

```python
from tensorrt_model_connect import build
from families.nemotron_h.edge_llm.config import (
    BuildExecutionInputs, NamedCheckpoint, with_execution,
)

# request is an ordinary BuildRequest owned by this family; draft_path is a Path.
build(with_execution(request, BuildExecutionInputs(
    "dflash", (NamedCheckpoint("draft", draft_path),),
)))
```

A failed explicit pair is never replaced by a base-only bundle. The full-model results above were collected during the rollout; publication-head checks must be reported separately.

Ordinary builds without an installed optional Edge SDK select native without a
warning unless the exact native compiler profile is known to crash. Malformed or incomplete installed packages still retain diagnostics and
warn before native fallback. Temporary checkpoint/engine staging uses the bundle
output directory filesystem (choose a scratch-backed output), not system /tmp.

## Declared build command

This family uses the existing cli.json protocol introduced in #1310. The family
owns its declaration, typed inputs and Python handler. The handler adapts those
inputs to the unchanged builder API, preserving native/Edge dispatch and bundle
publication. The legacy flat build command remains available for its existing
ordinary options; new family options use `trtmc nemotron_h build`.
Help is offline and does not need a local checkpoint. No shared parser hook or
family registry entry is added.

### Lightning DSpark profiles

The family command accepts the listed Lightning DSpark draft through the same
local companion mechanism. The ONNX exporter and Edge runtime own the complete
speculative graphs and scheduler; no shared Model Connect selection is involved.

```sh
trtmc nemotron_h build /models/lightning --precision fp16 \
  --execution-variant dspark --companion draft=/models/lightning-dspark \
  --max-sequence-length 256 --output lightning-dspark.bundle
```

Use `--execution-variant dspark_tree` for greedy DDTree. These profiles map the
published block8, anchor-only layout to a nine-position draft profile. Chain
verification uses nine positions and preserves sampling; tree verification uses
16 positions with draft fanout four and rejects non-greedy requests. The bundle
includes the required DSpark heads and metadata. Both capacity-256 paired profiles have their own real build, inference and
quality results above; standalone results are not used as paired qualification.

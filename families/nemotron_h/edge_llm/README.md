# Nemotron-H Edge-LLM execution

The family owns source-policy admission, command mapping, tokenizer/EOS semantics,
engine composition and runtime orchestration. TensorRT Edge-LLM owns model graphs,
lowering and execution. Provision the [optional native SDK](../../../cmake/edge_llm/README.md)
from official GitHub Edge-LLM 0.11.0, revision
`95515c2f87fba8982db5a519f9022277667b3cc9`. No internal source changes or
cross-compilation are used.

A compatible installed 0.11.0 wheel can provide the Python builder tools;
otherwise the family uses the provisioned SDK interpreter. The native C++ SDK
is still required. Interpreter discovery does not install packages or introduce
multi-version dispatch.

## Ordinary and paired builds

Ordinary compatible text builds use the installed experimental Python builder
with components=llm and original checkpoints. Plain source weights use FP16
compute; packed FP8/NVFP4 or mixed policies remain in their declared formats.
The adapter preserves source scales, exclusions and KV policy. For packed
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
Mamba/replay states required to commit accepted draft tokens; the experimental
builder's final-only state outputs do not implement this pair. No base-only
engine is substituted. ONNX projection weights are baked into plans and are not
duplicated in the bundle. Successfully consumed ONNX intermediates are removed.

Ordinary preparation failure warns, retains a diagnostic log and attempts the
unchanged native request once. Native fallback rejects packed checkpoints it
cannot interpret. Paired native fallback is unavailable and fails explicitly.
Cancellation, publication and runtime errors propagate without retry.

## Request and runtime contracts

Admission requires compatible Nemotron-H text topology and source quantization,
FP16 compute, TP 1/batch 1/context-parallel 1, and no unmapped build controls.
The published platform routes are native x86_64 SM80 for plain sources and
SM120 for plain/FP8/NVFP4 sources. These routes are admission rules, not proof for
every compatible checkpoint or capacity.

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
ownership. Supported ordinary sampling controls are forwarded; unmapped controls
are rejected. The current DFlash adapter uses block16/verify16 and remains
greedy-only; stochastic paired execution is not qualified. Runtime errors are not converted to native
inference.

## Documented Edge 0.11.0 scope and results

This scope follows the pinned public [supported-model catalog](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/getting_started/supported-models.md):
eight standalone Nemotron-H checkpoints and the listed Lightning DFlash,
DSpark chain, and greedy DSpark DDTree modes. Every listed case has a passing,
accuracy-failing, or resource-blocked disposition; this is not an all-model pass.
Nemotron Omni and ASR use distinct interfaces and are not this language family.

- **Nine passing profiles:** the table below.
- **Nano30B NVFP4:** build and inference work, but the unchanged output-128
  accuracy gate fails. The adapter remains available.
- **Super120B NVFP4:** not executed because compatible single-device capacity
  was insufficient; there is no qualified family tensor-parallel route.
  This is a resource limitation, not an observed Edge command failure.

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
Super120B is unexecuted because the available supported devices lack capacity
and this adapter has no qualified parallel route for that checkpoint.

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
warning. Malformed or incomplete installed packages still retain diagnostics and
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

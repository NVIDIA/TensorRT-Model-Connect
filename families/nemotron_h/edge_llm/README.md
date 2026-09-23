# Nemotron-H Edge-LLM execution

The family owns source-policy admission, command mapping, tokenizer/EOS semantics,
engine composition and runtime orchestration. TensorRT Edge-LLM owns model graphs,
lowering and execution. Provision the [optional native SDK](../../../cmake/edge_llm/README.md)
from official GitHub Edge-LLM 0.10.1, revision
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`. No internal source changes or
cross-compilation are used.

## Ordinary and paired builds

Ordinary compatible text builds use the installed experimental Python builder
with components=llm and original checkpoints. Plain source weights use FP16
compute; packed FP8/NVFP4 or mixed policies remain in their declared formats.
The adapter preserves source scales, exclusions and KV policy. For packed
sources, only supported external-weight kinds are requested; FP16 bias tensors
are baked into the engine rather than externalized through a missing recipe.

The explicit Lightning DFlash pair instead uses the original ONNX exporter and
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
are rejected. DFlash uses block16/verify16 and is greedy-only because the pinned
runtime forces greedy verification. Runtime errors are not converted to native
inference.

## Recorded model qualification

These are **historical local Model Connect build/inference qualifications**,
not assertions that every publication head or CI executes these profiles.
CUDA 13.3 and TensorRT 11.1.0.106 were used. Ordinary MC profiles use capacity 256;
the DFlash profile uses input/KV1024. All are text-only TP 1/batch 1.
Independent NED must be at most 0.15; the original Edge 128-token fixture gates
remain ROUGE-1 >=0.25 and ROUGE-L >=0.20.

| Exact source model | GPU | Independent gate | MC ROUGE-1 / ROUGE-L |
| --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | SM120 | NED 0.0 | 0.5371 / 0.2514 |
| NVIDIA-Nemotron-3-Nano-4B-FP8 | SM120 | NED 0.0 | 0.5402 / 0.2644 |
| NVIDIA-Nemotron-Nano-9B-v2 | SM80 | Existing HF pytest gate passed; numeric NED not serialized | 0.4318 / 0.2727 |
| NVIDIA-Nemotron-Nano-9B-v2-FP8 | SM120 | NED 0.0 | 0.3750 / 0.2614 |
| NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | SM120 | NED 0.0 | 0.4130 / 0.2391 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | SM120 | NED 0.0 | 0.4643 / 0.2500 |
| Same Lightning target + DFlash companion | SM120 | NED 0.0 | 0.4848 / 0.2545 |

All models are from the public `nvidia` namespace. Immutable revisions, in table
order, are:

- 4B-BF16: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`.
- 4B-FP8: `3fe6dab75665a93884214ad4b1b95cf02717d081`.
- 9B-v2: `6533e8de2c68e4536bf7c411d7a3ce5734111476`.
- 9B-v2-FP8: `8bc5eece2eb5514c4bca7f2ec655b91eb554f4c0`.
- 9B-v2-NVFP4: `8556c9164ddb43fe1f4f4ad730593b3c5e3f7328`.
- Lightning target: `bee7596271d1495f6992ae224aefde4410e816b8`.
- Lightning DFlash companion: `8abcc4db8f34a5080c31eef05d4467afc06c6b9e`.

The independent references use builtin HF BF16 mathematical execution. Packed
sources are decoded with official ModelOpt routines and strict full-state loading;
these oracles do not emulate activation/KV quantization rounding. DFlash reused
a source-byte/reference-function-verified independent reference rather than
regenerating it. Its Edge fixture uses explicit greedy controls, not a claim of
sampling parity.

Passing payloads were retired with approval; compact results and provenance were
retained. Replaying all seven model checks requires rebuilding those payloads.
Fresh publication compilation/unit/source checks must be reported separately.
Only the existing plain 9B case is a registered owning E2E among these profiles;
the other exact models/pair used local recipes with existing family helpers.

## Still outside these qualifications

The separate direct-Edge 9B-NVFP4 capacity 1024 run failed ROUGE-L 0.1957 against 0.20
on a different SM120 GPU. The MC256 pass does not resolve that failure.
The earlier 4B-BF16 capacity 4096 compiler failure, long-context/context-reuse,
TP4, other models/platforms and stochastic equivalence remain outside this proof.
Nano30B, Super120B and Omni are not qualified by the table above.
No quality gate, failed result or hardware limitation is hidden by these passes.

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

A failed explicit pair is never replaced by a base-only bundle. Previously
recorded full-model results above are historical, not fresh refactor-head E2Es.

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

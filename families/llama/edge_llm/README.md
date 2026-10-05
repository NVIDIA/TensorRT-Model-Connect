# Llama Edge-LLM execution

This family owns complete-network offload to Edge-LLM. The existing SDK route
uses the official GitHub 0.10.1 snapshot, revision `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.
CMake provisions the optional native SDK separately. Builds use its experimental
Python builder; inference uses its persistent C++ `LLMInferenceRuntime` API.
There is no implicit installation, download, cross-compilation or cross-family
model dispatch.

## Publication scope

The published route accepts matching dense Llama configuration shapes,
text generation, FP16 compute, original unquantized source weights, TP1 and
batch1. Ordinary execution is mapped to native Linux x86_64 SM80; the explicit
EAGLE3 pair is mapped to SM120. FP8/NVFP4 Llama checkpoints and other platforms
are not admitted by this publication. A matching shape is not qualification for
every checkpoint, generation policy or capacity.

A nonmatching ordinary request retains the original native builder. An Edge
preparation failure retains its diagnostic log, warns, then attempts that native
builder once with the unchanged request. Publication and runtime errors
propagate; they do not silently switch backends. The current-main native body,
including chat-template and multi-EOS handling, is preserved unchanged.

The family maps supported build arguments, preserves the checkpoint assets
required for external weights, and packages the resulting engines. TensorRT and
Edge own the network lowering and speculative decoding. Bundle extraction is
bounded-memory; the runtime, CUDA stream and plugin have scoped lifetimes.
Requests are serialized against the persistent Edge runtime.

Bounded-memory extraction still needs temporary **disk space** for every extracted
engine and checkpoint asset, in addition to the original bundle. FP16 Llama 3.1
8B bundles can require many gigabytes per live runtime instance. Set TMPDIR
before starting the process to a writable filesystem with enough space for the
complete extraction, for example TMPDIR=/path/to/scratch/trtmc-tmp after creating
that directory. /tmp is not necessarily RAM-backed, but its available capacity
must not be assumed. Extracted files are removed when the runtime is destroyed
or extraction fails; abrupt process termination can leave files to clean up.

Raw prompts preserve the checkpoint tokenizer's BOS TemplateProcessing policy.
Chat requests use the pinned Edge chat processing without a second BOS prefix.
Unmapped generation controls, runtime KV overrides, unsupported tokenizer
postprocessors and requests beyond engine capacity are rejected, not ignored.
The native default generation length remains 128 tokens.

## Explicit EAGLE3 companion

Supply `meta-llama/Llama-3.1-8B-Instruct` as the primary checkpoint and the local
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` checkpoint as a named companion:

```python
from tensorrt_model_connect import build
from families.llama.edge_llm.config import BuildExecutionInputs, NamedCheckpoint, with_execution

build(with_execution(request, BuildExecutionInputs(
    "eagle3", (NamedCheckpoint("draft", draft_checkpoint),)
)))
```

`request` is the ordinary Llama `BuildRequest`; both checkpoint paths are local.
Use explicit capacity 2048 for the recorded pair. Geometry, source precision
and the draft context limit are checked before construction. The pinned builder
receives `--spec-type eagle3`; the bundle contains both engines and checkpoints.
Pinned drafting defaults are top-k 10, six steps and verification size 60.
A failed explicit pair never becomes a base-only deployment. The separate
`trtmc llama build-speculative` native EAGLE3 prototype remains available with
its own profile and controls; it is not an automatic replacement for this Edge
profile. See [native speculative decoding](../SPECULATIVE_DECODING.md).

## Recorded model qualification

The following are **historical local build and inference results**, not fresh
inference on the final publication head. All use native Linux x86_64, CUDA 13.3,
TensorRT 11.1.0.106, FP16 compute, TP1 and batch1.

| Exact checkpoint / pair | GPU | Capacity | Independent HF comparison | Edge fixture ROUGE-1 / L |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | SM80 | 4096 | Exact raw/chat/EOS tokens | 0.4828 / 0.3218 |
| Llama-3.2-1B-Instruct | SM80 | 4096 | Exact raw/chat/EOS tokens | 0.5543 / 0.4130 |
| Llama-3.2-3B-Instruct | SM80 | 4096 | Exact raw/chat/EOS tokens | 0.4457 / 0.2500 |
| Llama-3.1-8B-Instruct + EAGLE3 | SM120 | 2048 | Exact base-HF raw/chat/EOS tokens | 0.4809 / 0.3169 |

Base models are in the public `meta-llama` namespace. Immutable revisions:

- 3.1-8B: `0e9e39f249a16976918f6564b8830bc894c89659`.
- 3.2-1B: `9213176726f574b556790deb65791e0c5aa438b6`.
- 3.2-3B: `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- EAGLE3 draft: `ada412b672e293d682423de84a095447bf38a637`.

The recorded public/direct API checks also covered raw/chat/EOS behavior,
capacity and unsupported-control rejection, repeated requests after errors,
and self-contained bundles. Original Edge fixtures used the MC-built engines;
context-reuse fixture ROUGE-1/L was 1.0/1.0 for all four profiles. The independent
comparisons passed the existing Llama criteria without changing thresholds
(default normalized edit distance 0.15 for chat and 0.25 for raw text).

Successful payloads were retired with approval; compact receipts and provenance
remain. Replaying these exact checks requires rebuilding the engines. Historical
local drivers are not newly registered CI cases or published test-framework code.
Fresh CPU, native compilation, existing C++ checks and PR CI must be reported
separately from model inference. No other Llama checkpoint, precision, platform,
long-context behavior or stochastic equivalence is established by this table.

## Family-owned build options

The existing family CLI reads this owner's cli.json and invokes cli.py.
Edge-specific inputs and selection remain in edge_llm/; the shared parser,
CLI protocol and build API gain no new options or hooks.
All variant validation and builder selection remain in this family.

```sh
trtmc llama build /path/to/target --precision fp16 \
  --execution-variant eagle3 --companion draft=/path/to/draft \
  -o model.bundle
```

Omitted CLI precision defaults to FP16 for this paired profile; ordinary builds
retain FP32 and explicit precision values are unchanged.

Options may precede or follow MODEL. `trtmc llama build /path/to/target --help`
shows these family options using local metadata; remote-ID help does not download
a checkpoint. For Python callers, use this family's request extension:

```python
from tensorrt_model_connect import build
from families.llama.edge_llm.config import (
    BuildExecutionInputs, NamedCheckpoint, with_execution,
)

# request is an ordinary BuildRequest owned by this family; draft_path is a Path.
build(with_execution(request, BuildExecutionInputs(
    "eagle3", (NamedCheckpoint("draft", draft_path),),
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
ordinary options; new family options use `trtmc llama build`.
Help is offline and does not need a local checkpoint. No shared parser hook or
family registry entry is added.

## Versioned providers: design example

The opt-in provider route supports exact releases **0.10.0** and **0.11.0**
without changing Model Connect's shared build or runtime selection. It currently
admits ordinary dense FP16 Llama generation on Linux x86_64 SM80, batch1/TP1.
The existing 0.10.1 SDK route, including its EAGLE3 profile, is retained;
`--edge-provider` does not yet implement speculative decoding.

```text
Llama CLI / request
  -> Llama edge_llm builder -> selected installation's Python builder
  -> bundle (release + native-library identity + engine assets)

Llama runtime
  -> exact-version provider DSO (small C-compatible function table)
  -> persistent isolated worker -> selected installation's C++ LLMRuntime
```

The worker calls upstream Python bindings to the C++ runtime; it is not a
second inference implementation. One worker/runtime is retained per live
Model Connect task, with serialized requests. Isolation prevents different
Edge/TensorRT plugin versions from sharing one process's native loader state.
It adds a process and IPC boundary, so latency and concurrent-version GPU
memory usage still need performance qualification.

Ownership and compatibility:

- `edge_llm/provider.py` owns the release-specific builder calls, request
  mapping, capacity checks, and runtime behavior. The C++ family adapter selects
  the provider from bundle metadata; the core has no Edge routing.
- `cmake/edge_llm/provider/` contains only the opaque-handle ABI, process
  transport, and installation identity mechanics. No model or family dispatch
  is placed there. Provider DSOs expose `trtmc_edge_provider_v1`; no Edge/STL
  types, exceptions, or allocated objects cross that boundary.
- The descriptor is trusted, machine-local deployment configuration, **not**
  executable content read from a model bundle. Python paths and native library
  paths are explicit. Nothing is downloaded or installed during build/run.
- A bundle records the exact Edge release, runtime/plugin SHA-256, TensorRT
  version, CUDA runtime version, architecture and SM. Runtime mismatches fail
  closed, rather than silently rebuilding or choosing another release.
  Source and wheel origins use the same contract, but their different native
  binaries are not assumed to make existing engines interchangeable.
- Generic packaging copies provider DSOs and workers, never machine-local
  descriptors. Existing native fallback on Edge preparation failure still warns;
  a successful native fallback is not proof that this provider worked.

### Wheel installation

The optional `edgellm` dependency selects `tensorrt-edgellm==0.11.0`.
A source checkout can install it with `python -m pip install '.[edgellm]'`;
a Model Connect wheel exposes the same extra. The plain installation does not
require Edge, and there is no runtime pip invocation.

Create an absolute-path descriptor, for example `/opt/providers/0.11.0.json`:

```json
{
  "schema_version": 1,
  "version": "0.11.0",
  "python": "/opt/edge-0.11/bin/python",
  "library_paths": ["/opt/tensorrt/lib", "/opt/cuda/lib64"]
}
```

The selected interpreter must have the Edge package and its dependencies.
The public wheel's `tensorrt_edgellm.runtime.load()` selects its native payload.
`library_paths` is optional when the native dependencies already resolve.

### Source installation

Use the official release source and native build prerequisites, including its
Python bindings and plugin. For 0.10.0, the upstream CuTe preparation requires
`nvidia-cutlass-dsl[cu13]==4.6.1`; generate native CuTe artifacts before enabling
them in CMake. Disabling CuTe did not build the unmodified 0.10.0 SDK in the
recorded environment. Do not cross-compile.

A source build supplies the same descriptor plus explicit module locations:

```json
{
  "schema_version": 1,
  "version": "0.10.0",
  "python": "/opt/edge-0.10/bin/python",
  "python_paths": ["/opt/TensorRT-Edge-LLM-0.10.0"],
  "library_paths": ["/opt/tensorrt/lib", "/opt/cuda/lib64"],
  "native_module": "/opt/edge-build/pybind/_edgellm_runtime.cpython-312-x86_64-linux-gnu.so",
  "plugin": "/opt/edge-build/libNvInfer_edgellm_plugin.so"
}
```

Use the actual extension filename produced by that interpreter's native build.
0.10.0 requires the source module; 0.11.0 accepts either that explicit-module
form or the wheel selector. Both validate the loaded package's exact version
and hash the actual native libraries. The interface does not depend on how
those files were installed.

The family maps 0.10.0 to upstream ONNX export and `LLMBuilder` with embedded
weights; 0.11.0 maps to upstream's direct builder with external checkpoint
weights. Embedded engines omit redundant original checkpoint weight shards
from the bundle. These are release-specific choices, not global Model Connect
build modes.

### Build and deploy

```sh
trtmc llama build /models/Llama-3.2-1B-Instruct --precision fp16 \
  --max-sequence-length 256 --edge-provider /opt/providers/0.11.0.json \
  -o /scratch/llama.bundle

mkdir -p /opt/trtmc/lib/edge_llm/providers
cp /opt/providers/0.11.0.json /opt/trtmc/lib/edge_llm/providers/0.11.0.json
trtmc run /scratch/llama.bundle --runtime-root /opt/trtmc/lib \
  --prompt "What is the capital of France? Answer in one word." \
  --use-chat-template true --max-new-tokens 32 --temperature 0 --top-k 1 --top-p 1
```

For a CMake build, the runtime root is the build directory. For a packaged
installation, it is `tensorrt_model_connect/bin`. The descriptor must sit next
to the selected family/provider DSOs under `edge_llm/providers/<version>.json`;
it is not copied from the builder's filesystem into the bundle.

Both releases reject unsupported controls and clipped requests without
weakening the family contract. 0.11.0 can check exact prompt capacity before
execution; 0.10.0 exposes prompt counts only afterwards, so that release may
execute before rejecting an over-capacity request. It never returns that
clipped result as success. Worker startup/transport failures propagate,
with bounded IPC and teardown; they do not select a different backend.

### Evidence boundary

This is a versioning design example, not catalog-wide qualification.
New source/wheel build, inference, independent-reference, packaging and CI
results are reported separately in its review. The earlier 0.10.1 table above
does not qualify either new release. The 0.10.0 source and 0.11.0 wheel Llama-3.2-1B comparisons recorded
exact Edge/HF tokens on capital, arithmetic and raw-continuation prompts, but
both implementations answered `14` for `9 + 7`: the existing semantic gate
correctly remained failed. Matching an incorrect reference is not semantic
qualification.

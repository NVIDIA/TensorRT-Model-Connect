# Llama Edge-LLM execution

When the installed Model Connect wheel includes a native runner and Edge SDK,
the family uses that SDK. A different SDK on `CMAKE_PREFIX_PATH` cannot
override the wheel's native runner. To select another release, install a
Model Connect wheel built for it, or use a matching standalone Model Connect
builder and runner in an environment without the wheel. For standalone
installations, `CMAKE_PREFIX_PATH` selects the SDK as usual.

This family owns complete-network offload. The C++ adapter links one native
Edge SDK at build time, and reads bundles for that release. Model builds do
not download dependencies or select a second runtime. Native TensorRT remains
the family-owned fallback.

## Installed Python builder discovery

- **Explicit 0.11.0 build** (official revision `95515c2f87fba8982db5a519f9022277667b3cc9`):
  the experimental builder can run in the current Python installation if Edge
  and TensorRT match the SDK; otherwise use the SDK's bundled interpreter.
- **Default Edge-enabled build: 0.12.0** (official revision `513dcd18291e587c80928911a480761352f3c2d8`):
  configure MC with `TRTMC_ENABLE_EDGELLM=ON` and
  `TRTMC_EDGELLM_RELEASE=0.12.0`. Pass
  `--edge-builder-python /absolute/path/to/python` to `trtmc llama build`.
  The interpreter needs the release's builder (PyPI or from source),
  Transformers 5.17.0 and matching TensorRT bindings. If omitted, the
  family checks the current interpreter. For a PyTorch
  `.bin` draft, the Edge builder and the native Python graph builder also need
  Torch. Edge externalizes the draft weights, so its interpreter must expose
  checkpoint byte offsets as well. For the documented EAGLE3 draft, the
  official reader returned valid offsets with CPU Torch 2.15.0 but not
  2.10.0. MC checks this capability rather than imposing a Torch version
  on all Llama builds. Native only needs to read the tensors; safetensors
  inputs do not need Torch for this purpose.

Both paths require the matching native CMake SDK, not just Python modules.
The family finds the SDK receipt in the installed MC package or CMake prefix;
the 0.12 receipt contains no machine-specific path to a build interpreter.
A provisioned source SDK records its checked revision. An external native SDK
records its package version and a null source revision; the family carries
that distinction into the bundle.
See [native provisioning](../../../cmake/edge_llm/README.md). Historical
0.11 validation below is not evidence of 0.12 qualification.

## Adapter scope

The ordinary route accepts matching dense Llama configuration shapes,
text generation, FP16 compute, original unquantized source weights, TP1 and
batch1. Ordinary execution is mapped to native Linux x86_64 SM80; the explicit
EAGLE3 pair is mapped to SM120. The packed-format
route admits ModelOpt FP8/NVFP4 8B sources on SM120 and, with 0.12, the
static compressed-tensors 1B FP8 source. It requests upstream `--dense auto`
to preserve the source format.
Other platforms are not admitted for Edge offload. A matching shape is not qualification for
every checkpoint, generation policy or capacity.

A nonmatching ordinary request retains the original native builder. An Edge
preparation failure retains its diagnostic log, warns, then attempts that native
builder once with the unchanged request. Publication and runtime errors
propagate; they do not silently switch backends. Native chat-template and
multi-EOS handling remain family-owned.

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
A failed explicit pair never becomes a base-only deployment. It now reuses the
existing `trtmc llama build-speculative` native EAGLE3 implementation, preserving
the target, companion, context capacity, and FP16 compute. It constructs graphs
with TensorRT Python APIs and executes them in the family C++ runtime, without
ONNX or an Edge SDK. The native fallback uses six draft steps and its existing
verification schedule; it does not duplicate Edge's top-k10/tree60 scheduler.
The native runtime preserves the family sampler for greedy and sampled requests,
including the seed. Sampled execution verifies real proposals one target row at
a time, preserving ordinary-target RNG order and committed-prefix feedback.
Unsupported generation controls still fail explicitly. This correctness-first
schedule is not a speculative speedup claim. The standalone command remains
available with its own controls. See
[native speculative decoding](../SPECULATIVE_DECODING.md).

Native fallback evidence: the exact base/draft revisions below built and ran on
A30/SM80 with TensorRT 11.1, FP16 compute and capacity 2048, with Edge and BYOK
disabled. Capital, arithmetic and raw-continuation cases all matched fresh
official-source FP32 token IDs exactly (NED 0.0, unchanged gates 0.15/0.15/0.25).
One persistent native C++ runtime served all three requests. A subsequent native
sampling check preserved all three original-FP32 quality passes and matched
ordinary execution of the same compiled target in 12 sampled public requests.
The existing C++ validation also passed 16 sampled policy/width/prompt
combinations with repeated seeds, accepted real draft tokens, and retained its
EOS/reset/capacity checks. These are retained TensorRT 11.1 results, not fresh
qualification of the repository-pinned SDK or other platforms.

## Prepared source native fallback

When the Edge route is unavailable, the same prepared ModelOpt FP8/NVFP4
checkpoint can use family-owned TensorRT Python graphs and the existing native
C++ runtime. No ONNX conversion, Edge SDK, ModelOpt execution, Torch source
reader, or GPU-specific model graph is required. The declared safetensors,
NumPy and ml_dtypes dependencies preserve the exact source packed weights and
calibrated activation, block and global scales.

Projection Q/DQ preserves source quantized arithmetic rather than silently
turning NVFP4 into ordinary FP16 inference. New ordinary prepared-model builds with calibrated FP8 K/V use linear
E4M3 past/present bindings and one-byte native cache storage. TensorRT uses
the trained scale of each layer's K and V to dequantize for attention. Older
compute-type cache bundles retain their own storage layout. Explicit matching
FP8/NVFP4 requests and automatic requests consume the prepared checkpoint;
neither recalibrates those weights.

```bash
python -m tensorrt_model_connect llama build /path/to/prepared-checkpoint \
  --precision fp16 --max-sequence-length 256 -o llama-prepared.bundle
trtmc run llama-prepared.bundle --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 1 --top-k 1
```

Native validation uses the exact packed revisions listed below, capacity 256,
TP1 and the unchanged capital/arithmetic/raw cases. The oracle is the original
FP32 model with each prepared checkpoint's exact tokenizer inputs, not a reused
oracle with a different chat template.

- FP8 on RTX4000 Ada / SM89: build and native C++ inference pass all three
  cases with exact original-FP32 generated IDs and NED 0.
- NVFP4 on the 20GB RTX4000 Ada / SM89: build succeeds, but native context
  creation runs out of memory requesting an additional 13,952,389,120 bytes.
  This is an executed capacity failure, not an unsupported-SM89 rejection.
  TensorRT warns that FP4 compute is unaccelerated on this device.
- NVFP4 on the 32GB RTX PRO4500 Blackwell / SM120: a fresh native build and
  C++ inference pass all three cases. Both chat cases have NED 0; raw
  continuation has NED 0.092025 against the unchanged 0.25 limit. This build
  uses the dependency-free source reader. Across both prepared checkpoints,
  all 803 FP8 and 1027 NVFP4 source tensors match the independent safetensors
  Torch reader byte-for-byte; this is a reader check, not extra model coverage.
- TensorRT 11.1's native Q/DQ graph rejects SM80 when it contains FP8 operations.
  No model-SKU denylist or silent precision downgrade is added to bypass this
  backend limitation.

These exact local results do not qualify other contexts, source revisions,
sampling policies, CPU architectures or operating systems.

## Edge 0.12.0 validation

With TensorRT 11.4, batch one and FP16 compute, the
ordinary configurations below were built and run through MC. Each route was
scored on the existing capital, arithmetic and raw-continuation cases against
an independent FP32 reference with that checkpoint's prompt tokens. A quality
miss does not prevent building or running the model.

| Source checkpoint | Edge route | Native TensorRT route |
| --- | --- | --- |
| Meta Llama 3.2 1B Instruct | 2/3; arithmetic miss | 2/3; arithmetic miss |
| RedHatAI Llama 3.2 1B Instruct FP8 | 2/3; arithmetic miss | 2/3; arithmetic miss |
| Meta Llama 3.2 3B Instruct | 3/3, NED 0 with both wheel and source tools | 3/3, NED 0 after warned fallback |
| Meta Llama 3.1 8B Instruct | 3/3, NED 0 | 3/3, NED 0 |
| NVIDIA Llama 3.1 8B Instruct FP8 | 3/3, NED 0 | 3/3, NED 0 |
| NVIDIA Llama 3.1 8B Instruct NVFP4 | 3/3, NED 0 | 3/3, NED 0 |

The documented Llama 3.1 8B Instruct + EAGLE3 pair also ran on SM120 at
capacity 256. Both the Edge C++ runtime and the warned native speculative
fallback matched the base-model FP32 reference on all three cases (NED 0).
The Edge bundle contained both speculative engines. In the existing native
verifier, a 96-token input produced 32 identical greedy tokens for tree,
chain and ordinary target decoding. Reset, capacity and all 16 sampling
checks passed, with 228 accepted draft tokens across the sampled checks.
These checks establish correctness for this pair, not a speedup.

The two NVIDIA prepared configurations preserve the trained weight format
and FP8 KV cache. On SM120, the Edge engine configuration declares FP8 KV;
the native TensorRT plans expose 128 linear E4M3 past/present bindings. No
additional calibration was performed. The RedHatAI checkpoint instead uses
static per-tensor FP8 weights and input scales with a compute-precision
cache. The ordinary original-weight Edge runs used SM80, not SM120. These
results do not qualify other checkpoints, hardware or context lengths.

## Fresh Edge 0.11.0 qualification

These actual Model Connect builds use the official 0.11.0 Python wheel with the
separately provisioned native SDK. The native adapters were compiled against
Model Connect base `95f0da259ffddfd95ff7a67dc944583c2ae7e043` plus the family
changes and SDK prerequisite. All five profiles also pass the same three cases
through real installed Model Connect wheels and the existing persistent native
worker. Module and runtime-library paths were verified against those
installations. These are local results, not final PR-head CI or review approval.

All profiles below use CUDA 13.3, TensorRT 11.1.0.106 and FP16
compute, TP1/batch1 and greedy generation of up to 32 tokens. Capital, arithmetic
and raw-continuation cases use the unchanged family semantic checks and NED
limits (0.15 chat, 0.25 raw), against an independent FP32 Hugging Face reference.

| Exact checkpoint / pair | GPU | Capacity | Model Connect / HF | Direct Edge / HF |
| --- | --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct | A100 SM80 | 256 | 3/3 pass, exact generated IDs, NED 0 | 3/3 pass, NED 0 |
| meta-llama/Llama-3.2-3B-Instruct | A30 SM80 | 256 | 3/3 pass, exact generated IDs, NED 0 | 3/3 pass, NED 0 |
| Llama-3.1-8B-Instruct + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | Blackwell SM120 | 2048 | 3/3 pass, exact base-HF IDs, NED 0 | 3/3 pass, NED 0 |

Immutable revisions:

- 3.1-8B: `0e9e39f249a16976918f6564b8830bc894c89659`.
- 3.2-3B: `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- EAGLE3 draft: `ada412b672e293d682423de84a095447bf38a637`.

The paired profile reuses the exact primary-checkpoint FP32 oracle with case and
revision checks. Direct Edge receives the same raw input tokens, including BOS;
its CLI exposes output text, so no direct-CLI output-token-ID parity is claimed.
This token alignment matters: a plain raw string without the tokenizer's BOS
postprocessing is a different input, not a runtime discrepancy.

Remaining failures and exclusions:

- Llama-3.2-1B-Instruct, revision
  `9213176726f574b556790deb65791e0c5aa438b6`, SM80/FP16/capacity256:
  build/inference pass, but Model Connect, direct Edge and independent FP32 HF
  all answer 14 to 9 + 7. The unchanged semantic gate requires 16. This historical exact-input result
  used a 5 October 2026 template date; it was not rerun during installed closure.
  It is not a fully quality-qualified profile; parity does not resolve the failure.
- meta-llama/Meta-Llama-3-8B-Instruct, revision
  `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`: checkpoint access returned HTTP403.
  Build, inference and quality were not executed. This is an access-blocked
  skipped experiment, not an Edge, TensorRT or Model Connect execution defect.

The official packed profiles have now been validated separately, with FP16
compute, native Blackwell SM120, capacity256, TP1/batch1 and greedy32:

| Checkpoint | Revision | MC / direct cases | Chat NED | Raw NED / gate |
| --- | --- | --- | --- | --- |
| nvidia/Llama-3.1-8B-Instruct-FP8 | `42d9515ebd69eea3a87351d079c671c3c5ff0a31` | 3/3 pass each | 0 | 0 / 0.25 |
| nvidia/Llama-3.1-8B-Instruct-NVFP4 | `bdb54e24298451af785c0ac63c1b485e9b7400a2` | 3/3 pass each | 0 | 0.092025 / 0.25 |

Both packed profiles retain FP8 KV cache, and direct Edge text exactly matches
Model Connect. Their independent reference uses original FP32 model weights
with each packed checkpoint's exact tokenizer inputs. It is not an evaluation
of packed weights through Hugging Face. The NVIDIA chat templates omit Meta's
default system/date message: the initial cached-reference check rejected that
input mismatch, and scoring was corrected without rebuilding engines, replaying
native inference or changing gates. Original failed scoring receipts remain.
The packed adapter maps source formats to upstream --dense auto; no new
quantization recipe or shared-core selection is introduced. Installed-wheel replay preserves the same results and generated IDs. A clean
package initially exposed a missing Edge runtime plugin; the SDK prerequisite
now includes it in the wheel install component, and the corrected installed
package passes all three Blackwell profiles without manual library copying.

Historical 0.10.1 results are not evidence for 0.11.0. No other precision,
platform, context length, stochastic behavior or catalog-wide coverage is
established by these five successful profiles.

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

A failed explicit pair is never replaced by a base-only bundle. Fresh results above cover only the exact listed local profiles; PR CI and
new-head installed-package validation remain separate.

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


The current 8B profile is Llama3.1 with its 131072-token source context, not
Llama3.0 merely because their layer geometry matches. Access to the attempted
`meta-llama/Meta-Llama-3-8B-Instruct` checkpoint was denied, so no Edge profile
for it is added. The original 1B route remains executable despite its recorded
quality failure; quality scores are not dispatch criteria.


### Native EAGLE3 sampling

The family-native EAGLE3 runtime maps temperature, top-k and top-p to the
existing Llama sampler; native seeds are preserved as well. Real draft
proposals still execute. Sampled verification uses ordinary one-row target
execution and consumes each target distribution exactly once, including the
replacement token after rejection. Only committed target features condition
draft feedback. The existing greedy device policy remains unchanged. This
conservative path makes no speculative speedup claim or promise of matching
another framework's random-number sequence. Repetition penalties remain
unsupported, as in the Edge adapter.

For Llama-3.1-8B-Instruct revision
0e9e39f249a16976918f6564b8830bc894c89659 with EAGLE3 draft revision
ada412b672e293d682423de84a095447bf38a637, a fresh FP16/context2048,
TP1/batch1 native build on x86 A30 / TensorRT 11.1 passed all three unchanged
original-FP32 quality cases (NED 0). Twelve public sampled comparisons matched
ordinary same-engine target output exactly, with non-greedy continuations.
Existing runtime validation additionally covers two draft widths, multiple
seeds, short/chunked prompts, repeated requests, sampled EOS, capacity limits
and genuine draft acceptance. These results do not qualify other checkpoints
or platforms.

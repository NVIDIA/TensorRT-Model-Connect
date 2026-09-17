# Llama Edge-LLM execution

This family owns complete-network offload to the official GitHub Edge-LLM
0.10.1 snapshot, revision `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.
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

Raw prompts preserve the checkpoint tokenizer's BOS TemplateProcessing policy.
Chat requests use the pinned Edge chat processing without a second BOS prefix.
Unmapped generation controls, runtime KV overrides, unsupported tokenizer
postprocessors and requests beyond engine capacity are rejected, not ignored.
The native default generation length remains 128 tokens.

## Explicit EAGLE3 companion

Supply `meta-llama/Llama-3.1-8B-Instruct` as the primary checkpoint and the local
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` checkpoint as a named companion:

```python
from tensorrt_model_connect import BuildExecutionInputs, NamedCheckpoint, build

build(request, execution=BuildExecutionInputs(
    "eagle3", (NamedCheckpoint("draft", draft_checkpoint),)
))
```

`request` is the ordinary Llama `BuildRequest`; both checkpoint paths are local.
Use explicit capacity 2048 for the recorded pair. Geometry, source precision
and the draft context limit are checked before construction. The pinned builder
receives `--spec-type eagle3`; the bundle contains both engines and checkpoints.
Pinned drafting defaults are top-k 10, six steps and verification size 60.
A failed explicit pair never becomes a base-only deployment: the warned native
fallback reports that native EAGLE3 execution is unsupported.

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

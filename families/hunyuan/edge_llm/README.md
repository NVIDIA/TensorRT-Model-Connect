# Hunyuan Edge 0.11 integration

This family owns dense Hunyuan metadata, graph/weight semantics, tokenizer,
runtime and validation. It imports no other model family.

The Edge route supports original dense FP16, TP1/batch1 on the exercised Linux
x86-64 SM80 platform. Other requests use the native TensorRT graph. The native
graph applies Q/K RMSNorm **after** RoPE and uses Hunyuan's fixed
DynamicNTKAlpha base; it does not substitute ordinary Llama semantics.
Build failure warns and retries native with the unchanged request. Cancellation
and publication errors do not retry. Inference errors propagate.

Enable the pinned 0.11 CMake SDK and set its install prefix in CMAKE_PREFIX_PATH.
An ABI-compatible installed tensorrt-edgellm==0.11.0 Python package is used
when available, otherwise the provisioned SDK's private Python is used.
The wheel alone does not replace the required C++ SDK development payload.
No installation or download occurs during adapter dependency discovery.

Build with the existing command:

    trtmc build /path/to/Hy-MT2-7B --precision fp16 --max-sequence-length 1024 -o model.bundle
    trtmc run model.bundle --use-chat-template true --temperature 0 --prompt '...'

Edge runtime controls that have no mapped upstream equivalent are rejected.
The current raw tokenizer contract inserts no BOS/EOS; the chat template owns
the start-of-text and user terminator. Capacity is checked before generation.

Hy-MT2-1.8B uses a family-owned pretokenizer for its exact three-Split tokenizer
contract. It preserves the original vocabulary, merges, added tokens and chat
template, and passes token IDs through Edge's public pretokenized-input API.
The fixed patterns use linear Unicode-codepoint scans; unknown tokenizer
contracts fail closed. Vocabulary-derived filtering preserves the checkpoint's
unknown-base-symbol policy. Edge still performs BPE merges, generation, sampling, EOS and output
decoding. Single-Split sibling checkpoints retain the original Edge path.
This avoids a reproduced Edge 0.11 byte-offset/codepoint-indexing failure
without modifying the checkpoint or upstream source. Direct upstream CLI
tokenization remains affected; adapter success is not an upstream fix.
The SDK installation includes the Edge plugin beside the family runtime DSO.

## Validation scope

The owning manifests compare deterministic translations to an independent
Hugging Face FP32/eager model with the existing 0.15 normalized-edit-distance
gate and explicit semantic answers. The project remains pinned to Transformers
5.2. Hunyuan's independent HF oracle
requires >=5.6,<6. The family declares the official Transformers 5.14.1 source
in tests/reference-source.json using the existing reference-source convention.
Provide that checkout through TRTMC_REFERENCE_SOURCE_DIR: an isolated child
uses its src/ directory with the existing test dependencies, while Model
Connect keeps the installed Transformers 5.2. No shared dependency is upgraded.
Alternatively set TRTMC_HUNYUAN_REFERENCE_PYTHON to an explicitly provisioned
reference interpreter; the tested oracle uses Transformers 5.14.1 and Torch 2.13.
Neither path installs or downloads packages during testing. The evidence records
the actual reference version and import path. Missing source/dependencies fail
rather than skip. CI must provision the declared reference checkout.
The isolated child defaults CPU thread pools to one thread, honors explicit
caller settings, and leaves the parent process environment unchanged.

Local-source Model Connect validation covers HY-MT1.5-7B, Hy-MT2-1.8B and
Hy-MT2-7B with the pinned manifests: original weights to FP16, SM80, TP1/batch1,
input/KV1024, greedy weather and idiom translations. Both texts and generated
token IDs match the independent FP32 oracle. For the 7B checkpoints, initial
reference-only host thread failures were recovered against the exact saved
runtime outputs with the unchanged oracle and assertions; the original failed
pytest records remain failures. These results do not qualify other precision,
full context, sampling, hardware or full-checkpoint native fallback. They also
do not establish installed-package or remote CI qualification.

Tokenizer validation separately covers original prompts, all added tokens,
Unicode and BPE boundary cases, and near-limit inputs. It does not replace
the owning model quality checks. Upstream smoke passes alone are never Model
Connect quality qualification.

A separate full-checkpoint Hy-MT2-1.8B native FP16 fallback build currently
hits a TensorRT attention-compiler internal assertion during decode compilation
(expected three live inputs, found four). This is distinct from the upstream
Edge tokenizer failure; neither full native FP16 nor runtime quality is qualified
by the successful tiny FP32 math check. The compiler cause is not yet reduced.

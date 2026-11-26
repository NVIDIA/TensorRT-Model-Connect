---
title: Nemotron-H with Edge-LLM
---

Nemotron-H owns its optional Edge-LLM builder, tokenizer contract, and
C++ runtime adapter. The ordinary family command selects Edge when a matching
native SDK and supported configuration are available.

```sh
trtmc nemotron_h build /models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --precision fp16 --max-sequence-length 256 --output nemotron.bundle
trtmc nemotron_h build /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --precision fp16 --max-sequence-length 256 \
  --execution-variant dspark --companion draft=/models/lightning-dspark \
  --output lightning-dspark.bundle
```

To opt in to Edge-LLM 0.12.0, configure the native SDK with
`-DTRTMC_EDGELLM_RELEASE=0.12.0`. If the matching builder is in a separate
wheel or source environment, add `--edge-builder-python /absolute/path/to/python`
to the family build command. This selects the Python tools, not a different
C++ runtime. Model Connect keeps its existing Transformers pin. An installed MC wheel
used this setup to build Nano 4B BF16 on A30. Its packaged C++ runtime ran the
bundle without the source checkpoint or a separate Edge SDK.

For the RedHatAI Lightning 30B FP8 checkpoint, Edge 0.12.0 uses its
checkpoint-direct builder and lowers the routed experts to FP16. This is a
mixed-precision engine, not an all-FP8 engine. The native fallback uses the
checkpoint’s original scales to construct its TensorRT Q/DQ graph.

The listed Lightning companion modes are `dflash`, `dspark`, and
`dspark_tree`. DFlash and DSpark tree require greedy generation; DSpark
chain preserves supported sampling controls. Edge-backed paired execution builds
both engines using the ONNX flow and never substitutes a base-only bundle.

For **0.12.0 ordinary decoding**, both MC routes passed the short, greedy and
sampled checks for Nano 4B BF16 (A30), Nano 4B FP8 and Nano 9B v2 FP8/NVFP4
(GB300). Unquantized 9B v2 passed both MC routes on GB300; standalone Edge
passed sampled generation but missed the original greedy gate (ROUGE-L
0.1954 versus 0.20). Its checkpoint template enables thinking for that prompt,
unlike MC’s explicit no-thinking request. Lightning 30B NVFP4 (GB300) and RedHatAI Lightning 30B FP8 (GB200)
passed on Edge; native TensorRT compilation failed. Nano 30B NVFP4 (GB300)
ran on Edge but missed the long-answer accuracy gates. Its native build
failed on GB300 and x86-64 SM120. Super 120B NVFP4 on GB300 failed in
TensorRT tactic selection with both checkpoint-direct and ONNX builds; the
native fallback crashed in serialization. No engine was produced.

At batch one, TP1 and capacity 256 on Linux ARM64 with CUDA 13.4 and
TensorRT 11.4.0.106, MC rejects those native retries even without an Edge SDK:
Nano 30B, Lightning 30B and Super 120B NVFP4 on SM103, and RedHatAI
Lightning 30B FP8 on SM100. The NVFP4 Lightning guard also covers the three
companion modes below. Working Edge builds remain available, including ones
that missed accuracy gates. The same Nano 30B native build also crashed on an RTX PRO 4500 Blackwell.
MC rejects that retry on x86-64 with the same compiler stack and capacity;
this is not a blanket SM120 exclusion.

For **0.12.0 speculative decoding**, the Lightning NVFP4 target and DFlash
companion passed the Edge short-answer and greedy long-answer gates on GB300,
with confirmed draft acceptance. Native target compilation failed before
loading the companion. On the same GB300, DSpark chain through MC Edge
matched the independent short answer and passed greedy generation. Sampled
generation missed the 0.20 ROUGE-L gate through both MC (0.149) and
standalone Edge (0.059). The native chain build crashed before inference. Separately, DSpark tree passed the MC Edge short-answer check and matched
standalone Edge on the greedy long-answer check. Its native build crashed;
sampling is not part of the tree profile.

**Historical 0.11.0 results, not 0.12.0 qualification:** Nine exact Edge-backed profiles passed: 4B BF16/FP8, 9B BF16/FP8/NVFP4, Lightning
NVFP4, and its three companion modes. Nano30B NVFP4 builds and runs but has a
recorded output-budget accuracy failure; it remains runnable. Super120B NVFP4
was not executed because compatible single-device capacity was insufficient.
These are bounded text-only results, not qualification of all contexts or
platforms. Nemotron Omni and ASR are separate interfaces.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_h/edge_llm/README.md)
for exact profiles, unchanged quality gates, source revisions, and limitations.
A compatible pip installation may provide Python tools; Edge-backed C++ inference
still requires the matching provisioned SDK. Preparation failures warn and attempt
the unchanged native request once; runtime failures propagate.

The measured native compiler failures are matched by layer order, attention
and state dimensions, and routed and shared expert widths. Other graph shapes
remain eligible for native compilation; eligibility is not a validation result.

## Native fallback

Without compatible Edge offload, the family uses TensorRT Python graph construction
and the Model Connect C++ runtime, without ONNX. Prepared FP8, NVFP4 and mixed
source formats, scales and FP8 KV storage are preserved; no replacement checkpoint
is calibrated. A paired request still builds the target and actual companion,
including the DSpark Markov head, rather than silently using ordinary decoding.

The **historical 0.11.0 rollout** native FP16-compute, TP1/batch1, capacity-256 results are separate from
the Edge results above: seven profiles pass, while Nano30B, ordinary Lightning
and Lightning DSpark chain retain accuracy failures and remain runnable.
Super120B has a demonstrated TensorRT compiler stack-exhaustion failure on the
tested ARM configuration and no native inference qualification. The larger-stack
diagnostic was stopped after more than 14 hours; this is not an observed Edge
failure or proof of a compiler hang. See the family recipe for individual scores,
speculative target-parity checks and the unresolved synthetic draft diagnostic.
These results do not qualify every platform or capacity, and tested GPU types
are not native family admission restrictions.

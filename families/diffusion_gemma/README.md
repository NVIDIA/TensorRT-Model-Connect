# DiffusionGemma

This family owns `google/diffusiongemma-26B-A4B-it` text continuation, including
its causal encoder, bidirectional block denoiser, shared checkpoint weights,
self-conditioning, and entropy-bound sampler. The initial checkpoint revision
is `f7f5b7f5fa82ffc52addd066915886d497f5517b`.

The build uses FP32 compute with the original checkpoint weights, batch size 1,
tensor parallel size 1, and context parallel size 1. The default sequence
capacity is 512 tokens, shared by the prompt and complete output canvases.
Every canvas contains 256 tokens. As in the official
implementation, the generation budget rounds up to a complete canvas and output
ends at the first configured EOS token.

The public builder accepts FP32 only; BF16 builds are not supported.

## Build

Prepare the native development environment using the repository's
[source build instructions](../../website/docs/getting-started/source-build.md).
From the checkout root:

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  google/diffusiongemma-26B-A4B-it \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --precision fp32 --max-sequence-length 512 \
  --output /tmp/diffusiongemma-26b-a4b-fp32-512.bundle
```

The bundle contains 62 engines: embedding/self-conditioning, 30 QKV stages,
30 attention/feedforward stages, and output/soft-conditioning. Its family-owned
`runtime.json` declares format version 2 and the `qkv_layer` engine layout.
For layer `i`, `qkv_i.plan` materializes normalized FP32 query, key, and value
tensors, with RoPE applied to the query and key. `layer_i.plan` consumes all
three tensors to perform attention and feedforward computation. These used
bindings preserve the chosen activation boundary before attention and form
the execution dataflow. TensorRT owns lowering and tactic selection within
each stage.

Both stages execute through the public backend interface, while the native
Task contract stays unchanged. The runtime releases each stage module after
copying its outputs, preserving the encoder caches independently of engine
residency. Plan weights and selected-expert workspace are substantial; use a
GPU with enough memory for the compiled layer profiles.

The native primary task is `text_continuation`, exposed through the public
`trtmc::TextContinuation` SDK interface. Its prefix can be text or token IDs.
For instruction requests, apply the checkpoint's chat template before passing
the prefix. Runtime configuration accepts `max_new_tokens` and `seed`.

## Validate the exact checkpoint

Build `trtmc`, `trtmc_backend_trt`, `trtmc_model_diffusion_gemma`,
`test_diffusion_gemma_diagnostic`, and `test_diffusion_gemma_sdk_cpp` with
`TRTMC_BUILD_TESTS=ON`. The independent reference uses the official Transformers
commit declared in [tests/requirements.txt](tests/requirements.txt), installed
in a separate Python environment from the repository's pinned base environment.

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
TRTMC_BINARY=/absolute/native-build/trtmc \
TRTMC_RUNTIME_ROOT=/absolute/native-build \
TRTMC_NATIVE_BUILD_DIR=/absolute/native-build \
TRTMC_DIFFUSION_GEMMA_MODEL_DIR=/absolute/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b \
TRTMC_DIFFUSION_GEMMA_REFERENCE_PYTHON=/absolute/reference-env/bin/python \
TRTMC_E2E_ARTIFACT_DIR=/tmp/diffusiongemma-evidence \
python3 -m pytest families/diffusion_gemma/tests/test_e2e.py \
  --e2e-testcase diffusiongemma-26b-a4b-fp32-512 -q
```

The default E2E builds through the public Python API. Set
`TRTMC_DIFFUSION_GEMMA_BUNDLE` to an existing bundle to reuse a completed build
while retaining the same native/reference comparisons and generation checks.

The checkpoint override must resolve to a snapshot directory named by the
manifest's pinned revision. This checks revision provenance, not the authenticity
of file contents; differently named local directories do not establish pinned
checkpoint qualification.

The oracle compares all encoder K/V tensors, two complete fixed-canvas decoder
logit tensors, and both self-conditioning outputs against the official model.
It also checks that denoising leaves the encoder caches unchanged and runs a
real instruction request through the public C++ Task API. Fixed canvases keep
this numerical comparison independent of the implementations' random number
generators. Raw observations and measured errors are retained in the artifact
directory.

Compilation and source checks alone establish no inference or parity result.
Qualification requires the complete selected E2E above to pass on the target
hardware.

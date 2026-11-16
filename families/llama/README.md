# Llama

The family supports autoregressive `text_generation` bundles and stateless
`text_prediction_positions_to_token_scores` bundles. A bundle is built for one
Task. Full forward executes the complete input sequence once, without KV-cache
inputs, outputs, updates or generation. The graph returns every input row's
vocabulary logits; the runtime copies the requested rows into an owned result.

## Stateless full forward

Build the native family and the public C++ SDK consumer:

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_llama llama_forward_runner
PYTHONPATH=core/builder:. python -m tensorrt_model_connect llama build \
  meta-llama/Llama-3.2-1B-Instruct \
  --task text_prediction_positions_to_token_scores \
  --precision fp32 --max-sequence-length 32 -o llama-forward.bundle
```

FP32 disables TF32; FP16 and BF16 use the existing family precision boundaries.
Full forward currently supports a single
sequence on one device, without quantization, mixed layer overrides,
speculative execution or Edge-LLM offload. Generation bundles retain their
existing builders and runtime. Rebuild a separate bundle to use this Task.

Use `trtmc::TextPredictionPositionsToTokenScores` from `trtmc/features.hpp`
(or its existing C v1 API in `trtmc/features.h`). The request contains:

- `token_ids`: the exact model vocabulary IDs; no tokenization, BOS/EOS
  insertion, truncation or sampling occurs.
- `attention_mask`: empty for all-valid input or one 0/1 per token; zero
  excludes that key from attention. Query rows are retained, including padding.
- `blocked_attention`: required row-major `[L,L]` bytes, query by key;
  one blocks attention and zero permits it, subject to other masks.
- `prediction_positions`: nonempty input row indices, retaining order and
  duplicates. Select `0..L-1` to obtain the complete `[L,vocabulary]` result.
  These are Llama logits at each input row, predicting the following token;
  selecting a row does not hide its token as in masked-language modeling.
- `segment_ids`: must be empty; Llama has no token-type embedding.

The family Config fields are `causal` (boolean, default true) and `position_ids`
(optional int64 list, one per input token, default `0..L-1`). All positions
must fit the bundle's `max_sequence_length`. The causal triangle, key-padding
mask and caller's blocked-attention matrix are combined. Set `causal=false`
and supply an all-zero blocked-attention matrix for bidirectional attention.
Rows with no visible keys are rejected before execution. Each call is
independent; results own their values across subsequent calls and model release.
The vocabulary identity denotes the checkpoint location, not compatibility
with another checkpoint's tokenizer.

The consumer below accepts convenient JSON requests, filling in the all-zero
blocked-attention matrix and all prediction positions when omitted:

```json
{"token_ids": [1, 123, 456, 789], "causal": false}
```

```sh
build/families/llama/llama_forward_runner \
  llama-forward.bundle build request.json logits.f32 5 30
```

It writes row-major float32 logits and prints a JSON latency summary. Timing is
the synchronous public Task call wall time, including mask construction, H2D,
full-sequence network execution, D2H, selected-row copying and result packing.
Build/load, request construction, warmup, serialization and result release are
excluded. This is an end-to-end forward latency, not an engine-only latency,
prefill latency or token-generation throughput. The existing `trtmc-bench`
token-scores operation currently accepts only masked-language-model Tasks, so
this family-owned SDK consumer provides the measurement path for this Task.

## Correctness and performance validation

The validation module uses the full FP32 Hugging Face model with SDPA's math
backend and disables TF32. It checks the bringup call with `use_cache=False,
is_causal=False` against an explicit bidirectional mask, and distinguishes it
from causal attention. The keyword's behavior can depend on the source
implementation; validation must establish it rather than infer it from its name.

```sh
PYTHONPATH=core/builder:. python -m families.llama.tests.forward_validation \
  --model /path/to/local/checkpoint-snapshot \
  --bundle llama-forward.bundle --precision fp32 \
  --runtime-root build --runner build/families/llama/llama_forward_runner \
  --output-dir /tmp/llama-forward-results --warmup 5 --iterations 30

TRTMC_NATIVE_BUILD_DIR="$PWD/build" PYTHONPATH=core/builder:apps/benchmark:. \
  python -m pytest families/llama/tests/test_forward.py -q
```

The validation checks all returned logits, causal and bidirectional masks,
key padding, custom blocked attention, explicit positions, row ordering and
duplicates, single-token inputs, independent A/B/A calls, owned results and
invalid input rejection. It records checkpoint location/revision, software,
GPU, shapes, tolerances and measured latency in `report.json`. Its primary
workload is batch 1, sequence length 16 with deterministic token IDs. Newer
compatible checkpoints and any supported GPU can be used; Rubin qualification
and reproduction of a historical checkpoint are separate from this capability.

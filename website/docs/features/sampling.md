# Top-p (Nucleus) Sampling

Restricts token sampling to the smallest vocabulary prefix whose cumulative
probability mass is >= `top_p`, then renormalizes and samples from that nucleus.
Combined with temperature scaling, optional top-k pre-filtering, and optional
min-p filtering.

## Algorithm

Current implementation status: top-p/top-k/min-p filtering is host-side. The
sampler requests CPU logits (`LogitsLocation::HOST`), so text generation copies
the logits to CPU before applying nucleus filtering and sampling. This PR does
not add an on-device top-p sampling kernel; the only on-device sampler path is
greedy argmax, and the optional torch-multinomial path still builds the filtered
distribution on host before invoking its CUDA multinomial helper.

1. Sort logits descending, take top-k candidates
   - `top_k <= 0` means no top-k limit
   - when `top_k <= 1` and `0 < top_p < 1`, top-p uses the full vocabulary
2. Apply temperature: `logit / temperature`
3. Softmax over top-k
4. Apply min-p if enabled by dropping tokens below `min_p * max_probability`
5. Find nucleus: smallest prefix with cumulative probability >= top_p
6. Renormalize filtered probabilities
7. Sample using xorshift64 RNG seeded by `seed`

Falls back to argmax (greedy) when all sampling parameters are default, when
`temperature <= 0`, or when `top_p <= 0`. Values above `top_p=1.0` are clamped
to disabled top-p behavior.

## Usage

### CLI
```bash
./build/trtmc run bundle.trtfb --prompt "Once upon a time" \
  --temperature 0.7 --top-p 0.9 --min-p 0.05 --top-k 50 --seed 42 \
  --hf-python /opt/venv/bin/python
```

### Defaults
- `temperature`: 1.0 (no scaling)
- `top_p`: 1.0 (disabled -- use all top_k tokens)
- `min_p`: 0.0 (disabled)
- `top_k`: 1 (greedy argmax unless `top_p` is active)
- `seed`: -1 (use deterministic default seed 42)

`--top-p 0.9` is meaningful without also passing `--top-k`: the sampler uses the
full vocabulary as the top-p candidate set. Pass `--top-k N` to apply top-k first
and then top-p/min-p inside those candidates.

## Testing

Unit tests: `tests/cpp/test_sampler.cpp`
- `test_top_p_nucleus_restricts_candidates` -- with logits {5,3,1,0.5,0.1}
  and top_p=0.7, verifies only token 0 (the nucleus) is ever sampled
- `test_top_p_disabled_allows_full_topk` -- top_p=1.0 allows all candidates
- `test_combined_top_k_top_p` -- combined filtering
- `test_top_p_alone_uses_full_vocab` -- verifies `top_p` is not reduced to greedy
  when `top_k` is left at its default value
- edge-case tests cover `top_k=0`, `top_p=0`, invalid/clamped values, and seeded
  `reset()` reproducibility

E2E test: `tests/e2e/models/qwen/manifests/qwen3-0.6b-topp.json` (`trace_id: IT-E2E-TOPP-001`)
-- runs Qwen3-0.6B with temperature=0.7/top_p=0.9/top_k=50/seed=42 under
the `sampling_top_p` invariant contract. The contract verifies that the runtime
CLI forwards the requested sampling flags, produces non-empty sampled text, and
replays the same seeded output on the configured determinism rerun. It does not
compare sampled text to a greedy external reference, because stochastic decoding
is not expected to match greedy Hugging Face output token-for-token.

The E2E test covers the user-visible sampling contract, not device placement.
The CPU/host-side implementation constraint is covered by the sampler interface
and unit tests.

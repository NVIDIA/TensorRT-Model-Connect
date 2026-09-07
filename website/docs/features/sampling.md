# Top-p (Nucleus) Sampling

Top-p sampling keeps the smallest probability prefix whose cumulative mass
reaches `top_p`, renormalizes it, and samples one token. Temperature, top-k,
min-p, and repetition penalty may further shape the distribution.

Sampling behavior is family-owned behind the abstract `ITextGeneration`
interface. The concrete details below describe the current Qwen runtime; another
family may support a smaller set and must reject or document unsupported
controls.

## Algorithm

Qwen currently samples host logits:

1. choose the full vocabulary or a top-k candidate set;
2. apply temperature and softmax;
3. remove candidates below `min_p * max_probability` when min-p is active;
4. keep the smallest prefix whose cumulative probability reaches `top_p`;
5. renormalize and sample with the family-owned RNG.

When sampling is disabled, Qwen uses deterministic argmax. The implementation
lives in `families/qwen/runtime/sampler.cpp`; it is not shared runtime policy.

## Usage

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Once upon a time" \
  --temperature 0.7 \
  --top-p 0.9 \
  --min-p 0.05 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --seed 42
```

Current Task defaults are `temperature=1.0`, `top_p=1.0`, `min_p=0.0`,
`top_k=1`, `repetition_penalty=1.0`, and `seed=-1`. The family may choose its
documented deterministic behavior when no random seed is supplied.

## Testing

Focused Qwen sampler checks live in:

```text
families/qwen/tests/cpp/test_qwen_sampler.cpp
families/qwen/tests/manifests/qwen3-0.6b-topp.json
```

The unit test proves filtering and seeded behavior. The E2E proves that the
public Task request reaches the Qwen implementation and produces repeatable,
non-empty output for the declared case. Stochastic text is not expected to
match a greedy reference token for token.

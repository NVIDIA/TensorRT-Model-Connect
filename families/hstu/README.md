# HSTU ranking (generative recommender)

This family runs one HSTU ranking request (Meta's Hierarchical Sequential
Transduction Units, ICML 2024) through native TensorRT engines. The
architecture is NVIDIA recsys-examples' KuaiRand-1K ranking model
([recsys-examples](https://github.com/NVIDIA/recsys-examples) `c6e2bb22`,
`examples/hstu`), evaluated without its KV cache
(`InferenceRankingGR.forward_nokvcache`): one full forward per request.

| Fact | Value |
| --- | --- |
| HSTU layers, heads, hidden, head dim | 3, 4, 512, 128 |
| Attention | SiLU(QKᵀ/√128)/scaling_seqlen, causal, candidates see history + self, no relative attention bias |
| Contextual features | 6 (user_id + five user buckets) |
| Tables | video_id 1 000 000 rows (the reference's declared capacity; production uses growing dynamic embeddings), action_weights 233 |
| Request | 4096 history (item, action) events + 100 candidates: 8298 HSTU tokens |
| Ranking head | 512 → 512 ReLU → 8 task logits per candidate |
| Precision | BF16 (the reference's only inference precision); FP32 for the structure test |

## Weights

No public trained HSTU ranking checkpoint exists, so the family writes a
deterministic seeded one:

```bash
PYTHONPATH=core/builder:. python -m families.hstu.weights <dir>          # 4096 / 100
PYTHONPATH=core/builder:. python -m families.hstu.weights --small <dir>  # 16 / 8
```

`config.json` records the checkpoint's name, origin (`synthetic`),
generator, seed and initializers. Every value is drawn from PCG64's raw
stream with exact float32 arithmetic, so the same seed writes the same bytes
on every platform and NumPy version. The bundle states the name, generator
and seed in `hstu.checkpoint.json`. The values set the parity test's
numerics, never the kernels' cost, and say nothing about ranking quality.

## Engines

| Section | recsys-examples module | Work |
| --- | --- | --- |
| `embedding.plan` | `InferenceEmbedding` | the table lookups |
| `hstu.plan` | `HSTUBlockInference` | interleave + concat + positions, 3 HSTU layers, the candidates' L2-normalized rows |
| `ranking_head.plan` | the ranking `MLP` | logits [candidates, tasks] |

The runtime (`runtime/plugin.cpp`) implements `trtmc::IStructuredDecision`
(the `structured_decision` task: `trtmc hstu decide <bundle> --record
<record.json>`, and `trtmc-bench`'s `decide` operation). The record is a JSON
document (`tests/fixtures/request.json`): the user's contextual ids, the
history's `video_id` and `action_weights` arrays and the candidates'
`video_id` array. Every id is a row index into its table, checked against the
table's size; recsys-examples instead hashes raw user and video ids into
dynamic tables, which this family does not model. Each engine reads the
previous engine's output buffers in place on one CUDA stream, and the result
answers one question per ranking task (8) with a logit and a sigmoid
probability per candidate, so `trtmc-bench` counts one request as 8 questions.
Candidate ids must be distinct (each is one option).

Every shape is static at the checkpoint's geometry, and a record of another
history length or candidate count is refused. Padding a shorter history is
not equivalent: under the causal mask, real events would attend to the pad
rows. Serving another length needs a bundle built at that length; jagged
(variable-length) attention is not implemented.

The attention keeps its pointwise arithmetic between the two GEMMs (scale,
SiLU, mask) in the GEMM dtype, as recsys-examples' padded `pytorch_hstu_mha`
does: with an FP32 pointwise there, TensorRT 11.4's Myelin fused-attention
codegen fails (`FMHA IR verification failed`, operands of mixed type). The
linear 1/scaling_seqlen factor scales P·V in FP32 instead.

## Validation

The oracle is a pure-PyTorch padded reference (`tests/reference.py`; its
mask follows recsys-examples' `_get_valid_attn_mask`). `tests/golden.py`
writes the golden FP32 logits and the reference's own BF16 spread;
`tests/parity.py` judges native logits against them. Both tolerances were
declared before any TensorRT output was compared (`tests/data/README.md`):

| Case | Precision | Tolerance |
| --- | --- | --- |
| `hstu-kuairand-1k-rank` (4096 history, 100 candidates) | BF16 | max \|Δ\| ≤ 2.5e-3, task cosine ≥ 0.9995 |
| `hstu-kuairand-1k-small-rank` (16 history, 8 candidates) | FP32, TF32 off | max \|Δ\| ≤ 1e-4, task cosine ≥ 0.99999 |

The small case proves the structure: at 4096 history events a candidate-mask
or interleave defect hides under BF16 noise, while at 16 every probed
structural defect moves the logits at least 3.5x past the small tolerance
(`tests/test_reference.py` asserts more than 2x for each).

```bash
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_hstu trtmc_cli_hstu \
  test_hstu_record
ctest --test-dir build -R hstu_record
TRTMC_RUNTIME_ROOT=$PWD/build TRTMC_BINARY=$PWD/build/trtmc \
PYTHONPATH=core/builder:apps/benchmark:. \
python -m pytest families/hstu/tests/test_e2e.py --e2e-model hstu -q
```

# ESMC

This family builds the ESMC protein encoders for the
`masked_text_to_token_scores` task:

| Checkpoint | Layers | Hidden | Heads | Intermediate | Revision |
|---|---|---|---|---|---|
| `biohub/ESMC-300M` | 30 | 960 | 15 | 2560 | `96a2d8d61654ac0c08881be44da6c867686449b9` |
| `biohub/ESMC-600M` | 36 | 1152 | 18 | 3072 | `5e366569e675ad5ae3512f66bf0b53ffe80ee83e` |

One engine returns the vocabulary logits for every row of a fixed window at
batch 1, in fp32 or fp16. The manifests use a 256-token window; the
checkpoints were trained to 2048 and a build may use any length up to that.

## What this checkpoint is

A pre-norm transformer whose tensor layout looks like a Llama, with five
differences from `families/esm` that each load every tensor and still run:

- **No projection bias anywhere.** Attention and MLP matrices carry a weight
  only, and the builder refuses a config that declares `attention_bias` or
  `mlp_bias`.
- **QK LayerNorm.** The query and the key are normalised after their
  projections and before the rotation, over the whole hidden width (960 or
  1152), not per head. Those two norms have a weight and no bias.
- **The MLP is SwiGLU**, `down(silu(gate(x)) * up(x))`, not
  `down(gelu(up(x)))`.
- **Every residual branch is divided by `sqrt(num_layers / 36)`** - 0.912871
  at 30 layers, exactly 1.0 at 36. Leaving it out is invisible on 600M and
  wrong on every row of 300M.
- **The masked-LM head keeps its own decoder.** `tie_word_embeddings` is false
  and `lm_head.decoder.weight` is in the file.

What is unchanged from a rotary pre-norm encoder: the closing norm is
weight-only (`esmc.norm` has no bias), positions enter only as a rotation
inside each layer (the checkpoint stores no position table and no
`inv_freq`), the rotation pairs the halves of a head, and the attention scale
multiplies the scores. Both layer norms and the QK norms take their statistics
in float32 at every precision: the residual stream reaches 1.0e+04, whose
square overflows float16.

## Tokenizer

`tokenizer.json` is a 33-entry BPE with no merges, no normaliser and no
pre-tokenizer, while the head is 64 columns wide. Three rules follow from it,
and the C++ tokenizer reproduces them:

- the added tokens (`<cls>`, `<pad>`, `<eos>`, `<unk>`, `<mask>`, and the chain
  break `|`) match whole and longest first, so `MK<mask>TV` is five ids;
- everything else is one id per character, so a residue string is one id per
  residue;
- a character outside the alphabet is one `<unk>` of its own - `mktv` is four
  `<unk>`s and a space is one `<unk>`, never dropped, and a multi-byte
  character is a single `<unk>`.

The bundle's `vocab.txt` is written at the engine's width: the 33 named tokens
in id order, then `<esmc_unused_33>` ... `<esmc_unused_63>` for the columns the
checkpoint's head has but the tokenizer does not name. No residue string
reaches those ids; they exist so the file's width matches the engine's.

## Validation

Measured on the three cases with the reference being `transformers==5.24.0`'s
`EsmcForMaskedLM` in fp32, no `trust_remote_code`. Hardware and stack: NVIDIA
A100-SXM4-40GB, TensorRT 11.1.0.106, CUDA 13.0, torch 2.12.0+cu130,
transformers 5.24.0, the repository's `Dockerfile.dev.x86-gpu` image. The same
engines were also measured on an NVIDIA L4 with the same image: fp32 4.2e-05
and fp16 1.1e-01 at the masked row, inside the same tolerances.

| | 300M fp32 | 300M fp16 | 600M fp32 | 600M fp16 |
|---|---|---|---|---|
| logit max abs at the masked row | 5.5e-05 | 1.3e-01 | 4.4e-05 | 4.7e-02 |
| logit max abs, all real rows | 9.9e-05 | 3.9e-01 | 6.1e-05 | 2.2e-01 |
| cosine at the masked row | 1.00000000 | 0.99999955 | 1.00000000 | 0.99999954 |
| masked token top-1 | matches | matches | matches | matches |
| engine plan | 1272 MiB | 638 MiB | 2196 MiB | 1100 MiB |
| build time on A100 | 20 s | 39 s | 21 s | 45 s |

The residual stream peaks at 1.03e+04 (300M) and 7.5e+03 (600M), which is why
every LayerNorm, including the two QK norms, takes its statistics in float32.

fp16 needs its own interpretation. The reference's own fp16 model is 1.1e-01
(300M) and 7.8e-02 (600M) away from its fp32 self, and the engines land in the
same range - the 300M engine is 1.2x the reference's own fp16 noise and the
600M engine is below it - so a tolerance below that would fail a correct
build. The fp16 gate in `tests/test_e2e.py` is 3.0e-01, a 2.4x to 6.4x margin
over a correct build, and it rejects the O(1) structural breaks; it cannot
separate the weakest break (a dropped residual scale, 1.5e-01 in fp16) from
fp16 noise. The fp32 case gates that one, and
`test_fp32_gate_rejects_unscaled_residuals` builds it to keep the claim
measured rather than argued.

An offline fp64 NumPy transcription of the graph in this family agrees with the
same reference to 1.6e-04 on every row of the 300M case, which is the fp32
rounding gap rather than a formula difference. The same transcription - not an
engine - measured the structural breaks the fp32 tolerance is chosen against,
all at the masked row: a dropped residual scale 5.2e-02, an erf-gelu MLP
6.2e+00, a tied decoder 2.0e+01, a per-head QK norm 2.0e+01, and no QK norm at
all 2.2e+01.

The bundle's C++ tokenizer was checked against the reference `tokenizer.json`
on the published vocabulary over 300 random strings (residues, whitespace,
unknown characters, multi-byte characters, partial special tokens): every id
matched.

Precisions:

- **fp32** is the default and the pre-merge case. Its gate is
  `_LOGIT_TOLERANCE` in `tests/test_e2e.py`.
- **fp16** is built and measured, with the separate tolerance and the caveat
  above.
- **bf16** is not built. The checkpoints were trained in bf16, but this family
  builds the two precisions above only.
- **fp8** is not implemented. There is no `quantization.py` and the builder
  refuses any quantization other than none.

## Known limitations

- Safetensors checkpoints only (`model.safetensors` or a
  `model.safetensors.index.json` shard set). A `pytorch_model.bin` is refused
  by name rather than converted on the fly.
- Single-chain inputs. The reference can take a `sequence_id` tensor and mask
  attention between chains; that input is not part of this Task and is not
  exposed here, so a multi-chain string is scored as one chain.
- Batch 1, and no tensor or context parallelism.
- No AIPerf reference adapter. `families/esm/reference/adapter.py` was left out
  on purpose; a profiling adapter can be added when it is needed.

# LFM2.5 vision and language

This family owns the SigLIP2 NaFlex encoder, multimodal projector, hybrid language
decoder, native RGB preprocessing, tokenizer, and generation state for
[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B).
The reference checkpoint revision is
`919fde3d022e3f90a4716006f993938ee8c2eb97`.

Build a BF16 bundle with the advertised 32,768-token context:

```bash
trtmc lfm2_vl build LiquidAI/LFM2.5-VL-1.6B \
  --revision 919fde3d022e3f90a4716006f993938ee8c2eb97 \
  --precision bf16 --max-sequence-length 32768 -o lfm25-vl.bundle
```

The native `images_text_to_text` Task accepts ordered images and text across
multiple conversation turns. Image parts retain their order within each message. The
`text_conversation` Task accepts text-only messages, tool definitions, and tool
results, returning typed tool calls with their generated token IDs. The runtime
does not execute tools.

```bash
trtmc run lfm25-vl.bundle --task images_text_to_text \
  --image cabin.png --prompt "Describe this image." --max-new-tokens 64
```

Generation options belong to this family: `temperature`, `top_k`, `top_p`,
`min_p`, `repetition_penalty`, `seed`, and `max_new_tokens`. A `top_k` of one
selects greedy decoding. For the model card's sampled policy, use
`temperature=0.1`, `top_k=50`, `min_p=0.15`, and `repetition_penalty=1.05`.
Conversation formatting also accepts `keep_past_thinking` (default false):
earlier assistant reasoning is removed, while the latest assistant reasoning
is retained, matching the checkpoint's chat template.

Image options are `min_image_tokens` (default 64), `max_image_tokens` (default
256), `max_tiles` (default 10), and `do_image_splitting` (default true). Tiles
are 512 by 512 pixels; tiled inputs include a thumbnail. The bundle accepts
at most 1,024 tokens per resized image and ten tiles per source image. RGB byte
images and normalized RGB Float32 images are accepted; normalized pixels are
converted to the corresponding RGB byte values before preprocessing.

Reference validation remains on Transformers 5.2.0. Before its loaders run,
family helpers reject config-supplied Hub/fallback kernel selectors, including
nested attention/expert settings, configuration-file redirects, and local
adapter redirects. The sampling
helper also checks the already loaded model before forwarding it. This is a
scoped validation-input mitigation; it neither patches the library generally
nor qualifies a newer Transformers release. Use the declared checkpoint for
reference validation.

The family owns its original-reference fixtures and complete model-card
acceptance matrix, including all eight advertised languages, multiple images,
OCR, aspect ratios, image budgets, tool use, and the full context boundary:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/lfm2_vl/tests --e2e-model lfm2_vl -q
```

Selected E2E requires `TRTMC_RUNTIME_ROOT` pointing to the native runtime
libraries, including the family DSO, C API, and TensorRT backend. The test builds
its own clients with a C++ compiler and CUDA toolkit headers; installed runtimes
do not need to ship test executables. An existing
pinned snapshot and bundle can be supplied through `TRTMC_LFM25_VL_CHECKPOINT`
and `TRTMC_LFM25_VL_BUNDLE`.

The manifest's `external_files` declares the original model-card photo for
preparation before offline E2E. For a local run, download it first and point
`TRTMC_LFM2_VL_MODEL_DIR` to the directory containing `model-card.jpg`:

```bash
mkdir -p /tmp/lfm2-vl-inputs
curl --fail --location --max-time 60 \
  'https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg' \
  -o /tmp/lfm2-vl-inputs/model-card.jpg
export TRTMC_LFM2_VL_MODEL_DIR=/tmp/lfm2-vl-inputs
```

When this directory is supplied, the test requires the staged file and never
falls back to a download. Without it, the test downloads the declared photo
with a 60-second timeout for standalone or network-enabled runs. `HF_HUB_OFFLINE`
controls checkpoint resolution, independently of this external image download.
Evidence records the declared URL, acquisition mode, and actual input file;
a local file's source is not authenticated by declaring the URL. Native and
reference inference use the same decoded RGB pixels.

To include real checkpoint token IDs and rendered-prompt BOS controls, set
`TRTMC_LFM2_VL_TOKENIZER_JSON` to the pinned checkpoint's `tokenizer.json` when
running `test_lfm2_vl_pretokenizer` and `test_lfm2_vl_runtime`.

Acceptance compares task behavior with the original model: caption facts,
ordered image colors, OCR content with whitespace normalized, answer language
and meaning, typed tool names/arguments/state, and the context endpoint. Exact
tokens, logits, and a 64-step coupled sampling comparison are retained as
diagnostics. A separate public Task case checks the recommended sampling policy.
The checkers reject unknown or contradictory answers. Shared reference answer
errors are recorded separately from native/reference semantic differences.

# Laguna

`families/laguna` builds poolside's `LagunaForCausalLM` checkpoints (`model_type` `laguna`) into one TensorRT engine
and serves them through the `text_continuation` Task. The validated checkpoint is
[`poolside/Laguna-XS.2`](https://huggingface.co/poolside/Laguna-XS.2) at revision
`a397bde04501591d0ac3d74a5d9d0a4ec298ffb8`: 33.4 B parameters, 40 layers, 256-expert top-8 MoE, BF16.

## Build

```bash
trtmc build poolside/Laguna-XS.2 --revision a397bde04501591d0ac3d74a5d9d0a4ec298ffb8 \
  --precision bf16 --max-sequence-length 2048 -o laguna-xs-2.bundle
```

`bf16` (default) keeps the checkpoint's weights as stored; the Laguna-XS.2 engine holds 62.35 GiB of constants.
`fp32` widens them and is meant for exact small-model parity runs. The engine has a 1..128-row prefill profile and
a one-row decode profile; every routed expert is evaluated on every row and unselected experts get weight zero. The
sliding window of the sliding-attention layers is applied inside the engine from the position IDs, so the runtime
keeps one append-only KV cache of `max_sequence_length` rows for every layer.

## Run

```bash
trtmc run laguna-xs-2.bundle --prompt "What is the largest planet in our solar system?" \
  --use-chat-template true --enable-thinking false --max-new-tokens 32
```

Config fields: `max_new_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `use_chat_template`,
`enable_thinking`, `system_prompt`. Generation stops at EOS IDs 2 (`〈|EOS|〉`) and 24 (`</assistant>`). The chat
format is the single-user-turn rendering of Laguna's `chat_template.jinja` (default system message, `<user>` /
`<assistant>` framing, `</think>` or `<think>` after the generation prompt); the builder enables it only when the
checkpoint's template contains every literal fragment of that framing (a fragment check, not a template evaluation).

## Provenance and licenses

- Model weights, tokenizer files and the chat template are poolside's, published under Apache-2.0 (the model card
  also refers to poolside's Acceptable Use Policy). Users download them separately; the repository contains none of
  them. The runtime's chat renderer, default system message and pre-tokenizer patterns reproduce that Apache-2.0
  material.
- `runtime/unicode_tables.h` holds the Unicode 16.0.0 `L*` and `N*` category ranges, derived from
  [UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt) (SHA-256
  `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`) and redistributed under the Unicode License v3
  in `UNICODE_LICENSE.txt`, which is installed with this family. Regenerate with the command below, then apply
  clang-format:

```bash
python families/laguna/tools/generate_unicode_tables.py UnicodeData.txt \
  families/laguna/runtime/unicode_tables.h
```

- The E2E reference and the tiny-model parity test use `transformers==5.22.2` (Laguna support starts at 5.7.0),
  `huggingface-hub==1.33.0` (transformers 5.22.2 needs at least 1.31; the pin keeps it on the 1.x line of the
  repository's images) and `accelerate`, all Apache-2.0, installed from this family's `requirements.txt`.

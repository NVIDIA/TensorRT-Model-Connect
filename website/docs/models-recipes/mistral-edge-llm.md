---
title: Mistral W8A16 with Edge-LLM
---

For `RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w8a16`, use the prepared
GPTQ weights with FP16 compute. The family can select the Edge-LLM 0.12.0
checkpoint-direct builder when the Model Connect C++ runtime includes the
matching SDK. The builder interpreter can contain a PyPI or source install.

```sh
trtmc mistral build RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w8a16 \
  --revision 5079e6662632d88ac726516beccf202c925402a0 \
  --precision fp16 --quantization w8a16 --max-sequence-length 256 \
  --edge-builder-python /path/to/edge-env/bin/python -o mistral.bundle
trtmc run mistral.bundle --prompt 'What is the capital of Japan? Answer in one word.' \
  --use-chat-template true --enable-thinking false --max-new-tokens 10
```

A failed Edge build warns and retries the native TensorRT graph using the
same stored weights. A runtime error propagates. See the
[family guide](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/mistral/README.md#prepared-gptq-w8a16)
for the distinction between this prepared checkpoint and Riva FP8.

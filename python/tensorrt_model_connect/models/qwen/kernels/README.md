# Qwen family kernels

This directory contains kernel-integration code owned by the Qwen family. It
does not add another Model Connect runtime abstraction: graph Recipes still
select the TensorRT region, and the shared TVM-FFI plugin still invokes the
selected DSO.

## FlashInfer linear decode POC

`export_flashinfer_decode_attention.py` exports a Qwen3-8B decode-attention DSO
for the `qwen.decode_attention_region@2` Recipe boundary:

```text
run(query, key, value, key_value_lengths, context)
```

The Qwen builder scales and rounds the query to BF16 before this boundary, so
the FlashInfer kernel uses a softmax scale of `1.0`.

FlashInfer 0.6.15's contiguous CuTe decode kernel uses the full K tensor shape
as the valid sequence length. Model Connect instead keeps a fixed-capacity
linear KV cache and supplies the current length as an `int32[1]` CUDA tensor.
`flashinfer_device_kv_length.patch` adds that optional device-length input to
the existing kernel. The default remains unchanged for existing FlashInfer
callers.

Build the POC from the repository root:

```bash
python -m pip install \
  "nvidia-cutlass-dsl==4.5.0" \
  "apache-tvm-ffi==0.1.12" \
  "flashinfer-python==0.6.15"

git clone --branch v0.6.15 --depth 1 \
  https://github.com/flashinfer-ai/flashinfer.git \
  artifacts/flashinfer-v0.6.15

git -C artifacts/flashinfer-v0.6.15 apply \
  "$PWD/python/tensorrt_model_connect/models/qwen/kernels/flashinfer_device_kv_length.patch"

PYTHONPATH="$PWD/artifacts/flashinfer-v0.6.15:$PWD/python" \
  python python/tensorrt_model_connect/models/qwen/kernels/export_flashinfer_decode_attention.py \
  --output artifacts/qwen3-flashinfer-linear.so
```

The exporter is intentionally fixed to the tutorial's Qwen3-8B BF16 decode
shape and SM 10.3. Use the ordinary Recipe build and load-time binding flow in
the [Bring Your Own Kernel with TVM-FFI tutorial](../../../../../website/docs/tutorials/beginner/bring-your-own-kernel.md).

This is a POC of the integration boundary, not a vendored FlashInfer fork. The
small optional-length change should live upstream before this becomes a
supported built-in kernel.

<!-- Collaborative review anchor: batch 2. -->

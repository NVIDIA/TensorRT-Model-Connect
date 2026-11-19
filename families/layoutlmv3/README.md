# LayoutLMv3 document features

This family accepts a document image, caller-supplied words and pixel boxes.
The FP32 encoder returns text-token features and their word mappings; it does
not perform OCR or apply a downstream prediction head.

Build and install a matching native package with `TRTMC_ENABLE_BYOK=ON` (the
default) before building bundles. Normalization, affine projections, GELU and
softmax use family-owned FP32 kernels through the public TVM-FFI BYOK bridge.
The remaining graph operations use TensorRT. Image resizing uses PyTorch's
CPU library to preserve the processor arithmetic.

The independent reference retains Transformers 5.20 in its separate environment.
Prepare it with `MODEL=layoutlmv3 python -m tools.ci pipeline setup`, then run
the family tests from the normal base environment with the native paths required
by the repository validation guide. The E2E adapter invokes the prepared
reference interpreter without changing the base Transformers installation.

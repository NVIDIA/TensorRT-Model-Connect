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

The resolved LibTorch CMake package must use the runtime toolchain's C++ ABI.
CMake checks its compile flags and imported target before applying them to the
family and tokenizer helper. Native-prefix installs retain the discovered Torch
library directory. Select a different family with `-DTRTMC_BUILD_FAMILIES=<family>`
or use an empty value for core-only builds to omit LayoutLMv3's dependencies.

Install this family's `requirements.txt` alongside a relocated wheel. It declares
the CUDA 13 cuBLASLt runtime. The loader prefers the wheel-relative NVIDIA
library, then the library discovered at build time, then the system SONAME.

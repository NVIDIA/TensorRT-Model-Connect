<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# V-JEPA2 video features

This family provides a fixed FP32 clip encoder for
[`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).
It returns patch features and their source-frame spans and spatial boxes.
It does not expose the predictor or classification heads.

The TensorRT graph uses family-owned FP32 normalization, linear, rotary,
softmax and GELU operations through the public TVM-FFI BYOK bridge. Build with
`TRTMC_ENABLE_BYOK=ON` (the default) and install the matching native wheel
before building or loading a bundle. TensorRT supplies each kernel's stream and
workspace. This family uses CUDA runtime and cuBLASLt; it does not call native
PyTorch GPU operators. The CPU resize computes only the columns needed by the
center crop and uses fused accumulation to preserve the processor's rounding.

The E2E case compares both forward and reversed clips with the original eager
Transformers reference. Its `reference_environment: isolated` manifest uses
`tests/requirements.txt`; the family invokes that prepared interpreter while
the builder remains in the normal package environment. Prepare the environment
through the repository's setup step before selecting the E2E case. The test
retains the original checkpoint, inputs and numerical gates.

See `NOTICE` for upstream algorithm attribution.

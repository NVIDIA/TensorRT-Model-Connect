/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Experimental FP32 specialization of the already-qualified no-Torch
 * cuBLASLt linear-probe infrastructure.  It exists only to prove the earliest
 * Wan2.2 call-0 time-path mismatch and is not a production plugin.
 */

// Include the CUDA/TRT declarations before remapping the two implementation
// constants.  Header guards then make the included reusable implementation see
// FP32 matrix layouts and FP32 TensorRT tensors without rewriting its lifecycle,
// serialization, heuristic enumeration, or qualification C ABI.
#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#define CUDA_R_16BF CUDA_R_32F
#define kBF16 kFLOAT
#include "../dit_linear_probe/wan22_dit_linear_probe_plugin.cu"
#undef kBF16
#undef CUDA_R_16BF

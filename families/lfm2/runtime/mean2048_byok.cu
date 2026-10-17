/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Mean2048 implementation adapted from PyTorch commit
 * 7661cd9c6b841b62b7f411aa52ec51f05457263b: MeanOps in
 * ATen/native/SharedReduceOps.h and setReduceConfig,
 * input_vectorized_thread_reduce_impl and block_x_reduce in
 * ATen/native/cuda/Reduce.cuh. One fixed source-derived algorithm.
 */
/*
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon
 * Bottou, Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research
 * Institute (Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute
 * (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 *
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to
 * further mark their specific copyright on a particular contribution, they
 * should indicate their copyright solely in the commit message of the change
 * when it is committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 * Laboratories America and IDIAP Research Institute nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {
constexpr int64_t kMaximumRows = 131072;
constexpr int64_t kReductionWidth = 2048;

void require(bool condition, const char* message) {
    if (!condition)
        TVM_FFI_THROW(ValueError) << message;
}
void require_metadata(tvm::ffi::TensorView tensor) {
    require(tensor.ndim() == 2, "mean2048 requires rank-two tensors");
    const auto dtype = tensor.dtype();
    require(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1,
            "mean2048 requires FP32 tensors");
    require(tensor.device().device_type == kDLCUDA && tensor.device().device_id >= 0,
            "mean2048 requires a CUDA device");
}
uintptr_t storage_address(tvm::ffi::TensorView tensor, size_t bytes) {
    require(tensor.data_ptr() != nullptr, "mean2048 requires data storage");
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto offset = tensor.byte_offset();
    constexpr auto maximum = std::numeric_limits<uintptr_t>::max();
    require(offset <= maximum - base, "mean2048 byte offset overflows address space");
    const auto address = base + static_cast<uintptr_t>(offset);
    require(address % alignof(float) == 0, "mean2048 requires FP32-aligned storage");
    require(bytes <= maximum - address, "mean2048 storage range overflows address space");
    return address;
}
struct Mean2048Contract {
    int64_t rows;
    int device;
    const float* input;
    float* output;
    size_t input_bytes;
    size_t output_bytes;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t grid_x;
    size_t shared_bytes;
    uint32_t input_alignment_bytes;
};
Mean2048Contract validate_contract(tvm::ffi::TensorView squares, tvm::ffi::TensorView shape_like,
                                   tvm::ffi::TensorView output) {
    require_metadata(squares);
    require_metadata(shape_like);
    require_metadata(output);
    const auto rows = squares.size(0);
    require(rows >= 1 && rows <= kMaximumRows, "mean2048 row count is outside its profile");
    require(squares.size(1) == kReductionWidth, "mean2048 reduction width must be 2048");
    for (const auto tensor : {shape_like, output}) {
        require(tensor.size(0) == rows && tensor.size(1) == 1,
                "mean2048 output and carrier must have shape [R,1]");
        require(tensor.device().device_id == squares.device().device_id,
                "mean2048 tensors must use the same CUDA device");
    }
    // Bound all dimensions before TensorView checks explicit stride products.
    for (const auto tensor : {squares, shape_like, output})
        require(tensor.IsContiguous(), "mean2048 requires contiguous tensors");
    const size_t input_bytes = static_cast<size_t>(rows) * kReductionWidth * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows) * sizeof(float);
    const auto source = storage_address(squares, input_bytes);
    const auto target = storage_address(output, output_bytes);
    require(source + input_bytes <= target || target + output_bytes <= source,
            "mean2048 input and output storage must not overlap");
    // Source setReduceConfig for vectorized width2048, warp32/maxthreads512.
    // No blockY/global split occurs for this bounded contiguous contract.
    uint32_t height = 1;
    while (height < 16 && static_cast<int64_t>(height * 2) <= rows)
        height *= 2;
    const uint32_t width = 512 / height;
    return {rows,
            squares.device().device_id,
            reinterpret_cast<const float*>(source),
            reinterpret_cast<float*>(target),
            input_bytes,
            output_bytes,
            width,
            height,
            static_cast<uint32_t>((rows + height - 1) / height),
            width > 32 ? 512 * sizeof(float) : 0,
            static_cast<uint32_t>(source % 16)};
}

struct alignas(16) Float4 {
    float val[4];
};
static_assert(sizeof(Float4) == 16 && alignof(Float4) == 16);

__global__ void mean2048_kernel(const float* input, float* output, int64_t rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const bool active = row < rows;
    float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
        const float* data = input + row * kReductionWidth;
        int end = kReductionWidth;
        const int shift = reinterpret_cast<uintptr_t>(data) % alignof(Float4) / sizeof(float);
        if (shift > 0) {
            // The guarded head loads are within the original logical row.
            data -= shift;
            end += shift;
            if (threadIdx.x >= shift && threadIdx.x < 4)
                accumulators[0] = __fadd_rn(accumulators[0], data[threadIdx.x]);
            end -= 4;
            data += 4;
        }
        int index = threadIdx.x;
        const int stride = blockDim.x;
        while (index * 4 + 3 < end) {
            const auto values = reinterpret_cast<const Float4*>(data)[index];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                accumulators[j] = __fadd_rn(accumulators[j], values.val[j]);
            index += stride;
        }
        const int tail_start = end - end % 4;
        const int tail_index = tail_start + threadIdx.x;
        if (tail_index < end)
            accumulators[0] = __fadd_rn(accumulators[0], data[tail_index]);
    }
    float value = accumulators[0];
#pragma unroll
    for (int j = 1; j < 4; ++j)
        value = __fadd_rn(value, accumulators[j]);
    extern __shared__ float shared[];
    if (blockDim.x > 32) {
        const int index = threadIdx.y * blockDim.x + threadIdx.x;
        shared[index] = value;
        for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                value = __fadd_rn(value, shared[index + offset]);
                shared[index] = value;
            }
        }
    }
    // Inactive tail rows must participate in every barrier and shuffle.
    __syncthreads();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, offset));
    if (active && threadIdx.x == 0)
        output[row] = __fmul_rn(value, 0.00048828125f);
}

void mean2048(tvm::ffi::TensorView squares, tvm::ffi::TensorView shape_like,
              tvm::ffi::TensorView output) {
    const auto contract = validate_contract(squares, shape_like, output);
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
    require(device == contract.device, "mean2048 tensor device is not the active CUDA device");
    // nullptr is a valid inherited CUDA default stream.
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    mean2048_kernel<<<dim3(contract.grid_x), dim3(contract.block_x, contract.block_y),
                      contract.shared_bytes, stream>>>(contract.input, contract.output,
                                                       contract.rows);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
}
} // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mean2048, mean2048);

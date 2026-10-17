/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * The reduction follows the FP32 mean implementation in PyTorch commit
 * 7661cd9c6b841b62b7f411aa52ec51f05457263b, specifically MeanOps in
 * aten/src/ATen/native/SharedReduceOps.h and thread_reduce_impl, block_x_reduce,
 * and setReduceConfig in aten/src/ATen/native/cuda/Reduce.cuh.
 * The FP32 reciprocal square root follows rsqrt_wrapper in the same commit's
 * aten/src/ATen/native/cuda/UnaryOpsKernel.cu: CUDA ::rsqrt(float), or ::rsqrtf.
 * The decode SUM3 follows the same commit's ReduceSumProdKernel.cu FP32
 * accumulation and Reduce.cuh contiguous width-three order with zero identities.
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
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin,
 * Jason Weston) Copyright (c) 2006      Idiap Research Institute (Samy Bengio) Copyright (c)
 * 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
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
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
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
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
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
constexpr int64_t kReductionWidth = 64;

void require(bool condition, const char* message) {
    if (!condition)
        TVM_FFI_THROW(ValueError) << message;
}

void require_metadata(tvm::ffi::TensorView tensor) {
    require(tensor.ndim() == 3, "LFM2 mean64 requires rank-three tensors");
    const auto dtype = tensor.dtype();
    require(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1,
            "LFM2 mean64 requires FP32 tensors");
    require(tensor.device().device_type == kDLCUDA && tensor.device().device_id >= 0,
            "LFM2 mean64 requires a CUDA device");
}

uintptr_t storage_address(tvm::ffi::TensorView tensor, size_t bytes) {
    require(tensor.data_ptr() != nullptr, "LFM2 mean64 requires data storage");
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto offset = tensor.byte_offset();
    constexpr auto maximum = std::numeric_limits<uintptr_t>::max();
    require(offset <= maximum - base, "LFM2 mean64 byte offset overflows address space");
    const auto address = base + static_cast<uintptr_t>(offset);
    require(address % alignof(float) == 0, "LFM2 mean64 requires FP32-aligned storage");
    require(bytes <= maximum - address, "LFM2 mean64 storage range overflows address space");
    return address;
}

struct Mean64Contract {
    int64_t query_rows;
    int64_t heads;
    int64_t reduction_rows;
    int device;
    const float* input;
    float* output;
    size_t input_bytes;
    size_t output_bytes;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t grid_x;
    size_t shared_bytes;
};

Mean64Contract validate_contract(tvm::ffi::TensorView squares, tvm::ffi::TensorView shape_like,
                                 tvm::ffi::TensorView output) {
    require_metadata(squares);
    require_metadata(shape_like);
    require_metadata(output);
    const int64_t rows = squares.size(0);
    const int64_t heads = squares.size(1);
    require(rows >= 1 && rows <= kMaximumRows, "LFM2 mean64 row count is outside its profile");
    require(heads == 8 || heads == 32, "LFM2 mean64 requires eight or thirty-two heads");
    require(squares.size(2) == kReductionWidth, "LFM2 mean64 reduction width must be 64");
    for (const auto tensor : {shape_like, output}) {
        require(tensor.size(0) == rows && tensor.size(1) == heads && tensor.size(2) == 1,
                "LFM2 mean64 output and shape carrier must have shape [R, H, 1]");
        require(tensor.device().device_id == squares.device().device_id,
                "LFM2 mean64 tensors must use the same CUDA device");
    }
    // Check contiguity only after bounding every dimension. Null strides denote
    // implicit contiguous DLTensor storage and must not be dereferenced.
    for (const auto tensor : {squares, shape_like, output})
        require(tensor.IsContiguous(), "LFM2 mean64 requires contiguous tensors");
    // Bounded dimensions make every element and byte count representable.
    const int64_t reduction_rows = rows * heads;
    const size_t input_bytes =
        static_cast<size_t>(reduction_rows) * kReductionWidth * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(reduction_rows) * sizeof(float);
    const uintptr_t input_address = storage_address(squares, input_bytes);
    const uintptr_t output_address = storage_address(output, output_bytes);
    require(input_address + input_bytes <= output_address ||
                output_address + output_bytes <= input_address,
            "LFM2 mean64 input and output storage must not overlap");

    // Shape-derived forms of the pinned CUDA reduction configuration for width64.
    // H is 8 or 32, so the only row count below16 is eight-row decode.
    const uint32_t block_x = reduction_rows == 8 ? 64 : 32;
    const uint32_t block_y = reduction_rows == 8 ? 8 : 16;
    const auto grid_x = static_cast<uint32_t>((reduction_rows + block_y - 1) / block_y);
    return {rows,
            heads,
            reduction_rows,
            squares.device().device_id,
            reinterpret_cast<const float*>(input_address),
            reinterpret_cast<float*>(output_address),
            input_bytes,
            output_bytes,
            block_x,
            block_y,
            grid_x,
            block_x == 64 ? 512 * sizeof(float) : 0};
}

__global__ void mean64_kernel(const float* input, float* output, int64_t rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const bool active = row < rows;
    float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int column = threadIdx.x + i * blockDim.x;
        if (active && column < kReductionWidth)
            accumulators[i] = __fadd_rn(accumulators[i], input[row * kReductionWidth + column]);
    }
    float value = accumulators[0];
#pragma unroll
    for (int i = 1; i < 4; ++i)
        value = __fadd_rn(value, accumulators[i]);

    extern __shared__ float shared[];
    if (blockDim.x > 32) {
        const int index = threadIdx.y * blockDim.x + threadIdx.x;
        shared[index] = value;
        __syncthreads();
        if (threadIdx.x < 32) {
            value = __fadd_rn(value, shared[index + 32]);
            shared[index] = value;
        }
    }
    // Inactive tail rows still participate in every block barrier and shuffle.
    __syncthreads();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, offset));
    if (active && threadIdx.x == 0)
        output[row] = __fmul_rn(value, 0.015625f);
}

void mean64(tvm::ffi::TensorView squares, tvm::ffi::TensorView shape_like,
            tvm::ffi::TensorView output) {
    const auto contract = validate_contract(squares, shape_like, output);
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
    require(device == contract.device, "LFM2 mean64 tensor device is not the active CUDA device");
    // A null inherited stream is CUDA's valid default stream.
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    mean64_kernel<<<dim3(contract.grid_x), dim3(contract.block_x, contract.block_y),
                    contract.shared_bytes, stream>>>(contract.input, contract.output,
                                                     contract.reduction_rows);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
}

struct RsqrtContract {
    int elements;
    int device;
    const float* input;
    float* output;
    size_t bytes;
    uint32_t grid_x;
};

RsqrtContract validate_rsqrt_contract(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    for (const auto tensor : {input, output}) {
        require(tensor.ndim() == 2 || tensor.ndim() == 3,
                "LFM2 rsqrt requires rank-two or rank-three tensors");
        const auto dtype = tensor.dtype();
        require(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1,
                "LFM2 rsqrt requires FP32 tensors");
        require(tensor.device().device_type == kDLCUDA && tensor.device().device_id >= 0,
                "LFM2 rsqrt requires a CUDA device");
    }
    require(input.ndim() == output.ndim(), "LFM2 rsqrt input and output ranks must match");
    const int64_t rows = input.size(0);
    require(rows >= 1 && rows <= kMaximumRows, "LFM2 rsqrt row count is outside its profile");
    const bool wide = input.ndim() == 2;
    const int64_t heads = wide ? 1 : input.size(1);
    if (wide) {
        require(input.size(1) == 1, "LFM2 wide rsqrt input must have shape [R, 1]");
        require(output.size(0) == rows && output.size(1) == 1,
                "LFM2 wide rsqrt input and output must have the same shape [R, 1]");
    } else {
        require(heads == 8 || heads == 32, "LFM2 rsqrt requires eight or thirty-two heads");
        require(input.size(2) == 1, "LFM2 rsqrt input must have shape [R, H, 1]");
        require(output.size(0) == rows && output.size(1) == heads && output.size(2) == 1,
                "LFM2 rsqrt input and output must have the same shape [R, H, 1]");
    }
    require(output.device().device_id == input.device().device_id,
            "LFM2 rsqrt tensors must use the same CUDA device");
    // Bound all dimensions before inspecting potentially explicit strides.
    require(input.IsContiguous() && output.IsContiguous(),
            "LFM2 rsqrt requires contiguous tensors");
    const int elements = static_cast<int>(rows * heads);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    const uintptr_t source = storage_address(input, bytes);
    const uintptr_t target = storage_address(output, bytes);
    require(source + bytes <= target || target + bytes <= source,
            "LFM2 rsqrt input and output storage must not overlap");
    return {elements,
            input.device().device_id,
            reinterpret_cast<const float*>(source),
            reinterpret_cast<float*>(target),
            bytes,
            static_cast<uint32_t>((elements + 255) / 256)};
}

__global__ void rsqrt_kernel(const float* input, float* output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = ::rsqrtf(input[index]);
}

void rsqrt_fp32(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    const auto contract = validate_rsqrt_contract(input, output);
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
    require(device == contract.device, "LFM2 rsqrt tensor device is not the active CUDA device");
    // Preserve CUDA's valid default stream as well as an inherited nonzero stream.
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    rsqrt_kernel<<<contract.grid_x, 256, 0, stream>>>(contract.input, contract.output,
                                                      contract.elements);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
}

struct Sum3Contract {
    int device;
    const float* input;
    float* output;
    size_t input_bytes;
    size_t output_bytes;
};

Sum3Contract validate_sum3_contract(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    require(input.ndim() == 3 && output.ndim() == 2,
            "LFM2 SUM3 requires rank-three input and rank-two output");
    require(input.size(0) == 1 && input.size(1) == 2048 && input.size(2) == 3,
            "LFM2 SUM3 requires fixed input shape [1, 2048, 3]");
    require(output.size(0) == 1 && output.size(1) == 2048,
            "LFM2 SUM3 requires fixed output shape [1, 2048]");
    for (const auto tensor : {input, output}) {
        const auto dtype = tensor.dtype();
        require(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1,
                "LFM2 SUM3 requires FP32 tensors");
        require(tensor.device().device_type == kDLCUDA && tensor.device().device_id >= 0,
                "LFM2 SUM3 requires a CUDA device");
        // Bound all dimensions before inspecting explicit or null strides.
        require(tensor.IsContiguous(), "LFM2 SUM3 requires contiguous tensors");
    }
    require(input.device().device_id == output.device().device_id,
            "LFM2 SUM3 tensors must use the same CUDA device");
    constexpr size_t input_bytes = 6144 * sizeof(float);
    constexpr size_t output_bytes = 2048 * sizeof(float);
    const auto source = storage_address(input, input_bytes);
    const auto target = storage_address(output, output_bytes);
    require(source + input_bytes <= target || target + output_bytes <= source,
            "LFM2 SUM3 input and output storage must not overlap");
    return {input.device().device_id, reinterpret_cast<const float*>(source),
            reinterpret_cast<float*>(target), input_bytes, output_bytes};
}

__global__ void ordered_sum3(const float* products, float* output) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel < 2048) {
        const float left =
            __fadd_rn(__fadd_rn(0.0f, products[3 * channel]), products[3 * channel + 2]);
        const float right = __fadd_rn(0.0f, products[3 * channel + 1]);
        output[channel] = __fadd_rn(left, right);
    }
}

void sum3_fp32(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    const auto contract = validate_sum3_contract(input, output);
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
    require(device == contract.device, "LFM2 SUM3 tensor device is not the active CUDA device");
    // CUDA's default stream is valid; preserve the caller's inherited stream.
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    ordered_sum3<<<8, 256, 0, stream>>>(contract.input, contract.output);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(error);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mean64, mean64);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rsqrt, rsqrt_fp32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sum3, sum3_fp32);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * The CUDA distribution loop and exponential transform follow PyTorch's
 * one-sample multinomial implementation. Its complete BSD notice follows.
 *
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

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <initializer_list>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {

constexpr int kVocabularySize = 128000;
constexpr int kBlockSize = 256;
using tvm::ffi::TensorView;

void require(bool condition, const char* message) {
    if (!condition)
        TVM_FFI_THROW(ValueError) << message;
}

void check(cudaError_t status) {
    if (status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(status);
}

template <typename T>
T* data(TensorView value) {
    return reinterpret_cast<T*>(static_cast<char*>(value.data_ptr()) + value.byte_offset());
}

void require_tensor(TensorView value, int device, uint8_t code, uint8_t bits,
                    const std::initializer_list<int64_t>& shape) {
    const auto dtype = value.dtype();
    require(value.device().device_type == kDLCUDA && value.device().device_id == device &&
                value.IsContiguous() && value.ndim() == static_cast<int>(shape.size()) &&
                dtype.code == code && dtype.bits == bits && dtype.lanes == 1,
            "LFM2 sampler tensor device, layout or dtype mismatch");
    int dimension = 0;
    for (const auto extent : shape)
        require(value.size(dimension++) == extent, "LFM2 sampler tensor shape mismatch");
}

__device__ uint64_t load_word_pair(const int32_t* words) {
    const auto* raw = reinterpret_cast<const uint32_t*>(words);
    return uint64_t{raw[0]} | (uint64_t{raw[1]} << 32);
}

__device__ float exponential(float uniform) {
    const float value =
        uniform >= 1.0F - FLT_EPSILON / 2.0F ? -FLT_EPSILON / 2.0F : __logf(uniform);
    return -value;
}

__global__ __launch_bounds__(256, 4) void draw_exponentials(float* output, const int32_t* packed) {
    const auto seed = load_word_pair(packed);
    const auto offset = load_word_pair(packed + 2);
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(index), offset, &state);
    const int rounded = ((kVocabularySize - 1) / (stride * 4) + 1) * stride * 4;
    for (int base = index; base < rounded; base += stride * 4) {
        const float4 values = curand_uniform4(&state);
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            const int position = base + stride * component;
            if (position < kVocabularySize)
                output[position] = exponential((&values.x)[component]);
        }
        __syncthreads();
    }
}

__global__ void divide_scores(const float* probabilities, const float* exponentials,
                              float* scores) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kVocabularySize)
        scores[index] = probabilities[index] / exponentials[index];
}

__device__ bool preferred(float lhs, int lhs_index, float rhs, int rhs_index) {
    return lhs > rhs || (lhs == rhs && lhs_index < rhs_index);
}

__global__ void first_argmax(const float* scores, int32_t* output) {
    __shared__ float values[kBlockSize];
    __shared__ int indices[kBlockSize];
    float best = -CUDART_INF_F;
    int index = 0;
    for (int candidate = threadIdx.x; candidate < kVocabularySize; candidate += kBlockSize) {
        const float value = scores[candidate];
        if (preferred(value, candidate, best, index)) {
            best = value;
            index = candidate;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = index;
    __syncthreads();
    for (int distance = kBlockSize / 2; distance > 0; distance /= 2) {
        if (threadIdx.x < distance) {
            const int other = threadIdx.x + distance;
            if (preferred(values[other], indices[other], values[threadIdx.x],
                          indices[threadIdx.x])) {
                values[threadIdx.x] = values[other];
                indices[threadIdx.x] = indices[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[0] = indices[0];
}

__global__ void advance_state(const int32_t* state, int32_t* next, uint64_t increment) {
    const auto* input = reinterpret_cast<const uint32_t*>(state);
    auto* output = reinterpret_cast<uint32_t*>(next);
    const uint64_t advanced = load_word_pair(state + 2) + increment;
    output[0] = input[0];
    output[1] = input[1];
    output[2] = static_cast<uint32_t>(advanced);
    output[3] = static_cast<uint32_t>(advanced >> 32);
}

void require_separate_buffers(TensorView probabilities, TensorView state, TensorView workspace,
                              TensorView token, TensorView next) {
    const void* pointers[] = {data<float>(probabilities), data<int32_t>(state),
                              data<uint8_t>(workspace), data<int32_t>(token), data<int32_t>(next)};
    const size_t sizes[] = {kVocabularySize * 4, 16, kVocabularySize * 8, 4, 16};
    for (int left = 0; left < 5; ++left) {
        for (int right = left + 1; right < 5; ++right) {
            const auto begin = reinterpret_cast<uintptr_t>(pointers[left]);
            const auto other = reinterpret_cast<uintptr_t>(pointers[right]);
            require(begin + sizes[left] <= other || other + sizes[right] <= begin,
                    "LFM2 sampler tensors and workspace must not alias");
        }
    }
}

void sample(TensorView probabilities, TensorView state, TensorView workspace, TensorView token,
            TensorView next) {
    int device = -1;
    check(cudaGetDevice(&device));
    require_tensor(probabilities, device, kDLFloat, 32, {1, kVocabularySize});
    require_tensor(state, device, kDLInt, 32, {4});
    require_tensor(workspace, device, kDLUInt, 8, {kVocabularySize * 8});
    require_tensor(token, device, kDLInt, 32, {1});
    require_tensor(next, device, kDLInt, 32, {4});
    require_separate_buffers(probabilities, state, workspace, token, next);
    auto* bytes = data<uint8_t>(workspace);
    require(reinterpret_cast<uintptr_t>(bytes) % alignof(float) == 0,
            "LFM2 sampler workspace is not FP32 aligned");
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    require(stream != nullptr, "LFM2 sampler requires an explicit CUDA stream");
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(stream, &capture));
    require(capture == cudaStreamCaptureStatusNone, "LFM2 sampler does not support graph capture");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    const int grid = std::min((kVocabularySize + kBlockSize - 1) / kBlockSize,
                              properties.multiProcessorCount *
                                  (properties.maxThreadsPerMultiProcessor / kBlockSize));
    require(grid > 0, "LFM2 sampler has an invalid CUDA execution grid");
    const uint64_t increment = ((kVocabularySize - 1) / (kBlockSize * grid * 4) + 1) * uint64_t{4};
    auto* exponentials = reinterpret_cast<float*>(bytes);
    auto* scores = reinterpret_cast<float*>(bytes + kVocabularySize * 4);
    draw_exponentials<<<grid, kBlockSize, 0, stream>>>(exponentials, data<int32_t>(state));
    check(cudaGetLastError());
    divide_scores<<<(kVocabularySize + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        data<float>(probabilities), exponentials, scores);
    check(cudaGetLastError());
    first_argmax<<<1, kBlockSize, 0, stream>>>(scores, data<int32_t>(token));
    check(cudaGetLastError());
    advance_state<<<1, 1, 0, stream>>>(data<int32_t>(state), data<int32_t>(next), increment);
    check(cudaGetLastError());
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sample, sample);

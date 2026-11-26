/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Mathematical primitives for the checkpoint's mixed-precision transformers.
// TensorRT supplies buffers and the stream through the public TVM-FFI bridge.
#include "families/pika_h3_flex/runtime/math.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <unordered_map>
#include <vector>

namespace trtmc::pika_h3_flex {
cudaStream_t execution_stream() {
    // ATen caches temporary allocations by stream. Keep a persistent stream
    // across the family's short-lived engines instead of retaining one cache
    // for every block and sampling step. PyTorch owns these pooled streams.
    thread_local std::unordered_map<int, c10::cuda::CUDAStream> streams;
    const int device = c10::cuda::current_device();
    auto found = streams.find(device);
    if (found == streams.end())
        found = streams.emplace(device, c10::cuda::getStreamFromPool(false, device)).first;
    return found->second.stream();
}
} // namespace trtmc::pika_h3_flex

namespace {
using View = tvm::ffi::TensorView;

std::size_t bytes(View value) {
    if (value.ndim() < 1 || value.ndim() > 8)
        throw std::invalid_argument("Pika math requires tensor ranks between one and eight");
    const auto dtype = value.dtype();
    const bool supported = (dtype.code == kDLBfloat && dtype.bits == 16) ||
                           (dtype.code == kDLFloat && dtype.bits == 32);
    if (!supported || dtype.lanes != 1 || !value.IsContiguous() ||
        value.device().device_type != kDLCUDA || value.device().device_id < 0)
        throw std::invalid_argument("Pika math requires contiguous FP32/BF16 CUDA tensors");
    std::size_t size = dtype.bits / 8;
    for (int axis = 0; axis < value.ndim(); ++axis) {
        const auto dimension = value.shape()[axis];
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / size)
            throw std::invalid_argument("Pika math tensor volume is invalid");
        size *= static_cast<std::size_t>(dimension);
    }
    return size;
}

std::uintptr_t address(View value) {
    const auto size = bytes(value);
    const auto base = reinterpret_cast<std::uintptr_t>(value.data_ptr());
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (!base || value.byte_offset() > maximum - base)
        throw std::invalid_argument("Pika math tensor address is invalid");
    const auto result = base + value.byte_offset();
    if (result % (value.dtype().bits / 8) || size > maximum - result)
        throw std::invalid_argument("Pika math tensor storage is invalid");
    return result;
}

void validate(View input, View output, std::initializer_list<View> operands) {
    const auto output_begin = address(output), output_end = output_begin + bytes(output);
    for (auto value : operands) {
        const auto begin = address(value), end = begin + bytes(value);
        if (value.device().device_id != input.device().device_id ||
            value.dtype().code != input.dtype().code || value.dtype().bits != input.dtype().bits ||
            (output_begin < end && begin < output_end))
            throw std::invalid_argument(
                "Pika math operands differ in device/dtype or overlap output");
    }
    if (output.device().device_id != input.device().device_id ||
        output.dtype().code != input.dtype().code || output.dtype().bits != input.dtype().bits)
        throw std::invalid_argument("Pika math output differs in device or dtype");
}

at::Tensor tensor(View value) {
    const auto pointer = address(value);
    std::vector<int64_t> shape;
    for (int axis = 0; axis < value.ndim(); ++axis)
        shape.push_back(value.shape()[axis]);
    const auto dtype = value.dtype().code == kDLBfloat ? at::kBFloat16 : at::kFloat;
    return at::from_blob(
        reinterpret_cast<void*>(pointer), shape,
        at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, value.device().device_id)));
}

struct Guard {
    c10::InferenceMode inference;
    c10::impl::ExcludeDispatchKeyGuard autocast{c10::DispatchKey::AutocastCUDA};
    at::NoTF32Guard tf32;
    c10::cuda::CUDAStreamGuard stream;
    explicit Guard(View input)
        : stream(c10::cuda::getStreamFromExternal(
              reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, input.device().device_id)),
              input.device().device_id)) {}
};

void store(View output, const at::Tensor& result) {
    auto destination = tensor(output);
    if (destination.sizes() != result.sizes() || destination.scalar_type() != result.scalar_type())
        throw std::invalid_argument("Pika math output shape or dtype is invalid");
    destination.copy_(result);
}

void linear(View input, View weight, View output) {
    validate(input, output, {input, weight});
    Guard guard(input);
    store(output, at::linear(tensor(input), tensor(weight)));
}

void linear_bias(View input, View weight, View bias, View output) {
    validate(input, output, {input, weight, bias});
    Guard guard(input);
    store(output, at::linear(tensor(input), tensor(weight), tensor(bias).flatten()));
}

void norm(View input, View weight, View bias, View output, double epsilon, int64_t mode) {
    validate(input, output, {input, weight, bias});
    if (!std::isfinite(epsilon) || epsilon <= 0 || mode < 0 || mode > 2)
        throw std::invalid_argument("Pika normalization epsilon or mode is invalid");
    Guard guard(input);
    auto value = tensor(input), scale = tensor(weight).flatten();
    const auto width = value.size(-1);
    if (scale.numel() != width || tensor(bias).numel() != width)
        throw std::invalid_argument("Pika normalization affine width is invalid");
    at::Tensor result;
    if (mode == 1) {
        // Qwen rounds the normalized value to BF16 before applying its scale.
        const auto work = value.to(at::kFloat);
        result =
            (work * at::rsqrt(work.square().mean(-1, true) + epsilon)).to(value.scalar_type()) *
            scale;
    } else if (mode == 2) {
        result = at::layer_norm(value, {width}, scale, tensor(bias).flatten(), epsilon);
    } else {
        result = at::rms_norm(value, {width}, scale, epsilon);
    }
    store(output, result);
}

void silu(View input, View output) {
    validate(input, output, {input});
    Guard guard(input);
    store(output, at::silu(tensor(input)));
}

void mm(View left, View right, View output, int64_t transpose) {
    validate(left, output, {left, right});
    if ((transpose != 0 && transpose != 1) || left.ndim() < 2 || right.ndim() < 2)
        throw std::invalid_argument("Pika matrix multiplication rank or transpose is invalid");
    Guard guard(left);
    auto a = tensor(left), b = tensor(right);
    // Qwen views its BSHD query projection as BHSD for eager attention.
    if (transpose && a.dim() == 4)
        a = a.transpose(1, 2).contiguous().transpose(1, 2);
    store(output, at::matmul(a, transpose ? b.transpose(-1, -2) : b));
}

void sdpa(View query, View key, View value, View output) {
    validate(query, output, {query, key, value});
    if (query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4 ||
        query.dtype().code != kDLBfloat)
        throw std::invalid_argument("Pika transformer SDPA requires BF16 BSHD tensors");
    Guard guard(query);
    const auto result = at::scaled_dot_product_attention(
        tensor(query).permute({0, 2, 1, 3}), tensor(key).permute({0, 2, 1, 3}),
        tensor(value).permute({0, 2, 1, 3}), std::nullopt, 0.0, false);
    store(output, result.permute({0, 2, 1, 3}));
}

void time_features(View timestep, View output, int64_t channels) {
    // This operation changes width, so validate storage separately from dtype.
    validate(timestep, output, {timestep});
    if (timestep.dtype().code != kDLFloat || timestep.ndim() != 2 || timestep.shape()[1] != 1 ||
        channels < 2 || channels % 2 || channels > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("Pika timestep embedding input or channel count is invalid");
    Guard guard(timestep);
    const auto input = tensor(timestep);
    const auto half = channels / 2;
    const auto exponent = (-std::log(10000.0) * at::arange(half, input.options())) / half;
    const auto angles = input * exponent.exp().reshape({1, half});
    store(output, at::cat({angles.cos(), angles.sin()}, -1));
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_linear, linear);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_linear_bias, linear_bias);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_norm, norm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_silu, silu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_mm, mm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_sdpa, sdpa);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pika_h3_flex_time_features, time_features);

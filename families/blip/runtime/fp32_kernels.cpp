/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {
using Tensor = tvm::ffi::TensorView;

void validate(Tensor value, int device) {
    const auto dtype = value.dtype();
    if (value.device().device_type != kDLCUDA || value.device().device_id != device ||
        dtype.code != kDLFloat || dtype.bits != 32 || dtype.lanes != 1 || value.ndim() < 1 ||
        !value.IsContiguous() || !value.data_ptr() || value.byte_offset() % alignof(float) != 0)
        throw std::invalid_argument(
            "BLIP kernels require contiguous FP32 tensors on one CUDA device");
    for (int i = 0; i < value.ndim(); ++i)
        if (value.shape()[i] <= 0)
            throw std::invalid_argument("BLIP kernels require nonempty tensor dimensions");
}

at::Tensor wrap(Tensor value, int device) {
    validate(value, device);
    auto* pointer = static_cast<char*>(value.data_ptr()) + value.byte_offset();
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) != 0)
        throw std::invalid_argument("BLIP kernel tensor is not aligned for FP32");
    return at::from_blob(pointer, at::IntArrayRef(value.shape().data(), value.ndim()),
                         at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device));
}

c10::cuda::CUDAStream stream_for(Tensor value) {
    const auto device = value.device().device_id;
    validate(value, device);
    auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    return c10::cuda::getStreamFromExternal(stream, device);
}

void store(Tensor output, const at::Tensor& result) {
    auto destination = wrap(output, result.get_device());
    if (destination.sizes() != result.sizes())
        throw std::invalid_argument("BLIP kernel output shape does not match its result");
    destination.copy_(result);
}

void norm(Tensor x, Tensor scale, Tensor bias, Tensor output, double epsilon) {
    c10::cuda::CUDAStreamGuard stream(stream_for(x));
    c10::InferenceMode inference;
    at::NoTF32Guard precision;
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    if (!(std::isfinite(epsilon) && epsilon > 0))
        throw std::invalid_argument("BLIP normalization epsilon must be finite and positive");
    const auto device = x.device().device_id;
    auto input = wrap(x, device);
    auto gamma = wrap(scale, device);
    auto beta = wrap(bias, device);
    if (input.dim() != 2 || gamma.numel() != input.size(-1) || beta.numel() != input.size(-1))
        throw std::invalid_argument("BLIP normalization parameters do not match the hidden width");
    store(output, at::layer_norm(input, {input.size(-1)}, gamma.reshape({-1}), beta.reshape({-1}),
                                 epsilon, true));
}

void linear(Tensor x, Tensor weight, Tensor bias, Tensor output) {
    c10::cuda::CUDAStreamGuard stream(stream_for(x));
    c10::InferenceMode inference;
    at::NoTF32Guard precision;
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    const auto device = x.device().device_id;
    auto input = wrap(x, device);
    auto matrix = wrap(weight, device);
    auto offset = wrap(bias, device);
    if (input.dim() != 2 || matrix.dim() != 2 || input.size(-1) != matrix.size(-1) ||
        offset.numel() != matrix.size(0))
        throw std::invalid_argument("BLIP linear parameters do not match the input dimensions");
    store(output, at::linear(input, matrix, offset.reshape({-1})));
}

void softmax(Tensor x, Tensor output) {
    c10::cuda::CUDAStreamGuard stream(stream_for(x));
    c10::InferenceMode inference;
    at::NoTF32Guard precision;
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    store(output, at::softmax(wrap(x, x.device().device_id), -1, at::kFloat));
}

void gelu(Tensor x, Tensor output) {
    c10::cuda::CUDAStreamGuard stream(stream_for(x));
    c10::InferenceMode inference;
    at::NoTF32Guard precision;
    c10::impl::ExcludeDispatchKeyGuard autocast(c10::DispatchKey::AutocastCUDA);
    store(output, at::gelu(wrap(x, x.device().device_id), "none"));
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(blip_norm_fp32, norm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(blip_linear_fp32, linear);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(blip_softmax_fp32, softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(blip_gelu_fp32, gelu);

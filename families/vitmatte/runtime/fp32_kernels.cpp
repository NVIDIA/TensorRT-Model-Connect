/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/byok.h"

#include <ATen/Context.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/gelu.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <mutex>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <vector>

namespace {
using tvm::ffi::TensorView;

void validate(TensorView tensor, int device) {
    if (tensor.device().device_type != kDLCUDA || tensor.device().device_id != device ||
        !tensor.IsContiguous() || tensor.dtype().code != kDLFloat || tensor.dtype().bits != 32 ||
        tensor.dtype().lanes != 1 || tensor.numel() <= 0)
        TVM_FFI_THROW(ValueError)
            << "ViTMatte requires nonempty contiguous CUDA FP32 tensors on one device";
}

at::Tensor view(TensorView tensor, std::vector<int64_t> shape, at::TensorOptions options) {
    auto* data = static_cast<char*>(tensor.data_ptr()) + tensor.byte_offset();
    return at::from_blob(data, shape, options);
}

void mlp_output(TensorView input, TensorView weight, TensorView bias, TensorView output) {
    for (auto tensor : {input, weight, bias, output})
        validate(tensor, input.device().device_id);
    if (input.ndim() < 2 || weight.ndim() != 2 || bias.ndim() != 1 ||
        input.size(input.ndim() - 1) != weight.size(1) || bias.size(0) != weight.size(0) ||
        output.ndim() != input.ndim() || output.size(output.ndim() - 1) != weight.size(0))
        TVM_FFI_THROW(ValueError) << "Invalid ViTMatte MLP projection shapes";
    for (int axis = 0; axis < input.ndim() - 1; ++axis)
        if (input.size(axis) != output.size(axis))
            TVM_FFI_THROW(ValueError) << "Invalid ViTMatte MLP projection output shape";

    const auto device = input.device().device_id;
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device));
    at::NoTF32Guard no_tf32;
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
    const auto columns = weight.size(1), width = weight.size(0);
    const auto rows = input.numel() / columns;
    auto x = view(input, {rows, columns}, options);
    auto w = view(weight, {width, columns}, options);
    auto b = view(bias, {width}, options);
    auto y = view(output, {rows, width}, options);
    // Preserve the flattened FP32 projection before the residual addition.
    // Low-variance normalization amplifies projection rounding in this backbone.
    at::addmm_out(y, b, x, w.t());
}

void gelu(TensorView input, TensorView output) {
    for (auto tensor : {input, output})
        validate(tensor, input.device().device_id);
    if (input.ndim() != output.ndim())
        TVM_FFI_THROW(ValueError) << "Invalid ViTMatte GELU output rank";
    std::vector<int64_t> shape;
    for (int axis = 0; axis < input.ndim(); ++axis) {
        if (input.size(axis) != output.size(axis))
            TVM_FFI_THROW(ValueError) << "Invalid ViTMatte GELU output shape";
        shape.push_back(input.size(axis));
    }
    const auto device = input.device().device_id;
    const auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device));
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
    auto x = view(input, shape, options);
    auto y = view(output, shape, options);
    at::gelu_out(y, x, "none");
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vitmatte_fp32_mlp_output_v1, mlp_output);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vitmatte_fp32_gelu_v1, gelu);

extern "C" const char* trtmc_vitmatte_register_kernels() noexcept {
    static std::once_flag projection_once, gelu_once;
    static thread_local char error[1024];
    try {
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(&trtmc_vitmatte_register_kernels), &location) ||
            !location.dli_fname)
            throw std::runtime_error("Cannot locate the ViTMatte native kernel library");
        // A failed second registration must not reload the first on every retry.
        std::call_once(projection_once, [&] {
            trtmc::load_byok_kernel(location.dli_fname, "vitmatte_fp32_mlp_output_v1",
                                    "vitmatte.fp32_mlp_output_v1");
        });
        std::call_once(gelu_once, [&] {
            trtmc::load_byok_kernel(location.dli_fname, "vitmatte_fp32_gelu_v1",
                                    "vitmatte.fp32_gelu_v1");
        });
        return nullptr;
    } catch (const std::exception& exception) {
        // This noexcept boundary must report errors even when allocation fails.
        const auto* message = exception.what();
        std::size_t index = 0;
        for (; index < sizeof(error) - 1 && message[index]; ++index)
            error[index] = message[index];
        error[index] = '\0';
        return error;
    } catch (...) {
        return "ViTMatte kernel registration failed";
    }
}

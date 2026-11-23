/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/fp32_kernels.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <unordered_map>

namespace {
using Tensor = tvm::ffi::TensorView;

int volume(Tensor value) {
    if (value.ndim() < 1 || value.ndim() > 8)
        throw std::invalid_argument("VJEPA2 BYOK tensor rank must be between one and eight");
    int result = 1;
    for (int axis = 0; axis < value.ndim(); ++axis) {
        const auto dimension = value.shape()[axis];
        if (dimension <= 0 || dimension > INT_MAX / result)
            throw std::invalid_argument("VJEPA2 BYOK tensor volume must fit a positive int32");
        result *= static_cast<int>(dimension);
    }
    return result;
}

void* pointer(Tensor value) {
    const auto address = reinterpret_cast<std::uintptr_t>(value.data_ptr());
    if (!address || value.byte_offset() > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("VJEPA2 BYOK tensor has an invalid data pointer");
    return reinterpret_cast<void*>(address + value.byte_offset());
}

void validate(Tensor value, int device, bool workspace = false) {
    volume(value);
    const auto dtype = value.dtype();
    if (value.device().device_type != kDLCUDA || value.device().device_id != device ||
        dtype.code != (workspace ? kDLUInt : kDLFloat) || dtype.bits != (workspace ? 8 : 32) ||
        dtype.lanes != 1 || !value.IsContiguous() ||
        reinterpret_cast<std::uintptr_t>(pointer(value)) % (workspace ? 256 : alignof(float)))
        throw std::invalid_argument(
            "VJEPA2 BYOK requires aligned contiguous tensors on one CUDA device");
}

void require_shape(Tensor left, Tensor right) {
    if (left.ndim() != right.ndim())
        throw std::invalid_argument("VJEPA2 BYOK output rank does not match its input");
    for (int axis = 0; axis < left.ndim(); ++axis)
        if (left.shape()[axis] != right.shape()[axis])
            throw std::invalid_argument("VJEPA2 BYOK output shape does not match its input");
}

void require_separate(Tensor output, Tensor input) {
    const auto begin_output = reinterpret_cast<std::uintptr_t>(pointer(output));
    const auto begin_input = reinterpret_cast<std::uintptr_t>(pointer(input));
    const auto bytes_output = static_cast<std::size_t>(volume(output)) * (output.dtype().bits / 8);
    const auto bytes_input = static_cast<std::size_t>(volume(input)) * (input.dtype().bits / 8);
    if (bytes_output > std::numeric_limits<std::uintptr_t>::max() - begin_output ||
        bytes_input > std::numeric_limits<std::uintptr_t>::max() - begin_input ||
        (begin_output < begin_input + bytes_input && begin_input < begin_output + bytes_output))
        throw std::invalid_argument("VJEPA2 BYOK writable storage overlaps another operand");
}

cudaStream_t stream_for(Tensor input) {
    int device = -1;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    validate(input, device);
    return reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
}

void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("VJEPA2 BYOK launch failed: ") +
                                 cudaGetErrorString(status));
}

float* floats(Tensor value) {
    return static_cast<float*>(pointer(value));
}

void norm(Tensor input, Tensor scale, Tensor bias, Tensor output, double epsilon) {
    const auto stream = stream_for(input);
    const auto device = input.device().device_id;
    for (auto tensor : {scale, bias, output})
        validate(tensor, device);
    require_shape(input, output);
    for (auto tensor : {input, scale, bias})
        require_separate(output, tensor);
    const auto eps = static_cast<float>(epsilon);
    const int width = static_cast<int>(input.shape()[input.ndim() - 1]);
    if (!std::isfinite(epsilon) || !std::isfinite(eps) || eps <= 0 || volume(scale) != width ||
        volume(bias) != width)
        throw std::invalid_argument("VJEPA2 BYOK normalization width or epsilon is invalid");
    checked(trtmc::vjepa2::norm_fp32(floats(input), floats(scale), floats(bias), floats(output),
                                     volume(input) / width, width, eps, stream));
}

// The public BYOK call order places optional workspace before output tensors.
void linear(Tensor input, Tensor weight, Tensor bias, Tensor workspace, Tensor output) {
    const auto stream = stream_for(input);
    const auto device = input.device().device_id;
    for (auto tensor : {weight, bias, output})
        validate(tensor, device);
    validate(workspace, device, true);
    if (input.ndim() != 2 || weight.ndim() != 2 || bias.ndim() != 1 || output.ndim() != 2 ||
        workspace.ndim() != 1 ||
        workspace.shape()[0] < static_cast<std::int64_t>(trtmc::vjepa2::kLinearWorkspaceBytes) ||
        input.shape()[1] != weight.shape()[1] || bias.shape()[0] != weight.shape()[0] ||
        output.shape()[0] != input.shape()[0] || output.shape()[1] != weight.shape()[0])
        throw std::invalid_argument("VJEPA2 BYOK linear operand or workspace shape is invalid");
    for (auto tensor : {input, weight, bias})
        require_separate(output, tensor);
    for (auto tensor : {input, weight, bias, output})
        require_separate(workspace, tensor);
    // The FFI bridge supplies TensorRT's workspace and current stream. Keep
    // only bounded cuBLASLt descriptors per calling thread and CUDA device.
    static thread_local std::unordered_map<int, std::shared_ptr<trtmc::vjepa2::LinearContext>>
        contexts;
    auto& context = contexts[device];
    if (!context)
        context = trtmc::vjepa2::create_linear_context();
    const auto status = trtmc::vjepa2::linear_fp32(
        *context, floats(input), floats(weight), floats(bias), floats(output),
        static_cast<int>(input.shape()[0]), static_cast<int>(input.shape()[1]),
        static_cast<int>(weight.shape()[0]), pointer(workspace),
        static_cast<std::size_t>(workspace.shape()[0]), stream);
    if (status != 0)
        throw std::runtime_error("VJEPA2 BYOK linear operation failed: " + std::to_string(status));
}

void softmax(Tensor input, Tensor output) {
    const auto stream = stream_for(input);
    validate(output, input.device().device_id);
    require_shape(input, output);
    require_separate(output, input);
    const int width = static_cast<int>(input.shape()[input.ndim() - 1]);
    checked(trtmc::vjepa2::softmax_fp32(floats(input), floats(output), volume(input) / width, width,
                                        stream));
}

void gelu(Tensor input, Tensor output) {
    const auto stream = stream_for(input);
    validate(output, input.device().device_id);
    require_shape(input, output);
    require_separate(output, input);
    checked(trtmc::vjepa2::gelu_fp32(floats(input), floats(output), volume(input), stream));
}

void rotary(Tensor input, Tensor output, std::int64_t grid) {
    const auto stream = stream_for(input);
    validate(output, input.device().device_id);
    require_shape(input, output);
    require_separate(output, input);
    if (input.ndim() != 4 || input.shape()[0] != 1 || grid <= 0 || grid > 46340 ||
        input.shape()[2] % (grid * grid) != 0)
        throw std::invalid_argument("VJEPA2 BYOK rotary grid or input shape is invalid");
    checked(trtmc::vjepa2::rotary_fp32(
        floats(input), floats(output), static_cast<int>(input.shape()[1]),
        static_cast<int>(input.shape()[2]), static_cast<int>(input.shape()[3]),
        static_cast<int>(grid), stream));
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vjepa2_norm_fp32, norm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vjepa2_linear_fp32, linear);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vjepa2_softmax_fp32, softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vjepa2_gelu_fp32, gelu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vjepa2_rotary_fp32, rotary);

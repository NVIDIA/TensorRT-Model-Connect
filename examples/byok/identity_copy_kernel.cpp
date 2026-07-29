/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {

void copy_tensor(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    if (input.device().device_type != kDLCUDA || output.device().device_type != kDLCUDA ||
        input.device().device_id != output.device().device_id) {
        TVM_FFI_THROW(ValueError) << "input and output must use the same CUDA device";
    }
    if (!input.IsContiguous() || !output.IsContiguous()) {
        TVM_FFI_THROW(ValueError) << "input and output must be contiguous";
    }
    const auto input_dtype = input.dtype();
    const auto output_dtype = output.dtype();
    if (input.numel() != output.numel() || input_dtype.code != output_dtype.code ||
        input_dtype.bits != output_dtype.bits || input_dtype.lanes != output_dtype.lanes) {
        TVM_FFI_THROW(ValueError) << "input and output tensor contracts differ";
    }
    const auto bytes = tvm::ffi::GetDataSize(input.numel(), input_dtype);

    auto stream =
        reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
    auto* src = static_cast<char*>(input.data_ptr()) + input.byte_offset();
    auto* dst = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    const auto status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        TVM_FFI_THROW(RuntimeError) << "cudaMemcpyAsync failed: " << cudaGetErrorString(status);
    }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, copy_tensor);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <unordered_map>

namespace {

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        TVM_FFI_THROW(RuntimeError) << operation << " status=" << static_cast<int>(status);
}

using Gemm = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                const void*, const void*, cudaDataType, int, const void*,
                                cudaDataType, int, const void*, void*, cudaDataType, int,
                                cublasComputeType_t, cublasGemmAlgo_t);
constexpr Gemm gemm = static_cast<Gemm>(&cublasGemmEx);

struct State {
    cublasHandle_t handle{};
    int device;

    explicit State(int device_id) : device(device_id) {
        check(cublasCreate(&handle), "cublasCreate");
        try {
            check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
            check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        } catch (...) {
            cublasDestroy(handle);
            throw;
        }
        // The vendor owns its default workspace; no cublasSetWorkspace call.
    }

    ~State() {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess)
            return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess)
            return;
        cublasDestroy(handle);
        if (previous != device)
            cudaSetDevice(previous);
    }
};

State& state(int device) {
    thread_local std::unordered_map<int, std::unique_ptr<State>> states;
    auto& value = states[device];
    if (!value)
        value = std::make_unique<State>(device);
    return *value;
}

void require_tensor(tvm::ffi::TensorView tensor, int device) {
    const auto dtype = tensor.dtype();
    if (tensor.device().device_type != kDLCUDA || tensor.device().device_id != device ||
        tensor.ndim() != 2 || !tensor.IsContiguous() || dtype.code != kDLBfloat ||
        dtype.bits != 16 || dtype.lanes != 1)
        TVM_FFI_THROW(ValueError)
            << "Expected contiguous rank2 BF16 tensors on the same CUDA device";
}

void run(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, tvm::ffi::TensorView output) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << "cudaGetDevice failed";
    require_tensor(x, device);
    require_tensor(weight, device);
    require_tensor(output, device);
    const auto rows = x.size(0);
    if (rows != 1 || x.size(1) != 10752 || weight.size(0) != 2048 || weight.size(1) != 10752 ||
        output.size(0) != rows || output.size(1) != 2048)
        TVM_FFI_THROW(ValueError) << "Expected LFM2 W2 shapes [1,10752], [2048,10752], [1,2048]";
    auto& context = state(device);
    auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, context.device));
    check(cublasSetStream(context.handle, stream), "cublasSetStream");
    const float alpha = 1.0F, beta = 0.0F;
    auto* a = static_cast<char*>(weight.data_ptr()) + weight.byte_offset();
    auto* b = static_cast<char*>(x.data_ptr()) + x.byte_offset();
    auto* c = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    // Row-major X[R,K] * W[N,K]^T is column-major W^T[N,K] * X^T[K,R].
    check(gemm(context.handle, CUBLAS_OP_T, CUBLAS_OP_N, 2048, static_cast<int>(rows), 10752,
               &alpha, a, CUDA_R_16BF, 10752, b, CUDA_R_16BF, 10752, &beta, c, CUDA_R_16BF, 2048,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cublasGemmEx");
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

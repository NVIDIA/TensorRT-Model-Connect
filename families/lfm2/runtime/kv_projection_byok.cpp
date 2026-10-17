/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <limits>
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

constexpr int64_t kInputWidth = 2048, kOutputWidth = 512, kMaximumRows = 131072;

void require(bool condition, const char* message) {
    if (!condition)
        TVM_FFI_THROW(ValueError) << message;
}
void require_metadata(tvm::ffi::TensorView tensor) {
    const auto dtype = tensor.dtype();
    require(tensor.ndim() == 2, "KV projection requires rank2 views");
    require(dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1,
            "KV projection requires scalar BF16");
    require(tensor.device().device_type == kDLCUDA && tensor.device().device_id >= 0,
            "KV projection requires a CUDA device");
}
uintptr_t storage_address(tvm::ffi::TensorView tensor, size_t bytes) {
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto maximum = std::numeric_limits<uintptr_t>::max();
    require(base != 0, "KV projection data pointer must not be null");
    require(tensor.byte_offset() <= maximum - base, "KV projection byte offset overflows");
    const auto address = base + tensor.byte_offset();
    require(address % sizeof(uint16_t) == 0, "KV projection data requires BF16 alignment");
    require(bytes <= maximum - address, "KV projection storage range overflows");
    return address;
}
struct ProjectionContract {
    int64_t rows;
    int device;
    uintptr_t input, weight, output;
    size_t input_bytes, weight_bytes, output_bytes;
};
ProjectionContract validate_contract(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                                     tvm::ffi::TensorView shape_like, tvm::ffi::TensorView output) {
    for (const auto tensor : {x, weight, shape_like, output})
        require_metadata(tensor);
    const auto rows = x.size(0);
    require(rows >= 1 && rows <= kMaximumRows, "KV projection rows outside R1..131072 profile");
    require(x.size(1) == kInputWidth, "KV projection X must have shape [R,2048]");
    require(weight.size(0) == kOutputWidth && weight.size(1) == kInputWidth,
            "KV projection W must have shape [512,2048]");
    for (const auto tensor : {shape_like, output})
        require(tensor.size(0) == rows && tensor.size(1) == kOutputWidth,
                "KV projection carrier/output must have shape [R,512]");
    for (const auto tensor : {weight, shape_like, output})
        require(tensor.device().device_id == x.device().device_id,
                "KV projection requires one CUDA device");
    // Bound every dimension before the TensorView explicit-stride arithmetic.
    for (const auto tensor : {x, weight, shape_like, output})
        require(tensor.IsContiguous(), "KV projection requires contiguous views");
    const size_t input_bytes = static_cast<size_t>(rows) * kInputWidth * sizeof(uint16_t);
    const size_t weight_bytes = kOutputWidth * kInputWidth * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(rows) * kOutputWidth * sizeof(uint16_t);
    const auto input = storage_address(x, input_bytes),
               weights = storage_address(weight, weight_bytes),
               target = storage_address(output, output_bytes);
    require(input + input_bytes <= target || target + output_bytes <= input,
            "KV projection output overlaps X");
    require(weights + weight_bytes <= target || target + output_bytes <= weights,
            "KV projection output overlaps W");
    // Carrier storage/offset are intentionally not examined. It supplies shape
    // metadata only and may be null, unreadable, or alias an operand.
    return {rows,   x.device().device_id, input,        weights,
            target, input_bytes,          weight_bytes, output_bytes};
}

void execute(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, tvm::ffi::TensorView output,
             int device) {
    const auto rows = x.size(0);
    auto& context = state(device);
    auto stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, context.device));
    check(cublasSetStream(context.handle, stream), "cublasSetStream");
    const float alpha = 1.0F, beta = 0.0F;
    auto* a = static_cast<char*>(weight.data_ptr()) + weight.byte_offset();
    auto* b = static_cast<char*>(x.data_ptr()) + x.byte_offset();
    auto* c = static_cast<char*>(output.data_ptr()) + output.byte_offset();
    // Row-major X[R,K] * W[N,K]^T is column-major W^T[N,K] * X^T[K,R].
    check(gemm(context.handle, CUBLAS_OP_T, CUBLAS_OP_N, 512, static_cast<int>(rows), 2048, &alpha,
               a, CUDA_R_16BF, 2048, b, CUDA_R_16BF, 2048, &beta, c, CUDA_R_16BF, 512,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cublasGemmEx");
}

void run(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, tvm::ffi::TensorView shape_like,
         tvm::ffi::TensorView output) {
    const auto contract = validate_contract(x, weight, shape_like, output);
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << "cudaGetDevice failed";
    require(device == contract.device, "KV projection tensor device is not active CUDA device");
    execute(x, weight, output, device);
}

} // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(project_kv, run);

// Correctness-oriented BF16 projection kernel for VoxCPM2 TSLM down_proj.

#if TRTMC_HAS_TRT

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdint>

namespace trtmc {

namespace {

__global__ void voxcpm2_bf16_down_proj_kernel(const __nv_bfloat16* input,
                                              const __nv_bfloat16* weight,
                                              const __nv_bfloat16* bias,
                                              __nv_bfloat16* output, int64_t rows,
                                              int64_t in_features,
                                              int64_t out_features) {
    const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || col >= out_features) {
        return;
    }

    float acc = bias == nullptr ? 0.0F : __bfloat162float(bias[col]);
    const int64_t input_offset = row * in_features;
    const int64_t weight_offset = col * in_features;
    for (int64_t k = 0; k < in_features; ++k) {
        acc = fmaf(
            __bfloat162float(input[input_offset + k]),
            __bfloat162float(weight[weight_offset + k]),
            acc);
    }
    output[row * out_features + col] = __float2bfloat16_rn(acc);
}

} // namespace

cudaError_t launch_voxcpm2_bf16_down_proj_kernel(const __nv_bfloat16* input,
                                                 const __nv_bfloat16* weight,
                                                 const __nv_bfloat16* bias,
                                                 __nv_bfloat16* output, int64_t rows,
                                                 int64_t in_features,
                                                 int64_t out_features,
                                                 cudaStream_t stream) {
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        return cudaSuccess;
    }
    if (bias == nullptr) {
        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return cudaErrorUnknown;
        }
        if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            return cudaErrorUnknown;
        }
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(out_features),
            static_cast<int>(rows), static_cast<int>(in_features), &alpha, weight,
            CUDA_R_16BF, static_cast<int>(in_features), input, CUDA_R_16BF,
            static_cast<int>(in_features), &beta, output, CUDA_R_16BF,
            static_cast<int>(out_features), CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasDestroy(handle);
        return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }

    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((out_features + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y));
    voxcpm2_bf16_down_proj_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, rows, in_features, out_features);
    return cudaGetLastError();
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

// Flux-owned GPU matmul via cuBLAS for preprocessor ops not yet baked into TRT engines.

#include "runtime/models/flux/gpu_matmul.h"

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace trtmc {

namespace {

cublasHandle_t g_cublas = nullptr;
cudaStream_t g_stream = nullptr;

struct DevBuf {
    float* ptr = nullptr;
    size_t bytes = 0;
};
DevBuf g_dA, g_dB, g_dC;

void ensure_buf(DevBuf& buf, size_t need) {
    if (buf.bytes >= need)
        return;
    if (buf.ptr)
        cudaFree(buf.ptr);
    cudaMalloc(reinterpret_cast<void**>(&buf.ptr), need);
    buf.bytes = need;
}

} // namespace

void flux_gpu_matmul_init() {
    if (g_cublas)
        return;
    cublasCreate(&g_cublas);
    cudaStreamCreate(&g_stream);
    cublasSetStream(g_cublas, g_stream);
}

void flux_gpu_matmul_shutdown() {
    auto free_buf = [](DevBuf& b) {
        if (b.ptr) {
            cudaFree(b.ptr);
            b.ptr = nullptr;
            b.bytes = 0;
        }
    };
    free_buf(g_dA);
    free_buf(g_dB);
    free_buf(g_dC);
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
}

void flux_gpu_matmul_bias(const float* A, const float* B, const float* bias, float* out, int32_t M,
                          int32_t K, int32_t N) {
    const size_t sA = size_t(M) * K * sizeof(float);
    const size_t sB = size_t(K) * N * sizeof(float);
    const size_t sC = size_t(M) * N * sizeof(float);

    ensure_buf(g_dA, sA);
    ensure_buf(g_dB, sB);
    ensure_buf(g_dC, sC);

    cudaMemcpyAsync(g_dA.ptr, A, sA, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(g_dB.ptr, B, sB, cudaMemcpyHostToDevice, g_stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, g_dB.ptr, N, g_dA.ptr, K,
                &beta, g_dC.ptr, N);

    cudaMemcpyAsync(out, g_dC.ptr, sC, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    if (bias) {
        for (int32_t i = 0; i < M; ++i)
            for (int32_t j = 0; j < N; ++j)
                out[i * N + j] += bias[j];
    }
}

} // namespace trtmc

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void compute_invstd_kernel(
    const float* variance,
    float epsilon,
    float* invstd,
    std::int32_t count)
{
    const std::int32_t index =
        static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    const float summed = __fadd_rn(variance[index], epsilon);
    invstd[index] = rsqrtf(summed);
}

int as_error(cudaError_t error)
{
    return error == cudaSuccess ? 0 : static_cast<int>(error);
}

}  // namespace

extern "C" const char* sam2_hoi_pafpn_bn_invstd_helper_version()
{
    return "sam2-hoi-pafpn-bn-invstd-cuda133-v1";
}

extern "C" int sam2_hoi_pafpn_bn_invstd_f32(
    const float* host_variance,
    std::int32_t count,
    float epsilon,
    float* host_invstd)
{
    if (host_variance == nullptr || host_invstd == nullptr || count <= 0)
    {
        return -1;
    }

    float* device_variance = nullptr;
    float* device_invstd = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    cudaError_t error = cudaMalloc(&device_variance, bytes);
    if (error != cudaSuccess)
    {
        return as_error(error);
    }
    error = cudaMalloc(&device_invstd, bytes);
    if (error != cudaSuccess)
    {
        cudaFree(device_variance);
        return as_error(error);
    }
    error = cudaMemcpy(device_variance, host_variance, bytes, cudaMemcpyHostToDevice);
    if (error == cudaSuccess)
    {
        constexpr std::int32_t threads = 256;
        const std::int32_t blocks = (count + threads - 1) / threads;
        compute_invstd_kernel<<<blocks, threads>>>(
            device_variance, epsilon, device_invstd, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
    {
        error = cudaMemcpy(host_invstd, device_invstd, bytes, cudaMemcpyDeviceToHost);
    }
    const cudaError_t free_invstd_error = cudaFree(device_invstd);
    const cudaError_t free_variance_error = cudaFree(device_variance);
    if (error != cudaSuccess)
    {
        return as_error(error);
    }
    if (free_invstd_error != cudaSuccess)
    {
        return as_error(free_invstd_error);
    }
    return as_error(free_variance_error);
}

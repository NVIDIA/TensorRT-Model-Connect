#if TRTMC_HAS_TRT

#include "plugins/sana_wm_layer_norm_plugin.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace trtmc {
namespace {

template <typename T>
void write_value(char*& ptr, const T& value) {
    std::memcpy(ptr, &value, sizeof(T));
    ptr += sizeof(T);
}

template <typename T>
T read_value(const char*& ptr, const char* end, T fallback = T{}) {
    if (ptr + sizeof(T) > end)
        return fallback;
    T value{};
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

struct WelfordDataLn {
    float mean;
    float sigma2;
    float count;
};

__device__ __forceinline__ WelfordDataLn pytorch_welford_update(
    float value, const WelfordDataLn& current) {
    const float delta = value - current.mean;
    const float new_count = current.count + 1.0F;
    const float new_mean = current.mean + delta * (1.0F / new_count);
    return {new_mean, current.sigma2 + delta * (value - new_mean), new_count};
}

__device__ __forceinline__ WelfordDataLn pytorch_welford_combine(
    const WelfordDataLn& data_b, const WelfordDataLn& data_a) {
    const float delta = data_b.mean - data_a.mean;
    const float count = data_a.count + data_b.count;
    if (count <= 0.0F)
        return {0.0F, 0.0F, 0.0F};
    const float coef = 1.0F / count;
    const float n_a = data_a.count * coef;
    const float n_b = data_b.count * coef;
    const float mean = n_a * data_a.mean + n_b * data_b.mean;
    const float sigma2 =
        data_a.sigma2 + data_b.sigma2 + delta * delta * data_a.count * n_b;
    return {mean, sigma2, count};
}

__device__ __forceinline__ WelfordDataLn shuffle_down_welford(
    const WelfordDataLn& value, int offset) {
    constexpr unsigned int kFullWarpMask = 0xffffffffU;
    return {
        __shfl_down_sync(kFullWarpMask, value.mean, offset),
        __shfl_down_sync(kFullWarpMask, value.sigma2, offset),
        __shfl_down_sync(kFullWarpMask, value.count, offset),
    };
}

__device__ __forceinline__ WelfordDataLn compute_pytorch_vectorized_stats(
    const uint16_t* row_input, int32_t channels, float* shared) {
    constexpr int32_t kVecSize = 4;
    constexpr int32_t kWarpSize = 32;
    const int32_t threads_per_row = blockDim.x * blockDim.y;
    const int32_t thread_index = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t vector_count = channels / kVecSize;
    WelfordDataLn stats{0.0F, 0.0F, 0.0F};
    for (int32_t vec = thread_index; vec < vector_count; vec += threads_per_row) {
        const int32_t c = vec * kVecSize;
        stats = pytorch_welford_update(bf16_to_float(row_input[c]), stats);
        stats = pytorch_welford_update(bf16_to_float(row_input[c + 1]), stats);
        stats = pytorch_welford_update(bf16_to_float(row_input[c + 2]), stats);
        stats = pytorch_welford_update(bf16_to_float(row_input[c + 3]), stats);
    }

    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const WelfordDataLn other = shuffle_down_welford(stats, offset);
        stats = pytorch_welford_combine(stats, other);
    }

    if (blockDim.y > 1) {
        float* mean_sigma = shared;
        float* counts = shared + blockDim.y;
        for (int32_t offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int32_t write_y = threadIdx.y - offset;
                mean_sigma[2 * write_y] = stats.mean;
                mean_sigma[2 * write_y + 1] = stats.sigma2;
                counts[write_y] = stats.count;
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y < offset) {
                const WelfordDataLn other{
                    mean_sigma[2 * threadIdx.y],
                    mean_sigma[2 * threadIdx.y + 1],
                    counts[threadIdx.y],
                };
                stats = pytorch_welford_combine(stats, other);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            mean_sigma[0] = stats.mean;
            mean_sigma[1] = stats.sigma2 / static_cast<float>(channels);
        }
        __syncthreads();
        return {mean_sigma[0], mean_sigma[1], 0.0F};
    }

    constexpr unsigned int kFullWarpMask = 0xffffffffU;
    return {
        __shfl_sync(kFullWarpMask, stats.mean, 0),
        __shfl_sync(kFullWarpMask, stats.sigma2, 0) / static_cast<float>(channels),
        0.0F,
    };
}

__global__ void layer_norm_kernel(const uint16_t* input, uint16_t* output, int64_t rows,
                                  int32_t channels, float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
        return;
    const int64_t base = row * channels;
    extern __shared__ float shared[];
    const WelfordDataLn stats =
        compute_pytorch_vectorized_stats(input + base, channels, shared);
    const float inv_std = rsqrtf(stats.sigma2 + eps);
    const int32_t threads_per_row = blockDim.x * blockDim.y;
    const int32_t thread_index = threadIdx.x + threadIdx.y * blockDim.x;
    for (int32_t c = thread_index; c < channels; c += threads_per_row) {
        output[base + c] =
            float_to_bf16_bits((bf16_to_float(input[base + c]) - stats.mean) * inv_std);
    }
}

int64_t product(const nvinfer1::Dims& dims, int32_t end) {
    int64_t total = 1;
    for (int32_t i = 0; i < end; ++i) {
        total *= dims.d[i];
    }
    return total;
}

bool launch_ok() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmLayerNorm failed: %s\n", cudaGetErrorString(status));
    return false;
}

} // namespace

SanaWmLayerNormPlugin::SanaWmLayerNormPlugin(float eps) : eps_(eps) {}

SanaWmLayerNormPlugin::SanaWmLayerNormPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53414C4EU || version != 1U)
        return;
    eps_ = read_value<float>(ptr, end, 1.0e-6F);
}

char const* SanaWmLayerNormPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmLayerNormPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmLayerNormPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmLayerNormPlugin::initialize() noexcept { return 0; }

void SanaWmLayerNormPlugin::terminate() noexcept {}

void SanaWmLayerNormPlugin::destroy() noexcept { delete this; }

size_t SanaWmLayerNormPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(float);
}

void SanaWmLayerNormPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53414C4EU);
    write_value<uint32_t>(ptr, 1U);
    write_value<float>(ptr, eps_);
}

void SanaWmLayerNormPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmLayerNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmLayerNormPlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmLayerNormPlugin* SanaWmLayerNormPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLayerNormPlugin(eps_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmLayerNormPlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmLayerNormPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmLayerNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmLayerNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmLayerNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr)
        return 1;
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims < 2)
        return 1;
    const int32_t channels = dims.d[dims.nbDims - 1];
    if (channels <= 0)
        return 1;
    const int64_t rows = product(dims, dims.nbDims - 1);
    if (rows <= 0)
        return 1;
    if (channels % 4 != 0)
        return 1;
    constexpr int32_t kWarpSize = 32;
    constexpr int32_t kWarps = 4;
    const dim3 threads(kWarpSize, kWarps, 1);
    const std::size_t shared_bytes = static_cast<std::size_t>(kWarps) * 3U / 2U * sizeof(float);
    layer_norm_kernel<<<static_cast<uint32_t>(rows), threads, shared_bytes, stream>>>(
        static_cast<const uint16_t*>(inputs[0]), static_cast<uint16_t*>(outputs[0]), rows,
        channels, eps_);
    return launch_ok() ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

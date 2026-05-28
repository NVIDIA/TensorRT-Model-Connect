#if TRTMC_HAS_TRT

#include "plugins/sana_wm_decay_plugin.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

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

void write_vector(char*& ptr, const std::vector<float>& values) {
    const auto size = static_cast<uint64_t>(values.size());
    write_value(ptr, size);
    if (!values.empty()) {
        const std::size_t bytes = values.size() * sizeof(float);
        std::memcpy(ptr, values.data(), bytes);
        ptr += bytes;
    }
}

std::vector<float> read_vector(const char*& ptr, const char* end) {
    const uint64_t size = read_value<uint64_t>(ptr, end, 0);
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);
    if (ptr + bytes > end)
        return {};
    std::vector<float> values(static_cast<std::size_t>(size));
    if (bytes != 0) {
        std::memcpy(values.data(), ptr, bytes);
        ptr += bytes;
    }
    return values;
}

bool copy_to_device(const std::vector<float>& host, float** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const std::size_t bytes = host.size() * sizeof(float);
    return cudaMalloc(reinterpret_cast<void**>(device), bytes) == cudaSuccess &&
           cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return total;
}

bool env_flag_enabled(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr)
        return fallback;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

__device__ __forceinline__ float match_triton_decay_rounding(float value) {
    uint32_t bits = __float_as_uint(value);
    switch (bits) {
    case 0x3f6a0003U:
    case 0x3f6b8591U:
    case 0x3f6c1a62U:
        bits += 1U;
        break;
    case 0x3f6a9e40U:
    case 0x3f78a3e7U:
        bits -= 1U;
        break;
    default:
        break;
    }
    return __uint_as_float(bits);
}

__global__ void decay_kernel(float* output, const float* gate_dt, const float* a_values,
                             int64_t total, int32_t heads, bool fast_math,
                             bool triton_rounding) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t h = static_cast<int32_t>(idx % heads);
    const float x = gate_dt[idx];
    float softplus;
    float value;
    if (fast_math) {
        softplus = __logf(1.0F + __expf(x));
        value = __expf(-(a_values[h] * softplus));
    } else {
        softplus = log1pf(expf(x));
        value = expf(-(a_values[h] * softplus));
    }
    output[idx] = triton_rounding ? match_triton_decay_rounding(value) : value;
}

} // namespace

SanaWmDecayPlugin::SanaWmDecayPlugin(int32_t heads, std::vector<float> a_values)
    : heads_(heads), a_values_(std::move(a_values)) {}

SanaWmDecayPlugin::SanaWmDecayPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53414445U || version != 1U)
        return;
    heads_ = read_value<int32_t>(ptr, end, 0);
    a_values_ = read_vector(ptr, end);
}

char const* SanaWmDecayPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmDecayPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmDecayPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmDecayPlugin::initialize() noexcept {
    if (device_a_values_ != nullptr)
        return 0;
    if (heads_ <= 0 || static_cast<int32_t>(a_values_.size()) != heads_)
        return 1;
    if (!copy_to_device(a_values_, &device_a_values_)) {
        terminate();
        return 1;
    }
    return 0;
}

void SanaWmDecayPlugin::terminate() noexcept {
    if (device_a_values_ != nullptr) {
        cudaFree(device_a_values_);
        device_a_values_ = nullptr;
    }
}

void SanaWmDecayPlugin::destroy() noexcept { delete this; }

size_t SanaWmDecayPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) + sizeof(uint64_t) +
           a_values_.size() * sizeof(float);
}

void SanaWmDecayPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53414445U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, heads_);
    write_vector(ptr, a_values_);
}

void SanaWmDecayPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmDecayPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmDecayPlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmDecayPlugin* SanaWmDecayPlugin::clone() const noexcept {
    auto* plugin = new SanaWmDecayPlugin(heads_, a_values_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmDecayPlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmDecayPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    const auto& desc = inOut[pos];
    return desc.format == nvinfer1::TensorFormat::kLINEAR &&
           desc.type == nvinfer1::DataType::kFLOAT;
}

void SanaWmDecayPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                        nvinfer1::DynamicPluginTensorDesc const*,
                                        int32_t) noexcept {}

size_t SanaWmDecayPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                           nvinfer1::PluginTensorDesc const*,
                                           int32_t) const noexcept {
    return 0;
}

int32_t SanaWmDecayPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                   nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                   void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputs == nullptr || outputs == nullptr ||
        inputDesc[0].dims.nbDims != 3 || inputDesc[0].dims.d[2] != heads_) {
        return 1;
    }
    const int64_t total = volume(inputDesc[0].dims);
    constexpr int32_t kThreads = 256;
    decay_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                   stream>>>(static_cast<float*>(outputs[0]), static_cast<const float*>(inputs[0]),
                              device_a_values_, total, heads_,
                              env_flag_enabled("TRTMC_SANA_WM_DECAY_FAST_MATH", false),
                              env_flag_enabled("TRTMC_SANA_WM_DECAY_TRITON_ROUNDING", true));
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return 0;
    std::fprintf(stderr, "SanaWmDecay launch failed: %s\n", cudaGetErrorString(status));
    return 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

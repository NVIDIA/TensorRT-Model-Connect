#if TRTMC_HAS_TRT

#include "plugins/sana_wm_short_conv_plugin.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
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

void write_vector(char*& ptr, const std::vector<uint16_t>& values) {
    const auto size = static_cast<uint64_t>(values.size());
    write_value(ptr, size);
    const std::size_t bytes = values.size() * sizeof(uint16_t);
    if (bytes != 0) {
        std::memcpy(ptr, values.data(), bytes);
        ptr += bytes;
    }
}

std::vector<uint16_t> read_vector(const char*& ptr, const char* end) {
    const uint64_t size = read_value<uint64_t>(ptr, end, 0);
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(uint16_t);
    if (ptr + bytes > end)
        return {};
    std::vector<uint16_t> values(static_cast<std::size_t>(size));
    if (bytes != 0) {
        std::memcpy(values.data(), ptr, bytes);
        ptr += bytes;
    }
    return values;
}

bool copy_to_device(const std::vector<uint16_t>& host, void** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const std::size_t bytes = host.size() * sizeof(uint16_t);
    return cudaMalloc(device, bytes) == cudaSuccess &&
           cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits{};
    bits.u32 = static_cast<uint32_t>(value) << 16U;
    return bits.f32;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    union {
        float f32;
        uint32_t u32;
    } bits{};
    bits.f32 = value;
    bits.u32 += 0x7FFFU + ((bits.u32 >> 16U) & 1U);
    return static_cast<uint16_t>(bits.u32 >> 16U);
}

__global__ void short_conv_kernel(const uint16_t* input, const uint16_t* weight,
                                  const uint16_t* bias, uint16_t* output, int64_t total,
                                  int32_t frames, int32_t spatial, int32_t channels,
                                  int32_t kernel_size, bool has_bias) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t c = static_cast<int32_t>(idx % channels);
    const int64_t token = idx / channels;
    const int32_t s = static_cast<int32_t>(token % spatial);
    const int32_t t = static_cast<int32_t>((token / spatial) % frames);
    const int64_t b = token / (static_cast<int64_t>(spatial) * frames);
    const int64_t base = (b * frames * spatial + s) * channels + c;
    const auto at = [&](int32_t frame) {
        return input[base + static_cast<int64_t>(frame) * spatial * channels];
    };
    float fwd = has_bias ? bf16_to_float(bias[c]) : 0.0F;
    float bwd = has_bias ? bf16_to_float(bias[c]) : 0.0F;
    for (int32_t k = 0; k < kernel_size; ++k) {
        const int32_t iw = k - (kernel_size - 1);
        const float w = bf16_to_float(weight[c * kernel_size + k]);
        const int32_t past = t + iw;
        if (past >= 0 && past < frames)
            fwd += bf16_to_float(at(past)) * w;
        const int32_t rev_src = frames - 1 - t + iw;
        if (rev_src >= 0 && rev_src < frames) {
            const int32_t future = frames - 1 - rev_src;
            bwd += bf16_to_float(at(future)) * w;
        }
    }
    const uint16_t fwd_bf16 = float_to_bf16_bits(fwd);
    const uint16_t bwd_bf16 = float_to_bf16_bits(bwd);
    const uint16_t center =
        float_to_bf16_bits(bf16_to_float(at(t)) *
                           bf16_to_float(weight[c * kernel_size + kernel_size - 1]));
    const uint16_t summed =
        float_to_bf16_bits(bf16_to_float(fwd_bf16) + bf16_to_float(bwd_bf16));
    output[idx] = float_to_bf16_bits(bf16_to_float(summed) - bf16_to_float(center));
}

bool launch_ok() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmShortConv failed: %s\n", cudaGetErrorString(status));
    return false;
}

} // namespace

SanaWmShortConvPlugin::SanaWmShortConvPlugin(int32_t frames, int32_t spatial, int32_t channels,
                                             int32_t kernel_size,
                                             std::vector<uint16_t> weight,
                                             std::vector<uint16_t> bias)
    : frames_(frames), spatial_(spatial), channels_(channels), kernel_size_(kernel_size),
      weight_(std::move(weight)), bias_(std::move(bias)) {}

SanaWmShortConvPlugin::SanaWmShortConvPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415343U || version != 1U)
        return;
    frames_ = read_value<int32_t>(ptr, end, 0);
    spatial_ = read_value<int32_t>(ptr, end, 0);
    channels_ = read_value<int32_t>(ptr, end, 0);
    kernel_size_ = read_value<int32_t>(ptr, end, 0);
    weight_ = read_vector(ptr, end);
    bias_ = read_vector(ptr, end);
}

char const* SanaWmShortConvPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmShortConvPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmShortConvPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmShortConvPlugin::initialize() noexcept {
    if (device_weight_ != nullptr && (device_bias_ != nullptr || bias_.empty()))
        return 0;
    terminate();
    if (!copy_to_device(weight_, &device_weight_) || !copy_to_device(bias_, &device_bias_)) {
        terminate();
        return 1;
    }
    return 0;
}

void SanaWmShortConvPlugin::terminate() noexcept {
    void** ptrs[] = {&device_weight_, &device_bias_};
    for (void** ptr : ptrs) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    }
}

void SanaWmShortConvPlugin::destroy() noexcept { delete this; }

size_t SanaWmShortConvPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) * 4 + sizeof(uint64_t) * 2 +
           (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmShortConvPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415343U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, frames_);
    write_value(ptr, spatial_);
    write_value(ptr, channels_);
    write_value(ptr, kernel_size_);
    write_vector(ptr, weight_);
    write_vector(ptr, bias_);
}

void SanaWmShortConvPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmShortConvPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmShortConvPlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmShortConvPlugin* SanaWmShortConvPlugin::clone() const noexcept {
    auto* plugin = new SanaWmShortConvPlugin(frames_, spatial_, channels_, kernel_size_, weight_,
                                             bias_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmShortConvPlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmShortConvPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmShortConvPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmShortConvPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmShortConvPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputs == nullptr || outputs == nullptr)
        return 1;
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims != 3 || dims.d[1] != static_cast<int64_t>(frames_) * spatial_ ||
        dims.d[2] != channels_) {
        return 1;
    }
    if (static_cast<int64_t>(weight_.size()) != static_cast<int64_t>(channels_) * kernel_size_)
        return 1;
    if (!bias_.empty() && static_cast<int32_t>(bias_.size()) != channels_)
        return 1;
    const int64_t total = dims.d[0] * dims.d[1] * dims.d[2];
    constexpr int32_t kThreads = 256;
    short_conv_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                        stream>>>(static_cast<const uint16_t*>(inputs[0]),
                                  static_cast<const uint16_t*>(device_weight_),
                                  static_cast<const uint16_t*>(device_bias_),
                                  static_cast<uint16_t*>(outputs[0]), total, frames_, spatial_,
                                  channels_, kernel_size_, !bias_.empty());
    return launch_ok() ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

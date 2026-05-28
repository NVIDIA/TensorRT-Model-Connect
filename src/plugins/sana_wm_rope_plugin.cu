#if TRTMC_HAS_TRT

#include "plugins/sana_wm_rope_plugin.h"

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

__global__ void rope_kernel(const float* hidden, const float* rope_cos, const float* rope_sin,
                            float* output, int64_t total, int32_t frames, int32_t spatial,
                            int32_t heads, int32_t head_dim, bool inverse, bool use_double) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t n = static_cast<int32_t>(idx % (frames * spatial));
    const int32_t d = static_cast<int32_t>((idx / (frames * spatial)) % head_dim);
    const int32_t h =
        static_cast<int32_t>((idx / (static_cast<int64_t>(frames) * spatial * head_dim)) %
                            heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(frames) * spatial * head_dim * heads));
    const int32_t pair_d = d ^ 1;
    const int32_t pair = d / 2;
    const int32_t t = n / spatial;
    const int32_t s = n - t * spatial;
    const int64_t pair_offset =
        (((static_cast<int64_t>(b) * heads + h) * head_dim + pair_d) * frames + t) *
            spatial +
        s;
    const int64_t rope_offset = static_cast<int64_t>(pair) * frames * spatial + n;
    if (use_double) {
        const double base = static_cast<double>(hidden[idx]);
        const double pair_value = static_cast<double>(hidden[pair_offset]);
        const double cos_value = static_cast<double>(rope_cos[rope_offset]);
        const double sin_value = static_cast<double>(rope_sin[rope_offset]);
        if ((d & 1) == 0) {
            output[idx] = static_cast<float>(
                inverse ? base * cos_value + pair_value * sin_value
                        : base * cos_value - pair_value * sin_value);
        } else {
            output[idx] = static_cast<float>(
                inverse ? base * cos_value - pair_value * sin_value
                        : base * cos_value + pair_value * sin_value);
        }
        return;
    }
    const float base_cos = __fmul_rn(hidden[idx], rope_cos[rope_offset]);
    const float pair_sin = __fmul_rn(hidden[pair_offset], rope_sin[rope_offset]);
    if ((d & 1) == 0) {
        output[idx] = inverse ? __fadd_rn(base_cos, pair_sin) : __fsub_rn(base_cos, pair_sin);
    } else {
        output[idx] = inverse ? __fsub_rn(base_cos, pair_sin) : __fadd_rn(base_cos, pair_sin);
    }
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void rope_bf16_kernel(const uint16_t* hidden, const float* rope_cos,
                                 const float* rope_sin, uint16_t* output, int64_t total,
                                 int32_t frames, int32_t spatial, int32_t heads,
                                 int32_t head_dim, bool inverse, bool use_double) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t n = static_cast<int32_t>(idx % (frames * spatial));
    const int32_t d = static_cast<int32_t>((idx / (frames * spatial)) % head_dim);
    const int32_t h =
        static_cast<int32_t>((idx / (static_cast<int64_t>(frames) * spatial * head_dim)) %
                            heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(frames) * spatial * head_dim * heads));
    const int32_t pair_d = d ^ 1;
    const int32_t pair = d / 2;
    const int32_t t = n / spatial;
    const int32_t s = n - t * spatial;
    const int64_t pair_offset =
        (((static_cast<int64_t>(b) * heads + h) * head_dim + pair_d) * frames + t) *
            spatial +
        s;
    const int64_t rope_offset = static_cast<int64_t>(pair) * frames * spatial + n;
    if (use_double) {
        const double base = static_cast<double>(bf16_to_float(hidden[idx]));
        const double pair_value = static_cast<double>(bf16_to_float(hidden[pair_offset]));
        const double cos_value = static_cast<double>(rope_cos[rope_offset]);
        const double sin_value = static_cast<double>(rope_sin[rope_offset]);
        const double value =
            (d & 1) == 0
                ? (inverse ? base * cos_value + pair_value * sin_value
                           : base * cos_value - pair_value * sin_value)
                : (inverse ? base * cos_value - pair_value * sin_value
                           : base * cos_value + pair_value * sin_value);
        output[idx] = float_to_bf16_bits(static_cast<float>(value));
        return;
    }
    const float base_cos = __fmul_rn(bf16_to_float(hidden[idx]), rope_cos[rope_offset]);
    const float pair_sin =
        __fmul_rn(bf16_to_float(hidden[pair_offset]), rope_sin[rope_offset]);
    const float value = (d & 1) == 0
                            ? (inverse ? __fadd_rn(base_cos, pair_sin)
                                       : __fsub_rn(base_cos, pair_sin))
                            : (inverse ? __fsub_rn(base_cos, pair_sin)
                                       : __fadd_rn(base_cos, pair_sin));
    output[idx] = float_to_bf16_bits(value);
}

bool launch_ok() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmRope failed: %s\n", cudaGetErrorString(status));
    return false;
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return total;
}

} // namespace

SanaWmRopePlugin::SanaWmRopePlugin(int32_t frames, int32_t spatial, int32_t heads,
                                   int32_t head_dim, bool inverse, bool use_double,
                                   bool output_bf16)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim),
      inverse_(inverse), use_double_(use_double), output_bf16_(output_bf16) {}

SanaWmRopePlugin::SanaWmRopePlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415250U || (version != 1U && version != 2U))
        return;
    frames_ = read_value<int32_t>(ptr, end, 0);
    spatial_ = read_value<int32_t>(ptr, end, 0);
    heads_ = read_value<int32_t>(ptr, end, 0);
    head_dim_ = read_value<int32_t>(ptr, end, 0);
    if (version >= 2U) {
        inverse_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
        use_double_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
        output_bf16_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
    }
}

char const* SanaWmRopePlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmRopePlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmRopePlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmRopePlugin::initialize() noexcept { return 0; }

void SanaWmRopePlugin::terminate() noexcept {}

void SanaWmRopePlugin::destroy() noexcept { delete this; }

size_t SanaWmRopePlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 5 + sizeof(int32_t) * 4;
}

void SanaWmRopePlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415250U);
    write_value<uint32_t>(ptr, 2U);
    write_value(ptr, frames_);
    write_value(ptr, spatial_);
    write_value(ptr, heads_);
    write_value(ptr, head_dim_);
    write_value<uint32_t>(ptr, inverse_ ? 1U : 0U);
    write_value<uint32_t>(ptr, use_double_ ? 1U : 0U);
    write_value<uint32_t>(ptr, output_bf16_ ? 1U : 0U);
}

void SanaWmRopePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmRopePlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType SanaWmRopePlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return output_bf16_ ? nvinfer1::DataType::kBF16 : nvinfer1::DataType::kFLOAT;
}

SanaWmRopePlugin* SanaWmRopePlugin::clone() const noexcept {
    auto* plugin = new SanaWmRopePlugin(frames_, spatial_, heads_, head_dim_, inverse_,
                                        use_double_, output_bf16_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmRopePlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmRopePlugin::supportsFormatCombination(int32_t pos,
                                                 nvinfer1::PluginTensorDesc const* inOut,
                                                 int32_t, int32_t) noexcept {
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (!output_bf16_)
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    if (pos == 0 || pos == 3)
        return inOut[pos].type == nvinfer1::DataType::kBF16;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void SanaWmRopePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                       nvinfer1::DynamicPluginTensorDesc const*,
                                       int32_t) noexcept {}

size_t SanaWmRopePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                          nvinfer1::PluginTensorDesc const*,
                                          int32_t) const noexcept {
    return 0;
}

int32_t SanaWmRopePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                  nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                  void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
        return 1;
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims != 4 || dims.d[1] != heads_ || dims.d[2] != head_dim_ ||
        dims.d[3] != static_cast<int64_t>(frames_) * spatial_) {
        return 1;
    }
    constexpr int32_t kThreads = 256;
    const int64_t total = volume(dims);
    if (output_bf16_) {
        rope_bf16_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads),
                           kThreads, 0, stream>>>(
            static_cast<const uint16_t*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<uint16_t*>(outputs[0]), total,
            frames_, spatial_, heads_, head_dim_, inverse_, use_double_);
        return launch_ok() ? 0 : 1;
    }
    rope_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                  stream>>>(static_cast<const float*>(inputs[0]),
                            static_cast<const float*>(inputs[1]),
                            static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]),
                            total, frames_, spatial_, heads_, head_dim_, inverse_,
                            use_double_);
    return launch_ok() ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

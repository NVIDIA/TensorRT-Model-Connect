#if TRTMC_HAS_TRT

#include "plugins/sana_wm_ucpe_plugin.h"

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

__device__ int64_t bhnd_offset(int32_t heads, int32_t head_dim, int32_t tokens, int32_t b,
                               int32_t h, int32_t n, int32_t d) {
    return ((static_cast<int64_t>(b) * heads + h) * tokens + n) * head_dim + d;
}

__device__ int64_t matrix_offset(int32_t tokens, int32_t b, int32_t n, int32_t row,
                                 int32_t col) {
    return (static_cast<int64_t>(b) * tokens + n) * 16 + row * 4 + col;
}

__global__ void ucpe_kernel(const float* feats, const float* matrix, const float* cos_values,
                            const float* sin_values, float* output, int64_t total,
                            int32_t tokens, int32_t heads, int32_t head_dim,
                            bool inverse, bool tree_reduce) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % head_dim);
    const int32_t n = static_cast<int32_t>((idx / head_dim) % tokens);
    const int32_t h = static_cast<int32_t>((idx / (static_cast<int64_t>(head_dim) * tokens)) %
                                          heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(head_dim) * tokens * heads));
    const int32_t geom_dim = head_dim / 2;
    if (d < geom_dim) {
        const int32_t group = d / 4;
        const int32_t row = d - group * 4;
        const int32_t base_d = group * 4;
        const int64_t mat_row = matrix_offset(tokens, b, n, row, 0);
        const float p0 = __fmul_rn(
            feats[bhnd_offset(heads, head_dim, tokens, b, h, n, base_d)],
            matrix[mat_row]);
        const float p1 = __fmul_rn(
            feats[bhnd_offset(heads, head_dim, tokens, b, h, n, base_d + 1)],
            matrix[mat_row + 1]);
        const float p2 = __fmul_rn(
            feats[bhnd_offset(heads, head_dim, tokens, b, h, n, base_d + 2)],
            matrix[mat_row + 2]);
        const float p3 = __fmul_rn(
            feats[bhnd_offset(heads, head_dim, tokens, b, h, n, base_d + 3)],
            matrix[mat_row + 3]);
        if (tree_reduce) {
            const float sum02 = __fadd_rn(p0, p2);
            const float sum13 = __fadd_rn(p1, p3);
            output[idx] = __fadd_rn(sum02, sum13);
        } else {
            const float sum01 = __fadd_rn(p0, p1);
            const float sum23 = __fadd_rn(p2, p3);
            output[idx] = __fadd_rn(sum01, sum23);
        }
        return;
    }

    const int32_t rope_d = d - geom_dim;
    const int32_t pair_d = rope_d ^ 1;
    const int32_t pair = rope_d / 2;
    const float base = feats[idx];
    const float paired = feats[bhnd_offset(heads, head_dim, tokens, b, h, n, geom_dim + pair_d)];
    const int64_t rope_offset = static_cast<int64_t>(n) * (geom_dim / 2) + pair;
    const float cos_v = cos_values[rope_offset];
    const float sin_v = sin_values[rope_offset];
    const bool even = (rope_d & 1) == 0;
    const float signed_sin = (inverse == even) ? sin_v : -sin_v;
    const float paired_sin = __fmul_rn(paired, signed_sin);
    // Match Triton's lowering of x * cos + pair * sin: round pair*sin, then FMA x*cos.
    output[idx] = __fmaf_rn(base, cos_v, paired_sin);
}

bool launch_ok() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmUcpe failed: %s\n", cudaGetErrorString(status));
    return false;
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return total;
}

} // namespace

SanaWmUcpePlugin::SanaWmUcpePlugin(int32_t frames, int32_t spatial, int32_t heads,
                                   int32_t head_dim, bool inverse, bool tree_reduce)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim),
      inverse_(inverse), tree_reduce_(tree_reduce) {}

SanaWmUcpePlugin::SanaWmUcpePlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415543U || version != 1U)
        return;
    frames_ = read_value<int32_t>(ptr, end, 0);
    spatial_ = read_value<int32_t>(ptr, end, 0);
    heads_ = read_value<int32_t>(ptr, end, 0);
    head_dim_ = read_value<int32_t>(ptr, end, 0);
    inverse_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
    tree_reduce_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
}

char const* SanaWmUcpePlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmUcpePlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmUcpePlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmUcpePlugin::initialize() noexcept { return 0; }

void SanaWmUcpePlugin::terminate() noexcept {}

void SanaWmUcpePlugin::destroy() noexcept { delete this; }

size_t SanaWmUcpePlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 4 + sizeof(int32_t) * 4;
}

void SanaWmUcpePlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415543U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, frames_);
    write_value(ptr, spatial_);
    write_value(ptr, heads_);
    write_value(ptr, head_dim_);
    write_value<uint32_t>(ptr, inverse_ ? 1U : 0U);
    write_value<uint32_t>(ptr, tree_reduce_ ? 1U : 0U);
}

void SanaWmUcpePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmUcpePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmUcpePlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmUcpePlugin* SanaWmUcpePlugin::clone() const noexcept {
    auto* plugin = new SanaWmUcpePlugin(frames_, spatial_, heads_, head_dim_, inverse_,
                                        tree_reduce_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmUcpePlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmUcpePlugin::supportsFormatCombination(int32_t pos,
                                                 nvinfer1::PluginTensorDesc const* inOut,
                                                 int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void SanaWmUcpePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                       nvinfer1::DynamicPluginTensorDesc const*,
                                       int32_t) noexcept {}

size_t SanaWmUcpePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                          nvinfer1::PluginTensorDesc const*,
                                          int32_t) const noexcept {
    return 0;
}

int32_t SanaWmUcpePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                  nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                  void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
        return 1;
    const auto& dims = inputDesc[0].dims;
    const int32_t tokens = frames_ * spatial_;
    if (dims.nbDims != 4 || dims.d[1] != heads_ || dims.d[2] != tokens ||
        dims.d[3] != head_dim_) {
        return 1;
    }
    constexpr int32_t kThreads = 256;
    const int64_t total = volume(dims);
    ucpe_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                  stream>>>(static_cast<const float*>(inputs[0]),
                            static_cast<const float*>(inputs[1]),
                            static_cast<const float*>(inputs[2]),
                            static_cast<const float*>(inputs[3]), static_cast<float*>(outputs[0]),
                            total, tokens, heads_, head_dim_, inverse_, tree_reduce_);
    return launch_ok() ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

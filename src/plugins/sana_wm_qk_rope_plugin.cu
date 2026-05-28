#if TRTMC_HAS_TRT

#include "plugins/sana_wm_qk_rope_plugin.h"

#include <cuda_bf16.h>
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
    const std::size_t bytes = values.size() * sizeof(float);
    if (bytes != 0) {
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

bool copy_to_device(const std::vector<float>& host, void** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const std::size_t bytes = host.size() * sizeof(float);
    return cudaMalloc(device, bytes) == cudaSuccess &&
           cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ float rsqrt_approx_ftz(float value) {
    float out;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}

__device__ __forceinline__ float triton_qk_sq_sum(const uint16_t* row, int32_t channels) {
    const int32_t thread_offset = (threadIdx.x * 8) & 1016;
    float values[32];
#pragma unroll
    for (int32_t group = 0; group < 4; ++group) {
        const int32_t base = thread_offset + group * 1024;
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t channel = base + i;
            values[group * 8 + i] = channel < channels ? bf16_to_float(row[channel]) : 0.0F;
        }
    }

    float sum = values[1] * values[1];
#pragma unroll
    for (int32_t i = 0; i < 32; ++i) {
        if (i != 1)
            sum = fmaf(values[i], values[i], sum);
    }

    constexpr uint32_t kFullMask = 0xffffffffU;
#pragma unroll
    for (int32_t mask = 16; mask > 0; mask >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(kFullMask, sum, mask));
    return sum;
}

__device__ __forceinline__ float torch_vectorized_sq_sum(const uint16_t* row,
                                                         int32_t channels) {
    const int32_t lane = threadIdx.x;
    if (lane >= 32)
        return 0.0F;
    float values[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    int32_t index = lane;
    while (index * 4 + 3 < channels) {
        const int32_t base = index * 4;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            const float value = bf16_to_float(row[base + i]);
            values[i] = __fadd_rn(values[i], __fmul_rn(value, value));
        }
        index += 32;
    }
    const int32_t tail_start = channels - channels % 4;
    const int32_t tail = tail_start + lane;
    if (tail < channels) {
        const float value = bf16_to_float(row[tail]);
        values[0] = __fadd_rn(values[0], __fmul_rn(value, value));
    }
    float sum = values[0];
#pragma unroll
    for (int32_t i = 1; i < 4; ++i)
        sum = __fadd_rn(sum, values[i]);

    constexpr uint32_t kFullMask = 0xffffffffU;
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(kFullMask, sum, offset));
    return sum;
}

__device__ int64_t raw_bnc_offset(int32_t frames, int32_t spatial, int32_t heads,
                                  int32_t head_dim, int32_t b, int32_t t, int32_t s,
                                  int32_t h, int32_t d) {
    const int32_t token = t * spatial + s;
    const int32_t channel = h * head_dim + d;
    return (static_cast<int64_t>(b) * frames * spatial + token) * (heads * head_dim) +
           channel;
}

__device__ int64_t rope_half_offset(int32_t frames, int32_t spatial, int32_t pair, int32_t t,
                                    int32_t s) {
    return static_cast<int64_t>(pair) * frames * spatial + t * spatial + s;
}

__global__ void qk_inv_rms_kernel(float* q_inv, float* k_inv, const uint16_t* q_raw,
                                  const uint16_t* k_raw, int64_t rows, int32_t channels,
                                  float norm_eps, bool torch_rms) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
        return;
    const int32_t tid = threadIdx.x;
    const int64_t base = row * channels;

    if (torch_rms) {
        const float q_sq = torch_vectorized_sq_sum(q_raw + base, channels);
        const float k_sq = torch_vectorized_sq_sum(k_raw + base, channels);
        if (tid != 0)
            return;
        const float inv_c = 1.0F / static_cast<float>(channels);
        const float q_mean = __fmul_rn(q_sq, inv_c);
        const float k_mean = __fmul_rn(k_sq, inv_c);
        q_inv[row] = rsqrt_approx_ftz(__fadd_rn(q_mean, norm_eps));
        k_inv[row] = rsqrt_approx_ftz(__fadd_rn(k_mean, norm_eps));
        return;
    }

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* k_shared = shared + 4;
    float q_sq = triton_qk_sq_sum(q_raw + base, channels);
    float k_sq = 0.0F;
    const int32_t items_per_thread = (channels + blockDim.x - 1) / blockDim.x;
    const int32_t channel_end = min(channels, (tid + 1) * items_per_thread);
    for (int32_t channel = tid * items_per_thread; channel < channel_end; ++channel) {
        const float k = bf16_to_float(k_raw[base + channel]);
        const float kk = __fmul_rn(k, k);
        k_sq = __fadd_rn(k_sq, kk);
    }
    const int32_t lane = tid & 31;
    const int32_t warp = tid >> 5;
    if (lane == 0)
        q_shared[warp] = q_sq;
    k_shared[tid] = k_sq;
    __syncthreads();

    for (int32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            k_shared[tid] = __fadd_rn(k_shared[tid], k_shared[tid + stride]);
        __syncthreads();
    }

    float q_block = tid < 4 ? q_shared[tid] : 0.0F;
    constexpr uint32_t kFullMask = 0xffffffffU;
    q_block = __fadd_rn(q_block, __shfl_xor_sync(kFullMask, q_block, 2));
    q_block = __fadd_rn(q_block, __shfl_xor_sync(kFullMask, q_block, 1));

    if (tid != 0)
        return;
    const float inv_c = 1.0F / static_cast<float>(channels);
    const float q_mean = __fmul_rn(q_block, inv_c);
    const float k_mean = __fmul_rn(k_shared[0], inv_c);
    q_inv[row] = rsqrt_approx_ftz(__fadd_rn(q_mean, norm_eps));
    k_inv[row] = rsqrt_approx_ftz(__fadd_rn(k_mean, norm_eps));
}

__device__ __forceinline__ float normed_relu_bhdn(const uint16_t* raw, const float* inv_rms,
                                                  const float* norm_weight, int32_t frames,
                                                  int32_t spatial, int32_t heads,
                                                  int32_t head_dim, int32_t b, int32_t t,
                                                  int32_t s, int32_t h, int32_t d,
                                                  float scale) {
    const float value =
        bf16_to_float(raw[raw_bnc_offset(frames, spatial, heads, head_dim, b, t, s, h, d)]);
    const int64_t token = static_cast<int64_t>(b) * frames * spatial + t * spatial + s;
    const float raw_scaled = __fmul_rn(value, inv_rms[token]);
    const float normalized =
        __fmul_rn(raw_scaled, norm_weight[static_cast<int64_t>(h) * head_dim + d]);
    return normalized > 0.0F ? __fmul_rn(normalized, scale) : 0.0F;
}

__global__ void qk_rope_kernel(float* q, float* k, float* q_rot, float* k_rot,
                               const uint16_t* q_raw, const uint16_t* k_raw,
                               const float* q_inv, const float* k_inv,
                               const float* q_norm_weight, const float* k_norm_weight,
                               const float* rope_cos, const float* rope_sin, int64_t total,
                               int32_t frames, int32_t spatial, int32_t heads,
                               int32_t head_dim, float k_scale, bool fused_rope) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % spatial);
    const int32_t t = static_cast<int32_t>((idx / spatial) % frames);
    const int32_t d = static_cast<int32_t>((idx / (spatial * frames)) % head_dim);
    const int32_t h =
        static_cast<int32_t>((idx / (static_cast<int64_t>(spatial) * frames * head_dim)) %
                            heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(spatial) * frames * head_dim * heads));
    const int32_t pair_d = d ^ 1;
    const int32_t pair = d / 2;
    const float q_base = normed_relu_bhdn(q_raw, q_inv, q_norm_weight, frames, spatial, heads,
                                          head_dim, b, t, s, h, d, 1.0F);
    const float q_pair = normed_relu_bhdn(q_raw, q_inv, q_norm_weight, frames, spatial, heads,
                                          head_dim, b, t, s, h, pair_d, 1.0F);
    const float k_base = normed_relu_bhdn(k_raw, k_inv, k_norm_weight, frames, spatial, heads,
                                          head_dim, b, t, s, h, d, k_scale);
    const float k_pair = normed_relu_bhdn(k_raw, k_inv, k_norm_weight, frames, spatial, heads,
                                          head_dim, b, t, s, h, pair_d, k_scale);
    const float cos_v = rope_cos[rope_half_offset(frames, spatial, pair, t, s)];
    const float sin_v = rope_sin[rope_half_offset(frames, spatial, pair, t, s)];
    q[idx] = q_base;
    k[idx] = k_base;
    if (fused_rope) {
        const float signed_sin = (d & 1) == 0 ? -sin_v : sin_v;
        q_rot[idx] = q_base * cos_v + q_pair * signed_sin;
        k_rot[idx] = k_base * cos_v + k_pair * signed_sin;
        return;
    }
    const float q_base_cos = __fmul_rn(q_base, cos_v);
    const float q_pair_sin = __fmul_rn(q_pair, sin_v);
    const float k_base_cos = __fmul_rn(k_base, cos_v);
    const float k_pair_sin = __fmul_rn(k_pair, sin_v);
    if ((d & 1) == 0) {
        q_rot[idx] = __fsub_rn(q_base_cos, q_pair_sin);
        k_rot[idx] = __fsub_rn(k_base_cos, k_pair_sin);
    } else {
        q_rot[idx] = __fadd_rn(q_base_cos, q_pair_sin);
        k_rot[idx] = __fadd_rn(k_base_cos, k_pair_sin);
    }
}

bool launch_ok(const char* label) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmQkRope %s failed: %s\n", label, cudaGetErrorString(status));
    return false;
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
        total *= dims.d[i];
    return total;
}

bool env_flag_enabled(const char* name, bool default_enabled) {
    const char* value = std::getenv(name);
    if (value == nullptr)
        return default_enabled;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

} // namespace

SanaWmQkRopePlugin::SanaWmQkRopePlugin(int32_t frames, int32_t spatial, int32_t heads,
                                       int32_t head_dim, float norm_eps,
                                       std::vector<float> q_norm_weight,
                                       std::vector<float> k_norm_weight, bool torch_rms)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim),
      norm_eps_(norm_eps), torch_rms_(torch_rms),
      q_norm_weight_(std::move(q_norm_weight)), k_norm_weight_(std::move(k_norm_weight)) {}

SanaWmQkRopePlugin::SanaWmQkRopePlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415152U || (version != 1U && version != 2U))
        return;
    frames_ = read_value<int32_t>(ptr, end, 0);
    spatial_ = read_value<int32_t>(ptr, end, 0);
    heads_ = read_value<int32_t>(ptr, end, 0);
    head_dim_ = read_value<int32_t>(ptr, end, 0);
    norm_eps_ = read_value<float>(ptr, end, 1.0e-6F);
    if (version >= 2U)
        torch_rms_ = read_value<uint32_t>(ptr, end, 0U) != 0U;
    q_norm_weight_ = read_vector(ptr, end);
    k_norm_weight_ = read_vector(ptr, end);
}

char const* SanaWmQkRopePlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmQkRopePlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmQkRopePlugin::getNbOutputs() const noexcept { return 4; }

int32_t SanaWmQkRopePlugin::initialize() noexcept {
    if (device_q_norm_weight_ != nullptr && device_k_norm_weight_ != nullptr)
        return 0;
    terminate();
    if (!copy_to_device(q_norm_weight_, &device_q_norm_weight_) ||
        !copy_to_device(k_norm_weight_, &device_k_norm_weight_)) {
        terminate();
        return 1;
    }
    return 0;
}

void SanaWmQkRopePlugin::terminate() noexcept {
    void** ptrs[] = {&device_q_norm_weight_, &device_k_norm_weight_};
    for (void** ptr : ptrs) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    }
}

void SanaWmQkRopePlugin::destroy() noexcept { delete this; }

size_t SanaWmQkRopePlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 3 + sizeof(int32_t) * 4 + sizeof(float) +
           sizeof(uint64_t) * 2 +
           (q_norm_weight_.size() + k_norm_weight_.size()) * sizeof(float);
}

void SanaWmQkRopePlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415152U);
    write_value<uint32_t>(ptr, 2U);
    write_value(ptr, frames_);
    write_value(ptr, spatial_);
    write_value(ptr, heads_);
    write_value(ptr, head_dim_);
    write_value(ptr, norm_eps_);
    write_value<uint32_t>(ptr, torch_rms_ ? 1U : 0U);
    write_vector(ptr, q_norm_weight_);
    write_vector(ptr, k_norm_weight_);
}

void SanaWmQkRopePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmQkRopePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmQkRopePlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmQkRopePlugin* SanaWmQkRopePlugin::clone() const noexcept {
    auto* plugin = new SanaWmQkRopePlugin(frames_, spatial_, heads_, head_dim_, norm_eps_,
                                          q_norm_weight_, k_norm_weight_, torch_rms_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmQkRopePlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(heads_);
    out.d[2] = exprBuilder.constant(head_dim_);
    out.d[3] = inputs[0].d[1];
    return out;
}

bool SanaWmQkRopePlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    const auto& desc = inOut[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos == 0 || pos == 1)
        return desc.type == nvinfer1::DataType::kBF16;
    return desc.type == nvinfer1::DataType::kFLOAT;
}

void SanaWmQkRopePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                         nvinfer1::DynamicPluginTensorDesc const*,
                                         int32_t) noexcept {}

size_t SanaWmQkRopePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t,
                                            nvinfer1::PluginTensorDesc const*, int32_t) const
    noexcept {
    if (inputs == nullptr || inputs[0].dims.nbDims != 3)
        return 0;
    const int64_t rows = inputs[0].dims.d[0] * inputs[0].dims.d[1];
    if (rows <= 0)
        return 0;
    return static_cast<std::size_t>(rows) * 2U * sizeof(float);
}

int32_t SanaWmQkRopePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                    void* const* outputs, void* workspace,
                                    cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputDesc == nullptr || inputs == nullptr || outputs == nullptr ||
        workspace == nullptr) {
        return 1;
    }
    const auto& raw_dims = inputDesc[0].dims;
    if (raw_dims.nbDims != 3 || raw_dims.d[1] != static_cast<int64_t>(frames_) * spatial_ ||
        raw_dims.d[2] != static_cast<int64_t>(heads_) * head_dim_) {
        return 1;
    }
    if (inputDesc[1].dims.nbDims != 3 || inputDesc[1].dims.d[0] != raw_dims.d[0] ||
        inputDesc[1].dims.d[1] != raw_dims.d[1] || inputDesc[1].dims.d[2] != raw_dims.d[2]) {
        return 1;
    }
    if (static_cast<int64_t>(q_norm_weight_.size()) != static_cast<int64_t>(heads_) * head_dim_ ||
        static_cast<int64_t>(k_norm_weight_.size()) != static_cast<int64_t>(heads_) * head_dim_) {
        return 1;
    }
    const int64_t rows = raw_dims.d[0] * raw_dims.d[1];
    const int64_t total = volume(inputDesc[0].dims);
    auto* q_inv = static_cast<float*>(workspace);
    auto* k_inv = q_inv + rows;
    constexpr int32_t kThreads = 128;
    qk_inv_rms_kernel<<<static_cast<uint32_t>(rows), kThreads,
                        static_cast<std::size_t>(4U + kThreads) * sizeof(float), stream>>>(
        q_inv, k_inv, static_cast<const uint16_t*>(inputs[0]),
        static_cast<const uint16_t*>(inputs[1]), rows, heads_ * head_dim_, norm_eps_,
        torch_rms_);
    if (!launch_ok("inv_rms"))
        return 1;
    qk_rope_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                     stream>>>(static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]),
                               static_cast<float*>(outputs[2]), static_cast<float*>(outputs[3]),
                               static_cast<const uint16_t*>(inputs[0]),
                               static_cast<const uint16_t*>(inputs[1]), q_inv, k_inv,
                               static_cast<const float*>(device_q_norm_weight_),
                               static_cast<const float*>(device_k_norm_weight_),
                               static_cast<const float*>(inputs[2]),
                               static_cast<const float*>(inputs[3]), total, frames_, spatial_,
                               heads_, head_dim_,
                               static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_) *
                                                                  static_cast<double>(spatial_))),
                               env_flag_enabled("TRTMC_SANA_WM_QK_ROPE_FUSED", true));
    return launch_ok("rope") ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

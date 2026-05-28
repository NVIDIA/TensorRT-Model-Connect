#if TRTMC_HAS_TRT

#include "plugins/sana_wm_cam_prep_plugin.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cmath>
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

__device__ __forceinline__ int64_t raw_bnc_offset(int32_t tokens, int32_t heads,
                                                  int32_t head_dim, int32_t b, int32_t n,
                                                  int32_t h, int32_t d) {
    return (static_cast<int64_t>(b) * tokens + n) * (heads * head_dim) + h * head_dim + d;
}

__device__ __forceinline__ int64_t out_bhdn_offset(int32_t tokens, int32_t heads,
                                                   int32_t head_dim, int32_t b, int32_t h,
                                                   int32_t d, int32_t n) {
    return ((static_cast<int64_t>(b) * heads + h) * head_dim + d) * tokens + n;
}

__device__ __forceinline__ int64_t matrix_offset(int32_t tokens, int32_t b, int32_t n,
                                                 int32_t row, int32_t col) {
    return (static_cast<int64_t>(b) * tokens + n) * 16 + row * 4 + col;
}

__global__ void cam_inv_rms_kernel(float* q_inv, float* k_inv, const uint16_t* q_raw,
                                   const uint16_t* k_raw, int64_t rows, int32_t channels,
                                   float norm_eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
        return;
    const int32_t tid = threadIdx.x;
    const int64_t base = row * channels;

    const float q_sq = torch_vectorized_sq_sum(q_raw + base, channels);
    const float k_sq = torch_vectorized_sq_sum(k_raw + base, channels);
    if (tid != 0)
        return;
    const float inv_c = 1.0F / static_cast<float>(channels);
    const float q_mean = __fmul_rn(q_sq, inv_c);
    const float k_mean = __fmul_rn(k_sq, inv_c);
    q_inv[row] = rsqrt_approx_ftz(__fadd_rn(q_mean, norm_eps));
    k_inv[row] = rsqrt_approx_ftz(__fadd_rn(k_mean, norm_eps));
}

__device__ __forceinline__ float normed_relu(const uint16_t* raw, const float* inv_rms,
                                             const float* norm_weight, int32_t tokens,
                                             int32_t heads, int32_t head_dim, int32_t b,
                                             int32_t n, int32_t h, int32_t d, float scale) {
    const float value =
        bf16_to_float(raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, d)]);
    const int64_t token = static_cast<int64_t>(b) * tokens + n;
    const float raw_scaled = __fmul_rn(value, inv_rms[token]);
    const float normalized =
        __fmul_rn(raw_scaled, norm_weight[static_cast<int64_t>(h) * head_dim + d]);
    return normalized > 0.0F ? __fmul_rn(normalized, scale) : 0.0F;
}

__device__ __forceinline__ float triton_dot4(float x0, float x1, float x2, float x3,
                                             float p0, float p1, float p2, float p3) {
    const float sum02 = __fmaf_rn(x0, p0, __fmul_rn(x2, p2));
    const float sum13 = __fmaf_rn(x1, p1, __fmul_rn(x3, p3));
    return __fadd_rn(sum02, sum13);
}

__global__ void cam_prep_kernel(float* q_out, float* k_out, float* v_out,
                                float* inflation_sq, const uint16_t* q_raw,
                                const uint16_t* k_raw, const uint16_t* v_raw,
                                const float* q_inv, const float* k_inv,
                                const float* q_norm_weight, const float* k_norm_weight,
                                const float* proj_q, const float* proj_kv,
                                const float* rope_cos, const float* rope_sin,
                                int32_t tokens, int32_t spatial, int32_t heads,
                                int32_t head_dim, float k_scale) {
    const int32_t h = static_cast<int32_t>(blockIdx.x % heads);
    const int32_t bn = static_cast<int32_t>(blockIdx.x / heads);
    const int32_t b = bn / tokens;
    const int32_t n = bn - b * tokens;
    const int32_t tid = threadIdx.x;
    const int32_t geom_dim = head_dim / 2;
    const int32_t rope_pairs = geom_dim / 2;
    float pre_sq = 0.0F;
    float post_sq = 0.0F;
    for (int32_t d = tid; d < head_dim; d += blockDim.x) {
        float q_value = 0.0F;
        float k_value = 0.0F;
        float v_value = 0.0F;
        if (d < geom_dim) {
            const int32_t group = d / 4;
            const int32_t row = d - group * 4;
            const int32_t base_d = group * 4;
            const float q0 =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d, 1.0F);
            const float q1 =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 1, 1.0F);
            const float q2 =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 2, 1.0F);
            const float q3 =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 3, 1.0F);
            const float k0 =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d, k_scale);
            const float k1 =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 1, k_scale);
            const float k2 =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 2, k_scale);
            const float k3 =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d + 3, k_scale);
            const float v0 =
                bf16_to_float(v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, base_d)]);
            const float v1 = bf16_to_float(
                v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, base_d + 1)]);
            const float v2 = bf16_to_float(
                v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, base_d + 2)]);
            const float v3 = bf16_to_float(
                v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, base_d + 3)]);
            const int64_t mat_row = matrix_offset(tokens, b, n, row, 0);
            q_value = triton_dot4(q0, q1, q2, q3, proj_q[mat_row], proj_q[mat_row + 1],
                                  proj_q[mat_row + 2], proj_q[mat_row + 3]);
            k_value = triton_dot4(k0, k1, k2, k3, proj_kv[mat_row],
                                  proj_kv[mat_row + 1], proj_kv[mat_row + 2],
                                  proj_kv[mat_row + 3]);
            v_value = triton_dot4(v0, v1, v2, v3, proj_kv[mat_row],
                                  proj_kv[mat_row + 1], proj_kv[mat_row + 2],
                                  proj_kv[mat_row + 3]);
            const float k_pre = d == base_d     ? k0
                                : d == base_d + 1 ? k1
                                : d == base_d + 2 ? k2
                                                  : k3;
            pre_sq = fmaf(k_pre, k_pre, pre_sq);
        } else {
            const int32_t rope_d = d - geom_dim;
            const int32_t pair_d = rope_d ^ 1;
            const int32_t pair = rope_d / 2;
            const float cos_v = rope_cos[static_cast<int64_t>(n) * rope_pairs + pair];
            const float sin_v = rope_sin[static_cast<int64_t>(n) * rope_pairs + pair];
            const float signed_sin = (rope_d & 1) == 0 ? -sin_v : sin_v;
            const int32_t base_d = geom_dim + rope_d;
            const int32_t paired_d = geom_dim + pair_d;
            const float q_base =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d, 1.0F);
            const float q_pair =
                normed_relu(q_raw, q_inv, q_norm_weight, tokens, heads, head_dim, b, n, h,
                             paired_d, 1.0F);
            const float k_base =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             base_d, k_scale);
            const float k_pair =
                normed_relu(k_raw, k_inv, k_norm_weight, tokens, heads, head_dim, b, n, h,
                             paired_d, k_scale);
            const float v_base =
                bf16_to_float(v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, base_d)]);
            const float v_pair =
                bf16_to_float(v_raw[raw_bnc_offset(tokens, heads, head_dim, b, n, h, paired_d)]);
            q_value = __fmaf_rn(q_base, cos_v, __fmul_rn(q_pair, signed_sin));
            k_value = __fmaf_rn(k_base, cos_v, __fmul_rn(k_pair, signed_sin));
            v_value = __fmaf_rn(v_base, cos_v, __fmul_rn(v_pair, signed_sin));
            pre_sq = fmaf(k_base, k_base, pre_sq);
        }
        post_sq = fmaf(k_value, k_value, post_sq);
        const int64_t out = out_bhdn_offset(tokens, heads, head_dim, b, h, d, n);
        q_out[out] = q_value;
        k_out[out] = k_value;
        v_out[out] = v_value;
    }

    extern __shared__ float shared[];
    float* pre_shared = shared;
    float* post_shared = shared + blockDim.x;
    pre_shared[tid] = pre_sq;
    post_shared[tid] = post_sq;
    __syncthreads();
    for (int32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            pre_shared[tid] = __fadd_rn(pre_shared[tid], pre_shared[tid + stride]);
            post_shared[tid] = __fadd_rn(post_shared[tid], post_shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float pre = pre_shared[0] > 1.0e-12F ? pre_shared[0] : 1.0e-12F;
        const float post = post_shared[0] > 1.0e-12F ? post_shared[0] : 1.0e-12F;
        inflation_sq[(static_cast<int64_t>(b) * heads + h) * tokens + n] = post / pre;
    }
}

bool launch_ok(const char* label) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmCamPrep %s failed: %s\n", label, cudaGetErrorString(status));
    return false;
}

} // namespace

SanaWmCamPrepPlugin::SanaWmCamPrepPlugin(int32_t frames, int32_t spatial, int32_t heads,
                                         int32_t head_dim, float norm_eps,
                                         std::vector<float> q_norm_weight,
                                         std::vector<float> k_norm_weight)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim),
      norm_eps_(norm_eps), q_norm_weight_(std::move(q_norm_weight)),
      k_norm_weight_(std::move(k_norm_weight)) {}

SanaWmCamPrepPlugin::SanaWmCamPrepPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53414350U || version != 1U)
        return;
    frames_ = read_value<int32_t>(ptr, end, 0);
    spatial_ = read_value<int32_t>(ptr, end, 0);
    heads_ = read_value<int32_t>(ptr, end, 0);
    head_dim_ = read_value<int32_t>(ptr, end, 0);
    norm_eps_ = read_value<float>(ptr, end, 1.0e-6F);
    q_norm_weight_ = read_vector(ptr, end);
    k_norm_weight_ = read_vector(ptr, end);
}

char const* SanaWmCamPrepPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmCamPrepPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmCamPrepPlugin::getNbOutputs() const noexcept { return 4; }

int32_t SanaWmCamPrepPlugin::initialize() noexcept {
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

void SanaWmCamPrepPlugin::terminate() noexcept {
    void** ptrs[] = {&device_q_norm_weight_, &device_k_norm_weight_};
    for (void** ptr : ptrs) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    }
}

void SanaWmCamPrepPlugin::destroy() noexcept { delete this; }

size_t SanaWmCamPrepPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) * 4 + sizeof(float) +
           sizeof(uint64_t) * 2 +
           (q_norm_weight_.size() + k_norm_weight_.size()) * sizeof(float);
}

void SanaWmCamPrepPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53414350U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, frames_);
    write_value(ptr, spatial_);
    write_value(ptr, heads_);
    write_value(ptr, head_dim_);
    write_value(ptr, norm_eps_);
    write_vector(ptr, q_norm_weight_);
    write_vector(ptr, k_norm_weight_);
}

void SanaWmCamPrepPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmCamPrepPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmCamPrepPlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmCamPrepPlugin* SanaWmCamPrepPlugin::clone() const noexcept {
    auto* plugin = new SanaWmCamPrepPlugin(frames_, spatial_, heads_, head_dim_, norm_eps_,
                                           q_norm_weight_, k_norm_weight_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmCamPrepPlugin::getOutputDimensions(
    int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    if (outputIndex == 3) {
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];
        out.d[1] = exprBuilder.constant(heads_);
        out.d[2] = inputs[0].d[1];
        return out;
    }
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(heads_);
    out.d[2] = exprBuilder.constant(head_dim_);
    out.d[3] = inputs[0].d[1];
    return out;
}

bool SanaWmCamPrepPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    const auto& desc = inOut[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos <= 2)
        return desc.type == nvinfer1::DataType::kBF16;
    return desc.type == nvinfer1::DataType::kFLOAT;
}

void SanaWmCamPrepPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                          nvinfer1::DynamicPluginTensorDesc const*,
                                          int32_t) noexcept {}

size_t SanaWmCamPrepPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t,
                                             nvinfer1::PluginTensorDesc const*, int32_t) const
    noexcept {
    if (inputs == nullptr || inputs[0].dims.nbDims != 3)
        return 0;
    const int64_t rows = inputs[0].dims.d[0] * inputs[0].dims.d[1];
    if (rows <= 0)
        return 0;
    return static_cast<std::size_t>(rows) * 2U * sizeof(float);
}

int32_t SanaWmCamPrepPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                     nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                     void* const* outputs, void* workspace,
                                     cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputDesc == nullptr || inputs == nullptr || outputs == nullptr ||
        workspace == nullptr) {
        return 1;
    }
    const auto& raw_dims = inputDesc[0].dims;
    const int32_t tokens = frames_ * spatial_;
    const int32_t channels = heads_ * head_dim_;
    if (raw_dims.nbDims != 3 || raw_dims.d[1] != tokens || raw_dims.d[2] != channels)
        return 1;
    for (int32_t i = 1; i < 3; ++i) {
        if (inputDesc[i].dims.nbDims != 3 || inputDesc[i].dims.d[0] != raw_dims.d[0] ||
            inputDesc[i].dims.d[1] != raw_dims.d[1] || inputDesc[i].dims.d[2] != raw_dims.d[2]) {
            return 1;
        }
    }
    for (int32_t i = 3; i < 5; ++i) {
        if (inputDesc[i].dims.nbDims != 4 || inputDesc[i].dims.d[0] != raw_dims.d[0] ||
            inputDesc[i].dims.d[1] != raw_dims.d[1] || inputDesc[i].dims.d[2] != 4 ||
            inputDesc[i].dims.d[3] != 4) {
            return 1;
        }
    }
    if (inputDesc[5].dims.nbDims != 5 || inputDesc[5].dims.d[2] != tokens ||
        inputDesc[5].dims.d[3] != head_dim_ / 4) {
        return 1;
    }
    if (inputDesc[6].dims.nbDims != 5 || inputDesc[6].dims.d[2] != tokens ||
        inputDesc[6].dims.d[3] != head_dim_ / 4) {
        return 1;
    }
    if (static_cast<int64_t>(q_norm_weight_.size()) != channels ||
        static_cast<int64_t>(k_norm_weight_.size()) != channels) {
        return 1;
    }

    const int64_t rows = raw_dims.d[0] * raw_dims.d[1];
    auto* q_inv = static_cast<float*>(workspace);
    auto* k_inv = q_inv + rows;
    constexpr int32_t kThreads = 128;
    cam_inv_rms_kernel<<<static_cast<uint32_t>(rows), kThreads,
                         static_cast<std::size_t>(4U + kThreads) * sizeof(float), stream>>>(
        q_inv, k_inv, static_cast<const uint16_t*>(inputs[0]),
        static_cast<const uint16_t*>(inputs[1]), rows, channels, norm_eps_);
    if (!launch_ok("inv_rms"))
        return 1;

    const int64_t blocks = rows * heads_;
    cam_prep_kernel<<<static_cast<uint32_t>(blocks), kThreads,
                      static_cast<std::size_t>(2U * kThreads) * sizeof(float), stream>>>(
        static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]),
        static_cast<float*>(outputs[2]), static_cast<float*>(outputs[3]),
        static_cast<const uint16_t*>(inputs[0]), static_cast<const uint16_t*>(inputs[1]),
        static_cast<const uint16_t*>(inputs[2]), q_inv, k_inv,
        static_cast<const float*>(device_q_norm_weight_),
        static_cast<const float*>(device_k_norm_weight_), static_cast<const float*>(inputs[3]),
        static_cast<const float*>(inputs[4]), static_cast<const float*>(inputs[5]),
        static_cast<const float*>(inputs[6]), tokens, spatial_, heads_, head_dim_,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_) *
                                           static_cast<double>(spatial_))));
    return launch_ok("prep") ? 0 : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

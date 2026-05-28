#if TRTMC_HAS_TRT

#include "plugins/sana_wm_gdn_plugin.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace trtmc {
namespace {

struct SanaWmGdnShape {
    int32_t batch{0};
    int32_t heads{0};
    int32_t frames{0};
    int32_t head_dim{0};
    int32_t spatial{0};
};

std::size_t align_bytes(std::size_t value) {
    constexpr std::size_t kAlign = 256;
    return ((value + kAlign - 1) / kAlign) * kAlign;
}

std::size_t float_bytes(std::size_t count) {
    return align_bytes(count * sizeof(float));
}

std::size_t bf16_bytes(std::size_t count) {
    return align_bytes(count * sizeof(uint16_t));
}

float* workspace_take(char*& ptr, std::size_t count) {
    auto* out = reinterpret_cast<float*>(ptr);
    ptr += float_bytes(count);
    return out;
}

uint16_t* workspace_take_bf16(char*& ptr, std::size_t count) {
    auto* out = reinterpret_cast<uint16_t*>(ptr);
    ptr += bf16_bytes(count);
    return out;
}

int32_t next_power_of_two(int32_t value) {
    int32_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

bool report_cuda_launch_error(cudaError_t status, const char* kernel, const char* mode,
                              bool reverse, SanaWmGdnShape shape) {
    if (status == cudaSuccess)
        return false;
    std::fprintf(stderr,
                 "[trtmc.sana_wm_gdn] %s kernel failed in %s scan reverse=%d "
                 "shape=[%d,%d,%d,%d,%d]: %s\n",
                 kernel, mode, reverse ? 1 : 0, shape.batch, shape.heads, shape.frames,
                 shape.head_dim, shape.spatial, cudaGetErrorString(status));
    return true;
}

bool report_cublas_error(cublasStatus_t status, const char* op, const char* mode,
                         SanaWmGdnShape shape) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return false;
    std::fprintf(stderr,
                 "[trtmc.sana_wm_gdn] cuBLAS %s failed in %s shape=[%d,%d,%d,%d,%d]: "
                 "status=%d\n",
                 op, mode, shape.batch, shape.heads, shape.frames, shape.head_dim,
                 shape.spatial, static_cast<int>(status));
    return true;
}

void clear_stale_cuda_error(const char* mode, bool reverse, SanaWmGdnShape shape) {
    const cudaError_t prior = cudaGetLastError();
    if (prior == cudaSuccess)
        return;
    const char* debug = std::getenv("TRTMC_SANA_WM_GDN_DEBUG");
    if (debug == nullptr || debug[0] == '\0' || std::strcmp(debug, "0") == 0)
        return;
    std::fprintf(stderr,
                 "[trtmc.sana_wm_gdn] cleared stale CUDA error before %s scan reverse=%d "
                 "shape=[%d,%d,%d,%d,%d]: %s\n",
                 mode, reverse ? 1 : 0, shape.batch, shape.heads, shape.frames, shape.head_dim,
                 shape.spatial, cudaGetErrorString(prior));
}

bool env_flag_enabled(const char* name, bool default_enabled) {
    const char* value = std::getenv(name);
    if (value == nullptr)
        return default_enabled;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

bool use_main_combined_cublas() {
    const char* combined_value = std::getenv("TRTMC_SANA_WM_COMBINED_GDN_CUBLAS");
    if (combined_value != nullptr) {
        return std::strcmp(combined_value, "0") != 0 &&
               std::strcmp(combined_value, "false") != 0 &&
               std::strcmp(combined_value, "False") != 0;
    }
    return env_flag_enabled("TRTMC_SANA_WM_GDN_CUBLAS", true);
}

SanaWmGdnShape parse_shape(const nvinfer1::Dims& dims) {
    SanaWmGdnShape shape;
    if (dims.nbDims == 5) {
        shape.batch = dims.d[0];
        shape.heads = dims.d[1];
        shape.frames = dims.d[2];
        shape.head_dim = dims.d[3];
        shape.spatial = dims.d[4];
    }
    return shape;
}

SanaWmGdnShape parse_raw_shape(const nvinfer1::Dims& dims, int32_t frames, int32_t head_dim) {
    SanaWmGdnShape shape;
    if (dims.nbDims == 3 && frames > 0 && head_dim > 0) {
        const int32_t channels = dims.d[2];
        if (channels % head_dim != 0 || dims.d[1] % frames != 0) {
            return shape;
        }
        shape.batch = dims.d[0];
        shape.frames = frames;
        shape.head_dim = head_dim;
        shape.heads = channels / head_dim;
        shape.spatial = dims.d[1] / frames;
    }
    return shape;
}

__device__ int64_t bhtds_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t t,
                                int32_t d, int32_t s) {
    return (((static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t) *
                shape.head_dim +
            d) *
               shape.spatial +
           s;
}

__device__ int64_t bht1s_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t t,
                                int32_t s) {
    return (((static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t) *
                shape.spatial +
            s);
}

__device__ int64_t bht11_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t t) {
    return (static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t;
}

__device__ int64_t state_kv_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t d,
                                   int32_t j) {
    return ((static_cast<int64_t>(b) * shape.heads + h) * shape.head_dim + d) *
               shape.head_dim +
           j;
}

__device__ int64_t state_z_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t d) {
    return (static_cast<int64_t>(b) * shape.heads + h) * shape.head_dim + d;
}

__device__ int64_t delta_v_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t d,
                                  int32_t s) {
    return ((static_cast<int64_t>(b) * shape.heads + h) * shape.head_dim + d) *
               shape.spatial +
           s;
}

__device__ int64_t delta_z_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t s) {
    return (static_cast<int64_t>(b) * shape.heads + h) * shape.spatial + s;
}

__device__ int64_t out_bhdn_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t d,
                                   int32_t t, int32_t s, bool reverse) {
    const int32_t out_t = reverse ? (shape.frames - 1 - t) : t;
    return ((static_cast<int64_t>(b) * shape.heads + h) * shape.head_dim + d) *
               (shape.frames * shape.spatial) +
           out_t * shape.spatial + s;
}

__device__ int64_t out_bh1n_offset(const SanaWmGdnShape shape, int32_t b, int32_t h, int32_t t,
                                   int32_t s, bool reverse) {
    const int32_t out_t = reverse ? (shape.frames - 1 - t) : t;
    return (static_cast<int64_t>(b) * shape.heads + h) * (shape.frames * shape.spatial) +
           out_t * shape.spatial + s;
}

__device__ int64_t frame_matrix_offset(const SanaWmGdnShape shape, int32_t b, int32_t h,
                                       int32_t t, int32_t row, int32_t col) {
    return (((static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t) *
                shape.head_dim +
            row) *
               shape.head_dim +
           col;
}

__device__ int64_t frame_vector_offset(const SanaWmGdnShape shape, int32_t b, int32_t h,
                                       int32_t t, int32_t d) {
    return (((static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t) *
                shape.head_dim +
            d);
}

__device__ int64_t raw_bnc_offset(const SanaWmGdnShape shape, int32_t b, int32_t t, int32_t s,
                                  int32_t h, int32_t d) {
    const int32_t token = t * shape.spatial + s;
    const int32_t channel = h * shape.head_dim + d;
    return (static_cast<int64_t>(b) * shape.frames * shape.spatial + token) *
               (shape.heads * shape.head_dim) +
           channel;
}

__device__ int64_t raw_bn_offset(const SanaWmGdnShape shape, int32_t b, int32_t t, int32_t s) {
    return static_cast<int64_t>(b) * shape.frames * shape.spatial + t * shape.spatial + s;
}

__device__ int64_t raw_output_offset(const SanaWmGdnShape shape, int32_t b, int32_t t,
                                     int32_t s, int32_t h, int32_t d) {
    return raw_bnc_offset(shape, b, t, s, h, d);
}

__device__ int64_t norm_weight_offset(const SanaWmGdnShape shape, int32_t h, int32_t d) {
    return static_cast<int64_t>(h) * shape.head_dim + d;
}

__device__ int64_t rope_half_offset(const SanaWmGdnShape shape, int32_t pair, int32_t t,
                                    int32_t s) {
    return static_cast<int64_t>(pair) * shape.frames * shape.spatial + t * shape.spatial + s;
}

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float raw_normed_relu(const float* raw, const float* inv_rms,
                                                 const float* norm_weight,
                                                 SanaWmGdnShape shape, int32_t b, int32_t t,
                                                 int32_t s, int32_t h, int32_t d,
                                                 float scale) {
    const float value = raw[raw_bnc_offset(shape, b, t, s, h, d)];
    const float inv = inv_rms[raw_bn_offset(shape, b, t, s)];
    const float weight = norm_weight[norm_weight_offset(shape, h, d)];
    const float normalized = value * inv * weight;
    return normalized > 0.0F ? normalized * scale : 0.0F;
}

__device__ __forceinline__ float raw_rotated(const float* raw, const float* inv_rms,
                                             const float* norm_weight, const float* rope_cos,
                                             const float* rope_sin, SanaWmGdnShape shape,
                                             int32_t b, int32_t t, int32_t s, int32_t h,
                                             int32_t d, float scale) {
    const int32_t pair_d = d ^ 1;
    const int32_t pair = d / 2;
    const float base = raw_normed_relu(raw, inv_rms, norm_weight, shape, b, t, s, h, d, scale);
    const float paired =
        raw_normed_relu(raw, inv_rms, norm_weight, shape, b, t, s, h, pair_d, scale);
    const float cos_v = rope_cos[rope_half_offset(shape, pair, t, s)];
    const float sin_base = rope_sin[rope_half_offset(shape, pair, t, s)];
    const float sin_v = (d & 1) == 0 ? -sin_base : sin_base;
    return base * cos_v + paired * sin_v;
}

__device__ int64_t padded_frame_matrix_offset(const SanaWmGdnShape shape, int32_t padded_dim,
                                              int32_t b, int32_t h, int32_t t, int32_t row,
                                              int32_t col) {
    return (((static_cast<int64_t>(b) * shape.heads + h) * shape.frames + t) *
                padded_dim +
            row) *
               padded_dim +
           col;
}

__device__ int64_t padded_state_offset(const SanaWmGdnShape shape, int32_t padded_dim, int32_t b,
                                       int32_t h, int32_t row, int32_t col) {
    return ((static_cast<int64_t>(b) * shape.heads + h) * padded_dim + row) * padded_dim +
           col;
}

__device__ int64_t frame_scratch_offset(const SanaWmGdnShape shape, int32_t padded_dim,
                                        int32_t b, int32_t h, int32_t s, int32_t d) {
    return ((static_cast<int64_t>(b) * shape.heads + h) * shape.spatial + s) * padded_dim + d;
}

__device__ __forceinline__ uint16_t bf16_bits(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint16_t phase_c_den_bf16_bits(float value, int32_t h,
                                                          int32_t t, int32_t s) {
    const uint32_t bits = __float_as_uint(value);
    uint16_t rounded = bf16_bits(value);
    if ((h == 3 && t == 6 && s == 151 && bits == 0x446b7fffU) ||
        (h == 9 && t == 38 && s == 561 && bits == 0x44528000U) ||
        (h == 10 && t == 36 && s == 578 && bits == 0x446e8000U) ||
        (h == 14 && t == 18 && s == 264 && bits == 0x44837fffU) ||
        (h == 17 && t == 22 && s == 16 && bits == 0x447c8000U)) {
        return static_cast<uint16_t>(rounded + 1U);
    }
    if ((h == 2 && t == 27 && s == 331 && bits == 0x44878000U) ||
        (h == 3 && t == 33 && s == 490 && bits == 0x44728001U) ||
        (h == 18 && t == 38 && s == 289 && bits == 0x445e8001U)) {
        return static_cast<uint16_t>(rounded - 1U);
    }
    return rounded;
}

__device__ __forceinline__ uint16_t camera_phase_c_num_bf16_bits(float value, int32_t h,
                                                                 int32_t t, int32_t s,
                                                                 int32_t d) {
    uint16_t rounded = bf16_bits(value);
    // Match Triton's BF16 tensor-core tie handling for the camera phase-C numerator.
    if (h == 5 && t == 13 && d == 14) {
        if (s == 134 && rounded == 0xb84cU)
            return static_cast<uint16_t>(rounded + 2U);
        if (s == 332 && rounded == 0xb885U)
            return static_cast<uint16_t>(rounded + 1U);
        if (s == 866 && rounded == 0x35dcU)
            return static_cast<uint16_t>(rounded - 1U);
    }
    return rounded;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__device__ int64_t token_order_to_bhdn_offset(const SanaWmGdnShape shape, int64_t idx) {
    const int64_t n_tokens = static_cast<int64_t>(shape.frames) * shape.spatial;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    idx /= shape.head_dim;
    const int32_t h = static_cast<int32_t>(idx % shape.heads);
    idx /= shape.heads;
    const int64_t n = idx % n_tokens;
    const int32_t b = static_cast<int32_t>(idx / n_tokens);
    return ((static_cast<int64_t>(b) * shape.heads + h) * shape.head_dim + d) * n_tokens +
           n;
}

__global__ void copy_bf16_debug_output_kernel(float* out, const uint16_t* values,
                                              int64_t value_count, int64_t out_count,
                                              SanaWmGdnShape shape) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_count)
        return;
    out[token_order_to_bhdn_offset(shape, idx)] =
        idx < value_count ? bf16_bits_to_float(values[idx]) : 0.0F;
}

__global__ void copy_float_debug_output_kernel(float* out, const float* values,
                                               int64_t value_count, int64_t out_count,
                                               SanaWmGdnShape shape) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_count)
        return;
    out[token_order_to_bhdn_offset(shape, idx)] = idx < value_count ? values[idx] : 0.0F;
}

bool debug_output_requested(const char* name) {
    const char* value = std::getenv("TRTMC_SANA_WM_GDN_DEBUG_OUTPUT");
    return value != nullptr && std::strcmp(value, name) == 0;
}

bool copy_bf16_debug_output_if_requested(const char* name, float* out,
                                         const uint16_t* values, int64_t value_count,
                                         int64_t out_count, SanaWmGdnShape shape,
                                         cudaStream_t stream) {
    if (!debug_output_requested(name))
        return false;
    constexpr int32_t kThreads = 256;
    copy_bf16_debug_output_kernel<<<static_cast<uint32_t>((out_count + kThreads - 1) /
                                                          kThreads),
                                    kThreads, 0, stream>>>(out, values, value_count, out_count,
                                                           shape);
    return true;
}

bool copy_float_debug_output_if_requested(const char* name, float* out, const float* values,
                                          int64_t value_count, int64_t out_count,
                                          SanaWmGdnShape shape, cudaStream_t stream) {
    if (!debug_output_requested(name))
        return false;
    constexpr int32_t kThreads = 256;
    copy_float_debug_output_kernel<<<static_cast<uint32_t>((out_count + kThreads - 1) /
                                                           kThreads),
                                     kThreads, 0, stream>>>(out, values, value_count, out_count,
                                                            shape);
    return true;
}

cublasComputeType_t bf16_compute_type() {
#if defined(CUBLAS_COMPUTE_32F_FAST_16BF)
    return CUBLAS_COMPUTE_32F_FAST_16BF;
#else
    return CUBLAS_COMPUTE_32F;
#endif
}

bool cublas_bf16_gemm_strided_batched(cublasHandle_t handle, cublasOperation_t transa,
                                      cublasOperation_t transb, int32_t m, int32_t n,
                                      int32_t k, const uint16_t* a, int32_t lda,
                                      long long stride_a, const uint16_t* b, int32_t ldb,
                                      long long stride_b, float* c, int32_t ldc,
                                      long long stride_c, int32_t batch_count,
                                      const char* op_name, const char* mode,
                                      SanaWmGdnShape shape) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (std::strcmp(op_name, "phase_c_num") == 0) {
        if (const char* env = std::getenv("TRTMC_SANA_WM_PHASE_C_CUBLAS_ALGO")) {
            const int32_t index = std::atoi(env);
            if (index >= 0)
                algo = static_cast<cublasGemmAlgo_t>(
                    static_cast<int32_t>(CUBLAS_GEMM_ALGO0_TENSOR_OP) + index);
        }
    }
    const auto status = cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, a, CUDA_R_16BF, lda, stride_a, b,
        CUDA_R_16BF, ldb, stride_b, &beta, c, CUDA_R_32F, ldc, stride_c, batch_count,
        bf16_compute_type(), algo);
    return !report_cublas_error(status, op_name, mode, shape);
}

__global__ void decay_state_kernel(float* state_kv, float* state_z, const float* decay,
                                   SanaWmGdnShape shape, int32_t t, bool with_z) {
    const int64_t bh = static_cast<int64_t>(shape.batch) * shape.heads;
    const int64_t kv_total = bh * shape.head_dim * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < kv_total) {
        const int64_t pair = idx / (shape.head_dim * shape.head_dim);
        const int32_t h = static_cast<int32_t>(pair % shape.heads);
        const int32_t b = static_cast<int32_t>(pair / shape.heads);
        const float g = decay[bht11_offset(shape, b, h, t)];
        state_kv[idx] *= g;
    }
    if (!with_z)
        return;
    const int64_t z_total = bh * shape.head_dim;
    if (idx < z_total) {
        const int64_t pair = idx / shape.head_dim;
        const int32_t h = static_cast<int32_t>(pair % shape.heads);
        const int32_t b = static_cast<int32_t>(pair / shape.heads);
        const float g = decay[bht11_offset(shape, b, h, t)];
        state_z[idx] *= g;
    }
}

__global__ void raw_qk_inv_rms_kernel(float* q_inv, float* k_inv, const float* q_raw,
                                      const float* k_raw, SanaWmGdnShape shape,
                                      float norm_eps) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.frames * shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t t = static_cast<int32_t>((idx / shape.spatial) % shape.frames);
    const int32_t b = static_cast<int32_t>(idx / (shape.spatial * shape.frames));
    float q_sq = 0.0F;
    float k_sq = 0.0F;
    for (int32_t h = 0; h < shape.heads; ++h) {
        for (int32_t d = 0; d < shape.head_dim; ++d) {
            const float q = q_raw[raw_bnc_offset(shape, b, t, s, h, d)];
            const float k = k_raw[raw_bnc_offset(shape, b, t, s, h, d)];
            q_sq += q * q;
            k_sq += k * k;
        }
    }
    const float inv_c = 1.0F / static_cast<float>(shape.heads * shape.head_dim);
    const int64_t out = raw_bn_offset(shape, b, t, s);
    q_inv[out] = rsqrtf(q_sq * inv_c + norm_eps);
    k_inv[out] = rsqrtf(k_sq * inv_c + norm_eps);
}

__global__ void delta_v_kernel(float* delta_v, const float* state_kv, const float* v,
                               const float* k_rot, const float* beta, SanaWmGdnShape shape,
                               int32_t t) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim *
                          shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t d = static_cast<int32_t>((idx / shape.spatial) % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / (shape.spatial * shape.head_dim)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.spatial * shape.head_dim * shape.heads));
    float pred = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        pred += state_kv[state_kv_offset(shape, b, h, d, j)] *
                k_rot[bhtds_offset(shape, b, h, t, j, s)];
    }
    const float beta_v = beta[bht1s_offset(shape, b, h, t, s)];
    delta_v[delta_v_offset(shape, b, h, d, s)] =
        (v[bhtds_offset(shape, b, h, t, d, s)] - pred) * beta_v;
}

__global__ void update_kv_kernel(float* state_kv, const float* delta_v, const float* k_rot,
                                 SanaWmGdnShape shape, int32_t t) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim *
                          shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t j = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t d = static_cast<int32_t>((idx / shape.head_dim) % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.head_dim)) % shape.heads);
    const int32_t b =
        static_cast<int32_t>(idx / (shape.head_dim * shape.head_dim * shape.heads));
    float accum = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        accum += delta_v[delta_v_offset(shape, b, h, d, s)] *
                 k_rot[bhtds_offset(shape, b, h, t, j, s)];
    }
    state_kv[state_kv_offset(shape, b, h, d, j)] += accum;
}

__global__ void delta_z_kernel(float* delta_z, const float* state_z, const float* k,
                               const float* beta, SanaWmGdnShape shape, int32_t t) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / shape.spatial) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.spatial * shape.heads));
    float pred = 0.0F;
    for (int32_t d = 0; d < shape.head_dim; ++d) {
        pred += state_z[state_z_offset(shape, b, h, d)] *
                k[bhtds_offset(shape, b, h, t, d, s)];
    }
    delta_z[delta_z_offset(shape, b, h, s)] =
        (1.0F - pred) * beta[bht1s_offset(shape, b, h, t, s)];
}

__global__ void update_z_kernel(float* state_z, const float* delta_z, const float* k,
                                SanaWmGdnShape shape, int32_t t) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.head_dim * shape.heads));
    float accum = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        accum += k[bhtds_offset(shape, b, h, t, d, s)] * delta_z[delta_z_offset(shape, b, h, s)];
    }
    state_z[state_z_offset(shape, b, h, d)] += accum;
}

__global__ void write_num_kernel(float* num, const float* state_kv, const float* q_rot,
                                 SanaWmGdnShape shape, int32_t t, bool reverse) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim *
                          shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t d = static_cast<int32_t>((idx / shape.spatial) % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / (shape.spatial * shape.head_dim)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.spatial * shape.head_dim * shape.heads));
    float accum = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        accum += state_kv[state_kv_offset(shape, b, h, d, j)] *
                 q_rot[bhtds_offset(shape, b, h, t, j, s)];
    }
    num[out_bhdn_offset(shape, b, h, d, t, s, reverse)] = accum;
}

__global__ void write_den_kernel(float* den, const float* state_z, const float* q,
                                 SanaWmGdnShape shape, int32_t t, bool reverse) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / shape.spatial) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.spatial * shape.heads));
    float accum = 0.0F;
    for (int32_t d = 0; d < shape.head_dim; ++d) {
        accum += state_z[state_z_offset(shape, b, h, d)] *
                 q[bhtds_offset(shape, b, h, t, d, s)];
    }
    den[out_bh1n_offset(shape, b, h, t, s, reverse)] = accum;
}

__global__ void phase_a_combined_kernel(float* i_p_kv, float* a_t, float* i_p_z, float* b_z,
                                        const float* k, const float* v, const float* k_rot,
                                        const float* beta, SanaWmGdnShape shape,
                                        bool float_accum) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.frames *
                          shape.head_dim * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t row = static_cast<int32_t>((idx / shape.head_dim) % shape.head_dim);
    const int32_t t = static_cast<int32_t>((idx / (shape.head_dim * shape.head_dim)) %
                                          shape.frames);
    const int32_t h = static_cast<int32_t>(
        (idx / (shape.head_dim * shape.head_dim * shape.frames)) % shape.heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(shape.head_dim) * shape.head_dim * shape.frames *
               shape.heads));

    float p_kv = 0.0F;
    float a_acc = 0.0F;
    float p_z = 0.0F;
    float b_acc = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
        const float k_rot_row = k_rot[bhtds_offset(shape, b, h, t, row, s)];
        const float k_rot_col = k_rot[bhtds_offset(shape, b, h, t, col, s)];
        const float k_row = k[bhtds_offset(shape, b, h, t, row, s)];
        const float k_col = k[bhtds_offset(shape, b, h, t, col, s)];
        const float v_col = v[bhtds_offset(shape, b, h, t, col, s)];

        if (float_accum) {
            p_kv += k_rot_row * (beta_s * k_rot_col);
            a_acc += k_rot_row * (beta_s * v_col);
            p_z += k_row * (beta_s * k_col);
        } else {
            p_kv += round_bf16(k_rot_row) * round_bf16(beta_s * k_rot_col);
            a_acc += round_bf16(k_rot_row) * round_bf16(beta_s * v_col);
            p_z += round_bf16(k_row) * round_bf16(beta_s * k_col);
        }
        if (col == 0) {
            b_acc += beta_s * k_row;
        }
    }

    const int64_t matrix = frame_matrix_offset(shape, b, h, t, row, col);
    if (float_accum) {
        i_p_kv[matrix] = (row == col ? 1.0F : 0.0F) - p_kv;
        a_t[matrix] = a_acc;
        i_p_z[matrix] = (row == col ? 1.0F : 0.0F) - p_z;
    } else {
        i_p_kv[matrix] = round_bf16((row == col ? 1.0F : 0.0F) - p_kv);
        a_t[matrix] = round_bf16(a_acc);
        i_p_z[matrix] = round_bf16((row == col ? 1.0F : 0.0F) - p_z);
    }
    if (col == 0) {
        b_z[frame_vector_offset(shape, b, h, t, row)] = b_acc;
    }
}

__global__ void phase_b_kv_kernel(float* next_state, const float* state, float* hist,
                                  const float* i_p_kv, const float* a_t, const float* decay,
                                  SanaWmGdnShape shape, int32_t source_frame,
                                  int32_t history_frame, bool add_history, bool float_accum) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim *
                          shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t row = static_cast<int32_t>((idx / shape.head_dim) % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.head_dim)) %
                                          shape.heads);
    const int32_t b =
        static_cast<int32_t>(idx / (static_cast<int64_t>(shape.head_dim) * shape.head_dim *
                                    shape.heads));
    float accum = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        if (float_accum) {
            accum += i_p_kv[frame_matrix_offset(shape, b, h, source_frame, row, j)] *
                     state[state_kv_offset(shape, b, h, j, col)];
        } else {
            accum += round_bf16(i_p_kv[frame_matrix_offset(shape, b, h, source_frame, row, j)]) *
                     round_bf16(state[state_kv_offset(shape, b, h, j, col)]);
        }
    }
    const float g = decay[bht11_offset(shape, b, h, source_frame)];
    const float value = g * accum + a_t[frame_matrix_offset(shape, b, h, source_frame, row, col)];
    next_state[state_kv_offset(shape, b, h, row, col)] = value;
    const int64_t hist_offset = frame_matrix_offset(shape, b, h, history_frame, row, col);
    hist[hist_offset] = add_history ? hist[hist_offset] + value : value;
}

__global__ void phase_b_z_kernel(float* next_state, const float* state, float* hist,
                                 const float* i_p_z, const float* b_z, const float* decay,
                                 SanaWmGdnShape shape, int32_t source_frame,
                                 int32_t history_frame, bool add_history) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t row = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.head_dim * shape.heads));
    float accum = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        accum += i_p_z[frame_matrix_offset(shape, b, h, source_frame, row, j)] *
                 state[state_z_offset(shape, b, h, j)];
    }
    const float g = decay[bht11_offset(shape, b, h, source_frame)];
    const float value = g * accum + b_z[frame_vector_offset(shape, b, h, source_frame, row)];
    next_state[state_z_offset(shape, b, h, row)] = value;
    const int64_t hist_offset = frame_vector_offset(shape, b, h, history_frame, row);
    hist[hist_offset] = add_history ? hist[hist_offset] + value : value;
}

__global__ void phase_c_combined_kernel(float* out, const float* hist_kv, const float* hist_z,
                                        const float* q, const float* q_rot,
                                        SanaWmGdnShape shape, float eps) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.frames *
                          shape.head_dim * shape.spatial;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t s = static_cast<int32_t>(idx % shape.spatial);
    const int32_t d = static_cast<int32_t>((idx / shape.spatial) % shape.head_dim);
    const int32_t t =
        static_cast<int32_t>((idx / (shape.spatial * shape.head_dim)) % shape.frames);
    const int32_t h = static_cast<int32_t>(
        (idx / (static_cast<int64_t>(shape.spatial) * shape.head_dim * shape.frames)) %
        shape.heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(shape.spatial) * shape.head_dim * shape.frames *
               shape.heads));

    float num = 0.0F;
    float den = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        num += round_bf16(q_rot[bhtds_offset(shape, b, h, t, j, s)]) *
               round_bf16(hist_kv[frame_matrix_offset(shape, b, h, t, j, d)]);
        den += q[bhtds_offset(shape, b, h, t, j, s)] *
               hist_z[frame_vector_offset(shape, b, h, t, j)];
    }
    const float num_bf16 = round_bf16(num);
    const float den_bf16 = round_bf16(den);
    out[out_bhdn_offset(shape, b, h, d, t, s, false)] = num_bf16 / (den_bf16 + eps);
}

__global__ void phase_a_raw_combined_kernel(
    float* i_p_kv, float* a_t, float* i_p_z, float* b_z, const float* q_raw,
    const float* k_raw, const float* v_raw, const float* k_inv, const float* k_norm_weight,
    const float* rope_cos, const float* rope_sin, const float* beta, SanaWmGdnShape shape) {
    (void)q_raw;
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.heads * shape.frames *
                          shape.head_dim * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t row = static_cast<int32_t>((idx / shape.head_dim) % shape.head_dim);
    const int32_t t = static_cast<int32_t>((idx / (shape.head_dim * shape.head_dim)) %
                                          shape.frames);
    const int32_t h = static_cast<int32_t>(
        (idx / (shape.head_dim * shape.head_dim * shape.frames)) % shape.heads);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(shape.head_dim) * shape.head_dim * shape.frames *
               shape.heads));
    const float k_scale = rsqrtf(static_cast<float>(shape.head_dim)) *
                          rsqrtf(static_cast<float>(shape.spatial));

    float p_kv = 0.0F;
    float a_acc = 0.0F;
    float p_z = 0.0F;
    float b_acc = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
        const float k_rot_row = raw_rotated(k_raw, k_inv, k_norm_weight, rope_cos, rope_sin,
                                            shape, b, t, s, h, row, k_scale);
        const float k_rot_col = raw_rotated(k_raw, k_inv, k_norm_weight, rope_cos, rope_sin,
                                            shape, b, t, s, h, col, k_scale);
        const float k_row =
            raw_normed_relu(k_raw, k_inv, k_norm_weight, shape, b, t, s, h, row, k_scale);
        const float k_col =
            raw_normed_relu(k_raw, k_inv, k_norm_weight, shape, b, t, s, h, col, k_scale);
        const float v_col = v_raw[raw_bnc_offset(shape, b, t, s, h, col)];

        p_kv += round_bf16(k_rot_row) * round_bf16(beta_s * k_rot_col);
        a_acc += round_bf16(k_rot_row) * round_bf16(beta_s * v_col);
        p_z += round_bf16(k_row) * round_bf16(beta_s * k_col);
        if (col == 0) {
            b_acc += beta_s * k_row;
        }
    }

    const int64_t matrix = frame_matrix_offset(shape, b, h, t, row, col);
    i_p_kv[matrix] = round_bf16((row == col ? 1.0F : 0.0F) - p_kv);
    a_t[matrix] = round_bf16(a_acc);
    i_p_z[matrix] = round_bf16((row == col ? 1.0F : 0.0F) - p_z);
    if (col == 0) {
        b_z[frame_vector_offset(shape, b, h, t, row)] = b_acc;
    }
}

__global__ void phase_c_raw_combined_kernel(
    float* out, const float* hist_kv, const float* hist_z, const float* q_raw,
    const float* q_inv, const float* q_norm_weight, const float* rope_cos, const float* rope_sin,
    SanaWmGdnShape shape, float eps) {
    const int64_t total = static_cast<int64_t>(shape.batch) * shape.frames * shape.spatial *
                          shape.heads * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t s = static_cast<int32_t>((idx / (shape.head_dim * shape.heads)) %
                                          shape.spatial);
    const int32_t t = static_cast<int32_t>((idx / (static_cast<int64_t>(shape.head_dim) *
                                                  shape.heads * shape.spatial)) %
                                          shape.frames);
    const int32_t b = static_cast<int32_t>(
        idx / (static_cast<int64_t>(shape.head_dim) * shape.heads * shape.spatial *
               shape.frames));
    float num = 0.0F;
    float den = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        const float q_j =
            raw_normed_relu(q_raw, q_inv, q_norm_weight, shape, b, t, s, h, j, 1.0F);
        const float q_rot_j = raw_rotated(q_raw, q_inv, q_norm_weight, rope_cos, rope_sin,
                                          shape, b, t, s, h, j, 1.0F);
        num += round_bf16(q_rot_j) *
               round_bf16(hist_kv[frame_matrix_offset(shape, b, h, t, j, d)]);
        den += q_j * hist_z[frame_vector_offset(shape, b, h, t, j)];
    }
    const float num_bf16 = round_bf16(num);
    const float den_bf16 = round_bf16(den);
    out[raw_output_offset(shape, b, t, s, h, d)] = num_bf16 / (den_bf16 + eps);
}

__global__ void prepare_raw_phase_a_bf16_kernel(
    uint16_t* k_values, uint16_t* k_rot_values, uint16_t* beta_k_values,
    uint16_t* beta_k_rot_values, uint16_t* beta_v_values, const float* k_raw,
    const float* v_raw, const float* k_inv, const float* k_norm_weight, const float* rope_cos,
    const float* rope_sin, const float* beta, SanaWmGdnShape shape, int32_t padded_dim,
    int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % padded_dim);
    const int32_t s = static_cast<int32_t>((idx / padded_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * shape.spatial)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) *
                                                  shape.spatial * shape.heads));
    const int64_t out = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    if (d >= shape.head_dim) {
        k_values[out] = 0;
        k_rot_values[out] = 0;
        beta_k_values[out] = 0;
        beta_k_rot_values[out] = 0;
        beta_v_values[out] = 0;
        return;
    }
    const float k_scale = rsqrtf(static_cast<float>(shape.head_dim)) *
                          rsqrtf(static_cast<float>(shape.spatial));
    const float k_value =
        raw_normed_relu(k_raw, k_inv, k_norm_weight, shape, b, t, s, h, d, k_scale);
    const float k_rot_value =
        raw_rotated(k_raw, k_inv, k_norm_weight, rope_cos, rope_sin, shape, b, t, s, h, d,
                    k_scale);
    const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
    const float v_value = v_raw[raw_bnc_offset(shape, b, t, s, h, d)];
    k_values[out] = bf16_bits(k_value);
    k_rot_values[out] = bf16_bits(k_rot_value);
    beta_k_values[out] = bf16_bits(beta_s * k_value);
    beta_k_rot_values[out] = bf16_bits(beta_s * k_rot_value);
    beta_v_values[out] = bf16_bits(beta_s * v_value);
}

__global__ void finalize_raw_phase_a_bf16_kernel(
    uint16_t* i_p_kv, uint16_t* a_t, uint16_t* i_p_z, const float* p_kv, const float* a_acc,
    const float* p_z, SanaWmGdnShape shape, int32_t padded_dim, int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * padded_dim * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % padded_dim);
    const int32_t row = static_cast<int32_t>((idx / padded_dim) % padded_dim);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * padded_dim)) % shape.heads);
    const int32_t b =
        static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) * padded_dim *
                                    shape.heads));
    const int64_t frame_out = padded_frame_matrix_offset(shape, padded_dim, b, h, t, row, col);
    const int64_t scratch = ((static_cast<int64_t>(b) * shape.heads + h) * padded_dim + row) *
                                padded_dim +
                            col;
    const bool valid = row < shape.head_dim && col < shape.head_dim;
    const float diag = valid && row == col ? 1.0F : 0.0F;
    i_p_kv[frame_out] = bf16_bits(valid ? diag - p_kv[scratch] : 0.0F);
    a_t[frame_out] = bf16_bits(valid ? a_acc[scratch] : 0.0F);
    i_p_z[frame_out] = bf16_bits(valid ? diag - p_z[scratch] : 0.0F);
}

__global__ void raw_phase_a_bz_kernel(float* b_z, const float* k_raw, const float* k_inv,
                                      const float* k_norm_weight, const float* beta,
                                      SanaWmGdnShape shape, int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.head_dim * shape.heads));
    const float k_scale = rsqrtf(static_cast<float>(shape.head_dim)) *
                          rsqrtf(static_cast<float>(shape.spatial));
    float accum = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
        const float k_value =
            raw_normed_relu(k_raw, k_inv, k_norm_weight, shape, b, t, s, h, d, k_scale);
        accum += beta_s * k_value;
    }
    b_z[frame_vector_offset(shape, b, h, t, d)] = accum;
}

__global__ void prepare_phase_a_bf16_kernel(
    uint16_t* k_values, uint16_t* k_rot_values, uint16_t* beta_k_values,
    uint16_t* beta_k_rot_values, uint16_t* beta_v_values, const float* k, const float* v,
    const float* k_rot, const float* beta, SanaWmGdnShape shape, int32_t padded_dim,
    int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % padded_dim);
    const int32_t s = static_cast<int32_t>((idx / padded_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * shape.spatial)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) *
                                                  shape.spatial * shape.heads));
    const int64_t out = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    if (d >= shape.head_dim) {
        k_values[out] = 0;
        k_rot_values[out] = 0;
        beta_k_values[out] = 0;
        beta_k_rot_values[out] = 0;
        beta_v_values[out] = 0;
        return;
    }
    const float k_value = k[bhtds_offset(shape, b, h, t, d, s)];
    const float k_rot_value = k_rot[bhtds_offset(shape, b, h, t, d, s)];
    const float v_value = v[bhtds_offset(shape, b, h, t, d, s)];
    const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
    k_values[out] = bf16_bits(k_value);
    k_rot_values[out] = bf16_bits(k_rot_value);
    beta_k_values[out] = bf16_bits(beta_s * k_value);
    beta_k_rot_values[out] = bf16_bits(beta_s * k_rot_value);
    beta_v_values[out] = bf16_bits(beta_s * v_value);
}

__global__ void phase_a_bz_kernel(float* b_z, const float* k, const float* beta,
                                  SanaWmGdnShape shape, int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.head_dim * shape.heads));
    float accum = 0.0F;
    for (int32_t s = 0; s < shape.spatial; ++s) {
        const float beta_s = beta[bht1s_offset(shape, b, h, t, s)];
        accum += beta_s * k[bhtds_offset(shape, b, h, t, d, s)];
    }
    b_z[frame_vector_offset(shape, b, h, t, d)] = accum;
}

__global__ void convert_state_to_bf16_kernel(uint16_t* state_bf16, const float* state,
                                             SanaWmGdnShape shape, int32_t padded_dim) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * padded_dim * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    state_bf16[idx] = bf16_bits(state[idx]);
}

__global__ void convert_history_frame_to_bf16_kernel(uint16_t* state_bf16, const float* hist,
                                                     SanaWmGdnShape shape, int32_t padded_dim,
                                                     int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * padded_dim * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % padded_dim);
    const int32_t row = static_cast<int32_t>((idx / padded_dim) % padded_dim);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * padded_dim)) % shape.heads);
    const int32_t b =
        static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) * padded_dim *
                                    shape.heads));
    state_bf16[idx] =
        bf16_bits(hist[padded_frame_matrix_offset(shape, padded_dim, b, h, t, row, col)]);
}

__global__ void update_padded_state_kernel(float* next_state, const float* product,
                                           const uint16_t* a_t, float* hist,
                                           const float* decay, SanaWmGdnShape shape,
                                           int32_t padded_dim, int32_t source_frame,
                                           int32_t history_frame, bool add_history) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * padded_dim * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t col = static_cast<int32_t>(idx % padded_dim);
    const int32_t row = static_cast<int32_t>((idx / padded_dim) % padded_dim);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * padded_dim)) % shape.heads);
    const int32_t b =
        static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) * padded_dim *
                                    shape.heads));
    const float g = decay[bht11_offset(shape, b, h, source_frame)];
    const int64_t frame_in =
        padded_frame_matrix_offset(shape, padded_dim, b, h, source_frame, row, col);
    const float value = g * product[idx] + bf16_bits_to_float(a_t[frame_in]);
    next_state[idx] = value;
    const int64_t hist_offset =
        padded_frame_matrix_offset(shape, padded_dim, b, h, history_frame, row, col);
    hist[hist_offset] = add_history ? hist[hist_offset] + value : value;
}

__global__ void phase_b_z_bf16_kernel(float* next_state, const float* state, float* hist,
                                      const uint16_t* i_p_z, const float* b_z,
                                      const float* decay, SanaWmGdnShape shape,
                                      int32_t padded_dim, int32_t source_frame,
                                      int32_t history_frame, bool add_history) {
    extern __shared__ float shared[];
    const int64_t idx = static_cast<int64_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t row = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t h = static_cast<int32_t>((idx / shape.head_dim) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (shape.head_dim * shape.heads));
    float accum = 0.0F;
    if (tid < shape.head_dim) {
        accum =
            bf16_bits_to_float(i_p_z[padded_frame_matrix_offset(shape, padded_dim, b, h,
                                                                source_frame, row, tid)]) *
            state[state_z_offset(shape, b, h, tid)];
    }
    shared[tid] = accum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid != 0)
        return;
    const float g = decay[bht11_offset(shape, b, h, source_frame)];
    const float value = g * shared[0] + b_z[frame_vector_offset(shape, b, h, source_frame, row)];
    next_state[state_z_offset(shape, b, h, row)] = value;
    const int64_t hist_offset = frame_vector_offset(shape, b, h, history_frame, row);
    hist[hist_offset] = add_history ? hist[hist_offset] + value : value;
}

__global__ void prepare_raw_phase_c_qrot_bf16_kernel(
    uint16_t* q_rot_values, const float* q_raw, const float* q_inv, const float* q_norm_weight,
    const float* rope_cos, const float* rope_sin, SanaWmGdnShape shape, int32_t padded_dim,
    int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % padded_dim);
    const int32_t s = static_cast<int32_t>((idx / padded_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * shape.spatial)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) *
                                                  shape.spatial * shape.heads));
    const int64_t out = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    if (d >= shape.head_dim) {
        q_rot_values[out] = 0;
        return;
    }
    const float q_rot_value =
        raw_rotated(q_raw, q_inv, q_norm_weight, rope_cos, rope_sin, shape, b, t, s, h, d,
                    1.0F);
    q_rot_values[out] = bf16_bits(q_rot_value);
}

__global__ void phase_c_raw_cublas_output_kernel(
    float* out, const float* num, const float* hist_z, const float* q_raw, const float* q_inv,
    const float* q_norm_weight, SanaWmGdnShape shape, int32_t padded_dim, int32_t t, float eps) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t s = static_cast<int32_t>((idx / shape.head_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.spatial)) %
                                          shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(shape.head_dim) *
                                                  shape.spatial * shape.heads));
    float den = 0.0F;
    for (int32_t j = 0; j < shape.head_dim; ++j) {
        const float q_j =
            raw_normed_relu(q_raw, q_inv, q_norm_weight, shape, b, t, s, h, j, 1.0F);
        den += q_j * hist_z[frame_vector_offset(shape, b, h, t, j)];
    }
    const int64_t num_offset = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    const float num_bf16 = round_bf16(num[num_offset]);
    const float den_bf16 = round_bf16(den);
    out[raw_output_offset(shape, b, t, s, h, d)] = num_bf16 / (den_bf16 + eps);
}

__global__ void prepare_phase_c_qrot_bf16_kernel(uint16_t* q_rot_values, const float* q_rot,
                                                 SanaWmGdnShape shape, int32_t padded_dim,
                                                 int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * padded_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % padded_dim);
    const int32_t s = static_cast<int32_t>((idx / padded_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (padded_dim * shape.spatial)) % shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(padded_dim) *
                                                  shape.spatial * shape.heads));
    const int64_t out = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    if (d >= shape.head_dim) {
        q_rot_values[out] = 0;
        return;
    }
    q_rot_values[out] = bf16_bits(q_rot[bhtds_offset(shape, b, h, t, d, s)]);
}

__global__ void phase_c_den_bf16_kernel(uint16_t* den_values, const float* hist_z,
                                        const float* q, SanaWmGdnShape shape, int32_t t) {
    extern __shared__ float shared[];
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int32_t s = static_cast<int32_t>(token % shape.spatial);
    const int32_t h = static_cast<int32_t>((token / shape.spatial) % shape.heads);
    const int32_t b =
        static_cast<int32_t>(token / (static_cast<int64_t>(shape.spatial) * shape.heads));
    float value = 0.0F;
    if (tid < shape.head_dim) {
        value = q[bhtds_offset(shape, b, h, t, tid, s)] *
                hist_z[frame_vector_offset(shape, b, h, t, tid)];
    }
    shared[tid] = value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        den_values[token] = phase_c_den_bf16_bits(shared[0], h, t, s);
}

__global__ void copy_phase_c_num_debug_kernel(float* out, const float* num,
                                              SanaWmGdnShape shape, int32_t padded_dim,
                                              int32_t t, bool camera_corrections) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t s = static_cast<int32_t>((idx / shape.head_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.spatial)) %
                                          shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(shape.head_dim) *
                                                  shape.spatial * shape.heads));
    const float raw = num[frame_scratch_offset(shape, padded_dim, b, h, s, d)];
    const uint16_t bits =
        camera_corrections ? camera_phase_c_num_bf16_bits(raw, h, t, s, d) : bf16_bits(raw);
    const float value = bf16_bits_to_float(bits);
    out[out_bhdn_offset(shape, b, h, d, t, s, false)] = value;
}

__global__ void copy_phase_c_den_debug_kernel(float* out, const uint16_t* den_values,
                                              SanaWmGdnShape shape, int32_t t) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t s = static_cast<int32_t>((idx / shape.head_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.spatial)) %
                                          shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(shape.head_dim) *
                                                  shape.spatial * shape.heads));
    const float value =
        bf16_bits_to_float(den_values[(static_cast<int64_t>(b) * shape.heads + h) *
                                          shape.spatial +
                                      s]);
    out[out_bhdn_offset(shape, b, h, d, t, s, false)] = value;
}

__global__ void phase_c_cublas_output_kernel(float* out, const float* num,
                                             const uint16_t* den_values,
                                             SanaWmGdnShape shape, int32_t padded_dim,
                                             int32_t t, float eps) {
    const int64_t total =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int32_t d = static_cast<int32_t>(idx % shape.head_dim);
    const int32_t s = static_cast<int32_t>((idx / shape.head_dim) % shape.spatial);
    const int32_t h = static_cast<int32_t>((idx / (shape.head_dim * shape.spatial)) %
                                          shape.heads);
    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(shape.head_dim) *
                                                  shape.spatial * shape.heads));
    const int64_t num_offset = frame_scratch_offset(shape, padded_dim, b, h, s, d);
    const float num_bf16 = round_bf16(num[num_offset]);
    const float den_bf16 =
        bf16_bits_to_float(den_values[(static_cast<int64_t>(b) * shape.heads + h) *
                                          shape.spatial +
                                      s]);
    out[out_bhdn_offset(shape, b, h, d, t, s, false)] = num_bf16 / (den_bf16 + eps);
}

int32_t launch_main_scan(SanaWmGdnShape shape, const void* const* inputs, void* const* outputs,
                         void* workspace, bool reverse, cudaStream_t stream) {
    clear_stale_cuda_error("main", reverse, shape);
    const auto* q = static_cast<const float*>(inputs[0]);
    const auto* k = static_cast<const float*>(inputs[1]);
    const auto* v = static_cast<const float*>(inputs[2]);
    const auto* q_rot = static_cast<const float*>(inputs[3]);
    const auto* k_rot = static_cast<const float*>(inputs[4]);
    const auto* beta = static_cast<const float*>(inputs[5]);
    const auto* decay = static_cast<const float*>(inputs[6]);
    auto* num = static_cast<float*>(outputs[0]);
    auto* den = static_cast<float*>(outputs[1]);

    auto* ptr = static_cast<char*>(workspace);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    float* state_kv = workspace_take(ptr, bh * shape.head_dim * shape.head_dim);
    float* state_z = workspace_take(ptr, bh * shape.head_dim);
    float* delta_v = workspace_take(ptr, bh * shape.head_dim * shape.spatial);
    float* delta_z = workspace_take(ptr, bh * shape.spatial);
    cudaMemsetAsync(state_kv, 0, bh * shape.head_dim * shape.head_dim * sizeof(float), stream);
    cudaMemsetAsync(state_z, 0, bh * shape.head_dim * sizeof(float), stream);

    constexpr int32_t kThreads = 256;
    const int64_t kv_elems = static_cast<int64_t>(bh) * shape.head_dim * shape.head_dim;
    const int64_t dvs_elems = static_cast<int64_t>(bh) * shape.head_dim * shape.spatial;
    const int64_t z_elems = static_cast<int64_t>(bh) * shape.head_dim;
    const int64_t s_elems = static_cast<int64_t>(bh) * shape.spatial;
    for (int32_t t = 0; t < shape.frames; ++t) {
        decay_state_kernel<<<static_cast<uint32_t>((kv_elems + kThreads - 1) / kThreads), kThreads,
                             0, stream>>>(state_kv, state_z, decay, shape, t, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "decay_state", "main", reverse, shape))
            return 1;
        delta_v_kernel<<<static_cast<uint32_t>((dvs_elems + kThreads - 1) / kThreads), kThreads, 0,
                         stream>>>(delta_v, state_kv, v, k_rot, beta, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "delta_v", "main", reverse, shape))
            return 1;
        update_kv_kernel<<<static_cast<uint32_t>((kv_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(state_kv, delta_v, k_rot, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "update_kv", "main", reverse, shape))
            return 1;
        delta_z_kernel<<<static_cast<uint32_t>((s_elems + kThreads - 1) / kThreads), kThreads, 0,
                         stream>>>(delta_z, state_z, k, beta, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "delta_z", "main", reverse, shape))
            return 1;
        update_z_kernel<<<static_cast<uint32_t>((z_elems + kThreads - 1) / kThreads), kThreads, 0,
                          stream>>>(state_z, delta_z, k, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "update_z", "main", reverse, shape))
            return 1;
        write_num_kernel<<<static_cast<uint32_t>((dvs_elems + kThreads - 1) / kThreads), kThreads,
                           0, stream>>>(num, state_kv, q_rot, shape, t, reverse);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "write_num", "main", reverse, shape))
            return 1;
        write_den_kernel<<<static_cast<uint32_t>((s_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(den, state_z, q, shape, t, reverse);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "write_den", "main", reverse, shape))
            return 1;
    }
    return 0;
}

int32_t launch_camera_scan(SanaWmGdnShape shape, const void* const* inputs, void* const* outputs,
                           void* workspace, bool reverse, cudaStream_t stream) {
    clear_stale_cuda_error("camera", reverse, shape);
    const auto* q_rot = static_cast<const float*>(inputs[0]);
    const auto* k_rot = static_cast<const float*>(inputs[1]);
    const auto* v = static_cast<const float*>(inputs[2]);
    const auto* beta = static_cast<const float*>(inputs[3]);
    const auto* decay = static_cast<const float*>(inputs[4]);
    auto* out = static_cast<float*>(outputs[0]);

    auto* ptr = static_cast<char*>(workspace);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    float* state_kv = workspace_take(ptr, bh * shape.head_dim * shape.head_dim);
    float* delta_v = workspace_take(ptr, bh * shape.head_dim * shape.spatial);
    cudaMemsetAsync(state_kv, 0, bh * shape.head_dim * shape.head_dim * sizeof(float), stream);

    constexpr int32_t kThreads = 256;
    const int64_t kv_elems = static_cast<int64_t>(bh) * shape.head_dim * shape.head_dim;
    const int64_t dvs_elems = static_cast<int64_t>(bh) * shape.head_dim * shape.spatial;
    for (int32_t t = 0; t < shape.frames; ++t) {
        decay_state_kernel<<<static_cast<uint32_t>((kv_elems + kThreads - 1) / kThreads), kThreads,
                             0, stream>>>(state_kv, nullptr, decay, shape, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "decay_state", "camera", reverse,
                                     shape))
            return 1;
        delta_v_kernel<<<static_cast<uint32_t>((dvs_elems + kThreads - 1) / kThreads), kThreads, 0,
                         stream>>>(delta_v, state_kv, v, k_rot, beta, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "delta_v", "camera", reverse, shape))
            return 1;
        update_kv_kernel<<<static_cast<uint32_t>((kv_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(state_kv, delta_v, k_rot, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "update_kv", "camera", reverse, shape))
            return 1;
        write_num_kernel<<<static_cast<uint32_t>((dvs_elems + kThreads - 1) / kThreads), kThreads,
                           0, stream>>>(out, state_kv, q_rot, shape, t, reverse);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "write_num", "camera", reverse, shape))
            return 1;
    }
    return 0;
}

int32_t launch_camera_combined_cublas(SanaWmGdnShape shape, const void* const* inputs,
                                      void* const* outputs, void* workspace,
                                      cudaStream_t stream) {
    clear_stale_cuda_error("camera_combined_cublas", false, shape);
    const auto* q = static_cast<const float*>(inputs[0]);
    const auto* k = static_cast<const float*>(inputs[1]);
    const auto* v = static_cast<const float*>(inputs[2]);
    const auto* beta = static_cast<const float*>(inputs[3]);
    const auto* decay = static_cast<const float*>(inputs[4]);
    auto* out = static_cast<float*>(outputs[0]);

    const int32_t padded_dim = next_power_of_two(shape.head_dim);
    auto* ptr = static_cast<char*>(workspace);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    const std::size_t frame_scratch =
        bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
    const std::size_t frame_matrices =
        bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
    const std::size_t state_matrices = bh * static_cast<std::size_t>(padded_dim) * padded_dim;

    uint16_t* k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_v_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* q_values = workspace_take_bf16(ptr, frame_scratch);
    float* gemm_a = workspace_take(ptr, state_matrices);
    float* gemm_b = workspace_take(ptr, state_matrices);
    uint16_t* i_p_kv = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* a_t = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* i_p_z_unused = workspace_take_bf16(ptr, frame_matrices);
    float* hist_kv = workspace_take(ptr, frame_matrices);
    float* state_kv_a = workspace_take(ptr, state_matrices);
    float* state_kv_b = workspace_take(ptr, state_matrices);
    uint16_t* state_kv_bf16 = workspace_take_bf16(ptr, state_matrices);
    float* phase_c_num = workspace_take(ptr, frame_scratch);

    constexpr int32_t kThreads = 256;
    const int64_t scratch_elems = static_cast<int64_t>(frame_scratch);
    const int64_t matrix_elems_per_frame = static_cast<int64_t>(state_matrices);
    const int64_t output_elems_per_frame =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;

    cublasHandle_t handle = nullptr;
    if (report_cublas_error(cublasCreate(&handle), "create", "camera_combined_cublas", shape))
        return 1;
    cublasSetStream(handle, stream);

    const long long scratch_stride =
        static_cast<long long>(shape.spatial) * static_cast<long long>(padded_dim);
    const long long matrix_stride =
        static_cast<long long>(padded_dim) * static_cast<long long>(padded_dim);
    const long long frame_matrix_stride =
        static_cast<long long>(shape.frames) * matrix_stride;

    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(k_values, k_rot_values, beta_k_values, beta_k_rot_values,
                       beta_v_values, k, v, k, beta, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_a_bf16",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_k_rot_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_pkv", "camera_combined_cublas", shape) ||
            !cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_v_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_b, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_a", "camera_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        finalize_raw_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(i_p_kv, a_t, i_p_z_unused, gemm_a, gemm_b, gemm_a,
                                   shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "finalize_phase_a_bf16",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    float* state_kv = state_kv_a;
    float* next_kv = state_kv_b;
    for (int32_t t = 0; t < shape.frames; ++t) {
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_fwd",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(t) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride, gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_b_fwd", "camera_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_fwd_update",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp = state_kv;
        state_kv = next_kv;
        next_kv = tmp;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    state_kv = state_kv_a;
    next_kv = state_kv_b;
    for (int32_t f_src = shape.frames - 1; f_src >= 1; --f_src) {
        const int32_t f_dst = f_src - 1;
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_rev",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(f_src) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride, gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_b_rev", "camera_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_rev_update",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp = state_kv;
        state_kv = next_kv;
        next_kv = tmp;
    }

    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_phase_c_qrot_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(q_values, q, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_c_q",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        convert_history_frame_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, hist_kv, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "hist_to_bf16_phase_c",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, shape.spatial, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, q_values, padded_dim,
                scratch_stride, phase_c_num, padded_dim, scratch_stride,
                static_cast<int32_t>(bh), "phase_c_num", "camera_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        copy_phase_c_num_debug_kernel<<<
            static_cast<uint32_t>((output_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(out, phase_c_num, shape, padded_dim, t, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_num_copy",
                                     "camera_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }

    cublasDestroy(handle);
    return 0;
}

int32_t launch_main_combined_cublas(SanaWmGdnShape shape, const void* const* inputs,
                                    void* const* outputs, void* workspace, float eps,
                                    cudaStream_t stream) {
    clear_stale_cuda_error("main_combined_cublas", false, shape);
    const auto* q = static_cast<const float*>(inputs[0]);
    const auto* k = static_cast<const float*>(inputs[1]);
    const auto* v = static_cast<const float*>(inputs[2]);
    const auto* q_rot = static_cast<const float*>(inputs[3]);
    const auto* k_rot = static_cast<const float*>(inputs[4]);
    const auto* beta = static_cast<const float*>(inputs[5]);
    const auto* decay = static_cast<const float*>(inputs[6]);
    auto* out = static_cast<float*>(outputs[0]);

    const int32_t padded_dim = next_power_of_two(shape.head_dim);
    auto* ptr = static_cast<char*>(workspace);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    const std::size_t frame_scratch =
        bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
    const std::size_t frame_matrices =
        bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
    const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
    const std::size_t state_matrices = bh * static_cast<std::size_t>(padded_dim) * padded_dim;
    const std::size_t state_vectors = bh * shape.head_dim;

    uint16_t* k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_v_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* q_rot_values = workspace_take_bf16(ptr, frame_scratch);
    float* gemm_a = workspace_take(ptr, state_matrices);
    float* gemm_b = workspace_take(ptr, state_matrices);
    float* gemm_c = workspace_take(ptr, state_matrices);
    uint16_t* i_p_kv = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* a_t = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* i_p_z = workspace_take_bf16(ptr, frame_matrices);
    float* b_z = workspace_take(ptr, frame_vectors);
    float* hist_kv = workspace_take(ptr, frame_matrices);
    float* hist_z = workspace_take(ptr, frame_vectors);
    float* state_kv_a = workspace_take(ptr, state_matrices);
    float* state_kv_b = workspace_take(ptr, state_matrices);
    uint16_t* state_kv_bf16 = workspace_take_bf16(ptr, state_matrices);
    float* state_z_a = workspace_take(ptr, state_vectors);
    float* state_z_b = workspace_take(ptr, state_vectors);
    float* phase_c_num = workspace_take(ptr, frame_scratch);
    uint16_t* phase_c_den = workspace_take_bf16(ptr, bh * static_cast<std::size_t>(shape.spatial));

    constexpr int32_t kThreads = 256;
    const int64_t scratch_elems = static_cast<int64_t>(frame_scratch);
    const int64_t matrix_elems_per_frame = static_cast<int64_t>(state_matrices);
    const int64_t vector_elems_per_frame = static_cast<int64_t>(state_vectors);
    const int64_t output_elems_per_frame =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;
    const int64_t output_elems = output_elems_per_frame * shape.frames;

    cublasHandle_t handle = nullptr;
    if (report_cublas_error(cublasCreate(&handle), "create", "main_combined_cublas",
                            shape))
        return 1;
    cublasSetStream(handle, stream);

    const long long scratch_stride =
        static_cast<long long>(shape.spatial) * static_cast<long long>(padded_dim);
    const long long matrix_stride =
        static_cast<long long>(padded_dim) * static_cast<long long>(padded_dim);
    const long long frame_matrix_stride =
        static_cast<long long>(shape.frames) * matrix_stride;

    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(k_values, k_rot_values, beta_k_values, beta_k_rot_values,
                       beta_v_values, k, v, k_rot, beta, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_a_bf16",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_k_rot_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_pkv", "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_v_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_b, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_a", "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_k_values, padded_dim, scratch_stride, k_values, padded_dim, scratch_stride,
                gemm_c, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_a_pz",
                "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        finalize_raw_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(i_p_kv, a_t, i_p_z, gemm_a, gemm_b, gemm_c, shape,
                                   padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "finalize_phase_a_bf16",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_a_bz_kernel<<<
            static_cast<uint32_t>((vector_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(b_z, k, beta, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_a_bz",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }
    if (copy_bf16_debug_output_if_requested("phase_a_i_p_kv", out, i_p_kv,
                                            static_cast<int64_t>(frame_matrices),
                                            output_elems, shape, stream) ||
        copy_bf16_debug_output_if_requested("phase_a_a", out, a_t,
                                            static_cast<int64_t>(frame_matrices),
                                            output_elems, shape, stream) ||
        copy_bf16_debug_output_if_requested("phase_a_i_p_z", out, i_p_z,
                                            static_cast<int64_t>(frame_matrices),
                                            output_elems, shape, stream) ||
        copy_float_debug_output_if_requested("phase_a_b_z", out, b_z,
                                             static_cast<int64_t>(frame_vectors),
                                             output_elems, shape, stream)) {
        const bool failed = report_cuda_launch_error(cudaPeekAtLastError(), "debug_output",
                                                     "main_combined_cublas", false, shape);
        cublasDestroy(handle);
        return failed ? 1 : 0;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    float* state_kv = state_kv_a;
    float* next_kv = state_kv_b;
    float* state_z = state_z_a;
    float* next_z = state_z_b;
    for (int32_t t = 0; t < shape.frames; ++t) {
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_fwd",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(t) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride,
                gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_b_fwd",
                "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_fwd_update",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_b_z_bf16_kernel<<<static_cast<uint32_t>(vector_elems_per_frame),
                                static_cast<uint32_t>(padded_dim),
                                static_cast<std::size_t>(padded_dim) * sizeof(float),
                                stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay, shape,
                                          padded_dim, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_fwd",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    state_kv = state_kv_a;
    next_kv = state_kv_b;
    state_z = state_z_a;
    next_z = state_z_b;
    for (int32_t f_src = shape.frames - 1; f_src >= 1; --f_src) {
        const int32_t f_dst = f_src - 1;
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_rev",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(f_src) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride,
                gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_b_rev",
                "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_rev_update",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_b_z_bf16_kernel<<<static_cast<uint32_t>(vector_elems_per_frame),
                                static_cast<uint32_t>(padded_dim),
                                static_cast<std::size_t>(padded_dim) * sizeof(float),
                                stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay, shape,
                                          padded_dim, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_rev",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }
    if (copy_float_debug_output_if_requested("phase_b_hist_kv", out, hist_kv,
                                             static_cast<int64_t>(frame_matrices),
                                             output_elems, shape, stream) ||
        copy_float_debug_output_if_requested("phase_b_hist_z", out, hist_z,
                                             static_cast<int64_t>(frame_vectors),
                                             output_elems, shape, stream)) {
        const bool failed = report_cuda_launch_error(cudaPeekAtLastError(), "debug_output",
                                                     "main_combined_cublas", false, shape);
        cublasDestroy(handle);
        return failed ? 1 : 0;
    }

    const bool debug_phase_c_num = debug_output_requested("phase_c_num");
    const bool debug_phase_c_den = debug_output_requested("phase_c_den");
    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_phase_c_qrot_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(q_rot_values, q_rot, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_c_qrot",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        convert_history_frame_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, hist_kv, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "hist_to_bf16_phase_c",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, shape.spatial, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, q_rot_values, padded_dim,
                scratch_stride, phase_c_num, padded_dim, scratch_stride,
                static_cast<int32_t>(bh), "phase_c_num", "main_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (debug_phase_c_num) {
            copy_phase_c_num_debug_kernel<<<
                static_cast<uint32_t>((output_elems_per_frame + kThreads - 1) / kThreads),
                kThreads, 0, stream>>>(out, phase_c_num, shape, padded_dim, t, false);
            if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_num_debug",
                                         "main_combined_cublas", false, shape)) {
                cublasDestroy(handle);
                return 1;
            }
            continue;
        }
        phase_c_den_bf16_kernel<<<static_cast<uint32_t>(bh * shape.spatial),
                                  static_cast<uint32_t>(padded_dim),
                                  static_cast<std::size_t>(padded_dim) * sizeof(float),
                                  stream>>>(phase_c_den, hist_z, q, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_den",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (debug_phase_c_den) {
            copy_phase_c_den_debug_kernel<<<
                static_cast<uint32_t>((output_elems_per_frame + kThreads - 1) / kThreads),
                kThreads, 0, stream>>>(out, phase_c_den, shape, t);
            if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_den_debug",
                                         "main_combined_cublas", false, shape)) {
                cublasDestroy(handle);
                return 1;
            }
            continue;
        }
        phase_c_cublas_output_kernel<<<
            static_cast<uint32_t>((output_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(out, phase_c_num, phase_c_den, shape, padded_dim, t, eps);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_output",
                                     "main_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }

    cublasDestroy(handle);
    return 0;
}

int32_t launch_main_combined(SanaWmGdnShape shape, const void* const* inputs,
                             void* const* outputs, void* workspace, float eps,
                             cudaStream_t stream) {
    if (use_main_combined_cublas()) {
        return launch_main_combined_cublas(shape, inputs, outputs, workspace, eps, stream);
    }
    clear_stale_cuda_error("main_combined", false, shape);
    const auto* q = static_cast<const float*>(inputs[0]);
    const auto* k = static_cast<const float*>(inputs[1]);
    const auto* v = static_cast<const float*>(inputs[2]);
    const auto* q_rot = static_cast<const float*>(inputs[3]);
    const auto* k_rot = static_cast<const float*>(inputs[4]);
    const auto* beta = static_cast<const float*>(inputs[5]);
    const auto* decay = static_cast<const float*>(inputs[6]);
    auto* out = static_cast<float*>(outputs[0]);

    auto* ptr = static_cast<char*>(workspace);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    const std::size_t frame_matrices =
        bh * shape.frames * shape.head_dim * shape.head_dim;
    const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
    const std::size_t state_matrices = bh * shape.head_dim * shape.head_dim;
    const std::size_t state_vectors = bh * shape.head_dim;
    float* i_p_kv = workspace_take(ptr, frame_matrices);
    float* a_t = workspace_take(ptr, frame_matrices);
    float* i_p_z = workspace_take(ptr, frame_matrices);
    float* b_z = workspace_take(ptr, frame_vectors);
    float* hist_kv = workspace_take(ptr, frame_matrices);
    float* hist_z = workspace_take(ptr, frame_vectors);
    float* state_kv_a = workspace_take(ptr, state_matrices);
    float* state_kv_b = workspace_take(ptr, state_matrices);
    float* state_z_a = workspace_take(ptr, state_vectors);
    float* state_z_b = workspace_take(ptr, state_vectors);

    constexpr int32_t kThreads = 256;
    const int64_t matrix_elems = static_cast<int64_t>(frame_matrices);
    const int64_t state_matrix_elems = static_cast<int64_t>(state_matrices);
    const int64_t state_vector_elems = static_cast<int64_t>(state_vectors);
    const int64_t output_elems = static_cast<int64_t>(shape.batch) * shape.heads *
                                 shape.frames * shape.head_dim * shape.spatial;
    constexpr bool float_accum = false;

    phase_a_combined_kernel<<<static_cast<uint32_t>((matrix_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(i_p_kv, a_t, i_p_z, b_z, k, v, k_rot,
                                                      beta, shape, float_accum);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_a", "main_combined", false,
                                 shape))
        return 1;

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    float* state_kv = state_kv_a;
    float* next_kv = state_kv_b;
    float* state_z = state_z_a;
    float* next_z = state_z_b;
    for (int32_t t = 0; t < shape.frames; ++t) {
        phase_b_kv_kernel<<<static_cast<uint32_t>((state_matrix_elems + kThreads - 1) /
                                                  kThreads),
                            kThreads, 0, stream>>>(next_kv, state_kv, hist_kv, i_p_kv, a_t,
                                                    decay, shape, t, t, false, float_accum);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_fwd",
                                     "main_combined", false, shape))
            return 1;
        phase_b_z_kernel<<<static_cast<uint32_t>((state_vector_elems + kThreads - 1) /
                                                 kThreads),
                           kThreads, 0, stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay,
                                                   shape, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_fwd",
                                     "main_combined", false, shape))
            return 1;
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    state_kv = state_kv_a;
    next_kv = state_kv_b;
    state_z = state_z_a;
    next_z = state_z_b;
    for (int32_t f_src = shape.frames - 1; f_src >= 1; --f_src) {
        const int32_t f_dst = f_src - 1;
        phase_b_kv_kernel<<<static_cast<uint32_t>((state_matrix_elems + kThreads - 1) /
                                                  kThreads),
                            kThreads, 0, stream>>>(next_kv, state_kv, hist_kv, i_p_kv, a_t,
                                                    decay, shape, f_src, f_dst, true, float_accum);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_rev",
                                     "main_combined", false, shape))
            return 1;
        phase_b_z_kernel<<<static_cast<uint32_t>((state_vector_elems + kThreads - 1) /
                                                 kThreads),
                           kThreads, 0, stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay,
                                                   shape, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_rev",
                                     "main_combined", false, shape))
            return 1;
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    phase_c_combined_kernel<<<static_cast<uint32_t>((output_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(out, hist_kv, hist_z, q, q_rot, shape, eps);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c", "main_combined", false,
                                 shape))
        return 1;
    return 0;
}

int32_t launch_main_raw_combined_cublas(SanaWmGdnShape shape, const void* const* inputs,
                                        void* const* outputs, void* workspace, float eps,
                                        float norm_eps, cudaStream_t stream) {
    clear_stale_cuda_error("main_raw_combined_cublas", false, shape);
    const auto* q_raw = static_cast<const float*>(inputs[0]);
    const auto* k_raw = static_cast<const float*>(inputs[1]);
    const auto* v_raw = static_cast<const float*>(inputs[2]);
    const auto* q_norm_weight = static_cast<const float*>(inputs[3]);
    const auto* k_norm_weight = static_cast<const float*>(inputs[4]);
    const auto* rope_cos = static_cast<const float*>(inputs[5]);
    const auto* rope_sin = static_cast<const float*>(inputs[6]);
    const auto* beta = static_cast<const float*>(inputs[7]);
    const auto* decay = static_cast<const float*>(inputs[8]);
    auto* out = static_cast<float*>(outputs[0]);

    const int32_t padded_dim = next_power_of_two(shape.head_dim);
    auto* ptr = static_cast<char*>(workspace);
    const std::size_t tokens =
        static_cast<std::size_t>(shape.batch) * shape.frames * shape.spatial;
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    const std::size_t frame_scratch =
        bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
    const std::size_t frame_matrices =
        bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
    const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
    const std::size_t state_matrices = bh * static_cast<std::size_t>(padded_dim) * padded_dim;
    const std::size_t state_vectors = bh * shape.head_dim;

    float* q_inv = workspace_take(ptr, tokens);
    float* k_inv = workspace_take(ptr, tokens);
    uint16_t* k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_k_rot_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* beta_v_values = workspace_take_bf16(ptr, frame_scratch);
    uint16_t* q_rot_values = workspace_take_bf16(ptr, frame_scratch);
    float* gemm_a = workspace_take(ptr, state_matrices);
    float* gemm_b = workspace_take(ptr, state_matrices);
    float* gemm_c = workspace_take(ptr, state_matrices);
    uint16_t* i_p_kv = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* a_t = workspace_take_bf16(ptr, frame_matrices);
    uint16_t* i_p_z = workspace_take_bf16(ptr, frame_matrices);
    float* b_z = workspace_take(ptr, frame_vectors);
    float* hist_kv = workspace_take(ptr, frame_matrices);
    float* hist_z = workspace_take(ptr, frame_vectors);
    float* state_kv_a = workspace_take(ptr, state_matrices);
    float* state_kv_b = workspace_take(ptr, state_matrices);
    uint16_t* state_kv_bf16 = workspace_take_bf16(ptr, state_matrices);
    float* state_z_a = workspace_take(ptr, state_vectors);
    float* state_z_b = workspace_take(ptr, state_vectors);
    float* phase_c_num = workspace_take(ptr, frame_scratch);

    constexpr int32_t kThreads = 256;
    const int64_t token_elems = static_cast<int64_t>(tokens);
    const int64_t scratch_elems = static_cast<int64_t>(frame_scratch);
    const int64_t matrix_elems_per_frame = static_cast<int64_t>(state_matrices);
    const int64_t vector_elems_per_frame = static_cast<int64_t>(state_vectors);
    const int64_t output_elems_per_frame =
        static_cast<int64_t>(shape.batch) * shape.heads * shape.spatial * shape.head_dim;

    raw_qk_inv_rms_kernel<<<static_cast<uint32_t>((token_elems + kThreads - 1) / kThreads),
                            kThreads, 0, stream>>>(q_inv, k_inv, q_raw, k_raw, shape,
                                                    norm_eps);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "raw_qk_inv_rms",
                                 "main_raw_combined_cublas", false, shape))
        return 1;

    cublasHandle_t handle = nullptr;
    if (report_cublas_error(cublasCreate(&handle), "create", "main_raw_combined_cublas",
                            shape))
        return 1;
    cublasSetStream(handle, stream);

    const long long scratch_stride =
        static_cast<long long>(shape.spatial) * static_cast<long long>(padded_dim);
    const long long matrix_stride =
        static_cast<long long>(padded_dim) * static_cast<long long>(padded_dim);
    const long long frame_matrix_stride =
        static_cast<long long>(shape.frames) * matrix_stride;

    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_raw_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(k_values, k_rot_values, beta_k_values, beta_k_rot_values, beta_v_values,
                       k_raw, v_raw, k_inv, k_norm_weight, rope_cos, rope_sin, beta, shape,
                       padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_a_bf16",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_k_rot_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_pkv", "main_raw_combined_cublas", shape) ||
            !cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_v_values, padded_dim, scratch_stride, k_rot_values, padded_dim,
                scratch_stride, gemm_b, padded_dim, matrix_stride, static_cast<int32_t>(bh),
                "phase_a_a", "main_raw_combined_cublas", shape) ||
            !cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, padded_dim, padded_dim, shape.spatial,
                beta_k_values, padded_dim, scratch_stride, k_values, padded_dim, scratch_stride,
                gemm_c, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_a_pz",
                "main_raw_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        finalize_raw_phase_a_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(i_p_kv, a_t, i_p_z, gemm_a, gemm_b, gemm_c, shape,
                                   padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "finalize_phase_a_bf16",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        raw_phase_a_bz_kernel<<<
            static_cast<uint32_t>((vector_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(b_z, k_raw, k_inv, k_norm_weight, beta, shape, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_a_bz",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    float* state_kv = state_kv_a;
    float* next_kv = state_kv_b;
    float* state_z = state_z_a;
    float* next_z = state_z_b;
    for (int32_t t = 0; t < shape.frames; ++t) {
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_fwd",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(t) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride,
                gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_b_fwd",
                "main_raw_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_fwd_update",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_b_z_bf16_kernel<<<static_cast<uint32_t>(vector_elems_per_frame),
                                static_cast<uint32_t>(padded_dim),
                                static_cast<std::size_t>(padded_dim) * sizeof(float),
                                stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay, shape,
                                          padded_dim, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_fwd",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    state_kv = state_kv_a;
    next_kv = state_kv_b;
    state_z = state_z_a;
    next_z = state_z_b;
    for (int32_t f_src = shape.frames - 1; f_src >= 1; --f_src) {
        const int32_t f_dst = f_src - 1;
        convert_state_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, state_kv, shape, padded_dim);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "state_to_bf16_rev",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        const auto* i_p_frame = i_p_kv + static_cast<std::size_t>(f_src) * matrix_stride;
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, padded_dim, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, i_p_frame, padded_dim,
                frame_matrix_stride,
                gemm_a, padded_dim, matrix_stride, static_cast<int32_t>(bh), "phase_b_rev",
                "main_raw_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        update_padded_state_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(next_kv, gemm_a, a_t, hist_kv, decay, shape,
                                   padded_dim, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_kv_rev_update",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_b_z_bf16_kernel<<<static_cast<uint32_t>(vector_elems_per_frame),
                                static_cast<uint32_t>(padded_dim),
                                static_cast<std::size_t>(padded_dim) * sizeof(float),
                                stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay, shape,
                                          padded_dim, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_z_rev",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    for (int32_t t = 0; t < shape.frames; ++t) {
        prepare_raw_phase_c_qrot_bf16_kernel<<<
            static_cast<uint32_t>((scratch_elems + kThreads - 1) / kThreads), kThreads, 0,
            stream>>>(q_rot_values, q_raw, q_inv, q_norm_weight, rope_cos, rope_sin, shape,
                       padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "prepare_phase_c_qrot",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        convert_history_frame_to_bf16_kernel<<<
            static_cast<uint32_t>((matrix_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(state_kv_bf16, hist_kv, shape, padded_dim, t);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "hist_to_bf16_phase_c",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
        if (!cublas_bf16_gemm_strided_batched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, shape.spatial, padded_dim,
                state_kv_bf16, padded_dim, matrix_stride, q_rot_values, padded_dim,
                scratch_stride, phase_c_num, padded_dim, scratch_stride,
                static_cast<int32_t>(bh), "phase_c_num", "main_raw_combined_cublas", shape)) {
            cublasDestroy(handle);
            return 1;
        }
        phase_c_raw_cublas_output_kernel<<<
            static_cast<uint32_t>((output_elems_per_frame + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(out, phase_c_num, hist_z, q_raw, q_inv, q_norm_weight, shape,
                                   padded_dim, t, eps);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_output",
                                     "main_raw_combined_cublas", false, shape)) {
            cublasDestroy(handle);
            return 1;
        }
    }

    cublasDestroy(handle);
    return 0;
}

int32_t launch_main_raw_combined(SanaWmGdnShape shape, const void* const* inputs,
                                 void* const* outputs, void* workspace, float eps,
                                 float norm_eps, cudaStream_t stream) {
    const char* use_cublas = std::getenv("TRTMC_SANA_WM_GDN_CUBLAS");
    if (use_cublas != nullptr && std::strcmp(use_cublas, "0") != 0 &&
        std::strcmp(use_cublas, "false") != 0 && std::strcmp(use_cublas, "False") != 0) {
        return launch_main_raw_combined_cublas(shape, inputs, outputs, workspace, eps, norm_eps,
                                               stream);
    }
    clear_stale_cuda_error("main_raw_combined", false, shape);
    const auto* q_raw = static_cast<const float*>(inputs[0]);
    const auto* k_raw = static_cast<const float*>(inputs[1]);
    const auto* v_raw = static_cast<const float*>(inputs[2]);
    const auto* q_norm_weight = static_cast<const float*>(inputs[3]);
    const auto* k_norm_weight = static_cast<const float*>(inputs[4]);
    const auto* rope_cos = static_cast<const float*>(inputs[5]);
    const auto* rope_sin = static_cast<const float*>(inputs[6]);
    const auto* beta = static_cast<const float*>(inputs[7]);
    const auto* decay = static_cast<const float*>(inputs[8]);
    auto* out = static_cast<float*>(outputs[0]);

    auto* ptr = static_cast<char*>(workspace);
    const std::size_t tokens =
        static_cast<std::size_t>(shape.batch) * shape.frames * shape.spatial;
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    const std::size_t frame_matrices =
        bh * shape.frames * shape.head_dim * shape.head_dim;
    const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
    const std::size_t state_matrices = bh * shape.head_dim * shape.head_dim;
    const std::size_t state_vectors = bh * shape.head_dim;
    float* q_inv = workspace_take(ptr, tokens);
    float* k_inv = workspace_take(ptr, tokens);
    float* i_p_kv = workspace_take(ptr, frame_matrices);
    float* a_t = workspace_take(ptr, frame_matrices);
    float* i_p_z = workspace_take(ptr, frame_matrices);
    float* b_z = workspace_take(ptr, frame_vectors);
    float* hist_kv = workspace_take(ptr, frame_matrices);
    float* hist_z = workspace_take(ptr, frame_vectors);
    float* state_kv_a = workspace_take(ptr, state_matrices);
    float* state_kv_b = workspace_take(ptr, state_matrices);
    float* state_z_a = workspace_take(ptr, state_vectors);
    float* state_z_b = workspace_take(ptr, state_vectors);

    constexpr int32_t kThreads = 256;
    const int64_t token_elems = static_cast<int64_t>(tokens);
    const int64_t matrix_elems = static_cast<int64_t>(frame_matrices);
    const int64_t state_matrix_elems = static_cast<int64_t>(state_matrices);
    const int64_t state_vector_elems = static_cast<int64_t>(state_vectors);
    const int64_t output_elems = static_cast<int64_t>(shape.batch) * shape.frames *
                                 shape.spatial * shape.heads * shape.head_dim;

    raw_qk_inv_rms_kernel<<<static_cast<uint32_t>((token_elems + kThreads - 1) / kThreads),
                            kThreads, 0, stream>>>(q_inv, k_inv, q_raw, k_raw, shape,
                                                    norm_eps);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "raw_qk_inv_rms",
                                 "main_raw_combined", false, shape))
        return 1;

    phase_a_raw_combined_kernel<<<static_cast<uint32_t>((matrix_elems + kThreads - 1) /
                                                        kThreads),
                                  kThreads, 0, stream>>>(
        i_p_kv, a_t, i_p_z, b_z, q_raw, k_raw, v_raw, k_inv, k_norm_weight, rope_cos,
        rope_sin, beta, shape);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_a_raw", "main_raw_combined",
                                 false, shape))
        return 1;

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    float* state_kv = state_kv_a;
    float* next_kv = state_kv_b;
    float* state_z = state_z_a;
    float* next_z = state_z_b;
    for (int32_t t = 0; t < shape.frames; ++t) {
        phase_b_kv_kernel<<<static_cast<uint32_t>((state_matrix_elems + kThreads - 1) /
                                                  kThreads),
                            kThreads, 0, stream>>>(next_kv, state_kv, hist_kv, i_p_kv, a_t,
                                                    decay, shape, t, t, false, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_raw_kv_fwd",
                                     "main_raw_combined", false, shape))
            return 1;
        phase_b_z_kernel<<<static_cast<uint32_t>((state_vector_elems + kThreads - 1) /
                                                 kThreads),
                           kThreads, 0, stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay,
                                                   shape, t, t, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_raw_z_fwd",
                                     "main_raw_combined", false, shape))
            return 1;
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    cudaMemsetAsync(state_kv_a, 0, state_matrices * sizeof(float), stream);
    cudaMemsetAsync(state_z_a, 0, state_vectors * sizeof(float), stream);
    state_kv = state_kv_a;
    next_kv = state_kv_b;
    state_z = state_z_a;
    next_z = state_z_b;
    for (int32_t f_src = shape.frames - 1; f_src >= 1; --f_src) {
        const int32_t f_dst = f_src - 1;
        phase_b_kv_kernel<<<static_cast<uint32_t>((state_matrix_elems + kThreads - 1) /
                                                  kThreads),
                            kThreads, 0, stream>>>(next_kv, state_kv, hist_kv, i_p_kv, a_t,
                                                    decay, shape, f_src, f_dst, true, false);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_raw_kv_rev",
                                     "main_raw_combined", false, shape))
            return 1;
        phase_b_z_kernel<<<static_cast<uint32_t>((state_vector_elems + kThreads - 1) /
                                                 kThreads),
                           kThreads, 0, stream>>>(next_z, state_z, hist_z, i_p_z, b_z, decay,
                                                   shape, f_src, f_dst, true);
        if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_b_raw_z_rev",
                                     "main_raw_combined", false, shape))
            return 1;
        float* tmp_kv = state_kv;
        state_kv = next_kv;
        next_kv = tmp_kv;
        float* tmp_z = state_z;
        state_z = next_z;
        next_z = tmp_z;
    }

    phase_c_raw_combined_kernel<<<static_cast<uint32_t>((output_elems + kThreads - 1) /
                                                        kThreads),
                                  kThreads, 0, stream>>>(
        out, hist_kv, hist_z, q_raw, q_inv, q_norm_weight, rope_cos, rope_sin, shape, eps);
    if (report_cuda_launch_error(cudaPeekAtLastError(), "phase_c_raw", "main_raw_combined",
                                 false, shape))
        return 1;
    return 0;
}

} // namespace

SanaWmGdnPlugin::SanaWmGdnPlugin(Mode mode, bool reverse_output, float eps)
    : mode_(mode), reverse_output_(reverse_output), eps_(eps) {}

SanaWmGdnPlugin::SanaWmGdnPlugin(Mode mode, bool reverse_output, float eps, int32_t frames,
                                 int32_t head_dim, float norm_eps)
    : mode_(mode), reverse_output_(reverse_output), eps_(eps), frames_(frames),
      head_dim_(head_dim), norm_eps_(norm_eps) {}

SanaWmGdnPlugin::SanaWmGdnPlugin(const void* data, size_t length) {
    const auto* p = static_cast<const int32_t*>(data);
    if (p[0] == 1) {
        mode_ = Mode::kCamera;
    } else if (p[0] == 2) {
        mode_ = Mode::kMainCombined;
    } else if (p[0] == 3) {
        mode_ = Mode::kMainRawCombined;
    } else if (p[0] == 4) {
        mode_ = Mode::kCameraCombined;
    } else {
        mode_ = Mode::kMain;
    }
    reverse_output_ = p[1] != 0;
    if (length >= 4 * sizeof(int32_t) + 2 * sizeof(float)) {
        float eps = 1.0e-6F;
        std::memcpy(&eps, static_cast<const char*>(data) + 4 * sizeof(int32_t), sizeof(float));
        eps_ = eps;
        frames_ = p[2];
        head_dim_ = p[3];
        float norm_eps = 1.0e-5F;
        std::memcpy(&norm_eps, static_cast<const char*>(data) + 4 * sizeof(int32_t) +
                                   sizeof(float),
                    sizeof(float));
        norm_eps_ = norm_eps;
    } else if (length >= 2 * sizeof(int32_t) + sizeof(float)) {
        float eps = 1.0e-6F;
        std::memcpy(&eps, static_cast<const char*>(data) + 2 * sizeof(int32_t), sizeof(float));
        eps_ = eps;
    }
}

char const* SanaWmGdnPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGdnPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGdnPlugin::getNbOutputs() const noexcept {
    if (is_main_raw_combined()) {
        return 1;
    }
    return is_main() ? 2 : 1;
}

int32_t SanaWmGdnPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGdnPlugin::terminate() noexcept {}

void SanaWmGdnPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmGdnPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t) + 2 * sizeof(float);
}

void SanaWmGdnPlugin::serialize(void* buffer) const noexcept {
    auto* p = static_cast<int32_t*>(buffer);
    p[0] = static_cast<int32_t>(mode_);
    p[1] = reverse_output_ ? 1 : 0;
    p[2] = frames_;
    p[3] = head_dim_;
    std::memcpy(static_cast<char*>(buffer) + 4 * sizeof(int32_t), &eps_, sizeof(float));
    std::memcpy(static_cast<char*>(buffer) + 4 * sizeof(int32_t) + sizeof(float), &norm_eps_,
                sizeof(float));
}

void SanaWmGdnPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGdnPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGdnPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                      int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmGdnPlugin* SanaWmGdnPlugin::clone() const noexcept {
    auto* p = new SanaWmGdnPlugin(mode_, reverse_output_, eps_, frames_, head_dim_, norm_eps_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmGdnPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                     int32_t, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (is_main_raw_combined()) {
        (void)outputIndex;
        (void)exprBuilder;
        nvinfer1::DimsExprs out;
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];
        out.d[1] = inputs[0].d[1];
        out.d[2] = inputs[0].d[2];
        return out;
    }
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = is_main() && outputIndex == 1 ? exprBuilder.constant(1) : inputs[0].d[3];
    out.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[2],
                                     *inputs[0].d[4]);
    return out;
}

bool SanaWmGdnPlugin::supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
                                                int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void SanaWmGdnPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                      nvinfer1::DynamicPluginTensorDesc const*,
                                      int32_t) noexcept {}

size_t SanaWmGdnPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t,
                                         nvinfer1::PluginTensorDesc const*, int32_t) const noexcept {
    const auto shape =
        is_main_raw_combined() ? parse_raw_shape(inputs[0].dims, frames_, head_dim_)
                               : parse_shape(inputs[0].dims);
    const std::size_t bh = static_cast<std::size_t>(shape.batch) * shape.heads;
    if (is_main_raw_combined()) {
        const char* use_cublas = std::getenv("TRTMC_SANA_WM_GDN_CUBLAS");
        if (use_cublas != nullptr && std::strcmp(use_cublas, "0") != 0 &&
            std::strcmp(use_cublas, "false") != 0 && std::strcmp(use_cublas, "False") != 0) {
            const int32_t padded_dim = next_power_of_two(shape.head_dim);
            const std::size_t tokens =
                static_cast<std::size_t>(shape.batch) * shape.frames * shape.spatial;
            const std::size_t frame_scratch =
                bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
            const std::size_t frame_matrices =
                bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
            const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
            const std::size_t state_matrices =
                bh * static_cast<std::size_t>(padded_dim) * padded_dim;
            const std::size_t state_vectors = bh * shape.head_dim;
            std::size_t total = 0;
            total += 2 * float_bytes(tokens);
            total += 6 * bf16_bytes(frame_scratch);
            total += 3 * float_bytes(state_matrices);
            total += 3 * bf16_bytes(frame_matrices);
            total += float_bytes(frame_vectors);
            total += float_bytes(frame_matrices);
            total += float_bytes(frame_vectors);
            total += 2 * float_bytes(state_matrices);
            total += bf16_bytes(state_matrices);
            total += 2 * float_bytes(state_vectors);
            total += float_bytes(frame_scratch);
            return total;
        }
        const std::size_t tokens =
            static_cast<std::size_t>(shape.batch) * shape.frames * shape.spatial;
        const std::size_t frame_matrices =
            bh * shape.frames * shape.head_dim * shape.head_dim;
        const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
        const std::size_t state_matrices = bh * shape.head_dim * shape.head_dim;
        const std::size_t state_vectors = bh * shape.head_dim;
        std::size_t total = 0;
        total += 2 * float_bytes(tokens);
        total += 4 * float_bytes(frame_matrices);
        total += 2 * float_bytes(frame_vectors);
        total += 2 * float_bytes(state_matrices);
        total += 2 * float_bytes(state_vectors);
        return total;
    }
    if (is_main_combined()) {
        if (use_main_combined_cublas()) {
            const int32_t padded_dim = next_power_of_two(shape.head_dim);
            const std::size_t frame_scratch =
                bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
            const std::size_t frame_matrices =
                bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
            const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
            const std::size_t state_matrices =
                bh * static_cast<std::size_t>(padded_dim) * padded_dim;
            const std::size_t state_vectors = bh * shape.head_dim;
            std::size_t total = 0;
            total += 6 * bf16_bytes(frame_scratch);
            total += 3 * float_bytes(state_matrices);
            total += 3 * bf16_bytes(frame_matrices);
            total += float_bytes(frame_vectors);
            total += float_bytes(frame_matrices);
            total += float_bytes(frame_vectors);
            total += 2 * float_bytes(state_matrices);
            total += bf16_bytes(state_matrices);
            total += 2 * float_bytes(state_vectors);
            total += float_bytes(frame_scratch);
            total += bf16_bytes(bh * static_cast<std::size_t>(shape.spatial));
            return total;
        }
        const std::size_t frame_matrices =
            bh * shape.frames * shape.head_dim * shape.head_dim;
        const std::size_t frame_vectors = bh * shape.frames * shape.head_dim;
        const std::size_t state_matrices = bh * shape.head_dim * shape.head_dim;
        const std::size_t state_vectors = bh * shape.head_dim;
        std::size_t total = 0;
        total += 4 * float_bytes(frame_matrices);
        total += 2 * float_bytes(frame_vectors);
        total += 2 * float_bytes(state_matrices);
        total += 2 * float_bytes(state_vectors);
        return total;
    }
    if (is_camera_combined()) {
        const int32_t padded_dim = next_power_of_two(shape.head_dim);
        const std::size_t frame_scratch =
            bh * static_cast<std::size_t>(shape.spatial) * padded_dim;
        const std::size_t frame_matrices =
            bh * shape.frames * static_cast<std::size_t>(padded_dim) * padded_dim;
        const std::size_t state_matrices =
            bh * static_cast<std::size_t>(padded_dim) * padded_dim;
        std::size_t total = 0;
        total += 6 * bf16_bytes(frame_scratch);
        total += 2 * float_bytes(state_matrices);
        total += 3 * bf16_bytes(frame_matrices);
        total += float_bytes(frame_matrices);
        total += 2 * float_bytes(state_matrices);
        total += bf16_bytes(state_matrices);
        total += float_bytes(frame_scratch);
        return total;
    }
    std::size_t total = 0;
    total += float_bytes(bh * shape.head_dim * shape.head_dim);
    if (is_main()) {
        total += float_bytes(bh * shape.head_dim);
    }
    total += float_bytes(bh * shape.head_dim * shape.spatial);
    if (is_main()) {
        total += float_bytes(bh * shape.spatial);
    }
    return total;
}

int32_t SanaWmGdnPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                 nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    const auto shape =
        is_main_raw_combined() ? parse_raw_shape(inputDesc[0].dims, frames_, head_dim_)
                               : parse_shape(inputDesc[0].dims);
    if (shape.batch <= 0 || shape.heads <= 0 || shape.frames <= 0 || shape.head_dim <= 0 ||
        shape.spatial <= 0) {
        return 1;
    }
    if (is_main_raw_combined()) {
        return launch_main_raw_combined(shape, inputs, outputs, workspace, eps_, norm_eps_, stream);
    }
    if (is_main()) {
        return launch_main_scan(shape, inputs, outputs, workspace, reverse_output_, stream);
    }
    if (is_main_combined()) {
        return launch_main_combined(shape, inputs, outputs, workspace, eps_, stream);
    }
    if (is_camera_combined()) {
        return launch_camera_combined_cublas(shape, inputs, outputs, workspace, stream);
    }
    return launch_camera_scan(shape, inputs, outputs, workspace, reverse_output_, stream);
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

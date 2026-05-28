#if TRTMC_HAS_TRT

#include "plugins/sana_wm_timestep_plugin.h"

#include <cublasLt.h>
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

std::size_t align_bytes(std::size_t value) {
    constexpr std::size_t kAlign = 256;
    return ((value + kAlign - 1) / kAlign) * kAlign;
}

std::size_t bf16_bytes(std::size_t count) { return align_bytes(count * sizeof(uint16_t)); }

uint16_t* workspace_take(char*& ptr, std::size_t count) {
    auto* out = reinterpret_cast<uint16_t*>(ptr);
    ptr += bf16_bytes(count);
    return out;
}

bool launch_ok(const char* kernel);

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
    if (!values.empty()) {
        const std::size_t bytes = values.size() * sizeof(uint16_t);
        std::memcpy(ptr, values.data(), bytes);
        ptr += bytes;
    }
}

void write_float_vector(char*& ptr, const std::vector<float>& values) {
    const auto size = static_cast<uint64_t>(values.size());
    write_value(ptr, size);
    if (!values.empty()) {
        const std::size_t bytes = values.size() * sizeof(float);
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

std::vector<float> read_float_vector(const char*& ptr, const char* end) {
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

bool copy_to_device(const std::vector<uint16_t>& host, void** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const std::size_t bytes = host.size() * sizeof(uint16_t);
    return cudaMalloc(device, bytes) == cudaSuccess &&
           cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool copy_float_to_device(const std::vector<float>& host, void** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    const std::size_t bytes = host.size() * sizeof(float);
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

__global__ void timestep_frequency_kernel(const float* timestep, const float* freqs,
                                          uint16_t* out, int32_t rows, int32_t frequency_dim) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t total = rows * frequency_dim;
    if (idx >= total)
        return;
    const int32_t dim = idx % frequency_dim;
    const int32_t row = idx / frequency_dim;
    const int32_t half = frequency_dim / 2;
    const int32_t freq_idx = dim < half ? dim : dim - half;
    const float arg = timestep[row] * freqs[freq_idx];
    const float value = dim < half ? cosf(arg) : sinf(arg);
    out[idx] = float_to_bf16_bits(value);
}

__global__ void linear_kernel(const uint16_t* input, const uint16_t* weight,
                              const uint16_t* bias, uint16_t* output, int32_t rows,
                              int32_t input_dim, int32_t output_dim, bool apply_silu) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t total = rows * output_dim;
    if (idx >= total)
        return;
    const int32_t out_col = idx % output_dim;
    const int32_t row = idx / output_dim;
    float acc = bf16_to_float(bias[out_col]);
    for (int32_t k = 0; k < input_dim; ++k) {
        acc += bf16_to_float(input[row * input_dim + k]) *
               bf16_to_float(weight[k * output_dim + out_col]);
    }
    if (apply_silu)
        acc = acc / (1.0F + expf(-acc));
    output[idx] = float_to_bf16_bits(acc);
}

void destroy_lt(cublasLtMatmulDesc_t desc) {
    if (desc != nullptr)
        cublasLtMatmulDescDestroy(desc);
}

void destroy_lt(cublasLtMatrixLayout_t layout) {
    if (layout != nullptr)
        cublasLtMatrixLayoutDestroy(layout);
}

void destroy_lt(cublasLtMatmulPreference_t preference) {
    if (preference != nullptr)
        cublasLtMatmulPreferenceDestroy(preference);
}

bool lt_linear(cublasLtHandle_t handle, const uint16_t* input, const uint16_t* weight,
               const uint16_t* bias, uint16_t* output, int32_t rows, int32_t input_dim,
               int32_t output_dim, cudaStream_t stream) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    auto log_status = [](const char* step, cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "SanaWmTimestepEmbed cuBLASLt %s failed status=%d\n", step,
                         static_cast<int>(status));
        }
        return status == CUBLAS_STATUS_SUCCESS;
    };
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    bool ok = log_status("desc", status);
    const cublasOperation_t trans = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                            sizeof(trans));
    ok = ok && log_status("transa", status);
    status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans,
                                            sizeof(trans));
    ok = ok && log_status("transb", status);
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status =
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                       sizeof(epilogue));
    ok = ok && log_status("epilogue", status);
    const void* bias_ptr = bias;
    status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias_ptr, sizeof(bias_ptr));
    ok = ok && log_status("bias", status);
    status =
        cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, output_dim, input_dim, output_dim);
    ok = ok && log_status("a_layout", status);
    status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, input_dim, rows, input_dim);
    ok = ok && log_status("b_layout", status);
    status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, output_dim, rows, output_dim);
    ok = ok && log_status("c_layout", status);
    status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, output_dim, rows, output_dim);
    ok = ok && log_status("d_layout", status);
    status = cublasLtMatmulPreferenceCreate(&preference);
    ok = ok && log_status("preference", status);
    const std::size_t max_workspace = 0;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
        sizeof(max_workspace));
    ok = ok && log_status("workspace_pref", status);
    cublasLtMatmulHeuristicResult_t heuristic{};
    int32_t returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc, c_desc, d_desc,
                                            preference, 1, &heuristic, &returned_results);
    const bool heuristic_ok = log_status("heuristic", status);
    if (heuristic_ok && returned_results <= 0)
        std::fprintf(stderr, "SanaWmTimestepEmbed cuBLASLt heuristic returned no algorithms\n");
    ok = ok && heuristic_ok && returned_results > 0;
    const float alpha = 1.0F;
    const float beta = 0.0F;
    status = cublasLtMatmul(handle, op_desc, &alpha, weight, a_desc, input, b_desc, &beta,
                            output, c_desc, output, d_desc, &heuristic.algo, nullptr, 0, stream);
    ok = ok && log_status("matmul", status);
    destroy_lt(preference);
    destroy_lt(d_desc);
    destroy_lt(c_desc);
    destroy_lt(b_desc);
    destroy_lt(a_desc);
    destroy_lt(op_desc);
    return ok;
}

__global__ void silu_kernel(const uint16_t* input, uint16_t* output, int32_t total) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total)
        return;
    const float value = bf16_to_float(input[idx]);
    output[idx] = float_to_bf16_bits(value / (1.0F + expf(-value)));
}

bool launch_ok(const char* kernel) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmTimestepEmbed %s failed: %s\n", kernel,
                 cudaGetErrorString(status));
    return false;
}

int32_t row_count(const nvinfer1::Dims& dims) {
    if (dims.nbDims != 3)
        return 0;
    return static_cast<int32_t>(dims.d[0] * dims.d[1] * dims.d[2]);
}

} // namespace

SanaWmTimestepEmbedPlugin::SanaWmTimestepEmbedPlugin(
    int32_t frequency_dim, int32_t hidden_size, std::vector<float> freqs,
    std::vector<uint16_t> w0, std::vector<uint16_t> b0, std::vector<uint16_t> w1,
    std::vector<uint16_t> b1, std::vector<uint16_t> w2, std::vector<uint16_t> b2)
    : frequency_dim_(frequency_dim), hidden_size_(hidden_size), freqs_(std::move(freqs)),
      w0_(std::move(w0)), b0_(std::move(b0)), w1_(std::move(w1)), b1_(std::move(b1)),
      w2_(std::move(w2)), b2_(std::move(b2)) {}

SanaWmTimestepEmbedPlugin::SanaWmTimestepEmbedPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415445U || version != 1U)
        return;
    frequency_dim_ = read_value<int32_t>(ptr, end, 0);
    hidden_size_ = read_value<int32_t>(ptr, end, 0);
    freqs_ = read_float_vector(ptr, end);
    w0_ = read_vector(ptr, end);
    b0_ = read_vector(ptr, end);
    w1_ = read_vector(ptr, end);
    b1_ = read_vector(ptr, end);
    w2_ = read_vector(ptr, end);
    b2_ = read_vector(ptr, end);
}

char const* SanaWmTimestepEmbedPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmTimestepEmbedPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmTimestepEmbedPlugin::getNbOutputs() const noexcept { return 2; }

int32_t SanaWmTimestepEmbedPlugin::initialize() noexcept {
    if (device_freqs_ != nullptr && device_w0_ != nullptr && device_b0_ != nullptr &&
        device_w1_ != nullptr && device_b1_ != nullptr && device_w2_ != nullptr &&
        device_b2_ != nullptr && lt_handle_ != nullptr) {
        return 0;
    }
    terminate();
    if (cublasLtCreate(&lt_handle_) != CUBLAS_STATUS_SUCCESS ||
        !copy_float_to_device(freqs_, &device_freqs_) || !copy_to_device(w0_, &device_w0_) ||
        !copy_to_device(b0_, &device_b0_) ||
        !copy_to_device(w1_, &device_w1_) || !copy_to_device(b1_, &device_b1_) ||
        !copy_to_device(w2_, &device_w2_) || !copy_to_device(b2_, &device_b2_)) {
        terminate();
        return 1;
    }
    return 0;
}

void SanaWmTimestepEmbedPlugin::terminate() noexcept {
    if (lt_handle_ != nullptr) {
        cublasLtDestroy(lt_handle_);
        lt_handle_ = nullptr;
    }
    void** ptrs[] = {&device_freqs_, &device_w0_, &device_b0_, &device_w1_,
                     &device_b1_,   &device_w2_, &device_b2_};
    for (void** ptr : ptrs) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    }
}

void SanaWmTimestepEmbedPlugin::destroy() noexcept { delete this; }

size_t SanaWmTimestepEmbedPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) * 2 + sizeof(uint64_t) * 7 +
           freqs_.size() * sizeof(float) +
           (w0_.size() + b0_.size() + w1_.size() + b1_.size() + w2_.size() + b2_.size()) *
               sizeof(uint16_t);
}

void SanaWmTimestepEmbedPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415445U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, frequency_dim_);
    write_value(ptr, hidden_size_);
    write_float_vector(ptr, freqs_);
    write_vector(ptr, w0_);
    write_vector(ptr, b0_);
    write_vector(ptr, w1_);
    write_vector(ptr, b1_);
    write_vector(ptr, w2_);
    write_vector(ptr, b2_);
}

void SanaWmTimestepEmbedPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmTimestepEmbedPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmTimestepEmbedPlugin::getOutputDataType(
    int32_t /*index*/, nvinfer1::DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const
    noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmTimestepEmbedPlugin* SanaWmTimestepEmbedPlugin::clone() const noexcept {
    auto* plugin =
        new SanaWmTimestepEmbedPlugin(frequency_dim_, hidden_size_, freqs_, w0_, b0_, w1_, b1_,
                                      w2_, b2_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmTimestepEmbedPlugin::getOutputDimensions(
    int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t /*nbInputs*/,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    out.d[3] = exprBuilder.constant(outputIndex == 0 ? hidden_size_ : 6 * hidden_size_);
    return out;
}

bool SanaWmTimestepEmbedPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t /*nbInputs*/,
    int32_t /*nbOutputs*/) noexcept {
    const auto& desc = inOut[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos == 0)
        return desc.type == nvinfer1::DataType::kFLOAT;
    return desc.type == nvinfer1::DataType::kBF16;
}

void SanaWmTimestepEmbedPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* /*in*/,
                                                int32_t /*nbInputs*/,
                                                nvinfer1::DynamicPluginTensorDesc const* /*out*/,
                                                int32_t /*nbOutputs*/) noexcept {}

size_t SanaWmTimestepEmbedPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    nvinfer1::PluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    const int32_t rows = row_count(inputs[0].dims);
    if (rows <= 0)
        return 0;
    return bf16_bytes(static_cast<std::size_t>(rows) * frequency_dim_) +
           bf16_bytes(static_cast<std::size_t>(rows) * hidden_size_) +
           bf16_bytes(static_cast<std::size_t>(rows) * hidden_size_);
}

int32_t SanaWmTimestepEmbedPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                           nvinfer1::PluginTensorDesc const* /*outputDesc*/,
                                           void const* const* inputs, void* const* outputs,
                                           void* workspace, cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputs == nullptr || outputs == nullptr || workspace == nullptr)
        return 1;
    const int32_t rows = row_count(inputDesc[0].dims);
    if (rows <= 0 || frequency_dim_ <= 0 || hidden_size_ <= 0)
        return 1;
    if (static_cast<int32_t>(freqs_.size()) != frequency_dim_ / 2)
        return 1;
    if (static_cast<int64_t>(w0_.size()) != static_cast<int64_t>(frequency_dim_) * hidden_size_ ||
        static_cast<int64_t>(w1_.size()) != static_cast<int64_t>(hidden_size_) * hidden_size_ ||
        static_cast<int64_t>(w2_.size()) !=
            static_cast<int64_t>(hidden_size_) * 6 * hidden_size_ ||
        static_cast<int32_t>(b0_.size()) != hidden_size_ ||
        static_cast<int32_t>(b1_.size()) != hidden_size_ ||
        static_cast<int32_t>(b2_.size()) != 6 * hidden_size_) {
        return 1;
    }

    auto* ptr = static_cast<char*>(workspace);
    uint16_t* t_freq = workspace_take(ptr, static_cast<std::size_t>(rows) * frequency_dim_);
    uint16_t* hidden = workspace_take(ptr, static_cast<std::size_t>(rows) * hidden_size_);
    uint16_t* hidden_silu = workspace_take(ptr, static_cast<std::size_t>(rows) * hidden_size_);
    auto* out_t = static_cast<uint16_t*>(outputs[0]);
    auto* out_t0 = static_cast<uint16_t*>(outputs[1]);
    const auto* timestep = static_cast<const float*>(inputs[0]);
    constexpr int32_t kThreads = 256;

    const int32_t freq_total = rows * frequency_dim_;
    timestep_frequency_kernel<<<(freq_total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        timestep, static_cast<const float*>(device_freqs_), t_freq, rows, frequency_dim_);
    if (!launch_ok("frequency"))
        return 1;
    if (!lt_linear(lt_handle_, t_freq, static_cast<const uint16_t*>(device_w0_),
                   static_cast<const uint16_t*>(device_b0_), hidden, rows, frequency_dim_,
                   hidden_size_, stream)) {
        std::fprintf(stderr, "SanaWmTimestepEmbed linear0 failed\n");
        return 1;
    }
    silu_kernel<<<(rows * hidden_size_ + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        hidden, hidden_silu, rows * hidden_size_);
    if (!launch_ok("silu_hidden"))
        return 1;
    if (!lt_linear(lt_handle_, hidden_silu, static_cast<const uint16_t*>(device_w1_),
                   static_cast<const uint16_t*>(device_b1_), out_t, rows, hidden_size_,
                   hidden_size_, stream)) {
        std::fprintf(stderr, "SanaWmTimestepEmbed linear1 failed\n");
        return 1;
    }
    silu_kernel<<<(rows * hidden_size_ + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        out_t, hidden_silu, rows * hidden_size_);
    if (!launch_ok("silu_t"))
        return 1;
    if (!lt_linear(lt_handle_, hidden_silu, static_cast<const uint16_t*>(device_w2_),
                   static_cast<const uint16_t*>(device_b2_), out_t0, rows, hidden_size_,
                   6 * hidden_size_, stream)) {
        std::fprintf(stderr, "SanaWmTimestepEmbed linear2 failed\n");
        return 1;
    }
    return 0;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

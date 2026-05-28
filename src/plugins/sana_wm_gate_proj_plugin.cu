#if TRTMC_HAS_TRT

#include "plugins/sana_wm_gate_proj_plugin.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace trtmc {
namespace {

constexpr std::size_t kLtWorkspaceBytes = 4U * 1024U * 1024U;

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

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void skinny_linear_sigmoid_kernel(const uint16_t* input, const uint16_t* weight,
                                             const uint16_t* bias, uint16_t* output,
                                             int32_t rows, int32_t input_dim,
                                             int32_t output_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(rows) * output_dim;
    if (idx >= total)
        return;
    const int32_t n = static_cast<int32_t>(idx % output_dim);
    const int32_t row = static_cast<int32_t>(idx / output_dim);
    float acc = bias != nullptr ? bf16_to_float(bias[n]) : 0.0F;
    const int64_t input_base = static_cast<int64_t>(row) * input_dim;
    for (int32_t k = input_dim - 1; k >= 0; --k) {
        acc += bf16_to_float(input[input_base + k]) *
               bf16_to_float(weight[static_cast<int64_t>(k) * output_dim + n]);
    }
    const uint16_t linear = float_to_bf16_bits(acc);
    const float value = bf16_to_float(linear);
    output[idx] = float_to_bf16_bits(1.0F / (1.0F + expf(-value)));
}

bool launch_skinny_linear_sigmoid(const uint16_t* input, const uint16_t* weight,
                                  const uint16_t* bias, uint16_t* output, int32_t rows,
                                  int32_t input_dim, int32_t output_dim,
                                  cudaStream_t stream) {
    constexpr int32_t kThreads = 256;
    const int64_t total = static_cast<int64_t>(rows) * output_dim;
    skinny_linear_sigmoid_kernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads),
                                   kThreads, 0, stream>>>(input, weight, bias, output, rows,
                                                           input_dim, output_dim);
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "SanaWmGateProj skinny sigmoid failed: %s\n",
                 cudaGetErrorString(status));
    return false;
}

bool lt_linear(cublasLtHandle_t handle, const uint16_t* input, const uint16_t* weight,
               const uint16_t* /*bias*/, uint16_t* output, int32_t rows, int32_t input_dim,
               int32_t output_dim, void* workspace, std::size_t workspace_size,
               cudaStream_t stream) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    auto log_status = [](const char* step, cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "SanaWmGateProj cuBLASLt %s failed status=%d\n", step,
                         static_cast<int>(status));
        }
        return status == CUBLAS_STATUS_SUCCESS;
    };

    bool ok = log_status(
        "desc", cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans = CUBLAS_OP_N;
    ok = ok && log_status("transa", cublasLtMatmulDescSetAttribute(
                                        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                        sizeof(trans)));
    ok = ok && log_status("transb", cublasLtMatmulDescSetAttribute(
                                        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans,
                                        sizeof(trans)));
    ok = ok && log_status("a_layout", cublasLtMatrixLayoutCreate(
                                          &a_desc, CUDA_R_16BF, output_dim, input_dim,
                                          output_dim));
    ok = ok && log_status("b_layout", cublasLtMatrixLayoutCreate(
                                          &b_desc, CUDA_R_16BF, input_dim, rows, input_dim));
    ok = ok && log_status("c_layout", cublasLtMatrixLayoutCreate(
                                          &c_desc, CUDA_R_16BF, output_dim, rows, output_dim));
    ok = ok && log_status("d_layout", cublasLtMatrixLayoutCreate(
                                          &d_desc, CUDA_R_16BF, output_dim, rows, output_dim));
    ok = ok && log_status("preference", cublasLtMatmulPreferenceCreate(&preference));
    const std::size_t max_workspace = workspace_size;
    ok = ok && log_status("workspace_pref", cublasLtMatmulPreferenceSetAttribute(
                                                preference,
                                                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &max_workspace, sizeof(max_workspace)));
    constexpr int32_t kMaxAlgos = 16;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos]{};
    int32_t returned_results = 0;
    const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle, op_desc, a_desc, b_desc, c_desc, d_desc, preference, kMaxAlgos, heuristics,
        &returned_results);
    ok = ok && log_status("heuristic", heuristic_status) && returned_results > 0;
    int32_t algo_index = returned_results > 1 ? 1 : 0;
    if (const char* env = std::getenv("TRTMC_SANA_WM_GATE_PROJ_ALGO_INDEX")) {
        algo_index = std::atoi(env);
        if (algo_index < 0 || algo_index >= returned_results)
            algo_index = 0;
    }
    const float alpha = 1.0F;
    const float beta_value = 0.0F;
    if (ok) {
        ok = log_status("matmul", cublasLtMatmul(handle, op_desc, &alpha, weight, a_desc,
                                                 input, b_desc, &beta_value, output, c_desc,
                                                 output, d_desc, &heuristics[algo_index].algo, workspace,
                                                 workspace_size, stream));
    }
    destroy_lt(preference);
    destroy_lt(d_desc);
    destroy_lt(c_desc);
    destroy_lt(b_desc);
    destroy_lt(a_desc);
    destroy_lt(op_desc);
    return ok;
}

int32_t row_count(const nvinfer1::Dims& dims) {
    if (dims.nbDims != 3)
        return 0;
    return static_cast<int32_t>(dims.d[0] * dims.d[1]);
}

} // namespace

SanaWmGateProjPlugin::SanaWmGateProjPlugin(int32_t input_dim, int32_t output_dim,
                                           std::vector<uint16_t> weight,
                                           std::vector<uint16_t> bias, int32_t activation)
    : input_dim_(input_dim), output_dim_(output_dim), weight_(std::move(weight)),
      bias_(std::move(bias)), activation_(activation) {}

SanaWmGateProjPlugin::SanaWmGateProjPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53414750U || (version != 1U && version != 2U))
        return;
    input_dim_ = read_value<int32_t>(ptr, end, 0);
    output_dim_ = read_value<int32_t>(ptr, end, 0);
    if (version >= 2U)
        activation_ = read_value<int32_t>(ptr, end, 0);
    weight_ = read_vector(ptr, end);
    bias_ = read_vector(ptr, end);
}

char const* SanaWmGateProjPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmGateProjPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmGateProjPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmGateProjPlugin::initialize() noexcept {
    if (lt_handle_ != nullptr && device_weight_ != nullptr &&
        (bias_.empty() || device_bias_ != nullptr))
        return 0;
    terminate();
    if (cublasLtCreate(&lt_handle_) != CUBLAS_STATUS_SUCCESS ||
        !copy_to_device(weight_, &device_weight_) || !copy_to_device(bias_, &device_bias_)) {
        terminate();
        return 1;
    }
    return 0;
}

void SanaWmGateProjPlugin::terminate() noexcept {
    if (lt_handle_ != nullptr) {
        cublasLtDestroy(lt_handle_);
        lt_handle_ = nullptr;
    }
    void** ptrs[] = {&device_weight_, &device_bias_};
    for (void** ptr : ptrs) {
        if (*ptr != nullptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
    }
}

void SanaWmGateProjPlugin::destroy() noexcept { delete this; }

size_t SanaWmGateProjPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) * 3 + sizeof(uint64_t) * 2 +
           (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmGateProjPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53414750U);
    write_value<uint32_t>(ptr, 2U);
    write_value(ptr, input_dim_);
    write_value(ptr, output_dim_);
    write_value(ptr, activation_);
    write_vector(ptr, weight_);
    write_vector(ptr, bias_);
}

void SanaWmGateProjPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGateProjPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGateProjPlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGateProjPlugin* SanaWmGateProjPlugin::clone() const noexcept {
    auto* plugin =
        new SanaWmGateProjPlugin(input_dim_, output_dim_, weight_, bias_, activation_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmGateProjPlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 3;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = exprBuilder.constant(output_dim_);
    return out;
}

bool SanaWmGateProjPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t, int32_t) noexcept {
    const auto& desc = inOut[pos];
    return desc.format == nvinfer1::TensorFormat::kLINEAR &&
           desc.type == nvinfer1::DataType::kBF16;
}

void SanaWmGateProjPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                           nvinfer1::DynamicPluginTensorDesc const*,
                                           int32_t) noexcept {}

size_t SanaWmGateProjPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t,
                                              nvinfer1::PluginTensorDesc const*,
                                              int32_t) const noexcept {
    (void)inputs;
    if (activation_ == 1)
        return 0;
    return kLtWorkspaceBytes;
}

int32_t SanaWmGateProjPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                      nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                      void* const* outputs, void* workspace,
                                      cudaStream_t stream) noexcept {
    if (initialize() != 0 || inputs == nullptr || outputs == nullptr)
        return 1;
    const int32_t rows = row_count(inputDesc[0].dims);
    if (rows <= 0 || inputDesc[0].dims.d[2] != input_dim_ || input_dim_ <= 0 ||
        output_dim_ <= 0)
        return 1;
    if (static_cast<int64_t>(weight_.size()) !=
            static_cast<int64_t>(input_dim_) * output_dim_ ||
        (!bias_.empty() && static_cast<int32_t>(bias_.size()) != output_dim_)) {
        return 1;
    }
    if (activation_ == 1) {
        return launch_skinny_linear_sigmoid(static_cast<const uint16_t*>(inputs[0]),
                                            static_cast<const uint16_t*>(device_weight_),
                                            static_cast<const uint16_t*>(device_bias_),
                                            static_cast<uint16_t*>(outputs[0]), rows,
                                            input_dim_, output_dim_, stream)
                   ? 0
                   : 1;
    }
    return lt_linear(lt_handle_, static_cast<const uint16_t*>(inputs[0]),
                     static_cast<const uint16_t*>(device_weight_),
                     static_cast<const uint16_t*>(device_bias_),
                     static_cast<uint16_t*>(outputs[0]), rows, input_dim_, output_dim_,
                     workspace, kLtWorkspaceBytes, stream)
               ? 0
               : 1;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT

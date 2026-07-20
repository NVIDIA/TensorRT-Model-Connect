/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Experimental Wan2.2 BF16 linear+bias probe.  This DSO intentionally uses
 * only TensorRT, CUDA, and cuBLASLt.  It must never gain a torch/ATen runtime
 * dependency: the point of the probe is to determine whether the upstream
 * autocast nn.Linear numerics can be reproduced by a deployable native plugin.
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::wan22::linear_probe {
// External linkage is intentional: this exact POD is also the qualification
// C ABI payload.  Keeping it out of the anonymous namespace prevents nvcc from
// giving query/get-info entry points local ELF visibility.
struct AlgoInfo {
    int32_t heuristic_index{-1};
    int32_t algorithm_id{-1};
    int32_t tile_id{-1};
    int32_t stages_id{-1};
    int32_t split_k{-1};
    int32_t reduction_scheme{-1};
    int32_t cta_swizzle{-1};
    int32_t custom_option{-1};
    int32_t inner_shape_id{-1};
    int32_t cluster_shape_id{-1};
    uint64_t workspace_bytes{0};
    float waves_count{0.0F};
};

namespace {

constexpr int32_t kMAX_HEURISTICS = 128;

struct ProbeConfig {
    int32_t m{0};
    int32_t n{0};
    int32_t k{0};
    int32_t heuristic_index{0};
    int32_t workspace_mib{1024};
};

static_assert(sizeof(ProbeConfig) == 5 * sizeof(int32_t));

template <typename T>
bool algo_attribute(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr,
                    T* value) {
    size_t written = 0;
    return value != nullptr &&
           cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, value, sizeof(T), &written) ==
               CUBLAS_STATUS_SUCCESS &&
           written == sizeof(T);
}

AlgoInfo describe_algorithm(const cublasLtMatmulHeuristicResult_t& result, int32_t index) {
    AlgoInfo info{};
    info.heuristic_index = index;
    info.workspace_bytes = static_cast<uint64_t>(result.workspaceSize);
    info.waves_count = result.wavesCount;
    algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_ID, &info.algorithm_id);

    uint32_t unsigned_value = 0;
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &unsigned_value))
        info.tile_id = static_cast<int32_t>(unsigned_value);
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &unsigned_value))
        info.stages_id = static_cast<int32_t>(unsigned_value);
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &unsigned_value))
        info.reduction_scheme = static_cast<int32_t>(unsigned_value);
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &unsigned_value))
        info.cta_swizzle = static_cast<int32_t>(unsigned_value);
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &unsigned_value))
        info.custom_option = static_cast<int32_t>(unsigned_value);

    int32_t signed_value = 0;
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &signed_value))
        info.split_k = signed_value;

    uint16_t short_value = 0;
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &short_value))
        info.inner_shape_id = static_cast<int32_t>(short_value);
    if (algo_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &short_value))
        info.cluster_shape_id = static_cast<int32_t>(short_value);
    return info;
}

class LinearContext {
  public:
    explicit LinearContext(ProbeConfig config) : config_(config) {}
    LinearContext(const LinearContext&) = delete;
    LinearContext& operator=(const LinearContext&) = delete;

    ~LinearContext() { reset(); }

    int initialize() {
        reset();
        if (config_.m <= 0 || config_.n <= 0 || config_.k <= 0 || config_.heuristic_index < 0 ||
            config_.workspace_mib < 0) {
            error_ = "invalid problem configuration";
            return 1;
        }
        if (!check(cublasLtCreate(&handle_), "cublasLtCreate"))
            return 1;

        // Match the representation used by PyTorch addmm/linear: row-major
        // X[M,K], W[N,K], and D[M,N] are viewed as column-major X^T[K,M],
        // W^T[K,N], and D^T[N,M].  The cuBLASLt problem is therefore
        // op(W^T)=W [N,K] times X^T [K,M], i.e. TNN in kernel traces.
        cublasOperation_t op_a = CUBLAS_OP_T;
        cublasOperation_t op_b = CUBLAS_OP_N;
        if (!check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "cublasLtMatmulDescCreate") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                                  sizeof(op_a)),
                   "set TRANSA") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                                  sizeof(op_b)),
                   "set TRANSB"))
            return 1;

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)),
                   "set BIAS epilogue"))
            return 1;

        if (!create_layout(&a_layout_, config_.k, config_.n, config_.k, "A(weight^T)") ||
            !create_layout(&b_layout_, config_.k, config_.m, config_.k, "B(x^T)") ||
            !create_layout(&c_layout_, config_.n, config_.m, config_.n, "C(output^T)") ||
            !create_layout(&d_layout_, config_.n, config_.m, config_.n, "D(output^T)"))
            return 1;

        if (!check(cublasLtMatmulPreferenceCreate(&preference_), "cublasLtMatmulPreferenceCreate"))
            return 1;
        const size_t workspace_limit = workspace_limit_bytes();
        if (!check(cublasLtMatmulPreferenceSetAttribute(preference_,
                                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace_limit, sizeof(workspace_limit)),
                   "set max workspace"))
            return 1;

        candidates_.resize(kMAX_HEURISTICS);
        int returned = 0;
        if (!check(cublasLtMatmulAlgoGetHeuristic(handle_, operation_, a_layout_, b_layout_,
                                                  c_layout_, d_layout_, preference_,
                                                  kMAX_HEURISTICS, candidates_.data(), &returned),
                   "cublasLtMatmulAlgoGetHeuristic"))
            return 1;
        candidates_.resize(static_cast<size_t>(std::max(returned, 0)));
        if (config_.heuristic_index >= returned) {
            error_ = "heuristic index " + std::to_string(config_.heuristic_index) +
                     " is outside returned candidate count " + std::to_string(returned);
            return 1;
        }
        const auto& selected = candidates_[static_cast<size_t>(config_.heuristic_index)];
        if (selected.state != CUBLAS_STATUS_SUCCESS) {
            error_ = "selected heuristic has non-success state " +
                     std::to_string(static_cast<int>(selected.state));
            return 1;
        }
        info_ = describe_algorithm(selected, config_.heuristic_index);
        initialized_ = true;
        return 0;
    }

    int run(const void* x, const void* weight, const void* bias, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || x == nullptr || weight == nullptr || bias == nullptr ||
            output == nullptr) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        if (workspace_bytes < static_cast<size_t>(info_.workspace_bytes)) {
            error_ = "workspace is smaller than selected algorithm requirement";
            return 1;
        }
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias, sizeof(bias)),
                   "set bias pointer"))
            return 1;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto& selected = candidates_[static_cast<size_t>(config_.heuristic_index)];
        return check(cublasLtMatmul(handle_, operation_, &alpha, weight, a_layout_, x, b_layout_,
                                    &beta, output, c_layout_, output, d_layout_, &selected.algo,
                                    workspace, workspace_bytes, stream),
                     "cublasLtMatmul")
                   ? 0
                   : 1;
    }

    const AlgoInfo& info() const { return info_; }
    const std::string& error() const { return error_; }
    int candidate_count() const { return static_cast<int>(candidates_.size()); }
    size_t workspace_limit_bytes() const {
        return static_cast<size_t>(config_.workspace_mib) * 1024U * 1024U;
    }
    std::vector<AlgoInfo> candidate_info() const {
        std::vector<AlgoInfo> result;
        result.reserve(candidates_.size());
        for (size_t index = 0; index < candidates_.size(); ++index) {
            if (candidates_[index].state == CUBLAS_STATUS_SUCCESS)
                result.push_back(
                    describe_algorithm(candidates_[index], static_cast<int32_t>(index)));
        }
        return result;
    }

  private:
    bool create_layout(cublasLtMatrixLayout_t* layout, uint64_t rows, uint64_t columns, int64_t ld,
                       const char* name) {
        if (!check(cublasLtMatrixLayoutCreate(layout, CUDA_R_16BF, rows, columns, ld),
                   (std::string("create ") + name + " layout").c_str()))
            return false;
        return true;
    }

    bool check(cublasStatus_t status, const char* operation) {
        if (status == CUBLAS_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed with cuBLAS status " +
                 std::to_string(static_cast<int>(status));
        return false;
    }

    void reset() {
        initialized_ = false;
        candidates_.clear();
        if (preference_ != nullptr)
            cublasLtMatmulPreferenceDestroy(preference_);
        if (d_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(d_layout_);
        if (c_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(c_layout_);
        if (b_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(b_layout_);
        if (a_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(a_layout_);
        if (operation_ != nullptr)
            cublasLtMatmulDescDestroy(operation_);
        if (handle_ != nullptr)
            cublasLtDestroy(handle_);
        preference_ = nullptr;
        d_layout_ = nullptr;
        c_layout_ = nullptr;
        b_layout_ = nullptr;
        a_layout_ = nullptr;
        operation_ = nullptr;
        handle_ = nullptr;
    }

    ProbeConfig config_{};
    cublasLtHandle_t handle_{nullptr};
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t a_layout_{nullptr};
    cublasLtMatrixLayout_t b_layout_{nullptr};
    cublasLtMatrixLayout_t c_layout_{nullptr};
    cublasLtMatrixLayout_t d_layout_{nullptr};
    cublasLtMatmulPreference_t preference_{nullptr};
    std::vector<cublasLtMatmulHeuristicResult_t> candidates_;
    AlgoInfo info_{};
    std::string error_;
    bool initialized_{false};
};

int32_t read_int32(const nvinfer1::PluginFieldCollection* fields, const char* name,
                   int32_t fallback) {
    if (fields == nullptr)
        return fallback;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        const auto& field = fields->fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0 && field.data != nullptr &&
            field.type == nvinfer1::PluginFieldType::kINT32 && field.length == 1) {
            int32_t value = 0;
            std::memcpy(&value, field.data, sizeof(value));
            return value;
        }
    }
    return fallback;
}

} // namespace

class DitLinearProbePlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitLinearProbe";
    static constexpr const char* kVERSION = "1";

    explicit DitLinearProbePlugin(ProbeConfig config) : config_(config) {}
    DitLinearProbePlugin(const void* data, size_t length) {
        if (data != nullptr && length == sizeof(config_))
            std::memcpy(&config_, data, sizeof(config_));
    }

    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return initialize_context(); }
    void terminate() noexcept override { context_.reset(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {
        if (initialize_context() != 0)
            std::fprintf(stderr, "Wan22DitLinearProbe attachToContext failed\n");
    }
    void detachFromContext() noexcept override { context_.reset(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return sizeof(config_); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &config_, sizeof(config_));
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kBF16;
    }
    DitLinearProbePlugin* clone() const noexcept override {
        auto* result = new DitLinearProbePlugin(config_);
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs result = inputs[0];
        if (result.nbDims > 0)
            result.d[result.nbDims - 1] = builder.constant(config_.n);
        return result;
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && position >= 0 && position < 4 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kBF16;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return static_cast<size_t>(std::max(config_.workspace_mib, 0)) * 1024U * 1024U;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        // TensorRT 11 may attach a deserialized V2 plugin without invoking the
        // legacy initialize() callback on that exact clone.  attachToContext()
        // is the primary lifecycle hook; this lazy guard is a last-resort
        // safety net and is exercised only before benchmark warmup.
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || inputs == nullptr || outputs == nullptr) {
            std::fprintf(stderr, "Wan22DitLinearProbe enqueue: context=%p inputs=%p outputs=%p\n",
                         static_cast<void*>(context_.get()), static_cast<const void*>(inputs),
                         static_cast<const void*>(outputs));
            return 1;
        }
        if (input_desc[0].dims.nbDims != 2 || input_desc[1].dims.nbDims != 2 ||
            input_desc[2].dims.nbDims != 1 || input_desc[0].dims.d[0] != config_.m ||
            input_desc[0].dims.d[1] != config_.k || input_desc[1].dims.d[0] != config_.n ||
            input_desc[1].dims.d[1] != config_.k || input_desc[2].dims.d[0] != config_.n) {
            std::fprintf(stderr,
                         "Wan22DitLinearProbe enqueue shape mismatch: x=[%d,%d] w=[%d,%d] "
                         "b=[%d] expected=[%d,%d]x[%d,%d]+[%d]\n",
                         input_desc[0].dims.d[0], input_desc[0].dims.d[1], input_desc[1].dims.d[0],
                         input_desc[1].dims.d[1], input_desc[2].dims.d[0], config_.m, config_.k,
                         config_.n, config_.k, config_.n);
            return 1;
        }
        const int32_t status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                             getWorkspaceSize(nullptr, 0, nullptr, 0), stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22DitLinearProbe enqueue: %s workspace=%p bytes=%zu\n",
                         context_->error().c_str(), workspace,
                         getWorkspaceSize(nullptr, 0, nullptr, 0));
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        context_ = std::make_unique<LinearContext>(config_);
        const int status = context_->initialize();
        if (status != 0) {
            std::fprintf(stderr, "Wan22DitLinearProbe initialize: %s\n", context_->error().c_str());
            context_.reset();
        }
        return status;
    }
    ProbeConfig config_{};
    std::unique_ptr<LinearContext> context_;
    std::string namespace_;
};

class DitLinearProbeCreator final : public nvinfer1::IPluginCreator {
  public:
    DitLinearProbeCreator() {
        field_entries_[0] = {"m", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[1] = {"n", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[2] = {"k", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[3] = {"heuristic_index", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[4] = {"workspace_mib", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {5, field_entries_};
    }
    char const* getPluginName() const noexcept override { return DitLinearProbePlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return DitLinearProbePlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        ProbeConfig config{};
        config.m = read_int32(fields, "m", 0);
        config.n = read_int32(fields, "n", 0);
        config.k = read_int32(fields, "k", 0);
        config.heuristic_index = read_int32(fields, "heuristic_index", 0);
        config.workspace_mib = read_int32(fields, "workspace_mib", 1024);
        return config.m > 0 && config.n > 0 && config.k > 0 && config.heuristic_index >= 0 &&
                       config.workspace_mib >= 0
                   ? new DitLinearProbePlugin(config)
                   : nullptr;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return length == sizeof(ProbeConfig) ? new DitLinearProbePlugin(data, length) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField field_entries_[5]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::linear_probe

static nvinfer1::PluginRegistrar<trtmc::wan22::linear_probe::DitLinearProbeCreator>
    plugin_registrar_wan22_dit_linear_probe{};

// The C ABI below is used only by the qualification script to enumerate and
// time cuBLASLt candidates without rebuilding an otherwise identical TensorRT
// plan for every candidate.  The winning candidate is then re-run through the
// actual TensorRT plugin before qualification is reported.
extern "C" {

using Wan22LinearProbeAlgoInfo = trtmc::wan22::linear_probe::AlgoInfo;

int trtmc_wan22_linear_probe_query(int32_t m, int32_t n, int32_t k, int32_t workspace_mib,
                                   Wan22LinearProbeAlgoInfo* output, int32_t capacity) {
    trtmc::wan22::linear_probe::ProbeConfig config{m, n, k, 0, workspace_mib};
    trtmc::wan22::linear_probe::LinearContext context(config);
    if (context.initialize() != 0) {
        std::fprintf(stderr, "Wan22 linear query: %s\n", context.error().c_str());
        return -1;
    }
    const auto candidates = context.candidate_info();
    if (output != nullptr && capacity > 0) {
        const int32_t copied = std::min(capacity, static_cast<int32_t>(candidates.size()));
        std::memcpy(output, candidates.data(), static_cast<size_t>(copied) * sizeof(*output));
    }
    return static_cast<int32_t>(candidates.size());
}

void* trtmc_wan22_linear_probe_create(int32_t m, int32_t n, int32_t k, int32_t heuristic_index,
                                      int32_t workspace_mib) {
    trtmc::wan22::linear_probe::ProbeConfig config{m, n, k, heuristic_index, workspace_mib};
    auto context = std::make_unique<trtmc::wan22::linear_probe::LinearContext>(config);
    if (context->initialize() != 0) {
        std::fprintf(stderr, "Wan22 linear create: %s\n", context->error().c_str());
        return nullptr;
    }
    return context.release();
}

void trtmc_wan22_linear_probe_destroy(void* opaque) {
    delete static_cast<trtmc::wan22::linear_probe::LinearContext*>(opaque);
}

uint64_t trtmc_wan22_linear_probe_workspace_bytes(void* opaque) {
    if (opaque == nullptr)
        return 0;
    return static_cast<trtmc::wan22::linear_probe::LinearContext*>(opaque)->info().workspace_bytes;
}

int trtmc_wan22_linear_probe_get_info(void* opaque, Wan22LinearProbeAlgoInfo* output) {
    if (opaque == nullptr || output == nullptr)
        return 1;
    *output = static_cast<trtmc::wan22::linear_probe::LinearContext*>(opaque)->info();
    return 0;
}

int trtmc_wan22_linear_probe_run(void* opaque, const void* x, const void* weight, const void* bias,
                                 void* output, void* workspace, uint64_t workspace_bytes,
                                 void* stream) {
    if (opaque == nullptr)
        return 1;
    return static_cast<trtmc::wan22::linear_probe::LinearContext*>(opaque)->run(
        x, weight, bias, output, workspace, static_cast<size_t>(workspace_bytes),
        static_cast<cudaStream_t>(stream));
}

void trtmc_wan22_dit_linear_probe_force_link() {}

} // extern "C"

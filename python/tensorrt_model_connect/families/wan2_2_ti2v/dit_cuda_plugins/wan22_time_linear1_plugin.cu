/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact first FP32 time-embedding linear for the fixed TI2V profile.
 * PyTorch addmm uses a cuBLASLt FP32 matmul with a fused bias epilogue and a
 * 32 MiB heuristic workspace pool.  Query the target device at runtime; never
 * serialize or hard-code the GB300 algorithm into a Thor plan.
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

#ifndef WAN22_TIME_LINEAR_NAMESPACE
#define WAN22_TIME_LINEAR_NAMESPACE time_linear1
#endif
#ifndef WAN22_TIME_LINEAR_PLUGIN_CLASS
#define WAN22_TIME_LINEAR_PLUGIN_CLASS TimeLinear1Plugin
#endif
#ifndef WAN22_TIME_LINEAR_CREATOR_CLASS
#define WAN22_TIME_LINEAR_CREATOR_CLASS TimeLinear1Creator
#endif
#ifndef WAN22_TIME_LINEAR_PLUGIN_NAME
#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitTimeLinear1"
#endif
#ifndef WAN22_TIME_LINEAR_INSTANCE_NAME
#define WAN22_TIME_LINEAR_INSTANCE_NAME "Wan22DitTimeLinear1"
#endif
#ifndef WAN22_TIME_LINEAR_M
#define WAN22_TIME_LINEAR_M 27'280
#endif
#ifndef WAN22_TIME_LINEAR_N
#define WAN22_TIME_LINEAR_N 3'072
#endif
#ifndef WAN22_TIME_LINEAR_K
#define WAN22_TIME_LINEAR_K 256
#endif
#ifndef WAN22_TIME_LINEAR_REGISTRAR
#define WAN22_TIME_LINEAR_REGISTRAR plugin_registrar_wan22_dit_time_linear1
#endif
#ifndef WAN22_TIME_LINEAR_PLAN_INFO_TYPE
#define WAN22_TIME_LINEAR_PLAN_INFO_TYPE Wan22DitTimeLinear1PlanInfo
#endif
#ifndef WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION
#define WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION trtmc_wan22_dit_time_linear1_plan_info
#endif

namespace trtmc::wan22::WAN22_TIME_LINEAR_NAMESPACE {

constexpr int32_t kM = WAN22_TIME_LINEAR_M;
constexpr int32_t kN = WAN22_TIME_LINEAR_N;
constexpr int32_t kK = WAN22_TIME_LINEAR_K;
constexpr size_t kWorkspaceLimitBytes = 32U * 1024U * 1024U;
constexpr int32_t kMaxHeuristics = 128;

struct PlanInfo {
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
    uint64_t algorithm_workspace_bytes{0};
    uint64_t workspace_limit_bytes{kWorkspaceLimitBytes};
    float waves_count{0.0F};
};

namespace {

template <typename T>
bool algorithm_attribute(const cublasLtMatmulAlgo_t& algorithm,
                         cublasLtMatmulAlgoConfigAttributes_t attribute, T* value) {
    size_t written = 0;
    return value != nullptr &&
           cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, value, sizeof(T),
                                                &written) == CUBLAS_STATUS_SUCCESS &&
           written == sizeof(T);
}

PlanInfo describe_algorithm(const cublasLtMatmulHeuristicResult_t& result, int32_t index) {
    PlanInfo info{};
    info.heuristic_index = index;
    info.algorithm_workspace_bytes = static_cast<uint64_t>(result.workspaceSize);
    info.workspace_limit_bytes = kWorkspaceLimitBytes;
    info.waves_count = result.wavesCount;
    algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_ID, &info.algorithm_id);

    uint32_t unsigned_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &unsigned_value))
        info.tile_id = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &unsigned_value))
        info.stages_id = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &unsigned_value))
        info.reduction_scheme = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &unsigned_value))
        info.cta_swizzle = static_cast<int32_t>(unsigned_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &unsigned_value))
        info.custom_option = static_cast<int32_t>(unsigned_value);

    int32_t signed_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &signed_value))
        info.split_k = signed_value;
    uint16_t short_value = 0;
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &short_value))
        info.inner_shape_id = static_cast<int32_t>(short_value);
    if (algorithm_attribute(result.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &short_value))
        info.cluster_shape_id = static_cast<int32_t>(short_value);
    return info;
}

class Context {
  public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { reset(); }

    int initialize() {
        reset();
        if (!check(cublasLtCreate(&handle_), "cublasLtCreate"))
            return 1;
        cublasOperation_t transpose_weight = CUBLAS_OP_T;
        cublasOperation_t transpose_input = CUBLAS_OP_N;
        if (!check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "cublasLtMatmulDescCreate") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &transpose_weight, sizeof(transpose_weight)),
                   "set TRANSA") ||
            !check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &transpose_input, sizeof(transpose_input)),
                   "set TRANSB"))
            return 1;

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)),
                   "set fused bias epilogue"))
            return 1;

        // Row-major X[M,K], W[N,K], and Y[M,N] are represented as
        // column-major X^T[K,M], W^T[K,N], and Y^T[N,M], matching addmm.
        if (!create_layout(&weight_layout_, kK, kN, kK, "weight") ||
            !create_layout(&input_layout_, kK, kM, kK, "input") ||
            !create_layout(&output_layout_, kN, kM, kN, "output"))
            return 1;
        if (!check(cublasLtMatmulPreferenceCreate(&preference_),
                   "cublasLtMatmulPreferenceCreate") ||
            !check(cublasLtMatmulPreferenceSetAttribute(
                       preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceLimitBytes,
                       sizeof(kWorkspaceLimitBytes)),
                   "set 32 MiB heuristic workspace"))
            return 1;

        candidates_.resize(kMaxHeuristics);
        int32_t returned = 0;
        if (!check(cublasLtMatmulAlgoGetHeuristic(
                       handle_, operation_, weight_layout_, input_layout_, output_layout_,
                       output_layout_, preference_, kMaxHeuristics, candidates_.data(), &returned),
                   "cublasLtMatmulAlgoGetHeuristic"))
            return 1;
        candidates_.resize(static_cast<size_t>(std::max(returned, 0)));
        for (size_t index = 0; index < candidates_.size(); ++index) {
            if (candidates_[index].state == CUBLAS_STATUS_SUCCESS) {
                selected_index_ = static_cast<int32_t>(index);
                info_ = describe_algorithm(candidates_[index], selected_index_);
                initialized_ = true;
                return 0;
            }
        }
        error_ = "target-local cuBLASLt query returned no usable FP32 fused-bias algorithm";
        return 1;
    }

    int run(const void* input, const void* weight, const void* bias, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || selected_index_ < 0 || input == nullptr || weight == nullptr ||
            bias == nullptr || output == nullptr || workspace == nullptr) {
            error_ = "run called with an uninitialized context or null pointer";
            return 1;
        }
        if (workspace_bytes < kWorkspaceLimitBytes) {
            error_ = "workspace is smaller than the official 32 MiB heuristic pool";
            return 1;
        }
        if (!check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias, sizeof(bias)),
                   "set bias pointer"))
            return 1;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto& selected = candidates_[static_cast<size_t>(selected_index_)];
        return check(cublasLtMatmul(handle_, operation_, &alpha, weight, weight_layout_, input,
                                    input_layout_, &beta, output, output_layout_, output,
                                    output_layout_, &selected.algo, workspace, workspace_bytes,
                                    stream),
                     "cublasLtMatmul")
                   ? 0
                   : 1;
    }

    const PlanInfo& info() const { return info_; }
    const std::string& error() const { return error_; }

  private:
    bool check(cublasStatus_t status, const char* operation) {
        if (status == CUBLAS_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed with cuBLAS status " +
                 std::to_string(static_cast<int32_t>(status));
        return false;
    }

    bool create_layout(cublasLtMatrixLayout_t* layout, uint64_t rows, uint64_t columns,
                       int64_t leading_dimension, const char* name) {
        return check(
            cublasLtMatrixLayoutCreate(layout, CUDA_R_32F, rows, columns, leading_dimension),
            (std::string("create ") + name + " layout").c_str());
    }

    void reset() {
        initialized_ = false;
        selected_index_ = -1;
        candidates_.clear();
        if (preference_ != nullptr)
            cublasLtMatmulPreferenceDestroy(preference_);
        if (output_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(output_layout_);
        if (input_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(input_layout_);
        if (weight_layout_ != nullptr)
            cublasLtMatrixLayoutDestroy(weight_layout_);
        if (operation_ != nullptr)
            cublasLtMatmulDescDestroy(operation_);
        if (handle_ != nullptr)
            cublasLtDestroy(handle_);
        preference_ = nullptr;
        output_layout_ = nullptr;
        input_layout_ = nullptr;
        weight_layout_ = nullptr;
        operation_ = nullptr;
        handle_ = nullptr;
        info_ = {};
    }

    cublasLtHandle_t handle_{nullptr};
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t weight_layout_{nullptr};
    cublasLtMatrixLayout_t input_layout_{nullptr};
    cublasLtMatrixLayout_t output_layout_{nullptr};
    cublasLtMatmulPreference_t preference_{nullptr};
    std::vector<cublasLtMatmulHeuristicResult_t> candidates_;
    int32_t selected_index_{-1};
    PlanInfo info_{};
    std::string error_;
    bool initialized_{false};
};

} // namespace

class WAN22_TIME_LINEAR_PLUGIN_CLASS final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = WAN22_TIME_LINEAR_PLUGIN_NAME;
    static constexpr const char* kVERSION = "1";

    WAN22_TIME_LINEAR_PLUGIN_CLASS() { initialize_context(); }
    WAN22_TIME_LINEAR_PLUGIN_CLASS(const void*, size_t) { initialize_context(); }
    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return initialize_context(); }
    void terminate() noexcept override { context_.reset(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {
        if (initialize_context() != 0)
            std::fprintf(stderr, "%s attachToContext failed\n", WAN22_TIME_LINEAR_INSTANCE_NAME);
    }
    void detachFromContext() noexcept override { context_.reset(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kFLOAT;
    }
    WAN22_TIME_LINEAR_PLUGIN_CLASS* clone() const noexcept override {
        auto* result = new WAN22_TIME_LINEAR_PLUGIN_CLASS();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output = inputs[0];
        if (output.nbDims == 2)
            output.d[1] = builder.constant(kN);
        return output;
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && position >= 0 && position < 4 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return kWorkspaceLimitBytes;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& x = input_desc[0].dims;
        const auto& weight = input_desc[1].dims;
        const auto& bias = input_desc[2].dims;
        if (x.nbDims != 2 || x.d[0] != kM || x.d[1] != kK || weight.nbDims != 2 ||
            weight.d[0] != kN || weight.d[1] != kK || bias.nbDims != 1 || bias.d[0] != kN) {
            std::fprintf(stderr, "%s input shape mismatch\n", WAN22_TIME_LINEAR_INSTANCE_NAME);
            return 1;
        }
        const int32_t status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                             kWorkspaceLimitBytes, stream);
        if (status != 0)
            std::fprintf(stderr, "%s enqueue: %s\n", WAN22_TIME_LINEAR_INSTANCE_NAME,
                         context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        context_ = std::make_unique<Context>();
        const int32_t status = context_->initialize();
        if (status != 0) {
            std::fprintf(stderr, "%s initialize: %s\n", WAN22_TIME_LINEAR_INSTANCE_NAME,
                         context_->error().c_str());
            context_.reset();
        }
        return status;
    }
    std::unique_ptr<Context> context_;
    std::string namespace_;
};

class WAN22_TIME_LINEAR_CREATOR_CLASS final : public nvinfer1::IPluginCreator {
  public:
    WAN22_TIME_LINEAR_CREATOR_CLASS() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override {
        return WAN22_TIME_LINEAR_PLUGIN_CLASS::kNAME;
    }
    char const* getPluginVersion() const noexcept override {
        return WAN22_TIME_LINEAR_PLUGIN_CLASS::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new WAN22_TIME_LINEAR_PLUGIN_CLASS();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return length == 0 ? new WAN22_TIME_LINEAR_PLUGIN_CLASS(data, length) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::WAN22_TIME_LINEAR_NAMESPACE

static nvinfer1::PluginRegistrar<
    trtmc::wan22::WAN22_TIME_LINEAR_NAMESPACE::WAN22_TIME_LINEAR_CREATOR_CLASS>
    WAN22_TIME_LINEAR_REGISTRAR{};

extern "C" {

using WAN22_TIME_LINEAR_PLAN_INFO_TYPE = trtmc::wan22::WAN22_TIME_LINEAR_NAMESPACE::PlanInfo;

int WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION(WAN22_TIME_LINEAR_PLAN_INFO_TYPE* output) {
    if (output == nullptr)
        return 1;
    trtmc::wan22::WAN22_TIME_LINEAR_NAMESPACE::Context context;
    if (context.initialize() != 0) {
        std::fprintf(stderr, "Wan22 time Linear1 plan query: %s\n", context.error().c_str());
        return 1;
    }
    *output = context.info();
    return 0;
}

} // extern "C"

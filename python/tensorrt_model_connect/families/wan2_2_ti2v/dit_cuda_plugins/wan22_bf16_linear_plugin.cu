/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact BF16 linear+bias for the five fixed Wan2.2 TI2V DiT shapes.
 * The target queries cuBLASLt at runtime and accepts only non-split-K
 * algorithms with no reduction scheme.  The qualified probe demonstrated that
 * every such candidate is bit-exact with upstream torch autocast addmm on
 * GB300.  Algorithm identifiers are deliberately never serialized; Thor
 * performs its own selection and still requires separate target qualification.
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::wan22::bf16_linear {

namespace {

constexpr int32_t kMaxHeuristics = 128;
constexpr size_t kWorkspaceLimitBytes = 0;

struct Config {
    int32_t m{0};
    int32_t n{0};
    int32_t k{0};
};

static_assert(sizeof(Config) == 3 * sizeof(int32_t));

template <typename T>
bool algorithm_attribute(const cublasLtMatmulAlgo_t& algorithm,
                         cublasLtMatmulAlgoConfigAttributes_t attribute, T* value) {
    size_t written = 0;
    return value != nullptr &&
           cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, value, sizeof(T),
                                                &written) == CUBLAS_STATUS_SUCCESS &&
           written == sizeof(T);
}

bool is_admissible(const cublasLtMatmulHeuristicResult_t& candidate) {
    if (candidate.state != CUBLAS_STATUS_SUCCESS)
        return false;
    int32_t split_k = 0;
    uint32_t reduction_scheme = 0;
    return algorithm_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k) &&
           algorithm_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                               &reduction_scheme) &&
           split_k == 1 && reduction_scheme == 0 && candidate.workspaceSize == 0;
}

bool is_qualified_shape(const Config& config) {
    return (config.m == 27'280 && config.k == 3'072 && (config.n == 3'072 || config.n == 14'336)) ||
           (config.m == 27'280 && config.k == 14'336 && config.n == 3'072) ||
           (config.m == 512 && config.k == 4'096 && config.n == 3'072) ||
           (config.m == 512 && config.k == 3'072 && config.n == 3'072);
}

class Context {
  public:
    explicit Context(Config config) : config_(config) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { reset(); }

    int initialize() {
        reset();
        if (!is_qualified_shape(config_)) {
            error_ = "shape is outside the five qualified Wan2.2 DiT BF16 linears";
            return 1;
        }
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
        // column-major X^T[K,M], W^T[K,N], and Y^T[N,M].  This is the TNN
        // addmm representation observed and qualified against upstream.
        if (!create_layout(&weight_layout_, config_.k, config_.n, config_.k, "weight") ||
            !create_layout(&input_layout_, config_.k, config_.m, config_.k, "input") ||
            !create_layout(&output_layout_, config_.n, config_.m, config_.n, "output"))
            return 1;

        if (!check(cublasLtMatmulPreferenceCreate(&preference_),
                   "cublasLtMatmulPreferenceCreate") ||
            !check(cublasLtMatmulPreferenceSetAttribute(
                       preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceLimitBytes,
                       sizeof(kWorkspaceLimitBytes)),
                   "set zero-workspace heuristic policy"))
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
            if (is_admissible(candidates_[index])) {
                selected_index_ = static_cast<int32_t>(index);
                selected_workspace_bytes_ = candidates_[index].workspaceSize;
                initialized_ = true;
                return 0;
            }
        }
        error_ = "target-local query returned no zero-workspace non-split-K/no-reduction algorithm";
        return 1;
    }

    int run(const void* input, const void* weight, const void* bias, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || selected_index_ < 0 || input == nullptr || weight == nullptr ||
            bias == nullptr || output == nullptr) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        if (selected_workspace_bytes_ > workspace_bytes ||
            (selected_workspace_bytes_ != 0 && workspace == nullptr)) {
            error_ = "workspace is smaller than the selected target-local algorithm requires";
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

    const std::string& error() const { return error_; }

  private:
    bool create_layout(cublasLtMatrixLayout_t* layout, uint64_t rows, uint64_t columns,
                       int64_t leading_dimension, const char* name) {
        return check(
            cublasLtMatrixLayoutCreate(layout, CUDA_R_16BF, rows, columns, leading_dimension),
            (std::string("create ") + name + " layout").c_str());
    }

    bool check(cublasStatus_t status, const char* operation) {
        if (status == CUBLAS_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed with cuBLAS status " +
                 std::to_string(static_cast<int32_t>(status));
        return false;
    }

    void reset() {
        initialized_ = false;
        selected_index_ = -1;
        selected_workspace_bytes_ = 0;
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
    }

    Config config_{};
    cublasLtHandle_t handle_{nullptr};
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t weight_layout_{nullptr};
    cublasLtMatrixLayout_t input_layout_{nullptr};
    cublasLtMatrixLayout_t output_layout_{nullptr};
    cublasLtMatmulPreference_t preference_{nullptr};
    std::vector<cublasLtMatmulHeuristicResult_t> candidates_;
    int32_t selected_index_{-1};
    size_t selected_workspace_bytes_{0};
    std::string error_;
    bool initialized_{false};
};

int32_t read_int32(const nvinfer1::PluginFieldCollection* fields, const char* name) {
    if (fields == nullptr)
        return 0;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        const auto& field = fields->fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0 && field.data != nullptr &&
            field.type == nvinfer1::PluginFieldType::kINT32 && field.length == 1) {
            int32_t value = 0;
            std::memcpy(&value, field.data, sizeof(value));
            return value;
        }
    }
    return 0;
}

} // namespace

class Bf16LinearPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitBf16Linear";
    static constexpr const char* kVERSION = "1";

    explicit Bf16LinearPlugin(Config config) : config_(config) {}
    Bf16LinearPlugin(const void* data, size_t length) {
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
            std::fprintf(stderr, "Wan22DitBf16Linear attachToContext failed\n");
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
    Bf16LinearPlugin* clone() const noexcept override {
        auto* result = new Bf16LinearPlugin(config_);
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output = inputs[0];
        if (output.nbDims == 2)
            output.d[1] = builder.constant(config_.n);
        return output;
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
        if (x.nbDims != 2 || x.d[0] != config_.m || x.d[1] != config_.k || weight.nbDims != 2 ||
            weight.d[0] != config_.n || weight.d[1] != config_.k || bias.nbDims != 1 ||
            bias.d[0] != config_.n) {
            std::fprintf(stderr, "Wan22DitBf16Linear input shape mismatch\n");
            return 1;
        }
        const int32_t status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                             kWorkspaceLimitBytes, stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22DitBf16Linear enqueue: %s\n", context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        context_.reset();
        try {
            auto context = std::make_unique<Context>(config_);
            const int32_t status = context->initialize();
            if (status != 0) {
                std::fprintf(stderr, "Wan22DitBf16Linear initialize: %s\n",
                             context->error().c_str());
                context_.reset();
                return status;
            }
            context_ = std::move(context);
            return 0;
        } catch (const std::exception& exception) {
            std::fprintf(stderr, "Wan22DitBf16Linear initialize exception: %s\n", exception.what());
            return 1;
        } catch (...) {
            std::fprintf(stderr, "Wan22DitBf16Linear initialize unknown exception\n");
            return 1;
        }
    }

    Config config_{};
    std::unique_ptr<Context> context_;
    std::string namespace_;
};

class Bf16LinearCreator final : public nvinfer1::IPluginCreator {
  public:
    Bf16LinearCreator() {
        field_entries_[0] = {"m", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[1] = {"n", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[2] = {"k", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {3, field_entries_};
    }
    char const* getPluginName() const noexcept override { return Bf16LinearPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return Bf16LinearPlugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        Config config{read_int32(fields, "m"), read_int32(fields, "n"), read_int32(fields, "k")};
        return is_qualified_shape(config) ? new Bf16LinearPlugin(config) : nullptr;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        if (data == nullptr || length != sizeof(Config))
            return nullptr;
        Config config{};
        std::memcpy(&config, data, sizeof(config));
        return is_qualified_shape(config) ? new Bf16LinearPlugin(config) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField field_entries_[3]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::bf16_linear

static nvinfer1::PluginRegistrar<trtmc::wan22::bf16_linear::Bf16LinearCreator>
    plugin_registrar_wan22_dit_bf16_linear{};

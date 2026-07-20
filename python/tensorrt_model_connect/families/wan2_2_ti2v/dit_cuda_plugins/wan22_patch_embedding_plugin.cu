/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact Wan2.2 TI2V-5B patch embedding for the one production profile.
 * PyTorch autocast executes a bias-free cuDNN Conv3d with FP32 accumulation,
 * materializes BF16 NCDHW output, then launches a separate BF16 bias add.  Do
 * not fuse that bias into the cuDNN graph without a new full-output proof.
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::wan22::patch_embedding {

constexpr int32_t kN = 1;
constexpr int32_t kC = 48;
constexpr int32_t kD = 31;
constexpr int32_t kH = 44;
constexpr int32_t kW = 80;
constexpr int32_t kK = 3072;
constexpr int32_t kOutD = 31;
constexpr int32_t kOutH = 22;
constexpr int32_t kOutW = 40;
constexpr int32_t kRows = kOutD * kOutH * kOutW;
constexpr int64_t kOutputElements = static_cast<int64_t>(kK) * kRows;
constexpr int64_t kUidWeight = 119;
constexpr int64_t kUidInput = 120;
constexpr int64_t kUidOutput = 121;

struct PlanInfo {
    int32_t heuristic_index{-1};
    int32_t reserved{0};
    int64_t engine_id{-1};
    uint64_t workspace_bytes{0};
    uint64_t cudnn_version{0};
};

namespace {

const char* status_name(cudnnStatus_t status) {
    const char* value = cudnnGetErrorString(status);
    return value != nullptr ? value : "unknown cuDNN status";
}

void destroy_backend(cudnnBackendDescriptor_t& descriptor) {
    if (descriptor != nullptr)
        cudnnBackendDestroyDescriptor(descriptor);
    descriptor = nullptr;
}

template <typename T>
bool set_attribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                   cudnnBackendAttributeType_t type, int64_t count, const T* value,
                   std::string& error) {
    const cudnnStatus_t status = cudnnBackendSetAttribute(descriptor, name, type, count, value);
    if (status == CUDNN_STATUS_SUCCESS)
        return true;
    error = "cudnnBackendSetAttribute(" + std::to_string(static_cast<int>(name)) +
            ") failed: " + status_name(status);
    return false;
}

bool finalize(cudnnBackendDescriptor_t descriptor, const char* name, std::string& error) {
    const cudnnStatus_t status = cudnnBackendFinalize(descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        return true;
    error = std::string("cudnnBackendFinalize(") + name + ") failed: " + status_name(status);
    return false;
}

int64_t engine_id_from_json(const std::string& value) {
    constexpr const char* marker = "\"engineId\":";
    const size_t position = value.find(marker);
    if (position == std::string::npos)
        return -1;
    const char* begin = value.c_str() + position + std::strlen(marker);
    char* end = nullptr;
    const long long parsed = std::strtoll(begin, &end, 10);
    return end != begin ? static_cast<int64_t>(parsed) : -1;
}

__global__ void add_bias_ncdhw(__nv_bfloat16* tensor, const __nv_bfloat16* bias) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < kOutputElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t channel = index / kRows;
        tensor[index] =
            __float2bfloat16_rn(__bfloat162float(tensor[index]) + __bfloat162float(bias[channel]));
    }
}

class Context {
  public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { reset(); }

    int initialize() {
        reset();
        if (!check(cudnnCreate(&handle_), "cudnnCreate"))
            return 1;
        if (!make_tensor(x_desc_, {kN, kC, kD, kH, kW},
                         {kC * kD * kH * kW, kD * kH * kW, kH * kW, kW, 1}, kUidInput, "x") ||
            !make_tensor(w_desc_, {kK, kC, 1, 2, 2}, {kC * 4, 4, 4, 2, 1}, kUidWeight, "w") ||
            !make_tensor(y_desc_, {kN, kK, kOutD, kOutH, kOutW},
                         {kK * kRows, kRows, kOutH * kOutW, kOutW, 1}, kUidOutput, "y") ||
            !make_convolution() || !make_operation_graph() || !select_plan())
            return 1;
        initialized_ = true;
        return 0;
    }

    int run(const void* latent, const void* weight, const void* bias, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || plan_ == nullptr || latent == nullptr || weight == nullptr ||
            bias == nullptr || output == nullptr ||
            (plan_workspace_bytes_ != 0 && workspace == nullptr)) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        if (workspace_bytes < plan_workspace_bytes_) {
            error_ = "workspace is smaller than the selected cuDNN execution plan";
            return 1;
        }
        if (!check(cudnnSetStream(handle_, stream), "cudnnSetStream"))
            return 1;
        if (!prepare_variant_pack(latent, weight, output, workspace))
            return 1;
        if (!check(cudnnBackendExecute(handle_, plan_, variant_pack_), "cudnnBackendExecute"))
            return 1;

        // Official PyTorch semantics: the cuDNN graph has no bias operation.
        // Add BF16 bias only after cuDNN has materialized BF16 NCDHW output.
        constexpr int32_t threads = 256;
        constexpr int32_t max_blocks = 65535;
        const int64_t required = (kOutputElements + threads - 1) / threads;
        const int32_t blocks = static_cast<int32_t>(std::min<int64_t>(required, max_blocks));
        add_bias_ncdhw<<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16*>(output),
                                                       static_cast<const __nv_bfloat16*>(bias));
        const cudaError_t cuda_status = cudaPeekAtLastError();
        if (cuda_status != cudaSuccess) {
            error_ =
                std::string("separate BF16 bias add failed: ") + cudaGetErrorString(cuda_status);
            return 1;
        }
        return 0;
    }

    size_t workspace_bytes() const { return plan_workspace_bytes_; }
    int32_t heuristic_index() const { return heuristic_index_; }
    int64_t engine_id() const { return engine_id_; }
    uint64_t cudnn_version() const { return static_cast<uint64_t>(cudnnGetVersion()); }
    const std::string& plan_json() const { return plan_json_; }
    const std::string& error() const { return error_; }

  private:
    bool check(cudnnStatus_t status, const char* operation) {
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        error_ = std::string(operation) + " failed: " + status_name(status);
        return false;
    }

    bool create_backend(cudnnBackendDescriptorType_t type, cudnnBackendDescriptor_t& output,
                        const char* name) {
        const cudnnStatus_t status = cudnnBackendCreateDescriptor(type, &output);
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        error_ = std::string("create ") + name + " failed: " + status_name(status);
        return false;
    }

    bool make_tensor(cudnnBackendDescriptor_t& output, std::array<int64_t, 5> dimensions,
                     std::array<int64_t, 5> strides, int64_t uid, const char* name) {
        if (!create_backend(CUDNN_BACKEND_TENSOR_DESCRIPTOR, output, name))
            return false;
        const cudnnDataType_t type = CUDNN_DATA_BFLOAT16;
        const int64_t alignment = 32;
        const bool is_virtual = false;
        return set_attribute(output, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type,
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5,
                             dimensions.data(), error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 5, strides.data(),
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid,
                             error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1,
                             &alignment, error_) &&
               set_attribute(output, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1,
                             &is_virtual, error_) &&
               finalize(output, name, error_);
    }

    bool make_convolution() {
        if (!create_backend(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, conv_desc_, "convolution"))
            return false;
        const cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
        const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        const int64_t spatial_dims = 3;
        const std::array<int64_t, 3> pads{0, 0, 0};
        const std::array<int64_t, 3> strides{1, 2, 2};
        const std::array<int64_t, 3> dilations{1, 1, 1};
        return set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1,
                             &compute_type, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                             CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1,
                             &spatial_dims, error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 3,
                             pads.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 3,
                             pads.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 3,
                             strides.data(), error_) &&
               set_attribute(conv_desc_, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 3,
                             dilations.data(), error_) &&
               finalize(conv_desc_, "convolution", error_);
    }

    bool make_operation_graph() {
        if (!create_backend(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, operation_,
                            "convolution forward operation"))
            return false;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        if (!set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc_, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                           CUDNN_TYPE_FLOAT, 1, &alpha, error_) ||
            !set_attribute(operation_, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                           CUDNN_TYPE_FLOAT, 1, &beta, error_) ||
            !finalize(operation_, "convolution forward operation", error_))
            return false;

        if (!create_backend(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, operation_graph_,
                            "operation graph"))
            return false;
        const std::array<cudnnBackendDescriptor_t, 1> operations{operation_};
        return set_attribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                             CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, operations.data(), error_) &&
               set_attribute(operation_graph_, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,
                             1, &handle_, error_) &&
               finalize(operation_graph_, "operation graph", error_);
    }

    bool select_plan() {
        if (!create_backend(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, heuristic_, "heuristic"))
            return false;
        const cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
        if (!set_attribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_graph_, error_) ||
            !set_attribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode,
                           error_) ||
            !finalize(heuristic_, "heuristic", error_))
            return false;

        int64_t count = 0;
        if (!check(cudnnBackendGetAttribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr),
                   "query heuristic result count"))
            return false;
        if (count <= 0 || count > 1024) {
            error_ = "cuDNN returned an invalid heuristic result count";
            return false;
        }
        configs_.resize(static_cast<size_t>(count), nullptr);
        for (auto& config : configs_) {
            if (!create_backend(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, config, "engine config"))
                return false;
        }
        int64_t returned = 0;
        if (!check(cudnnBackendGetAttribute(heuristic_, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &returned,
                                            configs_.data()),
                   "get heuristic results"))
            return false;
        configs_.resize(static_cast<size_t>(returned));

        // PyTorch's official path uses the first finalizable INSTANT config.
        // Re-query on every target instead of serializing an SM103 engine.
        for (size_t index = 0; index < configs_.size(); ++index) {
            cudnnBackendDescriptor_t candidate = nullptr;
            size_t workspace = 0;
            const cudnnStatus_t status = make_plan(configs_[index], candidate, workspace);
            if (status == CUDNN_STATUS_SUCCESS) {
                plan_ = candidate;
                plan_workspace_bytes_ = workspace;
                heuristic_index_ = static_cast<int32_t>(index);
                plan_json_ = read_plan_json(plan_);
                engine_id_ = engine_id_from_json(plan_json_);
                return true;
            }
            destroy_backend(candidate);
        }
        error_ = "no cuDNN INSTANT engine config produced a finalizable execution plan";
        return false;
    }

    cudnnStatus_t make_plan(cudnnBackendDescriptor_t config, cudnnBackendDescriptor_t& output,
                            size_t& workspace) {
        workspace = 0;
        cudnnStatus_t status =
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &output);
        if (status != CUDNN_STATUS_SUCCESS)
            return status;
        status = cudnnBackendSetAttribute(output, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
        if (status == CUDNN_STATUS_SUCCESS)
            status = cudnnBackendFinalize(output);
        if (status == CUDNN_STATUS_SUCCESS) {
            int64_t bytes = 0;
            int64_t count = 0;
            status = cudnnBackendGetAttribute(output, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                              CUDNN_TYPE_INT64, 1, &count, &bytes);
            if (status == CUDNN_STATUS_SUCCESS && count == 1 && bytes >= 0)
                workspace = static_cast<size_t>(bytes);
        }
        return status;
    }

    std::string read_plan_json(cudnnBackendDescriptor_t plan) {
        int64_t count = 0;
        if (cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, 0, &count, nullptr) != CUDNN_STATUS_SUCCESS ||
            count <= 0)
            return {};
        std::string result(static_cast<size_t>(count), '\0');
        if (cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, count, &count,
                                     result.data()) != CUDNN_STATUS_SUCCESS)
            return {};
        while (!result.empty() && result.back() == '\0')
            result.pop_back();
        return result;
    }

    bool prepare_variant_pack(const void* latent, const void* weight, void* output,
                              void* workspace) {
        if (variant_pack_ != nullptr && latent == cached_latent_ && weight == cached_weight_ &&
            output == cached_output_ && workspace == cached_workspace_)
            return true;
        destroy_backend(variant_pack_);
        if (!create_backend(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, variant_pack_, "variant pack"))
            return false;
        const std::array<int64_t, 3> uids{kUidInput, kUidWeight, kUidOutput};
        const std::array<void*, 3> pointers{const_cast<void*>(latent), const_cast<void*>(weight),
                                            output};
        return set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3,
                             uids.data(), error_) &&
               set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             CUDNN_TYPE_VOID_PTR, 3, pointers.data(), error_) &&
               set_attribute(variant_pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR,
                             1, &workspace, error_) &&
               finalize(variant_pack_, "variant pack", error_) &&
               cache_variant_pointers(latent, weight, output, workspace);
    }

    bool cache_variant_pointers(const void* latent, const void* weight, void* output,
                                void* workspace) {
        cached_latent_ = latent;
        cached_weight_ = weight;
        cached_output_ = output;
        cached_workspace_ = workspace;
        return true;
    }

    void reset() {
        initialized_ = false;
        destroy_backend(variant_pack_);
        destroy_backend(plan_);
        for (auto& config : configs_)
            destroy_backend(config);
        configs_.clear();
        destroy_backend(heuristic_);
        destroy_backend(operation_graph_);
        destroy_backend(operation_);
        destroy_backend(conv_desc_);
        destroy_backend(y_desc_);
        destroy_backend(w_desc_);
        destroy_backend(x_desc_);
        if (handle_ != nullptr)
            cudnnDestroy(handle_);
        handle_ = nullptr;
        cached_latent_ = nullptr;
        cached_weight_ = nullptr;
        cached_output_ = nullptr;
        cached_workspace_ = nullptr;
        plan_workspace_bytes_ = 0;
        heuristic_index_ = -1;
        engine_id_ = -1;
        plan_json_.clear();
    }

    cudnnHandle_t handle_{nullptr};
    cudnnBackendDescriptor_t x_desc_{nullptr};
    cudnnBackendDescriptor_t w_desc_{nullptr};
    cudnnBackendDescriptor_t y_desc_{nullptr};
    cudnnBackendDescriptor_t conv_desc_{nullptr};
    cudnnBackendDescriptor_t operation_{nullptr};
    cudnnBackendDescriptor_t operation_graph_{nullptr};
    cudnnBackendDescriptor_t heuristic_{nullptr};
    std::vector<cudnnBackendDescriptor_t> configs_;
    cudnnBackendDescriptor_t plan_{nullptr};
    cudnnBackendDescriptor_t variant_pack_{nullptr};
    size_t plan_workspace_bytes_{0};
    int32_t heuristic_index_{-1};
    int64_t engine_id_{-1};
    std::string plan_json_;
    std::string error_;
    const void* cached_latent_{nullptr};
    const void* cached_weight_{nullptr};
    void* cached_output_{nullptr};
    void* cached_workspace_{nullptr};
    bool initialized_{false};
};

} // namespace

class PatchEmbeddingPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitPatchEmbedding";
    static constexpr const char* kVERSION = "1";

    PatchEmbeddingPlugin() { initialize_context(); }
    PatchEmbeddingPlugin(const void*, size_t) { initialize_context(); }
    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return initialize_context(); }
    void terminate() noexcept override { context_.reset(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {
        if (initialize_context() != 0)
            std::fprintf(stderr, "Wan22DitPatchEmbedding attachToContext failed\n");
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
        return nvinfer1::DataType::kBF16;
    }
    PatchEmbeddingPlugin* clone() const noexcept override {
        auto* result = new PatchEmbeddingPlugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const*, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output{};
        output.nbDims = 5;
        output.d[0] = builder.constant(kN);
        output.d[1] = builder.constant(kK);
        output.d[2] = builder.constant(kOutD);
        output.d[3] = builder.constant(kOutH);
        output.d[4] = builder.constant(kOutW);
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
        return context_ != nullptr ? context_->workspace_bytes() : 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& x = input_desc[0].dims;
        const auto& w = input_desc[1].dims;
        const auto& b = input_desc[2].dims;
        if (x.nbDims != 5 || x.d[0] != kN || x.d[1] != kC || x.d[2] != kD || x.d[3] != kH ||
            x.d[4] != kW || w.nbDims != 5 || w.d[0] != kK || w.d[1] != kC || w.d[2] != 1 ||
            w.d[3] != 2 || w.d[4] != 2 || b.nbDims != 1 || b.d[0] != kK) {
            std::fprintf(stderr, "Wan22DitPatchEmbedding input shape mismatch\n");
            return 1;
        }
        const int status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                         getWorkspaceSize(nullptr, 0, nullptr, 0), stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22DitPatchEmbedding enqueue: %s\n", context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        context_ = std::make_unique<Context>();
        const int status = context_->initialize();
        if (status != 0) {
            std::fprintf(stderr, "Wan22DitPatchEmbedding initialize: %s\n",
                         context_->error().c_str());
            context_.reset();
        }
        return status;
    }
    std::unique_ptr<Context> context_;
    std::string namespace_;
};

class PatchEmbeddingCreator final : public nvinfer1::IPluginCreator {
  public:
    PatchEmbeddingCreator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return PatchEmbeddingPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return PatchEmbeddingPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new PatchEmbeddingPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return length == 0 ? new PatchEmbeddingPlugin(data, length) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::patch_embedding

static nvinfer1::PluginRegistrar<trtmc::wan22::patch_embedding::PatchEmbeddingCreator>
    plugin_registrar_wan22_dit_patch_embedding{};

extern "C" {

using Wan22DitPatchPlanInfo = trtmc::wan22::patch_embedding::PlanInfo;

int trtmc_wan22_dit_patch_plan_info(Wan22DitPatchPlanInfo* output) {
    if (output == nullptr)
        return 1;
    trtmc::wan22::patch_embedding::Context context;
    if (context.initialize() != 0) {
        std::fprintf(stderr, "Wan22 patch plan query: %s\n", context.error().c_str());
        return 1;
    }
    output->heuristic_index = context.heuristic_index();
    output->reserved = 0;
    output->engine_id = context.engine_id();
    output->workspace_bytes = context.workspace_bytes();
    output->cudnn_version = context.cudnn_version();
    return 0;
}

int trtmc_wan22_dit_patch_plan_json(char* output, int32_t capacity) {
    trtmc::wan22::patch_embedding::Context context;
    if (context.initialize() != 0)
        return -1;
    const std::string& value = context.plan_json();
    const int32_t required = static_cast<int32_t>(value.size() + 1);
    if (output != nullptr && capacity > 0) {
        const int32_t copied = std::min(capacity - 1, static_cast<int32_t>(value.size()));
        if (copied > 0)
            std::memcpy(output, value.data(), static_cast<size_t>(copied));
        output[std::max(copied, 0)] = '\0';
    }
    return required;
}

} // extern "C"

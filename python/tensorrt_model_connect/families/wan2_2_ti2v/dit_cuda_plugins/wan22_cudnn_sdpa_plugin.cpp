/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact Wan2.2 TI2V-5B self/cross SDPA for the fixed 720p profile.
 * This reproduces the cuDNN frontend graph used by PyTorch's cuDNN SDPA path,
 * but has no torch, ATen, c10, or Python runtime dependency.
 */

#include "wan22_cudnn_sdpa_plugin.h"

#include <NvInferRuntime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(CUDNN_FRONTEND_MAJOR_VERSION) || !defined(CUDNN_FRONTEND_MINOR_VERSION) ||            \
    !defined(CUDNN_FRONTEND_PATCH_VERSION)
#error "Wan22DitCudnnSdpa requires versioned cuDNN frontend headers"
#endif
static_assert(CUDNN_FRONTEND_MAJOR_VERSION == 1 && CUDNN_FRONTEND_MINOR_VERSION == 22 &&
                  CUDNN_FRONTEND_PATCH_VERSION == 1,
              "Wan22DitCudnnSdpa is qualified with cuDNN frontend 1.22.1");

namespace trtmc::wan22::cudnn_sdpa {
namespace fe = cudnn_frontend;

namespace {

constexpr int64_t kUidQ = 0;
constexpr int64_t kUidK = 1;
constexpr int64_t kUidV = 2;
constexpr int64_t kUidOutput = 3;
constexpr int64_t kUidScale = 5;
constexpr uint32_t kAllRequiredFields = (1U << 6U) - 1U;

const char* attention_kind_name(const Config& config) noexcept {
    return config.attention_kind == static_cast<int32_t>(AttentionKind::kCross) ? "cross" : "self";
}

bool read_config_fields(const nvinfer1::PluginFieldCollection* fields, Config& output) noexcept {
    if (fields == nullptr || fields->nbFields < 0 ||
        (fields->nbFields != 0 && fields->fields == nullptr))
        return false;

    Config parsed{};
    uint32_t present = 0;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        const auto& field = fields->fields[index];
        if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
            field.type != nvinfer1::PluginFieldType::kINT32)
            return false;

        int32_t value = 0;
        std::memcpy(&value, field.data, sizeof(value));
        int32_t* target = nullptr;
        uint32_t bit = 0;
        if (std::strcmp(field.name, kAttentionKindField) == 0) {
            target = &parsed.attention_kind;
            bit = 1U << 0U;
        } else if (std::strcmp(field.name, kBatchField) == 0) {
            target = &parsed.batch;
            bit = 1U << 1U;
        } else if (std::strcmp(field.name, kHeadsField) == 0) {
            target = &parsed.heads;
            bit = 1U << 2U;
        } else if (std::strcmp(field.name, kQSequenceField) == 0) {
            target = &parsed.q_sequence;
            bit = 1U << 3U;
        } else if (std::strcmp(field.name, kKvSequenceField) == 0) {
            target = &parsed.kv_sequence;
            bit = 1U << 4U;
        } else if (std::strcmp(field.name, kHeadDimensionField) == 0) {
            target = &parsed.head_dimension;
            bit = 1U << 5U;
        } else {
            return false;
        }
        if ((present & bit) != 0U)
            return false;
        *target = value;
        present |= bit;
    }
    if (present != kAllRequiredFields || !is_qualified_config(parsed))
        return false;
    output = parsed;
    return true;
}

bool deserialize_config(const void* data, size_t length, Config& output) noexcept {
    if (data == nullptr || length != sizeof(SerializedConfig))
        return false;
    SerializedConfig serialized{};
    std::memcpy(&serialized, data, sizeof(serialized));
    if (!is_valid_serialized_config(serialized))
        return false;
    output = serialized.config;
    return true;
}

bool dimensions_match(const nvinfer1::Dims& actual, int32_t batch, int32_t sequence, int32_t heads,
                      int32_t dimension) noexcept {
    return actual.nbDims == 4 && actual.d[0] == batch && actual.d[1] == sequence &&
           actual.d[2] == heads && actual.d[3] == dimension;
}

} // namespace

class Context {
  public:
    explicit Context(Config config) : config_(config) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() { reset(); }

    int initialize() {
        reset();
        if (!is_qualified_config(config_)) {
            error_ = "configuration is outside the two qualified Wan2.2 SDPA contracts";
            return 1;
        }
        const cudnnStatus_t create_status = cudnnCreate(&handle_);
        if (create_status != CUDNN_STATUS_SUCCESS) {
            error_ = std::string("cudnnCreate failed: ") + cudnnGetErrorString(create_status);
            return 1;
        }

        graph_ = std::make_unique<fe::graph::Graph>();
        graph_->set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        const int64_t b = config_.batch;
        const int64_t h = config_.heads;
        const int64_t q_sequence = config_.q_sequence;
        const int64_t kv_sequence = config_.kv_sequence;
        const int64_t d = config_.head_dimension;
        const std::vector<int64_t> q_dimensions{b, h, q_sequence, d};
        const std::vector<int64_t> kv_dimensions{b, h, kv_sequence, d};

        // TensorRT supplies physically contiguous BSHD tensors.  Official Wan
        // transposes BSHD -> BHSD as a view; these logical strides preserve
        // that exact storage without an extra materialization.
        const std::vector<int64_t> q_strides{h * q_sequence * d, d, h * d, 1};
        const std::vector<int64_t> kv_strides{h * kv_sequence * d, d, h * d, 1};
        q_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kUidQ)
                                .set_name("Q")
                                .set_dim(q_dimensions)
                                .set_stride(q_strides));
        k_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kUidK)
                                .set_name("K")
                                .set_dim(kv_dimensions)
                                .set_stride(kv_strides));
        v_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kUidV)
                                .set_name("V")
                                .set_dim(kv_dimensions)
                                .set_stride(kv_strides));
        scale_tensor_ = graph_->tensor(fe::graph::Tensor_attributes()
                                           .set_uid(kUidScale)
                                           .set_name("Attn_scale")
                                           .set_dim({1, 1, 1, 1})
                                           .set_stride({1, 1, 1, 1})
                                           .set_is_pass_by_value(true)
                                           .set_data_type(fe::DataType_t::FLOAT));

        auto attributes = fe::graph::SDPA_attributes()
                              .set_name("CUDNN_SDPA")
                              .set_generate_stats(false)
                              .set_causal_mask(false)
                              .set_attn_scale(scale_tensor_);
        auto result = graph_->sdpa(q_, k_, v_, attributes);
        output_ = result[0];
        output_->set_uid(kUidOutput).set_output(true).set_dim(q_dimensions).set_stride(q_strides);

        if (!check(graph_->validate(), "validate") ||
            !check(graph_->build_operation_graph(handle_), "build_operation_graph"))
            return 1;

        // Match the official PyTorch cuDNN path: query Heuristic Mode A on the
        // current target and execute the first supported plan.  No GB300 plan
        // identifier or knob value is compiled into or serialized by this
        // plugin, so Thor performs its own selection.
        if (!check(graph_->create_execution_plans({fe::HeurMode_t::A}),
                   "create_execution_plans(HeurMode A)"))
            return 1;
        candidate_count_ = graph_->get_execution_plan_count();
        if (!check(graph_->check_support(handle_), "check_support") ||
            !check(graph_->build_plans(handle_), "build_plans") ||
            !check(graph_->get_plan_name(plan_name_), "get_plan_name") ||
            !check(graph_->get_workspace_size(workspace_size_), "get_workspace_size"))
            return 1;
        if (workspace_size_ < 0) {
            error_ = "cuDNN returned a negative workspace size";
            return 1;
        }

        scale_ = 1.0F / std::sqrt(static_cast<float>(config_.head_dimension));
        initialized_ = true;
        return 0;
    }

    int run(const void* q, const void* k, const void* v, void* output, void* workspace,
            cudaStream_t stream) {
        if (!initialized_ || q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
            (workspace_size_ != 0 && workspace == nullptr)) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        const cudnnStatus_t stream_status = cudnnSetStream(handle_, stream);
        if (stream_status != CUDNN_STATUS_SUCCESS) {
            error_ = std::string("cudnnSetStream failed: ") + cudnnGetErrorString(stream_status);
            return 1;
        }
        std::unordered_map<int64_t, void*> variant_pack = {
            {kUidQ, const_cast<void*>(q)}, {kUidK, const_cast<void*>(k)},
            {kUidV, const_cast<void*>(v)}, {kUidScale, &scale_},
            {kUidOutput, output},
        };
        return check(graph_->execute(handle_, variant_pack, workspace), "execute") ? 0 : 1;
    }

    size_t workspace_size() const noexcept { return static_cast<size_t>(workspace_size_); }
    const std::string& error() const noexcept { return error_; }

  private:
    bool check(fe::error_t status, const char* operation) {
        if (status.is_good())
            return true;
        error_ = std::string(operation) + " failed: " + status.get_message();
        return false;
    }

    void reset() noexcept {
        initialized_ = false;
        output_.reset();
        scale_tensor_.reset();
        v_.reset();
        k_.reset();
        q_.reset();
        graph_.reset();
        if (handle_ != nullptr)
            cudnnDestroy(handle_);
        handle_ = nullptr;
        workspace_size_ = 0;
        candidate_count_ = 0;
        plan_name_.clear();
        error_.clear();
    }

    Config config_{};
    cudnnHandle_t handle_{nullptr};
    std::unique_ptr<fe::graph::Graph> graph_;
    std::shared_ptr<fe::graph::Tensor_attributes> q_;
    std::shared_ptr<fe::graph::Tensor_attributes> k_;
    std::shared_ptr<fe::graph::Tensor_attributes> v_;
    std::shared_ptr<fe::graph::Tensor_attributes> scale_tensor_;
    std::shared_ptr<fe::graph::Tensor_attributes> output_;
    float scale_{0.0F};
    int64_t workspace_size_{0};
    int64_t candidate_count_{0};
    std::string plan_name_;
    std::string error_;
    bool initialized_{false};
};

class CudnnSdpaPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    explicit CudnnSdpaPlugin(Config config)
        : config_(config), config_valid_(is_qualified_config(config)) {
        initialize_context();
    }

    CudnnSdpaPlugin(const void* data, size_t length) {
        config_valid_ = deserialize_config(data, length, config_);
        if (config_valid_)
            initialize_context();
    }

    char const* getPluginType() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return initialize_context(); }
    void terminate() noexcept override { context_.reset(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {
        if (initialize_context() != 0)
            std::fprintf(stderr, "Wan22DitCudnnSdpa attachToContext failed\n");
    }
    void detachFromContext() noexcept override { context_.reset(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return sizeof(SerializedConfig); }
    void serialize(void* buffer) const noexcept override {
        if (buffer == nullptr)
            return;
        const SerializedConfig serialized = make_serialized_config(config_);
        std::memcpy(buffer, &serialized, sizeof(serialized));
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value != nullptr ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t output_index,
                                         nvinfer1::DataType const* input_types,
                                         int32_t input_count) const noexcept override {
        if (output_index != 0 || input_types == nullptr || input_count != 3)
            return nvinfer1::DataType::kBF16;
        return nvinfer1::DataType::kBF16;
    }
    CudnnSdpaPlugin* clone() const noexcept override {
        auto* result = new (std::nothrow) CudnnSdpaPlugin(config_);
        if (result != nullptr)
            result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                            int32_t input_count,
                                            nvinfer1::IExprBuilder&) noexcept override {
        if (output_index != 0 || inputs == nullptr || input_count != 3)
            return {};
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && in_out != nullptr && position >= 0 &&
               position < 4 && in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kBF16;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return context_ != nullptr ? context_->workspace_size() : 0U;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        if (!dimensions_match(input_desc[0].dims, config_.batch, config_.q_sequence, config_.heads,
                              config_.head_dimension) ||
            !dimensions_match(input_desc[1].dims, config_.batch, config_.kv_sequence, config_.heads,
                              config_.head_dimension) ||
            !dimensions_match(input_desc[2].dims, config_.batch, config_.kv_sequence, config_.heads,
                              config_.head_dimension)) {
            std::fprintf(stderr,
                         "Wan22DitCudnnSdpa %s shape mismatch; expected Q=[%d,%d,%d,%d] "
                         "KV=[%d,%d,%d,%d] in physical BSHD order\n",
                         attention_kind_name(config_), config_.batch, config_.q_sequence,
                         config_.heads, config_.head_dimension, config_.batch, config_.kv_sequence,
                         config_.heads, config_.head_dimension);
            return 1;
        }
        const int status =
            context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace, stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22DitCudnnSdpa enqueue failed: %s\n",
                         context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        if (!config_valid_ || !is_qualified_config(config_))
            return 1;
        if (context_ != nullptr)
            return 0;
        try {
            auto context = std::unique_ptr<Context>(new (std::nothrow) Context(config_));
            if (context == nullptr)
                return 1;
            const int status = context->initialize();
            if (status != 0) {
                std::fprintf(stderr, "Wan22DitCudnnSdpa initialize failed: %s\n",
                             context->error().c_str());
                return status;
            }
            context_ = std::move(context);
            return 0;
        } catch (const std::exception& exception) {
            std::fprintf(stderr, "Wan22DitCudnnSdpa initialize exception: %s\n", exception.what());
            return 1;
        } catch (...) {
            std::fprintf(stderr, "Wan22DitCudnnSdpa initialize unknown exception\n");
            return 1;
        }
    }

    Config config_{};
    bool config_valid_{false};
    std::unique_ptr<Context> context_;
    std::string namespace_;
};

class CudnnSdpaCreator final : public nvinfer1::IPluginCreator {
  public:
    CudnnSdpaCreator() {
        field_entries_[0] = {kAttentionKindField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[1] = {kBatchField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[2] = {kHeadsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[3] = {kQSequenceField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[4] = {kKvSequenceField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[5] = {kHeadDimensionField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {6, field_entries_};
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        Config config{};
        if (!read_config_fields(fields, config)) {
            std::fprintf(stderr,
                         "Wan22DitCudnnSdpa creation requires all six exact INT32 shape fields\n");
            return nullptr;
        }
        auto* plugin = new (std::nothrow) CudnnSdpaPlugin(config);
        if (plugin != nullptr)
            plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        Config config{};
        if (!deserialize_config(data, length, config))
            return nullptr;
        auto* plugin = new (std::nothrow) CudnnSdpaPlugin(data, length);
        if (plugin != nullptr)
            plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value != nullptr ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField field_entries_[6]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::cudnn_sdpa

static nvinfer1::PluginRegistrar<trtmc::wan22::cudnn_sdpa::CudnnSdpaCreator>
    plugin_registrar_wan22_dit_cudnn_sdpa{};

extern "C" void trtmc_wan22_dit_cudnn_sdpa_force_link() {}

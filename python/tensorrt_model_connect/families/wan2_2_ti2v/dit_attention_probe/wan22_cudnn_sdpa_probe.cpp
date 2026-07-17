/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Isolated Wan2.2 block-0 cuDNN SDPA parity probe.  This is intentionally not
 * wired into the production DiT builder.  It reproduces the exact graph built
 * by PyTorch 2.12's aten/src/ATen/native/cudnn/MHA.cpp and exposes it as a
 * TensorRT plugin without any torch/ATen dependency.
 */

#include <NvInferRuntime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc::wan22::attention_probe {
namespace fe = cudnn_frontend;

namespace {

constexpr int64_t kQ = 0;
constexpr int64_t kK = 1;
constexpr int64_t kV = 2;
constexpr int64_t kO = 3;
constexpr int64_t kScale = 5;

struct ProbeConfig {
    int32_t batch{1};
    int32_t heads{24};
    int32_t q_sequence{27280};
    int32_t kv_sequence{27280};
    int32_t dimension{128};
    int32_t engine_id{10};
    int32_t kernel_config{36};
};

static_assert(sizeof(ProbeConfig) == 7 * sizeof(int32_t));

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

bool valid_config(const ProbeConfig& config) {
    return config.batch > 0 && config.heads > 0 && config.q_sequence > 0 &&
           config.kv_sequence > 0 && config.dimension > 0 && config.engine_id >= -1 &&
           config.kernel_config >= -1;
}

} // namespace

class SdpaContext {
  public:
    explicit SdpaContext(ProbeConfig config) : config_(config) {}
    SdpaContext(const SdpaContext&) = delete;
    SdpaContext& operator=(const SdpaContext&) = delete;
    ~SdpaContext() { reset(); }

    int initialize() {
        reset();
        if (!valid_config(config_)) {
            error_ = "invalid SDPA probe configuration";
            return 1;
        }
        if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS) {
            error_ = "cudnnCreate failed";
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
        const int64_t d = config_.dimension;
        const std::vector<int64_t> q_dimensions{b, h, q_sequence, d};
        const std::vector<int64_t> kv_dimensions{b, h, kv_sequence, d};
        // The official Wan tensors are physically contiguous BSHD.  PyTorch
        // transposes them to logical BHSD without materializing a copy.
        const std::vector<int64_t> q_strides{h * q_sequence * d, d, h * d, 1};
        const std::vector<int64_t> kv_strides{h * kv_sequence * d, d, h * d, 1};
        q_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kQ)
                                .set_name("Q")
                                .set_dim(q_dimensions)
                                .set_stride(q_strides));
        k_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kK)
                                .set_name("K")
                                .set_dim(kv_dimensions)
                                .set_stride(kv_strides));
        v_ = graph_->tensor(fe::graph::Tensor_attributes()
                                .set_uid(kV)
                                .set_name("V")
                                .set_dim(kv_dimensions)
                                .set_stride(kv_strides));
        scale_tensor_ = graph_->tensor(fe::graph::Tensor_attributes()
                                           .set_uid(kScale)
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
        output_->set_uid(kO).set_output(true).set_dim(q_dimensions).set_stride(q_strides);

        if (!check(graph_->validate(), "validate") ||
            !check(graph_->build_operation_graph(handle_), "build_operation_graph"))
            return 1;

        if (config_.engine_id >= 0) {
            std::unordered_map<fe::KnobType_t, int64_t> knobs;
            if (config_.kernel_config >= 0)
                knobs[fe::KnobType_t::KERNEL_CFG] = config_.kernel_config;
            if (!check(graph_->create_execution_plan(config_.engine_id, knobs),
                       "create_execution_plan"))
                return 1;
        } else if (!check(graph_->create_execution_plans({fe::HeurMode_t::A}),
                          "create_execution_plans")) {
            return 1;
        }
        candidate_count_ = graph_->get_execution_plan_count();
        if (!check(graph_->check_support(handle_), "check_support") ||
            !check(graph_->build_plans(handle_), "build_plans") ||
            !check(graph_->get_plan_name(plan_name_), "get_plan_name") ||
            !check(graph_->get_workspace_size(workspace_size_), "get_workspace_size"))
            return 1;

        scale_ = 1.0F / std::sqrt(static_cast<float>(config_.dimension));
        initialized_ = true;
        std::fprintf(stderr,
                     "Wan22CudnnSdpaProbe initialized plan=%s candidates=%lld workspace=%lld "
                     "scale=%.9g\n",
                     plan_name_.c_str(), static_cast<long long>(candidate_count_),
                     static_cast<long long>(workspace_size_), static_cast<double>(scale_));
        return 0;
    }

    int run(const void* q, const void* k, const void* v, void* output, void* workspace,
            size_t workspace_bytes, cudaStream_t stream) {
        if (!initialized_ || q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
            error_ = "run called with an uninitialized context or null tensor pointer";
            return 1;
        }
        if (workspace_bytes < static_cast<size_t>(workspace_size_)) {
            error_ = "TensorRT workspace is smaller than the cuDNN plan requirement";
            return 1;
        }
        const auto stream_status = cudnnSetStream(handle_, stream);
        if (stream_status != CUDNN_STATUS_SUCCESS) {
            error_ = std::string("cudnnSetStream failed: ") + cudnnGetErrorString(stream_status);
            return 1;
        }
        std::unordered_map<int64_t, void*> variant_pack = {
            {kQ, const_cast<void*>(q)},
            {kK, const_cast<void*>(k)},
            {kV, const_cast<void*>(v)},
            {kScale, &scale_},
            {kO, output},
        };
        const auto status = graph_->execute(handle_, variant_pack, workspace);
        return check(status, "execute") ? 0 : 1;
    }

    int64_t workspace_size() const { return workspace_size_; }
    int64_t candidate_count() const { return candidate_count_; }
    const std::string& plan_name() const { return plan_name_; }
    const std::string& error() const { return error_; }

  private:
    bool check(fe::error_t status, const char* operation) {
        if (status.is_good())
            return true;
        error_ = std::string(operation) + " failed: " + status.get_message();
        return false;
    }

    void reset() {
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
    }

    ProbeConfig config_{};
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

class CudnnSdpaProbePlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22CudnnSdpaProbe";
    static constexpr const char* kVERSION = "1";

    explicit CudnnSdpaProbePlugin(ProbeConfig config) : config_(config) {}
    CudnnSdpaProbePlugin(const void* data, size_t length) {
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
            std::fprintf(stderr, "Wan22CudnnSdpaProbe attachToContext failed\n");
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
    CudnnSdpaProbePlugin* clone() const noexcept override {
        auto* result = new CudnnSdpaProbePlugin(config_);
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
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
        return context_ == nullptr ? 0U : static_cast<size_t>(context_->workspace_size());
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (context_ == nullptr && initialize_context() != 0)
            return 1;
        if (context_ == nullptr || input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& dimensions = input_desc[0].dims;
        if (dimensions.nbDims != 4 || dimensions.d[0] != config_.batch ||
            dimensions.d[1] != config_.q_sequence || dimensions.d[2] != config_.heads ||
            dimensions.d[3] != config_.dimension) {
            std::fprintf(stderr,
                         "Wan22CudnnSdpaProbe expected physical BSHD [%d,%d,%d,%d], got "
                         "[%lld,%lld,%lld,%lld]\n",
                         config_.batch, config_.q_sequence, config_.heads, config_.dimension,
                         static_cast<long long>(dimensions.nbDims > 0 ? dimensions.d[0] : -1),
                         static_cast<long long>(dimensions.nbDims > 1 ? dimensions.d[1] : -1),
                         static_cast<long long>(dimensions.nbDims > 2 ? dimensions.d[2] : -1),
                         static_cast<long long>(dimensions.nbDims > 3 ? dimensions.d[3] : -1));
            return 1;
        }
        for (int32_t index = 1; index < 3; ++index) {
            const auto& kv_dimensions = input_desc[index].dims;
            if (kv_dimensions.nbDims != 4 || kv_dimensions.d[0] != config_.batch ||
                kv_dimensions.d[1] != config_.kv_sequence || kv_dimensions.d[2] != config_.heads ||
                kv_dimensions.d[3] != config_.dimension) {
                std::fprintf(stderr,
                             "Wan22CudnnSdpaProbe expected physical KV BSHD [%d,%d,%d,%d] "
                             "for input %d\n",
                             config_.batch, config_.kv_sequence, config_.heads, config_.dimension,
                             index);
                return 1;
            }
        }
        const auto status = context_->run(inputs[0], inputs[1], inputs[2], outputs[0], workspace,
                                          getWorkspaceSize(nullptr, 0, nullptr, 0), stream);
        if (status != 0)
            std::fprintf(stderr, "Wan22CudnnSdpaProbe enqueue failed: %s\n",
                         context_->error().c_str());
        return status;
    }

  private:
    int32_t initialize_context() noexcept {
        if (context_ != nullptr)
            return 0;
        auto context = std::make_unique<SdpaContext>(config_);
        const auto status = context->initialize();
        if (status != 0) {
            std::fprintf(stderr, "Wan22CudnnSdpaProbe initialize failed: %s\n",
                         context->error().c_str());
            return status;
        }
        context_ = std::move(context);
        return 0;
    }

    ProbeConfig config_{};
    std::unique_ptr<SdpaContext> context_;
    std::string namespace_;
};

class CudnnSdpaProbeCreator final : public nvinfer1::IPluginCreator {
  public:
    CudnnSdpaProbeCreator() {
        field_entries_[0] = {"batch", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[1] = {"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[2] = {"q_sequence", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[3] = {"kv_sequence", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[4] = {"dimension", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[5] = {"engine_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[6] = {"kernel_config", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {7, field_entries_};
    }
    char const* getPluginName() const noexcept override { return CudnnSdpaProbePlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return CudnnSdpaProbePlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        ProbeConfig config{};
        config.batch = read_int32(fields, "batch", config.batch);
        config.heads = read_int32(fields, "heads", config.heads);
        config.q_sequence = read_int32(fields, "q_sequence", config.q_sequence);
        config.kv_sequence = read_int32(fields, "kv_sequence", config.kv_sequence);
        config.dimension = read_int32(fields, "dimension", config.dimension);
        config.engine_id = read_int32(fields, "engine_id", config.engine_id);
        config.kernel_config = read_int32(fields, "kernel_config", config.kernel_config);
        return valid_config(config) ? new CudnnSdpaProbePlugin(config) : nullptr;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return length == sizeof(ProbeConfig) ? new CudnnSdpaProbePlugin(data, length) : nullptr;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField field_entries_[7]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22::attention_probe

static nvinfer1::PluginRegistrar<trtmc::wan22::attention_probe::CudnnSdpaProbeCreator>
    plugin_registrar_wan22_cudnn_sdpa_probe{};

extern "C" void trtmc_wan22_cudnn_sdpa_probe_force_link() {}

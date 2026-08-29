/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_encoder_plugin.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <istream>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <streambuf>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <utility>
#include <vector>

namespace trtmc::minimax_h3 {
namespace {

using Plugin = MiniMaxH3AudioEncoderPlugin;

constexpr std::size_t kMinimumModuleBytes = 300U << 20U;
constexpr std::size_t kMaximumModuleBytes = 400U << 20U;

bool is_fp32_linear(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kFLOAT &&
           desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool is_int8_linear(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kINT8 && desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool valid_sample_count(int32_t samples) noexcept {
    return samples >= Plugin::kMIN_SAMPLES && samples <= Plugin::kMAX_SAMPLES &&
           samples % Plugin::kHOP_LENGTH == 0;
}

bool input_network_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 3 && dims.d[0] == Plugin::kBATCH &&
           dims.d[1] == Plugin::kINPUT_CHANNELS &&
           (dims.d[2] == -1 || valid_sample_count(dims.d[2]));
}

bool input_runtime_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 3 && dims.d[0] == Plugin::kBATCH &&
           dims.d[1] == Plugin::kINPUT_CHANNELS && valid_sample_count(dims.d[2]);
}

bool output_network_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 3 && dims.d[0] == Plugin::kBATCH &&
           (dims.d[1] == -1 || (dims.d[1] >= Plugin::kMIN_SAMPLES / Plugin::kHOP_LENGTH &&
                                dims.d[1] <= Plugin::kMAX_SAMPLES / Plugin::kHOP_LENGTH)) &&
           dims.d[2] == Plugin::kOUTPUT_CHANNELS;
}

bool output_runtime_dims(nvinfer1::Dims const& dims, int32_t samples) noexcept {
    return dims.nbDims == 3 && dims.d[0] == Plugin::kBATCH &&
           dims.d[1] == samples / Plugin::kHOP_LENGTH && dims.d[2] == Plugin::kOUTPUT_CHANNELS;
}

bool module_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 1 && dims.d[0] >= static_cast<int32_t>(kMinimumModuleBytes) &&
           dims.d[0] <= static_cast<int32_t>(kMaximumModuleBytes);
}

bool dynamic_contract(nvinfer1::DynamicPluginTensorDesc const* inputs,
                      nvinfer1::DynamicPluginTensorDesc const* outputs) noexcept {
    if (!is_fp32_linear(inputs[0].desc) || !is_int8_linear(inputs[1].desc) ||
        !is_fp32_linear(outputs[0].desc) || !input_network_dims(inputs[0].desc.dims) ||
        !output_network_dims(outputs[0].desc.dims) || !input_runtime_dims(inputs[0].min) ||
        !input_runtime_dims(inputs[0].opt) || !input_runtime_dims(inputs[0].max) ||
        !module_dims(inputs[1].desc.dims) || !module_dims(inputs[1].min) ||
        !module_dims(inputs[1].opt) || !module_dims(inputs[1].max) ||
        inputs[1].min.d[0] != inputs[1].opt.d[0] || inputs[1].opt.d[0] != inputs[1].max.d[0]) {
        return false;
    }
    if (inputs[0].min.d[2] != Plugin::kMIN_SAMPLES || inputs[0].opt.d[2] != Plugin::kOPT_SAMPLES ||
        inputs[0].max.d[2] != Plugin::kMAX_SAMPLES) {
        return false;
    }
    return output_runtime_dims(outputs[0].min, inputs[0].min.d[2]) &&
           output_runtime_dims(outputs[0].opt, inputs[0].opt.d[2]) &&
           output_runtime_dims(outputs[0].max, inputs[0].max.d[2]);
}

bool runtime_contract(nvinfer1::PluginTensorDesc const* inputs,
                      nvinfer1::PluginTensorDesc const* outputs) noexcept {
    return is_fp32_linear(inputs[0]) && is_int8_linear(inputs[1]) && is_fp32_linear(outputs[0]) &&
           input_runtime_dims(inputs[0].dims) && module_dims(inputs[1].dims) &&
           output_runtime_dims(outputs[0].dims, inputs[0].dims.d[2]);
}

void report_error(char const* category, char const* detail) noexcept {
    std::fprintf(stderr, "[trtmc.minimax_h3.audio_encoder] %s: %s\n", category,
                 detail != nullptr ? detail : "unknown error");
}

class MemoryStreamBuffer final : public std::streambuf {
  public:
    explicit MemoryStreamBuffer(std::vector<std::uint8_t>& bytes) {
        begin_ = reinterpret_cast<char*>(bytes.data());
        end_ = begin_ + bytes.size();
        setg(begin_, begin_, end_);
    }

  protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                     std::ios_base::openmode mode) override {
        if ((mode & std::ios_base::in) == 0)
            return pos_type(off_type(-1));
        char* base = nullptr;
        if (direction == std::ios_base::beg)
            base = begin_;
        else if (direction == std::ios_base::cur)
            base = gptr();
        else if (direction == std::ios_base::end)
            base = end_;
        if (base == nullptr || offset < begin_ - base || offset > end_ - base)
            return pos_type(off_type(-1));
        char* position = base + offset;
        setg(begin_, position, end_);
        return pos_type(position - begin_);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
        if ((mode & std::ios_base::in) == 0)
            return pos_type(off_type(-1));
        auto const offset = static_cast<off_type>(position);
        if (offset < 0 || offset > end_ - begin_)
            return pos_type(off_type(-1));
        setg(begin_, begin_ + offset, end_);
        return position;
    }

  private:
    char* begin_{nullptr};
    char* end_{nullptr};
};

} // namespace

struct MiniMaxH3AudioEncoderPlugin::RuntimeState {
    std::mutex mutex;
    std::vector<std::uint8_t> host_module_bytes;
    std::optional<torch::jit::Module> module;
    int32_t device_index{-1};

    torch::jit::Module& getOrLoad(void const* device_bytes, std::size_t byte_count,
                                  cudaStream_t stream, int32_t requested_device) {
        std::lock_guard<std::mutex> lock(mutex);
        if (module.has_value()) {
            if (device_index != requested_device)
                throw std::runtime_error("audio encoder plugin instance changed CUDA device");
            return *module;
        }
        host_module_bytes.resize(byte_count);
        if (cudaMemcpyAsync(host_module_bytes.data(), device_bytes, byte_count,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::vector<std::uint8_t>().swap(host_module_bytes);
            throw std::runtime_error("unable to copy embedded TorchScript module from CUDA");
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::vector<std::uint8_t>().swap(host_module_bytes);
            throw std::runtime_error("unable to synchronize embedded TorchScript module copy");
        }
        if (host_module_bytes.size() < 4 || host_module_bytes[0] != 'P' ||
            host_module_bytes[1] != 'K' || host_module_bytes[2] != 3U ||
            host_module_bytes[3] != 4U) {
            std::vector<std::uint8_t>().swap(host_module_bytes);
            throw std::runtime_error("embedded audio encoder module is not TorchScript");
        }
        std::optional<torch::jit::Module> loaded_module;
        try {
            MemoryStreamBuffer buffer(host_module_bytes);
            std::istream input_stream(&buffer);
            loaded_module.emplace(
                torch::jit::load(input_stream, c10::Device(c10::kCUDA, requested_device)));
            loaded_module->eval();
        } catch (...) {
            std::vector<std::uint8_t>().swap(host_module_bytes);
            throw;
        }
        // torch::jit::load owns the decoded tensors. Do not retain another
        // 300-400 MiB host copy for the lifetime of every execution context.
        std::vector<std::uint8_t>().swap(host_module_bytes);
        module.emplace(std::move(*loaded_module));
        device_index = requested_device;
        return *module;
    }
};

MiniMaxH3AudioEncoderPlugin::MiniMaxH3AudioEncoderPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : runtime_(new (std::nothrow) RuntimeState()) {
    valid_ = runtime_ != nullptr && fields.nbFields == 0 && fields.fields == nullptr;
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

MiniMaxH3AudioEncoderPlugin::MiniMaxH3AudioEncoderPlugin(
    MiniMaxH3AudioEncoderPlugin const& other) noexcept
    : runtime_(new (std::nothrow) RuntimeState()), valid_(other.valid_ && runtime_ != nullptr) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

MiniMaxH3AudioEncoderPlugin::~MiniMaxH3AudioEncoderPlugin() = default;

nvinfer1::IPluginCapability*
MiniMaxH3AudioEncoderPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

MiniMaxH3AudioEncoderPlugin* MiniMaxH3AudioEncoderPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) MiniMaxH3AudioEncoderPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const* MiniMaxH3AudioEncoderPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const* MiniMaxH3AudioEncoderPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MiniMaxH3AudioEncoderPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t MiniMaxH3AudioEncoderPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 2 &&
                   output_count == 1 && dynamic_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3AudioEncoderPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
                                                        int32_t output_count,
                                                        nvinfer1::DataType const* input_types,
                                                        int32_t input_count) const noexcept {
    if (output_types == nullptr || input_types == nullptr || output_count != 1 ||
        input_count != 2 || input_types[0] != nvinfer1::DataType::kFLOAT ||
        input_types[1] != nvinfer1::DataType::kINT8) {
        return 1;
    }
    output_types[0] = nvinfer1::DataType::kFLOAT;
    return 0;
}

int32_t MiniMaxH3AudioEncoderPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 2 || output_count != 1 ||
        shape_input_count != 0 || inputs[0].nbDims != 3 || inputs[0].d[0] == nullptr ||
        inputs[0].d[1] == nullptr || inputs[0].d[2] == nullptr || inputs[1].nbDims != 1 ||
        inputs[1].d[0] == nullptr) {
        return 1;
    }
    (void)shape_inputs;
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    auto* hop = expression_builder.constant(kHOP_LENGTH);
    outputs[0].d[1] = expression_builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                                   *inputs[0].d[2], *hop);
    outputs[0].d[2] = expression_builder.constant(kOUTPUT_CHANNELS);
    return outputs[0].d[1] != nullptr && outputs[0].d[2] != nullptr ? 0 : 1;
}

bool MiniMaxH3AudioEncoderPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_output == nullptr || input_count != 2 || output_count != 1 || position < 0 ||
        position >= 3) {
        return false;
    }
    if (position == 0)
        return is_fp32_linear(input_output[0].desc) &&
               input_network_dims(input_output[0].desc.dims);
    if (position == 1)
        return is_int8_linear(input_output[1].desc) && module_dims(input_output[1].desc.dims);
    return is_fp32_linear(input_output[2].desc) && output_network_dims(input_output[2].desc.dims);
}

int32_t MiniMaxH3AudioEncoderPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t MiniMaxH3AudioEncoderPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*,
                                                          int32_t,
                                                          nvinfer1::DynamicPluginTensorDesc const*,
                                                          int32_t) const noexcept {
    return 0;
}

char const* MiniMaxH3AudioEncoderPlugin::getTimingCacheID() noexcept {
    return "torchscript-fp32-audio-encoder-hop800-v1";
}

char const* MiniMaxH3AudioEncoderPlugin::getMetadataString() noexcept {
    return "input=[2,1,samples]:fp32:linear;samples=64000..480000:step800;"
           "module_input=[module_bytes]:int8:linear;module_bytes=300..400MiB:fixed;"
           "output=[2,samples/800,32]:fp32:linear;module=torchscript-plan-constant-v1;"
           "weight_norm=cuda-frozen;cudnn_tf32=trace-true;matmul_tf32=false;"
           "graph_optimizer=false;python_runtime=false";
}

int32_t MiniMaxH3AudioEncoderPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                   int32_t input_count,
                                                   nvinfer1::PluginTensorDesc const* outputs,
                                                   int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 2 &&
                   output_count == 1 && runtime_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3AudioEncoderPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                             nvinfer1::PluginTensorDesc const* output_desc,
                                             void const* const* inputs, void* const* outputs, void*,
                                             cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        outputs[0] == nullptr || !runtime_contract(input_desc, output_desc) ||
        runtime_ == nullptr) {
        return 1;
    }
    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        c10::InferenceMode inference_mode;
        at::NoTF32Guard no_tf32;
        auto const torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
        int64_t const samples = input_desc[0].dims.d[2];
        int64_t const frames = output_desc[0].dims.d[1];
        auto const options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        auto input = at::from_blob(const_cast<void*>(inputs[0]), {kBATCH, 1, samples}, options);
        auto& module = runtime_->getOrLoad(
            inputs[1], static_cast<std::size_t>(input_desc[1].dims.d[0]), stream, device_index);
        // The profiling executor changes Conv1d lowering after the first call,
        // which is not bit-exact to the released eager path. In the pinned
        // libtorch ABI, GraphOptimizerEnabledGuard is backed by thread-local
        // state, so concurrent execution contexts do not affect one another.
        torch::jit::GraphOptimizerEnabledGuard optimizer_guard(false);
        auto result = module.forward({input}).toTensor();
        if (result.scalar_type() != at::kFloat || !result.is_cuda() || result.dim() != 3 ||
            result.size(0) != kBATCH || result.size(1) != frames ||
            result.size(2) != kOUTPUT_CHANNELS) {
            report_error("output contract", "TorchScript module returned an incompatible tensor");
            return 1;
        }
        auto output = at::from_blob(outputs[0], {kBATCH, frames, kOUTPUT_CHANNELS}, options);
        output.copy_(result);
        return 0;
    } catch (c10::Error const& error) {
        report_error("TorchScript execution", error.what());
    } catch (std::exception const& error) {
        report_error("native execution", error.what());
    } catch (...) {
        report_error("native execution", "unknown exception");
    }
    return 1;
}

nvinfer1::IPluginV3*
MiniMaxH3AudioEncoderPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const*
MiniMaxH3AudioEncoderPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc::minimax_h3

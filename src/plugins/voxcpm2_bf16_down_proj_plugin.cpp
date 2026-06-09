// VoxCPM2 BF16 down-projection TensorRT plugin.

#if TRTMC_HAS_TRT

#include "plugins/voxcpm2_bf16_down_proj_plugin.h"

#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <vector>

namespace trtmc {

cudaError_t launch_voxcpm2_bf16_down_proj_kernel(const __nv_bfloat16* input,
                                                 const __nv_bfloat16* weight,
                                                 const __nv_bfloat16* bias,
                                                 __nv_bfloat16* output, int64_t rows,
                                                 int64_t in_features,
                                                 int64_t out_features,
                                                 cudaStream_t stream);

VoxCPM2Bf16DownProjPlugin::VoxCPM2Bf16DownProjPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(num_inputs_)) {
        std::memcpy(&num_inputs_, data, sizeof(num_inputs_));
    }
}

char const* VoxCPM2Bf16DownProjPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* VoxCPM2Bf16DownProjPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t VoxCPM2Bf16DownProjPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t VoxCPM2Bf16DownProjPlugin::initialize() noexcept {
    return 0;
}

void VoxCPM2Bf16DownProjPlugin::terminate() noexcept {}

void VoxCPM2Bf16DownProjPlugin::destroy() noexcept {
    delete this;
}

size_t VoxCPM2Bf16DownProjPlugin::getSerializationSize() const noexcept {
    return sizeof(num_inputs_);
}

void VoxCPM2Bf16DownProjPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &num_inputs_, sizeof(num_inputs_));
}

void VoxCPM2Bf16DownProjPlugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    namespace_ = pluginNamespace ? pluginNamespace : "";
}

char const* VoxCPM2Bf16DownProjPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType
VoxCPM2Bf16DownProjPlugin::getOutputDataType(int32_t, nvinfer1::DataType const* inputTypes,
                                             int32_t) const noexcept {
    return inputTypes[0];
}

VoxCPM2Bf16DownProjPlugin* VoxCPM2Bf16DownProjPlugin::clone() const noexcept {
    auto* plugin = new VoxCPM2Bf16DownProjPlugin();
    plugin->namespace_ = namespace_;
    plugin->num_inputs_ = num_inputs_;
    return plugin;
}

nvinfer1::DimsExprs
VoxCPM2Bf16DownProjPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                               int32_t, nvinfer1::IExprBuilder&) noexcept {
    nvinfer1::DimsExprs output = inputs[0];
    if (output.nbDims > 0) {
        output.d[output.nbDims - 1] = inputs[1].d[0];
    }
    return output;
}

bool VoxCPM2Bf16DownProjPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void VoxCPM2Bf16DownProjPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                int32_t nbInputs,
                                                nvinfer1::DynamicPluginTensorDesc const*,
                                                int32_t) noexcept {
    num_inputs_ = nbInputs;
}

size_t VoxCPM2Bf16DownProjPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                   nvinfer1::PluginTensorDesc const*,
                                                   int32_t) const noexcept {
    return 0;
}

int32_t VoxCPM2Bf16DownProjPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                           nvinfer1::PluginTensorDesc const*,
                                           void const* const* inputs, void* const* outputs,
                                           void*, cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        outputs[0] == nullptr) {
        return -1;
    }

    const auto& input_dims = inputDesc[0].dims;
    const auto& weight_dims = inputDesc[1].dims;
    if (input_dims.nbDims < 2 || weight_dims.nbDims < 2) {
        return -1;
    }

    int64_t rows = 1;
    for (int32_t dim = 0; dim < input_dims.nbDims - 1; ++dim) {
        if (input_dims.d[dim] < 0) {
            return -1;
        }
        rows *= static_cast<int64_t>(input_dims.d[dim]);
    }
    const auto in_features = static_cast<int64_t>(input_dims.d[input_dims.nbDims - 1]);
    const auto out_features = static_cast<int64_t>(inputDesc[1].dims.d[0]);
    if (in_features <= 0 || out_features <= 0) {
        return -1;
    }
    const auto* input = static_cast<const __nv_bfloat16*>(inputs[0]);
    const auto* weight = static_cast<const __nv_bfloat16*>(inputs[1]);
    const auto* bias =
        (num_inputs_ > 2 && inputs[2] != nullptr) ? static_cast<const __nv_bfloat16*>(inputs[2])
                                                  : nullptr;
    auto* output = static_cast<__nv_bfloat16*>(outputs[0]);

    const auto status = launch_voxcpm2_bf16_down_proj_kernel(
        input, weight, bias, output, rows, in_features, out_features, stream);
    if (status != cudaSuccess) {
        std::cerr << "[VoxCPM2Bf16DownProjPlugin] CUDA launch failed: "
                  << cudaGetErrorString(status) << '\n';
        return -1;
    }
    return 0;
}

class VoxCPM2Bf16DownProjCreator : public nvinfer1::IPluginCreator {
  public:
    VoxCPM2Bf16DownProjCreator() {
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return VoxCPM2Bf16DownProjPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return VoxCPM2Bf16DownProjPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* pluginNamespace) noexcept override {
        namespace_ = pluginNamespace ? pluginNamespace : "";
    }

    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new VoxCPM2Bf16DownProjPlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new VoxCPM2Bf16DownProjPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string namespace_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::VoxCPM2Bf16DownProjCreator>
    pluginRegistrarVoxCPM2Bf16DownProj{};

extern "C" void voxcpm2_bf16_down_proj_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

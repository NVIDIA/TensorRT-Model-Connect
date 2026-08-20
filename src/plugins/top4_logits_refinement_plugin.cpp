/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TRT

#include "plugins/top4_logits_refinement_plugin.h"

#include <NvInferRuntime.h>
#include <cstring>
#include <memory>
#include <vector>

namespace trtmc {

char const* Top4LogitsRefinementPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* Top4LogitsRefinementPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t Top4LogitsRefinementPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t Top4LogitsRefinementPlugin::initialize() noexcept {
    return 0;
}

void Top4LogitsRefinementPlugin::terminate() noexcept {}

void Top4LogitsRefinementPlugin::destroy() noexcept {
    delete this;
}

size_t Top4LogitsRefinementPlugin::getSerializationSize() const noexcept {
    return 0;
}

void Top4LogitsRefinementPlugin::serialize(void*) const noexcept {}

void Top4LogitsRefinementPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

char const* Top4LogitsRefinementPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType Top4LogitsRefinementPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

Top4LogitsRefinementPlugin* Top4LogitsRefinementPlugin::clone() const noexcept {
    try {
        auto plugin = std::make_unique<Top4LogitsRefinementPlugin>();
        plugin->namespace_ = namespace_;
        return plugin.release();
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::DimsExprs
Top4LogitsRefinementPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool Top4LogitsRefinementPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_count != 4 || output_count != 1 || position < 0 || position >= 5)
        return false;
    if (input_output[position].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (position == 1)
        return input_output[position].type == nvinfer1::DataType::kHALF;
    return input_output[position].type == nvinfer1::DataType::kFLOAT;
}

void Top4LogitsRefinementPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}

size_t Top4LogitsRefinementPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return top4_logits_refinement_workspace_size();
}

int32_t Top4LogitsRefinementPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs,
                                            void* workspace, cudaStream_t stream) noexcept {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr || workspace == nullptr)
        return -1;
    if (input_desc[0].dims.nbDims != 2 || input_desc[1].dims.nbDims != 2)
        return -1;
    const int32_t vocab_size = input_desc[0].dims.d[1];
    const int32_t hidden_size = input_desc[1].dims.d[1];
    if (vocab_size <= 0 || hidden_size <= 0)
        return -1;
    return launch_top4_logits_refinement(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0],
                                         workspace, hidden_size, vocab_size, stream);
}

class Top4LogitsRefinementCreator final : public nvinfer1::IPluginCreator {
  public:
    Top4LogitsRefinementCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }

    char const* getPluginName() const noexcept override {
        return Top4LogitsRefinementPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return Top4LogitsRefinementPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }

    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        try {
            auto plugin = std::make_unique<Top4LogitsRefinementPlugin>();
            plugin->setPluginNamespace(namespace_.c_str());
            return plugin.release();
        } catch (...) {
            return nullptr;
        }
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const*,
                                           size_t length) noexcept override {
        if (length != 0)
            return nullptr;
        return createPlugin(nullptr, nullptr);
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::Top4LogitsRefinementCreator>
    plugin_registrar_top4_logits_refinement{};

#endif // TRTMC_HAS_TRT

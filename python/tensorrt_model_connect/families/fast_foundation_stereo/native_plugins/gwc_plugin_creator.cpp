/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gwc_plugin.h"

#include <NvInferRuntime.h>
#include <cstddef>
#include <string>

namespace trtmc {

class FastFoundationStereoCombinedVolumeCreator final : public nvinfer1::IPluginCreator {
  public:
    char const* getPluginName() const noexcept override {
        return FastFoundationStereoCombinedVolumePlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return FastFoundationStereoCombinedVolumePlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
    }

    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        auto* plugin = new FastFoundationStereoCombinedVolumePlugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           std::size_t length) noexcept override {
        auto* plugin = new FastFoundationStereoCombinedVolumePlugin(data, length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

  private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
    std::string namespace_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoCombinedVolumeCreator>
    pluginRegistrarFastFoundationStereoCombinedVolume{};

extern "C" void fast_foundation_stereo_combined_volume_plugin_force_link() {}

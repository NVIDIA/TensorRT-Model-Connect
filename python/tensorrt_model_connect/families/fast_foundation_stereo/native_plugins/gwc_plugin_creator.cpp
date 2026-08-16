/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gwc_plugin.h"

#include <NvInferRuntime.h>
#include <cstddef>
#include <string>

namespace trtmc {

class FastFoundationStereoGwcCreator final : public nvinfer1::IPluginCreator {
  public:
    char const* getPluginName() const noexcept override {
        return FastFoundationStereoGwcPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return FastFoundationStereoGwcPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
    }

    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        auto* plugin = new FastFoundationStereoGwcPlugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           std::size_t length) noexcept override {
        auto* plugin = new FastFoundationStereoGwcPlugin(data, length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

  private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
    std::string namespace_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoGwcCreator>
    pluginRegistrarFastFoundationStereoGwc{};

extern "C" void fast_foundation_stereo_gwc_plugin_force_link() {}

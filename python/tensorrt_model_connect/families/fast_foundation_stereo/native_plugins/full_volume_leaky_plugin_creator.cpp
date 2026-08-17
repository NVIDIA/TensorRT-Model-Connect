/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "full_volume_leaky_plugin.h"

#include <NvInferRuntime.h>
#include <array>
#include <new>

namespace trtmc {

class FastFoundationStereoFullVolumeLeakyCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    FastFoundationStereoFullVolumeLeakyCreator() noexcept {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) FastFoundationStereoFullVolumeLeakyPlugin(*fields);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return FastFoundationStereoFullVolumeLeakyPlugin::kPLUGIN_NAME;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return FastFoundationStereoFullVolumeLeakyPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.data();
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::array<char, 1> namespace_{{'\0'}};
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoFullVolumeLeakyCreator>
    pluginRegistrarFastFoundationStereoFullVolumeLeaky{};

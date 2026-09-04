/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins.h"

#include <NvInferRuntime.h>
#include <array>
#include <cstddef>
#include <new>
#include <string>

namespace trtmc {

template <typename Plugin>
class PluginV2Creator final : public nvinfer1::IPluginCreator {
  public:
    char const* getPluginName() const noexcept override { return Plugin::kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return Plugin::kPLUGIN_VERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        auto* plugin = new Plugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           std::size_t length) noexcept override {
        auto* plugin = new Plugin(data, length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

  private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
    std::string namespace_;
};

template <typename Plugin>
class PluginV3Creator final : public nvinfer1::IPluginCreatorV3One {
  public:
    PluginV3Creator() noexcept {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) Plugin(*fields);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Plugin::kPLUGIN_NAME;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return Plugin::kPLUGIN_VERSION;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.data();
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::array<char, 1> namespace_{{'\0'}};
};

using FastFoundationStereoCombinedVolumeCreator =
    PluginV2Creator<FastFoundationStereoCombinedVolumePlugin>;
using FastFoundationStereoGeometryVolumeConvc1Creator =
    PluginV2Creator<FastFoundationStereoGeometryVolumeConvc1Plugin>;
using FastFoundationStereoSpatialAttentionReduceCreator =
    PluginV2Creator<FastFoundationStereoSpatialAttentionReducePlugin>;
using FastFoundationStereoFullVolumeLeakyCreator =
    PluginV3Creator<FastFoundationStereoFullVolumeLeakyPlugin>;
using FastFoundationStereoPost8SumCreator = PluginV3Creator<FastFoundationStereoPost8SumPlugin>;

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoCombinedVolumeCreator>
    pluginRegistrarFastFoundationStereoCombinedVolume{};
static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoGeometryVolumeConvc1Creator>
    pluginRegistrarFastFoundationStereoGeometryVolumeConvc1{};
static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoSpatialAttentionReduceCreator>
    pluginRegistrarFastFoundationStereoSpatialAttentionReduce{};
static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoFullVolumeLeakyCreator>
    pluginRegistrarFastFoundationStereoFullVolumeLeaky{};
static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoPost8SumCreator>
    pluginRegistrarFastFoundationStereoPost8Sum{};

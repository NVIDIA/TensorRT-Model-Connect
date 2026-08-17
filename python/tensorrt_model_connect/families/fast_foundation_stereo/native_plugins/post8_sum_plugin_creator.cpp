/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "post8_sum_plugin.h"

#include <NvInferRuntime.h>
#include <array>
#include <new>

namespace trtmc {
namespace {

const std::array<nvinfer1::PluginField, 1> kFields{{
    {"tile_positions", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
}};

} // namespace

class FastFoundationStereoPost8SumCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    FastFoundationStereoPost8SumCreator() noexcept {
        fields_.nbFields = static_cast<int32_t>(kFields.size());
        fields_.fields = kFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) FastFoundationStereoPost8SumPlugin(*fields);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return FastFoundationStereoPost8SumPlugin::kPLUGIN_NAME;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return FastFoundationStereoPost8SumPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.data();
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::array<char, 1> namespace_{{'\0'}};
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::FastFoundationStereoPost8SumCreator>
    pluginRegistrarFastFoundationStereoPost8Sum{};

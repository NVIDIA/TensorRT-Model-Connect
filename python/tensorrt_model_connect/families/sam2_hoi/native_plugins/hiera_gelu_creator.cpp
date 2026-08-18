/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hiera_gelu_plugin.h"

#include <NvInferPlugin.h>
#include <string>

namespace trtmc::sam2_hoi {

class HieraGeluErfBf16Creator final : public nvinfer1::IPluginCreator {
  public:
    HieraGeluErfBf16Creator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override {
        return HieraGeluErfBf16Plugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return HieraGeluErfBf16Plugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value != nullptr ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::IPluginV2*
    createPlugin(char const* name,
                 nvinfer1::PluginFieldCollection const* fields) noexcept override {
        (void)name;
        if (fields == nullptr || fields->nbFields != 0) {
            return nullptr;
        }
        auto* plugin = new HieraGeluErfBf16Plugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const* name, void const* data,
                                           std::size_t length) noexcept override {
        (void)name;
        if (length != 0) {
            return nullptr;
        }
        auto* plugin = new HieraGeluErfBf16Plugin(data, length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::sam2_hoi

static nvinfer1::PluginRegistrar<trtmc::sam2_hoi::HieraGeluErfBf16Creator>
    pluginRegistrarSam2HoiHieraGeluErfBf16{};

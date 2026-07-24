/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/native_kv_append_plugin.h"

#include <NvInferRuntime.h>
#include <array>
#include <cstring>
#include <new>

namespace trtmc::runtime_kv {

class NativeKvAppendCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    NativeKvAppendCreator() {
        fields_[0] = {
            "abi_version",
            nullptr,
            nvinfer1::PluginFieldType::kINT32,
            1,
        };
        field_collection_.nbFields = static_cast<int32_t>(fields_.size());
        field_collection_.fields = fields_.data();
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return kNativeKvAppendPluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return kNativeKvAppendPluginVersion;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        int32_t abi_version = kNativeKvAppendPluginAbi;
        if (fields != nullptr) {
            for (int32_t index = 0; index < fields->nbFields; ++index) {
                auto const& field = fields->fields[index];
                if (field.name != nullptr && std::strcmp(field.name, "abi_version") == 0) {
                    if (field.data == nullptr || field.type != nvinfer1::PluginFieldType::kINT32 ||
                        field.length != 1) {
                        return nullptr;
                    }
                    abi_version = *static_cast<int32_t const*>(field.data);
                }
            }
        }
        if (abi_version != kNativeKvAppendPluginAbi)
            return nullptr;
        return new (std::nothrow) NativeKvAppendPlugin(abi_version);
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    std::array<nvinfer1::PluginField, 1> fields_{};
    nvinfer1::PluginFieldCollection field_collection_{};
};

} // namespace trtmc::runtime_kv

static nvinfer1::PluginRegistrar<trtmc::runtime_kv::NativeKvAppendCreator>
    plugin_registrar_native_kv_append{};

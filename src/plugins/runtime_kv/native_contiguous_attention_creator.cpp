/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/cudnn_attention.h"
#include "plugins/runtime_kv/native_contiguous_attention_plugin.h"

#include <NvInferRuntime.h>
#include <array>
#include <cstring>
#include <new>

namespace trtmc::runtime_kv {
namespace {

bool read_int32_field(nvinfer1::PluginFieldCollection const* fields, char const* name,
                      int32_t& value) noexcept {
    if (fields == nullptr) {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        auto const& field = fields->fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0) {
            if (field.data == nullptr || field.type != nvinfer1::PluginFieldType::kINT32 ||
                field.length != 1) {
                return false;
            }
            value = *static_cast<int32_t const*>(field.data);
            return true;
        }
    }
    return false;
}

} // namespace

class NativeContiguousAttentionCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    NativeContiguousAttentionCreator() {
        char const* names[] = {"abi_version", "num_query_heads", "num_kv_heads", "head_dim",
                               "chunk_limit"};
        for (size_t index = 0; index < fields_.size(); ++index) {
            fields_[index] = {names[index], nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        }
        field_collection_.nbFields = static_cast<int32_t>(fields_.size());
        field_collection_.fields = fields_.data();
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return kNativeContiguousAttentionPluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return kNativeContiguousAttentionPluginVersion;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        int32_t abi_version = 0;
        int32_t num_query_heads = 0;
        int32_t num_kv_heads = 0;
        int32_t head_dim = 0;
        int32_t chunk_limit = 0;
        if (!native_cudnn_attention_available() ||
            !read_int32_field(fields, "abi_version", abi_version) ||
            !read_int32_field(fields, "num_query_heads", num_query_heads) ||
            !read_int32_field(fields, "num_kv_heads", num_kv_heads) ||
            !read_int32_field(fields, "head_dim", head_dim) ||
            !read_int32_field(fields, "chunk_limit", chunk_limit) ||
            abi_version != kNativeContiguousAttentionPluginAbi || num_query_heads <= 0 ||
            num_kv_heads <= 0 || num_query_heads % num_kv_heads != 0 || head_dim <= 0 ||
            head_dim > 256 || chunk_limit <= 0) {
            return nullptr;
        }
        return new (std::nothrow) NativeContiguousAttentionPlugin(
            abi_version, num_query_heads, num_kv_heads, head_dim, chunk_limit);
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    std::array<nvinfer1::PluginField, 5> fields_{};
    nvinfer1::PluginFieldCollection field_collection_{};
};

} // namespace trtmc::runtime_kv

static nvinfer1::PluginRegistrar<trtmc::runtime_kv::NativeContiguousAttentionCreator>
    plugin_registrar_native_contiguous_attention{};

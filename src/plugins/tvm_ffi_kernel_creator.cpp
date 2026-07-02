/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TVM-FFI kernel plugin creator (IPluginCreator) + REGISTER_TENSORRT_PLUGIN.
// Uses the stable V2 plugin path for maximum platform compatibility.

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_kernel_plugin.h"

#include <NvInferRuntime.h>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {

class TvmFfiKernelCreator : public nvinfer1::IPluginCreator {
  public:
    TvmFfiKernelCreator() {
        fields_.push_back({"kernel_name", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
        fields_.push_back({"shape_spec", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return TvmFfiKernelPlugin::kPLUGIN_NAME; }

    char const* getPluginVersion() const noexcept override {
        return TvmFfiKernelPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        std::string kernel_name;
        std::string shape_spec;

        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "kernel_name") == 0 &&
                    f.type == nvinfer1::PluginFieldType::kCHAR) {
                    kernel_name = std::string(static_cast<const char*>(f.data),
                                              static_cast<std::size_t>(f.length));
                } else if (std::strcmp(f.name, "shape_spec") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kCHAR) {
                    shape_spec = std::string(static_cast<const char*>(f.data),
                                             static_cast<std::size_t>(f.length));
                }
            }
        }

        return new TvmFfiKernelPlugin(kernel_name, shape_spec);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new TvmFfiKernelPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

// V2 static registration — proven stable on all platforms.
// Can't use REGISTER_TENSORRT_PLUGIN(trtmc::TvmFfiKernelCreator) because the
// macro concatenates the name into an identifier and :: is invalid in identifiers.
static nvinfer1::PluginRegistrar<trtmc::TvmFfiKernelCreator> pluginRegistrarTvmFfiKernel{};

// Force-link symbol for static library usage.
extern "C" void tvm_ffi_plugin_force_link() {}

#endif // TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

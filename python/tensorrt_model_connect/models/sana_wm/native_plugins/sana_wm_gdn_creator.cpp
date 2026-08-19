/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TRT

#include "sana_wm_gdn_plugin.h"

#include <NvInferRuntime.h>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {

namespace {

bool plugin_field_has_type(const nvinfer1::PluginField& field, const char* name,
                           nvinfer1::PluginFieldType type) {
    return std::strcmp(field.name, name) == 0 && field.type == type && field.data != nullptr;
}

bool read_int_plugin_field(const nvinfer1::PluginField& field, const char* name, int32_t& out) {
    if (!plugin_field_has_type(field, name, nvinfer1::PluginFieldType::kINT32))
        return false;
    out = *static_cast<const int32_t*>(field.data);
    return true;
}

bool read_float_plugin_field(const nvinfer1::PluginField& field, const char* name, float& out) {
    if (!plugin_field_has_type(field, name, nvinfer1::PluginFieldType::kFLOAT32))
        return false;
    out = *static_cast<const float*>(field.data);
    return true;
}

#if TRTMC_HAS_LIBTORCH_CONV
bool read_float_array_plugin_field(const nvinfer1::PluginField& field, const char* name,
                                   const float*& data, int32_t& count) {
    if (!plugin_field_has_type(field, name, nvinfer1::PluginFieldType::kFLOAT32))
        return false;
    data = static_cast<const float*>(field.data);
    count = field.length;
    return true;
}
#endif

SanaWmGdnPlugin::Mode gdn_mode_from_int(int32_t mode) {
    switch (mode) {
    case 1:
        return SanaWmGdnPlugin::Mode::kCamera;
    case 2:
        return SanaWmGdnPlugin::Mode::kMainCombined;
    case 3:
        return SanaWmGdnPlugin::Mode::kMainRawCombined;
    case 4:
        return SanaWmGdnPlugin::Mode::kCameraCombined;
    default:
        return SanaWmGdnPlugin::Mode::kMain;
    }
}

} // namespace

class SanaWmGdnCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGdnCreator() {
        fields_.push_back({"mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"reverse_output", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return SanaWmGdnPlugin::kPLUGIN_NAME; }

    char const* getPluginVersion() const noexcept override {
        return SanaWmGdnPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t mode = 0;
        int32_t reverse = 0;
        int32_t frames = 0;
        int32_t head_dim = 0;
        float eps = 1.0e-6F;
        float norm_eps = 1.0e-5F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "mode", mode))
                    continue;
                if (read_int_plugin_field(f, "reverse_output", reverse))
                    continue;
                if (read_float_plugin_field(f, "eps", eps))
                    continue;
                if (read_int_plugin_field(f, "frames", frames))
                    continue;
                if (read_int_plugin_field(f, "head_dim", head_dim))
                    continue;
                read_float_plugin_field(f, "norm_eps", norm_eps);
            }
        }
        auto plugin_mode = gdn_mode_from_int(mode);
        return new SanaWmGdnPlugin(plugin_mode, reverse != 0, eps, frames, head_dim, norm_eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGdnPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmUcpeCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmUcpeCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"inverse", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"tree_reduce", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"downscale", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"double_rope", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"rope_only", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return SanaWmUcpePlugin::kPLUGIN_NAME; }

    char const* getPluginVersion() const noexcept override {
        return SanaWmUcpePlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        int32_t head_dim = 0;
        int32_t inverse = 0;
        int32_t tree_reduce = 1;
        int32_t downscale = 0;
        int32_t double_rope = 0;
        int32_t rope_only = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "frames", frames))
                    continue;
                if (read_int_plugin_field(f, "spatial", spatial))
                    continue;
                if (read_int_plugin_field(f, "heads", heads))
                    continue;
                if (read_int_plugin_field(f, "head_dim", head_dim))
                    continue;
                if (read_int_plugin_field(f, "inverse", inverse))
                    continue;
                if (read_int_plugin_field(f, "tree_reduce", tree_reduce))
                    continue;
                if (read_int_plugin_field(f, "downscale", downscale))
                    continue;
                if (read_int_plugin_field(f, "double_rope", double_rope))
                    continue;
                read_int_plugin_field(f, "rope_only", rope_only);
            }
        }
        return new SanaWmUcpePlugin(frames, spatial, heads, head_dim, inverse != 0,
                                    tree_reduce != 0, downscale != 0, double_rope != 0,
                                    rope_only != 0);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmUcpePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmCamPrepCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmCamPrepCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmCamPrepPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmCamPrepPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        int32_t head_dim = 0;
        float norm_eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "frames", frames))
                    continue;
                if (read_int_plugin_field(f, "spatial", spatial))
                    continue;
                if (read_int_plugin_field(f, "heads", heads))
                    continue;
                if (read_int_plugin_field(f, "head_dim", head_dim))
                    continue;
                read_float_plugin_field(f, "norm_eps", norm_eps);
            }
        }
        return new SanaWmCamPrepPlugin(frames, spatial, heads, head_dim, norm_eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmCamPrepPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

#if TRTMC_HAS_LIBTORCH_CONV
class SanaWmTorchConv2dCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmTorchConv2dCreator() {
        fields_.push_back({"out_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"in_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"pad_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"pad_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmTorchConv2dPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmTorchConv2dPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t out_channels = 0;
        int32_t in_channels = 0;
        int32_t kernel_h = 0;
        int32_t kernel_w = 0;
        int32_t pad_h = 0;
        int32_t pad_w = 0;
        int32_t groups = 1;
        const float* weight = nullptr;
        int32_t weight_count = 0;
        const float* bias = nullptr;
        int32_t bias_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "out_channels", out_channels))
                    continue;
                if (read_int_plugin_field(f, "in_channels", in_channels))
                    continue;
                if (read_int_plugin_field(f, "kernel_h", kernel_h))
                    continue;
                if (read_int_plugin_field(f, "kernel_w", kernel_w))
                    continue;
                if (read_int_plugin_field(f, "pad_h", pad_h))
                    continue;
                if (read_int_plugin_field(f, "pad_w", pad_w))
                    continue;
                if (read_int_plugin_field(f, "groups", groups))
                    continue;
                if (read_float_array_plugin_field(f, "weight", weight, weight_count))
                    continue;
                read_float_array_plugin_field(f, "bias", bias, bias_count);
            }
        }
        return new SanaWmTorchConv2dPlugin(out_channels, in_channels, kernel_h, kernel_w, pad_h,
                                           pad_w, groups, weight, weight_count, bias, bias_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmTorchConv2dPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmTorchConv3dCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmTorchConv3dCreator() {
        fields_.push_back({"out_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"in_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"stride_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"stride_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"stride_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"pad_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"pad_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"pad_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"dilation_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"dilation_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"dilation_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmTorchConv3dPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmTorchConv3dPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t out_channels = 0;
        int32_t in_channels = 0;
        int32_t kernel_t = 0;
        int32_t kernel_h = 0;
        int32_t kernel_w = 0;
        int32_t stride_t = 1;
        int32_t stride_h = 1;
        int32_t stride_w = 1;
        int32_t pad_t = 0;
        int32_t pad_h = 0;
        int32_t pad_w = 0;
        int32_t dilation_t = 1;
        int32_t dilation_h = 1;
        int32_t dilation_w = 1;
        int32_t groups = 1;
        int32_t output_t = 0;
        int32_t output_h = 0;
        int32_t output_w = 0;
        const float* weight = nullptr;
        int32_t weight_count = 0;
        const float* bias = nullptr;
        int32_t bias_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "out_channels", out_channels))
                    continue;
                if (read_int_plugin_field(f, "in_channels", in_channels))
                    continue;
                if (read_int_plugin_field(f, "kernel_t", kernel_t))
                    continue;
                if (read_int_plugin_field(f, "kernel_h", kernel_h))
                    continue;
                if (read_int_plugin_field(f, "kernel_w", kernel_w))
                    continue;
                if (read_int_plugin_field(f, "stride_t", stride_t))
                    continue;
                if (read_int_plugin_field(f, "stride_h", stride_h))
                    continue;
                if (read_int_plugin_field(f, "stride_w", stride_w))
                    continue;
                if (read_int_plugin_field(f, "pad_t", pad_t))
                    continue;
                if (read_int_plugin_field(f, "pad_h", pad_h))
                    continue;
                if (read_int_plugin_field(f, "pad_w", pad_w))
                    continue;
                if (read_int_plugin_field(f, "dilation_t", dilation_t))
                    continue;
                if (read_int_plugin_field(f, "dilation_h", dilation_h))
                    continue;
                if (read_int_plugin_field(f, "dilation_w", dilation_w))
                    continue;
                if (read_int_plugin_field(f, "groups", groups))
                    continue;
                if (read_int_plugin_field(f, "output_t", output_t))
                    continue;
                if (read_int_plugin_field(f, "output_h", output_h))
                    continue;
                if (read_int_plugin_field(f, "output_w", output_w))
                    continue;
                if (read_float_array_plugin_field(f, "weight", weight, weight_count))
                    continue;
                read_float_array_plugin_field(f, "bias", bias, bias_count);
            }
        }
        return new SanaWmTorchConv3dPlugin(
            out_channels, in_channels, kernel_t, kernel_h, kernel_w, stride_t, stride_h, stride_w,
            pad_t, pad_h, pad_w, dilation_t, dilation_h, dilation_w, groups, output_t, output_h,
            output_w, weight, weight_count, bias, bias_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmTorchConv3dPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmVaeRmsSiluCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmVaeRmsSiluCreator() {
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmVaeRmsSiluPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmVaeRmsSiluPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float eps = 1.0e-8F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_float_plugin_field(fc->fields[i], "eps", eps);
        }
        return new SanaWmVaeRmsSiluPlugin(eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmVaeRmsSiluPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmVaeDenormalizeCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmVaeDenormalizeCreator() {
        fields_.push_back({"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"scaling_factor", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"mean", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"std", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmVaeDenormalizePlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmVaeDenormalizePlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t channels = 0;
        float scaling_factor = 1.0F;
        const float* mean = nullptr;
        int32_t mean_count = 0;
        const float* std = nullptr;
        int32_t std_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "channels", channels))
                    continue;
                if (read_float_plugin_field(field, "scaling_factor", scaling_factor))
                    continue;
                if (read_float_array_plugin_field(field, "mean", mean, mean_count))
                    continue;
                read_float_array_plugin_field(field, "std", std, std_count);
            }
        }
        return new SanaWmVaeDenormalizePlugin(channels, scaling_factor, mean, mean_count, std,
                                              std_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmVaeDenormalizePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmVaeLayerNormCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmVaeLayerNormCreator() {
        fields_.push_back({"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmVaeLayerNormPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmVaeLayerNormPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t channels = 0;
        float eps = 1.0e-6F;
        const float* weight = nullptr;
        int32_t weight_count = 0;
        const float* bias = nullptr;
        int32_t bias_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "channels", channels))
                    continue;
                if (read_float_plugin_field(field, "eps", eps))
                    continue;
                if (read_float_array_plugin_field(field, "weight", weight, weight_count))
                    continue;
                read_float_array_plugin_field(field, "bias", bias, bias_count);
            }
        }
        return new SanaWmVaeLayerNormPlugin(channels, eps, weight, weight_count, bias, bias_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmVaeLayerNormPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGlumbconvTempCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGlumbconvTempCreator() {
        fields_.push_back({"batch", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"height", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"width", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"hidden", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"t_kernel", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"inverted_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"inverted_bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"depth_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"depth_bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"point_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"t_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGlumbconvTempPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmGlumbconvTempPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t batch = 0;
        int32_t frames = 0;
        int32_t height = 0;
        int32_t width = 0;
        int32_t channels = 0;
        int32_t hidden = 0;
        int32_t t_kernel = 3;
        const float* inverted_weight = nullptr;
        int32_t inverted_weight_count = 0;
        const float* inverted_bias = nullptr;
        int32_t inverted_bias_count = 0;
        const float* depth_weight = nullptr;
        int32_t depth_weight_count = 0;
        const float* depth_bias = nullptr;
        int32_t depth_bias_count = 0;
        const float* point_weight = nullptr;
        int32_t point_weight_count = 0;
        const float* t_weight = nullptr;
        int32_t t_weight_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "batch", batch))
                    continue;
                if (read_int_plugin_field(f, "frames", frames))
                    continue;
                if (read_int_plugin_field(f, "height", height))
                    continue;
                if (read_int_plugin_field(f, "width", width))
                    continue;
                if (read_int_plugin_field(f, "channels", channels))
                    continue;
                if (read_int_plugin_field(f, "hidden", hidden))
                    continue;
                if (read_int_plugin_field(f, "t_kernel", t_kernel))
                    continue;
                if (read_float_array_plugin_field(f, "inverted_weight", inverted_weight,
                                                  inverted_weight_count))
                    continue;
                if (read_float_array_plugin_field(f, "inverted_bias", inverted_bias,
                                                  inverted_bias_count))
                    continue;
                if (read_float_array_plugin_field(f, "depth_weight", depth_weight,
                                                  depth_weight_count))
                    continue;
                if (read_float_array_plugin_field(f, "depth_bias", depth_bias, depth_bias_count))
                    continue;
                if (read_float_array_plugin_field(f, "point_weight", point_weight,
                                                  point_weight_count))
                    continue;
                read_float_array_plugin_field(f, "t_weight", t_weight, t_weight_count);
            }
        }
        return new SanaWmGlumbconvTempPlugin(
            batch, frames, height, width, channels, hidden, t_kernel, inverted_weight,
            inverted_weight_count, inverted_bias, inverted_bias_count, depth_weight,
            depth_weight_count, depth_bias, depth_bias_count, point_weight, point_weight_count,
            t_weight, t_weight_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGlumbconvTempPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmTimestepEmbedCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmTimestepEmbedCreator() {
        fields_.push_back({"frequency_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"hidden_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"freqs", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"w0", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"b0", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"w1", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"b1", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"w2", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"b2", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmTimestepEmbedPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmTimestepEmbedPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frequency_dim = 0;
        int32_t hidden_size = 0;
        const float* freqs = nullptr;
        int32_t freqs_count = 0;
        const float* w0 = nullptr;
        int32_t w0_count = 0;
        const float* b0 = nullptr;
        int32_t b0_count = 0;
        const float* w1 = nullptr;
        int32_t w1_count = 0;
        const float* b1 = nullptr;
        int32_t b1_count = 0;
        const float* w2 = nullptr;
        int32_t w2_count = 0;
        const float* b2 = nullptr;
        int32_t b2_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "frequency_dim", frequency_dim))
                    continue;
                if (read_int_plugin_field(f, "hidden_size", hidden_size))
                    continue;
                if (read_float_array_plugin_field(f, "freqs", freqs, freqs_count))
                    continue;
                if (read_float_array_plugin_field(f, "w0", w0, w0_count))
                    continue;
                if (read_float_array_plugin_field(f, "b0", b0, b0_count))
                    continue;
                if (read_float_array_plugin_field(f, "w1", w1, w1_count))
                    continue;
                if (read_float_array_plugin_field(f, "b1", b1, b1_count))
                    continue;
                if (read_float_array_plugin_field(f, "w2", w2, w2_count))
                    continue;
                read_float_array_plugin_field(f, "b2", b2, b2_count);
            }
        }
        return new SanaWmTimestepEmbedPlugin(frequency_dim, hidden_size, freqs, freqs_count, w0,
                                             w0_count, b0, b0_count, w1, w1_count, b1, b1_count, w2,
                                             w2_count, b2, b2_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmTimestepEmbedPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmT2IModulateCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmT2IModulateCreator() {
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmT2IModulatePlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmT2IModulatePlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2*
    createPlugin(char const* /*name*/,
                 nvinfer1::PluginFieldCollection const* /*fc*/) noexcept override {
        return new SanaWmT2IModulatePlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmT2IModulatePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmCaptionEmbedCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmCaptionEmbedCreator() {
        fields_.push_back({"hidden_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmCaptionEmbedPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmCaptionEmbedPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t hidden_size = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_int_plugin_field(fc->fields[i], "hidden_size", hidden_size);
        }
        return new SanaWmCaptionEmbedPlugin(hidden_size);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmCaptionEmbedPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmCrossAttentionCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmCrossAttentionCreator() {
        fields_.push_back({"num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmCrossAttentionPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmCrossAttentionPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t num_heads = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_int_plugin_field(fc->fields[i], "num_heads", num_heads);
        }
        return new SanaWmCrossAttentionPlugin(num_heads);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmCrossAttentionPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmSoftmaxAttentionCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmSoftmaxAttentionCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"camera", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmSoftmaxAttentionPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmSoftmaxAttentionPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        int32_t head_dim = 0;
        int32_t camera = 0;
        float norm_eps = 1.0e-5F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "frames", frames))
                    continue;
                if (read_int_plugin_field(field, "spatial", spatial))
                    continue;
                if (read_int_plugin_field(field, "heads", heads))
                    continue;
                if (read_int_plugin_field(field, "head_dim", head_dim))
                    continue;
                if (read_float_plugin_field(field, "norm_eps", norm_eps))
                    continue;
                read_int_plugin_field(field, "camera", camera);
            }
        }
        return new SanaWmSoftmaxAttentionPlugin(frames, spatial, heads, head_dim, norm_eps,
                                                camera != 0);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmSoftmaxAttentionPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmTorchCamPrepCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmTorchCamPrepCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmTorchCamPrepPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmTorchCamPrepPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        int32_t head_dim = 0;
        float norm_eps = 1.0e-5F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "frames", frames))
                    continue;
                if (read_int_plugin_field(field, "spatial", spatial))
                    continue;
                if (read_int_plugin_field(field, "heads", heads))
                    continue;
                if (read_int_plugin_field(field, "head_dim", head_dim))
                    continue;
                read_float_plugin_field(field, "norm_eps", norm_eps);
            }
        }
        return new SanaWmTorchCamPrepPlugin(frames, spatial, heads, head_dim, norm_eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmTorchCamPrepPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmCameraBetaDiscountCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmCameraBetaDiscountCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmCameraBetaDiscountPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmCameraBetaDiscountPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "frames", frames))
                    continue;
                if (read_int_plugin_field(field, "spatial", spatial))
                    continue;
                read_int_plugin_field(field, "heads", heads);
            }
        }
        return new SanaWmCameraBetaDiscountPlugin(frames, spatial, heads);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmCameraBetaDiscountPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmFrameGateCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmFrameGateCreator() {
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmFrameGatePlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmFrameGatePlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2*
    createPlugin(char const* /*name*/,
                 nvinfer1::PluginFieldCollection const* /*fc*/) noexcept override {
        return new SanaWmFrameGatePlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmFrameGatePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmFrameMeanCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmFrameMeanCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmFrameMeanPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmFrameMeanPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                read_int_plugin_field(fc->fields[i], "frames", frames);
                read_int_plugin_field(fc->fields[i], "spatial", spatial);
            }
        }
        return new SanaWmFrameMeanPlugin(frames, spatial);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmFrameMeanPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLayerNormCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLayerNormCreator() {
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmLayerNormPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmLayerNormPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_float_plugin_field(fc->fields[i], "eps", eps);
        }
        return new SanaWmLayerNormPlugin(eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLayerNormPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmShortConvCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmShortConvCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmShortConvPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmShortConvPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t channels = 0;
        int32_t kernel_size = 0;
        const float* weight = nullptr;
        int32_t weight_count = 0;
        const float* bias = nullptr;
        int32_t bias_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "frames", frames))
                    continue;
                if (read_int_plugin_field(f, "spatial", spatial))
                    continue;
                if (read_int_plugin_field(f, "channels", channels))
                    continue;
                if (read_int_plugin_field(f, "kernel_size", kernel_size))
                    continue;
                if (read_float_array_plugin_field(f, "weight", weight, weight_count))
                    continue;
                read_float_array_plugin_field(f, "bias", bias, bias_count);
            }
        }
        return new SanaWmShortConvPlugin(frames, spatial, channels, kernel_size, weight,
                                         weight_count, bias, bias_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmShortConvPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGateProjCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGateProjCreator() {
        fields_.push_back({"input_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"activation", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"use_matmul_bias", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGateProjPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmGateProjPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t input_dim = 0;
        int32_t output_dim = 0;
        int32_t activation = 0;
        int32_t use_matmul_bias = 0;
        const float* weight = nullptr;
        int32_t weight_count = 0;
        const float* bias = nullptr;
        int32_t bias_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "input_dim", input_dim))
                    continue;
                if (read_int_plugin_field(f, "output_dim", output_dim))
                    continue;
                if (read_int_plugin_field(f, "activation", activation))
                    continue;
                if (read_int_plugin_field(f, "use_matmul_bias", use_matmul_bias))
                    continue;
                if (read_float_array_plugin_field(f, "weight", weight, weight_count))
                    continue;
                read_float_array_plugin_field(f, "bias", bias, bias_count);
            }
        }
        return new SanaWmGateProjPlugin(input_dim, output_dim, activation, use_matmul_bias, weight,
                                        weight_count, bias, bias_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGateProjPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmDecayCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmDecayCreator() {
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"a_log", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return SanaWmDecayPlugin::kPLUGIN_NAME; }

    char const* getPluginVersion() const noexcept override {
        return SanaWmDecayPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t heads = 0;
        const float* a_log_values = nullptr;
        int32_t a_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "heads", heads))
                    continue;
                read_float_array_plugin_field(f, "a_log", a_log_values, a_count);
            }
        }
        return new SanaWmDecayPlugin(heads, a_log_values, a_count);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmDecayPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGemmaRmsNormCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGemmaRmsNormCreator() {
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGemmaRmsNormPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmGemmaRmsNormPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_float_plugin_field(fc->fields[i], "eps", eps);
        }
        return new SanaWmGemmaRmsNormPlugin(eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGemmaRmsNormPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGemmaGatedGeluCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGemmaGatedGeluCreator() {
        fc_.nbFields = 0;
        fc_.fields = nullptr;
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGemmaGatedGeluPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmGemmaGatedGeluPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new SanaWmGemmaGatedGeluPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGemmaGatedGeluPlugin(data, length);
    }

  private:
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGemmaRopeCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGemmaRopeCreator() {
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"rotary_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"interleaved", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGemmaRopePlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmGemmaRopePlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t heads = 0;
        int32_t head_dim = 0;
        int32_t rotary_dim = 0;
        int32_t interleaved = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "heads", heads))
                    continue;
                if (read_int_plugin_field(f, "head_dim", head_dim))
                    continue;
                if (read_int_plugin_field(f, "rotary_dim", rotary_dim))
                    continue;
                read_int_plugin_field(f, "interleaved", interleaved);
            }
        }
        return new SanaWmGemmaRopePlugin(heads, head_dim, rotary_dim, interleaved != 0);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGemmaRopePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmGemmaAttentionCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGemmaAttentionCreator() {
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kv_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmGemmaAttentionPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmGemmaAttentionPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t heads = 0;
        int32_t kv_heads = 0;
        int32_t head_dim = 0;
        float scale = 1.0F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (read_int_plugin_field(f, "heads", heads))
                    continue;
                if (read_int_plugin_field(f, "kv_heads", kv_heads))
                    continue;
                if (read_int_plugin_field(f, "head_dim", head_dim))
                    continue;
                read_float_plugin_field(f, "scale", scale);
            }
        }
        return new SanaWmGemmaAttentionPlugin(heads, kv_heads, head_dim, scale);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmGemmaAttentionPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxTextNormalizeCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxTextNormalizeCreator() {
        fields_.push_back({"caption_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"layer_count", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"scale_factor", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxTextNormalizePlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxTextNormalizePlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t caption_channels = 0;
        int32_t layer_count = 0;
        float scale_factor = 8.0F;
        float eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "caption_channels", caption_channels))
                    continue;
                if (read_int_plugin_field(field, "layer_count", layer_count))
                    continue;
                if (read_float_plugin_field(field, "scale_factor", scale_factor))
                    continue;
                read_float_plugin_field(field, "eps", eps);
            }
        }
        return new SanaWmLtxTextNormalizePlugin(caption_channels, layer_count, scale_factor, eps);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxTextNormalizePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxRegisterCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxRegisterCreator() {
        fields_.push_back({"register_count", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"registers", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxRegisterPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxRegisterPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t register_count = 0;
        int32_t hidden_dim = 0;
        const float* registers = nullptr;
        int32_t value_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "register_count", register_count))
                    continue;
                if (read_int_plugin_field(field, "hidden_dim", hidden_dim))
                    continue;
                read_float_array_plugin_field(field, "registers", registers, value_count);
            }
        }
        return new SanaWmLtxRegisterPlugin(register_count, hidden_dim, registers, value_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxRegisterPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxConnectorBlockCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxConnectorBlockCreator() {
        fields_.push_back({"hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"ff_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"packed_weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxConnectorBlockPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxConnectorBlockPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t hidden_dim = 0;
        int32_t num_heads = 0;
        int32_t head_dim = 0;
        int32_t ff_dim = 0;
        const float* packed_weights = nullptr;
        int32_t weight_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "hidden_dim", hidden_dim))
                    continue;
                if (read_int_plugin_field(field, "num_heads", num_heads))
                    continue;
                if (read_int_plugin_field(field, "head_dim", head_dim))
                    continue;
                if (read_int_plugin_field(field, "ff_dim", ff_dim))
                    continue;
                read_float_array_plugin_field(field, "packed_weights", packed_weights,
                                              weight_count);
            }
        }
        return new SanaWmLtxConnectorBlockPlugin(hidden_dim, num_heads, head_dim, ff_dim,
                                                 packed_weights, weight_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxConnectorBlockPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxRmsNormCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxRmsNormCreator() {
        fields_.push_back({"eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxRmsNormPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxRmsNormPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i)
                read_float_plugin_field(fc->fields[i], "eps", eps);
        }
        return new SanaWmLtxRmsNormPlugin(eps);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxRmsNormPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxTimestepFrequencyCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxTimestepFrequencyCreator() {
        fields_.push_back({"frequency_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"max_period", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxTimestepFrequencyPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxTimestepFrequencyPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t frequency_dim = 256;
        float max_period = 10000.0F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                if (read_int_plugin_field(fc->fields[i], "frequency_dim", frequency_dim))
                    continue;
                read_float_plugin_field(fc->fields[i], "max_period", max_period);
            }
        }
        return new SanaWmLtxTimestepFrequencyPlugin(frequency_dim, max_period);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxTimestepFrequencyPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxVideoBlockCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxVideoBlockCreator() {
        for (const char* name :
             {"hidden_dim", "num_heads", "head_dim", "ff_dim", "context_tokens", "debug"}) {
            fields_.push_back({name, nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        }
        fields_.push_back({"packed_weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxVideoBlockPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxVideoBlockPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t hidden_dim = 0;
        int32_t num_heads = 0;
        int32_t head_dim = 0;
        int32_t ff_dim = 0;
        int32_t context_tokens = 0;
        int32_t debug = 0;
        const float* packed_weights = nullptr;
        int32_t weight_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "hidden_dim", hidden_dim))
                    continue;
                if (read_int_plugin_field(field, "num_heads", num_heads))
                    continue;
                if (read_int_plugin_field(field, "head_dim", head_dim))
                    continue;
                if (read_int_plugin_field(field, "ff_dim", ff_dim))
                    continue;
                if (read_int_plugin_field(field, "context_tokens", context_tokens))
                    continue;
                if (read_int_plugin_field(field, "debug", debug))
                    continue;
                read_float_array_plugin_field(field, "packed_weights", packed_weights,
                                              weight_count);
            }
        }
        return new SanaWmLtxVideoBlockPlugin(hidden_dim, num_heads, head_dim, ff_dim,
                                             context_tokens, debug != 0, packed_weights,
                                             weight_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxVideoBlockPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

class SanaWmLtxVideoOutputCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmLtxVideoOutputCreator() {
        for (const char* name : {"hidden_dim", "output_dim"})
            fields_.push_back({name, nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"packed_weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override {
        return SanaWmLtxVideoOutputPlugin::kPLUGIN_NAME;
    }
    char const* getPluginVersion() const noexcept override {
        return SanaWmLtxVideoOutputPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t hidden_dim = 0;
        int32_t output_dim = 0;
        const float* packed_weights = nullptr;
        int32_t weight_count = 0;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (read_int_plugin_field(field, "hidden_dim", hidden_dim))
                    continue;
                if (read_int_plugin_field(field, "output_dim", output_dim))
                    continue;
                read_float_array_plugin_field(field, "packed_weights", packed_weights,
                                              weight_count);
            }
        }
        return new SanaWmLtxVideoOutputPlugin(hidden_dim, output_dim, packed_weights, weight_count);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmLtxVideoOutputPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};
#endif

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmGdnCreator> pluginRegistrarSanaWmGdn{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmUcpeCreator> pluginRegistrarSanaWmUcpe{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmCamPrepCreator> pluginRegistrarSanaWmCamPrep{};
#if TRTMC_HAS_LIBTORCH_CONV
static nvinfer1::PluginRegistrar<trtmc::SanaWmTorchConv2dCreator>
    pluginRegistrarSanaWmTorchConv2d{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmTorchConv3dCreator>
    pluginRegistrarSanaWmTorchConv3d{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmVaeRmsSiluCreator> pluginRegistrarSanaWmVaeRmsSilu{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmVaeDenormalizeCreator>
    pluginRegistrarSanaWmVaeDenormalize{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmVaeLayerNormCreator>
    pluginRegistrarSanaWmVaeLayerNorm{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGlumbconvTempCreator>
    pluginRegistrarSanaWmGlumbconvTemp{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmTimestepEmbedCreator>
    pluginRegistrarSanaWmTimestepEmbed{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmT2IModulateCreator>
    pluginRegistrarSanaWmT2IModulate{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmCaptionEmbedCreator>
    pluginRegistrarSanaWmCaptionEmbed{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmCrossAttentionCreator>
    pluginRegistrarSanaWmCrossAttention{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmSoftmaxAttentionCreator>
    pluginRegistrarSanaWmSoftmaxAttention{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmTorchCamPrepCreator>
    pluginRegistrarSanaWmTorchCamPrep{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmCameraBetaDiscountCreator>
    pluginRegistrarSanaWmCameraBetaDiscount{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmFrameGateCreator> pluginRegistrarSanaWmFrameGate{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmFrameMeanCreator> pluginRegistrarSanaWmFrameMean{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLayerNormCreator> pluginRegistrarSanaWmLayerNorm{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmShortConvCreator> pluginRegistrarSanaWmShortConv{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGateProjCreator> pluginRegistrarSanaWmGateProj{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmDecayCreator> pluginRegistrarSanaWmDecay{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGemmaRmsNormCreator>
    pluginRegistrarSanaWmGemmaRmsNorm{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGemmaGatedGeluCreator>
    pluginRegistrarSanaWmGemmaGatedGelu{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGemmaRopeCreator> pluginRegistrarSanaWmGemmaRope{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmGemmaAttentionCreator>
    pluginRegistrarSanaWmGemmaAttention{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxTextNormalizeCreator>
    pluginRegistrarSanaWmLtxTextNormalize{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxRegisterCreator>
    pluginRegistrarSanaWmLtxRegister{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxConnectorBlockCreator>
    pluginRegistrarSanaWmLtxConnectorBlock{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxRmsNormCreator> pluginRegistrarSanaWmLtxRmsNorm{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxTimestepFrequencyCreator>
    pluginRegistrarSanaWmLtxTimestepFrequency{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxVideoBlockCreator>
    pluginRegistrarSanaWmLtxVideoBlock{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmLtxVideoOutputCreator>
    pluginRegistrarSanaWmLtxVideoOutput{};
#endif

extern "C" void sana_wm_gdn_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

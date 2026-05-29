#if TRTMC_HAS_TRT

#include "plugins/sana_wm_gdn_plugin.h"

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
                read_int_plugin_field(f, "tree_reduce", tree_reduce);
                read_int_plugin_field(f, "downscale", downscale);
            }
        }
        return new SanaWmUcpePlugin(frames, spatial, heads, head_dim, inverse != 0,
                                    tree_reduce != 0, downscale != 0);
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

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmGdnCreator> pluginRegistrarSanaWmGdn{};
static nvinfer1::PluginRegistrar<trtmc::SanaWmUcpeCreator> pluginRegistrarSanaWmUcpe{};

extern "C" void sana_wm_gdn_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

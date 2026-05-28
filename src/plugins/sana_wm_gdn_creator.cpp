#if TRTMC_HAS_TRT

#include "plugins/sana_wm_gdn_plugin.h"

#include <NvInferRuntime.h>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {

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
                if (std::strcmp(f.name, "mode") == 0 &&
                    f.type == nvinfer1::PluginFieldType::kINT32 && f.data != nullptr) {
                    mode = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "reverse_output") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kINT32 &&
                           f.data != nullptr) {
                    reverse = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "eps") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32 &&
                           f.data != nullptr) {
                    eps = *static_cast<const float*>(f.data);
                } else if (std::strcmp(f.name, "frames") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kINT32 && f.data != nullptr) {
                    frames = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "head_dim") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kINT32 && f.data != nullptr) {
                    head_dim = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "norm_eps") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32 &&
                           f.data != nullptr) {
                    norm_eps = *static_cast<const float*>(f.data);
                }
            }
        }
        auto plugin_mode = SanaWmGdnPlugin::Mode::kMain;
        if (mode == 1) {
            plugin_mode = SanaWmGdnPlugin::Mode::kCamera;
        } else if (mode == 2) {
            plugin_mode = SanaWmGdnPlugin::Mode::kMainCombined;
        } else if (mode == 3) {
            plugin_mode = SanaWmGdnPlugin::Mode::kMainRawCombined;
        } else if (mode == 4) {
            plugin_mode = SanaWmGdnPlugin::Mode::kCameraCombined;
        }
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

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmGdnCreator> pluginRegistrarSanaWmGdn{};

extern "C" void sana_wm_gdn_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

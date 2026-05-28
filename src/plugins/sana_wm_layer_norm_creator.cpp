#if TRTMC_HAS_TRT

#include "plugins/sana_wm_layer_norm_plugin.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {

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

    nvinfer1::IPluginV2* createPlugin(char const*, nvinfer1::PluginFieldCollection const* fc)
        noexcept override {
        float eps = 1.0e-6F;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (std::strcmp(field.name, "eps") == 0 && field.data != nullptr &&
                    field.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    eps = *static_cast<const float*>(field.data);
                }
            }
        }
        return new SanaWmLayerNormPlugin(eps);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data, size_t length) noexcept
        override {
        return new SanaWmLayerNormPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmLayerNormCreator>
    pluginRegistrarSanaWmLayerNorm{};

extern "C" void sana_wm_layer_norm_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

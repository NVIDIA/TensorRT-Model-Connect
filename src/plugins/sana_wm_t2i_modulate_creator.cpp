#if TRTMC_HAS_TRT

#include "plugins/sana_wm_t2i_modulate_plugin.h"

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trtmc {

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

    nvinfer1::IPluginV2* createPlugin(
        char const* /*name*/, nvinfer1::PluginFieldCollection const* /*fc*/) noexcept override {
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

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmT2IModulateCreator>
    pluginRegistrarSanaWmT2IModulate{};

extern "C" void sana_wm_t2i_modulate_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

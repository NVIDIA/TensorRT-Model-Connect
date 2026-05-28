#if TRTMC_HAS_TRT

#include "plugins/sana_wm_decay_plugin.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {
namespace {

std::vector<float> float_field_to_vector(const nvinfer1::PluginField& field) {
    std::vector<float> out;
    if (field.data == nullptr || field.length <= 0)
        return out;
    const auto* values = static_cast<const float*>(field.data);
    out.assign(values, values + field.length);
    return out;
}

} // namespace

class SanaWmDecayCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmDecayCreator() {
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"a_values", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmDecayPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmDecayPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t heads = 0;
        std::vector<float> a_values;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& field = fc->fields[i];
                if (std::strcmp(field.name, "heads") == 0 && field.data != nullptr) {
                    heads = *static_cast<const int32_t*>(field.data);
                } else if (std::strcmp(field.name, "a_values") == 0 &&
                           field.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    a_values = float_field_to_vector(field);
                }
            }
        }
        return new SanaWmDecayPlugin(heads, std::move(a_values));
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmDecayPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmDecayCreator> pluginRegistrarSanaWmDecay{};

extern "C" void sana_wm_decay_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

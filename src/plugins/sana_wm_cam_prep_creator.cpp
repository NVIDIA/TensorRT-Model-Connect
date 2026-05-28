#if TRTMC_HAS_TRT

#include "plugins/sana_wm_cam_prep_plugin.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
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

class SanaWmCamPrepCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmCamPrepCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"norm_eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
        fields_.push_back({"q_norm_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"k_norm_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
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

    nvinfer1::IPluginV2* createPlugin(char const*, nvinfer1::PluginFieldCollection const* fc)
        noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t heads = 0;
        int32_t head_dim = 0;
        float norm_eps = 1.0e-6F;
        std::vector<float> q_norm_weight;
        std::vector<float> k_norm_weight;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "frames") == 0 && f.data != nullptr) {
                    frames = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "spatial") == 0 && f.data != nullptr) {
                    spatial = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "heads") == 0 && f.data != nullptr) {
                    heads = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "head_dim") == 0 && f.data != nullptr) {
                    head_dim = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "norm_eps") == 0 && f.data != nullptr &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    norm_eps = *static_cast<const float*>(f.data);
                } else if (std::strcmp(f.name, "q_norm_weight") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    q_norm_weight = float_field_to_vector(f);
                } else if (std::strcmp(f.name, "k_norm_weight") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    k_norm_weight = float_field_to_vector(f);
                }
            }
        }
        return new SanaWmCamPrepPlugin(frames, spatial, heads, head_dim, norm_eps,
                                       std::move(q_norm_weight), std::move(k_norm_weight));
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data, size_t length)
        noexcept override {
        return new SanaWmCamPrepPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmCamPrepCreator> pluginRegistrarSanaWmCamPrep{};

extern "C" void sana_wm_cam_prep_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

#if TRTMC_HAS_TRT

#include "plugins/sana_wm_rope_plugin.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {

class SanaWmRopeCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmRopeCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"inverse", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"use_double", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_bf16", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmRopePlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmRopePlugin::kPLUGIN_VERSION;
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
        bool inverse = false;
        bool use_double = false;
        bool output_bf16 = false;
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
                } else if (std::strcmp(f.name, "inverse") == 0 && f.data != nullptr) {
                    inverse = *static_cast<const int32_t*>(f.data) != 0;
                } else if (std::strcmp(f.name, "use_double") == 0 && f.data != nullptr) {
                    use_double = *static_cast<const int32_t*>(f.data) != 0;
                } else if (std::strcmp(f.name, "output_bf16") == 0 && f.data != nullptr) {
                    output_bf16 = *static_cast<const int32_t*>(f.data) != 0;
                }
            }
        }
        return new SanaWmRopePlugin(frames, spatial, heads, head_dim, inverse, use_double,
                                    output_bf16);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data, size_t length)
        noexcept override {
        return new SanaWmRopePlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmRopeCreator> pluginRegistrarSanaWmRope{};

extern "C" void sana_wm_rope_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

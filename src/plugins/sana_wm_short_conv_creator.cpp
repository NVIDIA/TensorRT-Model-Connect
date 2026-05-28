#if TRTMC_HAS_TRT

#include "plugins/sana_wm_short_conv_plugin.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace trtmc {
namespace {

uint16_t fp32_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

std::vector<uint16_t> float_field_to_bf16(const nvinfer1::PluginField& field) {
    std::vector<uint16_t> out;
    if (field.data == nullptr || field.length <= 0)
        return out;
    const auto* values = static_cast<const float*>(field.data);
    out.reserve(static_cast<std::size_t>(field.length));
    for (int32_t i = 0; i < field.length; ++i)
        out.push_back(fp32_to_bf16_bits(values[i]));
    return out;
}

} // namespace

class SanaWmShortConvCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmShortConvCreator() {
        fields_.push_back({"frames", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"spatial", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
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

    nvinfer1::IPluginV2* createPlugin(char const*, nvinfer1::PluginFieldCollection const* fc)
        noexcept override {
        int32_t frames = 0;
        int32_t spatial = 0;
        int32_t channels = 0;
        int32_t kernel_size = 0;
        std::vector<uint16_t> weight;
        std::vector<uint16_t> bias;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "frames") == 0 && f.data != nullptr) {
                    frames = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "spatial") == 0 && f.data != nullptr) {
                    spatial = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "channels") == 0 && f.data != nullptr) {
                    channels = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "kernel_size") == 0 && f.data != nullptr) {
                    kernel_size = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "weight") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    weight = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "bias") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    bias = float_field_to_bf16(f);
                }
            }
        }
        return new SanaWmShortConvPlugin(frames, spatial, channels, kernel_size, std::move(weight),
                                         std::move(bias));
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data, size_t length) noexcept
        override {
        return new SanaWmShortConvPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmShortConvCreator>
    pluginRegistrarSanaWmShortConv{};

extern "C" void sana_wm_short_conv_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

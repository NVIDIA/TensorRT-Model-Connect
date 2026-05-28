#if TRTMC_HAS_TRT

#include "plugins/sana_wm_timestep_plugin.h"

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

class SanaWmTimestepEmbedCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmTimestepEmbedCreator() {
        fields_.push_back({"frequency_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"hidden_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"freqs", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"w0", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"b0", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"w1", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"b1", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"w2", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"b2", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
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
        std::vector<float> freqs;
        std::vector<uint16_t> w0;
        std::vector<uint16_t> b0;
        std::vector<uint16_t> w1;
        std::vector<uint16_t> b1;
        std::vector<uint16_t> w2;
        std::vector<uint16_t> b2;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "frequency_dim") == 0 && f.data != nullptr) {
                    frequency_dim = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "hidden_size") == 0 && f.data != nullptr) {
                    hidden_size = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "freqs") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32 && f.data != nullptr) {
                    const auto* values = static_cast<const float*>(f.data);
                    freqs.assign(values, values + f.length);
                } else if (std::strcmp(f.name, "w0") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    w0 = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "b0") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    b0 = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "w1") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    w1 = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "b1") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    b1 = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "w2") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    w2 = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "b2") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    b2 = float_field_to_bf16(f);
                }
            }
        }
        return new SanaWmTimestepEmbedPlugin(frequency_dim, hidden_size, std::move(freqs),
                                             std::move(w0), std::move(b0), std::move(w1),
                                             std::move(b1), std::move(w2), std::move(b2));
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

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmTimestepEmbedCreator>
    pluginRegistrarSanaWmTimestepEmbed{};

extern "C" void sana_wm_timestep_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

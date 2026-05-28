#if TRTMC_HAS_TRT

#include "plugins/sana_wm_gate_proj_plugin.h"

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

class SanaWmGateProjCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmGateProjCreator() {
        fields_.push_back({"input_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"output_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"activation", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
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

    nvinfer1::IPluginV2* createPlugin(char const*, nvinfer1::PluginFieldCollection const* fc)
        noexcept override {
        int32_t input_dim = 0;
        int32_t output_dim = 0;
        int32_t activation = 0;
        std::vector<uint16_t> weight;
        std::vector<uint16_t> bias;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "input_dim") == 0 && f.data != nullptr) {
                    input_dim = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "output_dim") == 0 && f.data != nullptr) {
                    output_dim = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "activation") == 0 && f.data != nullptr) {
                    activation = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "weight") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    weight = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "bias") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    bias = float_field_to_bf16(f);
                }
            }
        }
        return new SanaWmGateProjPlugin(input_dim, output_dim, std::move(weight),
                                        std::move(bias), activation);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data, size_t length)
        noexcept override {
        return new SanaWmGateProjPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmGateProjCreator>
    pluginRegistrarSanaWmGateProj{};

extern "C" void sana_wm_gate_proj_plugin_force_link() {}

#endif // TRTMC_HAS_TRT

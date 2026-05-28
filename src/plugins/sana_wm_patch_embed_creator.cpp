#if TRTMC_HAS_TRT && TRTMC_HAS_CUDNN

#include "plugins/sana_wm_patch_embed_plugin.h"

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

class SanaWmPatchEmbed3dCreator : public nvinfer1::IPluginCreator {
  public:
    SanaWmPatchEmbed3dCreator() {
        fields_.push_back({"out_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"in_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_t", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"kernel_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"algo", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"weight", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fields_.push_back({"bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return SanaWmPatchEmbed3dPlugin::kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override {
        return SanaWmPatchEmbed3dPlugin::kPLUGIN_VERSION;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }

    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const* /*name*/,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        int32_t out_channels = 0;
        int32_t in_channels = 0;
        int32_t kernel_t = 1;
        int32_t kernel_h = 1;
        int32_t kernel_w = 1;
        int32_t algo = 0;
        std::vector<uint16_t> weight;
        std::vector<uint16_t> bias;
        if (fc != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                const auto& f = fc->fields[i];
                if (std::strcmp(f.name, "out_channels") == 0 && f.data != nullptr) {
                    out_channels = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "in_channels") == 0 && f.data != nullptr) {
                    in_channels = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "kernel_t") == 0 && f.data != nullptr) {
                    kernel_t = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "kernel_h") == 0 && f.data != nullptr) {
                    kernel_h = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "kernel_w") == 0 && f.data != nullptr) {
                    kernel_w = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "algo") == 0 && f.data != nullptr) {
                    algo = *static_cast<const int32_t*>(f.data);
                } else if (std::strcmp(f.name, "weight") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    weight = float_field_to_bf16(f);
                } else if (std::strcmp(f.name, "bias") == 0 &&
                           f.type == nvinfer1::PluginFieldType::kFLOAT32) {
                    bias = float_field_to_bf16(f);
                }
            }
        }
        return new SanaWmPatchEmbed3dPlugin(out_channels, in_channels, kernel_t, kernel_h,
                                            kernel_w, algo, std::move(weight), std::move(bias));
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* /*name*/, void const* data,
                                           size_t length) noexcept override {
        return new SanaWmPatchEmbed3dPlugin(data, length);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::SanaWmPatchEmbed3dCreator>
    pluginRegistrarSanaWmPatchEmbed3d{};

extern "C" void sana_wm_patch_embed_plugin_force_link() {}

#endif // TRTMC_HAS_TRT && TRTMC_HAS_CUDNN

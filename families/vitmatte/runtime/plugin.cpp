/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/perception.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::vitmatte {
namespace {
class Pipeline final : public internal::IModel, public internal::IImageTrimapToAlpha {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, uint32_t height, uint32_t width)
        : engine_(std::move(engine)), height_(height), width_(width) {}
    const char* task() const noexcept override { return "image_trimap_to_alpha"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageTrimapToAlpha>(*this)};
    }
    internal::AlphaMatteResult run(const internal::ImageTrimapToAlphaRequest& request,
                                   internal::ConfigView config) override {
        internal::validate_config({}, config);
        const auto& image = request.image;
        const auto& trimap = request.trimap;
        const uint64_t area = static_cast<uint64_t>(height_) * width_;
        if (image.height != height_ || image.width != width_ || image.channels != 3 || !image.data ||
            trimap.height != height_ || trimap.width != width_ || trimap.states.size() != area || !trimap.states.data())
            throw std::invalid_argument("ViTMatte requires an RGB image and trimap matching its built profile");
        const bool byte = image.format == internal::ImageFormat::UInt8;
        if ((!byte && image.format != internal::ImageFormat::Float32) || image.byte_size != area * 3 * (byte ? 1 : sizeof(float)))
            throw std::invalid_argument("Invalid ViTMatte RGB storage");
        std::vector<float> pixels(4 * area);
        for (uint64_t index = 0; index < area; ++index) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float value = byte ? static_cast<const uint8_t*>(image.data)[index * 3 + channel] :
                    static_cast<const float*>(image.data)[index * 3 + channel] * 255.0F;
                if (!std::isfinite(value) || value < 0 || value > 255) throw std::invalid_argument("Invalid ViTMatte RGB value");
                pixels[channel * area + index] = (value - 127.5F) / 127.5F;
            }
            const auto state = trimap.states[index];
            if (state > 2) throw std::invalid_argument("Invalid ViTMatte trimap state");
            pixels[3 * area + index] = state == 1 ? 128.0F / 255.0F : state / 2.0F;
        }
        Tensor input;
        input.data = pixels.data(); input.shape = {1, 4, height_, width_}; input.dtype = DType::kFloat32;
        const auto outputs = engine_->forward({{"pixel_values", input}});
        const auto found = outputs.find("alpha");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 || found->second.numel() != area || !found->second.data)
            throw std::runtime_error("Invalid ViTMatte alpha output");
        internal::AlphaMatteResult result;
        result.alpha.rows = height_; result.alpha.columns = width_;
        const auto* values = static_cast<const float*>(found->second.data);
        result.alpha.values.assign(values, values + area);
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> engine_;
    uint32_t height_, width_;
};
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length) throw std::runtime_error(std::string("ViTMatte bundle lacks ") + name);
    return reader.read_section(name);
}
} // namespace
} // namespace trtmc::vitmatte
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "image_trimap_to_alpha")
        throw std::invalid_argument("ViTMatte requires image-trimap-to-alpha without an external cache");
    const auto raw = vitmatte::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int64_t height = cfg.at("height"), width = cfg.at("width");
    if (height < 32 || width < 32 || height > 1024 || width > 1024 || height % 32 || width % 32)
        throw std::runtime_error("Invalid ViTMatte image profile");
    const auto plan = vitmatte::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok()) throw std::runtime_error("ViTMatte engine failed to load");
    return new vitmatte::Pipeline(std::move(engine), static_cast<uint32_t>(height), static_cast<uint32_t>(width));
}

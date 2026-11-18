/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/preprocess.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::vjepa2 {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error(std::string("V-JEPA2 bundle lacks ") + name);
    return reader.read_section(name);
}
class Pipeline final : public internal::IModel, public internal::IVideoToTokenFeatures {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json cfg)
        : engine_(std::move(engine)), cfg_(std::move(cfg)) {}
    const char* task() const noexcept override { return "video_to_token_features"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IVideoToTokenFeatures>(*this)};
    }
    internal::VideoTokenFeaturesResult run(const internal::VideoToTokenFeaturesRequest& request,
                                           internal::ConfigView config) override {
        internal::validate_config({}, config);
        const int32_t frames = cfg_.at("frames"), size = cfg_.at("size"),
                      shortest = cfg_.at("shortest_edge");
        const int32_t grid = cfg_.at("grid"), temporal = cfg_.at("grid_frames"),
                      width = cfg_.at("hidden_size");
        const int32_t patch = cfg_.at("patch"), tubelet = cfg_.at("tubelet");
        if (request.video.frames.size() != static_cast<std::size_t>(frames))
            throw std::invalid_argument("V-JEPA2 clip differs from the built frame-count profile");
        const auto& first = request.video.frames[0];
        if (first.height == 0 || first.width == 0 || first.height > INT32_MAX ||
            first.width > INT32_MAX ||
            static_cast<uint64_t>(first.height) >
                std::numeric_limits<std::size_t>::max() / first.width / 3 / sizeof(float))
            throw std::invalid_argument("Invalid V-JEPA2 input frame dimensions");
        const auto geometry = image_geometry(first.height, first.width, shortest, size);
        const auto plane = static_cast<std::size_t>(size) * size;
        std::vector<float> pixels(3 * frames * plane);
        for (int32_t frame = 0; frame < frames; ++frame) {
            const auto& image = request.video.frames[frame];
            if (!image.data || image.format != internal::ImageFormat::Float32 ||
                image.channels != 3 || image.height != first.height || image.width != first.width ||
                image.byte_size !=
                    static_cast<std::size_t>(first.height) * first.width * 3 * sizeof(float))
                throw std::invalid_argument(
                    "V-JEPA2 requires equal-sized contiguous unit RGB float32 frames");
            const auto prepared = preprocess(static_cast<const float*>(image.data), image.height,
                                             image.width, shortest, size);
            for (int32_t channel = 0; channel < 3; ++channel)
                std::copy_n(prepared.data() + channel * plane, plane,
                            pixels.data() + (channel * frames + frame) * plane);
        }
        Tensor input;
        input.data = pixels.data();
        input.dtype = DType::kFloat32;
        input.shape = {1, 3, frames, size, size};
        const auto outputs = engine_->forward({{"pixel_values", input}});
        const auto found = outputs.find("features");
        const auto count = static_cast<std::size_t>(temporal) * grid * grid;
        if (found == outputs.end() || !found->second.data ||
            found->second.dtype != DType::kFloat32 ||
            found->second.shape != std::vector<int64_t>{static_cast<int64_t>(count), width})
            throw std::runtime_error("V-JEPA2 returned an invalid patch feature matrix");
        internal::VideoTokenFeaturesResult result;
        const auto* values = static_cast<const float*>(found->second.data);
        result.features = {{values, values + count * width}, count, static_cast<uint64_t>(width)};
        for (const float value : result.features.values)
            if (!std::isfinite(value))
                throw std::runtime_error("V-JEPA2 features contain a nonfinite value");
        result.grid_frames = temporal;
        result.grid_rows = result.grid_columns = grid;
        result.source_frame_count = frames;
        result.tokens.reserve(count);
        for (int32_t t = 0; t < temporal; ++t)
            for (int32_t row = 0; row < grid; ++row)
                for (int32_t column = 0; column < grid; ++column) {
                    internal::VideoFeatureToken token;
                    token.grid_frame = t;
                    token.grid_row = row;
                    token.grid_column = column;
                    token.frame_begin = t * tubelet;
                    token.frame_end = (t + 1) * tubelet;
                    token.x_min = float(geometry.left + column * patch) / geometry.width;
                    token.x_max = float(geometry.left + (column + 1) * patch) / geometry.width;
                    token.y_min = float(geometry.top + row * patch) / geometry.height;
                    token.y_max = float(geometry.top + (row + 1) * patch) / geometry.height;
                    result.tokens.push_back(token);
                }
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    nlohmann::json cfg_;
};
} // namespace
} // namespace trtmc::vjepa2
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("vjepa2 does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "video_to_token_features")
        throw std::invalid_argument(
            "V-JEPA2 requires video token features without an external cache");
    const auto raw = vjepa2::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t frames = cfg.at("frames"), size = cfg.at("size"),
                  shortest = cfg.at("shortest_edge");
    const int32_t patch = cfg.at("patch"), tubelet = cfg.at("tubelet"),
                  width = cfg.at("hidden_size");
    if (frames <= 0 || size <= 0 || shortest < size || patch <= 0 || tubelet <= 0 || width <= 0 ||
        size % patch || frames % tubelet || cfg.at("grid") != size / patch ||
        cfg.at("grid_frames") != frames / tubelet)
        throw std::runtime_error("Invalid V-JEPA2 patch profile");
    const auto plan = vjepa2::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("V-JEPA2 encoder failed to load");
    return new vjepa2::Pipeline(std::move(engine), cfg);
}

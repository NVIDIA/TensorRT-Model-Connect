/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace {
using namespace trtmc::internal;

class ImageFixture final : public IModel,
                           public ITextToImage,
                           public IImagesTextToImageEdit,
                           public IMaskedImageTextToImage,
                           public IBatchTextToImage {
  public:
    explicit ImageFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "single_only")
            return {bind<ITextToImage>(*this, fields_for(ITextToImage::kTask))};
        return {bind<ITextToImage>(*this, fields_for(ITextToImage::kTask)),
                bind<IImagesTextToImageEdit>(*this, fields_for(IImagesTextToImageEdit::kTask)),
                bind<IMaskedImageTextToImage>(*this, fields_for(IMaskedImageTextToImage::kTask)),
                bind<IBatchTextToImage>(*this, fields_for(IBatchTextToImage::kTask))};
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view task_id) const {
        if (mode_ == "single_only" && task_id != ITextToImage::kTask)
            throw UnsupportedTask("image fixture task is disabled in this bundle");
        static const ConfigField single[] = {
            {"level", ConfigKind::F64, ConfigValue{0.25}, "Generated red-channel intensity"}};
        static const ConfigField batch[] = {
            {"level", ConfigKind::F64, ConfigValue{0.25}, "Generated red-channel intensity"},
            {"seed", ConfigKind::I64, std::nullopt,
             "Optional seed marker; absence retains the batch marker"}};
        if (task_id == IBatchTextToImage::kTask)
            return batch;
        return single;
    }
    ImageResult run(const TextToImageRequest& input, ConfigView config) override {
        const auto level = parse(ITextToImage::kTask, config);
        if (mode_ == "worker" || input.prompt == "benchmark-worker") {
            ImageResult result;
            result.num_frames = 0;
            return result;
        }
        if (mode_ == "bad_empty")
            return {}; // Default one-frame empty output is not worker completion.
        return replay(image(level, static_cast<float>(input.prompt.size() % 32) / 32.0F, 0.125F),
                      input.initial_latents);
    }
    ImageResult run(const ImagesTextToImageEditRequest& input, ConfigView config) override {
        const auto level = parse(IImagesTextToImageEdit::kTask, config);
        if (input.images.empty())
            throw std::invalid_argument("source images are required");
        float first = 0;
        float last = 0;
        for (std::size_t i = 0; i < input.images.size(); ++i) {
            const auto value = first_pixel(input.images[i]);
            if (i == 0)
                first = value;
            last = value;
        }
        return replay(image(level, first, last), input.initial_latents);
    }
    ImageResult run(const MaskedImageTextToImageRequest& input, ConfigView config) override {
        const auto level = parse(IMaskedImageTextToImage::kTask, config);
        if (input.mask.values.size() != 1 ||
            (input.mask.values[0] != 0.0F && input.mask.values[0] != 1.0F))
            throw std::invalid_argument("fixture accepts one binary mask pixel");
        const auto source = first_pixel(input.source);
        return image(input.mask.values[0] == 0 ? source : level, source, 0.5F);
    }
    std::vector<ImageResult> run_batch(const BatchTextToImageRequest& input) override {
        std::vector<float> levels;
        for (const auto& item : input.items) {
            levels.push_back(parse(IBatchTextToImage::kTask, item.config));
            validate_replay(item.input.initial_latents);
            if (!item.input.initial_latents.empty() && input.items.size() != 1)
                throw std::invalid_argument("fixture replay supports exactly one batch item");
        }
        std::vector<ImageResult> results;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            if (input.items[i].input.prompt == "benchmark-worker") {
                ImageResult worker;
                worker.num_frames = 0;
                results.push_back(std::move(worker));
                continue;
            }
            float marker = 0.875F;
            for (const auto& entry : input.items[i].config)
                if (entry.name == "seed")
                    marker = static_cast<float>(std::get<std::int64_t>(entry.value)) / 255.0F;
            results.push_back(replay(image(levels[i], static_cast<float>(i) / 32.0F, marker),
                                     input.items[i].input.initial_latents));
            if (input.items[i].input.prompt == "benchmark-wide") {
                if (levels[i] != 0.5F || marker != 17.0F / 255.0F)
                    throw ConfigError("wide benchmark item requires its exact level and seed");
                results.back().width = 2;
                results.back().pixels.resize(6, 0.5F);
            }
        }
        return results; // The native-batch marker differs from single run.
    }

  private:
    void validate_replay(trtmc::Span<const float> values) const {
        if (values.empty())
            return;
        if (mode_ == "single_only")
            throw UnsupportedTask("this fixture implements generation but not latent replay");
        if (values.size() != 3)
            throw std::invalid_argument("image fixture latent layout requires three floats");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument("image fixture latents must be finite");
    }
    ImageResult replay(ImageResult result, trtmc::Span<const float> values) const {
        validate_replay(values);
        for (std::size_t i = 0; i < values.size(); ++i)
            result.pixels[i] += values[i];
        return result;
    }
    float parse(std::string_view task_id, ConfigView config) const {
        const auto field = fields_for(task_id)[0];
        double level = config_value_as<double>(*field.default_value);
        bool seen = false;
        bool seed_seen = false;
        for (const auto& entry : config) {
            if (entry.name == "seed" && task_id == IBatchTextToImage::kTask) {
                if (seed_seen || config_kind(entry.value) != ConfigKind::I64)
                    throw ConfigError("invalid or duplicate image seed");
                seed_seen = true;
                const auto seed = std::get<std::int64_t>(entry.value);
                if (seed < 0 || seed > 255)
                    throw ConfigError("fixture seed must be in [0,255]");
                continue;
            }
            if (entry.name != field.name)
                throw ConfigError("unsupported image config");
            if (seen)
                throw ConfigError("duplicate image config");
            if (config_kind(entry.value) != field.kind)
                throw ConfigError("image config type mismatch");
            seen = true;
            level = config_value_as<double>(entry.value);
        }
        if (!std::isfinite(level) || level < 0 || level > 1)
            throw ConfigError("level must be in [0,1]");
        return static_cast<float>(level);
    }
    static ImageResult image(float red, float green, float blue) {
        ImageResult result;
        result.height = result.width = result.num_frames = 1;
        result.channels = 3;
        result.pixels = {red, green, blue};
        return result;
    }
    static float first_pixel(const ImageView& image) {
        if (image.height != 1 || image.width != 1 || image.channels != 3)
            throw std::invalid_argument("fixture accepts one RGB pixel");
        if (image.format == ImageFormat::Float32)
            return static_cast<const float*>(image.data)[0];
        return static_cast<const std::uint8_t*>(image.data)[0] / 255.0F;
    }
    std::string mode_;
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "image_fixture")
        throw std::runtime_error("wrong fixture family");
    return new ImageFixture(context.reader.info().task);
}

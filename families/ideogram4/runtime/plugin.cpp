/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/ideogram4/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <stdexcept>

namespace trtmc::ideogram4 {
namespace {
constexpr std::int32_t kChannels = 128;
constexpr std::int32_t kFeatures = 53248;
const internal::ConfigField kFields[] = {
    {"seed", internal::ConfigKind::I64, std::int64_t{0},
     "Initial noise seed; explicit latents override it"},
    {"num_steps", internal::ConfigKind::I64, std::int64_t{20}, "V4_DEFAULT_20 sampler steps"},
};

std::vector<char> section(const BundleReader& bundle, const char* name) {
    const auto* entry = bundle.find_section(name);
    if (!entry || entry->length == 0)
        throw std::invalid_argument(std::string("ideogram4 missing section: ") + name);
    return bundle.read_section(name);
}

std::vector<float> forward(ITrtModule& engine, const TensorMap& inputs, const char* name,
                           std::size_t count) {
    const auto outputs = engine.forward(inputs);
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("ideogram4 unexpected engine output: ") + name);
    const auto* values = static_cast<const float*>(found->second.data);
    if (!std::all_of(values, values + count, [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("ideogram4 engine output is not finite");
    return {values, values + count};
}

template <typename T>
Tensor tensor(std::vector<T>& data, std::vector<std::int64_t> shape, DType dtype) {
    return Tensor{data.data(), std::move(shape), dtype};
}

struct Packed {
    std::int32_t length;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> segments;
    std::vector<std::int32_t> roles;
    std::vector<float> features;
    std::vector<float> latents;

    TensorMap inputs(float* time) {
        return {{"features", tensor(features, {1, length, kFeatures}, DType::kFloat32)},
                {"latents", tensor(latents, {1, length, kChannels}, DType::kFloat32)},
                {"time", Tensor{time, {1}, DType::kFloat32}},
                {"positions", tensor(positions, {1, length, 3}, DType::kInt32)},
                {"segments", tensor(segments, {1, length}, DType::kInt32)},
                {"roles", tensor(roles, {1, length}, DType::kInt32)}};
    }
};

class Pipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    Pipeline(std::unique_ptr<ITrtModule> text, std::unique_ptr<ITrtModule> positive,
             std::unique_ptr<ITrtModule> negative, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<Tokenizer> tokenizer, const nlohmann::json& config)
        : text_(std::move(text)), positive_(std::move(positive)), negative_(std::move(negative)),
          decoder_(std::move(decoder)), tokenizer_(std::move(tokenizer)),
          height_(config.at("height").get<std::int32_t>()),
          width_(config.at("width").get<std::int32_t>()),
          length_(config.at("text_length").get<std::int32_t>()),
          times_(config.at("times").get<std::vector<float>>()),
          guidance_(config.at("guidance").get<std::vector<float>>()) {
        if (height_ < 256 || width_ < 256 || height_ % 16 || width_ % 16 || length_ <= 0 ||
            length_ > 2048 || times_.size() != 21 || guidance_.size() != 20 ||
            config.at("precision") != "bf16" || config.at("sampler") != "V4_DEFAULT_20" ||
            config.at("latent_channels") != kChannels || config.at("features") != kFeatures)
            throw std::invalid_argument("ideogram4 invalid bundle dimensions or sampler");
        for (std::size_t i = 0; i < times_.size(); ++i) {
            if (!std::isfinite(times_[i]) || times_[i] < 0 || times_[i] > 1 ||
                (i && times_[i] > times_[i - 1]))
                throw std::invalid_argument("ideogram4 invalid time schedule");
        }
        if (!std::all_of(guidance_.begin(), guidance_.end(),
                         [](float x) { return std::isfinite(x) && x >= 0; }))
            throw std::invalid_argument("ideogram4 invalid guidance schedule");
    }

    const char* task() const noexcept override { return internal::ITextToImage::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToImage>(*this, kFields)};
    }

    internal::ImageResult run(const internal::TextToImageRequest& request,
                              internal::ConfigView config) override {
        internal::validate_config(kFields, config);
        if (*internal::config_get<std::int64_t>(config, kFields, "num_steps") != 20)
            throw internal::ConfigError("ideogram4 bundle implements the V4_DEFAULT_20 sampler");
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock())
            throw std::invalid_argument("ideogram4 pipeline already has an active request");
        // The foundation checkpoint consumes structured captions; prompt enhancement is an upstream
        // application.
        const auto caption = nlohmann::json::parse(request.prompt.begin(), request.prompt.end());
        if (!caption.is_object())
            throw std::invalid_argument(
                "ideogram4 prompt must be a structured JSON caption object");
        const std::string framed = "<|im_start|>user\n" + std::string(request.prompt) +
                                   "<|im_end|>\n<|im_start|>assistant\n";
        const auto encoded = tokenizer_->encode(framed);
        if (encoded.empty() || encoded.size() > static_cast<std::size_t>(length_))
            throw std::invalid_argument("ideogram4 framed prompt exceeds the bundle text profile");
        const auto offset = static_cast<std::int32_t>(length_ - encoded.size());
        std::vector<std::int32_t> ids(length_, 0), mask(length_, 0), text_positions(length_, 0);
        std::copy(encoded.begin(), encoded.end(), ids.begin() + offset);
        for (std::int32_t i = offset; i < length_; ++i) {
            mask[i] = 1;
            text_positions[i] = i - offset;
        }
        TensorMap text_inputs{{"token_ids", tensor(ids, {1, length_}, DType::kInt32)},
                              {"token_mask", tensor(mask, {1, length_}, DType::kInt32)},
                              {"positions", tensor(text_positions, {1, length_}, DType::kInt32)}};
        auto features =
            forward(*text_, text_inputs, "features", static_cast<std::size_t>(length_) * kFeatures);
        const std::int32_t gh = height_ / 16, gw = width_ / 16, count = gh * gw;
        Packed pos, neg;
        pos.length = length_ + count;
        neg.length = count;
        for (auto* pack : {&pos, &neg}) {
            pack->positions.assign(static_cast<std::size_t>(pack->length) * 3, 0);
            pack->segments.assign(pack->length, 1);
            pack->roles.assign(pack->length, 2);
            pack->features.assign(static_cast<std::size_t>(pack->length) * kFeatures, 0);
            pack->latents.assign(static_cast<std::size_t>(pack->length) * kChannels, 0);
        }
        std::copy(features.begin(), features.end(), pos.features.begin());
        for (std::int32_t i = 0; i < length_; ++i) {
            pos.segments[i] = i < offset ? -1 : 1;
            pos.roles[i] = i < offset ? 0 : 3;
            for (int axis = 0; axis < 3; ++axis)
                pos.positions[3 * i + axis] = text_positions[i];
        }
        for (std::int32_t i = 0; i < count; ++i) {
            const std::int32_t coordinates[] = {65536, 65536 + i / gw, 65536 + i % gw};
            std::copy(coordinates, coordinates + 3, pos.positions.begin() + 3 * (length_ + i));
            std::copy(coordinates, coordinates + 3, neg.positions.begin() + 3 * i);
        }
        std::vector<float> z(static_cast<std::size_t>(count) * kChannels);
        if (!request.initial_latents.empty()) {
            if (request.initial_latents.size() != z.size())
                throw std::invalid_argument(
                    "ideogram4 initial latents must have shape [1, H/16*W/16, 128]");
            std::copy(request.initial_latents.begin(), request.initial_latents.end(), z.begin());
        } else {
            const auto seed = *internal::config_get<std::int64_t>(config, kFields, "seed");
            std::mt19937 generator(static_cast<std::uint32_t>(seed));
            std::normal_distribution<float> normal;
            for (auto& value : z)
                value = normal(generator);
        }
        if (!std::all_of(z.begin(), z.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("ideogram4 initial latents must be finite");
        for (int step = 19; step >= 0; --step) {
            float time = times_[step + 1];
            std::copy(z.begin(), z.end(),
                      pos.latents.begin() + static_cast<std::size_t>(length_) * kChannels);
            neg.latents = z;
            const auto positive =
                forward(*positive_, pos.inputs(&time), "velocity", pos.latents.size());
            const auto negative = forward(*negative_, neg.inputs(&time), "velocity", z.size());
            const auto begin = static_cast<std::size_t>(length_) * kChannels;
            for (std::size_t i = 0; i < z.size(); ++i) {
                const float velocity =
                    guidance_[step] * positive[begin + i] + (1.0F - guidance_[step]) * negative[i];
                z[i] += velocity * (times_[step] - time);
            }
        }
        TensorMap decode{{"latents", tensor(z, {1, count, kChannels}, DType::kFloat32)}};
        const auto pixels =
            forward(*decoder_, decode, "pixels", static_cast<std::size_t>(height_) * width_ * 3);
        internal::ImageResult result;
        result.height = height_;
        result.width = width_;
        result.channels = 3;
        result.num_frames = 1;
        result.pixels.resize(pixels.size());
        const auto plane = static_cast<std::size_t>(height_) * width_;
        for (std::size_t i = 0; i < plane; ++i)
            for (std::size_t channel = 0; channel < 3; ++channel)
                result.pixels[3 * i + channel] =
                    std::nearbyint((std::clamp(pixels[channel * plane + i], -1.0F, 1.0F) + 1.0F) *
                                   127.5F) /
                    255.0F;
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> text_, positive_, negative_, decoder_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::int32_t height_, width_, length_;
    std::vector<float> times_, guidance_;
    std::mutex mutex_;
};
} // namespace

ITask* create(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("ideogram4 does not support --kv-cache-size");
    if (context.reader.info().task != internal::ITextToImage::kTask ||
        std::string(context.backend.name()) != "trt")
        throw std::invalid_argument(
            "ideogram4 requires text_to_image, TensorRT, and no KV cache allocation");
    const auto raw = section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(raw.begin(), raw.end());
    auto module = [&](const char* name) {
        const auto plan = section(context.reader, name);
        auto result = context.backend.create_module(plan.data(), plan.size(), {});
        if (!result || !result->ok())
            throw std::runtime_error(std::string("ideogram4 engine could not load: ") + name);
        return result;
    };
    auto text = module("text.plan");
    auto positive = module("positive.plan");
    auto negative = module("negative.plan");
    auto decoder = module("decoder.plan");
    const auto serialized = section(context.reader, "tokenizer.json");
    auto tokenizer =
        std::make_unique<Tokenizer>(std::string_view(serialized.data(), serialized.size()));
    return new Pipeline(std::move(text), std::move(positive), std::move(negative),
                        std::move(decoder), std::move(tokenizer), config);
}
} // namespace trtmc::ideogram4

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::ideogram4::create(context);
}

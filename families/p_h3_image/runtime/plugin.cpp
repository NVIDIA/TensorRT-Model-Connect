/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/p_h3_image/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

namespace trtmc::p_h3_image {
namespace {
using BF = std::vector<uint16_t>;
using F32 = std::vector<float>;
uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}
std::vector<char> section(const BundleReader& reader, const std::string& name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error("P-H3-Image bundle lacks " + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data;
    result.shape = shape;
    result.dtype = dtype;
    return result;
}
Tensor output(const TensorMap& values, const char* name, std::size_t count, DType dtype) {
    const auto found = values.find(name);
    if (found == values.end() || !found->second.data || found->second.dtype != dtype ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("Invalid P-H3-Image output ") + name);
    return found->second;
}
BF copy_bf(const TensorMap& values, const char* name, std::size_t count) {
    const auto view = output(values, name, count, DType::kBFloat16);
    const auto* data = static_cast<const uint16_t*>(view.data);
    return BF(data, data + count);
}
F32 copy_f32(const TensorMap& values, const char* name, std::size_t count) {
    const auto view = output(values, name, count, DType::kFloat32);
    const auto* data = static_cast<const float*>(view.data);
    return F32(data, data + count);
}
struct Engines {
    std::unique_ptr<ITrtModule> embedding, text_norm, connector, prepare, time, head, vae_input,
        vae_head;
    std::vector<std::unique_ptr<ITrtModule>> text, denoiser, vae;
};
struct Tiles {
    std::vector<int> starts, overlaps;
};
Tiles tiles(int size) {
    int count = (size + 255) / 256;
    while (256 * count - 64 * (count - 1) < size)
        ++count;
    Tiles result;
    result.overlaps.assign(count - 1, 64);
    const int remaining = (256 * count - 64 * (count - 1) - size) / 16;
    for (int i = 0; i < remaining; ++i)
        result.overlaps[i % (count - 1)] += 16;
    result.starts.push_back(0);
    for (int i = 0; i + 1 < count; ++i)
        result.starts.push_back(result.starts.back() + 256 - result.overlaps[i]);
    return result;
}

class Pipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    Pipeline(Engines engines, std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg, F32 sink,
             F32 means, F32 scales)
        : engines_(std::move(engines)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
          sink_(std::move(sink)), means_(std::move(means)), scales_(std::move(scales)),
          fields_{{"seed", internal::ConfigKind::I64, internal::ConfigValue{int64_t{42}},
                   "Initial Gaussian noise seed"}} {}
    const char* task() const noexcept override { return "text_to_image"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToImage>(*this, {fields_.data(), fields_.size()})};
    }
    internal::ImageResult run(const internal::TextToImageRequest& request,
                              internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto seed = *internal::config_get<int64_t>(config, fields, "seed");
        if (seed < 0 || seed > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Invalid image noise seed");
        auto ids = tokenizer_->encode(std::string(request.prompt));
        const int length = static_cast<int>(ids.size());
        if (!length || length > cfg_.at("max_prompt_tokens").get<int>())
            throw std::invalid_argument(
                "P-H3-Image requires 1–256 prompt tokens within the built profile");
        const int vocab = cfg_.at("text").at("vocab_size");
        for (const auto id : ids)
            if (id < 0 || id >= vocab)
                throw std::invalid_argument("Invalid P-H3-Image token");
        const int height = cfg_.at("height"), width = cfg_.at("width");
        const int lh = height / 16, lw = width / 16, rows = lh * lw / 4,
                  packed_rows = length + 4 + rows;
        const auto image_count = static_cast<std::size_t>(lh) * lw * 24;
        F32 noise(image_count + 128);
        if (request.initial_latents.empty()) {
            std::mt19937 random(static_cast<uint32_t>(seed));
            std::normal_distribution<float> normal(0, 1);
            for (auto& value : noise)
                value = normal(random);
        } else {
            if (request.initial_latents.size() != noise.size() || !request.initial_latents.data())
                throw std::invalid_argument(
                    "P-H3-Image replay requires NCHW image noise followed by 4x32 audio noise");
            std::copy_n(request.initial_latents.data(), noise.size(), noise.data());
        }
        for (const auto value : noise)
            if (!std::isfinite(value))
                throw std::invalid_argument("Nonfinite image noise");
        auto context = encode(ids);
        F32 video(image_count), audio(noise.begin() + image_count, noise.end());
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x)
                for (int c = 0; c < 24; ++c)
                    video[((y / 2) * (lw / 2) + x / 2) * 96 + c * 4 + (y % 2) * 2 + x % 2] =
                        noise[(c * lh + y) * lw + x];
        BF cos(packed_rows * 96), sin(packed_rows * 96);
        std::vector<int32_t> tags(packed_rows, 0);
        const double root_area = std::sqrt(static_cast<double>(lh) * lw);
        const double hy = (1 - lh / root_area) / 2, wx = (1 - lw / root_area) / 2;
        for (int row = 0; row < packed_rows; ++row) {
            std::array<double, 3> position{};
            if (row < length) {
                position[0] = row;
                tags[row] = 1;
            } else if (row < length + 4) {
                const int index = row - length;
                position[0] = length + index % 2;
                position[2] =
                    32 * (index < 2 ? wx : wx + (lw / 2 - 1) * (lw / root_area) / (lw / 2));
                tags[row] = 2;
            } else {
                const int index = row - length - 4;
                position[0] = length;
                position[1] = 32 * (hy + (index / (lw / 2)) * (lh / root_area) / (lh / 2));
                position[2] = 32 * (wx + (index % (lw / 2)) * (lw / root_area) / (lw / 2));
            }
            for (int axis = 0; axis < 3; ++axis)
                for (int f = 0; f < 16; ++f) {
                    const float frequency = 1.0F / std::pow(10000.0F, static_cast<float>(f) / 16);
                    const float angle = static_cast<float>(position[axis]) * frequency;
                    const int channel = axis * 16 + f;
                    cos[row * 96 + channel] = cos[row * 96 + channel + 48] = bf16(std::cos(angle));
                    sin[row * 96 + channel] = sin[row * 96 + channel + 48] = bf16(std::sin(angle));
                }
        }
        const int hidden_width = cfg_.at("denoiser").at("hidden_size"),
                  time_width = cfg_.at("denoiser").at("time_embed_dim");
        const auto sigmas = cfg_.at("sigmas").get<F32>();
        for (std::size_t step = 0; step + 1 < sigmas.size(); ++step) {
            auto hidden =
                copy_bf(engines_.prepare->forward(
                            {{"context", tensor(context.data(), {length, 5120}, DType::kBFloat16)},
                             {"video", tensor(video.data(), {rows, 96}, DType::kFloat32)},
                             {"audio", tensor(audio.data(), {4, 32}, DType::kFloat32)}}),
                        "result", packed_rows * hidden_width);
            float timestep = 1 - sigmas[step];
            auto time = copy_bf(
                engines_.time->forward({{"timestep", tensor(&timestep, {1, 1}, DType::kFloat32)}}),
                "result", time_width);
            for (auto& layer : engines_.denoiser) {
                hidden = copy_bf(
                    layer->forward(
                        {{"hidden",
                          tensor(hidden.data(), {packed_rows, hidden_width}, DType::kBFloat16)},
                         {"time", tensor(time.data(), {1, time_width}, DType::kBFloat16)},
                         {"tags", tensor(tags.data(), {packed_rows}, DType::kInt32)},
                         {"cos", tensor(cos.data(), {1, packed_rows, 1, 96}, DType::kBFloat16)},
                         {"sin", tensor(sin.data(), {1, packed_rows, 1, 96}, DType::kBFloat16)}}),
                    "result", packed_rows * hidden_width);
            }
            const auto prediction = engines_.head->forward(
                {{"hidden", tensor(hidden.data(), {packed_rows, hidden_width}, DType::kBFloat16)},
                 {"time", tensor(time.data(), {1, time_width}, DType::kBFloat16)}});
            const auto pv = output(prediction, "video", packed_rows * 96, DType::kFloat32);
            const auto pa = output(prediction, "audio", packed_rows * 32, DType::kFloat32);
            const auto* vv = static_cast<const float*>(pv.data) + (length + 4) * 96;
            const auto* av = static_cast<const float*>(pa.data) + length * 32;
            const float ratio = sigmas[step + 1] / sigmas[step];
            auto update = [&](F32& values, const float* velocity) {
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const float clean = values[i] + (1 - timestep) * velocity[i];
                    values[i] = ratio * values[i] + (1 - ratio) * clean;
                    if (!std::isfinite(values[i]))
                        throw std::runtime_error("Nonfinite P-H3-Image latent");
                }
            };
            update(video, vv);
            update(audio, av);
        }
        F32 latents(image_count);
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x)
                for (int c = 0; c < 24; ++c)
                    latents[(y * lw + x) * 24 + c] =
                        video[((y / 2) * (lw / 2) + x / 2) * 96 + c * 4 + (y % 2) * 2 + x % 2] *
                            scales_[c] +
                        means_[c];
        return decode(latents, height, width);
    }

  private:
    BF encode(std::vector<int32_t>& ids) {
        const int length = static_cast<int>(ids.size()), width = cfg_.at("text").at("hidden_size");
        const int dim = cfg_.at("text").at("head_dim");
        const float theta = cfg_.at("text").at("rope_parameters").at("rope_theta");
        auto hidden = copy_bf(engines_.embedding->forward(
                                  {{"input_ids", tensor(ids.data(), {length}, DType::kInt32)}}),
                              "result", length * width);
        BF cos(length * dim), sin(length * dim), mask(length * length, 0);
        for (int p = 0; p < length; ++p) {
            for (int i = 0; i < dim / 2; ++i) {
                const float angle = p * (1.0F / std::pow(theta, 2.0F * i / dim));
                cos[p * dim + i] = cos[p * dim + i + dim / 2] = bf16(std::cos(angle));
                sin[p * dim + i] = sin[p * dim + i + dim / 2] = bf16(std::sin(angle));
            }
            for (int j = p + 1; j < length; ++j)
                mask[p * length + j] = 0xff7fU;
        }
        const auto selected =
            cfg_.at("connector").at("hidden_state_indices").get<std::vector<int>>();
        BF captures(static_cast<std::size_t>(length) * width * selected.size());
        for (std::size_t layer = 0; layer < engines_.text.size(); ++layer) {
            hidden = copy_bf(
                engines_.text[layer]->forward(
                    {{"hidden", tensor(hidden.data(), {length, width}, DType::kBFloat16)},
                     {"cos", tensor(cos.data(), {1, length, 1, dim}, DType::kBFloat16)},
                     {"sin", tensor(sin.data(), {1, length, 1, dim}, DType::kBFloat16)},
                     {"mask", tensor(mask.data(), {1, 1, length, length}, DType::kBFloat16)}}),
                "result", length * width);
            if (layer + 1 == engines_.text.size())
                hidden = copy_bf(
                    engines_.text_norm->forward(
                        {{"hidden", tensor(hidden.data(), {length, width}, DType::kBFloat16)}}),
                    "result", length * width);
            const auto found =
                std::find(selected.begin(), selected.end(), static_cast<int>(layer + 1));
            if (found != selected.end()) {
                const auto offset = static_cast<std::size_t>(found - selected.begin()) * width;
                for (int row = 0; row < length; ++row)
                    std::copy_n(hidden.data() + row * width, width,
                                captures.data() + row * width * selected.size() + offset);
            }
        }
        auto result =
            copy_bf(engines_.connector->forward(
                        {{"hidden", tensor(captures.data(),
                                           {length, static_cast<int64_t>(width * selected.size())},
                                           DType::kBFloat16)}}),
                    "result", length * 5120);
        for (std::size_t i = 0; i < sink_.size(); ++i)
            result[i] = bf16(sink_[i]);
        return result;
    }
    F32 decode_tile(F32& latents) {
        constexpr int rows = 261, width = 2048;
        auto hidden =
            copy_bf(engines_.vae_input->forward(
                        {{"latents", tensor(latents.data(), {256, 24}, DType::kFloat32)}}),
                    "result", rows * width);
        for (auto& layer : engines_.vae)
            hidden =
                copy_bf(layer->forward(
                            {{"hidden", tensor(hidden.data(), {rows, width}, DType::kBFloat16)}}),
                        "result", rows * width);
        const auto patches =
            copy_f32(engines_.vae_head->forward(
                         {{"hidden", tensor(hidden.data(), {rows, width}, DType::kBFloat16)}}),
                     "result", rows * 768);
        F32 pixels(256 * 256 * 3);
        for (int y = 0; y < 256; ++y)
            for (int x = 0; x < 256; ++x)
                for (int c = 0; c < 3; ++c)
                    pixels[(y * 256 + x) * 3 + c] =
                        patches[((y / 16) * 16 + x / 16) * 768 + c * 256 + (y % 16) * 16 + x % 16];
        return pixels;
    }
    internal::ImageResult decode(const F32& latents, int height, int width) {
        const auto ys = tiles(height), xs = tiles(width);
        const auto ny = ys.starts.size(), nx = xs.starts.size();
        std::vector<std::vector<F32>> decoded(ny, std::vector<F32>(nx));
        F32 tile_latents(256 * 24);
        for (std::size_t i = 0; i < ny; ++i)
            for (std::size_t j = 0; j < nx; ++j) {
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x)
                        std::copy_n(latents.data() + (((ys.starts[i] / 16 + y) * (width / 16) +
                                                       xs.starts[j] / 16 + x) *
                                                      24),
                                    24, tile_latents.data() + (y * 16 + x) * 24);
                decoded[i][j] = decode_tile(tile_latents);
            }
        internal::ImageResult result;
        result.height = height;
        result.width = width;
        result.channels = 3;
        result.num_frames = 1;
        result.pixels.resize(static_cast<std::size_t>(height) * width * 3);
        for (std::size_t i = 0; i < ny; ++i)
            for (std::size_t j = 0; j < nx; ++j) {
                auto tile = decoded[i][j];
                if (i) {
                    const int overlap = ys.overlaps[i - 1];
                    for (int y = 0; y < overlap; ++y)
                        for (int x = 0; x < 256; ++x)
                            for (int c = 0; c < 3; ++c) {
                                const float weight = static_cast<float>(y) / overlap;
                                tile[(y * 256 + x) * 3 + c] =
                                    decoded[i - 1][j][((256 - overlap + y) * 256 + x) * 3 + c] *
                                        (1 - weight) +
                                    tile[(y * 256 + x) * 3 + c] * weight;
                            }
                }
                if (j) {
                    const int overlap = xs.overlaps[j - 1];
                    for (int y = 0; y < 256; ++y)
                        for (int x = 0; x < overlap; ++x)
                            for (int c = 0; c < 3; ++c) {
                                const float weight = static_cast<float>(x) / overlap;
                                tile[(y * 256 + x) * 3 + c] =
                                    decoded[i][j - 1][(y * 256 + 256 - overlap + x) * 3 + c] *
                                        (1 - weight) +
                                    tile[(y * 256 + x) * 3 + c] * weight;
                            }
                }
                const int keep_h = i + 1 < ny ? 256 - ys.overlaps[i] : 256;
                const int keep_w = j + 1 < nx ? 256 - xs.overlaps[j] : 256;
                constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F},
                    scale{0.229F, 0.224F, 0.225F};
                for (int y = 0; y < keep_h; ++y)
                    for (int x = 0; x < keep_w; ++x)
                        for (int c = 0; c < 3; ++c) {
                            const float value = tile[(y * 256 + x) * 3 + c] * scale[c] + mean[c];
                            if (!std::isfinite(value))
                                throw std::runtime_error("Nonfinite P-H3-Image pixel");
                            result.pixels[((ys.starts[i] + y) * width + xs.starts[j] + x) * 3 + c] =
                                std::clamp(value, 0.0F, 1.0F);
                        }
            }
        return result;
    }
    Engines engines_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    F32 sink_, means_, scales_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::p_h3_image

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("p_h3_image does not support --kv-cache-size");
    using namespace trtmc;
    using namespace p_h3_image;
    if (context.reader.info().task != "text_to_image")
        throw std::invalid_argument("P-H3-Image requires text_to_image without external state");
    const auto raw = section(context.reader, "runtime.json");
    auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int height = cfg.at("height"), width = cfg.at("width"),
              length = cfg.at("max_prompt_tokens");
    if (height < 256 || width < 256 || height > 9216 || width > 9216 || height % 256 ||
        width % 256 || static_cast<int64_t>(height) * width < 2 * 1024 * 1024 ||
        static_cast<int64_t>(height) * width > 9 * 1024 * 1024 || length < 1 || length > 256 ||
        cfg.at("text").at("hidden_size") != 2560 || cfg.at("text").at("num_hidden_layers") != 36 ||
        cfg.at("denoiser").at("hidden_size") != 5376 ||
        cfg.at("denoiser").at("time_embed_dim") != 2688 ||
        cfg.at("denoiser").at("num_layers") != 50 || cfg.at("vae").at("decoder_num_layers") != 36 ||
        cfg.at("connector").at("hidden_state_indices") !=
            nlohmann::json::array({13, 19, 25, 31, 36}))
        throw std::runtime_error("Invalid P-H3-Image runtime profile");
    const auto sigmas = cfg.at("sigmas").get<F32>();
    if (sigmas.size() != 9 || sigmas.front() != 1 || sigmas.back() != 0)
        throw std::runtime_error("P-H3-Image requires the eight-step checkpoint schedule");
    for (std::size_t i = 1; i < sigmas.size(); ++i)
        if (!std::isfinite(sigmas[i]) || sigmas[i] < 0 || sigmas[i] >= sigmas[i - 1])
            throw std::runtime_error("Invalid image schedule");
    auto means = cfg.at("vae").at("latents_mean").get<F32>();
    auto scales = cfg.at("vae").at("latents_std").get<F32>();
    for (const auto* values : {&means, &scales}) {
        if (values->size() != 24 ||
            !std::all_of(values->begin(), values->end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("P-H3-Image VAE statistics require 24 finite values");
    }
    auto load = [&](const std::string& name) {
        const auto plan = section(context.reader, name + ".plan");
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok())
            throw std::runtime_error("Failed to load P-H3-Image engine " + name);
        return engine;
    };
    Engines engines;
    engines.embedding = load("text.embedding");
    engines.text_norm = load("text.norm");
    engines.connector = load("text.connector");
    engines.prepare = load("denoise.prepare");
    engines.time = load("denoise.time");
    engines.head = load("denoise.head");
    engines.vae_input = load("vae.input");
    engines.vae_head = load("vae.head");
    for (int i = 0; i < 36; ++i)
        engines.text.push_back(load("text.layer." + std::to_string(i)));
    for (int i = 0; i < 50; ++i)
        engines.denoiser.push_back(load("denoise.layer." + std::to_string(i)));
    for (int i = 0; i < 36; ++i)
        engines.vae.push_back(load("vae.layer." + std::to_string(i)));
    const auto sink_bytes = section(context.reader, "text.sink.f32");
    if (sink_bytes.size() != 5120 * sizeof(float))
        throw std::runtime_error("Invalid text sink");
    F32 sink(5120);
    std::memcpy(sink.data(), sink_bytes.data(), sink_bytes.size());
    const auto tokenizer = section(context.reader, "tokenizer.json");
    return new Pipeline(std::move(engines),
                        CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false),
                        std::move(cfg), std::move(sink), std::move(means), std::move(scales));
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pika_h3_flex/runtime/engine.h"
#include "families/pika_h3_flex/runtime/random.h"
#include "families/pika_h3_flex/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace trtmc::pika_h3_flex {
namespace {
class Pipeline final : public internal::IModel, public internal::ITextToAudioVideo {
  public:
    Pipeline(Loader loader, std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : loader_(std::move(loader)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)), fields_{
            {"seed", internal::ConfigKind::I64, internal::ConfigValue{int64_t{42}}, "CPU-generator Gaussian stream seed"}} {}
    const char* task() const noexcept override { return "text_to_audio_video"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToAudioVideo>(*this, {fields_.data(), fields_.size()})};
    }
    internal::AudioVideoResult run(const internal::TextToAudioVideoRequest& request, internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto seed = *internal::config_get<int64_t>(config, fields, "seed");
        if (seed < 0 || seed > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("Invalid Pika noise seed");
        auto ids = tokenizer_->encode(std::string(request.prompt));
        if (ids.empty() || ids.size() > cfg_.at("max_prompt_tokens").get<std::size_t>())
            throw std::invalid_argument("Pika prompt is empty or exceeds the built context profile");
        for (const auto id : ids) if (id < 0 || id >= cfg_.at("text").at("vocab_size").get<int>())
            throw std::invalid_argument("Invalid Pika token ID");
        auto context = encode(ids);
        Random random(static_cast<uint32_t>(seed));
        auto initial = random.normal(24 * 37 * 30 * 52);
        auto audio = random.normal(414 * 32);
        F32 video(initial.size());
        for (int f = 0; f < 37; ++f) for (int y = 0; y < 30; ++y) for (int x = 0; x < 52; ++x) for (int c = 0; c < 24; ++c)
            video[((f * 15 + y / 2) * 26 + x / 2) * 96 + c * 4 + (y % 2) * 2 + x % 2] = initial[((c * 37 + f) * 30 + y) * 52 + x];
        initial.clear(); initial.shrink_to_fit();
        denoise(context, static_cast<int>(ids.size()), video, audio, random);
        const auto means = cfg_.at("vae").at("latents_mean").get<F32>();
        const auto scales = cfg_.at("vae").at("latents_std").get<F32>();
        F32 latents(video.size());
        for (int f = 0; f < 37; ++f) for (int y = 0; y < 30; ++y) for (int x = 0; x < 52; ++x) for (int c = 0; c < 24; ++c)
            latents[((f * 30 + y) * 52 + x) * 24 + c] = video[((f * 15 + y / 2) * 26 + x / 2) * 96 + c * 4 + (y % 2) * 2 + x % 2] * scales[c] + means[c];
        internal::AudioVideoResult result;
        result.video = decode_video(loader_, cfg_, latents);
        result.audio = decode_audio(audio);
        result.audio_start_seconds = 0.0;
        return result;
    }
  private:
    BF encode(std::vector<int32_t>& ids) const {
        const int length = static_cast<int>(ids.size());
        auto embedding = loader_.load("text.embedding");
        auto hidden = copy_bf(embedding->forward({{"input_ids", tensor(ids.data(), {length}, DType::kInt32)}}), "result", length * 5120);
        embedding.reset();
        BF cos(length * 128), sin(length * 128), mask(length * length, 0);
        for (int p = 0; p < length; ++p) {
            for (int i = 0; i < 64; ++i) {
                const float angle = p * (1.0F / std::pow(5000000.0F, 2.0F * i / 128));
                cos[p * 128 + i] = cos[p * 128 + i + 64] = bf16(std::cos(angle));
                sin[p * 128 + i] = sin[p * 128 + i + 64] = bf16(std::sin(angle));
            }
            for (int j = p + 1; j < length; ++j) mask[p * length + j] = 0xff7fU;
        }
        // The source conditions on hidden_states[50], before the final norm.
        // Each text block is consumed once, so release it before loading the next.
        for (int i = 0; i < 50; ++i) {
            auto layer = loader_.load("text.layer." + std::to_string(i));
            hidden = copy_bf(layer->forward({
                {"hidden", tensor(hidden.data(), {length, 5120}, DType::kBFloat16)},
                {"cos", tensor(cos.data(), {1, length, 1, 128}, DType::kBFloat16)},
                {"sin", tensor(sin.data(), {1, length, 1, 128}, DType::kBFloat16)},
                {"mask", tensor(mask.data(), {1, 1, length, length}, DType::kBFloat16)}}), "result", length * 5120);
        }
        return hidden;
    }
    void denoise(BF& context, int text_rows, F32& video, F32& audio, Random& random) const {
        constexpr int video_rows = 14430, audio_rows = 414, width = 5376;
        const int total = text_rows + audio_rows + video_rows;
        auto prepare = loader_.load("denoise.prepare"), time_engine = loader_.load("denoise.time"), head = loader_.load("denoise.head");
        std::vector<std::unique_ptr<ITrtModule>> layers;
        for (int i = 0; i < 50; ++i) layers.push_back(loader_.load("denoise.layer." + std::to_string(i)));
        BF cos(total * 96), sin(total * 96);
        const double root = std::sqrt(30.0 * 52), hy = (1 - 30.0 / root) / 2, wx = (1 - 52.0 / root) / 2;
        constexpr std::array<int, 5> spans{1, 4, 4, 4, 4};
        std::array<double, 37> times{};
        double elapsed = 0;
        for (int f = 0; f < 37; ++f) { times[f] = text_rows + elapsed; elapsed += (5.0 / 3) * spans[f % 5]; }
        for (int row = 0; row < total; ++row) {
            std::array<double, 3> position{};
            if (row < text_rows) position[0] = row;
            else if (row < text_rows + audio_rows) {
                const int i = row - text_rows;
                position[0] = text_rows + i % 207;
                position[2] = 32 * (i < 207 ? wx : wx + 25 * (52.0 / root) / 26);
            } else {
                const int i = row - text_rows - audio_rows;
                position[0] = times[i / 390];
                position[1] = 32 * (hy + (i % 390 / 26) * (30.0 / root) / 15);
                position[2] = 32 * (wx + (i % 26) * (52.0 / root) / 26);
            }
            for (int axis = 0; axis < 3; ++axis) for (int f = 0; f < 16; ++f) {
                const float angle = static_cast<float>(position[axis]) * (1.0F / std::pow(10000.0F, static_cast<float>(f) / 16));
                const int c = axis * 16 + f;
                cos[row * 96 + c] = cos[row * 96 + c + 48] = bf16(std::cos(angle));
                sin[row * 96 + c] = sin[row * 96 + c + 48] = bf16(std::sin(angle));
            }
        }
        const auto knots = cfg_.at("knots").get<std::vector<int>>();
        std::array<float, 7> vs{}, as{};
        for (int i = 0; i < 7; ++i) {
            const float value = static_cast<float>(knots[i]) / 1000;
            vs[i] = (12 * value) / (1 + 11 * value);
            as[i] = (3 * value) / (1 + 2 * value);
        }
        std::vector<int32_t> time_indices(total), adaln_indices(total);
        for (int step = 0; step < 6; ++step) {
            F32 timesteps{1 - vs[step]};
            if (1 - as[step] != timesteps[0]) timesteps.push_back(1 - as[step]);
            const int time_rows = static_cast<int>(timesteps.size());
            for (int row = 0; row < total; ++row) {
                const bool is_audio = row >= text_rows && row < text_rows + audio_rows;
                time_indices[row] = is_audio ? time_rows - 1 : 0;
                adaln_indices[row] = time_indices[row] * 3 + (row < text_rows ? 1 : is_audio ? 2 : 0);
            }
            auto time = copy_bf(time_engine->forward({{"timestep", tensor(timesteps.data(), {time_rows, 1}, DType::kFloat32)}}), "result", time_rows * 2688);
            auto hidden = copy_bf(prepare->forward({
                {"context", tensor(context.data(), {text_rows, 5120}, DType::kBFloat16)},
                {"video", tensor(video.data(), {video_rows, 96}, DType::kFloat32)},
                {"audio", tensor(audio.data(), {audio_rows, 32}, DType::kFloat32)}}), "result", total * width);
            for (auto& layer : layers)
                hidden = copy_bf(layer->forward({
                    {"hidden", tensor(hidden.data(), {total, width}, DType::kBFloat16)},
                    {"time", tensor(time.data(), {time_rows, 2688}, DType::kBFloat16)},
                    {"adaln_indices", tensor(adaln_indices.data(), {total}, DType::kInt32)},
                    {"cos", tensor(cos.data(), {1, total, 1, 96}, DType::kBFloat16)},
                    {"sin", tensor(sin.data(), {1, total, 1, 96}, DType::kBFloat16)}}), "result", total * width);
            const auto predicted = head->forward({
                {"hidden", tensor(hidden.data(), {total, width}, DType::kBFloat16)},
                {"time", tensor(time.data(), {time_rows, 2688}, DType::kBFloat16)},
                {"time_indices", tensor(time_indices.data(), {total}, DType::kInt32)}});
            const auto vview = output(predicted, "video", total * 96, DType::kFloat32);
            const auto aview = output(predicted, "audio", total * 32, DType::kFloat32);
            auto update = [&](F32& sample, const float* velocity, float sigma, float next) {
                const auto noise = random.normal(sample.size());
                for (std::size_t i = 0; i < sample.size(); ++i) {
                    const float clean = sample[i] + sigma * velocity[i];
                    sample[i] = (1 - next) * clean + next * noise[i];
                    if (!std::isfinite(sample[i])) throw std::runtime_error("Pika produced a nonfinite latent");
                }
            };
            update(video, static_cast<const float*>(vview.data) + (text_rows + audio_rows) * 96, vs[step], vs[step + 1]);
            update(audio, static_cast<const float*>(aview.data) + text_rows * 32, as[step], as[step + 1]);
        }
    }
    internal::AudioResult decode_audio(const F32& rows) const {
        const auto means = cfg_.at("audio_vae").at("latents_mean").get<F32>();
        const auto scales = cfg_.at("audio_vae").at("latents_std").get<F32>();
        F32 latents(2 * 32 * 207);
        for (int ch = 0; ch < 2; ++ch) for (int c = 0; c < 32; ++c) for (int t = 0; t < 207; ++t)
            latents[(ch * 32 + c) * 207 + t] = rows[(ch * 207 + t) * 32 + c] * scales[c] + means[c];
        auto input = loader_.load("audio.input");
        auto hidden = copy_f32(input->forward({{"latents", tensor(latents.data(), {2, 32, 1, 207}, DType::kFloat32)}}), "result", 2 * 1024 * 207);
        input.reset();
        constexpr std::array<int, 7> rates{5, 5, 2, 2, 2, 2, 2};
        int frames = 207, channels = 1024;
        for (int stage = 0; stage < 7; ++stage) {
            auto layer = loader_.load("audio.stage." + std::to_string(stage));
            auto result = copy_f32(layer->forward({{"hidden", tensor(hidden.data(), {2, channels, 1, frames}, DType::kFloat32)}}), "result", 2 * (channels / 2) * frames * rates[stage]);
            hidden = std::move(result); frames *= rates[stage]; channels /= 2;
        }
        auto head = loader_.load("audio.head");
        const auto wave = copy_f32(head->forward({{"hidden", tensor(hidden.data(), {2, 8, 1, frames}, DType::kFloat32)}}), "result", 2 * frames);
        constexpr int samples = 165333; // round(124 / 24 * 32000)
        internal::AudioResult result;
        result.sample_rate = 32000; result.channels = 2; result.samples.resize(samples * 2);
        for (int t = 0; t < samples; ++t) for (int ch = 0; ch < 2; ++ch) {
            const float value = wave[ch * frames + t];
            if (!std::isfinite(value) || value < -1 || value > 1) throw std::runtime_error("Invalid Pika audio output");
            result.samples[t * 2 + ch] = value;
        }
        return result;
    }
    Loader loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::pika_h3_flex

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    using namespace pika_h3_flex;
    if (context.kv_cache_size_bytes || context.reader.info().task != "text_to_audio_video")
        throw std::invalid_argument("Pika H3 Flex requires text_to_audio_video without external state");
    Loader loader(context);
    const auto raw = loader.section("runtime.json");
    auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    if (cfg.at("height") != 480 || cfg.at("width") != 832 || cfg.at("frames") != 124 ||
        cfg.at("latent_frames") != 37 || cfg.at("audio_latents") != 207 ||
        cfg.at("max_prompt_tokens").get<int>() < 1 || cfg.at("max_prompt_tokens").get<int>() > 128 ||
        cfg.at("text").at("hidden_size") != 5120 || cfg.at("text").at("head_dim") != 128 ||
        cfg.at("denoiser").at("hidden_size") != 5376 || cfg.at("denoiser").at("time_embed_dim") != 2688 ||
        cfg.at("denoiser").at("num_layers") != 50 || cfg.at("vae").at("decoder_num_layers") != 36 ||
        cfg.at("audio_vae").at("sampling_rate") != 32000 ||
        cfg.at("knots") != nlohmann::json::array({1000, 750, 500, 250, 100, 40, 0}))
        throw std::runtime_error("Invalid Pika H3 Flex runtime profile");
    const auto tokenizer = loader.section("tokenizer.json");
    return new Pipeline(std::move(loader), CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false), std::move(cfg));
}

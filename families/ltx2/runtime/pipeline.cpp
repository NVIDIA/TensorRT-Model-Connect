/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx2/runtime/pipeline.h"

#include "families/ltx2/runtime/portable_normal.h"
#include "families/ltx2/runtime/runtime_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16U;
    const uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t bits = sign;
    if (exp == 31U) {
        bits |= 0x7F800000U | (mant << 13U);
    } else if (exp != 0U) {
        bits |= ((exp - 15U + 127U) << 23U) | (mant << 13U);
    } else if (mant != 0U) {
        int32_t e = -1;
        do {
            ++e;
            mant <<= 1U;
        } while ((mant & 0x400U) == 0U);
        bits |= (static_cast<uint32_t>(127 - 15 - e) << 23U) | ((mant & 0x3FFU) << 13U);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float bf16_to_float(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

const Tensor& require_output(const TensorMap& outputs, const std::string& name, DType dtype,
                             std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end())
        throw std::runtime_error("LTX-2.5 engine output is missing: " + name);
    if (it->second.dtype != dtype || it->second.numel() != count || it->second.data == nullptr)
        throw std::runtime_error("LTX-2.5 engine output does not match its contract: " + name);
    return it->second;
}

std::vector<float> float_output(const TensorMap& outputs, const std::string& name,
                                std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end())
        throw std::runtime_error("LTX-2.5 engine output is missing: " + name);
    const auto& tensor = it->second;
    if (tensor.data == nullptr || tensor.numel() != count)
        throw std::runtime_error("LTX-2.5 engine output does not match its contract: " + name);
    std::vector<float> out(count);
    if (tensor.dtype == DType::kFloat32) {
        std::memcpy(out.data(), tensor.data, count * sizeof(float));
    } else if (tensor.dtype == DType::kFloat16) {
        const auto* src = static_cast<const uint16_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = half_to_float(src[i]);
    } else if (tensor.dtype == DType::kBFloat16) {
        const auto* src = static_cast<const uint16_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = bf16_to_float(src[i]);
    } else {
        throw std::runtime_error("LTX-2.5 engine output has an unsupported dtype: " + name);
    }
    return out;
}

std::vector<uint16_t> bf16_output(const TensorMap& outputs, const std::string& name,
                                  std::size_t count) {
    const auto& tensor = require_output(outputs, name, DType::kBFloat16, count);
    const auto* src = static_cast<const uint16_t*>(tensor.data);
    return {src, src + count};
}

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

// Optional family diagnostics (all off by default):
//   TRTMC_LTX2_INITIAL_LATENTS  raw fp32 file: packed video [S, C] then audio [Sa, Ca] noise
//                               (replaces the seeded noise, e.g. a reference pipeline's draw)
//   TRTMC_LTX2_DUMP_LATENTS     raw fp32 file written with the final video then audio latents
std::vector<float> read_f32_file(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error(std::string("cannot read LTX-2.5 initial latents: ") + path);
    const auto size = static_cast<std::size_t>(input.tellg());
    if (size % sizeof(float) != 0)
        throw std::runtime_error("LTX-2.5 initial latents file is not fp32");
    std::vector<float> values(size / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
    return values;
}

void maybe_dump(const std::vector<float>& video, const std::vector<float>& audio) {
    const char* path = std::getenv("TRTMC_LTX2_DUMP_LATENTS");
    if (path == nullptr || *path == '\0')
        return;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(video.data()),
                 static_cast<std::streamsize>(video.size() * 4));
    output.write(reinterpret_cast<const char*>(audio.data()),
                 static_cast<std::streamsize>(audio.size() * 4));
    std::cerr << "[ltx2] wrote final latents (" << video.size() << " + " << audio.size()
              << " fp32) to " << path << "\n";
}

const std::array<internal::ConfigField, 1>& config_fields() {
    static const std::array<internal::ConfigField, 1> fields{{
        {"seed", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Seed of the initial video and audio noise (portable std::mt19937 + normal draws)."},
    }};
    return fields;
}

internal::AudioVideoResult worker_completion(const LTX2Options& options) {
    internal::AudioVideoResult result;
    result.video.frames.num_frames = 0;
    result.video.frames.height = 0;
    result.video.frames.width = 0;
    result.video.frames.channels = 3;
    result.audio.sample_rate = static_cast<uint32_t>(options.audio_sample_rate);
    result.audio.channels = static_cast<uint32_t>(options.audio_channels);
    return result;
}

} // namespace

LTX2Options parse_ltx2_options(const std::string& runtime_json) {
    const auto doc = nlohmann::json::parse(runtime_json);
    LTX2Options o;
    o.video_frames = doc.at("video_frames").get<int32_t>();
    o.video_height = doc.at("video_height").get<int32_t>();
    o.video_width = doc.at("video_width").get<int32_t>();
    o.latent_frames = doc.at("latent_frames").get<int32_t>();
    o.latent_height = doc.at("latent_height").get<int32_t>();
    o.latent_width = doc.at("latent_width").get<int32_t>();
    o.latent_channels = doc.at("latent_channels").get<int32_t>();
    o.audio_frames = doc.at("audio_frames").get<int32_t>();
    o.audio_latent_channels = doc.at("audio_latent_channels").get<int32_t>();
    o.text_seq_len = doc.at("text_seq_len").get<int32_t>();
    o.frame_rate = doc.at("frame_rate").get<float>();
    o.pad_token_id = doc.at("pad_token_id").get<int32_t>();
    o.sigmas = doc.at("sigmas").get<std::vector<float>>();
    o.audio_sample_rate = doc.at("audio_sample_rate").get<int32_t>();
    o.audio_channels = doc.at("audio_channels").get<int32_t>();
    if (o.sigmas.size() < 2 || o.sigmas.back() != 0.0F)
        throw std::runtime_error("LTX-2.5 runtime.json sigmas must end with the terminal 0");
    if (doc.at("dit_batch").get<int32_t>() != 1)
        throw std::runtime_error("LTX-2.5 runtime runs the distilled (batch 1) denoiser");
    if (o.video_tokens() <= 0 || o.audio_frames <= 0 || o.text_seq_len <= 0)
        throw std::runtime_error("LTX-2.5 runtime.json has invalid shapes");
    return o;
}

LTX2Pipeline::LTX2Pipeline(std::unique_ptr<ITrtModule> text_encoder,
                           std::unique_ptr<ITrtModule> denoiser, std::unique_ptr<ITrtModule> vae,
                           std::unique_ptr<ITrtModule> audio, LTX2Options options,
                           std::shared_ptr<ITokenizer> tokenizer,
                           LTX2DistributedContext distributed)
    : distributed_(std::move(distributed)), text_encoder_(std::move(text_encoder)),
      denoiser_(std::move(denoiser)), vae_(std::move(vae)), audio_(std::move(audio)),
      options_(std::move(options)), tokenizer_(std::move(tokenizer)), progress_(distributed_.rank) {
}

LTX2Pipeline::~LTX2Pipeline() = default;

std::vector<internal::TaskInstance> LTX2Pipeline::task_bindings() {
    const auto& fields = config_fields();
    return {internal::bind<internal::ITextToAudioVideo>(*this, {fields.data(), fields.size()})};
}

LTX2Pipeline::TextContext LTX2Pipeline::encode(const std::string& text) {
    std::vector<int32_t> ids;
    std::vector<int32_t> mask;
    ltx2_prompt_ids(tokenizer_->encode(trim(text)), options_.text_seq_len, options_.pad_token_id,
                    ids, mask);
    const int64_t L = options_.text_seq_len;
    TensorMap inputs;
    inputs["input_ids"] = Tensor{ids.data(), {1, L}, DType::kInt32};
    inputs["attention_mask"] = Tensor{mask.data(), {1, L}, DType::kInt32};
    const auto outputs = text_encoder_->forward(inputs);
    const auto video_dim =
        static_cast<std::size_t>(text_encoder_->tensor_shape("video_context").back());
    const auto audio_dim =
        static_cast<std::size_t>(text_encoder_->tensor_shape("audio_context").back());
    TextContext context;
    context.video = bf16_output(outputs, "video_context", static_cast<std::size_t>(L) * video_dim);
    context.audio = bf16_output(outputs, "audio_context", static_cast<std::size_t>(L) * audio_dim);
    return context;
}

void LTX2Pipeline::run_dit(const std::vector<float>& video, const std::vector<float>& audio,
                           const TextContext& text, float timestep, std::vector<float>& video_out,
                           std::vector<float>& audio_out) {
    const int64_t S = options_.video_tokens();
    const int64_t Sa = options_.audio_frames;
    const int64_t L = options_.text_seq_len;
    std::vector<float> t{timestep};
    std::vector<float> keep{1.0F};
    TensorMap inputs;
    inputs["video_latent"] =
        Tensor{const_cast<float*>(video.data()), {1, S, options_.latent_channels}, DType::kFloat32};
    inputs["audio_latent"] = Tensor{
        const_cast<float*>(audio.data()), {1, Sa, options_.audio_latent_channels}, DType::kFloat32};
    inputs["video_context"] = Tensor{const_cast<uint16_t*>(text.video.data()),
                                     {1, L, static_cast<int64_t>(text.video.size()) / L},
                                     DType::kBFloat16};
    inputs["audio_context"] = Tensor{const_cast<uint16_t*>(text.audio.data()),
                                     {1, L, static_cast<int64_t>(text.audio.size()) / L},
                                     DType::kBFloat16};
    inputs["timestep"] = Tensor{t.data(), {1}, DType::kFloat32};
    inputs["stg_keep"] = Tensor{keep.data(), {1}, DType::kFloat32};
    inputs["av_keep"] = Tensor{keep.data(), {1}, DType::kFloat32};
    const auto outputs = denoiser_->forward(inputs);
    video_out = float_output(outputs, "video_velocity", video.size());
    audio_out = float_output(outputs, "audio_velocity", audio.size());
}

std::vector<float> LTX2Pipeline::decode_video(const std::vector<float>& video_latents) {
    TensorMap inputs;
    inputs["latents"] = Tensor{const_cast<float*>(video_latents.data()),
                               {1, options_.video_tokens(), options_.latent_channels},
                               DType::kFloat32};
    const auto outputs = vae_->forward(inputs);
    const auto count = static_cast<std::size_t>(options_.video_frames) * options_.video_height *
                       options_.video_width * 3U;
    return float_output(outputs, "frames", count);
}

std::vector<float> LTX2Pipeline::decode_audio(const std::vector<float>& audio_latents) {
    TensorMap inputs;
    inputs["audio_latents"] = Tensor{const_cast<float*>(audio_latents.data()),
                                     {1, options_.audio_frames, options_.audio_latent_channels},
                                     DType::kFloat32};
    const auto outputs = audio_->forward(inputs);
    const auto shape = audio_->tensor_shape("waveform");
    std::size_t count = 1;
    for (const auto dim : shape)
        count *= static_cast<std::size_t>(dim);
    return float_output(outputs, "waveform", count);
}

internal::AudioVideoResult LTX2Pipeline::run(const internal::TextToAudioVideoRequest& request,
                                             internal::ConfigView config) {
    const auto& fields = config_fields();
    internal::validate_config({fields.data(), fields.size()}, config);
    const auto seed =
        internal::config_get<std::int64_t>(config, {fields.data(), fields.size()}, "seed").value();
    const std::string prompt(request.prompt);
    const int32_t steps = static_cast<int32_t>(options_.sigmas.size()) - 1;
    const auto video_count =
        static_cast<std::size_t>(options_.video_tokens()) * options_.latent_channels;
    const auto audio_count =
        static_cast<std::size_t>(options_.audio_frames) * options_.audio_latent_channels;

    const auto t_start = Clock::now();
    if (progress_.enabled()) {
        std::ostringstream detail;
        detail << "world_size=" << distributed_.world_size << " frames=" << options_.video_frames
               << " width=" << options_.video_width << " height=" << options_.video_height
               << " steps=" << steps;
        progress_.start(detail.str());
        progress_.emit("encode_begin");
    }
    const auto text = encode(prompt);
    const auto t_text = Clock::now();
    progress_.emit("encode_end");

    std::vector<float> video(video_count);
    std::vector<float> audio(audio_count);
    if (const char* path = std::getenv("TRTMC_LTX2_INITIAL_LATENTS");
        path != nullptr && *path != '\0') {
        const auto values = read_f32_file(path);
        if (values.size() != video_count + audio_count)
            throw std::runtime_error(
                "TRTMC_LTX2_INITIAL_LATENTS must hold the packed video then audio noise");
        std::copy_n(values.begin(), video_count, video.begin());
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(video_count), audio_count,
                    audio.begin());
    } else {
        std::mt19937 generator(static_cast<uint32_t>(seed));
        ltx2::LibstdcxxNormalFloat normal;
        for (auto& v : video)
            v = normal(generator);
        for (auto& v : audio)
            v = normal(generator);
    }

    progress_.emit("denoise_begin");
    std::vector<double> step_ms;
    std::vector<float> video_v;
    std::vector<float> audio_v;
    for (int32_t step = 0; step < steps; ++step) {
        const auto step_start = Clock::now();
        const float sigma = options_.sigmas[static_cast<std::size_t>(step)];
        const float sigma_next = options_.sigmas[static_cast<std::size_t>(step) + 1];
        run_dit(video, audio, text, sigma * 1000.0F, video_v, audio_v);
        ltx2_euler_step(video, video_v, sigma, sigma_next);
        ltx2_euler_step(audio, audio_v, sigma, sigma_next);
        step_ms.push_back(elapsed_ms(step_start, Clock::now()));
        if (progress_.enabled()) {
            std::ostringstream detail;
            detail << "step=" << (step + 1) << "/" << steps << " step_ms=" << std::fixed
                   << std::setprecision(3) << step_ms.back();
            progress_.emit("step", detail.str());
        }
    }
    const auto t_denoise = Clock::now();
    progress_.emit("denoise_end");

    if (distributed_.world_size > 1 && distributed_.rank != 0) {
        std::cerr << "[ltx2] context-parallel rank " << distributed_.rank
                  << " finished denoising in " << elapsed_ms(t_text, t_denoise)
                  << " ms; rank 0 decodes the video and audio\n";
        progress_.emit("worker_done");
        return worker_completion(options_);
    }
    maybe_dump(video, audio);

    progress_.emit("vae_begin");
    auto frames = decode_video(video);
    const auto t_vae = Clock::now();
    progress_.emit("vae_end");
    progress_.emit("audio_begin");
    const auto wave = decode_audio(audio);
    const auto t_audio = Clock::now();
    progress_.emit("audio_end");

    internal::AudioVideoResult result;
    result.video.frames.pixels = std::move(frames);
    result.video.frames.height = options_.video_height;
    result.video.frames.width = options_.video_width;
    result.video.frames.channels = 3;
    result.video.frames.num_frames = options_.video_frames;
    result.video.timestamps_seconds.reserve(static_cast<std::size_t>(options_.video_frames));
    for (int32_t f = 0; f < options_.video_frames; ++f)
        result.video.timestamps_seconds.push_back(static_cast<double>(f) / options_.frame_rate);
    result.audio.samples = ltx2_interleave(wave, options_.audio_channels);
    result.audio.sample_rate = static_cast<uint32_t>(options_.audio_sample_rate);
    result.audio.channels = static_cast<uint32_t>(options_.audio_channels);
    result.audio_start_seconds = 0.0;
    result.video.inference_ms = elapsed_ms(t_start, t_audio);

    std::vector<double> sorted = step_ms;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    std::cerr << std::fixed << std::setprecision(3)
              << "[ltx2-perf-json] {\"world_size\":" << distributed_.world_size
              << ",\"text_encode_ms\":" << elapsed_ms(t_start, t_text)
              << ",\"denoise_ms\":" << elapsed_ms(t_text, t_denoise)
              << ",\"median_step_ms\":" << median
              << ",\"vae_decode_ms\":" << elapsed_ms(t_denoise, t_vae)
              << ",\"audio_decode_ms\":" << elapsed_ms(t_vae, t_audio)
              << ",\"generate_ms\":" << elapsed_ms(t_start, t_audio) << ",\"num_steps\":" << steps
              << "}\n";
    if (progress_.enabled()) {
        std::ostringstream detail;
        detail << "generate_ms=" << std::fixed << std::setprecision(3)
               << elapsed_ms(t_start, t_audio);
        progress_.emit("done", detail.str());
    }
    return result;
}

} // namespace trtmc

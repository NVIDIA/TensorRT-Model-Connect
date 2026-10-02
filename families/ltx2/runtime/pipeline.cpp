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
#include <cuda_runtime_api.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
//                               (replaces the seeded noise, e.g. a reference pipeline's draw).
//                               Two-stage runs append the stage 2 re-noise draws: packed
//                               full-resolution video [S2, C] then audio [Sa, Ca].
//   TRTMC_LTX2_DUMP_LATENTS     raw fp32 file written with the final video then audio latents;
//                               two-stage runs also write <file>.stage1 (stage 1 video, audio) and
//                               <file>.upsampled (upsampled video, audio)
//   TRTMC_LTX2_DECODE_LATENTS   raw fp32 file in the TRTMC_LTX2_DUMP_LATENTS layout; replaces the
//                               denoised latents before the decode (decoder checks, e.g. the
//                               single-GPU vs tile-parallel decode of identical latents)
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

void maybe_dump(const std::vector<float>& video, const std::vector<float>& audio,
                const char* suffix = "") {
    const char* base = std::getenv("TRTMC_LTX2_DUMP_LATENTS");
    if (base == nullptr || *base == '\0')
        return;
    const std::string path = std::string(base) + suffix;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(video.data()),
                 static_cast<std::streamsize>(video.size() * 4));
    output.write(reinterpret_cast<const char*>(audio.data()),
                 static_cast<std::streamsize>(audio.size() * 4));
    std::cerr << "[ltx2] wrote final latents (" << video.size() << " + " << audio.size()
              << " fp32) to " << path << "\n";
}

void replace_final_latents(std::vector<float>& video, std::vector<float>& audio) {
    const char* path = std::getenv("TRTMC_LTX2_DECODE_LATENTS");
    if (path == nullptr || *path == '\0')
        return;
    const auto values = read_f32_file(path);
    if (values.size() != video.size() + audio.size())
        throw std::runtime_error(
            "TRTMC_LTX2_DECODE_LATENTS must hold the packed final video then audio latents");
    std::copy_n(values.begin(), video.size(), video.begin());
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(video.size()), audio.size(),
                audio.begin());
    std::cerr << "[ltx2] decoding the latents of " << path << "\n";
}

const std::array<internal::ConfigField, 2>& config_fields() {
    static const std::array<internal::ConfigField, 2> fields{{
        {"seed", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Seed of the initial video and audio noise (portable std::mt19937 + normal draws)."},
        {"two_stage", internal::ConfigKind::Bool, internal::ConfigValue{false},
         "Two-stage pipeline: half-resolution stage 1, latent upsampler, full-resolution "
         "refinement (bundles built with trtmc ltx2 build --two-stage)."},
    }};
    return fields;
}

LTX2TwoStage parse_two_stage(const nlohmann::json& doc) {
    LTX2TwoStage two;
    two.latent_height = doc.at("latent_height").get<int32_t>();
    two.latent_width = doc.at("latent_width").get<int32_t>();
    two.sigmas = doc.at("sigmas").get<std::vector<float>>();
    two.stage2_sigmas = doc.at("stage2_sigmas").get<std::vector<float>>();
    two.noise_scale = doc.at("noise_scale").get<float>();
    for (const auto* schedule : {&two.sigmas, &two.stage2_sigmas}) {
        if (schedule->size() < 2 || schedule->back() != 0.0F)
            throw std::runtime_error("LTX-2.5 two-stage sigmas must end with the terminal 0");
    }
    if (two.latent_height <= 0 || two.latent_width <= 0)
        throw std::runtime_error("LTX-2.5 runtime.json has an invalid two-stage grid");
    return two;
}

ltx2::VaeTilePlan parse_tile_plan(const nlohmann::json& doc) {
    ltx2::VaeTilePlan plan;
    plan.tile_latent = doc.at("tile_latent").get<std::array<int32_t, 3>>();
    plan.tile_pixels = doc.at("tile_pixels").get<std::array<int32_t, 3>>();
    for (const auto& item : doc.at("tiles")) {
        ltx2::VaeTile tile;
        tile.latent_start = item.at("latent_start").get<std::array<int32_t, 3>>();
        tile.pixel_start = item.at("pixel_start").get<std::array<int32_t, 3>>();
        tile.ramps = item.at("ramps").get<std::array<std::array<int32_t, 2>, 3>>();
        tile.rank = item.at("rank").get<int32_t>();
        plan.tiles.push_back(tile);
    }
    return plan;
}

// Device scratch for one decode (tiles a worker rank sends or rank 0 receives).
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes != 0 && cudaMalloc(&ptr_, bytes) != cudaSuccess)
            throw std::runtime_error("LTX-2.5 tiled decode: cudaMalloc failed");
    }
    ~DeviceBuffer() {
        if (ptr_ != nullptr)
            cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint8_t* get() const { return static_cast<uint8_t*>(ptr_); }

  private:
    void* ptr_{nullptr};
};

void cuda_copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
    const auto status = cudaMemcpy(dst, src, bytes, kind);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("LTX-2.5 tiled decode: cudaMemcpy failed: ") +
                                 cudaGetErrorString(status));
}

std::size_t numel(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dim : shape)
        count *= static_cast<std::size_t>(dim);
    return count;
}

constexpr std::chrono::minutes kPeerTimeout{10};

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

LTX2Options parse_ltx2_options(const std::string& runtime_json, int32_t world_size) {
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
    if (doc.contains("audio_waveform_shape"))
        o.audio_waveform_shape = doc.at("audio_waveform_shape").get<std::vector<int64_t>>();
    if (doc.contains("two_stage")) {
        o.two_stage = parse_two_stage(doc.at("two_stage"));
        if (o.two_stage.latent_height * 2 != o.latent_height ||
            o.two_stage.latent_width * 2 != o.latent_width)
            throw std::runtime_error("LTX-2.5 two-stage grid must be half the video latent grid");
    }
    if (doc.contains("vae_tiling")) {
        const auto& tiling = doc.at("vae_tiling");
        if (tiling.at("world_size").get<int32_t>() != world_size)
            throw std::runtime_error("LTX-2.5 VAE tile plan was built for another world size");
        o.vae_tiling = parse_tile_plan(tiling);
        ltx2::vae_validate_plan(o.vae_tiling, {o.latent_frames, o.latent_height, o.latent_width},
                                {o.video_frames, o.video_height, o.video_width}, world_size);
    }
    return o;
}

LTX2Pipeline::LTX2Pipeline(std::unique_ptr<ITrtModule> text_encoder,
                           std::unique_ptr<ITrtModule> denoiser, std::unique_ptr<ITrtModule> vae,
                           std::unique_ptr<ITrtModule> audio, LTX2Options options,
                           std::shared_ptr<ITokenizer> tokenizer,
                           LTX2DistributedContext distributed,
                           std::unique_ptr<ITrtModule> upsampler)
    : distributed_(std::move(distributed)), text_encoder_(std::move(text_encoder)),
      denoiser_(std::move(denoiser)), vae_(std::move(vae)), audio_(std::move(audio)),
      upsampler_(std::move(upsampler)), options_(std::move(options)),
      tokenizer_(std::move(tokenizer)), progress_(distributed_.rank) {}

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
                           const TextContext& text, float timestep, int64_t video_tokens,
                           std::vector<float>& video_out, std::vector<float>& audio_out) {
    const int64_t S = video_tokens;
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

double LTX2Pipeline::StageTimes::median_ms() const {
    if (step_ms.empty())
        return 0.0;
    std::vector<double> sorted = step_ms;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
}

LTX2Pipeline::Noise LTX2Pipeline::initial_noise(int64_t seed, bool two_stage) const {
    const auto channels = static_cast<std::size_t>(options_.latent_channels);
    const auto full = static_cast<std::size_t>(options_.video_tokens()) * channels;
    const auto stage1 =
        two_stage
            ? static_cast<std::size_t>(options_.two_stage.video_tokens(options_.latent_frames)) *
                  channels
            : full;
    const auto audio =
        static_cast<std::size_t>(options_.audio_frames) * options_.audio_latent_channels;
    Noise noise;
    noise.video.resize(stage1);
    noise.audio.resize(audio);
    if (two_stage) {
        noise.video_stage2.resize(full);
        noise.audio_stage2.resize(audio);
    }
    const std::vector<std::vector<float>*> draws{&noise.video, &noise.audio, &noise.video_stage2,
                                                 &noise.audio_stage2};
    if (const char* path = std::getenv("TRTMC_LTX2_INITIAL_LATENTS");
        path != nullptr && *path != '\0') {
        const auto values = read_f32_file(path);
        std::size_t total = 0;
        for (const auto* d : draws)
            total += d->size();
        if (values.size() != total)
            throw std::runtime_error("TRTMC_LTX2_INITIAL_LATENTS must hold the packed video then "
                                     "audio noise (then the stage 2 video and audio draws)");
        auto it = values.begin();
        for (auto* d : draws) {
            std::copy_n(it, d->size(), d->begin());
            it += static_cast<std::ptrdiff_t>(d->size());
        }
        return noise;
    }
    std::mt19937 generator(static_cast<uint32_t>(seed));
    ltx2::LibstdcxxNormalFloat normal;
    for (auto* d : draws)
        for (auto& v : *d)
            v = normal(generator);
    return noise;
}

void LTX2Pipeline::denoise(std::vector<float>& video, std::vector<float>& audio,
                           const TextContext& text, const std::vector<float>& sigmas,
                           int64_t video_tokens, const char* stage, StageTimes& times) {
    const auto start = Clock::now();
    const int32_t steps = static_cast<int32_t>(sigmas.size()) - 1;
    std::vector<float> video_v;
    std::vector<float> audio_v;
    for (int32_t step = 0; step < steps; ++step) {
        const auto step_start = Clock::now();
        const float sigma = sigmas[static_cast<std::size_t>(step)];
        const float sigma_next = sigmas[static_cast<std::size_t>(step) + 1];
        run_dit(video, audio, text, sigma * 1000.0F, video_tokens, video_v, audio_v);
        ltx2_euler_step(video, video_v, sigma, sigma_next);
        ltx2_euler_step(audio, audio_v, sigma, sigma_next);
        times.step_ms.push_back(elapsed_ms(step_start, Clock::now()));
        if (progress_.enabled()) {
            std::ostringstream detail;
            detail << "stage=" << stage << " step=" << (step + 1) << "/" << steps
                   << " step_ms=" << std::fixed << std::setprecision(3) << times.step_ms.back();
            progress_.emit("step", detail.str());
        }
    }
    times.total_ms = elapsed_ms(start, Clock::now());
}

std::vector<float> LTX2Pipeline::upsample(const std::vector<float>& video, int64_t video_tokens) {
    if (!upsampler_)
        throw std::runtime_error("LTX-2.5 two-stage run without latent_upsampler.plan");
    TensorMap inputs;
    inputs["latents"] = Tensor{const_cast<float*>(video.data()),
                               {1, video_tokens, options_.latent_channels},
                               DType::kFloat32};
    const auto outputs = upsampler_->forward(inputs);
    return float_output(outputs, "upsampled",
                        static_cast<std::size_t>(options_.video_tokens()) *
                            options_.latent_channels);
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

uint8_t* LTX2Pipeline::tile_host_buffer(std::size_t bytes) {
    if (tile_host_bytes_ < bytes) {
        tile_host_.reset();
        tile_host_bytes_ = 0;
        void* ptr = nullptr;
        if (cudaMallocHost(&ptr, bytes) != cudaSuccess)
            throw std::runtime_error("LTX-2.5 tiled decode: cudaMallocHost failed");
        tile_host_ = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr),
                                              [](uint8_t* p) { cudaFreeHost(p); });
        tile_host_bytes_ = bytes;
    }
    return tile_host_.get();
}

LTX2Pipeline::Decoded LTX2Pipeline::decode_untiled(const std::vector<float>& video,
                                                   const std::vector<float>& audio) {
    Decoded out;
    auto start = Clock::now();
    out.frames = decode_video(video);
    out.tiles_ms = elapsed_ms(start, Clock::now());
    start = Clock::now();
    out.wave = decode_audio(audio);
    out.audio_ms = elapsed_ms(start, Clock::now());
    return out;
}

// Decodes this rank's tiles in tile order: into host slot k on rank 0, else packed into the
// device send buffer.
void LTX2Pipeline::decode_own_tiles(const std::vector<float>& video, uint8_t* host, void* device,
                                    Decoded& out) {
    const auto& plan = options_.vae_tiling;
    const auto bytes = plan.tile_values() * sizeof(uint16_t);
    const std::array<int32_t, 3> latent{options_.latent_frames, options_.latent_height,
                                        options_.latent_width};
    std::vector<float> tile_latents;
    std::size_t packed = 0;
    const auto start = Clock::now();
    for (std::size_t k = 0; k < plan.tiles.size(); ++k) {
        if (plan.tiles[k].rank != distributed_.rank)
            continue;
        const auto tile_start = Clock::now();
        ltx2::vae_gather_tile_latents(video, latent, options_.latent_channels, plan, plan.tiles[k],
                                      tile_latents);
        TensorMap inputs;
        inputs["latents"] =
            Tensor{tile_latents.data(),
                   {1, static_cast<int64_t>(tile_latents.size()) / options_.latent_channels,
                    options_.latent_channels},
                   DType::kFloat32};
        vae_->forward_async(inputs);
        vae_->sync();
        const void* frames = vae_->device_ptr("frames");
        if (host != nullptr)
            cuda_copy(host + k * bytes, frames, bytes, cudaMemcpyDeviceToHost);
        else
            cuda_copy(static_cast<uint8_t*>(device) + (packed++) * bytes, frames, bytes,
                      cudaMemcpyDeviceToDevice);
        ++out.tiles;
        if (progress_.enabled()) {
            std::ostringstream detail;
            detail << "tile=" << k << " tile_ms=" << std::fixed << std::setprecision(3)
                   << elapsed_ms(tile_start, Clock::now());
            progress_.emit("vae_tile", detail.str());
        }
    }
    out.tiles_ms = elapsed_ms(start, Clock::now());
}

// Rank 0: receives every worker's tiles (in their tile order) and the audio rank's waveform.
void LTX2Pipeline::receive_peer_tiles(uint8_t* host, Decoded& out) {
    const auto& plan = options_.vae_tiling;
    const auto bytes = plan.tile_values() * sizeof(uint16_t);
    const auto world = static_cast<std::size_t>(distributed_.world_size);
    const auto audio_rank = static_cast<std::size_t>(options_.audio_rank(distributed_.world_size));
    const auto wave_count = numel(options_.audio_waveform_shape);
    std::vector<std::size_t> peer_bytes(world, 0);
    for (const auto& tile : plan.tiles)
        peer_bytes[static_cast<std::size_t>(tile.rank)] += bytes;
    if (audio_rank != 0)
        peer_bytes[audio_rank] += wave_count * sizeof(float);
    std::vector<std::size_t> cursor(world, 0);
    std::size_t total = 0;
    for (std::size_t p = 1; p < world; ++p) {
        cursor[p] = total;
        total += peer_bytes[p];
    }
    DeviceBuffer recv(total);
    std::vector<ltx2::PeerTransfer> transfers;
    for (std::size_t p = 1; p < world; ++p) {
        if (peer_bytes[p] != 0)
            transfers.push_back(
                {static_cast<int>(p), recv.get() + cursor[p], peer_bytes[p], false});
    }
    const auto start = Clock::now();
    distributed_.channel->run(transfers, kPeerTimeout);
    for (std::size_t k = 0; k < plan.tiles.size(); ++k) {
        const auto rank = static_cast<std::size_t>(plan.tiles[k].rank);
        if (rank == 0)
            continue;
        cuda_copy(host + k * bytes, recv.get() + cursor[rank], bytes, cudaMemcpyDeviceToHost);
        cursor[rank] += bytes;
    }
    if (audio_rank != 0) {
        out.wave.resize(wave_count);
        cuda_copy(out.wave.data(), recv.get() + cursor[audio_rank], wave_count * sizeof(float),
                  cudaMemcpyDeviceToHost);
    }
    out.exchange_ms = elapsed_ms(start, Clock::now());
}

LTX2Pipeline::Decoded LTX2Pipeline::decode_tiled(const std::vector<float>& video,
                                                 const std::vector<float>& audio) {
    const auto& plan = options_.vae_tiling;
    const auto bytes = plan.tile_values() * sizeof(uint16_t);
    const int32_t audio_rank = options_.audio_rank(distributed_.world_size);
    Decoded out;
    if (distributed_.rank != 0) {
        std::size_t mine = 0;
        for (const auto& tile : plan.tiles)
            mine += tile.rank == distributed_.rank ? 1U : 0U;
        const auto wave_bytes = numel(options_.audio_waveform_shape) * sizeof(float);
        const bool sends_audio = distributed_.rank == audio_rank;
        DeviceBuffer send(mine * bytes + (sends_audio ? wave_bytes : 0));
        decode_own_tiles(video, nullptr, send.get(), out);
        if (sends_audio) {
            const auto start = Clock::now();
            (void)decode_audio(audio);
            cuda_copy(send.get() + mine * bytes, audio_->device_ptr("waveform"), wave_bytes,
                      cudaMemcpyDeviceToDevice);
            out.audio_ms = elapsed_ms(start, Clock::now());
        }
        const auto start = Clock::now();
        const auto total = mine * bytes + (sends_audio ? wave_bytes : 0);
        if (total != 0) // rank 0 posts a receive only for peers with data
            distributed_.channel->run({{0, send.get(), total, true}}, kPeerTimeout);
        out.exchange_ms = elapsed_ms(start, Clock::now());
        return out;
    }
    uint8_t* host = tile_host_buffer(plan.tiles.size() * bytes);
    decode_own_tiles(video, host, nullptr, out);
    if (distributed_.world_size > 1)
        receive_peer_tiles(host, out);
    std::vector<const uint16_t*> tiles(plan.tiles.size());
    for (std::size_t k = 0; k < tiles.size(); ++k)
        tiles[k] = reinterpret_cast<const uint16_t*>(host + k * bytes);
    // The host blend overlaps the audio decode when rank 0 decodes the audio.
    std::exception_ptr blend_error;
    std::thread blend([&] {
        try {
            const auto start = Clock::now();
            ltx2::vae_blend_tiles(plan, tiles, options_.video_frames, options_.video_height,
                                  options_.video_width, out.frames);
            out.blend_ms = elapsed_ms(start, Clock::now());
        } catch (...) {
            blend_error = std::current_exception();
        }
    });
    if (audio_rank == 0) {
        try {
            const auto start = Clock::now();
            out.wave = decode_audio(audio);
            out.audio_ms = elapsed_ms(start, Clock::now());
        } catch (...) {
            blend.join();
            throw;
        }
    }
    blend.join();
    if (blend_error)
        std::rethrow_exception(blend_error);
    return out;
}

internal::AudioVideoResult LTX2Pipeline::run(const internal::TextToAudioVideoRequest& request,
                                             internal::ConfigView config) {
    const auto& fields = config_fields();
    internal::validate_config({fields.data(), fields.size()}, config);
    const auto seed =
        internal::config_get<std::int64_t>(config, {fields.data(), fields.size()}, "seed").value();
    const bool two_stage =
        internal::config_get<bool>(config, {fields.data(), fields.size()}, "two_stage").value();
    if (two_stage && !options_.two_stage.enabled())
        throw internal::ConfigError(
            "two_stage=true needs a two-stage bundle (trtmc ltx2 build --two-stage)");
    const std::string prompt(request.prompt);
    const auto& stage1_sigmas = two_stage ? options_.two_stage.sigmas : options_.sigmas;
    const int32_t steps =
        static_cast<int32_t>(stage1_sigmas.size()) - 1 +
        (two_stage ? static_cast<int32_t>(options_.two_stage.stage2_sigmas.size()) - 1 : 0);

    // Ranks finish loading their engines at different times (the ranks load different decoders);
    // start together so the first collective does not charge one rank's load to the generation.
    if (distributed_.channel)
        distributed_.channel->barrier(kPeerTimeout);
    const auto t_start = Clock::now();
    if (progress_.enabled()) {
        std::ostringstream detail;
        detail << "world_size=" << distributed_.world_size << " frames=" << options_.video_frames
               << " width=" << options_.video_width << " height=" << options_.video_height
               << " steps=" << steps << " two_stage=" << (two_stage ? 1 : 0);
        progress_.start(detail.str());
        progress_.emit("encode_begin");
    }
    const auto text = encode(prompt);
    const auto t_text = Clock::now();
    progress_.emit("encode_end");

    auto noise = initial_noise(seed, two_stage);
    auto& video = noise.video;
    auto& audio = noise.audio;
    const int64_t stage1_tokens = two_stage
                                      ? options_.two_stage.video_tokens(options_.latent_frames)
                                      : options_.video_tokens();
    progress_.emit("denoise_begin");
    StageTimes stage1;
    StageTimes stage2;
    denoise(video, audio, text, stage1_sigmas, stage1_tokens, "stage1", stage1);
    double upsample_ms = 0.0;
    if (two_stage) {
        const auto up_start = Clock::now();
        if (distributed_.rank == 0)
            maybe_dump(video, audio, ".stage1");
        video = upsample(video, stage1_tokens);
        if (distributed_.rank == 0)
            maybe_dump(video, audio, ".upsampled");
        // diffusers prepare_latents / prepare_audio_latents: noise_scale * noise + (1 -
        // noise_scale) * x.
        const float s = options_.two_stage.noise_scale;
        ltx2_renoise(video, noise.video_stage2, s);
        ltx2_renoise(audio, noise.audio_stage2, s);
        upsample_ms = elapsed_ms(up_start, Clock::now());
        progress_.emit("upsample_end");
        denoise(video, audio, text, options_.two_stage.stage2_sigmas, options_.video_tokens(),
                "stage2", stage2);
    }
    const auto t_denoise = Clock::now();
    progress_.emit("denoise_end");
    replace_final_latents(video, audio);
    std::vector<double> step_ms = stage1.step_ms;
    step_ms.insert(step_ms.end(), stage2.step_ms.begin(), stage2.step_ms.end());

    const bool tiled = options_.vae_tiling.enabled();
    if (distributed_.world_size > 1 && distributed_.rank != 0 && !tiled) {
        std::cerr << "[ltx2] context-parallel rank " << distributed_.rank
                  << " finished denoising in " << elapsed_ms(t_text, t_denoise)
                  << " ms; rank 0 decodes the video and audio\n";
        progress_.emit("worker_done");
        return worker_completion(options_);
    }
    if (distributed_.rank == 0)
        maybe_dump(video, audio);

    progress_.emit("decode_begin");
    auto decoded = tiled ? decode_tiled(video, audio) : decode_untiled(video, audio);
    const auto t_audio = Clock::now();
    progress_.emit("decode_end");
    if (distributed_.rank != 0) {
        std::cerr << std::fixed << std::setprecision(3)
                  << "[ltx2-worker-perf-json] {\"rank\":" << distributed_.rank
                  << ",\"denoise_ms\":" << elapsed_ms(t_text, t_denoise)
                  << ",\"vae_tiles\":" << decoded.tiles << ",\"vae_tiles_ms\":" << decoded.tiles_ms
                  << ",\"audio_decode_ms\":" << decoded.audio_ms
                  << ",\"vae_send_ms\":" << decoded.exchange_ms << "}\n";
        progress_.emit("worker_done");
        return worker_completion(options_);
    }
    auto frames = std::move(decoded.frames);
    const auto& wave = decoded.wave;

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
              << ",\"median_step_ms\":" << median << ",\"two_stage\":" << (two_stage ? 1 : 0)
              << ",\"stage1_ms\":" << stage1.total_ms
              << ",\"stage1_median_step_ms\":" << stage1.median_ms()
              << ",\"upsample_ms\":" << upsample_ms << ",\"stage2_ms\":" << stage2.total_ms
              << ",\"stage2_median_step_ms\":" << stage2.median_ms() << ",\"decode_ms\":"
              << elapsed_ms(t_denoise, t_audio)
              // Untiled: the video then the audio decode. Tiled: the audio overlaps the blend
              // (one device) or runs on the audio rank, so the video path spans the phase.
              << ",\"vae_decode_ms\":"
              << (tiled ? elapsed_ms(t_denoise, t_audio) : decoded.tiles_ms)
              << ",\"audio_decode_ms\":" << decoded.audio_ms << ",\"vae_tiles\":" << decoded.tiles
              << ",\"vae_tiles_ms\":" << decoded.tiles_ms
              << ",\"vae_exchange_ms\":" << decoded.exchange_ms
              << ",\"vae_blend_ms\":" << decoded.blend_ms
              << ",\"audio_rank\":" << options_.audio_rank(distributed_.world_size)
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// LTX2Pipeline: native C++ runtime for Lightricks LTX-2.5 (text -> synchronized video + audio).
// All model execution goes through TensorRT component engines:
//   text_encoder.plan  Gemma 4 + LTX2TextConnectors -> video/audio text contexts
//   denoiser.plan      joint audio/video DiT (single device or context parallel)
//   vae.plan           video VAE decoder -> RGB frames (whole video, or one tile shape when the
//                      bundle carries a tile plan; context-parallel ranks decode disjoint tiles)
//   audio.plan         audio VAE decoder + vocoder with BWE -> 48 kHz stereo
//   latent_upsampler.plan  (two-stage bundles) 2x spatial latent upsampler between the stages

#include "families/ltx2/runtime/distributed_runtime.h"
#include "families/ltx2/runtime/progress_log.h"
#include "families/ltx2/runtime/tokenizer.h"
#include "families/ltx2/runtime/vae_tiling.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// Two-stage pipeline of a bundle built with `trtmc ltx2 build --two-stage`: stage 1 denoises the
// half-resolution grid, latent_upsampler.plan doubles it, stage 2 re-noises the video and audio
// latents to noise_scale and refines at full resolution.
struct LTX2TwoStage {
    int32_t latent_height{0};
    int32_t latent_width{0};
    std::vector<float> sigmas;        // stage 1 schedule including the terminal 0
    std::vector<float> stage2_sigmas; // stage 2 schedule including the terminal 0
    float noise_scale{0.0F};

    bool enabled() const { return latent_height > 0; }
    int64_t video_tokens(int32_t latent_frames) const {
        return int64_t(latent_frames) * latent_height * latent_width;
    }
};

struct LTX2Options {
    int32_t video_frames{121};
    int32_t video_height{544};
    int32_t video_width{960};
    int32_t latent_frames{16};
    int32_t latent_height{17};
    int32_t latent_width{30};
    int32_t latent_channels{128};
    int32_t audio_frames{126};
    int32_t audio_latent_channels{128};
    int32_t text_seq_len{1024};
    float frame_rate{24.0F};
    int32_t pad_token_id{0};
    // Full schedule including the terminal 0: the model runs at sigmas[0..n-1].
    std::vector<float> sigmas;
    int32_t audio_sample_rate{48000};
    int32_t audio_channels{2};
    // Tiled video decode (empty: vae.plan decodes the whole video on rank 0).
    ltx2::VaeTilePlan vae_tiling;
    // Waveform shape of audio.plan; lets another rank decode the audio for rank 0.
    std::vector<int64_t> audio_waveform_shape;
    LTX2TwoStage two_stage;

    int64_t video_tokens() const { return int64_t(latent_frames) * latent_height * latent_width; }
    // Rank that decodes the audio: the last context-parallel rank when the tiles spread the
    // video decode over every rank, else rank 0.
    int32_t audio_rank(int32_t world_size) const {
        return world_size > 1 && vae_tiling.enabled() && !audio_waveform_shape.empty()
                   ? world_size - 1
                   : 0;
    }
};

LTX2Options parse_ltx2_options(const std::string& runtime_json, int32_t world_size = 1);

// Context-parallel participation. The owner keeps the NCCL communicator used by the
// denoiser engine alive for the pipeline lifetime. Rank 0 returns the media; other ranks
// return the worker completion. With a tile plan every rank decodes its video tiles and sends
// them (and, on the audio rank, the waveform) to rank 0 over `channel`.
struct LTX2DistributedContext {
    std::shared_ptr<void> owner;
    int32_t rank{0};
    int32_t world_size{1};
    std::shared_ptr<ltx2::PeerChannel> channel;
};

class LTX2Pipeline final : public internal::IModel, public internal::ITextToAudioVideo {
  public:
    LTX2Pipeline(std::unique_ptr<ITrtModule> text_encoder, std::unique_ptr<ITrtModule> denoiser,
                 std::unique_ptr<ITrtModule> vae, std::unique_ptr<ITrtModule> audio,
                 LTX2Options options, std::shared_ptr<ITokenizer> tokenizer,
                 LTX2DistributedContext distributed = {},
                 std::unique_ptr<ITrtModule> upsampler = nullptr);
    ~LTX2Pipeline() override;

    const char* task() const noexcept override { return ITextToAudioVideo::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioVideoResult run(const internal::TextToAudioVideoRequest& request,
                                   internal::ConfigView config) override;

    struct TextContext {
        std::vector<uint16_t> video; // [1, L, 4096] bf16 bits
        std::vector<uint16_t> audio; // [1, L, 2048] bf16 bits
    };

    // Decoded media (rank 0) and the decode phase timings of this rank.
    struct Decoded {
        std::vector<float> frames;
        std::vector<float> wave;
        double tiles_ms{0.0};
        double audio_ms{0.0};
        double exchange_ms{0.0};
        double blend_ms{0.0};
        int32_t tiles{0};
    };

    // Initial noise: stage 1 video then audio; for two-stage runs the stage 2 re-noise draws
    // (video at full resolution, then audio) continue the same stream.
    struct Noise {
        std::vector<float> video;
        std::vector<float> audio;
        std::vector<float> video_stage2;
        std::vector<float> audio_stage2;
    };

    struct StageTimes {
        std::vector<double> step_ms;
        double total_ms{0.0};
        double median_ms() const;
    };

  private:
    TextContext encode(const std::string& text);
    void run_dit(const std::vector<float>& video, const std::vector<float>& audio,
                 const TextContext& text, float timestep, int64_t video_tokens,
                 std::vector<float>& video_out, std::vector<float>& audio_out);
    Noise initial_noise(int64_t seed, bool two_stage) const;
    void denoise(std::vector<float>& video, std::vector<float>& audio, const TextContext& text,
                 const std::vector<float>& sigmas, int64_t video_tokens, const char* stage,
                 StageTimes& times);
    std::vector<float> upsample(const std::vector<float>& video, int64_t video_tokens);
    std::vector<float> decode_video(const std::vector<float>& video_latents);
    std::vector<float> decode_audio(const std::vector<float>& audio_latents);
    Decoded decode_untiled(const std::vector<float>& video, const std::vector<float>& audio);
    Decoded decode_tiled(const std::vector<float>& video, const std::vector<float>& audio);
    void decode_own_tiles(const std::vector<float>& video, uint8_t* host, void* device,
                          Decoded& out);
    void receive_peer_tiles(uint8_t* host, Decoded& out);
    uint8_t* tile_host_buffer(std::size_t bytes);

    // Declared first so the communicator outlives every engine that uses it.
    LTX2DistributedContext distributed_;
    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> denoiser_;
    std::unique_ptr<ITrtModule> vae_;
    std::unique_ptr<ITrtModule> audio_;
    std::unique_ptr<ITrtModule> upsampler_;
    LTX2Options options_;
    std::shared_ptr<ITokenizer> tokenizer_;
    LTX2ProgressLog progress_;
    std::shared_ptr<uint8_t> tile_host_; // pinned decoded tiles on rank 0
    std::size_t tile_host_bytes_{0};
};

} // namespace trtmc

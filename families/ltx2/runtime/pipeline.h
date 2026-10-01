/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// LTX2Pipeline: native C++ runtime for Lightricks LTX-2.5 (text -> synchronized video + audio).
// All model execution goes through TensorRT component engines:
//   text_encoder.plan  Gemma 4 + LTX2TextConnectors -> video/audio text contexts
//   denoiser.plan      joint audio/video DiT (single device or context parallel)
//   vae.plan           video VAE decoder -> RGB frames
//   audio.plan         audio VAE decoder + vocoder with BWE -> 48 kHz stereo

#include "families/ltx2/runtime/progress_log.h"
#include "families/ltx2/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

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

    int64_t video_tokens() const { return int64_t(latent_frames) * latent_height * latent_width; }
};

LTX2Options parse_ltx2_options(const std::string& runtime_json);

// Context-parallel participation. The owner keeps the NCCL communicator used by the
// denoiser engine alive for the pipeline lifetime. Rank 0 decodes and returns media;
// other ranks return the worker completion.
struct LTX2DistributedContext {
    std::shared_ptr<void> owner;
    int32_t rank{0};
    int32_t world_size{1};
};

class LTX2Pipeline final : public internal::IModel, public internal::ITextToAudioVideo {
  public:
    LTX2Pipeline(std::unique_ptr<ITrtModule> text_encoder, std::unique_ptr<ITrtModule> denoiser,
                 std::unique_ptr<ITrtModule> vae, std::unique_ptr<ITrtModule> audio,
                 LTX2Options options, std::shared_ptr<ITokenizer> tokenizer,
                 LTX2DistributedContext distributed = {});
    ~LTX2Pipeline() override;

    const char* task() const noexcept override { return ITextToAudioVideo::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioVideoResult run(const internal::TextToAudioVideoRequest& request,
                                   internal::ConfigView config) override;

    struct TextContext {
        std::vector<uint16_t> video; // [1, L, 4096] bf16 bits
        std::vector<uint16_t> audio; // [1, L, 2048] bf16 bits
    };

  private:
    TextContext encode(const std::string& text);
    void run_dit(const std::vector<float>& video, const std::vector<float>& audio,
                 const TextContext& text, float timestep, std::vector<float>& video_out,
                 std::vector<float>& audio_out);
    std::vector<float> decode_video(const std::vector<float>& video_latents);
    std::vector<float> decode_audio(const std::vector<float>& audio_latents);

    // Declared first so the communicator outlives every engine that uses it.
    LTX2DistributedContext distributed_;
    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> denoiser_;
    std::unique_ptr<ITrtModule> vae_;
    std::unique_ptr<ITrtModule> audio_;
    LTX2Options options_;
    std::shared_ptr<ITokenizer> tokenizer_;
    LTX2ProgressLog progress_;
};

} // namespace trtmc

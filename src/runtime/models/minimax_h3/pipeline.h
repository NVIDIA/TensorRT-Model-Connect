/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using MiniMaxH3ModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string&, cudaStream_t)>;
using MiniMaxH3ProfileModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string&, cudaStream_t, int32_t)>;

enum class MiniMaxH3Workflow {
    kT2va,
    kFl2va,
    kRef2va,
};

struct MiniMaxH3Schedule {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
};

enum class MiniMaxH3KeyframeAnchor {
    kFirst,
    kLast,
};

struct MiniMaxH3PackedLayout {
    int32_t sequence_rows{0};
    std::vector<float> position_ids; // [sequence_rows, 3]
    std::vector<int32_t> token_tags;
    std::vector<int32_t> video_indices;
    std::vector<int32_t> audio_indices;
    std::vector<int32_t> text_indices;
    int32_t num_condition_video_rows{0};
    int32_t num_condition_audio_rows{0};
};

struct MiniMaxH3PreparedReferenceLayout {
    AudioVideoReferenceKind kind{AudioVideoReferenceKind::kImage};
    int32_t num_latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t num_audio_latents{0};
};

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift);
void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next);
std::vector<float> minimax_h3_unpack_audio_latents(const std::vector<float>& rows);
void validate_minimax_h3_request(const AudioVideoRequest& request);
MiniMaxH3PackedLayout
make_minimax_h3_fl2va_layout(const std::vector<int32_t>& text_token_tags, int32_t num_latent_frames,
                             int32_t latent_height, int32_t latent_width, int32_t num_audio_latents,
                             const std::vector<MiniMaxH3KeyframeAnchor>& keyframe_anchors = {});
MiniMaxH3PackedLayout
make_minimax_h3_ref2va_layout(const std::vector<int32_t>& text_token_tags,
                              const std::vector<MiniMaxH3PreparedReferenceLayout>& references,
                              int32_t num_latent_frames, int32_t latent_height,
                              int32_t latent_width, int32_t num_audio_latents);
std::vector<int32_t> make_minimax_h3_conditioned_timestep_indices(
    const MiniMaxH3PackedLayout& layout, int32_t video_slot, int32_t audio_slot,
    int32_t condition_video_slot, int32_t condition_audio_slot);

class MiniMaxH3Pipeline final : public IPipeline {
  public:
    MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                      std::string model_id, bool first_block_cache = false,
                      float cache_threshold = 0.025F,
                      MiniMaxH3Workflow workflow = MiniMaxH3Workflow::kT2va,
                      MiniMaxH3ProfileModuleLoader profile_loader = {});
    ~MiniMaxH3Pipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    AudioVideoResult generate_audio_video(const std::string& prompt,
                                          const GenerateConfig& cfg = {}) override;
    AudioVideoResult generate_audio_video(const AudioVideoRequest& request) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MiniMaxH3Pipeline"; }

  private:
    struct ResidentState;

    AudioVideoResult generate_joint(const std::string& prompt, const GenerateConfig& cfg,
                                    bool decode_audio);
    AudioVideoResult generate_fl2va(const AudioVideoRequest& request);
    AudioVideoResult generate_ref2va(const AudioVideoRequest& request);

    MiniMaxH3ModuleLoader loader_;
    MiniMaxH3ProfileModuleLoader profile_loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
    std::unique_ptr<ResidentState> resident_;
    bool first_block_cache_{false};
    float cache_threshold_{0.025F};
    MiniMaxH3Workflow workflow_{MiniMaxH3Workflow::kT2va};
};

} // namespace trtmc

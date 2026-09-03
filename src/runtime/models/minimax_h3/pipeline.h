/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/runtime_cache_control.h"
#include "runtime/models/minimax_h3/public_profile.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <array>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using MiniMaxH3ModuleLoader = std::function<std::unique_ptr<ITrtModule>(
    const std::string&, cudaStream_t, const std::vector<ModuleExternalBinding>&)>;

struct MiniMaxH3Schedule {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
};

struct MiniMaxH3VaeTileLayout {
    std::vector<int32_t> y_starts;
    std::vector<int32_t> x_starts;
    std::vector<int32_t> y_overlaps;
    std::vector<int32_t> x_overlaps;
};

struct MiniMaxH3Geometry {
    int32_t output_frames{0};
    int32_t output_height{0};
    int32_t output_width{0};
    int32_t video_latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t audio_latent_frames{0};
    int32_t audio_rows{0};
    // The denoiser's video binding is [condition_video_rows | target_video_rows].
    // T2VA has zero condition frames and video_rows == target_video_rows.
    int32_t condition_video_frames{0};
    int32_t condition_video_rows{0};
    int32_t target_video_rows{0};
    int32_t video_rows{0};
    int32_t vae_tile_rows{0};
    int32_t vae_tile_columns{0};
    int32_t vae_tile_count{0};
    int32_t vsa_video_tiles{0};
    int32_t vsa_top_video_tiles{0};
};

struct MiniMaxH3VsaMetadata {
    // Natural packed rows are text | stereo audio | video raster.  Each value
    // is the unique flattened slot in the segment-pure, tile-major VSA tensor.
    std::vector<int32_t> packed_row_to_tile_slot;
    // Inverse map consumed by the native CUDA tiling/untile kernels. Padding
    // slots are -1 and all live slots map to one natural packed row.
    std::vector<int32_t> tiled_slot_to_packed_row;
    std::vector<int32_t> prefix_valid_sizes;
    std::vector<int32_t> video_valid_sizes;
};

struct MiniMaxH3DenoiserMetadata {
    std::vector<float> positions;
    std::vector<int32_t> adaln_indices;
    std::vector<int32_t> timestep_indices;
    MiniMaxH3VsaMetadata vsa;
};

struct MiniMaxH3DenoiserConfig {
    bool native_vsa{false};
    int32_t scheduler_grid_points{50};
    int32_t transformer_forwards{49};
    float guidance_scale{1.0F};
    int32_t max_text_rows{537};
};

struct MiniMaxH3Ref2VAConfig {
    bool enabled{false};
    int32_t scheduler_grid_points{50};
    int32_t transformer_forwards{49};
    float video_shift{12.0F};
    float audio_shift{3.0F};
    float guidance_scale{1.0F};
    bool guidance_distilled{true};
    std::array<float, 32> audio_latent_mean{};
    std::array<float, 32> audio_latent_std{};
};

enum class MiniMaxH3SegmentPlanKind {
    kEntry,
    kTransition,
    kFinish,
};

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift);
MiniMaxH3VaeTileLayout make_minimax_h3_vae_tile_layout(int32_t output_height, int32_t output_width);
MiniMaxH3Geometry make_minimax_h3_geometry(int32_t output_frames, int32_t output_height,
                                           int32_t output_width);
MiniMaxH3Geometry make_minimax_h3_fl2va_geometry(const MiniMaxH3Geometry& target_geometry,
                                                 int32_t keyframe_count);
MiniMaxH3VsaMetadata make_minimax_h3_vsa_metadata(int32_t text_rows,
                                                  const MiniMaxH3Geometry& geometry);
MiniMaxH3DenoiserMetadata make_minimax_h3_denoiser_metadata(int32_t text_rows,
                                                            const MiniMaxH3Geometry& geometry,
                                                            bool native_vsa);
MiniMaxH3DenoiserMetadata
make_minimax_h3_fl2va_denoiser_metadata(const std::vector<int32_t>& text_token_tags,
                                        const std::vector<int32_t>& keyframe_anchors,
                                        const MiniMaxH3Geometry& geometry, bool native_vsa);
void validate_minimax_h3_prompt_token_count(std::size_t token_count, int32_t max_text_rows);
void validate_minimax_h3_monolithic_denoiser_plan(ITrtModule& module, bool native_vsa,
                                                  int32_t expected_max_text_rows = 0);
void validate_minimax_h3_segment_plan(ITrtModule& module, MiniMaxH3SegmentPlanKind kind);
std::vector<float> make_minimax_h3_position_ids(int32_t text_rows);
std::vector<float> make_minimax_h3_position_ids(int32_t text_rows,
                                                const MiniMaxH3Geometry& geometry);
std::vector<float> unpack_and_denormalize_minimax_h3_audio(const std::vector<float>& audio_rows,
                                                           int32_t audio_latent_frames);
void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next);

class MiniMaxH3Pipeline final : public IPipeline, public IRuntimeCacheControl {
  public:
    MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                      std::string model_id, bool first_block_cache = false,
                      float cache_threshold = 0.025F, MiniMaxH3DenoiserConfig denoiser_config = {},
                      MiniMaxH3Ref2VAConfig ref2va_config = {},
                      std::function<void()> runtime_cache_finalize = {});
    ~MiniMaxH3Pipeline() override;

    VideoResult generate_video(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    VideoResult generate_video(const VideoGenerationRequest& request) override;
    void finalize_runtime_cache() override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MiniMaxH3Pipeline"; }

  private:
    struct ResidentState;

    MiniMaxH3ModuleLoader loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
    std::unique_ptr<ResidentState> resident_;
    bool first_block_cache_{false};
    float cache_threshold_{0.025F};
    MiniMaxH3DenoiserConfig denoiser_config_{};
    MiniMaxH3Ref2VAConfig ref2va_config_{};
    std::function<void()> runtime_cache_finalize_;
    bool runtime_cache_finalize_started_{false};
    bool runtime_cache_contexts_released_{false};

    VideoResult generate_video_impl(const std::string& prompt, const GenerateConfig& cfg,
                                    bool include_audio);
    VideoResult generate_video_request_impl(const VideoGenerationRequest& request,
                                            bool include_audio);
    VideoResult generate_ref2va_request_impl(const VideoGenerationRequest& request,
                                             bool include_audio);
};

} // namespace trtmc

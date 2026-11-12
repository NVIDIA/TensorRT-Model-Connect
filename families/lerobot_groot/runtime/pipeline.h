/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lerobot_groot/runtime/prompt.h"
#include "families/lerobot_groot/runtime/tokenizer.h"
#include "trtmc/internal/action.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc::lerobot_groot {

struct RuntimeConfig {
    std::vector<std::string> cameras; // Checkpoint keys, in the order the model sees them.
    std::int32_t camera_size{0};      // Square cameras, already at the processor's target size.
    std::int32_t crop{0};             // Centred crop before INTER_AREA back to camera_size.
    std::int32_t chunk_size{0};       // Leading steps of the native horizon returned per call.
    std::int32_t action_horizon{0};
    std::int32_t state_dim{0};
    std::int32_t action_dim{0};
    std::int32_t max_state_dim{0};
    std::int32_t max_action_dim{0};
    std::int32_t num_inference_steps{0};
    std::int32_t max_instruction_tokens{0};
    bool formalize_language{true};
    std::vector<float> state_min, state_max;
    bool clip_state{true};
    std::vector<float> action_min, action_max;
    bool clip_action{true};
    bool libero_gripper{false};

    std::int32_t grid() const; // Merged image tokens per side.
};

// GR00T N1.7 flow-matching policy: one condition pass over the cameras, prompt and state, then
// `num_inference_steps` Euler steps of the DiT action head. The cross-attention keys/values the
// condition engine writes stay on the device for the denoising steps.
class Pipeline final : public internal::IModel, public internal::IImagesStateTextToActionChunk {
  public:
    Pipeline(std::unique_ptr<ITrtModule> condition, std::unique_ptr<ITrtModule> denoise,
             std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::ImagesStateTextActionChunkResult
    run(const internal::ImagesStateTextToActionChunkRequest& request,
        internal::ConfigView config) override;

  private:
    struct Options {
        bool has_seed{false};
        std::uint64_t seed{0};
        std::vector<float> noise; // Empty unless supplied explicitly.
    };
    Options parse_options(internal::ConfigView config) const;
    std::vector<float> initial_noise(const Options& options);
    std::vector<float> pack_pixels(const internal::ImagesStateTextObservation& observation) const;
    std::vector<float> normalized_state(Span<const float> state) const;
    Prompt prompt(std::string_view text) const;
    internal::ImagesStateTextActionChunkResult pack_actions(const std::vector<float>& sample) const;
    void validate_engines();

    std::unique_ptr<ITrtModule> condition_;
    std::unique_ptr<ITrtModule> denoise_;
    std::unique_ptr<ITokenizer> tokenizer_;
    RuntimeConfig config_;
    std::string primary_task_;
    TemplateTokens template_;
    std::mt19937_64 rng_;
    std::int64_t max_tokens_{0};
    std::int64_t cross_blocks_{0};
    std::int64_t cross_width_{0};
    DeviceTensor cross_k_;
    DeviceTensor cross_v_;
    DeviceTensor state_features_;
};

} // namespace trtmc::lerobot_groot

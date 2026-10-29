/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lerobot_pi05/runtime/paligemma_bpe.h"
#include "trtmc/internal/action.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc::lerobot_pi05 {

// Per-feature normalization. Empty statistics mean the value passes through, as
// LeRobot skips a feature that has none.
struct Normalization {
    float eps{1.0e-8F};
    std::vector<float> mean;
    std::vector<float> std;
    bool identity() const { return mean.empty(); }
};

struct RuntimeConfig {
    std::vector<std::string> cameras; // Real cameras, in the checkpoint's order.
    std::int32_t empty_cameras{0};    // Masked placeholder slots appended after them.
    std::int32_t camera_height{0};
    std::int32_t camera_width{0};
    std::int32_t tokens_per_image{0};
    std::int32_t prompt_tokens{0};
    std::int32_t prefix_tokens{0};
    std::int32_t chunk_size{0};
    std::int32_t state_dim{0};
    std::int32_t action_dim{0};
    std::int32_t max_action_dim{0};
    std::int32_t num_inference_steps{0};
    Normalization state_normalization;
    Normalization action_normalization;

    std::int32_t camera_slots() const {
        return static_cast<std::int32_t>(cameras.size()) + empty_cameras;
    }
};

// Pi 0.5 flow-matching policy: one prefix pass over the cameras and prompt, then
// `num_inference_steps` Euler steps of the action expert.
class Pipeline final : public internal::IModel, public internal::IImagesStateTextToActionChunk {
  public:
    Pipeline(std::unique_ptr<ITrtModule> prefix, std::unique_ptr<ITrtModule> denoise,
             PaligemmaBpeTokenizer tokenizer, RuntimeConfig config, std::string primary_task);

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

    std::unique_ptr<ITrtModule> prefix_;
    std::unique_ptr<ITrtModule> denoise_;
    PaligemmaBpeTokenizer tokenizer_;
    RuntimeConfig config_;
    std::string primary_task_;
    std::mt19937_64 rng_;
};

} // namespace trtmc::lerobot_pi05

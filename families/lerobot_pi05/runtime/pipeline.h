/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lerobot_pi05/runtime/normalization.h"
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
    std::vector<float> action_min; // Dataset range per action dimension; empty when unknown.
    std::vector<float> action_max;

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
    // What the prefix engine reads, and the masks the denoise steps reuse.
    struct PrefixInputs {
        std::vector<float> pixels;
        std::vector<std::int32_t> token_ids;
        std::vector<std::int32_t> positions;
        std::vector<float> attention_bias;
        std::vector<float> prefix_bias;
        std::int32_t real_tokens{0};
    };
    Options parse_options(internal::ConfigView config) const;
    std::vector<float> initial_noise(const Options& options);
    std::vector<float> pack_pixels(const internal::ImagesStateTextObservation& observation) const;
    std::vector<float> normalized_state(Span<const float> state) const;
    PrefixInputs prefix_inputs(const internal::ImagesStateTextObservation& observation) const;
    std::vector<float> denoise(PrefixInputs& inputs, const Tensor& k_cache, const Tensor& v_cache,
                               std::vector<float> sample);
    internal::ImagesStateTextActionChunkResult pack_actions(const std::vector<float>& sample) const;
    void validate_engines();

    std::unique_ptr<ITrtModule> prefix_;
    std::unique_ptr<ITrtModule> denoise_;
    PaligemmaBpeTokenizer tokenizer_;
    RuntimeConfig config_;
    std::string primary_task_;
    std::mt19937_64 rng_;
    std::size_t cache_numel_{0}; // Elements in one K or V cache, from the engines themselves.
};

} // namespace trtmc::lerobot_pi05

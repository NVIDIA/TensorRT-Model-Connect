/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/magpie/magpie_config.h"

#include <algorithm>
#include <cstdint>

namespace trtmc {

struct MagpieDecoderPlan {
    int32_t hidden{0};
    int32_t num_cb{0};
    int32_t cb_size{0};
    int32_t total_logits{0};
    bool use_cfg{false};
    bool use_gpu_kernels{false};
    bool use_gpu_greedy{false};
    bool use_gpu_sampling{false};
    bool use_cross_attn_tracking{false};
    int32_t estimated_frames{0};
    int32_t finished_limit{0};
    int32_t max_source_positions{0};
    int32_t text_consumed_threshold{1};
};

inline bool should_enable_magpie_cfg(const MagpieTTSConfig& config, bool has_uncond_cache,
                                     bool has_uncond_resources, bool has_uncond_cross_kv) {
    return config.cfg_scale > 1.0F && has_uncond_cache && has_uncond_resources &&
           has_uncond_cross_kv;
}

inline bool should_enable_magpie_gpu_greedy(bool use_gpu_kernels, bool greedy) {
    return use_gpu_kernels && greedy;
}

inline MagpieDecoderPlan make_magpie_decoder_plan(const MagpieTTSConfig& config,
                                                  bool has_uncond_cache, bool has_uncond_resources,
                                                  bool has_uncond_cross_kv, bool use_gpu_kernels,
                                                  bool has_cross_attn_output,
                                                  bool has_cross_attn_weights,
                                                  int32_t text_length) {
    MagpieDecoderPlan plan;
    plan.hidden = config.hidden_size;
    plan.num_cb = config.num_codebooks;
    plan.cb_size = config.codebook_size;
    plan.total_logits = plan.num_cb * plan.cb_size;
    plan.use_cfg = should_enable_magpie_cfg(config, has_uncond_cache, has_uncond_resources,
                                            has_uncond_cross_kv);
    plan.use_gpu_kernels = use_gpu_kernels;
    plan.use_gpu_greedy = should_enable_magpie_gpu_greedy(use_gpu_kernels, config.greedy);
    plan.use_gpu_sampling = use_gpu_kernels && !config.greedy;
    plan.finished_limit = config.finished_limit_with_eot;
    plan.max_source_positions = config.max_source_positions;
    plan.use_cross_attn_tracking =
        has_cross_attn_output && has_cross_attn_weights && text_length > 0;
    plan.estimated_frames = (!plan.use_cross_attn_tracking && text_length > 0)
                                ? static_cast<int32_t>(static_cast<float>(text_length) * 3.0F)
                                : 0;
    plan.text_consumed_threshold =
        std::max(static_cast<int32_t>(static_cast<float>(text_length) * 0.9F), 1);
    return plan;
}

} // namespace trtmc

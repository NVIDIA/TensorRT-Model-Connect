/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace trtmc::qwen_image2_1 {

// Static contract of one bundle, written by the family builder into runtime.json.
struct RuntimeConfig {
    std::int32_t image_height{0};
    std::int32_t image_width{0};
    std::int32_t image_channels{0};
    std::int32_t latent_channels{0};
    std::int32_t vae_scale_factor{0};

    // Text slots after the system turn, system-turn tokens dropped from the encoder output,
    // and the id used to fill unused input slots.
    std::int32_t text_tokens{0};
    std::int32_t text_hidden_size{0};
    std::int32_t prefix_tokens{0};
    std::int32_t pad_token_id{0};
    std::string prompt_prefix;
    std::string prompt_suffix;

    // FlowMatchEulerDiscreteScheduler with exponential dynamic shifting.
    std::int32_t num_train_timesteps{0};
    double base_shift{0.0};
    double max_shift{0.0};
    std::int32_t base_image_seq_len{0};
    std::int32_t max_image_seq_len{0};
    double shift_terminal{0.0};

    std::int32_t default_num_steps{0};

    std::int32_t latent_height() const { return image_height / vae_scale_factor; }
    std::int32_t latent_width() const { return image_width / vae_scale_factor; }
    std::int32_t image_tokens() const { return latent_height() * latent_width(); }
    std::size_t latent_count() const {
        return static_cast<std::size_t>(latent_channels) * static_cast<std::size_t>(image_tokens());
    }
};

// Parses and validates runtime.json. Throws std::runtime_error on a malformed bundle.
RuntimeConfig parse_runtime_config(const std::string& document);

} // namespace trtmc::qwen_image2_1

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/flux3/runtime/runtime_config.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::flux3 {

struct Engines {
    std::unique_ptr<ITrtModule> text_encoder;
    std::unique_ptr<ITrtModule> denoiser;
    std::unique_ptr<ITrtModule> vae;
};

// The token latents [1, latent_height * latent_width, latent_channels].
std::vector<std::int64_t> latent_tensor_shape(const RuntimeConfig& config);

// Each check throws std::runtime_error unless the engine has exactly the
// bundle contract's tensors, dtypes and shapes for this runtime.json.
void validate_text_encoder(const ITrtModule& engine, const RuntimeConfig& config);
void validate_denoiser(const ITrtModule& engine, const RuntimeConfig& config);
void validate_vae(const ITrtModule& engine, const RuntimeConfig& config);

} // namespace trtmc::flux3

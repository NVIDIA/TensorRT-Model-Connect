/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "plugin_helpers.h"
#include "preprocessor_weights_helpers.h"
#include "runtime/models/z_image/z_image_diffusion_types.h"
#include "utils/json_helpers.h"

namespace trtmc {

ZImageDiffusionConfig make_diffusion_config(const std::string& json);

// Z-Image diffusion resources loaded once, then dispatched to the Z-Image factory.
struct DiffusionParts {
    LoadedModule denoiser;
    LoadedModule vae;
    LoadedModule vision;
    LoadedModule vae_encoder;
    std::vector<LoadedModule> text_encoders;
    ZImageDiffusionConfig config;
    ZImageCommonPreprocessorWeights weights;
    std::shared_ptr<ITokenizer> tokenizer;
};

DiffusionParts load_diffusion_parts(IBackend* backend, const BundleFile& bundle,
                                    const std::string& json,
                                    const ModuleCreateOptions& options = {},
                                    const std::string& denoiser_section_name = "denoiser_plan",
                                    const ModuleCreateOptions* denoiser_options = nullptr);

} // namespace trtmc

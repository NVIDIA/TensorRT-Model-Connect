/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/minicpmv4_7/runtime/image_preprocess.h"
#include "families/minicpmv4_7/runtime/sampler.h"
#include "families/minicpmv4_7/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/language.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace trtmc::minicpmv4_7 {

// Request options of the images_text_to_text binding.
Span<const internal::ConfigField> request_config_fields();

// Pixels in [0, 255] of a three-channel image; Float32 views carry value / 255 as the CLI loads
// them. The sides and the byte size are checked before the copy is allocated.
std::vector<uint8_t> rgb_bytes(const internal::ImageView& image);

// What the host prepares for one request before any engine work: the sampler of the validated
// options, the preprocessed images, the prompt ids and the generation budget.
struct PreparedRequest {
    Sampler sampler;
    std::vector<ImageCrops> images;
    std::vector<int32_t> ids;
    int64_t max_new_tokens{0};
};

struct RequestContext {
    const nlohmann::json& prompts; // provider recipes keyed by enable_thinking ("false", "true")
    const ProcessorConfig& processor;
    const ITokenizer& tokenizer;
    int32_t capacity{0};
};

// Validates every option before it maps messages or preprocesses images, so an invalid request
// fails before any image work and before the caller runs the engines.
PreparedRequest prepare_request(const internal::ImagesTextToTextRequest& request,
                                internal::ConfigView supplied, const RequestContext& context);

} // namespace trtmc::minicpmv4_7

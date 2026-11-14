/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/language.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc {

struct GemmaImageConfig {
    int patch_size{0}, pooling_kernel{0}, soft_tokens{0}, hidden_size{0};
    int image_token{0}, begin_token{0}, end_token{0};
    bool merged_patches{false};

    bool valid_token_ids(int vocab_size) const {
        return image_token >= 0 && image_token < vocab_size && begin_token >= 0 &&
               begin_token < vocab_size && end_token >= 0 && end_token < vocab_size;
    }
};

struct GemmaImagePatches {
    std::vector<float> pixels;
    std::vector<int32_t> positions;
    int soft_tokens{0};
};

GemmaImagePatches gemma4_image_patches(const internal::ImageView&, const GemmaImageConfig&);
std::string gemma4_image_prompt(const internal::ImagesTextToTextRequest&,
                                const TextGenerationConfig&);

std::string gemma4_media_prompt(const internal::ImageAudioTextToTextRequest&,
                                const TextGenerationConfig&);

class GemmaVisionEncoder {
  public:
    GemmaVisionEncoder(std::unique_ptr<ITrtModule>, GemmaImageConfig);
    const GemmaImageConfig& config() const { return config_; }
    std::vector<float> encode(const GemmaImagePatches&);

  private:
    std::unique_ptr<ITrtModule> module_;
    GemmaImageConfig config_;
};

} // namespace trtmc

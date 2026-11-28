/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/muse_glimmer/runtime/media_request.h"
#include "families/muse_glimmer/runtime/tokenizer.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>

namespace trtmc::muse_glimmer {
struct MediaPrefix {
    std::vector<int32_t> ids;
    std::vector<int32_t> feature_at;
    std::vector<uint16_t> features;
};
class NativeVision {
  public:
    NativeVision(std::unique_ptr<ITrtModule> module, const nlohmann::json& metadata);
    MediaPrefix prepare(const MediaTurns&, const Tokenizer&, const std::string& bos,
                        const TextGenerationConfig&, int token_budget);

  private:
    std::unique_ptr<ITrtModule> module_;
    int patch_, side_, head_dim_, width_, max_patches_, image_limit_, video_limit_, image_id_,
        video_id_;
    float theta_;
    double fps_;
};
} // namespace trtmc::muse_glimmer

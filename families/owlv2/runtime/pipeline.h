/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/owlv2/runtime/image_preprocess_seam.h"
#include "families/owlv2/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Owlv2RuntimeConfig {
    Owlv2PreprocessConfig preprocess;
    std::int32_t max_text_len{16};
    std::int32_t num_patches{3600};
    std::int32_t grid{60};
    std::int32_t query_dim{512};
    std::int32_t pad_token_id{0};
    float score_threshold{0.1F};
};

// The image tower runs once per request and the text tower once per phrase.
// Everything that depends only on pixels - the boxes and the three per-patch
// class terms - comes out of the image engine, so adding a phrase costs one
// short text pass and a dot product, not another pass over the image.
class Owlv2Pipeline final : public internal::IModel, public internal::IImageTextToBoxes {
  public:
    Owlv2Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                  std::shared_ptr<ITokenizer> tokenizer, Owlv2RuntimeConfig config);

    const char* task() const noexcept override { return internal::IImageTextToBoxes::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::GroundedBoxesResult run(const internal::ImageTextToBoxesRequest& request,
                                      internal::ConfigView config) override;

  private:
    std::vector<float> embed_query(const std::string& phrase);

    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> text_;
    std::shared_ptr<ITokenizer> tokenizer_;
    Owlv2RuntimeConfig config_;
};

// Split a dotted prompt into phrases, the convention the other family on this
// task already uses. Exposed for the unit test.
std::vector<std::string> split_owlv2_phrases(std::string_view text);

} // namespace trtmc

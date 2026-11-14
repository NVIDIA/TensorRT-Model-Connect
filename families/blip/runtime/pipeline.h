/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/blip/runtime/image_preprocess.h"
#include "families/blip/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

namespace trtmc::blip {

struct RuntimeConfig {
    BlipPreprocessConfig preprocess;
    std::int32_t image_tokens{0}, vision_width{0}, vocab_size{0}, max_length{0};
    std::int32_t bos_token_id{0}, eos_token_id{0};
};

class Pipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config);
    const char* task() const noexcept override { return "images_text_to_text"; }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::ImagesTextToTextRequest&, internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> vision_, decoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    RuntimeConfig config_;
    std::vector<internal::ConfigField> fields_;
};

} // namespace trtmc::blip

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/layoutlm/runtime/document_qa.h"
#include "families/layoutlm/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::layoutlm {

struct RuntimeConfig {
    std::int32_t max_sequence_length{512};
    std::int32_t doc_stride{256};
    std::int32_t max_answer_length{15};
    // Largest coordinate, width or height the 2-D position tables cover.
    std::int32_t max_2d_position{1023};
    std::string ocr_language{"eng"};
    std::int32_t ocr_page_segmentation_mode{3};
};

// One document image and one question in, the extracted answer out: Tesseract
// reads the words and their boxes, the question and the words are tokenized
// into overlapping windows, the encoder scores answer starts and ends per
// window, and the best span over all windows is returned as text.
class DocumentQaPipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    DocumentQaPipeline(std::unique_ptr<ITrtModule> encoder,
                       std::unique_ptr<WordTokenizer> tokenizer, RuntimeConfig config,
                       std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::TextResult run(const internal::ImagesTextToTextRequest& request,
                             internal::ConfigView config) override;

  private:
    TokenizedWords tokenize(const std::vector<std::string>& words) const;
    void check_boxes(const Window& window) const;

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<WordTokenizer> tokenizer_;
    RuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc::layoutlm

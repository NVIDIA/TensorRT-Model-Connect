/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// EncoderPipeline: single-pass encoding and embedding.

#include "families/roberta/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class EncoderPipeline final : public IEmbedding, public IEncoding {
  public:
    const char* task() const noexcept override {
        if (mode_ == "embedding")
            return IEmbedding::kTask;
        return IEncoding::kTask;
    }

    EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string mode,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    EmbeddingResult embed(const std::string& text) override;
    EmbeddingResult encode(const std::string& text) override;

    // Token-ID-based encoding (for unit tests and internal callers).
    EmbeddingResult encode_ids(const std::vector<int32_t>& input_ids);

  private:
    std::unique_ptr<ITrtModule> encoder_;
    std::string mode_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc

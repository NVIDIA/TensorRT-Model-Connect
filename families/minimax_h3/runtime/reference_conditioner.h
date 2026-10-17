/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/qwen_reference.h"
#include "trtmc/runtime/trt_module.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

struct ResizedQwenImage {
    const uint8_t* rgb{nullptr};
    std::size_t bytes{0};
    int32_t height{0};
    int32_t width{0};
};

enum class ConditionerResidency { Staged, Resident };

using ReferenceModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string&, cudaStream_t)>;

struct ReferenceImageConditioning {
    QwenImagePresentation presentation;
    std::vector<float> embeddings; // Owned FP32 [presentation rows,5120].
};

// Executes the family-owned vision -> language plan composition through the
// native backend contract. Caller supplies exactly resized Qwen RGB images;
// original-reference LANCZOS setup and any processor smart-resize are separate.
// This component does not expose a public Ref2VA task or run a denoiser.
class ReferenceImageConditioner {
  public:
    ReferenceImageConditioner(ReferenceModuleLoader loader, const ITokenizer& tokenizer,
                              cudaStream_t stream, int32_t max_rows,
                              ConditionerResidency residency);

    ReferenceImageConditioning encode(const std::vector<ResizedQwenImage>& images,
                                      const std::string& prompt);

  private:
    ReferenceModuleLoader loader_;
    const ITokenizer& tokenizer_;
    cudaStream_t stream_;
    int32_t max_rows_;
    ConditionerResidency residency_;
    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> language_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3

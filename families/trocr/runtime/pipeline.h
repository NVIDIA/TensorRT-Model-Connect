/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/trocr/runtime/detokenizer.h"
#include "families/trocr/runtime/image_preprocess.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::trocr {

struct RuntimeConfig {
    PreprocessConfig preprocess;
    std::int32_t decoder_start_token_id{0};
    std::int32_t eos_token_id{0};
    std::vector<std::int32_t> special_token_ids;
    std::int32_t vocab_size{0};
    std::int32_t layers{0};
    std::int32_t heads{0};
    std::int32_t head_dim{0};
    std::int32_t encoder_sequence_length{0};
    // Decoder positions the cache holds: the start token plus generated tokens.
    std::int32_t max_sequence_length{0};
    DType work_dtype{DType::kFloat32};
};

// Device storage the two engines share. It is allocated before the engines
// load, which bind it by name, and it outlives them.
struct DecodeBuffers {
    DeviceTensor cross_key, cross_value;
    DeviceTensor cache_key, cache_value;
    DeviceTensor new_key, new_value;
};

DecodeBuffers allocate_buffers(const RuntimeConfig& config);

class TrocrPipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    TrocrPipeline(RuntimeConfig config, std::unique_ptr<DecodeBuffers> buffers,
                  std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                  ByteLevelDetokenizer detokenizer);

    const char* task() const noexcept override;
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::ImagesTextToTextRequest& request,
                   internal::ConfigView config) override;

  private:
    std::int32_t decode_step(std::int32_t token, std::int32_t position);

    RuntimeConfig config_;
    std::unique_ptr<DecodeBuffers> buffers_;
    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> decoder_;
    ByteLevelDetokenizer detokenizer_;
    // One request at a time owns the engines and the cache.
    std::mutex mutex_;
};

// The request contract, exposed for the CPU test: one user message (or
// several) holding exactly one image and only empty text parts. TrOCR reads
// the text in the image; it takes no instruction, history or tools.
const internal::ImageView& single_image(const internal::ImagesTextToTextRequest& request);

// The optional runtime parameters and their resolution against the capacity.
Span<const internal::ConfigField> config_fields() noexcept;
std::int32_t resolve_max_new_tokens(internal::ConfigView config, std::int32_t capacity);

// The first index of the largest logit, as torch.argmax picks it.
std::int32_t greedy_token(const float* logits, std::int32_t vocab_size);

} // namespace trtmc::trocr

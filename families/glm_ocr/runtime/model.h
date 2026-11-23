/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/glm_ocr/runtime/image.h"
#include "families/glm_ocr/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace trtmc::glm_ocr {

// runtime.json written by model.py.
struct RuntimeConfig {
    DType cache_dtype{DType::kFloat32};
    int max_sequence_length{0};
    int hidden_size{0};
    int num_layers{0};
    int num_key_value_heads{0};
    int head_dim{0};
    int vocab_size{0};
    float rope_theta{0.0F};
    std::array<int, 3> mrope_section{};
    int vision_out_hidden_size{0};
    ImageConfig image;
    std::int32_t image_token_id{0};
    std::unordered_set<std::int32_t> eos_token_ids;
};

RuntimeConfig parse_runtime_config(const std::vector<char>& json);

// Generation is greedy. The sampling controls are accepted only with values that keep it greedy
// (top_k 1, or temperature 0); anything else throws internal::ConfigError.
void check_greedy_controls(double temperature, std::int64_t top_k, double top_p);

// Owns a CUDA stream; destroying it releases the stream.
struct StreamDeleter {
    void operator()(cudaStream_t stream) const noexcept { cudaStreamDestroy(stream); }
};
using StreamHandle = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;

class GlmOcrModel final : public internal::IModel, public internal::IImagesTextToText {
  public:
    GlmOcrModel(const BundleReader& reader, IBackend& backend);
    ~GlmOcrModel() override;
    GlmOcrModel(const GlmOcrModel&) = delete;
    GlmOcrModel& operator=(const GlmOcrModel&) = delete;

    const char* task() const noexcept override { return internal::IImagesTextToText::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::ImagesTextToTextRequest& request,
                   internal::ConfigView config) override;

  private:
    std::vector<float> encode_image(const VisionInput& input);
    std::vector<float> prefill(const std::vector<float>& embeds, const std::vector<float>& cos,
                               const std::vector<float>& sin, std::size_t rows);
    std::vector<float> decode_step(std::int32_t token, std::int32_t position, std::size_t filled);
    void embed_row(std::int32_t token, float* out) const;
    std::int32_t argmax(const std::vector<float>& logits) const;

    RuntimeConfig config_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::vector<char> embedding_; // BF16 [vocab, hidden]
    StreamHandle stream_;         // declared before the engines and caches, so it outlives them
    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    std::vector<DeviceTensor> key_cache_;   // per layer [max_sequence_length, kv_heads, head_dim]
    std::vector<DeviceTensor> value_cache_; // per layer, same shape
    std::size_t cache_row_bytes_{0};
    std::mutex mutex_;
};

} // namespace trtmc::glm_ocr

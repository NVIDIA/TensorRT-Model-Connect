/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/diffusion_gemma/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc::diffusion_gemma {

struct RuntimeConfig {
    std::int32_t hidden_size{0};
    std::int32_t vocab_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_attention_heads{0};
    std::int32_t num_key_value_heads{0};
    std::int32_t num_global_key_value_heads{0};
    std::int32_t head_dim{0};
    std::int32_t global_head_dim{0};
    std::int32_t sliding_window{0};
    std::int32_t max_sequence_length{0};
    std::int32_t canvas_length{256};
    std::int32_t max_new_tokens{256};
    std::int32_t max_denoising_steps{48};
    std::int32_t stability_threshold{1};
    std::int32_t bos_token_id{2};
    float confidence_threshold{0.005F};
    float entropy_bound{0.1F};
    float t_min{0.4F};
    float t_max{0.8F};
    std::vector<std::int32_t> eos_token_ids;
    std::vector<std::int32_t> unsupported_token_ids;
    std::vector<std::string> layer_types;
    DType dtype{DType::kBFloat16};
};

// Completed canvases contain vocabulary IDs, not media operands. Only an
// external prompt uses the text bundle's unsupported-media sentinel policy.
void validate_token_ids(const std::vector<std::int32_t>& ids, const RuntimeConfig& config,
                        bool external_prompt);

// Own output bytes because backend TensorMap values borrow its staging buffers.
struct OwnedTensor {
    std::vector<std::uint8_t> bytes;
    std::vector<std::int64_t> shape;
    DType dtype{DType::kFloat32};

    OwnedTensor() = default;
    explicit OwnedTensor(const Tensor& tensor);
    Tensor view() const;
};

struct DecoderObservation {
    OwnedTensor logits;
    OwnedTensor self_conditioning;
};

using TraceCallback = std::function<void(const std::string&, const Tensor&)>;

// The encoder is causal and commits its keys and values. Denoising reads that
// prefix but does not commit the current canvas until the canvas is finalized.
class Pipeline final : public internal::IModel,
                       public internal::ITextContinuation,
                       public ITextGeneration {
  public:
    Pipeline(BundleReader reader, IBackend& backend, RuntimeConfig config,
             std::unique_ptr<ITokenizer> tokenizer);
    const char* task() const noexcept override { return "text_continuation"; }
    std::vector<internal::TaskInstance> task_bindings() override;
    std::int32_t default_max_new_tokens() const override { return config_.max_new_tokens; }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override;
    internal::TextResult run(const internal::TextContinuationRequest&,
                             internal::ConfigView) override;

    void reset();
    void encode(const std::vector<std::int32_t>& input_ids, const TraceCallback& trace = {});
    DecoderObservation denoise(const std::vector<std::int32_t>& canvas,
                               const OwnedTensor& self_conditioning, float temperature,
                               const TraceCallback& trace = {});
    OwnedTensor zero_conditioning(std::size_t length) const;
    const RuntimeConfig& config() const { return config_; }
    const std::vector<OwnedTensor>& keys() const { return keys_; }
    const std::vector<OwnedTensor>& values() const { return values_; }

  private:
    std::unique_ptr<ITrtModule> load(const std::string& name);
    ITrtModule& layer(std::size_t index);
    void validate_ids(const std::vector<std::int32_t>& ids, bool external_prompt = true) const;
    void encode_tokens(const std::vector<std::int32_t>& input_ids, bool external_prompt,
                       const TraceCallback& trace = {});
    OwnedTensor embed(const std::vector<std::int32_t>& ids, const OwnedTensor& self_conditioning,
                      bool decoder);
    OwnedTensor transform(OwnedTensor hidden, bool decoder, const TraceCallback& trace);
    TextResult generate_ids(const std::vector<std::int32_t>& prompt, std::int32_t max_new_tokens,
                            std::int64_t seed, std::int32_t max_steps);

    BundleReader reader_;
    IBackend& backend_;
    RuntimeConfig config_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITrtModule> embedding_;
    std::unique_ptr<ITrtModule> output_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::vector<OwnedTensor> keys_;
    std::vector<OwnedTensor> values_;
    std::int64_t prefix_length_{0};
};

std::vector<float> attention_mask(std::int64_t sequence_length, std::int64_t past_length,
                                  std::int64_t sliding_window, bool decoder);

struct CanvasSample {
    std::vector<std::int32_t> argmax;
    std::vector<std::int32_t> next_canvas;
    double mean_entropy{0.0};
};
CanvasSample sample_canvas(const float* raw_logits, std::size_t canvas_length,
                           std::size_t vocabulary_size, float temperature, double entropy_bound,
                           std::mt19937_64& random);

} // namespace trtmc::diffusion_gemma

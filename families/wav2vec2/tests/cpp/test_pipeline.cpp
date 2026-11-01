/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wav2vec2/runtime/pipeline.h"

#include <iostream>
#include <memory>
#include <optional>

namespace {
using namespace trtmc;

class CtcEngine final : public ITrtModule {
  public:
    TensorMap forward(const TensorMap&) override {
        return {{"logits", {logits_, {4, 4}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    float logits_[16]{0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0};
};
} // namespace

int test_pipeline_language() {
    using namespace trtmc;
    Wav2Vec2RuntimeConfig cfg;
    cfg.do_normalize = false;
    cfg.conv_kernel = {1};
    cfg.conv_stride = {1};
    cfg.max_samples = 4;
    cfg.vocabulary.tokens = {"<pad>", "|", "日", "本"};
    cfg.vocabulary.blank_token_id = 0;
    cfg.vocabulary.word_delimiter_token = "|";
    Wav2Vec2Pipeline pipeline(std::make_unique<CtcEngine>(), std::move(cfg));
    const float samples[]{0, 1, 0, -1};
    for (std::optional<std::string_view> language :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"ja"},
          std::optional<std::string_view>{"en"}}) {
        internal::SpeechTranscriptionRequest request;
        request.audio.samples = {samples, 4};
        request.source_language = language;
        try {
            const auto result = pipeline.run(request, {});
            if (result.text != "日本" || result.token_ids != std::vector<int32_t>{2, 3})
                return 1;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: checkpoint-defined language: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hubert/runtime/pipeline.h"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
using namespace trtmc;

// Four frames of logits over a four-token vocabulary, one per input sample
// (kernel 1, stride 1), so the pipeline's frame check holds for four samples.
class CtcEngine final : public ITrtModule {
  public:
    TensorMap forward(const TensorMap& inputs) override {
        const auto found = inputs.find("audio");
        seen_samples = found == inputs.end() ? -1 : found->second.numel();
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

    std::int64_t seen_samples{0};

  private:
    // Frames decode to: delimiter, A, delimiter, B -> "A B" with ids {1, 2, 1, 3}.
    float logits_[16]{0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 4};
};

HubertRuntimeConfig config() {
    HubertRuntimeConfig cfg;
    cfg.do_normalize = false;
    cfg.conv_kernel = {1};
    cfg.conv_stride = {1};
    cfg.min_samples = 4;
    cfg.max_samples = 4;
    cfg.vocabulary.tokens = {"<pad>", "|", "A", "B"};
    cfg.vocabulary.blank_token_id = 0;
    cfg.vocabulary.word_delimiter_token = "|";
    return cfg;
}

template <typename Error, typename Call>
bool refuses(Call&& call) {
    try {
        call();
    } catch (const Error&) {
        return true;
    }
    return false;
}

} // namespace

int test_pipeline_language() {
    // The checkpoint's CTC vocabulary decides the language; a source-language
    // hint neither changes nor rejects the transcript.
    HubertPipeline pipeline(std::make_unique<CtcEngine>(), config());
    const float samples[]{0, 1, 0, -1};
    for (std::optional<std::string_view> language :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"en"},
          std::optional<std::string_view>{"fr"}}) {
        internal::SpeechTranscriptionRequest request;
        request.audio.samples = {samples, 4};
        request.source_language = language;
        try {
            const auto result = pipeline.run(request, {});
            // The leading delimiter is stripped from the text, not from the ids.
            if (result.text != "A B" || result.token_ids != std::vector<int32_t>{1, 2, 1, 3}) {
                std::cerr << "FAIL: transcript " << result.text << '\n';
                return 1;
            }
        } catch (const std::exception& error) {
            std::cerr << "FAIL: checkpoint-defined language: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}

int test_pipeline_refusals() {
    int failed = 0;
    auto engine = std::make_unique<CtcEngine>();
    auto* raw = engine.get();
    HubertPipeline pipeline(std::move(engine), config());
    const float samples[]{0, 1, 0, -1, 2, 3, 4, 5};

    internal::SpeechTranscriptionRequest wrong_rate;
    wrong_rate.audio.samples = {samples, 4};
    wrong_rate.audio.sample_rate = 8000;
    if (!refuses<std::invalid_argument>([&] { pipeline.run(wrong_rate, {}); })) {
        std::cerr << "FAIL: an 8 kHz clip is not refused\n";
        ++failed;
    }

    internal::SpeechTranscriptionRequest too_long;
    too_long.audio.samples = {samples, 8};
    if (!refuses<std::invalid_argument>([&] { pipeline.run(too_long, {}); })) {
        std::cerr << "FAIL: a clip outside the engine's sample range is not refused\n";
        ++failed;
    }

    // Stereo: eight interleaved values are four frames, mixed down before the engine.
    internal::SpeechTranscriptionRequest stereo;
    stereo.audio.samples = {samples, 8};
    stereo.audio.channels = 2;
    stereo.audio.sample_rate = 16000;
    if (pipeline.run(stereo, {}).text != "A B" || raw->seen_samples != 4) {
        std::cerr << "FAIL: a stereo clip is not mixed to mono before the engine\n";
        ++failed;
    }

    internal::SpeechTranscriptionRequest empty;
    if (!refuses<std::invalid_argument>([&] { pipeline.run(empty, {}); })) {
        std::cerr << "FAIL: an empty clip is not refused\n";
        ++failed;
    }
    return failed;
}

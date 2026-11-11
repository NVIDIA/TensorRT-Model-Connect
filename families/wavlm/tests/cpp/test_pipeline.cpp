/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wavlm/runtime/pipeline.h"

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

namespace {
using namespace trtmc;

class CtcEngine final : public ITrtModule {
  public:
    TensorMap forward(const TensorMap&) override {
        ++calls;
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

    int calls{0};

  private:
    // Frames: 'a', '|', 'b', '[PAD]'.
    float logits_[16]{0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4};
};

WavLMRuntimeConfig config() {
    WavLMRuntimeConfig cfg;
    cfg.do_normalize = false;
    cfg.conv_kernel = {1};
    cfg.conv_stride = {1};
    cfg.min_samples = 4;
    cfg.max_samples = 4;
    cfg.vocabulary.tokens = {"|", "a", "b", "[PAD]"};
    cfg.vocabulary.blank_token_id = 3;
    cfg.vocabulary.word_delimiter_token = "|";
    return cfg;
}

bool refused(WavLMPipeline& pipeline, const internal::SpeechTranscriptionRequest& request) {
    try {
        static_cast<void>(pipeline.run(request, {}));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int test_pipeline_contract() {
    using namespace trtmc;
    auto engine = std::make_unique<CtcEngine>();
    auto* calls = &engine->calls;
    WavLMPipeline pipeline(std::move(engine), config());
    const float samples[]{0, 1, 0, -1};

    // The CTC vocabulary decides the output; a language hint does not change it.
    for (std::optional<std::string_view> language :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"en"},
          std::optional<std::string_view>{"fr"}}) {
        internal::SpeechTranscriptionRequest request;
        request.audio.samples = {samples, 4};
        request.source_language = language;
        try {
            const auto result = pipeline.run(request, {});
            if (result.text != "a b" || result.token_ids != std::vector<int32_t>{1, 0, 2}) {
                std::cerr << "FAIL: transcript '" << result.text << "'\n";
                return 1;
            }
        } catch (const std::exception& error) {
            std::cerr << "FAIL: checkpoint-defined vocabulary: " << error.what() << '\n';
            return 1;
        }
    }

    // A clip at another rate or outside the profile is refused before the engine runs.
    const int before = *calls;
    internal::SpeechTranscriptionRequest resampled;
    resampled.audio.samples = {samples, 4};
    resampled.audio.sample_rate = 8000;
    internal::SpeechTranscriptionRequest short_clip;
    short_clip.audio.samples = {samples, 3};
    if (!refused(pipeline, resampled) || !refused(pipeline, short_clip) || *calls != before) {
        std::cerr << "FAIL: an unusable clip reached the engine\n";
        return 1;
    }
    return 0;
}

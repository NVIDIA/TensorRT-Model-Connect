/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/pipeline.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class Module : public trtmc::ITrtModule {
  public:
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<std::int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

class Vision final : public Module {
  public:
    int calls{0};
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& pixels = inputs.at("pixel_values");
        check(pixels.shape == std::vector<std::int64_t>({1, 3, 4, 4}), "Image shape changed");
        const auto* positions =
            static_cast<const std::int32_t*>(inputs.at("vision_position_ids").data);
        check(std::vector<std::int32_t>(positions, positions + 4) ==
                  std::vector<std::int32_t>({0, 1, 4, 5}),
              "Image rotary positions changed");
        features_ = {*static_cast<const float*>(pixels.data), 2};
        return {{"image_features", {features_.data(), {1, 2}, trtmc::DType::kFloat32}}};
    }

  private:
    std::vector<float> features_;
};

class Decoder final : public Module {
  public:
    int prefills{0};
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto length = static_cast<std::int32_t>(inputs.at("inputs_embeds").shape[0]);
        const auto past = static_cast<std::int32_t>(inputs.at("past_key").shape[2]);
        const auto total = length + past;
        if (!past)
            ++prefills;
        const auto* positions = static_cast<const std::int32_t*>(inputs.at("position_ids").data);
        const auto* mask = static_cast<const float*>(inputs.at("attention_mask").data);
        for (int row = 0; row < length; ++row) {
            check(positions[row] == past + row, "Decoder positions lost absolute offset");
            for (int column = 0; column < total; ++column)
                check((mask[row * total + column] == 0) == (column <= past + row),
                      "Causal mask changed");
        }
        check_cache(inputs.at("past_key"), past, 1);
        check_cache(inputs.at("past_value"), past, -1);
        key_.resize(static_cast<std::size_t>(2 * total * 2));
        value_.resize(key_.size());
        for (int layer = 0; layer < 2; ++layer)
            for (int token = 0; token < total; ++token)
                for (int column = 0; column < 2; ++column) {
                    const auto offset =
                        static_cast<std::size_t>((layer * total + token) * 2 + column);
                    key_[offset] = static_cast<float>(100 * layer + 10 * token + column);
                    value_[offset] = -key_[offset];
                }
        logits_.assign(6, -1);
        const auto* embedding = static_cast<const float*>(inputs.at("inputs_embeds").data);
        logits_[past ? 5 : embedding[2] > 0.5F ? 0 : 1] = 1;
        return {{"logits", {logits_.data(), {1, 6}, trtmc::DType::kFloat32}},
                {"present_key", {key_.data(), {2, 1, total, 2}, trtmc::DType::kFloat32}},
                {"present_value", {value_.data(), {2, 1, total, 2}, trtmc::DType::kFloat32}}};
    }

  private:
    void check_cache(const trtmc::Tensor& tensor, int length, int sign) {
        const auto* values = static_cast<const float*>(tensor.data);
        for (int layer = 0; layer < 2; ++layer)
            for (int token = 0; token < length; ++token)
                for (int column = 0; column < 2; ++column)
                    check(values[(layer * length + token) * 2 + column] ==
                              sign * (100 * layer + 10 * token + column),
                          "KV cache ownership or append layout changed");
    }
    std::vector<float> key_, value_, logits_;
};

template <class Function>
void rejects(Function action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Unsupported request was accepted");
}

const char* tokenizer_json =
    R"({"model":{"type":"BPE","vocab":{"A":0,"B":1,"Ċ":2,"Ã":3,"©":4}},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":5,"content":"<eos>","special":true}]})";

} // namespace

int main() {
    try {
        using namespace trtmc;
        using namespace trtmc::lighton_ocr;
        Tokenizer tokenizer(tokenizer_json, 6);
        check(tokenizer.decode({3, 4, 5}) == "é", "Byte-level Unicode decoding changed");
        check(tokenizer.decode({3, 5}) == "�", "Truncated UTF-8 replacement changed");
        auto vision = std::make_unique<Vision>();
        auto decoder = std::make_unique<Decoder>();
        auto* observed_vision = vision.get();
        auto* observed_decoder = decoder.get();
        RuntimeConfig config;
        config.max_sequence_length = 8;
        config.max_new_tokens = 4;
        config.hidden_size = 2;
        config.vocab_size = 6;
        config.num_hidden_layers = 2;
        config.num_key_value_heads = 1;
        config.head_dim = 2;
        config.image_token_id = 4;
        config.eos_token_ids = {5};
        config.prompt_prefix_ids = config.prompt_suffix_ids = {2};
        config.image.patch_size = config.image.spatial_merge_size = 2;
        config.image.max_image_size = 8;
        config.image.mean = {0, 0, 0};
        config.image.standard_deviation = {1, 1, 1};
        Pipeline pipeline(std::move(vision), std::move(decoder), std::vector<float>(12, 0),
                          std::move(tokenizer), config);
        std::vector<unsigned char> pixels(4 * 4 * 3, 255);
        internal::ImageView image{
            pixels.data(), pixels.size(), 4, 4, 3, internal::ImageFormat::UInt8};
        std::vector<internal::ImagesTextPart> parts{image, internal::TextPartView{""}};
        const internal::ImagesTextMessage message{internal::MessageRole::User,
                                                  {parts.data(), parts.size()}};
        const internal::ImagesTextToTextRequest request{{&message, 1}};
        const auto first = pipeline.run(request, {});
        check(first.text == "A" && first.token_ids == std::vector<std::int32_t>({0, 5}),
              "Greedy EOS contract changed");
        std::fill(pixels.begin(), pixels.end(), 0);
        check(pipeline.run(request, {}).text == "B", "Second image did not replace vision input");
        std::fill(pixels.begin(), pixels.end(), 255);
        check(pipeline.run(request, {}).token_ids == first.token_ids && first.text == "A",
              "A/B/A request reset or result ownership failed");
        check(observed_decoder->prefills == 3, "Each request must start from an empty KV cache");
        const auto numeric = pipeline.observe(image, {0});
        check(numeric.cache_prefix_unchanged &&
                  numeric.prompt_ids == std::vector<std::int32_t>({2, 4, 2}) &&
                  numeric.logits.size() == 2,
              "Diagnostic framing or immutable cache prefix changed");
        const auto calls = observed_vision->calls;
        parts.back() = internal::TextPartView{"instruction"};
        rejects([&] { pipeline.run(request, {}); });
        parts.back() = image;
        rejects([&] { pipeline.run(request, {}); });
        parts.back() = internal::TextPartView{""};
        std::vector<internal::ConfigEntry> options{{"max_new_tokens", std::int64_t(8)}};
        rejects([&] { pipeline.run(request, {options.data(), options.size()}); });
        options = {{"lora_adapter_id", std::string_view("x")}};
        rejects([&] { pipeline.run(request, {options.data(), options.size()}); });
        options = {{"temperature", 0.5}};
        rejects([&] { pipeline.run(request, {options.data(), options.size()}); });
        check(observed_vision->calls == calls, "Invalid requests executed vision before rejection");
        check(pipeline.run(request, {}).text == "A", "Rejected request prevented recovery");
        std::cout << "LightOnOCR native lifecycle, cache, decoding and rejection checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

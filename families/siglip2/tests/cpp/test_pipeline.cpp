/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The pipeline's own rules, with fixture engines: what each engine receives, the padded
// text window, normalisation, the per-pair sigmoid and the refusals.

#include "families/siglip2/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& what) {
    if (!condition)
        throw std::runtime_error(what);
}

class FixtureModule final : public trtmc::ITrtModule {
  public:
    explicit FixtureModule(std::string output) : output_(std::move(output)) {}
    std::vector<float> values = {3.0F, 0.0F, 4.0F, 0.0F};
    trtmc::TensorMap seen;
    std::vector<std::vector<std::int32_t>> ids_seen;
    std::vector<std::vector<std::int64_t>> pixel_shapes;
    std::vector<std::vector<std::int64_t>> tap_shapes;

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        if (inputs.count("input_ids")) {
            const auto& ids = inputs.at("input_ids");
            const auto* data = static_cast<const std::int32_t*>(ids.data);
            ids_seen.emplace_back(data, data + ids.numel());
        } else {
            pixel_shapes.push_back(inputs.at("pixel_values").shape);
            tap_shapes.push_back(inputs.at("position_weights").shape);
        }
        trtmc::Tensor out;
        out.data = values.data();
        out.shape = {1, static_cast<std::int64_t>(values.size())};
        out.dtype = trtmc::DType::kFloat32;
        return {{output_, out}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 2; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::string output_;
};

class FixtureTokenizer final : public trtmc::siglip2::ITokenizer {
  public:
    std::vector<std::int32_t> ids{7, 8, 1};
    std::vector<std::int32_t> encode(const std::string&) const override { return ids; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    // A vocabulary of ids 0..15.
    std::string token_for_id(std::int32_t id) const override {
        return id >= 0 && id < 16 ? "t" : std::string{};
    }
};

struct Fixture {
    FixtureModule* vision;
    FixtureModule* text;
    std::shared_ptr<FixtureTokenizer> tokenizer;
    std::unique_ptr<trtmc::Siglip2Pipeline> pipeline;
};

Fixture make(trtmc::Siglip2RuntimeConfig config = {}) {
    config.max_text_len = 4;
    config.embedding_dim = 4;
    config.logit_scale = 10.0F;
    config.logit_bias = -2.0F;
    auto vision = std::make_unique<FixtureModule>("image_embeds");
    auto text = std::make_unique<FixtureModule>("text_embeds");
    auto tokenizer = std::make_shared<FixtureTokenizer>();
    Fixture fixture{vision.get(), text.get(), tokenizer, nullptr};
    fixture.pipeline = std::make_unique<trtmc::Siglip2Pipeline>(
        std::move(vision), std::move(text), tokenizer, config, "text_image_to_relevance");
    return fixture;
}

trtmc::internal::ImageView image_of(const std::vector<float>& pixels, std::uint32_t height,
                                    std::uint32_t width) {
    trtmc::internal::ImageView image;
    image.data = pixels.data();
    image.byte_size = pixels.size() * sizeof(float);
    image.height = height;
    image.width = width;
    image.channels = 3;
    image.format = trtmc::internal::ImageFormat::Float32;
    return image;
}

void test_text_window() {
    auto fixture = make();
    // Short: padded to the window with the pad id after the end token.
    require(fixture.pipeline->text_window("x") == std::vector<std::int32_t>{7, 8, 1, 0},
            "short prompt window");
    // Long: the head is kept and the end token re-appended at the last row.
    fixture.tokenizer->ids = {5, 6, 7, 8, 9, 1};
    require(fixture.pipeline->text_window("x") == std::vector<std::int32_t>{5, 6, 7, 1},
            "long prompt window");
    fixture.tokenizer->ids = {5, 6, 7, 1};
    require(fixture.pipeline->text_window("x") == std::vector<std::int32_t>{5, 6, 7, 1},
            "exact-length window");
}

void test_embeddings_and_relevance() {
    auto fixture = make();
    const std::vector<float> pixels(20 * 30 * 3, 0.5F);
    const auto image = image_of(pixels, 20, 30);

    trtmc::internal::ImageToEmbeddingRequest image_request{image};
    const auto image_result = fixture.pipeline->run(image_request, {});
    require(image_result.values.size() == 4 && std::abs(image_result.values[0] - 0.6F) < 1e-7F &&
                std::abs(image_result.values[2] - 0.8F) < 1e-7F,
            "image embedding is not L2-normalised");
    require(image_result.pooling == "attention_probe" && image_result.normalization == "l2" &&
                image_result.embedding_space == "siglip2",
            "image embedding metadata");
    // The image is resized to its own grid of 16-pixel patches within the 256 budget.
    const auto [height, width] = trtmc::siglip2_target_size(20, 30, 16, 256);
    const std::int64_t patches = (height / 16) * (width / 16);
    require(fixture.vision->pixel_shapes.back() == std::vector<std::int64_t>{patches, 768},
            "pixel_values shape");
    require(fixture.vision->tap_shapes.back() == std::vector<std::int64_t>{patches, 256},
            "position_weights shape");

    trtmc::internal::TextToEmbeddingRequest text_request{"x",
                                                         trtmc::internal::EmbeddingRole::Default};
    const auto text_result = fixture.pipeline->run(text_request, {});
    require(text_result.pooling == "last_token" && text_result.normalization == "l2",
            "text embedding metadata");
    require(fixture.text->ids_seen.back() == std::vector<std::int32_t>{7, 8, 1, 0},
            "text engine did not receive the padded window");

    trtmc::internal::TextImageToRelevanceRequest pair{"x", image};
    const auto relevance = fixture.pipeline->run(pair, {});
    // Both unit vectors are (0.6, 0, 0.8, 0): cosine 1, logit 10 * 1 - 2 = 8.
    require(relevance.kind == trtmc::internal::ScoreKind::Probability, "score kind");
    require(std::abs(relevance.score - 1.0F / (1.0F + std::exp(-8.0F))) < 1e-6F,
            "relevance is not sigmoid(scale * cosine + bias)");
}

void test_refusals() {
    auto fixture = make();
    bool thrown = false;
    try {
        trtmc::internal::TextToEmbeddingRequest request{"x", trtmc::internal::EmbeddingRole::Query};
        fixture.pipeline->run(request, {});
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    require(thrown, "a query role was accepted");

    thrown = false;
    try {
        const std::vector<float> gray(6, 0.5F);
        auto image = image_of(gray, 2, 3);
        image.channels = 1;
        trtmc::internal::ImageToEmbeddingRequest request{image};
        fixture.pipeline->run(request, {});
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    require(thrown, "a one-channel image was accepted");

    thrown = false;
    try {
        trtmc::Siglip2RuntimeConfig config;
        config.resample = 3;
        make(config);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    require(thrown, "a bicubic bundle was accepted");

    thrown = false;
    try {
        trtmc::Siglip2RuntimeConfig config;
        config.pad_token_id = 16;
        make(config);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    require(thrown, "a pad id outside the tokenizer's vocabulary was accepted");
}

} // namespace

void test_runtime_config() {
    const std::string valid =
        R"({"patch_size":16,"max_num_patches":256,"position_grid":16,"image_mean":[0.5,0.5,0.5],)"
        R"("image_std":[0.5,0.5,0.5],"resample":2,"rescale_factor":0.00392156862745098,)"
        R"("max_text_len":64,"pad_token_id":0,"embedding_dim":768,"logit_scale":117.3,)"
        R"("logit_bias":-12.9})";
    const auto parse = [](const std::string& text) {
        return trtmc::siglip2::parse_runtime_config(std::vector<char>(text.begin(), text.end()));
    };
    require(parse(valid).max_text_len == 64, "a valid runtime.json was refused");
    // A field of another JSON type or beyond its range is refused, not converted: 5.5 would
    // pad with token 5, 4294967552 would wrap to 256, true would count as 1 and 1e39 would
    // become an infinity. A zero deviation, refused by the builder, is refused here too.
    const std::vector<std::pair<std::string, std::string>> bad = {
        {"\"pad_token_id\":0", "\"pad_token_id\":5.5"},
        {"\"max_text_len\":64", "\"max_text_len\":64.0"},
        {"\"max_num_patches\":256", "\"max_num_patches\":4294967552"},
        {"\"patch_size\":16", "\"patch_size\":true"},
        {"\"logit_scale\":117.3", "\"logit_scale\":true"},
        {"\"image_mean\":[0.5,0.5,0.5]", "\"image_mean\":[true,0.5,0.5]"},
        {"\"image_mean\":[0.5,0.5,0.5]", "\"image_mean\":[1e39,0.5,0.5]"},
        {"\"logit_bias\":-12.9", "\"logit_bias\":1e39"},
        {"\"image_std\":[0.5,0.5,0.5]", "\"image_std\":[0.5,0,0.5]"},
    };
    for (const auto& [field, value] : bad) {
        std::string text = valid;
        text.replace(text.find(field), field.size(), value);
        bool thrown = false;
        try {
            parse(text);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        require(thrown, "runtime.json accepted " + value);
    }
}

int main() {
    try {
        test_text_window();
        test_embeddings_and_relevance();
        test_refusals();
        test_runtime_config();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

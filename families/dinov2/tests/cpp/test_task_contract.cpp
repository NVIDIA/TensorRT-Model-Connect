/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/pipeline.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        shape = image.shape;
        const auto* data = static_cast<const float*>(image.data);
        pixels.assign(data, data + image.numel());
        return {{"last_hidden_state", {hidden.data(), hidden_shape, dtype}}};
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
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override { return name == "last_hidden_state"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return dtype; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    std::vector<float> hidden;
    std::vector<int64_t> hidden_shape;
    std::vector<float> pixels;
    std::vector<int64_t> shape;
    trtmc::DType dtype{trtmc::DType::kFloat32};
};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

using trtmc::internal::ImageFeatureTokenRole;

// A 4x4 crop of patch 2 gives a 2x2 grid; with two registers there are 7 tokens.
constexpr int64_t kTokens = 7;
constexpr int64_t kHidden = 3;

trtmc::Dinov2RuntimeConfig config(int32_t registers = 2) {
    trtmc::Dinov2RuntimeConfig result;
    result.preprocess = {4, 4, 4, {0.25F, 0.5F, 0.75F}, {0.5F, 0.25F, 0.125F}};
    result.patch_size = 2;
    result.hidden_size = static_cast<int32_t>(kHidden);
    result.num_register_tokens = registers;
    return result;
}

std::unique_ptr<RecordingModule> module(int64_t tokens = kTokens) {
    auto result = std::make_unique<RecordingModule>();
    result->hidden.resize(static_cast<std::size_t>(tokens * kHidden));
    for (std::size_t index = 0; index < result->hidden.size(); ++index)
        result->hidden[index] = static_cast<float>(index) - 4.5F;
    result->hidden_shape = {1, tokens, kHidden};
    return result;
}

trtmc::internal::ImageToTokenAndPooledFeaturesRequest request(const std::vector<float>& pixels,
                                                              uint32_t height, uint32_t width) {
    return {{pixels.data(), pixels.size() * sizeof(float), height, width, 3,
             trtmc::internal::ImageFormat::Float32}};
}

void test_binding_tokens_and_owned_features() {
    trtmc::internal::ImageTokenAndPooledFeaturesResult result;
    std::vector<float> expected;
    {
        auto engine = module();
        auto* recording = engine.get();
        expected = recording->hidden;
        trtmc::Dinov2FeaturePipeline model(std::move(engine), config());
        const auto bindings = model.task_bindings();
        require(bindings.size() == 1 &&
                    bindings[0].key.id == "image_to_token_and_pooled_features" &&
                    bindings[0].key.major == 1 && bindings[0].key.minor == 0,
                "family must publish exactly its implemented semantic task");
        require(bindings[0].fields.empty(), "family has no runtime config options");
        require(bindings[0].implementation ==
                    static_cast<trtmc::internal::IImageToTokenAndPooledFeatures*>(&model),
                "bind must preserve the interface subobject address");
        require(std::string(model.task()) == "image_to_token_and_pooled_features",
                "bundle primary task must match the binding");

        // An 8x4 (HxW) image resizes to 8x4 and crops rows 2..5 at the configured 4x4 crop.
        const std::vector<float> pixels(8 * 4 * 3, 0.75F);
        result = static_cast<trtmc::internal::IImageToTokenAndPooledFeatures*>(
                     bindings[0].implementation)
                     ->run(request(pixels, 8, 4), {});
        require(recording->shape == std::vector<int64_t>({1, 3, 4, 4}),
                "preprocessing must produce the engine NCHW crop");
        // 0.75 is stored as byte 191, exactly as the reference processor sees it.
        const float value = static_cast<float>(191) / 255.0F;
        require(recording->pixels.front() == (value - 0.25F) / 0.5F &&
                    recording->pixels.back() == (value - 0.75F) / 0.125F,
                "preprocessing must preserve checkpoint normalization");
        recording->hidden.assign(recording->hidden.size(), 0.0F);
        model.run(request(pixels, 8, 4), {});
    }
    const auto& tokens = result.tokens;
    require(tokens.features.values == expected && tokens.features.rows == kTokens &&
                tokens.features.columns == kHidden,
            "result must own every token row across another call and model destruction");
    require(tokens.grid_rows == 2 && tokens.grid_columns == 2, "grid is the crop patch grid");
    require(tokens.tokens.size() == kTokens &&
                tokens.tokens[0].role == ImageFeatureTokenRole::Class &&
                tokens.tokens[1].role == ImageFeatureTokenRole::Register &&
                tokens.tokens[2].role == ImageFeatureTokenRole::Register,
            "class then register rows precede patches in engine order");
    const auto& last = tokens.tokens.back();
    require(last.role == ImageFeatureTokenRole::Patch && last.grid_row == 1 &&
                last.grid_column == 1,
            "patch rows are row-major over the grid");
    const auto& first = tokens.tokens[3];
    require(first.x_min == 0.0F && first.x_max == 0.5F && first.y_min == 0.25F &&
                first.y_max == 0.5F,
            "patch footprint maps through the center crop to normalized source coordinates");
    require(result.pooled.values ==
                std::vector<float>(expected.begin(), expected.begin() + kHidden),
            "pooled output is the final CLS row");
    require(result.pooled.pooling == "cls" && result.pooled.normalization == "none",
            "pooling metadata states the actual reduction");
}

void test_uint8_matches_float_input() {
    auto first = module();
    auto* first_recording = first.get();
    trtmc::Dinov2FeaturePipeline float_model(std::move(first), config());
    auto second = module();
    auto* second_recording = second.get();
    trtmc::Dinov2FeaturePipeline byte_model(std::move(second), config());
    std::vector<uint8_t> bytes(6 * 5 * 3);
    std::vector<float> floats(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<uint8_t>((index * 37U) % 256U);
        floats[index] = static_cast<float>(bytes[index]) / 255.0F;
    }
    float_model.run(request(floats, 6, 5), {});
    byte_model.run({{bytes.data(), bytes.size(), 6, 5, 3, trtmc::internal::ImageFormat::UInt8}},
                   {});
    require(first_recording->pixels == second_recording->pixels,
            "float [0,1] input must preprocess exactly like its 8-bit source");
}

void test_invalid_inputs_and_outputs() {
    auto engine = module();
    auto* recording = engine.get();
    trtmc::Dinov2FeaturePipeline model(std::move(engine), config());
    const std::vector<float> pixels(4 * 4 * 3, 0.5F);
    const auto input = request(pixels, 4, 4);
    model.run(input, {});

    const trtmc::internal::ConfigEntry unsupported{"pooling", std::string_view("mean")};
    rejects<trtmc::internal::ConfigError>([&] { model.run(input, {&unsupported, 1}); },
                                          "unsupported config must fail");
    auto invalid = input;
    invalid.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject truncated image");
    invalid = input;
    invalid.image.channels = 4;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject non-RGB image");
    std::vector<float> nan_pixels = pixels;
    nan_pixels[5] = std::numeric_limits<float>::quiet_NaN();
    rejects<std::invalid_argument>([&] { model.run(request(nan_pixels, 4, 4), {}); },
                                   "reject non-finite pixels");
    require(recording->calls == 1, "invalid input and config must fail before engine execution");

    recording->hidden_shape = {1, kTokens - 1, kHidden};
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject a short token output");
    recording->hidden_shape = {1, kTokens, kHidden};
    recording->dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject wrong output dtype");

    rejects<std::runtime_error>(
        [] {
            auto bad = config();
            bad.patch_size = 3;
            trtmc::Dinov2FeaturePipeline unused(module(), bad);
        },
        "reject a crop that is not a whole patch grid");
}

} // namespace

int main() {
    try {
        test_binding_tokens_and_owned_features();
        test_uint8_matches_float_input();
        test_invalid_inputs_and_outputs();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    std::cout << "dinov2 task contract tests passed\n";
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mask2former/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kSize = 4;
constexpr int kClasses = 3;

// Scores that select class 1 on the left half and class 2 on the right half of a 4x4 map.
std::vector<float> halves() {
    std::vector<float> scores(kClasses * kSize * kSize, 0.0F);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            scores[(1 * kSize + y) * kSize + x] = x < kSize / 2 ? 2.0F : 0.5F;
            scores[(2 * kSize + y) * kSize + x] = x < kSize / 2 ? 0.5F : 2.0F;
        }
    }
    return scores;
}

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        shape = image.shape;
        const auto* data = static_cast<const float*>(image.data);
        pixels.assign(data, data + image.numel());
        return {{"semantic_scores", {scores.data(), output_shape, output_dtype}}};
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
    bool has_output(const std::string& name) const override { return name == "semantic_scores"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return name == "pixel_values" ? std::vector<int64_t>{1, 3, kSize, kSize} : engine_scores;
    }
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
    std::vector<float> scores = halves();
    std::vector<int64_t> engine_scores{1, kClasses, kSize, kSize};
    std::vector<int64_t> output_shape{1, kClasses, kSize, kSize};
    trtmc::DType output_dtype{trtmc::DType::kFloat32};
    std::vector<float> pixels;
    std::vector<int64_t> shape;
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

trtmc::mask2former::PreprocessConfig preprocessing() {
    trtmc::mask2former::PreprocessConfig config;
    config.height = kSize;
    config.width = kSize;
    config.rescale_factor = 1.0 / 255.0;
    config.mean = {0.5F, 0.25F, 0.0F};
    config.std = {0.5F, 0.25F, 1.0F};
    return config;
}

trtmc::mask2former::SemanticHead head() {
    return {kSize, kSize, {"first", "second", "third"}};
}

trtmc::internal::ImageToSemanticSegmentationRequest
request(const std::vector<float>& pixels, std::uint32_t height, std::uint32_t width) {
    return {{pixels.data(), pixels.size() * sizeof(float), height, width, 3,
             trtmc::internal::ImageFormat::Float32}};
}

void test_binding_and_owned_result() {
    trtmc::internal::SemanticSegmentationResult result;
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module),
                                                                  preprocessing(), head());
        const auto bindings = model.task_bindings();
        require(bindings.size() == 1 && bindings[0].key.id == "image_to_semantic_segmentation" &&
                    bindings[0].key.major == 1 && bindings[0].key.minor == 0,
                "family must publish exactly its implemented semantic task");
        require(bindings[0].fields.empty(), "family has no runtime config options");
        require(bindings[0].implementation ==
                    static_cast<trtmc::internal::IImageToSemanticSegmentation*>(&model),
                "bind must preserve the interface subobject address");
        require(std::string(model.task()) == "image_to_semantic_segmentation",
                "bundle primary task must match the binding");

        const std::vector<float> pixels(kSize * kSize * 3, 0.75F);
        result =
            static_cast<trtmc::internal::IImageToSemanticSegmentation*>(bindings[0].implementation)
                ->run(request(pixels, kSize, kSize), {});
        require(recording->shape == std::vector<int64_t>({1, 3, kSize, kSize}),
                "preprocessing must feed the engine one NCHW image");
        // 0.75 quantizes to 191; float32(191 / 255) normalized per channel.
        const float rescaled = static_cast<float>(191.0 * (1.0 / 255.0));
        const float expected[3] = {(rescaled - 0.5F) / 0.5F, (rescaled - 0.25F) / 0.25F,
                                   (rescaled - 0.0F) / 1.0F};
        for (int channel = 0; channel < 3; ++channel)
            for (int index = 0; index < kSize * kSize; ++index)
                require(recording->pixels[channel * kSize * kSize + index] == expected[channel],
                        "preprocessing must rescale and normalize every channel plane");
        recording->scores.assign(recording->scores.size(), 0.0F);
        model.run(request(pixels, kSize, kSize), {});
    }
    require(result.height == kSize && result.width == kSize, "labels keep the image size");
    for (int y = 0; y < kSize; ++y)
        for (int x = 0; x < kSize; ++x)
            require(result.labels[y * kSize + x] == (x < kSize / 2 ? 1 : 2),
                    "labels must own the argmax across later calls and model destruction");
    require(result.class_ids == std::vector<std::int32_t>({0, 1, 2}) &&
                result.class_names == std::vector<std::string>({"first", "second", "third"}),
            "every class ID is named in order");
    require(result.vocabulary_id.empty() && !result.ignore_label && !result.background_label,
            "no invented vocabulary, ignore or background label");
    require(result.class_scores.empty() && result.score_height == 0 && result.score_width == 0,
            "the family does not publish its unnormalized score maps");
}

void test_original_resolution_and_uint8_input() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module), preprocessing(),
                                                              head());
    const std::vector<float> pixels(2 * 8 * 3, 0.5F);
    const auto result = model.run(request(pixels, 2, 8), {});
    require(result.height == 2 && result.width == 8 && result.labels.size() == 16,
            "labels must come back at the original image resolution");
    for (int x = 0; x < 8; ++x)
        require(result.labels[x] == (x < 4 ? 1 : 2) && result.labels[8 + x] == result.labels[x],
                "the score map is resized to the original image before the argmax");
    const auto float_pixels = recording->pixels;

    const std::vector<std::uint8_t> bytes(2 * 8 * 3, 128);
    trtmc::internal::ImageToSemanticSegmentationRequest uint8{
        {bytes.data(), bytes.size(), 2, 8, 3, trtmc::internal::ImageFormat::UInt8}};
    const auto from_bytes = model.run(uint8, {});
    require(from_bytes.labels == result.labels && recording->pixels == float_pixels,
            "0.5 quantizes to 128: float32 and 8-bit inputs must agree");
}

void test_scores_are_resized_before_the_argmax() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module), preprocessing(),
                                                              head());
    // Class 1 scores 100 in the left half and class 2 scores 1 in the right half. Resizing the
    // scores from width 4 to width 8 keeps class 1 at column 4 (0.25 * 100 against 0.75 * 1);
    // resizing the argmax map instead would take column 2's class 2 there.
    recording->scores.assign(recording->scores.size(), 0.0F);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            recording->scores[(1 * kSize + y) * kSize + x] = x < kSize / 2 ? 100.0F : 0.0F;
            recording->scores[(2 * kSize + y) * kSize + x] = x < kSize / 2 ? 0.0F : 1.0F;
        }
    }
    const std::vector<float> pixels(2 * 8 * 3, 0.5F);
    const auto result = model.run(request(pixels, 2, 8), {});
    const std::vector<std::int32_t> row{1, 1, 1, 1, 1, 2, 2, 2};
    require(result.labels.size() == 16, "labels come back at the image resolution");
    for (int y = 0; y < 2; ++y)
        require(std::vector<std::int32_t>(result.labels.begin() + y * 8,
                                          result.labels.begin() + (y + 1) * 8) == row,
                "the scores are resized to the image before the argmax");
}

void test_invalid_inputs_and_outputs() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module), preprocessing(),
                                                              head());
    std::vector<float> pixels(kSize * kSize * 3, 0.25F);
    const auto input = request(pixels, kSize, kSize);
    const trtmc::internal::ConfigEntry unsupported{"threshold", 0.5};
    rejects<trtmc::internal::ConfigError>([&] { model.run(input, {&unsupported, 1}); },
                                          "unsupported config must fail");
    auto invalid = input;
    invalid.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject truncated image");
    invalid = input;
    invalid.image.channels = 4;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject non-RGB image");
    invalid = input;
    invalid.image.width = 0;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject empty image");
    pixels[5] = 1.5F;
    rejects<std::invalid_argument>([&] { model.run(input, {}); }, "reject values above one");
    pixels[5] = std::nanf("");
    rejects<std::invalid_argument>([&] { model.run(input, {}); }, "reject NaN pixels");
    pixels[5] = 0.25F;
    require(recording->calls == 0, "invalid input and config must fail before engine execution");

    recording->output_shape = {1, kClasses, kSize, kSize - 1};
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject incomplete scores");
    recording->output_shape = {1, kClasses, kSize, kSize};
    recording->output_dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject wrong output dtype");
    recording->output_dtype = trtmc::DType::kFloat32;
    recording->scores[0] = std::numeric_limits<float>::infinity();
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject nonfinite scores");
    // A 1x1 image reads only score rows and columns 1 and 2; the score at (0, 0) must still count.
    const std::vector<float> one_pixel(3, 0.25F);
    recording->scores[0] = std::nanf("");
    rejects<std::runtime_error>([&] { model.run(request(one_pixel, 1, 1), {}); },
                                "reject nonfinite scores that the resize does not sample");
    recording->scores[0] = 0.0F;
    require(model.run(request(one_pixel, 1, 1), {}).labels.size() == 1,
            "the same request with finite scores succeeds");
}

void test_engine_must_match_runtime_metadata() {
    auto module = std::make_unique<RecordingModule>();
    module->engine_scores = {1, kClasses + 1, kSize, kSize};
    rejects<std::runtime_error>(
        [&] {
            trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module),
                                                                      preprocessing(), head());
        },
        "an engine with another class count must be rejected at load");
    auto zero_std = preprocessing();
    zero_std.std[1] = 0.0F;
    rejects<std::runtime_error>(
        [&] {
            trtmc::mask2former::Mask2FormerSemanticSegmentation model(
                std::make_unique<RecordingModule>(), zero_std, head());
        },
        "zero image_std must be rejected at load");
}

void test_normalization_must_stay_finite() {
    // Finite and nonzero, but 1 / 1e-40 overflows float32.
    auto tiny_std = preprocessing();
    tiny_std.std[2] = 1e-40F;
    // 255 * 1e300 does not fit float.
    auto huge_rescale = preprocessing();
    huge_rescale.rescale_factor = 1e300;
    // An infinite std turns every value into a finite zero.
    auto infinite_std = preprocessing();
    infinite_std.std[0] = std::numeric_limits<float>::infinity();
    const std::vector<std::uint8_t> pixel(3, 255);
    for (const auto& config : {tiny_std, huge_rescale, infinite_std}) {
        auto module = std::make_unique<RecordingModule>();
        rejects<std::runtime_error>(
            [&] {
                trtmc::mask2former::Mask2FormerSemanticSegmentation model(std::move(module), config,
                                                                          head());
            },
            "a normalization that overflows float32 must be rejected at load");
        rejects<std::invalid_argument>(
            [&] { trtmc::mask2former::preprocess(pixel.data(), 1, 1, config); },
            "preprocessing must refuse a normalization that overflows float32");
    }
}

std::vector<char> characters(const std::string& text) {
    return {text.begin(), text.end()};
}

void test_runtime_json_is_read_strictly() {
    const std::string valid =
        R"({"input_height": 4, "input_width": 4, "score_height": 4, "score_width": 4, )"
        R"("num_classes": 3, "rescale_factor": 0.00392156862745098, "image_mean": [0.5, 0.25, 0.0], )"
        R"("image_std": [0.5, 0.25, 1.0], "labels": ["first", "second", "third"]})";
    const auto metadata = trtmc::mask2former::parse_runtime_json(characters(valid));
    require(metadata.preprocess.height == 4 && metadata.preprocess.width == 4 &&
                metadata.preprocess.rescale_factor == 0.00392156862745098 &&
                metadata.preprocess.mean[1] == 0.25F && metadata.preprocess.std[2] == 1.0F &&
                metadata.head.score_height == 4 && metadata.head.score_width == 4 &&
                metadata.head.labels == std::vector<std::string>({"first", "second", "third"}),
            "a valid runtime.json is read in full");
    const std::pair<std::string, std::string> invalid[] = {
        {R"("input_height": 4)", R"("input_height": 4.5)"},
        // 2^32 + 4 wraps to 4 as int32.
        {R"("input_height": 4)", R"("input_height": 4294967300)"},
        {R"("input_width": 4)", R"("input_width": -4)"},
        {R"("score_width": 4)", R"("score_width": null)"},
        {R"("num_classes": 3)", R"("num_classes": 3.0)"},
        {R"("num_classes": 3)", R"("num_classes": 4)"},
        {R"("input_height": 4)", R"("input_height": 4, "input_height": 8)"},
        {R"("image_std": [0.5, 0.25, 1.0])", R"("image_std": [0.5, 0.25, 1e300])"},
        {R"("image_mean": [0.5, 0.25, 0.0])", R"("image_mean": [0.5, 0.25])"},
        {R"("rescale_factor": 0.00392156862745098)", R"("rescale_factor": "0.0039")"},
        {R"("third"])", R"(3])"},
    };
    for (const auto& change : invalid) {
        auto text = valid;
        const auto position = text.find(change.first);
        require(position != std::string::npos, "every invalid runtime.json changes the valid one");
        text.replace(position, change.first.size(), change.second);
        rejects<std::runtime_error>(
            [&] { trtmc::mask2former::parse_runtime_json(characters(text)); },
            change.second.c_str());
    }
}

} // namespace

int main() {
    try {
        test_binding_and_owned_result();
        test_original_resolution_and_uint8_input();
        test_scores_are_resized_before_the_argmax();
        test_invalid_inputs_and_outputs();
        test_engine_must_match_runtime_metadata();
        test_normalization_must_stay_finite();
        test_runtime_json_is_read_strictly();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vitpose/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// A tiny model: an 8 x 6 crop, 8 x 6 heatmaps, three keypoints and a 3-tap DARK kernel.
constexpr int kHeight = 8;
constexpr int kWidth = 6;
constexpr int kKeypoints = 3;
constexpr int kPlane = kHeight * kWidth;

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        shape = image.shape;
        const auto* data = static_cast<const float*>(image.data);
        pixels.assign(data, data + image.numel());
        return {{"heatmaps", {heatmaps.data(), output_shape, output_dtype}}};
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
    bool has_output(const std::string& name) const override { return name == "heatmaps"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return name == "pixel_values" ? engine_input : engine_output;
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
    std::vector<float> heatmaps = std::vector<float>(kKeypoints * kPlane, -1.0F);
    std::vector<int64_t> engine_input{1, 3, kHeight, kWidth};
    std::vector<int64_t> engine_output{1, kKeypoints, kHeight, kWidth};
    std::vector<int64_t> output_shape{1, kKeypoints, kHeight, kWidth};
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

trtmc::vitpose::RuntimeMetadata metadata() {
    trtmc::vitpose::RuntimeMetadata result;
    result.preprocess.height = kHeight;
    result.preprocess.width = kWidth;
    result.preprocess.padding_factor = 1.25;
    result.preprocess.rescale_factor = 0.00392156862745098;
    result.preprocess.mean = {0.485F, 0.456F, 0.406F};
    result.preprocess.std = {0.229F, 0.224F, 0.225F};
    result.heatmap_height = kHeight;
    result.heatmap_width = kWidth;
    result.dark_kernel = 3;
    result.labels = {"Nose", "L_Eye", "R_Eye"};
    return result;
}

trtmc::internal::ImageTextToPointsRequest request(const std::vector<float>& pixels,
                                                  std::uint32_t height, std::uint32_t width,
                                                  std::string_view text = "person") {
    return {{pixels.data(), pixels.size() * sizeof(float), height, width, 3,
             trtmc::internal::ImageFormat::Float32},
            text};
}

// One symmetric peak: the DARK gradient is zero there, so the keypoint is the cell itself.
void peak(std::vector<float>& heatmaps, int keypoint, int x, int y, float value) {
    for (int index = 0; index < kPlane; ++index)
        heatmaps[keypoint * kPlane + index] = 0.01F;
    heatmaps[keypoint * kPlane + y * kWidth + x] = value;
}

// transform_preds of an integer heatmap cell through the crop of `box`.
std::pair<float, float> image_point(const trtmc::vitpose::PersonBox& box, int x, int y) {
    const auto geometry = trtmc::vitpose::crop_geometry(box, metadata().preprocess);
    const float side_x = geometry.scale[0] * 200.0F;
    const float side_y = geometry.scale[1] * 200.0F;
    return {static_cast<float>(x) * (side_x / static_cast<float>(kWidth - 1)) + geometry.center[0] -
                side_x * 0.5F,
            static_cast<float>(y) * (side_y / static_cast<float>(kHeight - 1)) +
                geometry.center[1] - side_y * 0.5F};
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 1e-4F * std::max(1.0F, std::fabs(expected));
}

void test_binding_and_located_keypoints() {
    trtmc::internal::GroundedPointsResult result;
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        trtmc::vitpose::VitPosePoints model(std::move(module), metadata());
        const auto bindings = model.task_bindings();
        require(bindings.size() == 1 && bindings[0].key.id == "image_text_to_points" &&
                    bindings[0].key.major == 1 && bindings[0].key.minor == 0,
                "family must publish exactly its implemented semantic task");
        require(bindings[0].fields.size() == 1 && bindings[0].fields[0].name == "person_box" &&
                    bindings[0].fields[0].kind == trtmc::internal::ConfigKind::F64List &&
                    !bindings[0].fields[0].default_value,
                "the person box is the one config field, defaulting to the whole image");
        require(bindings[0].implementation ==
                    static_cast<trtmc::internal::IImageTextToPoints*>(&model),
                "bind must preserve the interface subobject address");
        require(std::string(model.task()) == "image_text_to_points",
                "bundle primary task must match the binding");

        peak(recording->heatmaps, 0, 2, 5, 0.9F);
        peak(recording->heatmaps, 2, 4, 1, 0.3F); // keypoint 1 stays negative everywhere
        const std::vector<float> pixels(24 * 18 * 3, 0.5F);
        result = static_cast<trtmc::internal::IImageTextToPoints*>(bindings[0].implementation)
                     ->run(request(pixels, 24, 18), {});
        require(recording->calls == 1 &&
                    recording->shape == std::vector<int64_t>({1, 3, kHeight, kWidth}),
                "preprocessing must feed the engine one NCHW crop");
        recording->heatmaps.assign(recording->heatmaps.size(), 0.5F);
    }
    require(result.points.size() == 2, "a keypoint without a positive heatmap value is left out");
    require(result.points[0].label == "Nose" && result.points[1].label == "R_Eye",
            "located keypoints keep their labels and order");
    require(result.points[0].confidence == 0.9F && result.points[1].confidence == 0.3F,
            "the confidence is the heatmap maximum");
    const trtmc::vitpose::PersonBox whole{0.0, 0.0, 18.0, 24.0};
    const auto nose = image_point(whole, 2, 5);
    const auto eye = image_point(whole, 4, 1);
    require(
        near(result.points[0].point.x, nose.first) && near(result.points[0].point.y, nose.second) &&
            near(result.points[1].point.x, eye.first) && near(result.points[1].point.y, eye.second),
        "keypoints map back through the crop of the whole image");
    require(result.raw_response.empty() && result.parse_complete,
            "no generated text: the result is complete as returned");
}

void test_query_box_and_8bit_input() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::vitpose::VitPosePoints model(std::move(module), metadata());
    peak(recording->heatmaps, 0, 1, 1, 0.5F);
    std::vector<float> pixels(24 * 18 * 3);
    std::vector<std::uint8_t> bytes(pixels.size());
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(index * 37 % 256);
        pixels[index] = static_cast<float>(bytes[index]) / 255.0F;
    }
    const auto from_text = model.run(request(pixels, 24, 18), {});
    const auto first_pixels = recording->pixels;
    const auto empty_text = model.run(request(pixels, 24, 18, ""), {});
    require(empty_text.points.size() == from_text.points.size() &&
                empty_text.points[0].point.x == from_text.points[0].point.x &&
                recording->pixels == first_pixels,
            "empty text is the person query");
    rejects<std::invalid_argument>([&] { model.run(request(pixels, 24, 18, "dog"), {}); },
                                   "another query must be rejected");
    rejects<std::invalid_argument>([&] { model.run(request(pixels, 24, 18, " person"), {}); },
                                   "the query is matched exactly");

    trtmc::internal::ImageTextToPointsRequest uint8{
        {bytes.data(), bytes.size(), 24, 18, 3, trtmc::internal::ImageFormat::UInt8}, "person"};
    model.run(uint8, {});
    require(recording->pixels == first_pixels,
            "8-bit input and its value / 255 float32 decode must give the same crop");

    const double whole[] = {0.0, 0.0, 18.0, 24.0};
    const trtmc::internal::ConfigEntry whole_box{"person_box", trtmc::Span<const double>(whole, 4)};
    const auto boxed = model.run(request(pixels, 24, 18), {&whole_box, 1});
    require(recording->pixels == first_pixels &&
                boxed.points[0].point.x == from_text.points[0].point.x,
            "a person box of the whole image is the default");
    const double inner[] = {3.0, 4.0, 9.5, 20.0};
    const trtmc::internal::ConfigEntry inner_box{"person_box", trtmc::Span<const double>(inner, 4)};
    const auto cropped = model.run(request(pixels, 24, 18), {&inner_box, 1});
    require(recording->pixels != first_pixels, "a person box changes the crop");
    const auto expected = image_point({3.0, 4.0, 6.5, 16.0}, 1, 1);
    require(near(cropped.points[0].point.x, expected.first) &&
                near(cropped.points[0].point.y, expected.second),
            "keypoints map back through the crop of the person box");
}

void test_invalid_boxes() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::vitpose::VitPosePoints model(std::move(module), metadata());
    const std::vector<float> pixels(24 * 18 * 3, 0.25F);
    const double infinite = std::numeric_limits<double>::infinity();
    const std::vector<std::vector<double>> invalid = {
        {0.0, 0.0, 18.0},                // three values
        {0.0, 0.0, 18.0, 24.0, 1.0},     // five values
        {std::nan(""), 0.0, 18.0, 24.0}, // not finite
        {0.0, 0.0, infinite, 24.0},
        {-0.5, 0.0, 18.0, 24.0}, // outside the image
        {0.0, 0.0, 18.5, 24.0},
        {0.0, 0.0, 18.0, 24.25},
        {5.0, 5.0, 5.5, 20.0}, // narrower than one pixel
        {5.0, 5.0, 10.0, 5.9},
        {10.0, 5.0, 5.0, 20.0}, // corners swapped
    };
    for (const auto& values : invalid) {
        const trtmc::internal::ConfigEntry entry{
            "person_box", trtmc::Span<const double>(values.data(), values.size())};
        rejects<trtmc::internal::ConfigError>(
            [&] { model.run(request(pixels, 24, 18), {&entry, 1}); },
            "an invalid person box must be rejected");
    }
    const trtmc::internal::ConfigEntry unknown{"threshold", 0.5};
    rejects<trtmc::internal::ConfigError>(
        [&] { model.run(request(pixels, 24, 18), {&unknown, 1}); },
        "an unknown config must be rejected");
    const trtmc::internal::ConfigEntry scalar{"person_box", 0.5};
    rejects<trtmc::internal::ConfigError>([&] { model.run(request(pixels, 24, 18), {&scalar, 1}); },
                                          "a person box of the wrong kind must be rejected");
    require(recording->calls == 0, "invalid config must fail before engine execution");
}

void test_invalid_inputs_and_outputs() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::vitpose::VitPosePoints model(std::move(module), metadata());
    std::vector<float> pixels(24 * 18 * 3, 0.25F);
    const auto input = request(pixels, 24, 18);
    auto invalid = input;
    invalid.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject a truncated image");
    invalid = input;
    invalid.image.channels = 4;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject a non-RGB image");
    invalid = input;
    invalid.image.width = 0;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject an empty image");
    pixels[5] = 1.5F;
    rejects<std::invalid_argument>([&] { model.run(input, {}); }, "reject values above one");
    pixels[5] = -0.25F;
    rejects<std::invalid_argument>([&] { model.run(input, {}); }, "reject values below zero");
    pixels[5] = std::nanf("");
    rejects<std::invalid_argument>([&] { model.run(input, {}); }, "reject NaN pixels");
    pixels[5] = 0.25F;
    require(recording->calls == 0, "invalid input must fail before engine execution");

    recording->output_shape = {1, kKeypoints, kHeight, kWidth - 1};
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject incomplete heatmaps");
    recording->output_shape = {1, kKeypoints, kHeight, kWidth};
    recording->output_dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject a wrong output dtype");
    recording->output_dtype = trtmc::DType::kFloat32;
    recording->heatmaps[kPlane * kKeypoints - 1] = std::numeric_limits<float>::infinity();
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject non-finite heatmaps");
    recording->heatmaps[kPlane * kKeypoints - 1] = std::nanf("");
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject NaN heatmaps");
    recording->heatmaps[kPlane * kKeypoints - 1] = 0.0F;
    require(model.run(input, {}).parse_complete, "the same request with finite heatmaps succeeds");
}

void test_engine_and_metadata_contract() {
    auto module = std::make_unique<RecordingModule>();
    module->engine_output = {1, kKeypoints + 1, kHeight, kWidth};
    rejects<std::runtime_error>(
        [&] { trtmc::vitpose::VitPosePoints model(std::move(module), metadata()); },
        "an engine with another keypoint count must be rejected at load");
    module = std::make_unique<RecordingModule>();
    module->engine_input = {1, 3, kHeight + 1, kWidth};
    rejects<std::runtime_error>(
        [&] { trtmc::vitpose::VitPosePoints model(std::move(module), metadata()); },
        "an engine with another crop size must be rejected at load");
    auto even_kernel = metadata();
    even_kernel.dark_kernel = 4;
    auto zero_std = metadata();
    zero_std.preprocess.std[1] = 0.0F;
    auto tiny_std = metadata(); // finite and non-zero, but 1 / 1e-40 overflows float32
    tiny_std.preprocess.std[2] = 1e-40F;
    auto no_padding = metadata();
    no_padding.preprocess.padding_factor = 0.0;
    auto huge_padding = metadata(); // finite, but beyond float32: every crop would fail
    huge_padding.preprocess.padding_factor = 1e300;
    // 1 / rescale_factor beyond float32: far beyond, and a little above FLT_MAX, which a plain
    // conversion would round down to a finite FLT_MAX.
    auto huge_reciprocal = metadata();
    huge_reciprocal.preprocess.rescale_factor = 1e-100;
    auto edge_reciprocal = metadata();
    edge_reciprocal.preprocess.rescale_factor = 1.0 / 3.4028235e38;
    edge_reciprocal.preprocess.mean = {0.0F, 0.0F, 0.0F};
    edge_reciprocal.preprocess.std = {1.0F, 1.0F, 1.0F};
    for (const auto& config : {even_kernel, zero_std, tiny_std, no_padding, huge_padding,
                               huge_reciprocal, edge_reciprocal}) {
        rejects<std::runtime_error>(
            [&] {
                trtmc::vitpose::VitPosePoints model(std::make_unique<RecordingModule>(), config);
            },
            "metadata outside the runtime contract must be rejected at load");
    }
}

void test_geometry_values_beyond_float32_are_rejected() {
    // Each would be converted from double to float32; beyond FLT_MAX that conversion is undefined.
    auto config = metadata().preprocess;
    const trtmc::vitpose::PersonBox huge{1e300, 0.0, 10.0, 10.0};
    rejects<std::invalid_argument>([&] { trtmc::vitpose::crop_geometry(huge, config); },
                                   "a box centre beyond float32 must be rejected");
    const trtmc::vitpose::PersonBox wide{0.0, 0.0, 1e300, 10.0};
    rejects<std::invalid_argument>([&] { trtmc::vitpose::crop_geometry(wide, config); },
                                   "a box size beyond float32 must be rejected");
    config.padding_factor = 1e300;
    rejects<std::invalid_argument>(
        [&] { trtmc::vitpose::crop_geometry({0.0, 0.0, 10.0, 10.0}, config); },
        "a box padding beyond float32 must be rejected");
}

std::vector<char> characters(const std::string& text) {
    return {text.begin(), text.end()};
}

void test_runtime_json_is_read_strictly() {
    const std::string valid =
        R"({"input_height": 256, "input_width": 192, "rescale_factor": 0.00392156862745098, )"
        R"("image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225], )"
        R"("box_padding": 1.25, "heatmap_height": 64, "heatmap_width": 48, "dark_kernel": 11, )"
        R"("labels": ["Nose", "L_Eye", "R_Eye"]})";
    const auto parsed = trtmc::vitpose::parse_runtime_json(characters(valid));
    require(parsed.preprocess.height == 256 && parsed.preprocess.width == 192 &&
                parsed.preprocess.rescale_factor == 0.00392156862745098 &&
                parsed.preprocess.padding_factor == 1.25 && parsed.preprocess.mean[1] == 0.456F &&
                parsed.preprocess.std[2] == 0.225F && parsed.heatmap_height == 64 &&
                parsed.heatmap_width == 48 && parsed.dark_kernel == 11 &&
                parsed.labels == std::vector<std::string>({"Nose", "L_Eye", "R_Eye"}),
            "a valid runtime.json is read in full");
    const std::pair<std::string, std::string> invalid[] = {
        {R"("input_height": 256)", R"("input_height": 256.5)"},
        // 2^32 + 256 wraps to 256 as int32.
        {R"("input_height": 256)", R"("input_height": 4294967552)"},
        {R"("input_width": 192)", R"("input_width": -192)"},
        {R"("heatmap_width": 48)", R"("heatmap_width": null)"},
        {R"("dark_kernel": 11)", R"("dark_kernel": 11.0)"},
        {R"("input_height": 256)", R"("input_height": 256, "input_height": 128)"},
        {R"("box_padding": 1.25)", R"("box_padding": "1.25")"},
        {R"("image_std": [0.229, 0.224, 0.225])", R"("image_std": [0.229, 0.224, 1e300])"},
        {R"("image_mean": [0.485, 0.456, 0.406])", R"("image_mean": [0.485, 0.456])"},
        {R"("rescale_factor": 0.00392156862745098)", R"("rescale_factor": "0.0039")"},
        {R"("R_Eye"])", R"(3])"},
        {R"("R_Eye"])", R"("Nose"])"},
        {R"("R_Eye"])", R"(""])"},
        {R"("dark_kernel": 11)", R"("dark_kernel": 11, "skeleton": [])"},
    };
    for (const auto& [from, to] : invalid) {
        auto text = valid;
        const auto position = text.find(from);
        require(position != std::string::npos, "the invalid case must edit the valid runtime.json");
        text.replace(position, from.size(), to);
        rejects<std::runtime_error>([&] { trtmc::vitpose::parse_runtime_json(characters(text)); },
                                    "an invalid runtime.json must be rejected");
    }
    for (const std::string& text : {std::string("[]"), std::string("{"), std::string("")}) {
        rejects<std::runtime_error>([&] { trtmc::vitpose::parse_runtime_json(characters(text)); },
                                    "runtime.json must be one JSON object");
    }
}

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"binding_and_located_keypoints", test_binding_and_located_keypoints},
        {"query_box_and_8bit_input", test_query_box_and_8bit_input},
        {"invalid_boxes", test_invalid_boxes},
        {"invalid_inputs_and_outputs", test_invalid_inputs_and_outputs},
        {"engine_and_metadata_contract", test_engine_and_metadata_contract},
        {"geometry_values_beyond_float32_are_rejected",
         test_geometry_values_beyond_float32_are_rejected},
        {"runtime_json_is_read_strictly", test_runtime_json_is_read_strictly},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

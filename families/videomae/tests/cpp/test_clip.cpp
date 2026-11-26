/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side clip handling of the videomae runtime: tube layout, request
// checks, normalization and reconstruction, without an engine or a GPU.

#include "families/videomae/runtime/bundle_config.h"
#include "families/videomae/runtime/clip.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::videomae::ClipGeometry;
using trtmc::videomae::PixelNormalization;
using trtmc::videomae::TubeSelection;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

template <class Error>
void expect_throws(const std::function<void()>& call, const std::string& what,
                   const std::string& fragment = "") {
    try {
        call();
    } catch (const Error& error) {
        expect(fragment.empty() || std::string(error.what()).find(fragment) != std::string::npos,
               what + " (message: " + error.what() + ")");
        return;
    } catch (const std::exception& error) {
        expect(false, what + " threw the wrong type: " + error.what());
        return;
    }
    expect(false, what + " did not throw");
}

// 4 frames of 4x4 pixels in tubes of 2 frames x 2x2 pixels: 2 tubelets x 2 x 2
// = 8 tubes of 8 pixels (24 values).
const ClipGeometry kGeometry{4, 4, 2, 2};

std::size_t plane(std::int64_t frame, std::int64_t row, std::int64_t column) {
    return static_cast<std::size_t>((frame * kGeometry.size + row) * kGeometry.size + column);
}

// A mask whose tubes in `masked` are 1, every other value 0.
std::vector<float> tube_mask(const std::vector<std::int64_t>& masked) {
    std::vector<float> values(static_cast<std::size_t>(4 * 4 * 4), 0.0F);
    for (const auto tube : masked) {
        const auto t = tube / 4, row = tube / 2 % 2, column = tube % 2;
        for (std::int64_t f = 0; f < 2; ++f)
            for (std::int64_t y = 0; y < 2; ++y)
                for (std::int64_t x = 0; x < 2; ++x)
                    values[plane(t * 2 + f, row * 2 + y, column * 2 + x)] = 1.0F;
    }
    return values;
}

TubeSelection select(const std::vector<float>& values) {
    return trtmc::videomae::select_tubes({{values.data(), values.size()}, 4, 4, 4}, kGeometry);
}

std::vector<trtmc::internal::ImageView> frames_of(const std::vector<std::uint8_t>& clip,
                                                  std::size_t count, std::uint32_t size,
                                                  std::uint32_t channels) {
    std::vector<trtmc::internal::ImageView> frames;
    const std::size_t bytes = std::size_t{size} * size * channels;
    for (std::size_t f = 0; f < count; ++f)
        frames.push_back({clip.data() + f * bytes, bytes, size, size, channels,
                          trtmc::internal::ImageFormat::UInt8});
    return frames;
}

void geometry_layout() {
    expect(kGeometry.grid() == 2 && kGeometry.tubes() == 8, "tube count");
    expect(kGeometry.tube_pixels() == 8 && kGeometry.tube_values() == 24, "tube size");
    trtmc::videomae::validate_geometry(kGeometry);
    // The checkpoint's clip: 16 frames of 224x224 in 2x16x16 tubes.
    const ClipGeometry base{16, 224, 16, 2};
    trtmc::videomae::validate_geometry(base);
    expect(base.tubes() == 1568 && base.tube_values() == 1536, "VideoMAE-Base tube layout");
    expect_throws<std::runtime_error>([] { trtmc::videomae::validate_geometry({4, 5, 2, 2}); },
                                      "a frame size that is not whole patches");
    expect_throws<std::runtime_error>([] { trtmc::videomae::validate_geometry({3, 4, 2, 2}); },
                                      "a frame count that is not whole tubelets");
    expect_throws<std::runtime_error>([] { trtmc::videomae::validate_geometry({2, 2, 2, 2}); },
                                      "a single tube");
    // Malformed bundle metadata: refused before any product could overflow.
    expect_throws<std::runtime_error>(
        [] { trtmc::videomae::validate_geometry({2, 2147483646, 2, 1}); },
        "a frame size whose pixel count overflows", "oversized clip");
    expect_throws<std::runtime_error>(
        [] { trtmc::videomae::validate_geometry({65536, 26754, 2, 2}); },
        "a clip too large in total", "oversized clip");
}

void tube_selection() {
    const auto selection = select(tube_mask({1, 6}));
    expect((selection.masked == std::vector<std::int32_t>{1, 6}), "masked tubes, ascending");
    expect((selection.visible == std::vector<std::int32_t>{0, 2, 3, 4, 5, 7}),
           "visible tubes, ascending");

    auto mixed = tube_mask({1});
    mixed[plane(1, 0, 3)] = 0.0F; // second frame of tube 1 (rows 0-1, columns 2-3)
    expect_throws<std::invalid_argument>([&] { select(mixed); }, "a tube that is half masked",
                                         "is mixed");
    auto soft = tube_mask({2});
    for (auto& value : soft)
        if (value == 1.0F)
            value = 0.5F;
    expect_throws<std::invalid_argument>([&] { select(soft); }, "a soft mask value",
                                         "must be 0 (condition) or 1");
    auto nan = tube_mask({3});
    nan[plane(0, 2, 2)] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>([&] { select(nan); }, "a NaN mask value");
    expect_throws<std::invalid_argument>([] { select(tube_mask({})); }, "a mask with no tube",
                                         "no tube to reconstruct");
    expect_throws<std::invalid_argument>([] { select(tube_mask({0, 1, 2, 3, 4, 5, 6, 7})); },
                                         "a mask with every tube", "at least one tube");
    const std::vector<float> short_mask(10, 0.0F);
    expect_throws<std::invalid_argument>(
        [&] {
            trtmc::videomae::select_tubes({{short_mask.data(), short_mask.size()}, 4, 4, 4},
                                          kGeometry);
        },
        "a mask that does not cover the clip");
}

void source_frames() {
    std::vector<std::uint8_t> clip(4 * 4 * 4 * 3);
    for (std::size_t i = 0; i < clip.size(); ++i)
        clip[i] = static_cast<std::uint8_t>(i * 7 % 256);
    const auto frames = frames_of(clip, 4, 4, 3);
    const auto values = trtmc::videomae::source_clip({frames.data(), frames.size()}, kGeometry);
    bool exact = values.size() == clip.size();
    for (std::size_t i = 0; exact && i < clip.size(); ++i)
        exact = values[i] == static_cast<float>(static_cast<double>(clip[i]) * (1.0 / 255.0));
    expect(exact, "8-bit frames are rescaled by 1/255 in double, then rounded to float32");

    expect_throws<std::invalid_argument>(
        [&] { trtmc::videomae::source_clip({frames.data(), 3}, kGeometry); }, "three frames",
        "exactly 4 frames");
    const auto small = frames_of(clip, 4, 2, 3);
    expect_throws<std::invalid_argument>(
        [&] { trtmc::videomae::source_clip({small.data(), small.size()}, kGeometry); },
        "frames of another size", "4x4 RGB");
    const auto gray = frames_of(clip, 4, 4, 1);
    expect_throws<std::invalid_argument>(
        [&] { trtmc::videomae::source_clip({gray.data(), gray.size()}, kGeometry); },
        "grayscale frames", "with 1 channels");

    std::vector<float> floats(4 * 4 * 3, 0.25F);
    floats[5] = 1.5F;
    std::vector<trtmc::internal::ImageView> float_frames(
        4, {floats.data(), floats.size() * sizeof(float), 4, 4, 3,
            trtmc::internal::ImageFormat::Float32});
    expect_throws<std::invalid_argument>(
        [&] {
            trtmc::videomae::source_clip({float_frames.data(), float_frames.size()}, kGeometry);
        },
        "a float32 value above 1", "[0, 1]");
}

void normalization_layout() {
    std::vector<float> clip(4 * 4 * 4 * 3);
    for (std::size_t i = 0; i < clip.size(); ++i)
        clip[i] = static_cast<float>(i % 17) / 16.0F;
    const PixelNormalization normalization{{0.5F, 0.25F, 0.125F}, {0.5F, 2.0F, 0.25F}};
    const auto values = trtmc::videomae::pixel_values(clip, kGeometry, normalization);
    bool layout = values.size() == clip.size();
    for (std::size_t frame = 0; layout && frame < 4; ++frame)
        for (std::size_t pixel = 0; layout && pixel < 16; ++pixel)
            for (std::size_t c = 0; layout && c < 3; ++c)
                layout = values[(frame * 3 + c) * 16 + pixel] ==
                         (clip[(frame * 16 + pixel) * 3 + c] - normalization.mean[c]) /
                             normalization.std[c];
    expect(layout, "pixel_values is [1, frames, 3, size, size], normalized per channel");
}

void reconstruction() {
    // Tube 5 = tubelet 1, row 0, column 1: frames 2-3, rows 0-1, columns 2-3.
    std::vector<float> clip(4 * 4 * 4 * 3, 0.0F);
    std::vector<double> source(8 * 3);
    std::size_t pixel = 0;
    for (std::int64_t f = 2; f < 4; ++f)
        for (std::int64_t y = 0; y < 2; ++y)
            for (std::int64_t x = 2; x < 4; ++x, ++pixel)
                for (std::size_t c = 0; c < 3; ++c) {
                    const float value =
                        0.1F * static_cast<float>(c + 1) + 0.05F * static_cast<float>(pixel % 3);
                    clip[plane(f, y, x) * 3 + c] = value;
                    source[pixel * 3 + c] = value;
                }
    for (std::size_t i = 0; i < clip.size(); ++i)
        if (clip[i] == 0.0F)
            clip[i] = 0.75F; // every other tube: a visible value that must survive
    const TubeSelection selection{{0, 1, 2, 3, 4, 6, 7}, {5}};
    std::vector<float> prediction(24);
    for (std::size_t i = 0; i < prediction.size(); ++i)
        prediction[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.5F;
    prediction[0] = 100.0F;  // clamped to 1
    prediction[1] = -100.0F; // clamped to 0

    std::vector<double> expected(24);
    for (std::size_t c = 0; c < 3; ++c) {
        double mean = 0.0, spread = 0.0;
        for (std::size_t p = 0; p < 8; ++p)
            mean += source[p * 3 + c];
        mean /= 8.0;
        for (std::size_t p = 0; p < 8; ++p)
            spread += (source[p * 3 + c] - mean) * (source[p * 3 + c] - mean);
        const double scale = std::sqrt(spread / 7.0) + 1e-6;
        for (std::size_t p = 0; p < 8; ++p)
            expected[p * 3 + c] =
                std::fmin(1.0, std::fmax(0.0, prediction[p * 3 + c] * scale + mean));
    }
    auto result = clip;
    trtmc::videomae::reconstruct(result, prediction.data(), selection, kGeometry);
    // A tolerance, not equality: a compiler may contract the product and sum into one FMA.
    bool matches = true;
    pixel = 0;
    for (std::int64_t f = 2; f < 4; ++f)
        for (std::int64_t y = 0; y < 2; ++y)
            for (std::int64_t x = 2; x < 4; ++x, ++pixel)
                for (std::size_t c = 0; c < 3; ++c)
                    matches = matches && std::fabs(result[plane(f, y, x) * 3 + c] -
                                                   expected[pixel * 3 + c]) < 1e-6;
    expect(matches, "a masked tube is its prediction times the unbiased std (+1e-6) plus the mean");
    expect(result[plane(2, 0, 2) * 3] == 1.0F && result[plane(2, 0, 2) * 3 + 1] == 0.0F,
           "reconstructed values are clamped to [0, 1]");
    std::size_t kept = 0, visible = 0;
    for (std::size_t i = 0; i < clip.size(); ++i)
        if (clip[i] == 0.75F) {
            ++visible;
            kept += result[i] == 0.75F;
        }
    expect(visible == 7 * 24 && kept == visible, "conditioning tubes keep their source pixels");

    prediction[3] = std::numeric_limits<float>::infinity();
    expect_throws<std::runtime_error>(
        [&] {
            auto copy = clip;
            trtmc::videomae::reconstruct(copy, prediction.data(), selection, kGeometry);
        },
        "a non-finite prediction", "non-finite");
}

// runtime.json as the builder writes it for the checkpoint.
const std::string kRuntimeJson =
    R"({"num_frames": 16, "image_size": 224, "patch_size": 16, "tubelet_size": 2, )"
    R"("image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]})";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    text.replace(text.find(from), from.size(), to);
    return text;
}

void runtime_config() {
    const auto config = trtmc::videomae::parse_runtime_config(kRuntimeJson);
    expect(config.geometry.frames == 16 && config.geometry.size == 224 &&
               config.geometry.patch == 16 && config.geometry.tubelet == 2,
           "runtime.json geometry");
    expect(config.normalization.mean[1] == 0.456F && config.normalization.std[2] == 0.225F,
           "runtime.json channel statistics");
    const std::map<std::string, std::string> refused{
        {"a float where an integer belongs", replaced(kRuntimeJson, "16,", "16.0,")},
        {"a boolean where an integer belongs",
         replaced(kRuntimeJson, "\"tubelet_size\": 2", "\"tubelet_size\": true")},
        {"a null where an integer belongs", replaced(kRuntimeJson, "224", "null")},
        {"a repeated key",
         replaced(kRuntimeJson, "\"num_frames\": 16,", "\"num_frames\": 16, \"num_frames\": 18,")},
        {"an unknown key", replaced(kRuntimeJson, "{", "{\"norm_pix_loss\": false, ")},
        {"a missing key", replaced(kRuntimeJson, "\"patch_size\": 16, ", "")},
        {"a boolean statistic", replaced(kRuntimeJson, "0.485", "true")},
        {"two statistics", replaced(kRuntimeJson, ", 0.406]", "]")},
        {"a zero standard deviation", replaced(kRuntimeJson, "0.224", "0")},
        {"a statistic beyond float32", replaced(kRuntimeJson, "0.229", "1e300")},
        {"a standard deviation that overflows the normalization",
         replaced(kRuntimeJson, "0.229", "1e-40")},
        {"an integer beyond int64", replaced(kRuntimeJson, "224", "18446744073709551615")},
        {"a negative frame count", replaced(kRuntimeJson, "16,", "-16,")},
        {"not an object", "[1, 2, 3]"},
        {"invalid JSON", "{\"num_frames\": 16"},
    };
    for (const auto& [name, text] : refused)
        expect_throws<std::runtime_error>([&] { trtmc::videomae::parse_runtime_config(text); },
                                          "runtime.json with " + name);
}

// An engine stand-in that reports tensors the way the TensorRT module does: a dynamic
// input reports its opt shape, an output the shape inferred for the largest profile.
class FakeEngine final : public trtmc::ITrtModule {
  public:
    std::map<std::string, std::pair<trtmc::DType, std::vector<std::int64_t>>> inputs, outputs;
    std::set<std::string> dynamic;
    std::map<std::pair<std::string, trtmc::ProfileShapeSelector>, std::vector<std::int64_t>>
        profiles;

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(inputs, true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(outputs, false); }
    bool has_input(const std::string& name) const override { return inputs.count(name) != 0; }
    bool has_output(const std::string& name) const override { return outputs.count(name) != 0; }
    trtmc::DType tensor_dtype(const std::string& name) const override { return find(name).first; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return find(name).second;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t profile,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto found = profiles.find({name, selector});
        return profile == 0 && found != profiles.end() ? found->second : std::vector<int64_t>{};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(find(name).second.size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return inputs.count(name) != 0 && dynamic.count(name) != 0;
    }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    using Tensors = std::map<std::string, std::pair<trtmc::DType, std::vector<std::int64_t>>>;
    static std::vector<trtmc::TensorInfo> info(const Tensors& tensors, bool input) {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, tensor] : tensors)
            result.push_back({name, tensor.second, tensor.first, input});
        return result;
    }
    const std::pair<trtmc::DType, std::vector<std::int64_t>>& find(const std::string& name) const {
        const auto input = inputs.find(name);
        return input != inputs.end() ? input->second : outputs.at(name);
    }
};

FakeEngine checkpoint_engine() {
    FakeEngine engine;
    engine.inputs = {{"pixel_values", {trtmc::DType::kFloat32, {1, 16, 3, 224, 224}}},
                     {"visible_index", {trtmc::DType::kInt32, {784}}},
                     {"masked_index", {trtmc::DType::kInt32, {784}}}};
    engine.outputs = {{"logits", {trtmc::DType::kFloat32, {1, 1567, 1536}}}};
    engine.dynamic = {"visible_index", "masked_index"};
    for (const char* name : {"visible_index", "masked_index"}) {
        engine.profiles[{name, trtmc::ProfileShapeSelector::kMin}] = {1};
        engine.profiles[{name, trtmc::ProfileShapeSelector::kOpt}] = {784};
        engine.profiles[{name, trtmc::ProfileShapeSelector::kMax}] = {1567};
    }
    return engine;
}

void engine_io() {
    const ClipGeometry base{16, 224, 16, 2};
    const auto good = checkpoint_engine();
    trtmc::videomae::validate_engine(good, base);
    std::map<std::string, std::function<void(FakeEngine&)>> defects{
        {"another frame size", [](FakeEngine& e) { e.inputs["pixel_values"].second[3] = 256; }},
        {"float16 pixels",
         [](FakeEngine& e) { e.inputs["pixel_values"].first = trtmc::DType::kFloat16; }},
        {"float32 tube indices",
         [](FakeEngine& e) { e.inputs["masked_index"].first = trtmc::DType::kFloat32; }},
        {"rank-2 tube indices", [](FakeEngine& e) { e.inputs["visible_index"].second = {1, 784}; }},
        {"a fixed-length tube index",
         [](FakeEngine& e) {
             e.dynamic.erase("masked_index");
             e.inputs["masked_index"].second = {160};
         }},
        {"a profile without a single visible tube",
         [](FakeEngine& e) {
             e.profiles[{"visible_index", trtmc::ProfileShapeSelector::kMin}] = {2};
         }},
        {"a profile short of tubes - 1 masked tubes",
         [](FakeEngine& e) {
             e.profiles[{"masked_index", trtmc::ProfileShapeSelector::kMax}] = {1408};
         }},
        {"another tube size", [](FakeEngine& e) { e.outputs["logits"].second[2] = 768; }},
        {"logits of a fixed masked length",
         [](FakeEngine& e) { e.outputs["logits"].second[1] = 1408; }},
        {"a missing input", [](FakeEngine& e) { e.inputs.erase("masked_index"); }},
        {"an extra output",
         [](FakeEngine& e) { e.outputs["features"] = {trtmc::DType::kFloat32, {1, 8}}; }},
    };
    for (const auto& [name, defect] : defects) {
        auto engine = checkpoint_engine();
        defect(engine);
        expect_throws<std::runtime_error>([&] { trtmc::videomae::validate_engine(engine, base); },
                                          "an engine with " + name);
    }
}

} // namespace

int main() {
    geometry_layout();
    tube_selection();
    source_frames();
    normalization_layout();
    reconstruction();
    runtime_config();
    engine_io();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "videomae clip checks passed\n";
    return EXIT_SUCCESS;
}

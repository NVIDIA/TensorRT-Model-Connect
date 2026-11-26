/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dpt/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <typename Function>
bool throws_runtime_error(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

constexpr std::int64_t kSide = 384;

// An engine double that reports configurable I/O and returns a configurable
// depth map, and checks what the pipeline hands it.
class FakeEngine final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto found = inputs.find("pixel_values");
        input_ok = inputs.size() == 1U && found != inputs.end() && found->second.data != nullptr &&
                   found->second.dtype == trtmc::DType::kFloat32 &&
                   found->second.shape == std::vector<std::int64_t>{1, 3, kSide, kSide};
        return {{output_name, {depth.data(), output_shape, output_dtype}}};
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
    std::vector<trtmc::TensorInfo> input_info() const override { return inputs; }
    std::vector<trtmc::TensorInfo> output_info() const override { return outputs; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
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
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return healthy; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<trtmc::TensorInfo> inputs{
        {"pixel_values", {1, 3, kSide, kSide}, trtmc::DType::kFloat32, true}};
    std::vector<trtmc::TensorInfo> outputs{
        {"predicted_depth", {1, 1, kSide, kSide}, trtmc::DType::kFloat32, false}};
    bool healthy{true};
    std::string output_name{"predicted_depth"};
    std::vector<std::int64_t> output_shape{1, 1, kSide, kSide};
    trtmc::DType output_dtype{trtmc::DType::kFloat32};
    std::vector<float> depth = std::vector<float>(static_cast<std::size_t>(kSide * kSide), 2.0F);
    int calls{0};
    bool input_ok{false};
};

const std::map<std::string, std::string> kHealthyRuntime = {
    {"input_image_h", "384"},
    {"input_image_w", "384"},
    {"rescale_factor", "0.00392156862745098"},
    {"image_mean", "[0.5, 0.5, 0.5]"},
    {"image_std", "[0.5, 0.5, 0.5]"},
};

std::string runtime_json(const std::map<std::string, std::string>& values,
                         const std::string& extra = "") {
    std::string text = "{";
    for (const auto& [key, value] : values)
        text += (text.size() > 1 ? ", \"" : "\"") + key + "\": " + value;
    return text + extra + "}";
}

std::string with(const char* key, const char* value) {
    auto values = kHealthyRuntime;
    values[key] = value;
    return runtime_json(values);
}

std::string without(const char* key) {
    auto values = kHealthyRuntime;
    values.erase(key);
    return runtime_json(values);
}

trtmc::dpt::PreprocessConfig healthy_config() {
    return trtmc::dpt::parse_runtime_config(runtime_json(kHealthyRuntime));
}

void test_runtime_json_is_read_exactly() {
    const auto config = healthy_config();
    check(config.input_image_h == 384 && config.input_image_w == 384 &&
              config.rescale_factor == 0.00392156862745098 && config.image_mean[1] == 0.5F &&
              config.image_std[2] == 0.5F,
          "the build's runtime.json is read as written");

    const std::vector<std::pair<std::string, std::string>> refused = {
        {"an integer written as a float", with("input_image_h", "384.0")},
        {"a boolean for an integer", with("input_image_w", "true")},
        {"a negative size", with("input_image_h", "-384")},
        {"a zero size", with("input_image_w", "0")},
        {"a size that wraps to 384 in int32", with("input_image_h", "4294967680")},
        {"a size past int64", with("input_image_w", "18446744073709551615")},
        {"a number written as a string", with("rescale_factor", "\"0.00392156862745098\"")},
        {"an explicit null", with("rescale_factor", "null")},
        {"a zero rescale factor", with("rescale_factor", "0")},
        {"a negative rescale factor", with("rescale_factor", "-0.0039")},
        {"a rescale factor that overflows float32", with("rescale_factor", "1e308")},
        {"two means", with("image_mean", "[0.5, 0.5]")},
        {"a mean past float32", with("image_mean", "[0.5, 0.5, 1e39]")},
        {"a zero std", with("image_std", "[0.5, 0, 0.5]")},
        {"a std that rounds to zero in float32", with("image_std", "[0.5, 1e-50, 0.5]")},
        {"a std that normalizes a sample to infinity", with("image_std", "[0.5, 1e-40, 0.5]")},
        {"a std given as one number", with("image_std", "0.5")},
        {"a missing key", without("image_std")},
        {"an unknown key", runtime_json(kHealthyRuntime, ", \"resample\": 2")},
        {"a repeated key", runtime_json(kHealthyRuntime, ", \"input_image_h\": 512")},
        {"an array instead of an object", "[384, 384]"},
        {"truncated JSON", "{\"input_image_h\": 384"},
    };
    for (const auto& [what, text] : refused) {
        check(throws_runtime_error([&] { trtmc::dpt::parse_runtime_config(text); }),
              "runtime.json with " + what + " is refused");
    }
}

std::unique_ptr<FakeEngine> healthy_engine() {
    return std::make_unique<FakeEngine>();
}

bool refused_at_load(std::unique_ptr<FakeEngine> engine,
                     trtmc::dpt::PreprocessConfig config = healthy_config()) {
    return throws_runtime_error(
        [&] { trtmc::dpt::DptPipeline pipeline(std::move(engine), config); });
}

void test_engine_io_is_checked_at_load() {
    check(!refused_at_load(healthy_engine()), "an engine of the runtime contract loads");

    auto engine = healthy_engine();
    engine->inputs[0].name = "image";
    check(refused_at_load(std::move(engine)), "an engine input of another name is refused");
    engine = healthy_engine();
    engine->inputs[0].dtype = trtmc::DType::kFloat16;
    check(refused_at_load(std::move(engine)), "an fp16 engine input is refused");
    engine = healthy_engine();
    engine->inputs[0].shape = {1, 3, kSide, kSide - 1};
    check(refused_at_load(std::move(engine)), "an engine input of another size is refused");
    engine = healthy_engine();
    engine->inputs[0].shape = {-1, 3, kSide, kSide};
    check(refused_at_load(std::move(engine)), "a dynamic engine input is refused");
    engine = healthy_engine();
    engine->inputs.push_back(engine->inputs[0]);
    engine->inputs[1].name = "mask";
    check(refused_at_load(std::move(engine)), "an engine with a second input is refused");

    engine = healthy_engine();
    engine->outputs[0].name = "depth";
    check(refused_at_load(std::move(engine)), "an engine output of another name is refused");
    engine = healthy_engine();
    engine->outputs[0].dtype = trtmc::DType::kFloat16;
    check(refused_at_load(std::move(engine)), "an fp16 engine output is refused");
    engine = healthy_engine();
    engine->outputs[0].shape = {1, kSide, kSide};
    check(refused_at_load(std::move(engine)), "an engine output of another rank is refused");
    engine = healthy_engine();
    engine->outputs.clear();
    check(refused_at_load(std::move(engine)), "an engine without an output is refused");

    engine = healthy_engine();
    engine->healthy = false;
    check(refused_at_load(std::move(engine)), "an engine that failed to load is refused");
    auto config = healthy_config();
    config.input_image_w = 512;
    check(refused_at_load(healthy_engine(), config),
          "an engine whose size differs from runtime.json is refused");
    config = healthy_config();
    config.image_std = {0.5F, 0.0F, 0.5F};
    check(refused_at_load(healthy_engine(), config),
          "a configuration the image operations refuse is refused at load");
}

std::vector<float> grey_image(std::int32_t height, std::int32_t width) {
    return std::vector<float>(static_cast<std::size_t>(height) * width * 3U, 0.5F);
}

void test_depth_is_checked_before_and_after_the_resize() {
    constexpr std::int32_t kHeight = 4;
    constexpr std::int32_t kWidth = 6;
    const auto image = grey_image(kHeight, kWidth);

    auto engine = healthy_engine();
    auto* fake = engine.get();
    trtmc::dpt::DptPipeline pipeline(std::move(engine), healthy_config());
    const auto result = pipeline.estimate_depth(image.data(), kHeight, kWidth);
    bool constant = result.depth.size() == static_cast<std::size_t>(kHeight * kWidth);
    for (const float value : result.depth)
        constant = constant && std::fabs(value - 2.0F) <= 1e-5F;
    check(fake->calls == 1 && fake->input_ok, "the engine gets one float32 [1, 3, 384, 384] input");
    check(result.height == kHeight && result.width == kWidth && constant,
          "a constant depth map resizes to a constant map of the image's size");

    fake->depth[1000] = std::numeric_limits<float>::quiet_NaN();
    check(throws_runtime_error([&] { pipeline.estimate_depth(image.data(), kHeight, kWidth); }),
          "a NaN from the engine is refused before the resize");
    fake->depth[1000] = std::numeric_limits<float>::infinity();
    check(throws_runtime_error([&] { pipeline.estimate_depth(image.data(), kHeight, kWidth); }),
          "an infinity from the engine is refused before the resize");

    // The bicubic taps of a 384 -> 4 resize are -0.09375, 0.59375, 0.59375 and
    // -0.09375: their running sum passes 1 and overflows a map of FLT_MAX.
    fake->depth.assign(fake->depth.size(), std::numeric_limits<float>::max());
    check(throws_runtime_error([&] { pipeline.estimate_depth(image.data(), kHeight, kWidth); }),
          "a finite engine map that overflows in the resize is refused");

    fake->depth.assign(fake->depth.size(), 2.0F);
    fake->output_shape = {1, kSide, kSide};
    check(throws_runtime_error([&] { pipeline.estimate_depth(image.data(), kHeight, kWidth); }),
          "an engine output of another shape is refused at run time");
    fake->output_shape = {1, 1, kSide, kSide};
    fake->output_name = "depth";
    check(throws_runtime_error([&] { pipeline.estimate_depth(image.data(), kHeight, kWidth); }),
          "a missing engine output is refused at run time");
}

} // namespace

int main() {
    test_runtime_json_is_read_exactly();
    test_engine_io_is_checked_at_load();
    test_depth_is_checked_before_and_after_the_resize();
    if (failures != 0) {
        std::fprintf(stderr, "%d dpt pipeline check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("dpt pipeline checks passed\n");
    return EXIT_SUCCESS;
}

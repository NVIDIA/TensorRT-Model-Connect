/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam3/plugin_helpers.h"
#include "runtime/models/sam3/sam3_pipeline.h"
#include "runtime/models/sam3/sam3_video_processor.h"
#ifdef TRTMC_HAS_CUDA_KERNELS
#include "runtime/models/sam3/sam3_video_kernels.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

bool close(float actual, float expected) {
    return std::fabs(actual - expected) < 1.0e-5F;
}

trtmc::BundleFile make_sam3_clip_tokenizer_bundle() {
    static constexpr std::string_view config = R"({
      "tokenizer_add_special_tokens": true,
      "tokenizer_special_prefix_ids": [10],
      "tokenizer_special_suffix_ids": [11]
    })";
    static constexpr std::string_view tokenizer_json = R"({
      "model": {
        "type": "BPE",
        "end_of_word_suffix": "</w>",
        "vocab": {
          "a</w>": 0,
          "b</w>": 1,
          "1</w>": 2,
          "0</w>": 3,
          "\u0120": 4,
          "\u0120</w>": 5,
          "-</w>": 6,
          "'": 7,
          "s</w>": 8,
          "<|startoftext|>": 10,
          "<|endoftext|>": 11
        },
        "merges": []
      },
      "normalizer": {
        "type": "Sequence",
        "normalizers": [
          {"type": "NFC"},
          {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
          {"type": "Lowercase"}
        ]
      },
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {
            "type": "Split",
            "pattern": {
              "Regex": "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
            },
            "behavior": "Removed",
            "invert": true
          },
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
          }
        ]
      }
    })";

    trtmc::BundleFile bundle;
    trtmc::BundleSection config_section;
    config_section.name = "config.json";
    config_section.data.assign(config.begin(), config.end());
    bundle.sections.push_back(std::move(config_section));
    trtmc::BundleSection tokenizer_section;
    tokenizer_section.name = "tokenizer.json";
    tokenizer_section.data.assign(tokenizer_json.begin(), tokenizer_json.end());
    bundle.sections.push_back(std::move(tokenizer_section));
    return bundle;
}

void test_sam3_clip_tokenizer_matches_meta_segmentation() {
    auto bundle = make_sam3_clip_tokenizer_bundle();
    auto tokenizer = trtmc::create_tokenizer_from_bundle(bundle);
    check(tokenizer != nullptr, "sam3 creates native CLIP tokenizer");
    check(tokenizer->encode("a b") == std::vector<int32_t>({10, 0, 1, 11}),
          "sam3 CLIP tokenizer matches Meta Removed split whitespace semantics");
    check(tokenizer->encode("A\t  B") == std::vector<int32_t>({10, 0, 1, 11}),
          "sam3 CLIP tokenizer matches Meta lowercase and whitespace normalization");
    check(tokenizer->encode("10 b") == std::vector<int32_t>({10, 2, 3, 1, 11}),
          "sam3 CLIP tokenizer matches Meta single-digit segmentation");
    check(tokenizer->encode("A-B") == std::vector<int32_t>({10, 0, 6, 1, 11}),
          "sam3 CLIP tokenizer matches Meta punctuation segmentation");
    check(tokenizer->encode("A's") == std::vector<int32_t>({10, 0, 7, 8, 11}),
          "sam3 CLIP tokenizer matches Meta contraction segmentation");
}

void test_clip_non_removed_split_keeps_standalone_space_token() {
    auto bundle = make_sam3_clip_tokenizer_bundle();
    auto& tokenizer_data = bundle.sections.back().data;
    std::string tokenizer_json(tokenizer_data.begin(), tokenizer_data.end());
    const auto behavior = tokenizer_json.find("\"behavior\": \"Removed\"");
    check(behavior != std::string::npos, "sam3 synthetic CLIP tokenizer has Removed behavior");
    tokenizer_json.replace(behavior, std::strlen("\"behavior\": \"Removed\""),
                           "\"behavior\": \"Isolated\"");
    tokenizer_data.assign(tokenizer_json.begin(), tokenizer_json.end());

    auto tokenizer = trtmc::create_tokenizer_from_bundle(bundle);
    check(tokenizer != nullptr, "sam3 creates non-Removed CLIP tokenizer control");
    check(tokenizer->encode("a b") == std::vector<int32_t>({10, 0, 4, 1, 11}),
          "sam3 does not filter whitespace outside the Removed split contract");
}

#ifdef TRTMC_HAS_CUDA_KERNELS
std::uint32_t bfloat16_reference_bits(std::uint32_t bits) {
    const std::uint32_t exponent = bits & 0x7F800000U;
    const std::uint32_t mantissa = bits & 0x007FFFFFU;
    if (exponent == 0x7F800000U && mantissa != 0U)
        bits |= 0x00010000U;
    else
        bits += 0x00007FFFU + ((bits >> 16U) & 1U);
    return bits & 0xFFFF0000U;
}

void test_bfloat16_round_copy_supports_exact_alias() {
    const std::array<std::uint32_t, 8> input_bits{
        0x00000000U, 0x3F8CCCCDU, 0x3F808000U, 0x3F818000U,
        0x7F800000U, 0xFF800000U, 0x7F800001U, 0xFFC12345U,
    };
    std::array<float, input_bits.size()> input{};
    std::memcpy(input.data(), input_bits.data(), sizeof(input_bits));
    float* device = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(input)) == cudaSuccess,
          "sam3 BF16 alias allocates storage");
    check(cudaMemcpy(device, input.data(), sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess,
          "sam3 BF16 alias uploads input");
    trtmc::sam3_round_bfloat16_copy(device, device, input.size(), nullptr);
    check(cudaGetLastError() == cudaSuccess, "sam3 BF16 alias launches");
    std::array<std::uint32_t, input_bits.size()> actual{};
    check(cudaMemcpy(actual.data(), device, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
          "sam3 BF16 alias downloads output");
    check(cudaFree(device) == cudaSuccess, "sam3 BF16 alias releases storage");
    for (std::size_t index = 0; index < input_bits.size(); ++index)
        check(actual[index] == bfloat16_reference_bits(input_bits[index]),
              "sam3 BF16 alias matches round-to-nearest-even");
}

void test_cuda_preprocess_matches_cpu_meta_pillow() {
    constexpr int32_t input_size = 2;
    constexpr int32_t output_size = 4;
    constexpr unsigned int precision = 22;
    const std::array<trtmc::Sam3CudaResizeAxisEntry, output_size> entries{{
        {0, 0, 1},
        {0, 1, 2},
        {0, 3, 2},
        {1, 5, 1},
    }};
    const std::array<int32_t, 6> weights{
        4194304, 3145728, 1048576, 1048576, 3145728, 4194304,
    };
    const std::array<uint8_t, 12> source_u8{
        0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 240,
    };
    std::array<float, source_u8.size()> source{};
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(source_u8[index]) / 255.0F;

    trtmc::Sam3Config config;
    config.image_size = output_size;
    config.image_mean = {0.5F, 0.5F, 0.5F};
    config.image_std = {0.5F, 0.5F, 0.5F};
    const auto expected =
        trtmc::preprocess_sam3_image(source.data(), input_size, input_size, config);

    std::array<float, 3U * 256U> normalization_lut{};
    for (std::size_t value = 0; value < 256U; ++value) {
        const float pixel = static_cast<float>(value) / 255.0F;
        const std::array<float, 3> one_pixel{pixel, pixel, pixel};
        trtmc::Sam3Config one_pixel_config = config;
        one_pixel_config.image_size = 1;
        const auto normalized =
            trtmc::preprocess_sam3_image(one_pixel.data(), 1, 1, one_pixel_config);
        for (std::size_t channel = 0; channel < 3U; ++channel)
            normalization_lut[channel * 256U + value] = normalized[channel];
    }

    float* device_source = nullptr;
    uint8_t* device_quantized = nullptr;
    uint8_t* device_horizontal = nullptr;
    trtmc::Sam3CudaResizeAxisEntry* device_entries = nullptr;
    int32_t* device_weights = nullptr;
    float* device_lut = nullptr;
    float* device_output = nullptr;
    int* device_status = nullptr;
    check(
        cudaMalloc(reinterpret_cast<void**>(&device_source), sizeof(source)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_quantized), source.size()) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_horizontal),
                       input_size * output_size * 3U) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_entries), sizeof(entries)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_weights), sizeof(weights)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_lut), sizeof(normalization_lut)) ==
                cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_output), expected.size() * sizeof(float)) ==
                cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(int)) == cudaSuccess,
        "sam3 CUDA Meta preprocessing allocates test buffers");
    check(cudaMemcpy(device_source, source.data(), sizeof(source), cudaMemcpyHostToDevice) ==
                  cudaSuccess &&
              cudaMemcpy(device_entries, entries.data(), sizeof(entries), cudaMemcpyHostToDevice) ==
                  cudaSuccess &&
              cudaMemcpy(device_weights, weights.data(), sizeof(weights), cudaMemcpyHostToDevice) ==
                  cudaSuccess &&
              cudaMemcpy(device_lut, normalization_lut.data(), sizeof(normalization_lut),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(device_status, 0, sizeof(int)) == cudaSuccess,
          "sam3 CUDA Meta preprocessing uploads test data");

    check(trtmc::sam3_cuda_preprocess_image(
              device_source, input_size, input_size, device_quantized, device_horizontal,
              device_entries, output_size, device_weights, weights.size(), precision,
              device_entries, output_size, device_weights, weights.size(), precision, device_output,
              0, output_size, output_size, device_lut, device_status, nullptr),
          "sam3 CUDA Meta preprocessing accepts fixed-22 int32 plans");
    std::vector<float> actual(expected.size());
    int status = -1;
    check(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(&status, device_status, sizeof(status), cudaMemcpyDeviceToHost) ==
                  cudaSuccess,
          "sam3 CUDA Meta preprocessing downloads output");
    check(status == 0 && actual == expected,
          "sam3 CUDA fixed-22 resize and FP16 LUT are bit-exact with the CPU Meta path");

    check(cudaFree(device_status) == cudaSuccess && cudaFree(device_output) == cudaSuccess &&
              cudaFree(device_lut) == cudaSuccess && cudaFree(device_weights) == cudaSuccess &&
              cudaFree(device_entries) == cudaSuccess &&
              cudaFree(device_horizontal) == cudaSuccess &&
              cudaFree(device_quantized) == cudaSuccess && cudaFree(device_source) == cudaSuccess,
          "sam3 CUDA Meta preprocessing releases test buffers");
}
#endif

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        return ids;
    }

    std::string decode(const std::vector<int32_t>& /*ids*/) const override { return {}; }
    int32_t id_for_token(std::string_view /*token*/) const override { return -1; }
    std::string token_for_id(int32_t /*id*/) const override { return {}; }

    std::vector<int32_t> ids{7, 8};
    mutable std::string last_text;
};

class FakeSam3TextModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto ids_it = inputs.find("input_ids");
        const auto mask_it = inputs.find("attention_mask");
        if (ids_it == inputs.end() || mask_it == inputs.end() || !ids_it->second.data ||
            !mask_it->second.data) {
            throw std::runtime_error("missing SAM3 text inputs");
        }
        const auto* ids = static_cast<const int32_t*>(ids_it->second.data);
        const auto* mask = static_cast<const int32_t*>(mask_it->second.data);
        saw_expected_ids = ids[0] == 7 && ids[1] == 8 && ids[2] == 0 && ids[3] == 0;
        saw_expected_mask = mask[0] == 1 && mask[1] == 1 && mask[2] == 0 && mask[3] == 0;
        saw_shape = ids_it->second.shape == std::vector<int64_t>{4} &&
                    mask_it->second.shape == std::vector<int64_t>{4};

        features_ = {1.0F, 2.0F, 3.0F, 4.0F};
        hidden_ = {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};

        trtmc::Tensor features;
        features.data = features_.data();
        features.shape = {4, 1};
        features.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor hidden;
        hidden.data = hidden_.data();
        hidden.shape = {4, 2};
        hidden.dtype = trtmc::DType::kFloat32;

        return {{"sam3_text_features", features}, {"sam3_text_hidden_states", hidden}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& /*name*/) const override { return false; }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& /*name*/) const override { return {}; }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_expected_ids{false};
    bool saw_expected_mask{false};
    bool saw_shape{false};

  private:
    std::vector<float> features_;
    std::vector<float> hidden_;
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3VisionModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto image_it = inputs.find("pixel_values");
        if (image_it == inputs.end() || !image_it->second.data)
            throw std::runtime_error("missing SAM3 image input");
        const auto* pixels = static_cast<const float*>(image_it->second.data);
        last_input_address = pixels;
        saw_shape = image_it->second.shape == std::vector<int64_t>({1, 3, 4, 4});
        last_pixels.assign(pixels, pixels + 3 * 4 * 4);
        saw_normalized_pixels = close(pixels[0], -0.498046875F) && close(pixels[16], 0.00390625F) &&
                                close(pixels[32], 0.498046875F);

        trtmc::TensorMap out;
        for (int32_t level = 0; level < 3; ++level) {
            const auto level_index = static_cast<std::size_t>(level);
            fpn_hidden_[level_index] = {10.0F + static_cast<float>(level),
                                        11.0F + static_cast<float>(level)};
            fpn_position_[level_index] = {20.0F + static_cast<float>(level),
                                          21.0F + static_cast<float>(level)};

            trtmc::Tensor hidden;
            hidden.data = fpn_hidden_[level_index].data();
            hidden.shape = {1, 2, 1, 1};
            hidden.dtype = trtmc::DType::kFloat32;
            out["sam3_fpn_hidden_" + std::to_string(level)] = hidden;

            trtmc::Tensor pos;
            pos.data = fpn_position_[level_index].data();
            pos.shape = {1, 2, 1, 1};
            pos.dtype = trtmc::DType::kFloat32;
            out["sam3_fpn_position_" + std::to_string(level)] = pos;

            tracker_hidden_[level_index] = {30.0F + static_cast<float>(level)};
            trtmc::Tensor tracker_hidden;
            tracker_hidden.data = tracker_hidden_[level_index].data();
            tracker_hidden.shape = {1, 1, 1, 1};
            tracker_hidden.dtype = trtmc::DType::kFloat32;
            out["sam3_tracker_feature_" + std::to_string(level)] = tracker_hidden;

            tracker_position_[level_index] = {40.0F + static_cast<float>(level)};
            trtmc::Tensor tracker_pos;
            tracker_pos.data = tracker_position_[level_index].data();
            tracker_pos.shape = {1, 1, 1, 1};
            tracker_pos.dtype = trtmc::DType::kFloat32;
            out["sam3_tracker_position_" + std::to_string(level)] = tracker_pos;
        }
        return out;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& /*name*/) const override { return false; }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& /*name*/) const override { return {}; }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_shape{false};
    bool saw_normalized_pixels{false};
    std::vector<float> last_pixels;
    const float* last_input_address{nullptr};

  private:
    std::vector<float> fpn_hidden_[3];
    std::vector<float> fpn_position_[3];
    std::vector<float> tracker_hidden_[3];
    std::vector<float> tracker_position_[3];
    std::shared_ptr<void> keep_alive_;
};

class FakeDeviceSam3VisionModule final : public trtmc::TrtModule {
  public:
    FakeDeviceSam3VisionModule() {
        try {
            for (int32_t level = 0; level < 3; ++level) {
                allocate_output("sam3_fpn_hidden_" + std::to_string(level));
                allocate_output("sam3_fpn_position_" + std::to_string(level), 1);
                allocate_output("sam3_tracker_feature_" + std::to_string(level));
            }
            allocate_output("sam3_tracker_position_2");
        } catch (...) {
            release_outputs();
            throw;
        }
    }

    ~FakeDeviceSam3VisionModule() override { release_outputs(); }

    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        forward_async(inputs);
        sync();
        return {};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& inputs) override {
        const auto image = inputs.find("pixel_values");
        if (image == inputs.end() || image->second.data == nullptr ||
            image->second.shape != std::vector<int64_t>({1, 3, 4, 4}))
            throw std::runtime_error("invalid fake device SAM3 image input");
        last_input_address = static_cast<const float*>(image->second.data);
        const int32_t call_id = ++calls;
        write_outputs_for_call(call_id);
    }
    void sync() override {
        ++sync_calls;
        if (cudaStreamSynchronize(nullptr) != cudaSuccess)
            throw std::runtime_error("fake device SAM3 vision synchronization failed");
    }
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"pixel_values", tensor_shape("pixel_values"), trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        std::vector<trtmc::TensorInfo> infos;
        infos.reserve(outputs_.size());
        for (const auto& [name, pointer] : outputs_) {
            (void)pointer;
            infos.push_back({name, tensor_shape(name), trtmc::DType::kFloat32, false});
        }
        return infos;
    }
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override { return outputs_.count(name) != 0; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "pixel_values")
            return {1, 3, 4, 4};
        if (outputs_.count(name) != 0)
            return {1, 1, 1, 1};
        return {};
    }
    std::vector<int64_t>
    input_profile_shape(const std::string& name, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "pixel_values")
            return const_cast<std::byte*>(input_device_token_.data());
        const auto output = outputs_.find(name);
        return output == outputs_.end() ? nullptr : output->second;
    }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    int32_t calls{0};
    int32_t sync_calls{0};
    const float* last_input_address{nullptr};

  private:
    static float output_marker(const std::string& name, int32_t call_id) {
        int32_t offset = 0;
        if (name.rfind("sam3_fpn_hidden_", 0) == 0) {
            offset = 10 * (name.back() - '0');
        } else if (name.rfind("sam3_tracker_feature_", 0) == 0) {
            offset = 30 + 10 * (name.back() - '0');
        } else if (name.rfind("sam3_fpn_position_", 0) == 0) {
            offset = 60 + 10 * (name.back() - '0');
        } else if (name == "sam3_tracker_position_2") {
            offset = 90;
        }
        return static_cast<float>(call_id * 100 + offset);
    }

    void write_outputs_for_call(int32_t call_id) {
        for (const auto& [name, pointer] : outputs_) {
            const float marker = output_marker(name, call_id);
            if (cudaMemcpy(pointer, &marker, sizeof(marker), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                throw std::runtime_error("fake device SAM3 vision output fill failed");
            }
        }
    }

    void allocate_output(const std::string& name, int32_t values = 1) {
        void* pointer = nullptr;
        if (values <= 0 ||
            cudaMalloc(&pointer, static_cast<std::size_t>(values) * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("fake device SAM3 vision allocation failed");
        }
        owned_output_allocations_.push_back(pointer);
        outputs_.emplace(name, pointer);
    }

    void release_outputs() noexcept {
        for (void* pointer : owned_output_allocations_)
            (void)cudaFree(pointer);
        owned_output_allocations_.clear();
        outputs_.clear();
    }

    std::array<std::byte, 1024> input_device_token_{};
    std::vector<void*> owned_output_allocations_;
    std::unordered_map<std::string, void*> outputs_;
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3CoreModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto text_it = inputs.find("sam3_text_features");
        const auto mask_it = inputs.find("sam3_text_attention_mask");
        if (text_it == inputs.end() || mask_it == inputs.end() || !text_it->second.data ||
            !mask_it->second.data) {
            throw std::runtime_error("missing SAM3 core text inputs");
        }
        saw_text_shape = text_it->second.shape == std::vector<int64_t>({1, 4, 1});
        const auto* mask = static_cast<const int32_t*>(mask_it->second.data);
        saw_mask = mask[0] == 1 && mask[1] == 1 && mask[2] == 0 && mask[3] == 0;
        saw_vision_inputs = true;
        for (int32_t level = 0; level < 3; ++level) {
            saw_vision_inputs = saw_vision_inputs &&
                                inputs.count("sam3_fpn_hidden_" + std::to_string(level)) == 1 &&
                                inputs.count("sam3_fpn_position_" + std::to_string(level)) == 1;
        }

        pred_masks_ =
            three_detections ? std::vector<float>{2.0F,  -2.0F, -2.0F, -2.0F, -2.0F, 2.0F,
                                                  -2.0F, -2.0F, -2.0F, -2.0F, 2.0F,  -2.0F}
            : two_detections
                ? std::vector<float>{2.0F, 2.0F, -2.0F, -2.0F, -2.0F, 2.0F, 2.0F, -2.0F}
                : std::vector<float>{-1.0F, -1.0F, -1.0F, -1.0F, -2.0F, 2.0F, 2.0F, -2.0F};
        if (cleanup_probe_detection_masks) {
            // Query zero contains a one-pixel hole. fill_hole_area=1 must
            // change its final logit from -2.0 to +0.1. Query one keeps a
            // foreground pixel at that location so the public overlap result
            // distinguishes correct cleanup from a skipped cleanup.
            pred_masks_ = three_detections
                              ? std::vector<float>{
                                    2.0F,  2.0F,  2.0F,  -2.0F,
                                    -2.0F, -2.0F, -2.0F, 2.0F,
                                    -2.0F, -2.0F, 2.0F,  -2.0F,
                                }
                              : std::vector<float>{
                                    2.0F, 2.0F, 2.0F, -2.0F,
                                    -2.0F, -2.0F, -2.0F, 2.0F,
                                };
        }
        if (late_hard_conditioning_probe) {
            pred_masks_ = {-2.0F, -2.0F, -2.0F, 2.0F, 2.0F, 2.0F, 2.0F, -2.0F};
        }
        if (three_overlap_detections) {
            pred_masks_ = {2.0F,  -2.0F, -2.0F, -2.0F, 2.0F, 2.0F,
                           -2.0F, -2.0F, 2.0F,  -2.0F, 2.0F, -2.0F};
        }
        if (empty_first_detection_mask) {
            std::fill_n(pred_masks_.begin(), 4, -2.0F);
        }
        if (threshold_boundary_detection_mask && !two_detections && !three_detections) {
            pred_masks_[4] = 0.5F;
            pred_masks_[5] = 0.49F;
            pred_masks_[6] = -0.25F;
            pred_masks_[7] = -0.25F;
        }
        pred_boxes_ = three_detections
                          ? std::vector<float>{0.25F, 0.25F, 0.25F, 0.25F, 0.75F, 0.25F,
                                               0.25F, 0.25F, 0.25F, 0.75F, 0.25F, 0.25F}
                          : std::vector<float>{0.0F, 0.0F, 1.0F, 1.0F, 0.25F, 0.5F, 0.75F, 1.0F};
        pred_logits_ = three_detections       ? std::vector<float>{2.0F, 2.0F, 2.0F}
                       : tie_detection_scores ? std::vector<float>{2.0F, 2.0F}
                                              : std::vector<float>{0.0F, 2.0F};
        if (cleanup_probe_detection_masks && three_detections)
            pred_logits_ = {0.0F, 1.0F, 2.0F};
        if (three_overlap_detections)
            pred_logits_ = {0.0F, 1.0F, 2.0F};
        if (detections_first_frame_only && calls > 1)
            std::fill(pred_logits_.begin(), pred_logits_.end(), -20.0F);
        if (second_detection_first_frame_only && calls > 1)
            pred_logits_[1] = -20.0F;
        if (late_hard_conditioning_probe && calls == 1)
            pred_logits_[1] = -20.0F;
        presence_logits_ = {2.0F};
        trtmc::Tensor masks;
        masks.data = pred_masks_.data();
        masks.shape = {1, three_detections ? 3 : 2, 2, 2};
        masks.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor boxes;
        boxes.data = pred_boxes_.data();
        boxes.shape = {1, three_detections ? 3 : 2, 4};
        boxes.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor logits;
        logits.data = pred_logits_.data();
        logits.shape = {1, three_detections ? 3 : 2};
        logits.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor presence;
        presence.data = presence_logits_.data();
        presence.shape = {1, 1};
        presence.dtype = trtmc::DType::kFloat32;

        return {{"pred_masks", masks},
                {"pred_boxes", boxes},
                {"pred_logits", logits},
                {"presence_logits", presence}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        for (int32_t level = 0; level < 3; ++level) {
            const auto suffix = std::to_string(level);
            if (name == "sam3_fpn_hidden_" + suffix || name == "sam3_fpn_position_" + suffix) {
                return true;
            }
        }
        return false;
    }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return has_input(name) ? std::vector<int64_t>{1, 1, 1, 1} : std::vector<int64_t>{};
    }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto input = bound_inputs_.find(name);
        return input == bound_inputs_.end() ? nullptr : input->second;
    }
    void bind_external(const std::string& name, void* ptr) override {
        if (has_input(name))
            bound_inputs_[name] = ptr;
    }
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_text_shape{false};
    bool saw_mask{false};
    bool saw_vision_inputs{false};
    bool two_detections{false};
    bool three_detections{false};
    bool three_overlap_detections{false};
    bool empty_first_detection_mask{false};
    bool tie_detection_scores{false};
    bool detections_first_frame_only{false};
    bool second_detection_first_frame_only{false};
    bool threshold_boundary_detection_mask{false};
    bool cleanup_probe_detection_masks{false};
    bool late_hard_conditioning_probe{false};
    int32_t calls{0};

  private:
    std::vector<float> pred_masks_;
    std::vector<float> pred_boxes_;
    std::vector<float> pred_logits_;
    std::vector<float> presence_logits_;
    std::unordered_map<std::string, void*> bound_inputs_;
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3TrackerModule : public trtmc::TrtModule {
  public:
    explicit FakeSam3TrackerModule(bool recurrent, bool memory_only = false,
                                   float object_score_logit = 2.0F, bool device_recurrent = false,
                                   bool own_stream = false, std::size_t device_batch_size = 1,
                                   bool hard_memory = false)
        : recurrent_(recurrent), memory_only_(memory_only), object_score_logit_(object_score_logit),
          device_recurrent_(device_recurrent), device_batch_size_(device_batch_size),
          hard_memory_(hard_memory) {
        if (own_stream && cudaStreamCreate(&owned_stream_) != cudaSuccess) {
            throw std::runtime_error("fake SAM3 tracker stream creation failed");
        }
        if (!device_recurrent_)
            return;
        try {
            if (recurrent_) {
                allocate_device_tensor("memory_features", device_batch_size_ * 4 * 64);
                allocate_device_tensor("memory_position", device_batch_size_ * 4 * 64);
                allocate_device_tensor("pred_masks", device_batch_size_ * 4);
                allocate_device_tensor("object_pointer", device_batch_size_ * 256);
                allocate_device_tensor("object_score_logits", device_batch_size_);
                allocate_device_tensor("selected_iou", device_batch_size_);
            }
            if (memory_only_) {
                allocate_device_tensor("new_memory_features", device_batch_size_ * 4 * 64);
                allocate_device_tensor("new_memory_position", device_batch_size_ * 4 * 64);
            }
        } catch (...) {
            release_device_tensors();
            if (owned_stream_ != nullptr)
                (void)cudaStreamDestroy(owned_stream_);
            throw;
        }
    }

    ~FakeSam3TrackerModule() override {
        release_device_tensors();
        if (owned_stream_ != nullptr)
            (void)cudaStreamDestroy(owned_stream_);
    }

    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (on_forward)
            on_forward();
        std::size_t batch_size = 1;
        std::vector<float> downloaded_device_memory;
        std::vector<float> downloaded_device_position;
        if (memory_only_) {
            const char* mask_name = hard_memory_ ? "owned_tracker_mask" : "final_mask";
            const auto suppression = inputs.find("suppress_area_shrinkage");
            if (hard_memory_ && suppression != inputs.end())
                throw std::runtime_error("SAM3 hard memory received a soft suppression flag");
            if (!hard_memory_ && suppression == inputs.end())
                throw std::runtime_error("SAM3 soft memory is missing its suppression flag");
            saw_features = inputs.count("tracker_feature_2") == 1 && inputs.count(mask_name) == 1 &&
                           inputs.count("object_score_logits") == 1 &&
                           (hard_memory_ || suppression != inputs.end());
            const auto bound_feature = bound_inputs_.find("tracker_feature_2");
            if (bound_feature != bound_inputs_.end()) {
                float marker = 0.0F;
                if (cudaMemcpy(&marker, bound_feature->second, sizeof(marker),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("fake SAM3 memory feature marker download failed");
                }
                vision_feature_markers.push_back(marker);
            }
            const auto& final_mask = inputs.at(mask_name);
            const auto* final_mask_values = static_cast<const float*>(final_mask.data);
            final_mask_input_addresses.push_back(final_mask_values);
            batch_size = final_mask.shape.size() == 4
                             ? static_cast<std::size_t>(final_mask.shape.front())
                             : 1;
            if (batch_size == 0 || final_mask.numel() % batch_size != 0)
                throw std::runtime_error("invalid SAM3 memory encoder batch");
            memory_batch_sizes.push_back(batch_size);
            const auto item_values = final_mask.numel() / batch_size;
            const auto* score_values =
                static_cast<const float*>(inputs.at("object_score_logits").data);
            const auto* suppression_values =
                hard_memory_ ? nullptr : static_cast<const int32_t*>(suppression->second.data);
            for (std::size_t batch = 0; batch < batch_size; ++batch) {
                last_final_mask.assign(final_mask_values + batch * item_values,
                                       final_mask_values + (batch + 1) * item_values);
                last_memory_score = score_values[batch];
                final_masks_history.push_back(last_final_mask);
                memory_scores_history.push_back(last_memory_score);
                if (suppression_values != nullptr)
                    memory_suppressions_history.push_back(suppression_values[batch]);
            }
        } else if (recurrent_) {
            saw_features =
                inputs.count("tracker_feature_0") == 1 && inputs.count("tracker_feature_1") == 1 &&
                inputs.count("tracker_feature_2") == 1 && inputs.count("tracker_position_2") == 1;
            shared_features_batch_one = saw_features;
            for (int32_t level = 0; level < 3 && shared_features_batch_one; ++level) {
                const auto& shape = inputs.at("tracker_feature_" + std::to_string(level)).shape;
                shared_features_batch_one = !shape.empty() && shape.front() == 1;
            }
            const auto memory_offset = inputs.find("memory_temporal_offsets");
            const auto memory = inputs.find("memory_features");
            const auto memory_position = inputs.find("memory_position");
            const auto pointers = inputs.find("object_pointers");
            const auto pointer_offset = inputs.find("object_pointer_temporal_offsets");
            const auto max_pointers = inputs.find("max_object_pointers_to_use");
            if (memory == inputs.end() || memory_position == inputs.end() ||
                memory_offset == inputs.end() || pointers == inputs.end() ||
                pointer_offset == inputs.end() || max_pointers == inputs.end()) {
                throw std::runtime_error("missing SAM3 recurrent policy inputs");
            }
            const bool batch_leading_layout =
                memory->second.shape.size() == 4 && memory_offset->second.shape.size() == 2 &&
                pointers->second.shape.size() == 3 && pointer_offset->second.shape.size() == 2;
            if (!batch_leading_layout)
                throw std::runtime_error("invalid SAM3 recurrent step ranks");
            batch_size = static_cast<std::size_t>(memory->second.shape.front());
            const auto expected_batch = static_cast<int64_t>(device_batch_size_);
            const bool matching_dimensions =
                memory->second.shape.front() == expected_batch &&
                memory_position->second.shape == memory->second.shape &&
                memory_offset->second.shape.front() == expected_batch &&
                pointers->second.shape.front() == expected_batch &&
                pointer_offset->second.shape.front() == expected_batch &&
                memory->second.shape[1] == memory_offset->second.shape[1] &&
                pointers->second.shape[1] == pointer_offset->second.shape[1] &&
                memory->second.shape[3] == 64 && pointers->second.shape[2] == 256;
            if (batch_size != device_batch_size_ || !matching_dimensions)
                throw std::runtime_error("invalid SAM3 recurrent step batch shape");
            batch_sizes.push_back(batch_size);
            last_memory_shape = memory->second.shape;
            last_memory_offset_shape = memory_offset->second.shape;
            last_pointer_shape = pointers->second.shape;
            last_pointer_offset_shape = pointer_offset->second.shape;
            const auto* memory_values = static_cast<const int32_t*>(memory_offset->second.data);
            last_memory_offsets.assign(memory_values,
                                       memory_values + memory_offset->second.numel());
            memory_offsets_history.push_back(last_memory_offsets);
            const float* memory_features = static_cast<const float*>(memory->second.data);
            const float* memory_positions = static_cast<const float*>(memory_position->second.data);
            if (memory_features == nullptr) {
                saw_device_memory_input = true;
                downloaded_device_memory.resize(memory->second.numel());
                if (cudaMemcpy(downloaded_device_memory.data(),
                               device_tensors_.at("memory_features"),
                               downloaded_device_memory.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("fake SAM3 device memory download failed");
                }
                memory_features = downloaded_device_memory.data();
            }
            if (memory_positions == nullptr) {
                saw_device_position_input = true;
                downloaded_device_position.resize(memory_position->second.numel());
                if (cudaMemcpy(downloaded_device_position.data(),
                               device_tensors_.at("memory_position"),
                               downloaded_device_position.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("fake SAM3 device memory position download failed");
                }
                memory_positions = downloaded_device_position.data();
            }
            last_memory_value = *memory_features;
            last_memory_position_value = *memory_positions;
            last_memory_frame_values.clear();
            last_memory_position_frame_values.clear();
            const std::size_t frame_count = memory_offset->second.numel();
            const std::size_t values_per_frame = memory->second.numel() / frame_count;
            for (std::size_t frame = 0; frame < frame_count; ++frame) {
                last_memory_frame_values.push_back(memory_features[frame * values_per_frame]);
                last_memory_position_frame_values.push_back(
                    memory_positions[frame * values_per_frame]);
            }
            const auto* pointer_values = static_cast<const int32_t*>(pointer_offset->second.data);
            last_pointer_offsets.assign(pointer_values,
                                        pointer_values + pointer_offset->second.numel());
            const auto* pointer_features = static_cast<const float*>(pointers->second.data);
            last_pointer_frame_values.clear();
            const std::size_t pointer_count = pointers->second.numel() / 256U;
            for (std::size_t pointer = 0; pointer < pointer_count; ++pointer)
                last_pointer_frame_values.push_back(pointer_features[pointer * 256U]);
            last_max_pointers = *static_cast<const int32_t*>(max_pointers->second.data);
        } else {
            saw_features =
                inputs.count("tracker_feature_0") == 1 && inputs.count("tracker_feature_1") == 1 &&
                inputs.count("tracker_feature_2") == 1 && inputs.count("detector_mask") == 1;
            const auto& detector_mask = inputs.at("detector_mask");
            const auto* detector_mask_values = static_cast<const float*>(detector_mask.data);
            last_detector_mask.assign(detector_mask_values,
                                      detector_mask_values + detector_mask.numel());
            detector_masks_history.push_back(last_detector_mask);
        }

        pred_masks_.clear();
        selected_iou_.clear();
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const std::size_t logical_call = recurrent_items_ + batch + 1;
            std::vector<float> item_mask{-2.0F, 2.0F, 2.0F, -2.0F};
            if (recurrent_ && !recurrent_mask_override.empty())
                item_mask = recurrent_mask_override;
            if (recurrent_ && logical_call <= recurrent_masks_by_item.size())
                item_mask = recurrent_masks_by_item[logical_call - 1];
            if (recurrent_ && contained_pair_masks) {
                item_mask = logical_call % 2 == 1 ? std::vector<float>{2.0F, -2.0F, -2.0F, -2.0F}
                                                  : std::vector<float>{3.0F, 3.0F, 3.0F, 3.0F};
            }
            if (recurrent_ && scripted_occlusion && (logical_call == 1 || logical_call == 4)) {
                item_mask = {-2.0F, -2.0F, -2.0F, -2.0F};
            }
            pred_masks_.insert(pred_masks_.end(), item_mask.begin(), item_mask.end());
            const float selected_iou = logical_call <= scripted_selected_ious.size()
                                           ? scripted_selected_ious[logical_call - 1]
                                           : 1.0F;
            selected_iou_.push_back(selected_iou);
        }
        if (recurrent_)
            recurrent_items_ += batch_size;
        object_pointer_.assign(batch_size * 256, pointer_value);
        object_score_.assign(batch_size, object_score_logit_);
        memory_.clear();
        if (memory_only_) {
            for (std::size_t batch = 0; batch < batch_size; ++batch) {
                memory_.insert(memory_.end(), 4 * 64,
                               memory_value_base + static_cast<float>(calls) +
                                   static_cast<float>(batch) + 0.3333F);
            }
        } else {
            memory_.assign(4 * 64, 0.125F);
        }
        memory_position_.assign(batch_size * 4 * 64, 0.6667F);

        trtmc::Tensor mask;
        mask.data = pred_masks_.data();
        mask.shape = {static_cast<int64_t>(batch_size), 1, 2, 2};
        mask.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor pointer;
        pointer.data = object_pointer_.data();
        pointer.shape = {static_cast<int64_t>(batch_size), 1, 256};
        pointer.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor score;
        score.data = object_score_.data();
        score.shape = {static_cast<int64_t>(batch_size), 1, 1};
        score.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor selected_iou;
        selected_iou.data = selected_iou_.data();
        selected_iou.shape = {static_cast<int64_t>(batch_size), 1, 1};
        selected_iou.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor memory;
        memory.data = memory_.data();
        memory.shape = batch_size == 1
                           ? std::vector<int64_t>{4, 1, 64}
                           : std::vector<int64_t>{static_cast<int64_t>(batch_size), 4, 64};
        memory.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor position;
        position.data = memory_position_.data();
        position.shape = memory.shape;
        position.dtype = trtmc::DType::kFloat32;

        trtmc::TensorMap outputs;
        if (!memory_only_) {
            outputs = {
                {"pred_masks", mask}, {"object_pointer", pointer}, {"object_score_logits", score}};
            if (recurrent_)
                outputs["selected_iou"] = selected_iou;
        }
        if (memory_only_) {
            outputs["new_memory_features"] = memory;
            outputs["new_memory_position"] = position;
        } else if (!recurrent_) {
            outputs["memory_features"] = memory;
            outputs["memory_position"] = position;
        }
        return outputs;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& inputs) override {
        if (!device_recurrent_)
            return;
        ++async_calls;
        const auto outputs = forward(inputs);
        for (const auto& [name, output] : outputs) {
            const auto destination = device_tensors_.find(name);
            if (destination == device_tensors_.end())
                continue;
            if (cudaMemcpyAsync(destination->second, output.data, output.numel() * sizeof(float),
                                cudaMemcpyHostToDevice, owned_stream_) != cudaSuccess) {
                throw std::runtime_error("fake SAM3 tracker async output upload failed");
            }
        }
    }
    void sync() override {
        if (!device_recurrent_)
            return;
        ++sync_calls;
        if (cudaStreamSynchronize(owned_stream_) != cudaSuccess)
            throw std::runtime_error("fake SAM3 tracker synchronization failed");
    }
    cudaStream_t stream() const override { return owned_stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return is_vision_input(name) || device_tensors_.count(name) != 0 ||
               (recurrent_ &&
                (name == "memory_temporal_offsets" || name == "object_pointer_temporal_offsets" ||
                 name == "max_object_pointers_to_use"));
    }
    bool has_output(const std::string& name) const override {
        if (recurrent_ && name == "selected_iou")
            return true;
        return name == "pred_masks" || name == "object_pointer" || name == "object_score_logits" ||
                       name == "selected_iou" || name == "new_memory_features" ||
                       name == "new_memory_position"
                   ? device_tensors_.count(name) != 0
                   : false;
    }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "memory_features" || name == "memory_position")
            return {static_cast<int64_t>(device_batch_size_), 1, 4, 64};
        if (name == "new_memory_features" || name == "new_memory_position") {
            if (device_batch_size_ == 1)
                return {4, 1, 64};
            return {static_cast<int64_t>(device_batch_size_), 4, 64};
        }
        if (name == "pred_masks")
            return {static_cast<int64_t>(device_batch_size_), 1, 2, 2};
        if (name == "object_pointer")
            return {static_cast<int64_t>(device_batch_size_), 1, 256};
        if (name == "object_score_logits" || name == "selected_iou")
            return {static_cast<int64_t>(device_batch_size_), 1, 1};
        return is_vision_input(name) ? std::vector<int64_t>{1, 1, 1, 1} : std::vector<int64_t>{};
    }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto device = device_tensors_.find(name);
        if (device != device_tensors_.end())
            return device->second;
        const auto input = bound_inputs_.find(name);
        return input == bound_inputs_.end() ? nullptr : input->second;
    }
    void bind_external(const std::string& name, void* ptr) override {
        if (is_vision_input(name))
            bound_inputs_[name] = ptr;
    }
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    int32_t calls{0};
    bool saw_features{false};
    bool shared_features_batch_one{false};
    std::vector<std::size_t> batch_sizes;
    std::vector<int64_t> last_memory_shape;
    std::vector<int64_t> last_memory_offset_shape;
    std::vector<int64_t> last_pointer_shape;
    std::vector<int64_t> last_pointer_offset_shape;
    std::vector<int32_t> last_memory_offsets;
    std::vector<std::vector<int32_t>> memory_offsets_history;
    std::vector<int32_t> last_pointer_offsets;
    std::vector<float> last_pointer_frame_values;
    int32_t last_max_pointers{0};
    float last_memory_value{0.0F};
    std::vector<float> last_memory_frame_values;
    float last_memory_position_value{0.0F};
    std::vector<float> last_memory_position_frame_values;
    std::vector<float> last_final_mask;
    std::vector<std::vector<float>> final_masks_history;
    std::vector<const float*> final_mask_input_addresses;
    std::vector<float> last_detector_mask;
    std::vector<std::vector<float>> detector_masks_history;
    std::vector<float> memory_scores_history;
    std::vector<int32_t> memory_suppressions_history;
    std::vector<std::size_t> memory_batch_sizes;
    std::vector<float> vision_feature_markers;
    float memory_value_base{0.0F};
    float last_memory_score{0.0F};
    bool scripted_occlusion{false};
    bool contained_pair_masks{false};
    std::vector<float> recurrent_mask_override;
    std::vector<std::vector<float>> recurrent_masks_by_item;
    std::vector<float> scripted_selected_ious;
    float pointer_value{0.25F};
    bool saw_device_memory_input{false};
    bool saw_device_position_input{false};
    int32_t async_calls{0};
    int32_t sync_calls{0};
    std::function<void()> on_forward;

  private:
    void allocate_device_tensor(const std::string& name, std::size_t values) {
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, values * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("fake SAM3 tracker device allocation failed");
        device_tensors_.emplace(name, pointer);
    }

    void release_device_tensors() noexcept {
        for (const auto& [name, pointer] : device_tensors_) {
            (void)name;
            (void)cudaFree(pointer);
        }
        device_tensors_.clear();
    }

    bool is_vision_input(const std::string& name) const {
        if (memory_only_)
            return name == "tracker_feature_2";
        for (int32_t level = 0; level < 3; ++level) {
            if (name == "tracker_feature_" + std::to_string(level))
                return true;
        }
        return recurrent_ && name == "tracker_position_2";
    }

    bool recurrent_{false};
    bool memory_only_{false};
    float object_score_logit_{2.0F};
    bool device_recurrent_{false};
    std::size_t device_batch_size_{1};
    bool hard_memory_{false};
    std::size_t recurrent_items_{0};
    std::vector<float> pred_masks_;
    std::vector<float> object_pointer_;
    std::vector<float> object_score_;
    std::vector<float> selected_iou_;
    std::vector<float> memory_;
    std::vector<float> memory_position_;
    std::unordered_map<std::string, void*> bound_inputs_;
    std::unordered_map<std::string, void*> device_tensors_;
    cudaStream_t owned_stream_{nullptr};
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3HardMaskResizeModule final : public FakeSam3TrackerModule {
  public:
    explicit FakeSam3HardMaskResizeModule(std::size_t expected_batch)
        : FakeSam3TrackerModule(false), expected_batch_(expected_batch) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto found = inputs.find("tracker_mask");
        if (found == inputs.end() || found->second.dtype != trtmc::DType::kFloat32 ||
            found->second.data == nullptr || found->second.shape.size() != 4 ||
            found->second.shape[0] != static_cast<int64_t>(expected_batch_) ||
            found->second.shape[1] != 1) {
            throw std::runtime_error("invalid fake SAM3 hard-mask resize input");
        }
        const auto source_height = static_cast<std::size_t>(found->second.shape[2]);
        const auto source_width = static_cast<std::size_t>(found->second.shape[3]);
        constexpr std::size_t output_size = 4;
        const auto* source = static_cast<const float*>(found->second.data);
        const auto output_values = expected_batch_ * output_size * output_size;
        if (!scripted_output.empty()) {
            if (scripted_output.size() != output_values)
                throw std::runtime_error("invalid scripted SAM3 hard-mask resize output");
            output_ = scripted_output;
        } else {
            output_.assign(output_values, 0.0F);
            for (std::size_t batch = 0; batch < expected_batch_; ++batch) {
                for (std::size_t y = 0; y < output_size; ++y) {
                    for (std::size_t x = 0; x < output_size; ++x) {
                        const auto source_y =
                            std::min(y * source_height / output_size, source_height - 1);
                        const auto source_x =
                            std::min(x * source_width / output_size, source_width - 1);
                        output_[(batch * output_size + y) * output_size + x] =
                            source[(batch * source_height + source_y) * source_width + source_x];
                    }
                }
            }
        }
        ++calls;
        batch_sizes.push_back(expected_batch_);
        trtmc::Tensor output;
        output.data = output_.data();
        output.shape = {static_cast<int64_t>(expected_batch_), 1, 4, 4};
        output.dtype = trtmc::DType::kFloat32;
        return {{"resized_tracker_mask", output}};
    }

    int32_t calls{0};
    std::vector<std::size_t> batch_sizes;
    std::vector<float> scripted_output;

  private:
    std::size_t expected_batch_{0};
    std::vector<float> output_;
};

trtmc::Sam3Config make_config() {
    trtmc::Sam3Config cfg;
    cfg.text_max_position_embeddings = 4;
    cfg.text_pad_token_id = 0;
    cfg.image_size = 4;
    cfg.low_res_mask_size = 2;
    return cfg;
}

struct ParallelTrackerInitSchedule {
    void rendezvous() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [this] { return entered == 2; }))
            throw std::runtime_error("fake SAM3 tracker-init lanes did not overlap");
    }

    std::mutex mutex;
    std::condition_variable cv;
    int32_t entered{0};
};

struct BlockedTrackerInitSchedule {
    void enter_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered;
        changed.notify_all();
        changed.wait(lock, [this] { return released; });
    }

    bool wait_for_both() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [this] { return entered == 2; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        changed.notify_all();
    }

    std::mutex mutex;
    std::condition_variable changed;
    int32_t entered{0};
    bool released{false};
};

struct ParallelTrackerInitFixture {
    std::unique_ptr<trtmc::Sam3Pipeline> pipeline;
    FakeDeviceSam3VisionModule* vision{nullptr};
    FakeSam3TrackerModule* init{nullptr};
    FakeSam3TrackerModule* sibling{nullptr};
};

ParallelTrackerInitFixture
make_parallel_tracker_init_fixture(const std::shared_ptr<ParallelTrackerInitSchedule>& schedule,
                                   int32_t fill_hole_area = 0) {
    auto text = std::make_unique<FakeSam3TextModule>();
    auto vision = std::make_unique<FakeDeviceSam3VisionModule>();
    auto* vision_ptr = vision.get();
    auto core = std::make_unique<FakeSam3CoreModule>();
    core->two_detections = true;
    core->tie_detection_scores = true;
    core->cleanup_probe_detection_masks = fill_hole_area > 0;
    auto init = std::make_unique<FakeSam3TrackerModule>(false, false, 2.0F, false, true);
    auto* init_ptr = init.get();
    auto sibling = std::make_unique<FakeSam3TrackerModule>(false, false, 2.0F, false, true);
    auto* sibling_ptr = sibling.get();
    if (schedule != nullptr) {
        init->on_forward = [schedule] { schedule->rendezvous(); };
        sibling->on_forward = [schedule] { schedule->rendezvous(); };
    }
    auto step = std::make_unique<FakeSam3TrackerModule>(true);
    auto memory = std::make_unique<FakeSam3TrackerModule>(false, true);
    auto hard_memory =
        std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, false, false, 1, true);
    auto hard_memory_batch2 =
        std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, false, false, 2, true);
    auto hard_mask_resize = std::make_unique<FakeSam3HardMaskResizeModule>(1);
    auto hard_mask_resize_batch2 = std::make_unique<FakeSam3HardMaskResizeModule>(2);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto config = make_config();
    config.fill_hole_area = fill_hole_area;
    config.detection_threshold = 0.4F;
    config.new_detection_threshold = 0.4F;
    config.detection_nms_threshold = 0.0F;
    auto pipeline = std::make_unique<trtmc::Sam3Pipeline>(
        std::move(text), std::move(vision), std::move(core), tokenizer, config, "facebook/sam3",
        std::move(init), std::move(step), std::move(memory), nullptr, nullptr, std::move(sibling),
        std::move(hard_memory), std::move(hard_memory_batch2), std::move(hard_mask_resize),
        std::move(hard_mask_resize_batch2));
    return {std::move(pipeline), vision_ptr, init_ptr, sibling_ptr};
}

bool same_video_result(const trtmc::Sam3VideoFrameResult& lhs,
                       const trtmc::Sam3VideoFrameResult& rhs) {
    return lhs.frame_idx == rhs.frame_idx && lhs.height == rhs.height && lhs.width == rhs.width &&
           lhs.object_ids == rhs.object_ids && lhs.masks == rhs.masks &&
           lhs.detection_scores == rhs.detection_scores &&
           lhs.tracker_scores == rhs.tracker_scores && lhs.boxes == rhs.boxes &&
           lhs.removed_object_ids == rhs.removed_object_ids &&
           lhs.suppressed_object_ids == rhs.suppressed_object_ids;
}

std::uint64_t float_bit_hash(const std::vector<float>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const float value : values) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void test_preprocess_matches_customer_meta_pillow_resize() {
    // Oracles were produced through the customer Meta image-folder loader in
    // the pinned L4 environment. They cover Pillow's fixed-22 bilinear uint8
    // resize followed by Meta's FP16 ToTensor/subtract/divide order, expanded
    // to the runtime's FP32 TensorRT input in CHW order.
    trtmc::Sam3Config config;
    config.image_size = 1008;
    config.image_mean = {0.5F, 0.5F, 0.5F};
    config.image_std = {0.5F, 0.5F, 0.5F};

    {
        constexpr int32_t height = 96;
        constexpr int32_t width = 128;
        std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const int32_t value = (y * 37 + x * 17 + channel * 53 + (x * y) % 251) & 255;
                    pixels[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
                        static_cast<float>(value) / 255.0F;
                }
            }
        }
        const auto output = trtmc::preprocess_sam3_image(pixels.data(), height, width, config);
        const auto output_hash = float_bit_hash(output);
        check(output_hash == 0x099ca8c2a291e783ULL,
              "sam3 96x128 preprocessing is bit-exact with customer Meta/Pillow oracle");
    }

    {
        constexpr int32_t height = 1152;
        constexpr int32_t width = 1344;
        std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const std::array<int32_t, 3> values = {
                    ((x + y) & 1) * 255,
                    (x & 1) * 255,
                    (((x / 2 + y / 3) & 1) * 255),
                };
                for (int32_t channel = 0; channel < 3; ++channel) {
                    pixels[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
                        static_cast<float>(values[static_cast<std::size_t>(channel)]) / 255.0F;
                }
            }
        }
        const auto output = trtmc::preprocess_sam3_image(pixels.data(), height, width, config);
        const auto output_hash = float_bit_hash(output);
        check(output_hash == 0x92ddbd5d8ae52783ULL,
              "sam3 high-frequency downsample is bit-exact with customer Meta/Pillow oracle");
    }

    {
        constexpr int32_t height = 1013;
        constexpr int32_t width = 1031;
        std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const int32_t value = (y * 37 + x * 17 + channel * 53 + (x * y) % 251) & 255;
                    pixels[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
                        static_cast<float>(value) / 255.0F;
                }
            }
        }
        const auto output = trtmc::preprocess_sam3_image(pixels.data(), height, width, config);
        check(float_bit_hash(output) == 0xb640d1d167f10783ULL,
              "sam3 odd downsample is bit-exact with customer Meta/Pillow oracle");
    }
}

void test_preprocess_uses_explicit_fma_at_uint8_half_steps() {
    trtmc::Sam3Config config;
    config.image_size = 16;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    constexpr int32_t size = 16;
    std::vector<float> pixels(static_cast<std::size_t>(size * size * 3));
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const auto quantized = static_cast<float>(index % 255U);
        const float half_step = (quantized + 0.5F) / 255.0F;
        switch (index % 5U) {
        case 0:
            pixels[index] = std::nextafter(half_step, -std::numeric_limits<float>::infinity());
            break;
        case 1:
            pixels[index] = half_step;
            break;
        case 2:
            pixels[index] = std::nextafter(half_step, std::numeric_limits<float>::infinity());
            break;
        case 3:
            pixels[index] = -0.0F;
            break;
        default:
            pixels[index] = 1.25F;
            break;
        }
    }

    const auto output = trtmc::preprocess_sam3_image(pixels.data(), size, size, config);
    bool exact = output.size() == pixels.size();
    const std::size_t plane = static_cast<std::size_t>(size * size);
    for (std::size_t pixel = 0; exact && pixel < plane; ++pixel) {
        for (std::size_t channel = 0; channel < 3U; ++channel) {
            const float source = pixels[pixel * 3U + channel];
            const auto quantized = static_cast<float>(
                std::clamp(static_cast<int>(
                               std::floor(std::fma(std::clamp(source, 0.0F, 1.0F), 255.0F, 0.5F))),
                           0, 255));
            // FP16 storage changes the normalized value, but by much less
            // than half the distance between adjacent uint8 buckets. This
            // tight comparison therefore still pins the selected bucket.
            const float expected = quantized / 255.0F;
            exact = std::fabs(output[channel * plane + pixel] - expected) < 1.0e-3F;
        }
    }
    check(exact, "sam3 uint8 quantization pins fused multiply-add at adversarial half steps");
}
struct VideoFixture {
    std::unique_ptr<trtmc::Sam3Pipeline> pipeline;
    FakeDeviceSam3VisionModule* vision{nullptr};
    FakeSam3CoreModule* core{nullptr};
    FakeSam3TrackerModule* init{nullptr};
    FakeSam3TrackerModule* step{nullptr};
    FakeSam3TrackerModule* step_batch2{nullptr};
    FakeSam3TrackerModule* memory{nullptr};
    FakeSam3TrackerModule* memory_batch2{nullptr};
    FakeSam3TrackerModule* hard_memory{nullptr};
    FakeSam3TrackerModule* hard_memory_batch2{nullptr};
    FakeSam3HardMaskResizeModule* hard_mask_resize{nullptr};
    FakeSam3HardMaskResizeModule* hard_mask_resize_batch2{nullptr};
};

VideoFixture make_video_fixture(std::size_t detections = 1, bool batch2 = false,
                                bool device_recurrent = false, float tracker_logit = 2.0F,
                                const std::function<void(trtmc::Sam3Config&)>& configure = {},
                                bool exact_hard_mask_resize = true) {
    auto text = std::make_unique<FakeSam3TextModule>();
    auto vision = std::make_unique<FakeDeviceSam3VisionModule>();
    auto* vision_ptr = vision.get();
    auto core = std::make_unique<FakeSam3CoreModule>();
    auto* core_ptr = core.get();
    core->two_detections = detections == 2;
    core->three_detections = detections == 3;
    core->tie_detection_scores = detections == 2;
    auto init = std::make_unique<FakeSam3TrackerModule>(false);
    auto* init_ptr = init.get();
    auto step = std::make_unique<FakeSam3TrackerModule>(true, false, tracker_logit,
                                                        device_recurrent, device_recurrent);
    auto* step_ptr = step.get();
    auto memory = std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, device_recurrent,
                                                          device_recurrent);
    auto* memory_ptr = memory.get();
    auto hard_memory = std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, device_recurrent,
                                                               device_recurrent, 1, true);
    auto* hard_memory_ptr = hard_memory.get();
    auto hard_memory_batch2 = std::make_unique<FakeSam3TrackerModule>(
        false, true, 2.0F, device_recurrent, device_recurrent, 2, true);
    auto* hard_memory_batch2_ptr = hard_memory_batch2.get();
    std::unique_ptr<FakeSam3HardMaskResizeModule> hard_mask_resize;
    std::unique_ptr<FakeSam3HardMaskResizeModule> hard_mask_resize_batch2;
    FakeSam3HardMaskResizeModule* hard_mask_resize_ptr = nullptr;
    FakeSam3HardMaskResizeModule* hard_mask_resize_batch2_ptr = nullptr;
    if (exact_hard_mask_resize) {
        hard_mask_resize = std::make_unique<FakeSam3HardMaskResizeModule>(1);
        hard_mask_resize_batch2 = std::make_unique<FakeSam3HardMaskResizeModule>(2);
        hard_mask_resize_ptr = hard_mask_resize.get();
        hard_mask_resize_batch2_ptr = hard_mask_resize_batch2.get();
    }

    std::unique_ptr<FakeSam3TrackerModule> step_batch2;
    std::unique_ptr<FakeSam3TrackerModule> memory_batch2;
    FakeSam3TrackerModule* step_batch2_ptr = nullptr;
    FakeSam3TrackerModule* memory_batch2_ptr = nullptr;
    if (batch2) {
        step_batch2 = std::make_unique<FakeSam3TrackerModule>(
            true, false, tracker_logit, device_recurrent, device_recurrent, 2);
        memory_batch2 = std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, device_recurrent,
                                                                device_recurrent, 2);
        step_batch2_ptr = step_batch2.get();
        memory_batch2_ptr = memory_batch2.get();
    }

    auto config = make_config();
    config.fill_hole_area = 0;
    if (configure)
        configure(config);
    if (detections > 1) {
        config.detection_threshold = 0.4F;
        config.new_detection_threshold = 0.4F;
        config.detection_nms_threshold = 0.0F;
    }
    auto pipeline = std::make_unique<trtmc::Sam3Pipeline>(
        std::move(text), std::move(vision), std::move(core), std::make_shared<FakeTokenizer>(),
        config, "facebook/sam3", std::move(init), std::move(step), std::move(memory),
        std::move(step_batch2), std::move(memory_batch2), nullptr, std::move(hard_memory),
        std::move(hard_memory_batch2), std::move(hard_mask_resize),
        std::move(hard_mask_resize_batch2));
    return {std::move(pipeline),
            vision_ptr,
            core_ptr,
            init_ptr,
            step_ptr,
            step_batch2_ptr,
            memory_ptr,
            memory_batch2_ptr,
            hard_memory_ptr,
            hard_memory_batch2_ptr,
            hard_mask_resize_ptr,
            hard_mask_resize_batch2_ptr};
}

std::vector<trtmc::Sam3VideoFrameResult> run_video(trtmc::Sam3Pipeline& pipeline,
                                                   std::size_t frame_count, int32_t height = 2,
                                                   int32_t width = 2, float first_value = 0.5F) {
    check(frame_count != 0, "sam3 video helper requires a prompt frame");
    std::vector<std::vector<float>> pixels(frame_count);
    std::vector<trtmc::Sam3VideoFrameView> views;
    views.reserve(frame_count);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        pixels[frame].assign(static_cast<std::size_t>(height * width * 3),
                             first_value + static_cast<float>(frame) * 0.001F);
        views.push_back({pixels[frame].data(), height, width});
    }
    auto session = pipeline.create_sam3_video_session("person");
    auto prompt = session->accept_prompt_frame(pixels.front().data(), height, width);
    return session->propagate_borrowed_continuation(std::move(prompt), views.data(), views.size());
}

void test_video_bundle_requires_both_hard_memory_plans() {
    auto hard_memory =
        std::make_unique<FakeSam3TrackerModule>(false, true, 2.0F, false, false, 1, true);
    trtmc::Sam3Pipeline pipeline(
        std::make_unique<FakeSam3TextModule>(), std::make_unique<FakeDeviceSam3VisionModule>(),
        std::make_unique<FakeSam3CoreModule>(), std::make_shared<FakeTokenizer>(), make_config(),
        "facebook/sam3", std::make_unique<FakeSam3TrackerModule>(false),
        std::make_unique<FakeSam3TrackerModule>(true),
        std::make_unique<FakeSam3TrackerModule>(false, true), nullptr, nullptr, nullptr,
        std::move(hard_memory));
    bool rejected = false;
    try {
        (void)pipeline.create_sam3_video_session("person");
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("sam3_tracker_hard_memory_batch2_engine_plan") !=
                   std::string::npos;
    }
    check(rejected, "sam3 video refuses a bundle missing the hard B2 memory plan");
}

void test_b1_device_binding_and_prompt_memory() {
    auto fixture = make_video_fixture(2, true);
    for (int32_t level = 0; level < 3; ++level) {
        const auto suffix = std::to_string(level);
        check(fixture.core->device_ptr("sam3_fpn_hidden_" + suffix) ==
                      fixture.vision->device_ptr("sam3_fpn_hidden_" + suffix) &&
                  fixture.core->device_ptr("sam3_fpn_position_" + suffix) ==
                      fixture.vision->device_ptr("sam3_fpn_position_" + suffix) &&
                  fixture.init->device_ptr("tracker_feature_" + suffix) ==
                      fixture.vision->device_ptr("sam3_tracker_feature_" + suffix) &&
                  fixture.step->device_ptr("tracker_feature_" + suffix) ==
                      fixture.vision->device_ptr("sam3_tracker_feature_" + suffix),
              "sam3 B1 workspace binds every detector/tracker feature directly");
    }
    check(fixture.step->device_ptr("tracker_position_2") ==
                  fixture.vision->device_ptr("sam3_tracker_position_2") &&
              fixture.memory->device_ptr("tracker_feature_2") ==
                  fixture.vision->device_ptr("sam3_tracker_feature_2") &&
              fixture.memory_batch2->device_ptr("tracker_feature_2") ==
                  fixture.vision->device_ptr("sam3_tracker_feature_2") &&
              fixture.hard_memory->device_ptr("tracker_feature_2") ==
                  fixture.vision->device_ptr("sam3_tracker_feature_2") &&
              fixture.hard_memory_batch2->device_ptr("tracker_feature_2") ==
                  fixture.vision->device_ptr("sam3_tracker_feature_2"),
          "sam3 B1 workspace binds recurrent position and soft/hard memory consumers");

    auto session = fixture.pipeline->create_sam3_video_session("person");
    std::vector<float> pixels(12, 0.5F);
    (void)session->accept_prompt_frame(pixels.data(), 2, 2);
    check(fixture.vision->calls == 1 && fixture.vision->sync_calls == 0 &&
              fixture.hard_memory_batch2->calls == 1,
          "sam3 prompt finishes hard memory after the vision completion fence");
}

#ifdef TRTMC_HAS_CUDA_KERNELS
void test_prompt_only_session_cleanup_restores_cuda_device() {
    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2)
        return;

    int32_t original_device = 0;
    check(cudaGetDevice(&original_device) == cudaSuccess,
          "sam3 prompt-only cleanup queries the caller CUDA device");
    check(cudaSetDevice(0) == cudaSuccess,
          "sam3 prompt-only cleanup selects its owning CUDA device");
    {
        auto fixture = make_video_fixture();
        auto session = fixture.pipeline->create_sam3_video_session("person");
        std::vector<float> pixels(12, 0.5F);
        (void)session->accept_prompt_frame(pixels.data(), 2, 2);
        check(cudaSetDevice(1) == cudaSuccess && cudaGetLastError() == cudaSuccess,
              "sam3 prompt-only cleanup selects a different caller CUDA device");
        session.reset();
        int32_t current_device = 0;
        check(cudaGetDevice(&current_device) == cudaSuccess && current_device == 1 &&
                  cudaGetLastError() == cudaSuccess,
              "sam3 prompt-only cleanup frees on its owner and restores the caller device");
        check(cudaSetDevice(0) == cudaSuccess,
              "sam3 prompt-only cleanup restores the fixture CUDA device before teardown");
    }
    check(cudaSetDevice(original_device) == cudaSuccess,
          "sam3 prompt-only cleanup restores the test CUDA device");
}
#endif

void test_interleaved_sessions_finish_frame_zero_memory_before_return() {
    auto fixture = make_video_fixture();
    auto first = fixture.pipeline->create_sam3_video_session("first");
    auto second = fixture.pipeline->create_sam3_video_session("second");
    std::vector<float> first_pixels(12, 0.25F);
    std::vector<float> second_pixels(12, 0.75F);
    auto first_prompt = first->accept_prompt_frame(first_pixels.data(), 2, 2);
    auto second_prompt = second->accept_prompt_frame(second_pixels.data(), 2, 2);

    const std::array<trtmc::Sam3VideoFrameView, 1> first_frames{
        trtmc::Sam3VideoFrameView{first_pixels.data(), 2, 2}};
    const std::array<trtmc::Sam3VideoFrameView, 1> second_frames{
        trtmc::Sam3VideoFrameView{second_pixels.data(), 2, 2}};
    const auto first_results = first->propagate_borrowed_continuation(
        std::move(first_prompt), first_frames.data(), first_frames.size());
    const auto second_results = second->propagate_borrowed_continuation(
        std::move(second_prompt), second_frames.data(), second_frames.size());

    check(first_results.size() == 1 && second_results.size() == 1 &&
              fixture.hard_memory->vision_feature_markers == std::vector<float>({150.0F, 250.0F}) &&
              fixture.memory->vision_feature_markers == std::vector<float>({150.0F, 250.0F}),
          "sam3 interleaved prompts retain their own frame-zero feature for soft refresh");
    check(fixture.hard_memory->device_ptr("tracker_feature_2") ==
              fixture.vision->device_ptr("sam3_tracker_feature_2"),
          "sam3 frame-zero hard memory preserves the canonical external binding");
}

void test_image_pcs_runs_native_preprocess_and_full_result() {
    auto* text_ptr = new FakeSam3TextModule();
    auto* vision_ptr = new FakeSam3VisionModule();
    auto* core_ptr = new FakeSam3CoreModule();
    trtmc::Sam3Pipeline pipeline(std::unique_ptr<trtmc::TrtModule>(text_ptr),
                                 std::unique_ptr<trtmc::TrtModule>(vision_ptr),
                                 std::unique_ptr<trtmc::TrtModule>(core_ptr),
                                 std::make_shared<FakeTokenizer>(), make_config());

    std::vector<float> pixels(static_cast<std::size_t>(2 * 4 * 3));
    for (std::size_t index = 0; index < pixels.size(); index += 3U) {
        pixels[index] = 0.25F;
        pixels[index + 1U] = 0.5F;
        pixels[index + 2U] = 0.75F;
    }
    const auto result = pipeline.segment_prompted_text(pixels.data(), 2, 4, "ear");
    check(text_ptr->saw_expected_ids, "sam3 image PCS encodes the text prompt");
    check(vision_ptr->saw_shape, "sam3 image PCS sends the exact B1 vision shape");
    check(vision_ptr->saw_normalized_pixels,
          "sam3 image PCS applies customer Meta image preprocessing");
    check(core_ptr->saw_text_shape && core_ptr->saw_mask,
          "sam3 image PCS passes the projected prompt to the core");
    check(core_ptr->saw_vision_inputs, "sam3 image PCS passes all FPN features to the core");
    check(result.num_masks == 1 && result.height == 2 && result.width == 4 &&
              result.masks == std::vector<float>({0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F}),
          "sam3 image PCS returns the familiar resized binary mask");
    check(result.iou_scores.size() == 1 && close(result.iou_scores[0], 0.775803F) &&
              result.boxes.size() == 4 && close(result.boxes[0], 1.0F) &&
              close(result.boxes[1], 1.0F) && close(result.boxes[2], 3.0F) &&
              close(result.boxes[3], 2.0F),
          "sam3 image PCS returns the complete customer score and box result");
}

void test_prompt_then_borrowed_tail_is_strictly_ordered() {
    std::vector<int32_t> order;
    std::vector<const float*> borrowed;
    trtmc::Sam3VideoFrameProcessor processor;
    processor.accept_prompt = [&](const trtmc::Sam3VideoFrame& frame) {
        check(order.empty() && frame.frame_idx == 0 && frame.borrowed_pixels == nullptr,
              "sam3 prompt callback runs first with owned frame zero");
        order.push_back(0);
        trtmc::Sam3VideoFrameResult result;
        result.frame_idx = 0;
        result.height = frame.height;
        result.width = frame.width;
        return result;
    };
    processor.continue_borrowed = [&](trtmc::Sam3VideoFrameResult prompt,
                                      const std::vector<trtmc::Sam3VideoFrame>& tail,
                                      int32_t total_frames) {
        check(order == std::vector<int32_t>({0}) && prompt.frame_idx == 0 && total_frames == 3 &&
                  tail.size() == 2,
              "sam3 borrowed tail starts only after prompt completion");
        std::vector<trtmc::Sam3VideoFrameResult> results;
        results.push_back(std::move(prompt));
        for (const auto& frame : tail) {
            order.push_back(frame.frame_idx);
            borrowed.push_back(frame.pixel_data());
            trtmc::Sam3VideoFrameResult result;
            result.frame_idx = frame.frame_idx;
            result.height = frame.height;
            result.width = frame.width;
            results.push_back(std::move(result));
        }
        return results;
    };

    trtmc::Sam3VideoSegmentationSession session("person", std::move(processor), 3);
    std::array<std::vector<float>, 3> pixels{
        std::vector<float>(12, 0.1F), std::vector<float>(12, 0.2F), std::vector<float>(12, 0.3F)};
    std::array<trtmc::Sam3VideoFrameView, 3> views{
        {{pixels[0].data(), 2, 2}, {pixels[1].data(), 2, 2}, {pixels[2].data(), 2, 2}}};
    auto prompt = session.accept_prompt_frame(pixels[0].data(), 2, 2);
    const auto results =
        session.propagate_borrowed_continuation(std::move(prompt), views.data(), views.size());
    check(order == std::vector<int32_t>({0, 1, 2}) && results.size() == 3 &&
              borrowed == std::vector<const float*>({pixels[1].data(), pixels[2].data()}),
          "sam3 executes the borrowed tail in strict temporal order without copying it");
}

void test_recurrent_tracker_bfloat16_state() {
    constexpr float rounded_first_feature = 1.3359375F;
    constexpr float rounded_second_feature = 2.328125F;
    constexpr float rounded_position = 0.66796875F;
    auto fixture = make_video_fixture();
    fixture.core->threshold_boundary_detection_mask = true;
    const auto results = run_video(*fixture.pipeline, 3);
    check(results.size() == 3 && results[0].object_ids == std::vector<int32_t>({0}) &&
              results[1].object_ids == std::vector<int32_t>({0}) && fixture.init->calls == 1 &&
              fixture.step->calls == 2 && fixture.hard_memory->calls == 1 &&
              fixture.memory->calls == 3,
          "sam3 recurrent B1 path initializes once and advances in temporal order");
    check(
        fixture.step->last_memory_frame_values ==
                std::vector<float>({rounded_first_feature, rounded_second_feature}) &&
            fixture.step->last_memory_position_frame_values.size() == 2 &&
            std::all_of(fixture.step->last_memory_position_frame_values.begin(),
                        fixture.step->last_memory_position_frame_values.end(),
                        [rounded_position](float value) { return close(value, rounded_position); }),
        "sam3 recurrent B1 features and positions are BF16-rounded in FP32 carriers");
    check(results[1].tracker_scores.size() == 1 && close(results[1].tracker_scores[0], 0.880797F),
          "sam3 recurrent result reports the tracker probability");
}

void test_frame_zero_prompt_and_propagation_follow_meta_schedule() {
    auto fixture = make_video_fixture(2, true);
    fixture.core->detections_first_frame_only = true;
    fixture.memory_batch2->memory_value_base = 16.0F;
    fixture.hard_memory_batch2->memory_value_base = 64.0F;
    std::vector<float> pixels(12, 0.5F);
    auto session = fixture.pipeline->create_sam3_video_session("person");
    auto prompt = session->accept_prompt_frame(pixels.data(), 2, 2);
    const auto prompt_snapshot = prompt;

    check(prompt_snapshot.object_ids == std::vector<int32_t>({0, 1}) &&
              prompt_snapshot.masks ==
                  std::vector<float>({1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}) &&
              fixture.memory->calls == 0 && fixture.memory_batch2->calls == 0 &&
              fixture.hard_memory->calls == 0 && fixture.hard_memory_batch2->calls == 1,
          "sam3 prompt returns detector masks after finishing frame-zero hard memory");

    const std::array<trtmc::Sam3VideoFrameView, 2> views{
        {{pixels.data(), 2, 2}, {pixels.data(), 2, 2}}};
    const auto propagated =
        session->propagate_borrowed_continuation(std::move(prompt), views.data(), views.size());
    const std::vector<float> first_memory{
        1024.0F,
        -1024.0F,
        -1024.0F,
        -1024.0F,
    };
    const std::vector<float> second_memory{
        -1024.0F,
        1024.0F,
        1024.0F,
        -1024.0F,
    };
    const std::vector<float> first_hard_memory{
        1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    const std::vector<float> second_hard_memory{
        0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F,
        1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F,
    };
    const std::vector<float> propagated_frame_zero_masks{
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F,
    };
    check(propagated.size() == 2 && propagated.front().object_ids == std::vector<int32_t>({0, 1}) &&
              propagated.front().masks == propagated_frame_zero_masks &&
              propagated.front().masks != prompt_snapshot.masks,
          "sam3 continuation emits Meta's consolidated tracker-mask frame-zero result");
    check(
        fixture.memory->calls == 0 && fixture.memory_batch2->calls == 2 &&
            fixture.hard_memory_batch2->calls == 1 &&
            fixture.memory_batch2->memory_batch_sizes == std::vector<std::size_t>({2, 2}) &&
            std::vector<std::vector<float>>(fixture.memory_batch2->final_masks_history.begin(),
                                            fixture.memory_batch2->final_masks_history.begin() +
                                                2) ==
                std::vector<std::vector<float>>({first_memory, second_memory}) &&
            std::vector<int32_t>(fixture.memory_batch2->memory_suppressions_history.begin(),
                                 fixture.memory_batch2->memory_suppressions_history.begin() + 2) ==
                std::vector<int32_t>({0, 0}) &&
            std::vector<float>(fixture.memory_batch2->memory_scores_history.begin(),
                               fixture.memory_batch2->memory_scores_history.begin() + 2) ==
                std::vector<float>({10.0F, 10.0F}) &&
            fixture.hard_memory_batch2->memory_batch_sizes == std::vector<std::size_t>({2}) &&
            fixture.hard_memory_batch2->final_masks_history ==
                std::vector<std::vector<float>>({first_hard_memory, second_hard_memory}) &&
            fixture.hard_memory_batch2->memory_scores_history ==
                std::vector<float>({10.0F, 10.0F}) &&
            fixture.step_batch2->last_memory_frame_values == std::vector<float>({17.375F, 18.375F}),
        "sam3 propagation replaces hard frame-zero memory with soft memory consumed by frame one");
}

void test_frame_zero_zero_detections_is_stable() {
    auto fixture = make_video_fixture(1, false, false, 2.0F, [](trtmc::Sam3Config& config) {
        config.detection_threshold = 1.0F;
    });
    std::vector<float> pixels(12, 0.5F);
    auto session = fixture.pipeline->create_sam3_video_session("person");
    auto prompt = session->accept_prompt_frame(pixels.data(), 2, 2);
    check(prompt.object_ids.empty() && prompt.masks.empty() && fixture.init->calls == 0 &&
              fixture.memory->calls == 0 && fixture.hard_memory->calls == 0,
          "sam3 empty prompt creates no tracker or memory work");
    const std::array<trtmc::Sam3VideoFrameView, 1> views{{{pixels.data(), 2, 2}}};
    const auto propagated =
        session->propagate_borrowed_continuation(std::move(prompt), views.data(), views.size());
    check(propagated.size() == 1 && propagated.front().object_ids.empty() &&
              propagated.front().masks.empty() && fixture.init->calls == 0 &&
              fixture.memory->calls == 0 && fixture.hard_memory->calls == 0,
          "sam3 empty propagated frame zero remains empty without memory work");
}

void test_late_new_track_uses_uncleaned_hard_memory() {
    auto fixture = make_video_fixture(2, false, false, 2.0F,
                                      [](trtmc::Sam3Config& config) { config.fill_hole_area = 1; });
    fixture.core->late_hard_conditioning_probe = true;
    fixture.step->recurrent_mask_override = {-2.0F, -2.0F, -2.0F, 2.0F};

    const auto results = run_video(*fixture.pipeline, 2);
    check(results.size() == 2 && results[0].object_ids == std::vector<int32_t>({0}) &&
              results[1].object_ids == std::vector<int32_t>({0, 1}),
          "sam3 hard-memory probe introduces its second object after frame zero");
    check(fixture.memory->calls == 2 && fixture.memory_batch2 == nullptr &&
              fixture.hard_memory->calls == 2 && fixture.hard_memory_batch2->calls == 0,
          "sam3 frame-zero and late singletons use hard memory while recurrent updates stay soft");
    check(fixture.hard_memory->final_masks_history.back() ==
                  std::vector<float>({1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                      0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F}) &&
              fixture.hard_memory->memory_scores_history == std::vector<float>({10.0F, 10.0F}),
          "sam3 hard memory receives Meta's owned 1008-grid mask and present-object score");
}

void test_recurrent_tracker_uses_b2_pairs_and_b1_odd_tail() {
    {
        constexpr float rounded_first_feature = 1.3359375F;
        constexpr float rounded_second_feature = 2.328125F;
        constexpr float rounded_position = 0.66796875F;
        auto fixture = make_video_fixture(2, true);
        // Keep both logical rows visible so this test isolates equal-history
        // engine grouping rather than final-result overlap compaction.
        fixture.step->recurrent_masks_by_item = {
            {2.0F, -2.0F, -2.0F, -2.0F},
            {-2.0F, 2.0F, 2.0F, -2.0F},
        };
        fixture.step_batch2->recurrent_masks_by_item = fixture.step->recurrent_masks_by_item;
        const auto results = run_video(*fixture.pipeline, 2);
        check(results[0].object_ids == std::vector<int32_t>({0, 1}) &&
                  results[1].object_ids == std::vector<int32_t>({0, 1}) &&
                  fixture.step->calls == 0 && fixture.step_batch2->calls == 1 &&
                  fixture.step_batch2->batch_sizes == std::vector<std::size_t>({2}),
              "sam3 equal-history pair uses only the fixed B2 recurrent engine");
        check(fixture.memory->calls == 0 && fixture.memory_batch2->calls == 2 &&
                  fixture.memory_batch2->memory_batch_sizes == std::vector<std::size_t>({2, 2}) &&
                  fixture.hard_memory_batch2->calls == 1,
              "sam3 paired prompt uses hard B2 then soft-refreshes before recurrent update");
        check(fixture.step_batch2->last_memory_frame_values ==
                      std::vector<float>({rounded_first_feature, rounded_second_feature}) &&
                  fixture.step_batch2->last_memory_position_frame_values ==
                      std::vector<float>({rounded_position, rounded_position}),
              "sam3 recurrent B2 features and positions are BF16-rounded in FP32 carriers");
    }
    {
        auto fixture = make_video_fixture(3, true, false, 2.0F, {}, true);
        fixture.core->detections_first_frame_only = true;
        const auto results = run_video(*fixture.pipeline, 2);
        check(results[0].object_ids == std::vector<int32_t>({0, 1, 2}) &&
                  std::is_sorted(results[1].object_ids.begin(), results[1].object_ids.end()) &&
                  fixture.step_batch2->calls == 1 && fixture.step->calls == 1,
              "sam3 odd recurrent row uses one exact B2 pair and one B1 tail");
        std::unordered_map<int32_t, bool> unique;
        for (const auto id : results[1].object_ids)
            unique.emplace(id, true);
        check(unique.size() == results[1].object_ids.size(),
              "sam3 B1 tail preserves one output per logical track");
        check(fixture.memory_batch2->calls == 2 && fixture.hard_memory_batch2->calls == 1,
              "sam3 odd memory row soft-refreshes and updates its B2 rows");
        check(fixture.memory->calls == 2,
              "sam3 odd memory row soft-refreshes and updates its B1 tail");
        check(fixture.hard_memory->calls == 1,
              "sam3 odd memory row uses one hard frame-zero B1 tail");
        check(fixture.hard_mask_resize_batch2->calls == 1 && fixture.hard_mask_resize->calls == 1 &&
                  fixture.hard_mask_resize_batch2->batch_sizes == std::vector<std::size_t>({2}) &&
                  fixture.hard_mask_resize->batch_sizes == std::vector<std::size_t>({1}),
              "sam3 hard-mask resize executes one B2 chunk followed by its B1 tail");
        const std::array<const std::vector<float>*, 3> hard_masks{
            &fixture.hard_memory_batch2->final_masks_history[0],
            &fixture.hard_memory_batch2->final_masks_history[1],
            &fixture.hard_memory->final_masks_history[0],
        };
        bool globally_owned =
            std::all_of(hard_masks.begin(), hard_masks.end(), [](const auto* mask) {
                return mask->size() == 16 &&
                       std::all_of(mask->begin(), mask->end(),
                                   [](float value) { return value == 0.0F || value == 1.0F; });
            });
        for (std::size_t pixel = 0; pixel < 16 && globally_owned; ++pixel) {
            const auto claims = static_cast<int32_t>((*hard_masks[0])[pixel]) +
                                static_cast<int32_t>((*hard_masks[1])[pixel]) +
                                static_cast<int32_t>((*hard_masks[2])[pixel]);
            globally_owned = claims <= 1;
        }
        check(globally_owned && std::count(hard_masks[2]->begin(), hard_masks[2]->end(), 1.0F) > 0,
              "sam3 B1 hard-memory tail participates in the same 1008-grid ownership as B2");
    }
}

void test_hard_mask_resize_stream_preserves_stable_global_ownership() {
    auto fixture = make_video_fixture(3, true);
    fixture.hard_mask_resize_batch2->scripted_output.assign(32, 1.0F);
    fixture.hard_mask_resize_batch2->scripted_output[16 + 1] = 2.0F;
    fixture.hard_mask_resize->scripted_output.assign(16, 1.0F);
    fixture.hard_mask_resize->scripted_output[2] = 3.0F;

    const auto results = run_video(*fixture.pipeline, 1);
    std::vector<float> expected_first(16, 1.0F);
    expected_first[1] = 0.0F;
    expected_first[2] = 0.0F;
    std::vector<float> expected_second(16, 0.0F);
    expected_second[1] = 1.0F;
    std::vector<float> expected_tail(16, 0.0F);
    expected_tail[2] = 1.0F;
    check(results.size() == 1 && results.front().object_ids == std::vector<int32_t>({0, 1, 2}) &&
              fixture.hard_memory_batch2->final_masks_history ==
                  std::vector<std::vector<float>>({expected_first, expected_second}) &&
              fixture.hard_memory->final_masks_history ==
                  std::vector<std::vector<float>>({expected_tail}),
          "sam3 streamed hard-mask ownership keeps strict first-row ties across B2 and B1");
}

void test_recurrent_area_policy_uses_high_resolution_global_geometry() {
    auto fixture = make_video_fixture(2, true);
    fixture.core->detections_first_frame_only = true;
    const std::vector<float> first_mask{-4.0F, -4.0F, -4.0F, 1.0F};
    const std::vector<float> second_mask{-4.0F, -4.0F, -2.0F, 1.0F};
    fixture.step_batch2->recurrent_masks_by_item = {first_mask, second_mask};

    const auto results = run_video(*fixture.pipeline, 2);
    check(results.size() == 2 && fixture.memory_batch2->calls == 2 && fixture.memory->calls == 0,
          "sam3 high-resolution area-policy probe refreshes then updates B2 memory");
    check(std::vector<std::vector<float>>(fixture.memory_batch2->final_masks_history.end() - 2,
                                          fixture.memory_batch2->final_masks_history.end()) ==
                  std::vector<std::vector<float>>({first_mask, second_mask}) &&
              std::vector<int32_t>(fixture.memory_batch2->memory_suppressions_history.end() - 2,
                                   fixture.memory_batch2->memory_suppressions_history.end()) ==
                  std::vector<int32_t>({0, 0}) &&
              std::vector<float>(fixture.memory_batch2->memory_scores_history.end() - 2,
                                 fixture.memory_batch2->memory_scores_history.end()) ==
                  std::vector<float>({10.0F, 10.0F}),
          "sam3 area policy retains the second object from its 8x8 ratio rather than its "
          "2x2 ratio and sends original overlapping logits to TensorRT");
    check(fixture.hard_memory_batch2->memory_suppressions_history.empty(),
          "sam3 hard conditioning remains independent from recurrent area shrinkage");
}

void test_recurrent_area_policy_is_global_across_b2_and_b1_tail() {
    auto fixture = make_video_fixture(3, true);
    fixture.core->detections_first_frame_only = true;
    const std::vector<float> winner(4, 3.0F);
    const std::vector<float> empty(4, -2.0F);
    const std::vector<float> global_loser(4, 2.0F);
    fixture.step_batch2->recurrent_masks_by_item = {
        winner,
        empty,
        global_loser,
        global_loser,
    };
    fixture.step->recurrent_mask_override = global_loser;

    const auto results = run_video(*fixture.pipeline, 2);
    check(results.size() == 2 && fixture.memory_batch2->calls == 2 && fixture.memory->calls == 2,
          "sam3 three-object area-policy probe refreshes and updates B2 plus B1 memory");
    check(std::vector<int32_t>(fixture.memory_batch2->memory_suppressions_history.end() - 2,
                               fixture.memory_batch2->memory_suppressions_history.end()) ==
                  std::vector<int32_t>({0, 1}) &&
              std::vector<float>(fixture.memory_batch2->memory_scores_history.end() - 2,
                                 fixture.memory_batch2->memory_scores_history.end()) ==
                  std::vector<float>({10.0F, -10.0F}) &&
              fixture.memory->memory_suppressions_history.back() == 1 &&
              fixture.memory->memory_scores_history.back() == -10.0F &&
              fixture.memory->final_masks_history.back() == global_loser,
          "sam3 B1 tail carries the suppression decision made against all three objects");
}

void test_recurrent_area_policy_suppresses_empty_singleton_after_resize() {
    auto fixture = make_video_fixture();
    fixture.core->detections_first_frame_only = true;
    const std::vector<float> empty(4, -2.0F);
    fixture.step->recurrent_mask_override = empty;

    const auto results = run_video(*fixture.pipeline, 2);
    check(results.size() == 2 && fixture.memory->calls == 2 &&
              fixture.memory->memory_suppressions_history.back() == 1 &&
              fixture.memory->memory_scores_history.back() == -10.0F &&
              fixture.memory->final_masks_history.back() == empty,
          "sam3 empty singleton is rejected while its original low-resolution logits remain "
          "the soft-plan input");
}

void test_device_recurrent_memory_rounds_features_and_positions() {
    constexpr float rounded_first_feature = 1.3359375F;
    constexpr float rounded_second_feature = 2.328125F;
    constexpr float rounded_position = 0.66796875F;
    {
        auto fixture = make_video_fixture(1, false, true);
        (void)run_video(*fixture.pipeline, 2);
        check(fixture.step->saw_device_memory_input && fixture.step->saw_device_position_input &&
                  close(fixture.step->last_memory_value, rounded_first_feature) &&
                  close(fixture.step->last_memory_position_value, rounded_position),
              "sam3 device B1 features and positions are BF16-rounded in FP32 carriers");
    }
    {
        auto fixture = make_video_fixture(2, true, true);
        (void)run_video(*fixture.pipeline, 2);
        check(fixture.step_batch2->saw_device_memory_input &&
                  fixture.step_batch2->saw_device_position_input &&
                  fixture.step_batch2->last_memory_frame_values ==
                      std::vector<float>({rounded_first_feature, rounded_second_feature}) &&
                  fixture.step_batch2->last_memory_position_frame_values ==
                      std::vector<float>({rounded_position, rounded_position}),
              "sam3 device B2 features and positions are BF16-rounded in FP32 carriers");
    }
}

void test_parallel_mask_cleanup_preserves_full_results() {
    const auto enable_cleanup = [](trtmc::Sam3Config& config) { config.fill_hole_area = 1; };
    const std::vector<std::vector<float>> visible_recurrent_masks{
        {2.0F, -2.0F, -2.0F, -2.0F}, {-2.0F, 2.0F, -2.0F, -2.0F}, {-2.0F, -2.0F, 2.0F, 2.0F},
        {2.0F, -2.0F, -2.0F, -2.0F}, {-2.0F, 2.0F, -2.0F, -2.0F}, {-2.0F, -2.0F, 2.0F, 2.0F},
    };
    auto first = make_video_fixture(3, false, false, 2.0F, enable_cleanup);
    first.core->detections_first_frame_only = true;
    first.core->cleanup_probe_detection_masks = true;
    first.step->recurrent_masks_by_item = visible_recurrent_masks;
    const auto reference = run_video(*first.pipeline, 3, 2, 3);
    auto second = make_video_fixture(3, false, false, 2.0F, enable_cleanup);
    second.core->detections_first_frame_only = true;
    second.core->cleanup_probe_detection_masks = true;
    second.step->recurrent_masks_by_item = visible_recurrent_masks;
    const auto repeated = run_video(*second.pipeline, 3, 2, 3);
    const auto all_rows_visible = [](const auto& results) {
        return std::all_of(results.begin(), results.end(),
                           [](const auto& frame) { return frame.object_ids.size() == 3; });
    };
    check(reference.size() == 3 && repeated.size() == reference.size() &&
              all_rows_visible(reference) && all_rows_visible(repeated),
          "sam3 parallel cleanup handles every active tracker row");
    for (std::size_t index = 0; index < reference.size(); ++index)
        check(same_video_result(reference[index], repeated[index]),
              "sam3 parallel cleanup preserves deterministic full frame results");
}

void test_parallel_tracker_init_and_cleanup_overlap() {
    auto schedule = std::make_shared<ParallelTrackerInitSchedule>();
    auto fixture = make_parallel_tracker_init_fixture(schedule);
    const auto results = run_video(*fixture.pipeline, 1);
    check(schedule->entered == 2 && fixture.init->calls == 1 && fixture.sibling->calls == 1 &&
              results.front().object_ids == std::vector<int32_t>({0, 1}),
          "sam3 pair initialization overlaps two contexts and commits canonical object order");

    auto blocked = make_parallel_tracker_init_fixture(nullptr, 1);
    auto gate = std::make_shared<BlockedTrackerInitSchedule>();
    blocked.init->on_forward = [gate] { gate->enter_and_wait(); };
    blocked.sibling->on_forward = [gate] { gate->enter_and_wait(); };
    auto future = std::async(std::launch::async, [&] { return run_video(*blocked.pipeline, 1); });
    const bool both_started = gate->wait_for_both();
    const bool result_waits_for_init =
        future.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready;
    gate->release();
    const auto overlapped = future.get();
    check(both_started && result_waits_for_init && blocked.init->calls == 1 &&
              blocked.sibling->calls == 1 &&
              overlapped.front().object_ids == std::vector<int32_t>({0, 1}) &&
              overlapped.front().masks ==
                  std::vector<float>({1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F}),
          "sam3 consolidated new-detection cleanup remains exact while both init lanes overlap");
}

void test_conditioning_and_pointer_history() {
    const auto configure = [](trtmc::Sam3Config& config) {
        config.high_confidence_threshold = 0.7F;
    };
    auto config_fixture = make_video_fixture(1, false, false, 2.0F, configure);
    const auto conditioning_results = run_video(*config_fixture.pipeline, 82);
    check(conditioning_results.size() == 82, "sam3 conditioning run returns every frame");
    check(config_fixture.step->last_memory_offsets ==
              std::vector<int32_t>({0, 0, 0, 0, 6, 5, 4, 3, 2}),
          "sam3 retains ordered conditioning and recent memory offsets");
    check(config_fixture.step->last_memory_frame_values.size() == 9,
          "sam3 retains four conditioning and five recent memories");

    auto pointers = make_video_fixture(1, false, false, 2.0F, configure);
    const auto pointer_results = run_video(*pointers.pipeline, 1024);
    check(pointer_results.size() == 1024 && pointers.step->last_pointer_offsets.size() == 18 &&
              pointers.step->last_pointer_offsets[0] == 15 &&
              pointers.step->last_pointer_offsets[3] == 63 &&
              pointers.step->last_pointer_offsets[4] == 1 &&
              pointers.step->last_pointer_offsets.back() == 14 &&
              pointers.step->last_max_pointers == 16,
          "sam3 P19 profile preserves Meta's four conditioning and fourteen normal valid "
          "pointers");
}

void test_pointer_history_matches_meta_five_frame_boundary() {
    const auto expected = std::array<std::vector<int32_t>, 4>{
        std::vector<int32_t>{1},
        std::vector<int32_t>{2},
        std::vector<int32_t>{3, 1},
        std::vector<int32_t>{4, 1, 2},
    };
    for (std::size_t recurrent_frame = 1; recurrent_frame <= expected.size(); ++recurrent_frame) {
        auto fixture = make_video_fixture();
        const auto results = run_video(*fixture.pipeline, recurrent_frame + 1);
        check(results.size() == recurrent_frame + 1 &&
                  fixture.step->last_pointer_offsets == expected[recurrent_frame - 1],
              "sam3 five-frame pointer history matches the captured Meta tracker boundary");
    }
}

void test_memory_quality_selection_uses_meta_ordinal_slots() {
    auto fixture = make_video_fixture();
    fixture.step->scripted_selected_ious = {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F};
    const auto results = run_video(*fixture.pipeline, 9);
    check(results.size() == 9 &&
              fixture.step->last_memory_offsets == std::vector<int32_t>({0, 5, 4, 3, 2, 1}) &&
              fixture.step->last_pointer_offsets == std::vector<int32_t>({8, 1, 2, 3, 4}),
          "sam3 filters low-effective-IoU memories, keeps t-1, and uses ordinal slots");
}

void test_memory_quality_is_shared_within_appearance_cohort() {
    auto fixture = make_video_fixture(2, true);
    fixture.core->detections_first_frame_only = true;
    fixture.step_batch2->recurrent_masks_by_item = {
        {2.0F, -2.0F, -2.0F, -2.0F}, {-2.0F, 2.0F, 2.0F, -2.0F},  {2.0F, -2.0F, -2.0F, -2.0F},
        {-2.0F, 2.0F, 2.0F, -2.0F},  {2.0F, -2.0F, -2.0F, -2.0F}, {-2.0F, 2.0F, 2.0F, -2.0F},
    };
    // With logit 2, the two row-local effective qualities are 0 and
    // approximately 0.03046: they straddle the 0.01 threshold, while their
    // Meta cohort mean (approximately 0.01523) retains the frame for both.
    fixture.step_batch2->scripted_selected_ious = {0.0F, 0.04F, 1.0F, 1.0F, 1.0F, 1.0F};
    const auto results = run_video(*fixture.pipeline, 4);
    check(results.size() == 4 && fixture.step->calls == 0 && fixture.step_batch2->calls == 3 &&
              fixture.step_batch2->last_memory_offsets == std::vector<int32_t>({0, 2, 1, 0, 2, 1}),
          "sam3 applies Meta's shared cohort quality before recurrent history selection");
}

void test_memory_quality_keeps_later_appearance_cohort_independent() {
    auto fixture = make_video_fixture(2);
    fixture.core->late_hard_conditioning_probe = true;
    const std::vector<float> first_mask{-2.0F, -2.0F, -2.0F, 2.0F};
    const std::vector<float> second_mask{2.0F, 2.0F, 2.0F, -2.0F};
    fixture.step->recurrent_masks_by_item = {
        first_mask, first_mask, second_mask, first_mask, second_mask, first_mask, second_mask,
    };
    // Frame two gives the older cohort a sub-threshold score and the cohort
    // introduced on frame one a score above threshold. A global mean would
    // incorrectly retain frame two for both states.
    fixture.step->scripted_selected_ious = {0.0F, 0.0F, 0.04F, 1.0F, 1.0F, 1.0F, 1.0F};
    const auto results = run_video(*fixture.pipeline, 5);
    const auto history_size = fixture.step->memory_offsets_history.size();
    check(results.size() == 5 && history_size >= 2 &&
              fixture.step->memory_offsets_history[history_size - 2] ==
                  std::vector<int32_t>({0, 1}) &&
              fixture.step->memory_offsets_history[history_size - 1] ==
                  std::vector<int32_t>({0, 2, 1}),
          "sam3 does not average frame quality across separately introduced cohorts");
}

void test_memory_quality_recomputes_after_cohort_member_removal() {
    auto fixture = make_video_fixture(2, false, false, 2.0F, [](trtmc::Sam3Config& config) {
        config.hotstart_duplicate_threshold = 1;
    });
    fixture.core->second_detection_first_frame_only = true;
    fixture.core->tie_detection_scores = true;
    fixture.step->contained_pair_masks = true;
    // The frame-one cohort mean passes because the soon-removed second row is
    // strong. Meta remove_object slices that row from stored outputs and
    // recomputes the survivor's frame quality, which is zero.
    fixture.step->scripted_selected_ious = {0.0F, 0.04F, 1.0F, 1.0F};
    const auto results = run_video(*fixture.pipeline, 4);
    check(results.size() == 4 && results[1].removed_object_ids == std::vector<int32_t>({1}) &&
              fixture.step->last_memory_offsets == std::vector<int32_t>({0, 1}),
          "sam3 recomputes shared historical quality after removing a cohort member");
}

void test_reconditioning_uses_raw_tracker_logit() {
    auto fixture = make_video_fixture(1, false, false, 0.8F, [](trtmc::Sam3Config& config) {
        config.high_confidence_threshold = 0.7F;
    });
    fixture.init->pointer_value = 0.75F;
    fixture.step->recurrent_mask_override = {-1.0F, 3.0F, 3.0F, -1.0F};
    const auto results = run_video(*fixture.pipeline, 18);
    const auto refreshed_pointer_count =
        static_cast<std::size_t>(std::count(fixture.step->last_pointer_frame_values.begin(),
                                            fixture.step->last_pointer_frame_values.end(), 0.75F));
    const auto& frame_sixteen_memory = fixture.memory->final_masks_history.at(16);
    const bool cleared_nearby_memory =
        std::none_of(fixture.step->last_memory_frame_values.begin(),
                     fixture.step->last_memory_frame_values.end(),
                     [](float value) { return value > 10.0F && value < 17.0F; });
    check(fixture.init->calls == 2 && refreshed_pointer_count == 2 &&
              frame_sixteen_memory == std::vector<float>({-1.0F, 3.0F, 3.0F, -1.0F}) &&
              cleared_nearby_memory &&
              fixture.step->last_memory_offsets == std::vector<int32_t>({0, 0, 6, 5, 4, 3, 2}) &&
              results.back().tracker_scores.size() == 1 &&
              close(results.back().tracker_scores[0], 0.689974F),
          "sam3 periodic reconditioning refreshes the pointer, keeps tracker-mask memory, "
          "clears singleton history, and preserves probability metadata");
}

void test_reconditioning_promotes_the_full_appearance_cohort() {
    auto fixture = make_video_fixture(2, true, false, 2.0F, [](trtmc::Sam3Config& config) {
        config.high_confidence_threshold = 0.7F;
        config.hotstart_unmatch_threshold = 100;
        config.hotstart_duplicate_threshold = 100;
    });
    fixture.core->second_detection_first_frame_only = true;
    fixture.init->pointer_value = 0.75F;
    const std::vector<float> first_mask{3.0F, 1.0F, -1.0F, -3.0F};
    const std::vector<float> second_mask{-3.0F, 1.0F, 3.0F, -1.0F};
    for (std::size_t frame = 1; frame < 18; ++frame) {
        fixture.step_batch2->recurrent_masks_by_item.push_back(first_mask);
        fixture.step_batch2->recurrent_masks_by_item.push_back(second_mask);
    }

    const auto results = run_video(*fixture.pipeline, 18);
    check(results.size() == 18 && fixture.init->calls == 3,
          "sam3 periodic reconditioning refreshes only the matched cohort row");
    check(fixture.step->calls == 0 && fixture.step_batch2->calls == 17,
          "sam3 reconditioned cohort retains one shared recurrent batch signature");
    check(fixture.memory_batch2->final_masks_history.at(32) == first_mask,
          "sam3 periodic reconditioning refreshes only the matched row while promoting its "
          "full appearance cohort to conditioning history");
}

trtmc::Sam3VideoFrameResult run_overlap_case(bool tie_scores, bool three_overlap = false,
                                             bool empty_first = false) {
    auto fixture = make_video_fixture(three_overlap ? 3 : 2);
    fixture.core->three_overlap_detections = three_overlap;
    fixture.core->empty_first_detection_mask = empty_first;
    fixture.core->tie_detection_scores = tie_scores;
    std::vector<float> pixels(12, 0.5F);
    auto session = fixture.pipeline->create_sam3_video_session("person");
    return session->accept_prompt_frame(pixels.data(), 2, 2);
}

void test_association_overlap_and_stable_ties() {
    const auto scored = run_overlap_case(false);
    check(scored.object_ids == std::vector<int32_t>({0, 1}) &&
              scored.masks == std::vector<float>({1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F}),
          "sam3 overlap assigns shared pixels to the higher tracker score");
    const auto tied = run_overlap_case(true);
    check(tied.masks == std::vector<float>({1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}),
          "sam3 frame-zero detector overlap uses stable first-object ties");
    const auto displaced = run_overlap_case(false, true);
    check(displaced.object_ids == std::vector<int32_t>({0, 1, 2}) &&
              displaced.masks == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                                     1.0F, 0.0F, 1.0F, 0.0F}) &&
              displaced.detection_scores.size() == 3 && displaced.tracker_scores.size() == 3 &&
              displaced.boxes == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 1.0F}),
          "sam3 frame-zero overlap retains fully covered rows with their pre-overlap box");
    const auto compacted = run_overlap_case(false, false, true);
    check(compacted.object_ids == std::vector<int32_t>({1}) &&
              compacted.masks == std::vector<float>({0.0F, 1.0F, 1.0F, 0.0F}),
          "sam3 overlap compacts owner indices after an empty leading mask");
}

void test_recent_occlusion_and_hotstart_policy() {
    auto occlusion = make_video_fixture(2);
    occlusion.core->detections_first_frame_only = true;
    occlusion.step->scripted_occlusion = true;
    const auto occlusion_results = run_video(*occlusion.pipeline, 4);
    check(occlusion_results.back().object_ids == std::vector<int32_t>({0}) &&
              occlusion_results.back().suppressed_object_ids.empty(),
          "sam3 recent occlusion hiding is not reported as a hotstart event");

    auto hotstart = make_video_fixture(2, false, false, 2.0F, [](trtmc::Sam3Config& config) {
        config.hotstart_duplicate_threshold = 1;
    });
    hotstart.core->second_detection_first_frame_only = true;
    hotstart.step->contained_pair_masks = true;
    hotstart.core->tie_detection_scores = true;
    const auto hotstart_results = run_video(*hotstart.pipeline, 3);
    check(hotstart_results[1].object_ids == std::vector<int32_t>({0}) &&
              hotstart_results[1].removed_object_ids == std::vector<int32_t>({1}) &&
              hotstart_results[2].removed_object_ids == std::vector<int32_t>({1}),
          "sam3 hotstart removes duplicates after frame policy and retains removal history");
    check(hotstart.hard_memory->calls == 0 && hotstart.hard_memory_batch2->calls == 1 &&
              hotstart.memory->calls == 5 && hotstart.memory->final_masks_history.size() == 5 &&
              hotstart.memory->final_masks_history[2] ==
                  std::vector<float>({2.0F, -2.0F, -2.0F, -2.0F}) &&
              hotstart.memory->memory_suppressions_history[2] == 1 &&
              close(hotstart.memory->memory_scores_history[2], -10.0F),
          "sam3 hotstart memory update passes the soon-removed track's original mask plus its "
          "high-resolution suppression decision before erasure");
}

void test_recurrent_pool_survives_serial_sessions() {
    auto fixture = make_video_fixture(1, false, true);
    const auto first = run_video(*fixture.pipeline, 2);
    const auto second = run_video(*fixture.pipeline, 2, 2, 2, 0.25F);
    check(first.size() == 2 && second.size() == 2 && fixture.step->async_calls == 2 &&
              fixture.step->saw_device_memory_input && fixture.hard_memory->async_calls == 2 &&
              fixture.memory->async_calls == 4,
          "sam3 pipeline recurrent pool supports serial sessions on device-resident state");
}

} // namespace

int main() {
    test_sam3_clip_tokenizer_matches_meta_segmentation();
    test_clip_non_removed_split_keeps_standalone_space_token();
#ifdef TRTMC_HAS_CUDA_KERNELS
    test_bfloat16_round_copy_supports_exact_alias();
    test_cuda_preprocess_matches_cpu_meta_pillow();
    test_prompt_only_session_cleanup_restores_cuda_device();
#endif
    test_preprocess_matches_customer_meta_pillow_resize();
    test_preprocess_uses_explicit_fma_at_uint8_half_steps();
    test_recurrent_area_policy_uses_high_resolution_global_geometry();
    test_recurrent_area_policy_is_global_across_b2_and_b1_tail();
    test_recurrent_area_policy_suppresses_empty_singleton_after_resize();
    test_video_bundle_requires_both_hard_memory_plans();
    test_b1_device_binding_and_prompt_memory();
    test_interleaved_sessions_finish_frame_zero_memory_before_return();
    test_image_pcs_runs_native_preprocess_and_full_result();
    test_prompt_then_borrowed_tail_is_strictly_ordered();
    test_recurrent_tracker_bfloat16_state();
    test_frame_zero_prompt_and_propagation_follow_meta_schedule();
    test_frame_zero_zero_detections_is_stable();
    test_late_new_track_uses_uncleaned_hard_memory();
    test_recurrent_tracker_uses_b2_pairs_and_b1_odd_tail();
    test_hard_mask_resize_stream_preserves_stable_global_ownership();
    test_device_recurrent_memory_rounds_features_and_positions();
    test_parallel_mask_cleanup_preserves_full_results();
    test_parallel_tracker_init_and_cleanup_overlap();
    test_conditioning_and_pointer_history();
    test_pointer_history_matches_meta_five_frame_boundary();
    test_memory_quality_selection_uses_meta_ordinal_slots();
    test_memory_quality_is_shared_within_appearance_cohort();
    test_memory_quality_keeps_later_appearance_cohort_independent();
    test_memory_quality_recomputes_after_cohort_member_removal();
    test_reconditioning_uses_raw_tracker_logit();
    test_reconditioning_promotes_the_full_appearance_cohort();
    test_association_overlap_and_stable_ties();
    test_recent_occlusion_and_hotstart_policy();
    test_recurrent_pool_survives_serial_sessions();
    std::cout << "PASS\n";
    return 0;
}

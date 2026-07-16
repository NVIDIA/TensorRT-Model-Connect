/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
        saw_normalized_pixels = close(pixels[0], -1.0219197F) && close(pixels[16], 0.2051821F) &&
                                close(pixels[32], 1.5245316F);

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
            pred_masks_ = {2.0F, 2.0F, 2.0F, -2.0F, -2.0F, -2.0F, -2.0F, 2.0F};
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
        if (three_overlap_detections)
            pred_logits_ = {0.0F, 1.0F, 2.0F};
        if (detections_first_frame_only && calls > 1)
            std::fill(pred_logits_.begin(), pred_logits_.end(), -20.0F);
        if (second_detection_first_frame_only && calls > 1)
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
    int32_t calls{0};

  private:
    std::vector<float> pred_masks_;
    std::vector<float> pred_boxes_;
    std::vector<float> pred_logits_;
    std::vector<float> presence_logits_;
    std::unordered_map<std::string, void*> bound_inputs_;
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3TrackerModule final : public trtmc::TrtModule {
  public:
    explicit FakeSam3TrackerModule(bool recurrent, bool memory_only = false,
                                   float object_score_logit = 2.0F, bool device_recurrent = false,
                                   bool own_stream = false, std::size_t device_batch_size = 1)
        : recurrent_(recurrent), memory_only_(memory_only), object_score_logit_(object_score_logit),
          device_recurrent_(device_recurrent), device_batch_size_(device_batch_size) {
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
            }
            if (memory_only_) {
                allocate_device_tensor("new_memory_features", 4 * 64);
                allocate_device_tensor("new_memory_position", 4 * 64);
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
        if (memory_only_) {
            saw_features = inputs.count("tracker_feature_2") == 1 &&
                           inputs.count("final_mask") == 1 &&
                           inputs.count("object_score_logits") == 1;
            const auto& final_mask = inputs.at("final_mask");
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
            for (std::size_t batch = 0; batch < batch_size; ++batch) {
                last_final_mask.assign(final_mask_values + batch * item_values,
                                       final_mask_values + (batch + 1) * item_values);
                last_memory_score = score_values[batch];
                final_masks_history.push_back(last_final_mask);
                memory_scores_history.push_back(last_memory_score);
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
            const auto pointers = inputs.find("object_pointers");
            const auto pointer_offset = inputs.find("object_pointer_temporal_offsets");
            const auto max_pointers = inputs.find("max_object_pointers_to_use");
            if (memory == inputs.end() || memory_offset == inputs.end() ||
                pointers == inputs.end() || pointer_offset == inputs.end() ||
                max_pointers == inputs.end()) {
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
            const float* memory_features = static_cast<const float*>(memory->second.data);
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
            last_memory_value = *memory_features;
            last_memory_frame_values.clear();
            const std::size_t frame_count = memory_offset->second.numel();
            const std::size_t values_per_frame = memory->second.numel() / frame_count;
            for (std::size_t frame = 0; frame < frame_count; ++frame)
                last_memory_frame_values.push_back(memory_features[frame * values_per_frame]);
            const auto* pointer_values = static_cast<const int32_t*>(pointer_offset->second.data);
            last_pointer_offsets.assign(pointer_values,
                                        pointer_values + pointer_offset->second.numel());
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
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const std::size_t logical_call = recurrent_items_ + batch + 1;
            std::vector<float> item_mask{-2.0F, 2.0F, 2.0F, -2.0F};
            if (recurrent_ && contained_pair_masks) {
                item_mask = logical_call % 2 == 1 ? std::vector<float>{2.0F, -2.0F, -2.0F, -2.0F}
                                                  : std::vector<float>{3.0F, 3.0F, 3.0F, 3.0F};
            }
            if (recurrent_ && scripted_occlusion && (logical_call == 1 || logical_call == 4)) {
                item_mask = {-2.0F, -2.0F, -2.0F, -2.0F};
            }
            pred_masks_.insert(pred_masks_.end(), item_mask.begin(), item_mask.end());
        }
        if (recurrent_)
            recurrent_items_ += batch_size;
        object_pointer_.assign(batch_size * 256, 0.25F);
        object_score_.assign(batch_size, object_score_logit_);
        memory_.clear();
        if (memory_only_) {
            for (std::size_t batch = 0; batch < batch_size; ++batch) {
                memory_.insert(memory_.end(), 4 * 64,
                               static_cast<float>(calls) + static_cast<float>(batch) + 0.3333F);
            }
        } else {
            memory_.assign(4 * 64, 0.125F);
        }
        memory_position_.assign(batch_size * 4 * 64, 0.5F);

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
        return name == "pred_masks" || name == "object_pointer" || name == "object_score_logits" ||
                       name == "new_memory_features" || name == "new_memory_position"
                   ? device_tensors_.count(name) != 0
                   : false;
    }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "memory_features" || name == "memory_position")
            return {static_cast<int64_t>(device_batch_size_), 1, 4, 64};
        if (name == "new_memory_features" || name == "new_memory_position")
            return {4, 1, 64};
        if (name == "pred_masks")
            return {static_cast<int64_t>(device_batch_size_), 1, 2, 2};
        if (name == "object_pointer")
            return {static_cast<int64_t>(device_batch_size_), 1, 256};
        if (name == "object_score_logits")
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
    std::vector<int32_t> last_pointer_offsets;
    int32_t last_max_pointers{0};
    float last_memory_value{0.0F};
    std::vector<float> last_memory_frame_values;
    std::vector<float> last_final_mask;
    std::vector<std::vector<float>> final_masks_history;
    std::vector<const float*> final_mask_input_addresses;
    std::vector<float> last_detector_mask;
    std::vector<std::vector<float>> detector_masks_history;
    std::vector<float> memory_scores_history;
    std::vector<std::size_t> memory_batch_sizes;
    float last_memory_score{0.0F};
    bool scripted_occlusion{false};
    bool contained_pair_masks{false};
    bool saw_device_memory_input{false};
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
    std::size_t recurrent_items_{0};
    std::vector<float> pred_masks_;
    std::vector<float> object_pointer_;
    std::vector<float> object_score_;
    std::vector<float> memory_;
    std::vector<float> memory_position_;
    std::unordered_map<std::string, void*> bound_inputs_;
    std::unordered_map<std::string, void*> device_tensors_;
    cudaStream_t owned_stream_{nullptr};
    std::shared_ptr<void> keep_alive_;
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
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto config = make_config();
    config.fill_hole_area = fill_hole_area;
    config.detection_threshold = 0.4F;
    config.new_detection_threshold = 0.4F;
    config.detection_nms_threshold = 0.0F;
    auto pipeline = std::make_unique<trtmc::Sam3Pipeline>(
        std::move(text), std::move(vision), std::move(core), tokenizer, config, "facebook/sam3",
        std::move(init), std::move(step), std::move(memory), nullptr, nullptr, std::move(sibling));
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

void test_preprocess_matches_native_uint8_antialiased_resize() {
    // Oracles were produced through the pinned release environment's actual
    // Sam3ImageProcessorFast (facebook/sam3 revision 3c879f3), torch
    // 2.12.0+cu130 and torchvision 0.27.0 on aarch64. They cover resize,
    // rescale, and normalization in CHW order, not a hand-written surrogate.
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
        check(output_hash == 0xc62246a4f256d20dULL,
              "sam3 96x128 preprocessing is bit-exact with native torchvision oracle");
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
        check(output_hash == 0x8f733fa08ddf2663ULL,
              "sam3 high-frequency downsample is bit-exact with native antialias oracle");
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
        check(float_bit_hash(output) == 0xd2c5efe13ee2985aULL,
              "sam3 odd non-integer downsample is bit-exact with native antialias oracle");
    }
}

void test_preprocess_uses_explicit_fma_at_uint8_half_steps() {
    trtmc::Sam3Config config;
    config.image_size = 16;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
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
            const auto expected = static_cast<float>(
                std::clamp(static_cast<int>(
                               std::floor(std::fma(std::clamp(source, 0.0F, 1.0F), 255.0F, 0.5F))),
                           0, 255));
            exact = output[channel * plane + pixel] == expected;
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
};

VideoFixture make_video_fixture(std::size_t detections = 1, bool batch2 = false,
                                bool device_recurrent = false, float tracker_logit = 2.0F,
                                const std::function<void(trtmc::Sam3Config&)>& configure = {}) {
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
        std::move(step_batch2), std::move(memory_batch2));
    return {std::move(pipeline), vision_ptr, core_ptr,         init_ptr, step_ptr,
            step_batch2_ptr,     memory_ptr, memory_batch2_ptr};
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

void test_b1_device_binding_and_snapshot() {
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
                  fixture.vision->device_ptr("sam3_tracker_feature_2"),
          "sam3 B1 workspace binds recurrent position and B1/B2 memory consumers");

    auto session = fixture.pipeline->create_sam3_video_session("person");
    std::vector<float> pixels(12, 0.5F);
    (void)session->accept_prompt_frame(pixels.data(), 2, 2);
    check(fixture.vision->calls == 1 && fixture.vision->sync_calls == 0,
          "sam3 vision snapshot uses its completion fence without redundant module sync");
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
    check(vision_ptr->saw_normalized_pixels, "sam3 image PCS applies native image preprocessing");
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
    auto fixture = make_video_fixture();
    fixture.core->threshold_boundary_detection_mask = true;
    const auto results = run_video(*fixture.pipeline, 3);
    check(results.size() == 3 && results[0].object_ids == std::vector<int32_t>({0}) &&
              results[1].object_ids == std::vector<int32_t>({0}) && fixture.init->calls == 1 &&
              fixture.step->calls == 2 && fixture.memory->calls == 2,
          "sam3 recurrent B1 path initializes once and advances in temporal order");
    check(fixture.step->last_memory_frame_values.size() == 2 &&
              !close(fixture.step->last_memory_frame_values[1], 1.3333F),
          "sam3 recurrent memory is BF16-rounded before its next use");
    check(results[1].tracker_scores.size() == 1 && close(results[1].tracker_scores[0], 0.880797F),
          "sam3 recurrent result reports the tracker probability");
}

void test_recurrent_tracker_uses_b2_and_odd_tail() {
    {
        auto fixture = make_video_fixture(2, true);
        const auto results = run_video(*fixture.pipeline, 2);
        check(results[0].object_ids == std::vector<int32_t>({0, 1}) &&
                  results[1].object_ids == std::vector<int32_t>({0, 1}) &&
                  fixture.step->calls == 0 && fixture.step_batch2->calls == 1 &&
                  fixture.step_batch2->batch_sizes == std::vector<std::size_t>({2}),
              "sam3 equal-history pair uses only the fixed B2 recurrent engine");
        check(fixture.memory->calls == 0 && fixture.memory_batch2->calls == 1 &&
                  fixture.memory_batch2->memory_batch_sizes == std::vector<std::size_t>({2}),
              "sam3 paired state update uses the fixed B2 memory engine");
    }
    {
        auto fixture = make_video_fixture(3, true);
        const auto results = run_video(*fixture.pipeline, 2);
        check(results[0].object_ids == std::vector<int32_t>({0, 1, 2}) &&
                  std::is_sorted(results[1].object_ids.begin(), results[1].object_ids.end()) &&
                  fixture.step_batch2->calls == 2 && fixture.step->calls == 0,
              "sam3 odd recurrent row uses one exact pair and one padded B2 tail");
        std::unordered_map<int32_t, bool> unique;
        for (const auto id : results[1].object_ids)
            unique.emplace(id, true);
        check(unique.size() == results[1].object_ids.size(),
              "sam3 padded B2 tail creates no duplicate logical track");
        check(fixture.memory_batch2->calls == 1 && fixture.memory->calls == 1,
              "sam3 odd memory row commits one B2 pair and one singleton tail");
    }
}

void test_parallel_mask_cleanup_preserves_full_results() {
    const auto enable_cleanup = [](trtmc::Sam3Config& config) { config.fill_hole_area = 1; };
    auto first = make_video_fixture(3, false, false, 2.0F, enable_cleanup);
    first.core->detections_first_frame_only = true;
    first.core->cleanup_probe_detection_masks = true;
    const auto reference = run_video(*first.pipeline, 3, 2, 3);
    auto second = make_video_fixture(3, false, false, 2.0F, enable_cleanup);
    second.core->detections_first_frame_only = true;
    second.core->cleanup_probe_detection_masks = true;
    const auto repeated = run_video(*second.pipeline, 3, 2, 3);
    check(reference.size() == 3 && reference.front().object_ids.size() == 3 &&
              repeated.size() == reference.size(),
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
              overlapped.front().masks ==
                  std::vector<float>({1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F}),
          "sam3 new-detection cleanup remains exact while both init lanes overlap");
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
    check(pointer_results.size() == 1024 && pointers.step->last_pointer_offsets.size() == 78 &&
              pointers.step->last_pointer_offsets.front() == 1023 &&
              pointers.step->last_pointer_offsets[63] == 15 &&
              pointers.step->last_pointer_offsets[64] == 1 &&
              pointers.step->last_pointer_offsets.back() == 14 &&
              pointers.step->last_max_pointers == 16,
          "sam3 P79 policy retains all conditioning pointers then the recent window");
}

void test_reconditioning_uses_raw_tracker_logit() {
    auto fixture = make_video_fixture(1, false, false, 0.8F, [](trtmc::Sam3Config& config) {
        config.high_confidence_threshold = 0.7F;
    });
    const auto results = run_video(*fixture.pipeline, 18);
    check(fixture.step->last_memory_offsets == std::vector<int32_t>({0, 0, 6, 5, 4, 3, 2}) &&
              results.back().tracker_scores.size() == 1 &&
              close(results.back().tracker_scores[0], 0.689974F),
          "sam3 raw tracker logit drives reconditioning while metadata remains probability");
}

trtmc::Sam3VideoFrameResult run_overlap_case(bool tie_scores, bool three_overlap = false,
                                             bool empty_first = false) {
    auto fixture = make_video_fixture(three_overlap ? 3 : 2);
    fixture.core->three_overlap_detections = three_overlap;
    fixture.core->empty_first_detection_mask = empty_first;
    fixture.core->tie_detection_scores = tie_scores;
    return run_video(*fixture.pipeline, 1).front();
}

void test_association_overlap_and_stable_ties() {
    const auto scored = run_overlap_case(false);
    check(scored.object_ids == std::vector<int32_t>({0, 1}) &&
              scored.masks == std::vector<float>({1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F}),
          "sam3 overlap assigns shared pixels to the higher tracker score");
    const auto tied = run_overlap_case(true);
    check(tied.masks == std::vector<float>({1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}),
          "sam3 equal tracker scores use stable object-ID order");
    const auto displaced = run_overlap_case(false, true);
    check(displaced.object_ids == std::vector<int32_t>({0, 1, 2}) &&
              displaced.masks == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                                     1.0F, 0.0F, 1.0F, 0.0F}),
          "sam3 later scores displace ownership without dropping object metadata");
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
    check(hotstart.memory->calls == 3 && hotstart.memory->final_masks_history.size() == 3 &&
              hotstart.memory->final_masks_history[0] ==
                  std::vector<float>({-10.0F, -10.0F, -10.0F, -10.0F}) &&
              close(hotstart.memory->memory_scores_history[0], -10.0F),
          "sam3 hotstart memory update sees the soon-removed track before erasure");
}

void test_recurrent_pool_survives_serial_sessions() {
    auto fixture = make_video_fixture(1, false, true);
    const auto first = run_video(*fixture.pipeline, 2);
    const auto second = run_video(*fixture.pipeline, 2, 2, 2, 0.25F);
    check(first.size() == 2 && second.size() == 2 && fixture.step->async_calls == 2 &&
              fixture.step->saw_device_memory_input && fixture.memory->async_calls == 2,
          "sam3 pipeline recurrent pool supports serial sessions on device-resident state");
}

} // namespace

int main() {
#ifdef TRTMC_HAS_CUDA_KERNELS
    test_bfloat16_round_copy_supports_exact_alias();
#endif
    test_preprocess_matches_native_uint8_antialiased_resize();
    test_preprocess_uses_explicit_fma_at_uint8_half_steps();
    test_b1_device_binding_and_snapshot();
    test_image_pcs_runs_native_preprocess_and_full_result();
    test_prompt_then_borrowed_tail_is_strictly_ordered();
    test_recurrent_tracker_bfloat16_state();
    test_recurrent_tracker_uses_b2_and_odd_tail();
    test_parallel_mask_cleanup_preserves_full_results();
    test_parallel_tracker_init_and_cleanup_overlap();
    test_conditioning_and_pointer_history();
    test_reconditioning_uses_raw_tracker_logit();
    test_association_overlap_and_stable_ties();
    test_recent_occlusion_and_hotstart_policy();
    test_recurrent_pool_survives_serial_sessions();
    std::cout << "PASS\n";
    return 0;
}

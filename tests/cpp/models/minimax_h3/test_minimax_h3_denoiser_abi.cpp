/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct TensorSpec {
    trtmc::DType dtype{trtmc::DType::kFloat32};
    std::vector<int64_t> shape;
    bool input{true};
    bool dynamic{false};
    std::vector<int64_t> minimum;
    std::vector<int64_t> optimum;
    std::vector<int64_t> maximum;
};

class FakeModule final : public trtmc::ITrtModule {
  public:
    std::unordered_map<std::string, TensorSpec> tensors;

    void add_dynamic(const std::string& name, trtmc::DType dtype, std::vector<int64_t> minimum,
                     std::vector<int64_t> optimum, std::vector<int64_t> maximum) {
        tensors.emplace(name, TensorSpec{dtype, maximum, true, true, std::move(minimum),
                                         std::move(optimum), maximum});
    }
    void add_static(const std::string& name, trtmc::DType dtype, std::vector<int64_t> shape) {
        tensors.emplace(name, TensorSpec{dtype, std::move(shape), true});
    }
    void add_output(const std::string& name, trtmc::DType dtype, std::vector<int64_t> maximum) {
        tensors.emplace(name, TensorSpec{dtype, std::move(maximum), false});
    }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(/*input=*/true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(/*input=*/false); }
    bool has_input(const std::string& name) const override {
        const auto it = tensors.find(name);
        return it != tensors.end() && it->second.input;
    }
    bool has_output(const std::string& name) const override {
        const auto it = tensors.find(name);
        return it != tensors.end() && !it->second.input;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return tensors.at(name).shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto& spec = tensors.at(name);
        if (selector == trtmc::ProfileShapeSelector::kMin)
            return spec.minimum;
        if (selector == trtmc::ProfileShapeSelector::kOpt)
            return spec.optimum;
        return spec.maximum;
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensors.at(name).shape.size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return tensors.at(name).dynamic;
    }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<trtmc::TensorInfo> info(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, spec] : tensors) {
            if (spec.input == input)
                result.push_back({name, spec.shape, spec.dtype, input});
        }
        return result;
    }
};

constexpr int32_t kMinRows = 19285;
constexpr int32_t kOptRows = 37838;
constexpr int32_t kMaxRows = 112367;

void add_packed(FakeModule& module, const std::string& name, trtmc::DType dtype,
                int32_t width = 0) {
    if (width == 0)
        module.add_dynamic(name, dtype, {kMinRows}, {kOptRows}, {kMaxRows});
    else
        module.add_dynamic(name, dtype, {kMinRows, width}, {kOptRows, width}, {kMaxRows, width});
}

void add_modulation(FakeModule& module, const std::string& name) {
    module.add_static(name, trtmc::DType::kBFloat16, {12, 6, 5376});
}

void add_segment_outputs(FakeModule& module) {
    module.add_output("next_residual_hidden", trtmc::DType::kBFloat16, {kMaxRows, 5376});
    for (const char* name : {"vsa_query", "vsa_key", "vsa_value", "vsa_gate"})
        module.add_output(name, trtmc::DType::kBFloat16, {56, kMaxRows, 128});
}

FakeModule make_entry() {
    FakeModule module;
    module.add_dynamic("video_hidden_states", trtmc::DType::kFloat32, {18870, 96}, {37296, 96},
                       {108576, 96});
    module.add_dynamic("audio_hidden_states", trtmc::DType::kFloat32, {414, 32}, {414, 32},
                       {1150, 32});
    module.add_dynamic("encoder_hidden_states", trtmc::DType::kFloat32, {1, 5120}, {128, 5120},
                       {2641, 5120});
    add_packed(module, "position_ids", trtmc::DType::kFloat32, 3);
    add_packed(module, "adaln_indices", trtmc::DType::kInt32);
    add_modulation(module, "current_block_modulation");
    add_segment_outputs(module);
    return module;
}

FakeModule make_transition() {
    FakeModule module;
    add_packed(module, "residual_hidden", trtmc::DType::kBFloat16, 5376);
    module.add_dynamic("vsa_attention_output", trtmc::DType::kBFloat16, {56, kMinRows, 128},
                       {56, kOptRows, 128}, {56, kMaxRows, 128});
    add_packed(module, "position_ids", trtmc::DType::kFloat32, 3);
    add_packed(module, "adaln_indices", trtmc::DType::kInt32);
    add_modulation(module, "current_block_modulation");
    add_modulation(module, "next_block_modulation");
    add_segment_outputs(module);
    return module;
}

FakeModule make_finish() {
    FakeModule module;
    add_packed(module, "residual_hidden", trtmc::DType::kBFloat16, 5376);
    module.add_dynamic("vsa_attention_output", trtmc::DType::kBFloat16, {56, kMinRows, 128},
                       {56, kOptRows, 128}, {56, kMaxRows, 128});
    add_packed(module, "adaln_indices", trtmc::DType::kInt32);
    add_packed(module, "timestep_indices", trtmc::DType::kInt32);
    module.add_dynamic("video_hidden_states", trtmc::DType::kFloat32, {18870, 96}, {37296, 96},
                       {108576, 96});
    module.add_dynamic("audio_hidden_states", trtmc::DType::kFloat32, {414, 32}, {414, 32},
                       {1150, 32});
    add_modulation(module, "current_block_modulation");
    module.add_static("final_modulation", trtmc::DType::kBFloat16, {4, 2, 5376});
    module.add_output("video_velocity", trtmc::DType::kFloat32, {108576, 96});
    module.add_output("audio_velocity", trtmc::DType::kFloat32, {1150, 32});
    return module;
}

template <typename Function>
bool rejects(const Function& function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void require(bool condition, const char* label) {
    if (!condition)
        throw std::runtime_error(label);
}

void test_segmented_contract() {
    auto entry = make_entry();
    auto transition = make_transition();
    auto finish = make_finish();
    trtmc::validate_minimax_h3_segment_plan(entry, trtmc::MiniMaxH3SegmentPlanKind::kEntry);
    trtmc::validate_minimax_h3_segment_plan(transition,
                                            trtmc::MiniMaxH3SegmentPlanKind::kTransition);
    trtmc::validate_minimax_h3_segment_plan(finish, trtmc::MiniMaxH3SegmentPlanKind::kFinish);

    entry.tensors.at("position_ids").maximum = {38247, 3};
    require(rejects([&] {
                trtmc::validate_minimax_h3_segment_plan(entry,
                                                        trtmc::MiniMaxH3SegmentPlanKind::kEntry);
            }),
            "segmented entry accepted a legacy packed-row maximum");

    auto legacy_video_entry = make_entry();
    legacy_video_entry.tensors.at("video_hidden_states").maximum = {106488, 96};
    require(rejects([&] {
                trtmc::validate_minimax_h3_segment_plan(legacy_video_entry,
                                                        trtmc::MiniMaxH3SegmentPlanKind::kEntry);
            }),
            "segmented entry accepted a target-only maximum profile");
}

void test_static_legacy_plan_fails_closed() {
    FakeModule module;
    module.add_static("video_hidden_states", trtmc::DType::kFloat32, {37296, 96});
    module.add_static("audio_hidden_states", trtmc::DType::kFloat32, {414, 32});
    module.add_static("encoder_hidden_states", trtmc::DType::kFloat32, {537, 5120});
    module.add_static("position_ids", trtmc::DType::kFloat32, {38247, 3});
    module.add_static("adaln_indices", trtmc::DType::kInt32, {38247});
    module.add_static("timestep_indices", trtmc::DType::kInt32, {38247});
    for (int32_t layer = 0; layer < 50; ++layer)
        module.add_static("block_modulation_" + std::to_string(layer), trtmc::DType::kBFloat16,
                          {12, 6, 5376});
    module.add_static("final_modulation", trtmc::DType::kBFloat16, {4, 2, 5376});
    module.add_output("video_velocity", trtmc::DType::kFloat32, {37296, 96});
    module.add_output("audio_velocity", trtmc::DType::kFloat32, {414, 32});
    require(rejects([&] { trtmc::validate_minimax_h3_monolithic_denoiser_plan(module, false); }),
            "legacy static denoiser plan did not fail closed");
}

void test_dynamic_dense_legacy_profile_remains_compatible() {
    FakeModule module;
    module.add_dynamic("video_hidden_states", trtmc::DType::kFloat32, {18870, 96}, {37296, 96},
                       {106488, 96});
    module.add_dynamic("audio_hidden_states", trtmc::DType::kFloat32, {414, 32}, {414, 32},
                       {1150, 32});
    module.add_dynamic("encoder_hidden_states", trtmc::DType::kFloat32, {1, 5120}, {128, 5120},
                       {537, 5120});
    module.add_dynamic("position_ids", trtmc::DType::kFloat32, {19285, 3}, {37838, 3}, {108175, 3});
    for (const char* name : {"adaln_indices", "timestep_indices"})
        module.add_dynamic(name, trtmc::DType::kInt32, {19285}, {37838}, {108175});
    for (int32_t layer = 0; layer < 50; ++layer)
        module.add_static("block_modulation_" + std::to_string(layer), trtmc::DType::kBFloat16,
                          {12, 6, 5376});
    module.add_static("final_modulation", trtmc::DType::kBFloat16, {4, 2, 5376});
    module.add_output("video_velocity", trtmc::DType::kFloat32, {106488, 96});
    module.add_output("audio_velocity", trtmc::DType::kFloat32, {1150, 32});
    trtmc::validate_minimax_h3_monolithic_denoiser_plan(module, false);
}

} // namespace

int main() {
    try {
        test_segmented_contract();
        test_static_legacy_plan_fails_closed();
        test_dynamic_dense_legacy_profile_remains_compatible();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

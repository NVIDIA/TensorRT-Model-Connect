/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_conditioner.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trtmc;
using namespace trtmc::minimax_h3;
constexpr int64_t kWidth = 5120;
constexpr std::array<const char*, 4> kFeatures = {"visual_features", "deepstack_features_0",
                                                  "deepstack_features_1", "deepstack_features_2"};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

class FixtureTokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        return {text.begin(), text.end()};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view text) const override {
        if (text == "<|vision_start|>")
            return 151652;
        if (text == "<|vision_end|>")
            return 151653;
        if (text == "<|image_pad|>")
            return 151655;
        return -1;
    }
    std::string token_for_id(int32_t) const override { return {}; }
};

struct State {
    int vision_loads{0}, language_loads{0}, live{0}, vision_resets{0};
    bool fail_language{false};
    std::vector<std::array<uint16_t, 4>> expected_features;
};

// CPU-only native-contract double, not a model oracle. Its purpose is to test
// real TensorMap assembly, BF16 bit preservation, ownership and residency.
class Module final : public ITrtModule {
  public:
    Module(State& state, bool vision) : state_(state), vision_(vision) {
        ++state_.live;
        if (vision_)
            ++state_.vision_loads;
        else
            ++state_.language_loads;
    }
    ~Module() override { --state_.live; }
    TensorMap forward(const TensorMap& inputs) override {
        TensorMap result;
        if (vision_) {
            require(inputs.size() == 5, "vision native input count");
            const auto& pixels = inputs.at("pixel_values");
            require(pixels.dtype == DType::kFloat32 && pixels.shape[1] == 1536, "vision pixel ABI");
            const auto rows = pixels.shape[0] / 4;
            std::vector<std::array<uint16_t, 4>> markers(static_cast<std::size_t>(rows));
            for (std::size_t branch = 0; branch < 4; ++branch) {
                features_[branch].resize(static_cast<std::size_t>(rows * kWidth));
                for (int64_t row = 0; row < rows; ++row) {
                    const auto marker = static_cast<uint16_t>(
                        0x3f00 + state_.expected_features.size() * 16 + row * 16 + branch);
                    markers[static_cast<std::size_t>(row)][branch] = marker;
                    std::fill_n(features_[branch].data() + row * kWidth, kWidth, marker);
                }
                result.emplace(kFeatures[branch],
                               Tensor{features_[branch].data(), {rows, kWidth}, DType::kBFloat16});
            }
            state_.expected_features.insert(state_.expected_features.end(), markers.begin(),
                                            markers.end());
        } else {
            if (state_.fail_language)
                throw std::runtime_error("fixture language failure");
            require(inputs.size() == 7, "language native input count");
            const auto& ids = inputs.at("input_ids");
            const auto rows = ids.shape[0];
            const auto& mask = inputs.at("visual_mask");
            require(mask.dtype == DType::kInt32 && mask.shape == std::vector<int64_t>({rows, 1}),
                    "native mask is int32, not unsupported bool");
            require(inputs.at("position_ids").shape == std::vector<int64_t>({3, rows}),
                    "language MRoPE shape");
            const auto* tokens = static_cast<const int32_t*>(ids.data);
            const auto* visual = static_cast<const int32_t*>(mask.data);
            std::size_t packed_row = 0;
            for (int64_t row = 0; row < rows; ++row) {
                require((tokens[row] == 151655) == (visual[row] == 1),
                        "mask excludes vision boundaries and labels");
                for (std::size_t branch = 0; branch < 4; ++branch) {
                    const auto& feature = inputs.at(kFeatures[branch]);
                    require(feature.dtype == DType::kBFloat16 &&
                                feature.shape == std::vector<int64_t>({rows, kWidth}),
                            "dense BF16 feature ABI");
                    const auto* bits = static_cast<const uint16_t*>(feature.data);
                    const auto expected =
                        visual[row] ? state_.expected_features.at(packed_row)[branch] : uint16_t{0};
                    for (int64_t column = 0; column < kWidth; ++column)
                        require(bits[row * kWidth + column] == expected,
                                "feature payload/order is preserved without numeric conversion");
                }
                if (visual[row])
                    ++packed_row;
            }
            require(packed_row == state_.expected_features.size(),
                    "all images fill their own visual rows");
            embeddings_.resize(static_cast<std::size_t>(rows * kWidth));
            for (int64_t row = 0; row < rows; ++row)
                std::fill_n(embeddings_.data() + row * kWidth, kWidth, static_cast<float>(7 + row));
            result.emplace("encoder_hidden_states",
                           Tensor{embeddings_.data(), {rows, kWidth}, DType::kFloat32});
        }
        return result;
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused device fixture path");
    }
    void forward_device_async(const DeviceTensorMap&) override {
        throw std::logic_error("unused device fixture path");
    }
    void forward_async(const TensorMap&) override {
        throw std::logic_error("unused async fixture path");
    }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        if (vision_)
            return name == "pixel_values" || name == "position_indices" ||
                   name == "position_weights" || name == "rotary_cos" || name == "rotary_sin";
        return name == "input_ids" || name == "position_ids" || name == "visual_mask" ||
               name == "visual_features" || name == "deepstack_features_0" ||
               name == "deepstack_features_1" || name == "deepstack_features_2";
    }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string& name) const override {
        if (name == "input_ids" || name == "position_ids" || name == "visual_mask" ||
            name == "position_indices")
            return DType::kInt32;
        if (name == "visual_features" || name.find("deepstack_features_") == 0)
            return DType::kBFloat16;
        return DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return vision_ ? std::vector<int64_t>{65536, 1536} : std::vector<int64_t>{128};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 2; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {
        if (vision_) {
            ++state_.vision_resets;
            state_.expected_features.clear();
        }
    }
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    State& state_;
    bool vision_;
    std::array<std::vector<uint16_t>, 4> features_;
    std::vector<float> embeddings_;
};

void test_residency(ConditionerResidency residency) {
    State state;
    FixtureTokenizer tokenizer;
    ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t) {
        require(name == "vision.plan" || name == "text_encoder.plan",
                "family owns exact component plan names");
        return std::make_unique<Module>(state, name == "vision.plan");
    };
    std::vector<uint8_t> first(32 * 64 * 3, 63), second(32 * 32 * 3, 197);
    const std::vector<ResizedQwenImage> images = {{first.data(), first.size(), 32, 64},
                                                  {second.data(), second.size(), 32, 32}};
    {
        ReferenceImageConditioner conditioner(loader, tokenizer, nullptr, 128, residency);
        for (int request = 0; request < 2; ++request) {
            const auto result = conditioner.encode(images, "walk");
            require(result.presentation.visual_rows.size() == 3, "two-image feature count");
            require(result.embeddings.size() == result.presentation.input_ids.size() * kWidth,
                    "owned conditioning shape");
            require(result.embeddings.front() == 7.0F &&
                        result.embeddings.back() ==
                            static_cast<float>(6 + result.presentation.input_ids.size()),
                    "output remains valid after staged module destruction");
            require(state.live == (residency == ConditionerResidency::Staged ? 0 : 2),
                    "selected residency is honored");
        }
        const int loads = residency == ConditionerResidency::Staged ? 2 : 1;
        require(state.vision_loads == loads && state.language_loads == loads &&
                    state.vision_resets == 2,
                "resident contexts reuse; staged contexts reload");
        state.fail_language = true;
        bool failed = false;
        try {
            (void)conditioner.encode(images, "walk");
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(failed && state.live == 0, "failure releases both component states");
        state.fail_language = false;
        require(!conditioner.encode(images, "walk").embeddings.empty(),
                "next request recovers without stale bindings");
    }
    require(state.live == 0, "conditioner owns all resident state");
}

} // namespace

int main() {
    try {
        test_residency(ConditionerResidency::Staged);
        test_residency(ConditionerResidency::Resident);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

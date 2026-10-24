/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_visual_encoder.h"
#include "families/minimax_h3/runtime/torch_cuda_normal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace trtmc;
using namespace trtmc::minimax_h3;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

struct State {
    int live{0}, peak{0}, encoder_loads{0}, posterior_loads{0}, noise_calls{0};
    int encoder_resets{0}, posterior_resets{0}, tile_index{0};
    int fail_tile{-1};
    bool staged{true}, bad_posterior_shape{false};
};

State* noise_state = nullptr;

// CPU-only ITrtModule double. It checks orchestration and ownership, NOT
// trained VAE math, Gaussian generation, or TensorRT posterior correctness.
class Module final : public ITrtModule {
  public:
    Module(State& state, bool encoder) : state_(state), encoder_(encoder) {
        ++state_.live;
        state_.peak = std::max(state_.peak, state_.live);
        if (encoder_)
            ++state_.encoder_loads;
        else
            ++state_.posterior_loads;
    }
    ~Module() override {
        std::fill(output_.begin(), output_.end(), -12345.0F);
        --state_.live;
    }
    TensorMap forward(const TensorMap& inputs) override {
        if (encoder_) {
            if (state_.tile_index == state_.fail_tile)
                throw std::runtime_error("fixture tile failure");
            require(inputs.size() == 1, "encoder input count");
            const auto& pixels = inputs.at("pixels");
            require(pixels.shape == std::vector<int64_t>({1, 3, 1, 256, 256}) &&
                        pixels.dtype == DType::kFloat32,
                    "single-frame FP32 VisualVAE tile, not a repeated video");
            const auto* data = static_cast<const float*>(pixels.data);
            require(std::abs(data[0] - (127.0F / 255.0F - 0.485F) / 0.229F) < 1e-6F,
                    "encoder receives ImageNet pixels");
            output_.resize(48 * 16 * 16);
            for (int channel = 0; channel < 48; ++channel)
                std::fill_n(output_.data() + channel * 16 * 16, 16 * 16,
                            state_.tile_index / 16.0F + channel / 64.0F);
            ++state_.tile_index;
            return {{"moments", Tensor{output_.data(), {1, 48, 1, 16, 16}, DType::kFloat32}}};
        }
        require(inputs.size() == 2 && state_.tile_index == 121,
                "posterior runs once after every 2048-square tile");
        const auto& moments = inputs.at("moments");
        const auto& noise = inputs.at("noise");
        require(moments.shape == std::vector<int64_t>({48, 128 * 128}) &&
                    noise.shape == std::vector<int64_t>({24, 128 * 128}),
                "stitch moments before sampling; keep channel-major layout");
        const auto* values = static_cast<const float*>(moments.data);
        require(values[0] == 0.0F && values[48 * 128 * 128 - 1] == 120 / 16.0F + 47 / 64.0F,
                "first and last raw tiles survive stitching");
        const auto* random = static_cast<const float*>(noise.data);
        require(random[0] == -8.0F / 64 &&
                    random[24 * 128 * 128 - 1] ==
                        (static_cast<int>((24 * 128 * 128 - 1) % 17) - 8) / 64.0F,
                "posterior receives owned fresh CPU noise");
        output_.resize(24 * 128 * 128);
        for (int channel = 0; channel < 24; ++channel)
            for (int pixel = 0; pixel < 128 * 128; ++pixel)
                output_[channel * 128 * 128 + pixel] = (channel + 1) * 2.0F + (pixel % 5) / 8.0F;
        return {{"sampled_latents", Tensor{output_.data(),
                                           {state_.bad_posterior_shape ? 23 : 24, 128 * 128},
                                           DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused device fixture");
    }
    void forward_device_async(const DeviceTensorMap&) override {
        throw std::logic_error("unused async device fixture");
    }
    void forward_async(const TensorMap&) override {
        throw std::logic_error("unused async fixture");
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
        return encoder_ ? name == "pixels" : name == "moments" || name == "noise";
    }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             ProfileShapeSelector selector) const override {
        if (encoder_)
            return {1, 3, selector == ProfileShapeSelector::kMax ? 17 : 1, 256, 256};
        return {name == "moments" ? 48 : 24,
                selector == ProfileShapeSelector::kMax ? 65536 : 16384};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return encoder_ ? 5 : 2; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {
        if (encoder_) {
            state_.tile_index = 0;
            ++state_.encoder_resets;
        } else {
            ++state_.posterior_resets;
        }
    }
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    State& state_;
    bool encoder_;
    std::vector<float> output_;
};

ReferenceLatentNormalization normalization() {
    ReferenceLatentNormalization value;
    for (std::size_t channel = 0; channel < 24; ++channel) {
        value.mean[channel] = static_cast<float>(channel) / 4;
        value.std[channel] = 1.0F + static_cast<float>(channel) / 16;
    }
    return value;
}

void exercise(VisualReferenceResidency residency) {
    State state;
    state.staged = residency == VisualReferenceResidency::Staged;
    noise_state = &state;
    VisualReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t) {
        require(name == "vae_encoder.plan" || name == "vae_reference_posterior.plan",
                "family-owned visual-reference component names");
        return std::make_unique<Module>(state, name == "vae_encoder.plan");
    };
    const PreparedReferenceImage image{{2048, 2048}, std::vector<uint8_t>(2048 * 2048 * 3, 127)};
    {
        ReferenceVisualEncoder encoder(loader, normalization(), nullptr, residency);
        for (int request = 0; request < 2; ++request) {
            const auto output = encoder.encode_image(image);
            require(output.latent_height == 128 && output.latent_width == 128 &&
                        output.latents.size() == 24 * 128 * 128,
                    "owned single-frame full reference latent shape");
            for (int channel = 0; channel < 24; ++channel)
                for (int pixel : {0, 1, 16383}) {
                    const float expected =
                        ((channel + 1) * 2.0F + (pixel % 5) / 8.0F - channel / 4.0F) /
                        (1.0F + channel / 16.0F);
                    require(output.latents[channel * 128 * 128 + pixel] == expected,
                            "normalize owned rounded samples after module destruction");
                }
            require(state.live == (state.staged ? 0 : 2), "requested component residency");
        }
        require(state.noise_calls == 2, "fresh independent posterior draw for each image");
        require(state.encoder_loads == (state.staged ? 2 : 1) &&
                    state.posterior_loads == (state.staged ? 2 : 1),
                "resident contexts are reused rather than recreated");
        require(state.peak == (state.staged ? 1 : 2),
                "staged encoder released before posterior load");
        state.fail_tile = 5;
        bool failed = false;
        try {
            (void)encoder.encode_image(image);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(failed && state.live == 0 && state.noise_calls == 2,
                "tile failure releases state without sampling incomplete moments");
        state.fail_tile = -1;
        require(!encoder.encode_image(image).latents.empty(), "next request recovers");
        state.bad_posterior_shape = true;
        failed = false;
        try {
            (void)encoder.encode_image(image);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(failed && state.live == 0, "bad posterior output is rejected before normalization");
        PreparedReferenceImage preview{{32, 32}, std::vector<uint8_t>(32 * 32 * 3)};
        failed = false;
        try {
            (void)encoder.encode_image(preview);
        } catch (const std::invalid_argument&) {
            failed = true;
        }
        require(failed && state.live == 0, "no lower-quality reference preview accepted");
    }
    require(state.live == 0, "encoder owns both component lifetimes");
    noise_state = nullptr;
}

} // namespace

namespace trtmc::minimax_h3 {

// Deliberate link-time CPU RNG double for this standalone interface test.
// The production DSO links torch_cuda_normal.cu, not this implementation.
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset) {
    require(noise_state && seed == 42 && offset == 0 && count == 24 * 128 * 128,
            "original image-reference CPU seed, fresh stream and full latent count");
    require(!noise_state->staged || noise_state->live == 0,
            "staged GPU encoder is released before posterior allocation");
    ++noise_state->noise_calls;
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index)
        values[index] = (static_cast<int>(index % 17) - 8) / 64.0F;
    return values;
}

} // namespace trtmc::minimax_h3

int main() {
    try {
        auto bad = normalization();
        bad.std[3] = 0;
        bool rejected = false;
        try {
            ReferenceVisualEncoder encoder(
                [](const std::string&, cudaStream_t) { return std::unique_ptr<ITrtModule>{}; }, bad,
                nullptr, VisualReferenceResidency::Staged);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "zero latent normalization std rejected before loading");
        bad = normalization();
        bad.mean[2] = std::numeric_limits<float>::quiet_NaN();
        rejected = false;
        try {
            ReferenceVisualEncoder encoder(
                [](const std::string&, cudaStream_t) { return std::unique_ptr<ITrtModule>{}; }, bad,
                nullptr, VisualReferenceResidency::Staged);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "nonfinite latent normalization rejected before loading");
        exercise(VisualReferenceResidency::Staged);
        exercise(VisualReferenceResidency::Resident);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

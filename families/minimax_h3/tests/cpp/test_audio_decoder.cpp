/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/audio_decoder.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::minimax_h3;
constexpr int64_t kFrames = 3, kSamples = kFrames * 800;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Fn>
void must_throw(Fn&& function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected explicit rejection");
}

struct State {
    int loads{0}, live{0}, failure{0};
    cudaStream_t stream{nullptr};
    std::vector<float> expected;
};

// CPU lifecycle double, never an audio-quality or waveform-model oracle.
class Module final : public ITrtModule {
  public:
    explicit Module(State& state) : state_(state) { ++state.live; }
    ~Module() override { --state_.live; }
    TensorMap forward(const TensorMap& inputs) override {
        require(inputs.size() == 1, "one audio operand");
        const auto& tensor = inputs.at("normalized_audio_latents");
        require(tensor.dtype == DType::kFloat32 &&
                    tensor.shape == std::vector<int64_t>({2, 32, kFrames}),
                "FP32 stereo-as-mono-batch ABI");
        const auto* values = static_cast<const float*>(tensor.data);
        for (int64_t s = 0; s < 2; ++s)
            for (int64_t c = 0; c < 32; ++c)
                for (int64_t t = 0; t < kFrames; ++t)
                    require(values[(s * 32 + c) * kFrames + t] ==
                                state_.expected[(s * kFrames + t) * 32 + c],
                            "row-to-channel permutation changes no bits");
        if (state_.failure == 1)
            throw std::runtime_error("fixture backend failure");
        output_.resize(2 * kSamples);
        for (int64_t s = 0; s < 2; ++s)
            for (int64_t t = 0; t < kSamples; ++t)
                output_[s * kSamples + t] =
                    static_cast<float>(t) / 10000.0F + static_cast<float>(s) * 0.25F;
        if (state_.failure == 3)
            output_[0] = std::numeric_limits<float>::infinity();
        if (state_.failure == 4)
            output_[0] = 1.01F;
        return {{"waveform",
                 {output_.data(),
                  {2, 1, kSamples},
                  state_.failure == 5 ? DType::kFloat16 : DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return state_.stream; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {{}}; }
    std::vector<TensorInfo> output_info() const override { return {{}}; }
    bool has_input(const std::string& name) const override {
        return name == "normalized_audio_latents";
    }
    bool has_output(const std::string& name) const override { return name == "waveform"; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "waveform")
            return {2, 1, kSamples};
        return {2, 32, state_.failure == 2 ? kFrames + 1 : kFrames};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 3; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    State& state_;
    std::vector<float> output_;
};

void test_lifecycle(AudioDecoderResidency residency) {
    State state;
    for (int64_t i = 0; i < 2 * kFrames * 32; ++i)
        state.expected.push_back(static_cast<float>(i) / 1000.0F);
    int marker = 0;
    state.stream = reinterpret_cast<cudaStream_t>(&marker);
    ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t stream) {
        require(name == "audio_vae.plan" && stream == state.stream, "exact plan and caller stream");
        ++state.loads;
        return std::make_unique<Module>(state);
    };
    must_throw([&] { AudioDecoder invalid(loader, 0); });
    must_throw([&] { AudioDecoder invalid({}, kFrames); });
    {
        AudioDecoder owner(loader, kFrames, state.stream, residency);
        must_throw([&] { owner.run({}); });
        auto bad = state.expected;
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        must_throw([&] { owner.run(bad); });
        require(state.loads == 0, "bad inputs rejected without loading");
        const auto cold = owner.run(state.expected);
        const auto warm = owner.run(state.expected);
        require(cold.sample_rate == 32000 && cold.channels == 2 &&
                    cold.interleaved_samples.size() == 2 * kSamples &&
                    cold.interleaved_samples == warm.interleaved_samples,
                "owned cold/warm stereo waveform and metadata");
        for (int64_t t = 0; t < kSamples; ++t)
            for (int64_t s = 0; s < 2; ++s)
                require(cold.interleaved_samples[t * 2 + s] ==
                            static_cast<float>(t) / 10000.0F + static_cast<float>(s) * 0.25F,
                        "correct channel interleaving");
        const bool resident = residency == AudioDecoderResidency::Resident;
        require(state.loads == (resident ? 1 : 2) && state.live == (resident ? 1 : 0),
                "staged releases and resident reuses real owner state");
        state.failure = 1;
        must_throw([&] { owner.run(state.expected); });
        require(state.live == 0, "backend exception discards failed context");
        state.failure = 0;
        require(owner.run(state.expected).interleaved_samples == cold.interleaved_samples,
                "recovery creates a fresh module");
    }
    require(state.live == 0, "owner destruction releases resident module");
    for (int failure = 2; failure <= 5; ++failure) {
        state.failure = failure;
        AudioDecoder owner(loader, kFrames, state.stream, residency);
        must_throw([&] { owner.run(state.expected); });
        require(state.live == 0, "ABI/waveform failures release module");
    }
}
} // namespace

int main() {
    try {
        test_lifecycle(AudioDecoderResidency::Staged);
        test_lifecycle(AudioDecoderResidency::Resident);
        std::cout << "MiniMax-H3 native audio lifecycle checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

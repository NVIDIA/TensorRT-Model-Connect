/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_layout.h"
#include "families/minimax_h3/runtime/video_decoder.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::minimax_h3;
constexpr std::size_t kOneTile = 3 * 28 * 256 * 256;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Fn>
void must_throw(Fn&& fn) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected decoder rejection");
}
struct State {
    VideoDecoderConfig config;
    VideoDecodeGeometry geometry;
    std::vector<float> cthw;
    int loads{0}, live{0}, failure{0}, forwards{0};
    cudaStream_t stream{nullptr};
};

// CPU-only module double. It verifies orchestration and exact input packing;
// its deterministic decoded data is NOT a learned VAE or quality oracle.
class Module final : public ITrtModule {
  public:
    explicit Module(State& state) : state_(state) { ++state.live; }
    ~Module() override { --state_.live; }
    TensorMap forward(const TensorMap& inputs) override {
        const auto& g = state_.geometry;
        const auto& c = state_.config;
        const std::size_t groups = (g.tiles() + c.tile_batch - 1) / c.tile_batch;
        const std::size_t index = calls_ % (groups * g.clips);
        const auto clip = index / groups, first = (index % groups) * c.tile_batch;
        const auto& tensor = inputs.at("latent_tiles");
        require(inputs.size() == 1 && tensor.dtype == DType::kFloat32 &&
                    tensor.shape == std::vector<int64_t>({c.tile_batch, 24, 7, 16, 16}),
                "exact decoder batch ABI");
        const auto* values = static_cast<const float*>(tensor.data);
        std::size_t target = 0;
        for (int32_t slot = 0; slot < c.tile_batch; ++slot) {
            const auto tile = std::min(first + slot, g.tiles() - 1);
            const auto y0 = g.vertical.starts[tile / g.horizontal.starts.size()] / 16;
            const auto x0 = g.horizontal.starts[tile % g.horizontal.starts.size()] / 16;
            for (int32_t channel = 0; channel < 24; ++channel)
                for (int32_t time = 0; time < 7; ++time)
                    for (int32_t y = 0; y < 16; ++y)
                        for (int32_t x = 0; x < 16; ++x) {
                            const auto t = std::min(clip * 5 + time,
                                                    static_cast<std::size_t>(c.latent_frames - 1));
                            const auto source =
                                ((channel * c.latent_frames + t) * c.latent_height + y0 + y) *
                                    c.latent_width +
                                x0 + x;
                            volatile float scaled =
                                state_.cthw[source] * c.normalization.std[channel];
                            const float expected = scaled + c.normalization.mean[channel];
                            require(values[target++] == expected,
                                    "patch inverse, channel stats, tail repeat and unused batch "
                                    "padding");
                        }
        }
        if (state_.failure == 1)
            throw std::runtime_error("fixture failure");
        pixels_.resize(static_cast<std::size_t>(c.tile_batch) * kOneTile);
        const auto meaningful = std::min(static_cast<std::size_t>(c.tile_batch), g.tiles() - first);
        for (std::size_t slot = 0; slot < meaningful; ++slot)
            for (std::size_t i = 0; i < kOneTile; ++i)
                pixels_[slot * kOneTile + i] =
                    static_cast<float>(
                        static_cast<int32_t>((i * 13 + (first + slot) * 37 + clip * 101) % 251) -
                        125) /
                    32.0F;
        if (state_.failure == 3)
            pixels_[0] = std::numeric_limits<float>::quiet_NaN();
        ++calls_;
        ++state_.forwards;
        return {
            {"decoded_tiles", {pixels_.data(), {c.tile_batch, 3, 28, 256, 256}, DType::kFloat32}}};
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
    bool has_input(const std::string& name) const override { return name == "latent_tiles"; }
    bool has_output(const std::string& name) const override { return name == "decoded_tiles"; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "decoded_tiles")
            return {state_.config.tile_batch, 3, 28, 256, 256};
        return {state_.config.tile_batch, 24, state_.failure == 2 ? 8 : 7, 16, 16};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 5; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    State& state_;
    std::size_t calls_{0};
    std::vector<float> pixels_;
};

std::vector<float> test(int32_t batch, VideoDecoderResidency residency) {
    VideoDecoderConfig config;
    config.latent_frames = 6;
    config.latent_height = 16;
    config.latent_width = 20;
    config.tile_batch = batch;
    for (int32_t c = 0; c < 24; ++c) {
        config.normalization.mean[c] = static_cast<float>(c) / 37;
        config.normalization.std[c] = 1.1F + static_cast<float>(c) / 53;
    }
    State state{config, make_video_decode_geometry(6, 16, 20), {}};
    state.cthw.resize(24 * 6 * 16 * 20);
    for (std::size_t i = 0; i < state.cthw.size(); ++i)
        state.cthw[i] = static_cast<float>(static_cast<int32_t>(i % 257) - 128) / 128.0F;
    const auto rows = pack_reference_video_rows(state.cthw, 6, 16, 20);
    ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t stream) {
        require(name == "vae.plan" && stream == state.stream, "exact plan and stream");
        ++state.loads;
        return std::make_unique<Module>(state);
    };
    std::vector<float> retained;
    {
        VideoDecoder owner(loader, config, state.stream, residency);
        must_throw([&] { owner.run({}); });
        require(state.loads == 0, "invalid row count rejected before load");
        const auto cold = owner.run(rows);
        const auto warm = owner.run(rows);
        require(cold.rgb == warm.rgb && cold.frames == 18 && cold.height == 256 &&
                    cold.width == 320,
                "owned THWC cold/warm output");
        const bool resident = residency == VideoDecoderResidency::Resident;
        require(state.loads == (resident ? 1 : 2) && state.live == (resident ? 1 : 0),
                "staged release / resident reuse");
        require(state.forwards == (batch == 1 ? 4 : 2),
                "all real tiles and partial static batch handled");
        require(cold.forwards == static_cast<uint64_t>(batch == 1 ? 2 : 1) &&
                    warm.forwards == cold.forwards,
                "result reports actual successful engine forward calls");
        retained = cold.rgb;
        state.failure = 1;
        must_throw([&] { owner.run(rows); });
        require(state.live == 0, "mid-decode failure drops module");
        state.failure = 0;
        require(owner.run(rows).rgb == retained, "decode recovers with a fresh module");
    }
    require(state.live == 0, "decoder destruction releases resident module");
    for (int failure = 2; failure <= 3; ++failure) {
        state.failure = failure;
        VideoDecoder owner(loader, config, state.stream, residency);
        must_throw([&] { owner.run(rows); });
        require(state.live == 0, "ABI/nonfinite output failure drops module");
    }
    return retained;
}
void test_cumulative() {
    VideoDecoderConfig config;
    config.latent_frames = 67;
    config.latent_height = config.latent_width = 16;
    config.tile_batch = 1;
    config.normalization.std.fill(1.0F);
    State state{config, make_video_decode_geometry(67, 16, 16), {}};
    state.cthw.resize(24 * 67 * 16 * 16);
    for (std::size_t i = 0; i < state.cthw.size(); ++i)
        state.cthw[i] = static_cast<float>(i % 251) / 251.0F;
    auto rows = pack_reference_video_rows(state.cthw, 67, 16, 16);
    ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t) {
        require(name == "vae.plan", "same original static video tile plan");
        ++state.loads;
        return std::make_unique<Module>(state);
    };
    VideoDecoder owner(loader, config, nullptr, VideoDecoderResidency::Staged);
    const auto decoded = owner.run(rows);
    require(decoded.frames == 226 && decoded.rgb.size() == std::size_t{226} * 256 * 256 * 3 &&
                decoded.forwards == 13 && state.forwards == 13 && state.live == 0,
            "all 67 cumulative latent frames reach 13 real tile invocations and 226 output frames");
}
} // namespace

int main() {
    try {
        const auto baseline = test(1, VideoDecoderResidency::Staged);
        require(test(1, VideoDecoderResidency::Resident) == baseline,
                "resident and staged assembly match");
        require(test(28, VideoDecoderResidency::Staged) == baseline,
                "single-tile and original batch assembly match");
        require(test(28, VideoDecoderResidency::Resident) == baseline,
                "both memory policies preserve assembly");
        test_cumulative();
        std::cout << "MiniMax-H3 native video lifecycle checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

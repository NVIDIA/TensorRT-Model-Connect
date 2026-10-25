/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Explicit engine/RNG doubles: this test qualifies orchestration, not a model.
#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/reference_denoiser.h"
#include "families/minimax_h3/runtime/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace trtmc;
using namespace trtmc::minimax_h3;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F>
void rejects(F&& call) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

struct State {
    ReferenceDenoiserConfig config{{2, 4, 6, 3}, 1, 8, 4, 100};
    int adaln_loads{0}, dit_loads{0}, adaln_live{0}, dit_live{0}, step{0};
    bool fail{false};
    bool continuation{false};
    std::vector<std::array<uint64_t, 3>> draws;
    std::vector<float> prefix;
    std::vector<float> audio_prefix;
    std::vector<int32_t> tags = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
};
State* rng_state = nullptr;

ImageReferenceLayout expected_layout(const State& state) {
    if (state.continuation)
        return make_continuation_reference_layout(state.tags, {{4, 6}, {2, 4}},
            state.config.target, 100, {1, 1}, 17.5, 21);
    return make_image_reference_layout(state.tags, {{4, 6}, {2, 4}}, state.config.target, 100);
}

class Module final : public ITrtModule {
  public:
    Module(State& state, bool adaln) : state_(state), adaln_(adaln) {
        if (adaln_) {
            require(state_.dit_live == 0, "release the old DiT before changing the AdaLN cache");
            ++state_.adaln_live;
            ++state_.adaln_loads;
        } else {
            require(state_.adaln_live == 0, "AdaLN must be released before DiT load");
            ++state_.dit_live;
            ++state_.dit_loads;
        }
    }
    ~Module() override {
        if (adaln_)
            --state_.adaln_live;
        else
            --state_.dit_live;
    }
    TensorMap forward(const TensorMap& input) override {
        if (adaln_) {
            require(input.size() == 1 && input.at("timesteps").shape == std::vector<int64_t>({4}),
                    "native time values, not CPU trig features");
            const auto vs = make_minimax_h3_schedule(50, 12.0F);
            const auto as = make_minimax_h3_schedule(50, 3.0F);
            const auto plan = make_image_reference_timesteps(expected_layout(state_),
                vs.timesteps[step_], as.timesteps[step_]);
            std::array<float, 4> times{};
            std::copy(plan.timesteps.begin(), plan.timesteps.end(), times.begin());
            require(std::equal(times.begin(), times.end(),
                               static_cast<const float*>(input.at("timesteps").data)),
                    "clean audio receives its own unmodified AdaLN time value");
            block_.assign(12 * 6 * 8, static_cast<uint16_t>(0x3f00 + step_));
            final_.assign(4 * 2 * 8, static_cast<uint16_t>(0x3e00 + step_));
            ++step_;
            return {{"block_modulation_0", Tensor{block_.data(), {12, 6, 8}, DType::kBFloat16}},
                    {"final_modulation", Tensor{final_.data(), {4, 2, 8}, DType::kBFloat16}}};
        }
        if (state_.fail && step_ == 10)
            throw std::runtime_error("fixture forward failure");
        require(input.size() == 11, "all reference DiT operands supplied exactly once");
        const auto& video = input.at("video_hidden_states");
        require(video.shape == std::vector<int64_t>({state_.continuation ? 26 : 20, 96}),
                "own-reference geometry plus target");
        require(input.at("audio_hidden_states").shape ==
                    std::vector<int64_t>({state_.continuation ? 8 : 6, 32}),
                "only continuation adds clean audio guide rows");
        require(input.at("target_video_indices").shape == std::vector<int64_t>({12}),
                "only generated rows are projected");
        const auto* values = static_cast<const float*>(video.data);
        const auto* audio = static_cast<const float*>(input.at("audio_hidden_states").data);
        if (step_ == 0) {
            state_.prefix.assign(values, values + (state_.continuation ? 14 : 8) * 96);
            state_.audio_prefix.assign(audio, audio + (state_.continuation ? 2 : 0) * 32);
        }
        require(std::equal(state_.prefix.begin(), state_.prefix.end(), values),
                "reference prefix never updated by scheduler");
        require(std::equal(state_.audio_prefix.begin(), state_.audio_prefix.end(), audio),
                "clean stereo guide never updated by scheduler");
        require(*static_cast<const uint16_t*>(input.at("block_modulation_0").data) ==
                    0x3f00 + step_,
                "borrowed AdaLN buffers copied before next forward");
        require(*static_cast<const uint16_t*>(input.at("final_modulation").data) == 0x3e00 + step_,
                "final AdaLN buffer lifetime");
        const auto layout = expected_layout(state_);
        const std::vector<float> positions(layout.positions.begin(), layout.positions.end());
        require(std::equal(positions.begin(), positions.end(),
                           static_cast<const float*>(input.at("position_ids").data)),
                "guide and target share the explicit global media clock");
        const auto vs = make_minimax_h3_schedule(50, 12.0F);
        const auto as = make_minimax_h3_schedule(50, 3.0F);
        const auto expected =
            make_image_reference_timesteps(layout, vs.timesteps[step_], as.timesteps[step_]);
        require(std::equal(expected.timestep_indices.begin(), expected.timestep_indices.end(),
                           static_cast<const int32_t*>(input.at("timestep_indices").data)),
                "row times reach the recurrent DiT");
        require(std::equal(expected.adaln_indices.begin(), expected.adaln_indices.end(),
                           static_cast<const int32_t*>(input.at("adaln_indices").data)),
                "time and modality stay distinct");
        video_.assign(12 * 96, 0.25F + step_);
        audio_.assign(6 * 32, -0.5F - step_);
        ++step_;
        ++state_.step;
        return {{"video_velocity", Tensor{video_.data(), {12, 96}, DType::kFloat32}},
                {"audio_velocity", Tensor{audio_.data(), {6, 32}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return adaln_ ? name == "timesteps" : name != "timesteps";
    }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string& name) const override {
        if (name.find("modulation") != std::string::npos)
            return DType::kBFloat16;
        if (name.find("indices") != std::string::npos)
            return DType::kInt32;
        return DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "timesteps")
            return {4};
        if (name == "final_modulation")
            return {4, 2, 8};
        if (name == "block_modulation_0")
            return {12, 6, 8};
        if (name == "target_video_indices")
            return {12};
        if (name == "target_audio_indices")
            return {6};
        if (name == "video_hidden_states")
            return {-1, 96};
        if (name == "audio_hidden_states")
            return {-1, 32};
        if (name == "encoder_hidden_states")
            return {-1, 4};
        if (name == "position_ids")
            return {-1, 3};
        return {-1};
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             ProfileShapeSelector selector) const override {
        auto shape = tensor_shape(name);
        for (auto& value : shape)
            if (value < 0)
                value = selector == ProfileShapeSelector::kMin ? 1 : 100;
        return shape;
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override { throw std::logic_error("unused"); }
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {
        throw std::logic_error("unused");
    }
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensor_shape(name).size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        const auto shape = tensor_shape(name);
        return std::find(shape.begin(), shape.end(), -1) != shape.end();
    }
    void reset_execution_context() override { step_ = 0; }
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    State& state_;
    bool adaln_;
    int step_{0};
    std::vector<uint16_t> block_, final_;
    std::vector<float> video_, audio_;
};

void run_case(ReferenceDenoiserResidency residency) {
    State state;
    rng_state = &state;
    ReferenceModuleLoader load = [&](const std::string& name, cudaStream_t) {
        require(name == "adaln.plan" || name == "denoiser.plan", "declared Ref2VA plans only");
        return std::make_unique<Module>(state, name == "adaln.plan");
    };
    ReferenceImageConditioning text;
    text.presentation.h3_token_tags = state.tags;
    text.presentation.input_ids.assign(state.tags.size(), 42);
    text.presentation.visual_rows = {2, 3, 4, 5, 6, 7, 10, 11};
    text.embeddings.assign(state.tags.size() * 4, 0.25F);
    const std::vector<EncodedVisualReference> images = {
        {4, 6, std::vector<float>(24 * 4 * 6, 0.5F)},
        {2, 4, std::vector<float>(24 * 2 * 4, -0.25F)}};
    {
        ImageReferenceDenoiser owner(load, state.config, nullptr, residency);
        const auto first = owner.run(text, images, 77);
        require(first.forwards == 49 && first.video_rows.size() == 12 * 96 &&
                    first.audio_rows.size() == 6 * 32,
                "generated-only result and full requested forward count");
        require(state.draws ==
                    std::vector<std::array<uint64_t, 3>>(
                        {{576, 77, 0}, {192, 77, 576}, {1152, 77, 768}, {192, 77, 1920}}),
                "request RNG order and per-draw offsets");
        auto prior_prefix = state.prefix;
        volatile float first_clean = 0.999F * 0.5F;
        volatile float first_random = (1.0F - 0.999F) * (77.0F / 4096.0F);
        volatile float second_clean = 0.999F * -0.25F;
        volatile float second_random = (1.0F - 0.999F) * ((77.0F + 576.0F) / 4096.0F);
        require(prior_prefix[0] == first_clean + first_random &&
                    prior_prefix[6 * 96] == second_clean + second_random,
                "original FP32 noise augmentation and independent reference geometry");
        const auto second = owner.run(text, images, 77);
        require(first.video_rows == second.video_rows && first.audio_rows == second.audio_rows,
                "same seed cold/warm deterministic contract");
        require(state.prefix == prior_prefix,
                "condition is freshly reproduced, not rewritten by the previous request");
        const bool resident = residency == ReferenceDenoiserResidency::Resident;
        require(state.adaln_loads == (resident ? 1 : 2) && state.dit_loads == (resident ? 1 : 2),
                "staged/resident component lifetimes");
        auto bad = text;
        bad.presentation.visual_rows.resize(7);
        const auto draws = state.draws.size();
        rejects([&] { owner.run(bad, images, 78); });
        require(state.draws.size() == draws, "invalid geometry rejected before RNG or GPU work");
        bad = text;
        bad.presentation.visual_rows[1] = bad.presentation.visual_rows[0];
        rejects([&] { owner.run(bad, images, 78); });
        require(state.draws.size() == draws, "duplicate Qwen rows rejected before model work");
        const auto different = owner.run(text, images, 78);
        require(different.video_rows != first.video_rows && state.prefix != prior_prefix,
                "seed affects references as well as generated noise");
        require(state.draws[state.draws.size() - 4][2] == 0,
                "each request begins a fresh request generator");
        state.fail = true;
        rejects([&] { owner.run(text, images, 79); });
        require(state.adaln_live == 0 && state.dit_live == 0,
                "failure releases contexts and modulation cache");
        state.fail = false;
        require(owner.run(text, images, 79).forwards == 49,
                "mid-loop failure does not poison retry");
    }
    require(state.adaln_live == 0 && state.dit_live == 0, "destructor releases resident module");
    require(images[0].latents == std::vector<float>(24 * 4 * 6, 0.5F),
            "clean input latents stay immutable");
    rng_state = nullptr;
}

void run_continuation_case(ReferenceDenoiserResidency residency) {
    State state;
    rng_state = &state;
    ReferenceModuleLoader load = [&](const std::string& name, cudaStream_t) {
        require(name == "adaln.plan" || name == "denoiser.plan", "declared family plans only");
        return std::make_unique<Module>(state, name == "adaln.plan");
    };
    ReferenceImageConditioning text;
    text.presentation.h3_token_tags = state.tags;
    text.presentation.input_ids.assign(state.tags.size(), 42);
    text.presentation.visual_rows = {2, 3, 4, 5, 6, 7, 10, 11};
    text.embeddings.assign(state.tags.size() * 4, 0.25F);
    const std::vector<EncodedVisualReference> images = {
        {4, 6, std::vector<float>(24 * 4 * 6, 0.5F)},
        {2, 4, std::vector<float>(24 * 2 * 4, -0.25F)}};
    ReferenceLatentGuide guide{{1, 4, 6, 1}, std::vector<float>(6 * 96, 2.0F),
                               std::vector<float>(2 * 32, 3.0F)};
    std::fill(guide.audio_rows.begin() + 32, guide.audio_rows.end(), -4.0F);
    const auto clean = guide;
    {
        ImageReferenceDenoiser owner(load, state.config, nullptr, residency);
        const auto legacy = owner.run(text, images, 77);
        state.continuation = true;
        const auto first = owner.run_continuation(text, images, guide, 91, 17.5, 21);
        require(first.forwards == 49 && first.video_rows.size() == 12 * 96 &&
                    first.audio_rows.size() == 6 * 32,
                "guides do not leak into generated-only outputs");
        const std::vector<std::array<uint64_t, 3>> draws(state.draws.end() - 5, state.draws.end());
        require(draws == std::vector<std::array<uint64_t, 3>>(
                    {{576, 91, 0}, {192, 91, 576}, {576, 91, 768},
                     {1152, 91, 1344}, {192, 91, 2496}}),
                "guide video uses one native C/T/H/W draw; clean audio consumes no RNG");
        volatile float reference = 0.999F * 2.0F;
        volatile float random = (1.0F - 0.999F) * ((91.0F + 768.0F + 6.0F) / 4096.0F);
        require(state.prefix[8 * 96 + 2] == reference + random,
                "guide noise is patchified before mixing the packed clean guide");
        require(state.audio_prefix == guide.audio_rows, "stereo guide remains exactly clean");
        const auto second = owner.run_continuation(text, images, guide, 91, 17.5, 21);
        require(first.video_rows == second.video_rows && first.audio_rows == second.audio_rows,
                "guided cold/warm results preserve deterministic request-local RNG");
        const auto count = state.draws.size();
        const auto loads = state.adaln_loads + state.dit_loads;
        auto bad = guide;
        bad.shape.width = 4;
        rejects([&] { owner.run_continuation(text, images, bad, 91, 17.5, 21); });
        bad = guide;
        bad.shape.frames = 3;
        rejects([&] { owner.run_continuation(text, images, bad, 91, 17.5, 21); });
        bad = guide;
        bad.shape.audio_latents = 4;
        rejects([&] { owner.run_continuation(text, images, bad, 91, 17.5, 21); });
        bad = guide;
        bad.video_rows.pop_back();
        rejects([&] { owner.run_continuation(text, images, bad, 91, 17.5, 21); });
        bad = guide;
        bad.audio_rows[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { owner.run_continuation(text, images, bad, 91, 17.5, 21); });
        for (const auto offset : {-1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()})
            rejects([&] { owner.run_continuation(text, images, guide, 91, offset, 21); });
        rejects([&] { owner.run_continuation(text, images, guide, 91, 17.5, 14); });
        require(state.draws.size() == count && state.adaln_loads + state.dit_loads == loads,
                "invalid guide geometry, payload and clock fail before RNG or module work");
        state.continuation = false;
        const auto after = owner.run(text, images, 77);
        require(after.video_rows == legacy.video_rows && after.audio_rows == legacy.audio_rows,
                "switching back retains the exact image-only result");
        const bool resident = residency == ReferenceDenoiserResidency::Resident;
        require(state.adaln_loads == (resident ? 3 : 4) &&
                    state.dit_loads == (resident ? 3 : 4),
                "switching clean-audio presence rebuilds AdaLN without overlapping DiT weights");
        state.continuation = true;
        state.fail = true;
        rejects([&] { owner.run_continuation(text, images, guide, 91, 17.5, 21); });
        require(state.adaln_live == 0 && state.dit_live == 0,
                "guided failure releases all native modules and cached modulations");
        state.fail = false;
        require(owner.run_continuation(text, images, guide, 91, 17.5, 21).forwards == 49,
                "guided execution recovers after an explicit failed forward");
    }
    require(guide.video_rows == clean.video_rows && guide.audio_rows == clean.audio_rows,
            "clean guide buffers are immutable across sampling and failures");
    require(state.adaln_live == 0 && state.dit_live == 0, "guided owner releases resident state");
    rng_state = nullptr;
}
} // namespace

namespace trtmc::minimax_h3 {
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset) {
    require(rng_state != nullptr, "RNG fixture owner");
    rng_state->draws.push_back({count, seed, offset});
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index)
        values[index] = static_cast<float>(seed + offset + index) / 4096.0F;
    return values;
}
uint64_t torch_cuda_normal_consumed_offset(std::size_t count) {
    return count;
}
} // namespace trtmc::minimax_h3

int main() {
    try {
        run_case(ReferenceDenoiserResidency::Staged);
        run_case(ReferenceDenoiserResidency::Resident);
        run_continuation_case(ReferenceDenoiserResidency::Staged);
        run_continuation_case(ReferenceDenoiserResidency::Resident);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

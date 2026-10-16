/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/sampler.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

trtmc::Lfm2SampleResult sample_once(const std::vector<float>& logits,
                                    const trtmc::Lfm2SamplingParams& params) {
    auto sampler = trtmc::create_lfm2_sampler(params);
    return sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params, {});
}

void test_hf_repetition_penalty() {
    const float logits[] = {0.5F, 4.0F, -1.0F, 3.5F};
    const auto adjusted = trtmc::lfm2_apply_repetition_penalty(logits, 4, 2.0F, {1, 2, 1, -1, 99});
    check(adjusted.size() == 4, "penalty preserves vocabulary");
    check(std::abs(adjusted[0] - 0.5F) < 1.0e-6F, "unseen token unchanged");
    check(std::abs(adjusted[1] - 2.0F) < 1.0e-6F, "positive seen score divided once");
    check(std::abs(adjusted[2] + 2.0F) < 1.0e-6F, "negative seen score multiplied once");

    trtmc::Lfm2SamplingParams params;
    params.repetition_penalty = 2.0F;
    auto sampler = trtmc::create_lfm2_sampler(params);
    const auto result = sampler->sample(logits, 4, params, {1, 2});
    check(result.token_id == 3, "history penalty is applied before greedy selection");
}

void test_lfm2_top_k_default_resolution() {
    trtmc::TextGenerationConfig request;
    request.min_p = 0.15F;
    request.temperature = 0.3F;
    const auto defaults = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(defaults.top_k == 1, "generic request retains its source value");
    check(trtmc::lfm2_resolve_top_k(defaults) == 1,
          "generic top_k remains authoritative when other sampling knobs are active");

    request.top_k = 0;
    const auto full_vocab = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(full_vocab) == 0, "explicit top_k zero keeps full vocabulary");

    request.top_k = -1;
    const auto negative_full_vocab = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(negative_full_vocab) == -1,
          "negative top_k keeps the C++ API full-vocabulary contract");

    request.top_k = 17;
    const auto explicit_k = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(explicit_k) == 17, "explicit positive top_k is authoritative");

    request.top_k = 50;
    const auto model_card_k = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(model_card_k) == 50,
          "explicit model-card top_k 50 is preserved");
}

void test_eos_and_seeded_sampling() {
    trtmc::Lfm2SamplingParams params;
    params.temperature = 0.3F;
    params.min_p = 0.15F;
    params.seed = 123;
    params.eos_token_ids = {2, 7};
    const float logits[] = {1.0F, 0.5F, 3.0F, -2.0F};
    auto first = trtmc::create_lfm2_sampler(params);
    auto second = trtmc::create_lfm2_sampler(params);
    const auto a = first->sample(logits, 4, params, {});
    const auto b = second->sample(logits, 4, params, {});
    check(a.token_id == b.token_id, "seeded sampling is reproducible");
    check(trtmc::lfm2_is_eos_token(params, 2) && trtmc::lfm2_is_eos_token(params, 7),
          "all configured EOS ids stop generation");
}

void test_distribution_filters_and_temperature() {
    const std::vector<float> logits = {2.0F, 1.0F, 0.0F};
    trtmc::Lfm2SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.seed = 3;

    const auto unfiltered = sample_once(logits, params);
    check(unfiltered.token_id == 1, "seed samples the second top-k candidate");
    check(std::abs(unfiltered.logprob + 1.3132617F) < 1.0e-6F,
          "sample log probability uses the filtered distribution");

    params.temperature = 0.5F;
    check(sample_once(logits, params).token_id == 0, "temperature scales logits before sampling");

    params.temperature = 1.0F;
    params.top_k = 1;
    const auto top_k = sample_once(logits, params);
    check(top_k.token_id == 0, "top-k limits the candidate set");
    check(std::abs(top_k.logprob) < 1.0e-6F, "single-candidate top-k is renormalized");

    params.top_k = 2;
    params.min_p = 0.4F;
    check(sample_once(logits, params).token_id == 0,
          "min-p filters relative to the maximum probability");

    params.min_p = 0.0F;
    params.top_p = 0.7F;
    check(sample_once(logits, params).token_id == 0,
          "top-p retains the smallest prefix that reaches the threshold");
}

void test_seed_fallback_and_reset() {
    const std::vector<float> logits = {2.0F, 1.0F, 0.0F};
    trtmc::Lfm2SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.seed = 3;

    auto sampler = trtmc::create_lfm2_sampler(params);
    const auto first = sampler->sample(logits.data(), 3, params, {});
    (void)sampler->sample(logits.data(), 3, params, {});
    sampler->reset();
    const auto replay = sampler->sample(logits.data(), 3, params, {});
    check(first.token_id == replay.token_id && std::abs(first.logprob - replay.logprob) < 1.0e-6F,
          "reset restores the initial seeded sequence");

    params.seed = 1;
    check(sample_once(logits, params).token_id == 0, "the request seed controls sampling");

    params.seed = -1;
    const auto fallback = sample_once(logits, params);
    params.seed = 42;
    const auto explicit_fallback = sample_once(logits, params);
    check(fallback.token_id == explicit_fallback.token_id &&
              std::abs(fallback.logprob - explicit_fallback.logprob) < 1.0e-6F,
          "an unspecified seed uses the documented deterministic fallback");

    params.seed = 0;
    const auto zero_seed = sample_once(logits, params);
    params.seed = 1;
    const auto nonzero_state = sample_once(logits, params);
    check(zero_seed.token_id == nonzero_state.token_id &&
              std::abs(zero_seed.logprob - nonzero_state.logprob) < 1.0e-6F,
          "zero seed maps to a nonzero generator state");
}

class FakeSamplerModule final : public trtmc::ITrtModule {
  public:
    enum class BadState { None, Seed, Unchanged, Unaligned, SignedOverflow };
    std::vector<trtmc::TensorInfo> specs{
        {"probabilities", {1, 128000}, trtmc::DType::kFloat32, true},
        {"rng_state", {4}, trtmc::DType::kInt32, true},
        {"token_id", {1}, trtmc::DType::kInt32, false},
        {"next_state", {4}, trtmc::DType::kInt32, false}};
    std::vector<float> probabilities;
    std::vector<std::array<std::uint32_t, 4>> received_states;
    std::array<std::uint32_t, 4> returned{};
    int32_t selected{7};
    std::uint64_t increment{8};
    BadState bad_state{BadState::None};
    bool omit_state{false};
    bool wrong_output_dtype{false};
    bool dynamic{false};
    bool graph{false};
    cudaStream_t stream_value{reinterpret_cast<cudaStream_t>(1)};

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        check(inputs.size() == 2, "sampler receives only probabilities and explicit state");
        const auto& p = inputs.at("probabilities");
        const auto& s = inputs.at("rng_state");
        check(p.dtype == trtmc::DType::kFloat32 && p.shape == std::vector<int64_t>({1, 128000}),
              "sampler receives full FP32 vocabulary shape");
        check(s.dtype == trtmc::DType::kInt32 && s.shape == std::vector<int64_t>{4},
              "sampler receives packed INT32 state");
        const auto* values = static_cast<const float*>(p.data);
        probabilities.assign(values, values + 128000);
        std::memcpy(returned.data(), s.data, sizeof(returned));
        received_states.push_back(returned);
        auto offset = std::uint64_t{returned[2]} | (std::uint64_t{returned[3]} << 32);
        if (bad_state == BadState::Seed)
            returned[0] ^= 1;
        if (bad_state == BadState::SignedOverflow)
            offset = std::uint64_t{1} << 63;
        else if (bad_state != BadState::Unchanged)
            offset += bad_state == BadState::Unaligned ? 3 : increment;
        returned[2] = static_cast<std::uint32_t>(offset);
        returned[3] = static_cast<std::uint32_t>(offset >> 32);
        trtmc::TensorMap outputs{{"token_id", {&selected, {1}, trtmc::DType::kInt32}}};
        if (!omit_state)
            outputs["next_state"] = {returned.data(),
                                     {4},
                                     wrong_output_dtype ? trtmc::DType::kFloat32
                                                        : trtmc::DType::kInt32};
        return outputs;
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return stream_value; }
    void enable_cuda_graph() override { graph = true; }
    bool cuda_graph_active() const override { return graph; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return entries(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return entries(false); }
    bool has_input(const std::string& name) const override {
        return std::any_of(specs.begin(), specs.end(),
                           [&](const auto& value) { return value.name == name && value.is_input; });
    }
    bool has_output(const std::string& name) const override {
        return std::any_of(specs.begin(), specs.end(), [&](const auto& value) {
            return value.name == name && !value.is_input;
        });
    }
    trtmc::DType tensor_dtype(const std::string& name) const override { return spec(name).dtype; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return spec(name).shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override { return tensor_shape(name).size(); }
    bool input_is_dynamic(const std::string&) const override { return dynamic; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    const trtmc::TensorInfo& spec(const std::string& name) const {
        const auto found = std::find_if(specs.begin(), specs.end(),
                                        [&](const auto& value) { return value.name == name; });
        if (found == specs.end())
            throw std::runtime_error("fake sampler missing tensor");
        return *found;
    }
    std::vector<trtmc::TensorInfo> entries(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& value : specs)
            if (value.is_input == input)
                result.push_back(value);
        return result;
    }
};

std::vector<float> philox_logits() {
    std::vector<float> logits(128000, -1000.0F);
    logits[3] = 4.0F;
    logits[7] = 3.0F;
    return logits;
}

void test_module_sampler_processors_and_request_state() {
    FakeSamplerModule module;
    trtmc::Lfm2SamplingParams params;
    params.top_k = 2;
    params.seed = 0;
    params.repetition_penalty = 2;
    params.eos_token_ids = {7};
    const auto logits = philox_logits();
    auto sampler = trtmc::create_lfm2_sampler(params, module);
    const auto result = sampler->sample(logits.data(), logits.size(), params, {3, 3});
    const float expected = 1.0F / (1.0F + std::exp(-1.0F));
    check(result.token_id == 7 && result.is_eos, "module token preserves configured EOS handling");
    check(std::abs(result.logprob - std::log(expected)) < 1.0e-6F,
          "module sample log probability uses unchanged CPU processors");
    check(std::abs(module.probabilities[7] - expected) < 1.0e-6F &&
              std::abs(module.probabilities[3] - (1.0F - expected)) < 1.0e-6F &&
              module.probabilities[0] == 0.0F,
          "repetition and top-k precede full-vocabulary transport");
    (void)sampler->sample(logits.data(), logits.size(), params, {});
    check(module.received_states[0] == std::array<std::uint32_t, 4>{0, 0, 0, 0},
          "Philox seed zero remains zero");
    check(module.received_states[1] == std::array<std::uint32_t, 4>{0, 0, 8, 0},
          "runtime carries stage-owned increments rather than assuming four");
    sampler->reset();
    (void)sampler->sample(logits.data(), logits.size(), params, {});
    check(module.received_states.back() == module.received_states.front(),
          "sampler reset restores explicit request state");
    const auto first_params = params;
    params.seed = -1;
    auto another = trtmc::create_lfm2_sampler(params, module);
    (void)another->sample(logits.data(), logits.size(), params, {});
    check(module.received_states.back() == std::array<std::uint32_t, 4>{42, 0, 0, 0},
          "another request has independent fallback-seed state on the same module");
    (void)sampler->sample(logits.data(), logits.size(), first_params, {});
    check(module.received_states.back() == std::array<std::uint32_t, 4>{0, 0, 8, 0},
          "interleaved sampler objects keep their own states");
}

void test_module_sampler_greedy_and_failures() {
    const auto logits = philox_logits();
    trtmc::Lfm2SamplingParams params;
    FakeSamplerModule greedy;
    auto sampler = trtmc::create_lfm2_sampler(params, greedy);
    check(sampler->sample(logits.data(), logits.size(), params, {}).token_id == 3 &&
              greedy.received_states.empty(),
          "default greedy bypasses sampler engine");
    params.top_k = 50;
    params.seed = 42;
    params.temperature = 0;
    sampler = trtmc::create_lfm2_sampler(params, greedy);
    check(sampler->sample(logits.data(), logits.size(), params, {}).token_id == 3 &&
              greedy.received_states.empty(),
          "zero temperature remains greedy with a seed");
    params.temperature = 1;
    for (const auto bad :
         {FakeSamplerModule::BadState::Seed, FakeSamplerModule::BadState::Unchanged,
          FakeSamplerModule::BadState::Unaligned, FakeSamplerModule::BadState::SignedOverflow}) {
        FakeSamplerModule module;
        module.bad_state = bad;
        auto tested = trtmc::create_lfm2_sampler(params, module);
        bool rejected = false;
        try {
            (void)tested->sample(logits.data(), logits.size(), params, {});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid seed/offset transition is rejected");
    }
    for (int invalid = 0; invalid < 3; ++invalid) {
        FakeSamplerModule module;
        module.omit_state = invalid == 0;
        module.wrong_output_dtype = invalid == 1;
        module.selected = invalid == 2 ? 128000 : 7;
        auto tested = trtmc::create_lfm2_sampler(params, module);
        bool rejected = false;
        try {
            (void)tested->sample(logits.data(), logits.size(), params, {});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid sampler outputs are rejected");
    }
}

void test_module_sampler_contract() {
    trtmc::Lfm2SamplingParams params;
    for (int invalid = 0; invalid < 9; ++invalid) {
        FakeSamplerModule module;
        if (invalid == 0)
            module.specs[0].dtype = trtmc::DType::kBFloat16;
        if (invalid == 1)
            module.specs[0].shape = {128000};
        if (invalid == 2)
            module.specs[1].dtype = trtmc::DType::kFloat32;
        if (invalid == 3)
            module.specs[3].shape = {1};
        if (invalid == 4)
            module.dynamic = true;
        if (invalid == 5)
            module.graph = true;
        if (invalid == 6)
            module.stream_value = nullptr;
        if (invalid == 7)
            module.specs[1].name = "wrong_state";
        if (invalid == 8)
            module.specs.push_back({"extra", {1}, trtmc::DType::kInt32, false});
        bool rejected = false;
        try {
            (void)trtmc::create_lfm2_sampler(params, module);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "new sampler module contract is checked even for greedy requests");
    }
}

} // namespace

int main() {
    test_hf_repetition_penalty();
    test_lfm2_top_k_default_resolution();
    test_eos_and_seeded_sampling();
    test_distribution_filters_and_temperature();
    test_seed_fallback_and_reset();
    test_module_sampler_processors_and_request_state();
    test_module_sampler_greedy_and_failures();
    test_module_sampler_contract();
    return failures;
}

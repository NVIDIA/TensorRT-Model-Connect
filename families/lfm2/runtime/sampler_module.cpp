/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/sampler.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace trtmc {
namespace {

class Lfm2PhiloxSampler final : public Lfm2ISampler {
  public:
    Lfm2PhiloxSampler(std::uint64_t seed, ITrtModule& module) : seed_(seed), module_(module) {}

    Lfm2SampleResult sample(const float* logits, int32_t vocab_size,
                            const Lfm2SamplingParams& params,
                            const std::vector<int32_t>& token_history) override {
        if (vocab_size != 128000)
            throw std::invalid_argument("LFM2 Philox sampler requires a 128000-token vocabulary");
        if (!lfm2_sampling_uses_distribution(params))
            return create_lfm2_sampler(params)->sample(logits, vocab_size, params, token_history);
        const auto distribution =
            lfm2_sampling_distribution(logits, vocab_size, params, token_history);
        if (distribution.indices.empty())
            return {0, 0.0F, lfm2_is_eos_token(params, 0)};
        std::vector<float> probabilities(static_cast<std::size_t>(vocab_size));
        for (std::size_t index = 0; index < distribution.indices.size(); ++index)
            probabilities[distribution.indices[index]] = distribution.probabilities[index];
        const std::array<std::uint32_t, 4> raw_state{
            static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(seed_ >> 32),
            static_cast<std::uint32_t>(offset_), static_cast<std::uint32_t>(offset_ >> 32)};
        std::array<std::int32_t, 4> state{};
        std::memcpy(state.data(), raw_state.data(), sizeof(state));
        TensorMap inputs;
        inputs["probabilities"] = {probabilities.data(), {1, vocab_size}, DType::kFloat32};
        inputs["rng_state"] = {state.data(), {4}, DType::kInt32};
        const auto outputs = module_.forward(inputs);
        const auto token = outputs.find("token_id");
        const auto next = outputs.find("next_state");
        if (outputs.size() != 2 || token == outputs.end() || next == outputs.end() ||
            token->second.data == nullptr || next->second.data == nullptr ||
            token->second.dtype != DType::kInt32 || next->second.dtype != DType::kInt32 ||
            token->second.shape != std::vector<int64_t>{1} ||
            next->second.shape != std::vector<int64_t>{4})
            throw std::runtime_error("LFM2 sampler returned invalid token/state tensors");
        int32_t selected = 0;
        std::array<std::uint32_t, 4> returned{};
        std::memcpy(&selected, token->second.data, sizeof(selected));
        std::memcpy(returned.data(), next->second.data, sizeof(returned));
        const auto returned_seed = std::uint64_t{returned[0]} | (std::uint64_t{returned[1]} << 32);
        const auto returned_offset =
            std::uint64_t{returned[2]} | (std::uint64_t{returned[3]} << 32);
        if (selected < 0 || selected >= vocab_size || returned_seed != seed_ ||
            returned_offset <= offset_ || returned_offset % 4 != 0 ||
            returned_offset > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()))
            throw std::runtime_error("LFM2 sampler returned an invalid token or RNG transition");
        offset_ = returned_offset;
        return {selected,
                std::log(std::max(probabilities[selected], std::numeric_limits<float>::min())),
                lfm2_is_eos_token(params, selected)};
    }
    void reset() override { offset_ = 0; }

  private:
    std::uint64_t seed_;
    std::uint64_t offset_{0};
    ITrtModule& module_;
};

} // namespace

void lfm2_validate_sampler_module(const ITrtModule& module) {
    if (!module.ok() || module.stream() == nullptr || module.input_info().size() != 2 ||
        module.output_info().size() != 2 || !module.has_input("probabilities") ||
        !module.has_input("rng_state") || !module.has_output("token_id") ||
        !module.has_output("next_state") || module.cuda_graph_active() ||
        module.cuda_graph_captured())
        throw std::invalid_argument("LFM2 sampler module has an invalid execution contract");
    if (module.tensor_dtype("probabilities") != DType::kFloat32 ||
        module.tensor_shape("probabilities") != std::vector<int64_t>({1, 128000}) ||
        module.input_is_dynamic("probabilities") || module.input_is_dynamic("rng_state"))
        throw std::invalid_argument("LFM2 sampler requires fixed FP32 [1,128000] probabilities");
    for (const char* name : {"rng_state", "next_state", "token_id"}) {
        const std::vector<int64_t> shape{std::string_view(name) == "token_id" ? 1 : 4};
        if (module.tensor_dtype(name) != DType::kInt32 || module.tensor_shape(name) != shape)
            throw std::invalid_argument("LFM2 sampler requires INT32 token/state tensors");
    }
}

std::unique_ptr<Lfm2ISampler> create_lfm2_sampler(const Lfm2SamplingParams& params,
                                                  ITrtModule& module) {
    lfm2_validate_sampler_module(module);
    if (!lfm2_sampling_uses_distribution(params))
        return create_lfm2_sampler(params);
    const auto seed = params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : 42ULL;
    return std::make_unique<Lfm2PhiloxSampler>(seed, module);
}

} // namespace trtmc

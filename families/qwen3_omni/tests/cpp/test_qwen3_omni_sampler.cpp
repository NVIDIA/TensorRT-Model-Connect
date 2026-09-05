/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/residual_multinomial_kernel.h"
#include "families/qwen3_omni/runtime/sampler.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using trtmc::qwen3_omni::ResidualCodeSampler;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Callback>
bool throws(Callback&& callback) {
    try {
        callback();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::vector<float> increasing_logits() {
    std::vector<float> logits(2048);
    for (std::size_t index = 0; index < logits.size(); ++index)
        logits[index] = static_cast<float>(index) / 100.0F;
    return logits;
}

void test_same_seed_is_deterministic() {
    const auto logits = increasing_logits();
    ResidualCodeSampler first(42);
    ResidualCodeSampler second(42);
    const std::vector<std::int32_t> expected = {
        2037, 2036, 2024, 2010, 2015, 2010, 2045, 2039, 2046, 2011,
        2046, 2044, 2046, 2010, 2022, 2023, 2020, 2043, 2044, 2043,
    };

    for (const auto token : expected) {
        check(first.sample(logits.data(), logits.size()) == token,
              "seed 42 follows the explicit token sequence");
        check(second.sample(logits.data(), logits.size()) == token,
              "independent sampler with seed 42 follows the same sequence");
    }
    check(first.draws() == expected.size(), "one successful sample consumes one RNG draw");
    check(second.draws() == expected.size(), "deterministic peer consumes the same draws");
}

void test_rng_state_continues_across_calls() {
    const auto logits = increasing_logits();
    ResidualCodeSampler request_sampler(42);
    const auto first = request_sampler.sample(logits.data(), logits.size());
    const auto second = request_sampler.sample(logits.data(), logits.size());
    ResidualCodeSampler restarted(42);

    check(first == 2037, "first request draw matches the seed-42 sequence");
    check(second == 2036, "second request draw advances the persistent RNG state");
    check(restarted.sample(logits.data(), logits.size()) == first,
          "constructing a new request sampler restarts the sequence");
    check(request_sampler.draws() == 2, "persistent request sampler records both draws");
}

void test_pinned_transformers_draw() {
    // Nonzero probabilities captured from pinned Transformers 5.2.0 with
    // torch 2.12.0+cu130 immediately before residual draw 16. This separates
    // CUDA exponential-race parity from logits filtering.
    const std::pair<std::int32_t, float> entries[] = {
        {99, 0.10544615983963013F},   {198, 0.01832379400730133F},  {208, 0.02076357789337635F},
        {268, 0.016170693561434746F}, {309, 0.023528218269348145F}, {322, 0.10544615983963013F},
        {349, 0.01832379400730133F},  {351, 0.03021083027124405F},  {436, 0.016170693561434746F},
        {478, 0.01832379400730133F},  {534, 0.016170693561434746F}, {558, 0.01832379400730133F},
        {644, 0.02076357789337635F},  {694, 0.023528218269348145F}, {722, 0.03423335403203964F},
        {772, 0.026660963892936707F}, {776, 0.08212155103683472F},  {820, 0.04395649582147598F},
        {1329, 0.04395649582147598F}, {1349, 0.05644126236438751F}, {1428, 0.016170693561434746F},
        {1523, 0.01832379400730133F}, {1566, 0.09305590391159058F}, {1618, 0.026660963892936707F},
        {1776, 0.04980923980474472F}, {1876, 0.01832379400730133F}, {1928, 0.0387914776802063F},
    };
    std::vector<float> probabilities(2048, 0.0F);
    for (const auto& [token, probability] : entries)
        probabilities[static_cast<std::size_t>(token)] = probability;

    float* device_probabilities = nullptr;
    std::int32_t* device_token = nullptr;
    const auto probability_bytes = probabilities.size() * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&device_probabilities), probability_bytes) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_token), sizeof(std::int32_t)) != cudaSuccess) {
        cudaFree(device_probabilities);
        cudaFree(device_token);
        check(false, "CUDA allocations for the pinned Transformers draw succeed");
        return;
    }
    const cudaError_t upload = cudaMemcpy(device_probabilities, probabilities.data(),
                                          probability_bytes, cudaMemcpyHostToDevice);
    trtmc::qwen3_omni::launch_residual_exponential_race(device_probabilities, probabilities.size(),
                                                        42, 16, device_token, nullptr);
    const cudaError_t launch = cudaGetLastError();
    std::int32_t token = -1;
    const cudaError_t download =
        cudaMemcpy(&token, device_token, sizeof(token), cudaMemcpyDeviceToHost);
    cudaFree(device_probabilities);
    cudaFree(device_token);

    check(upload == cudaSuccess, "pinned Transformers probabilities upload to CUDA");
    check(launch == cudaSuccess, "pinned Transformers exponential race launches");
    check(download == cudaSuccess, "pinned Transformers token returns from CUDA");
    check(token == 776, "pinned Transformers residual draw selects token 776");
}

void test_top_k_and_top_p_boundaries() {
    // One candidate alone exceeds top-p=0.8, so the nucleus has one token
    // even though top-k retained fifty candidates.
    std::vector<float> peaked_logits(2048, 0.0F);
    peaked_logits[7] = 10.0F;
    ResidualCodeSampler peaked_sampler(99);
    for (int draw = 0; draw < 32; ++draw) {
        check(peaked_sampler.sample(peaked_logits.data(), peaked_logits.size()) == 7,
              "top-p collapses a sufficiently peaked distribution to one token");
    }
}

void test_invalid_inputs_do_not_advance_rng() {
    auto logits = increasing_logits();
    ResidualCodeSampler sampler(42);
    std::vector<float> too_short(49, 0.0F);
    std::vector<float> empty;

    check(throws([&] { sampler.sample(nullptr, logits.size()); }), "null logits fail closed");
    check(throws([&] { sampler.sample(empty.data(), empty.size()); }), "empty logits fail closed");
    check(throws([&] { sampler.sample(too_short.data(), too_short.size()); }),
          "vocabulary smaller than top-k fails closed");

    logits[3] = std::numeric_limits<float>::quiet_NaN();
    check(throws([&] { sampler.sample(logits.data(), logits.size()); }), "NaN logits fail closed");
    logits[3] = std::numeric_limits<float>::infinity();
    check(throws([&] { sampler.sample(logits.data(), logits.size()); }),
          "infinite logits fail closed");
    check(sampler.draws() == 0, "invalid inputs consume no RNG draws");

    logits[3] = 0.03F;
    check(sampler.sample(logits.data(), logits.size()) == 2037,
          "first valid sample still starts at the original seed");
    check(sampler.draws() == 1, "valid recovery consumes exactly one draw");
}

} // namespace

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    test_same_seed_is_deterministic();
    test_rng_state_continues_across_calls();
    test_pinned_transformers_draw();
    test_top_k_and_top_p_boundaries();
    test_invalid_inputs_do_not_advance_rng();

    if (failures != 0)
        std::cerr << failures << " Qwen3-Omni sampler test(s) failed\n";
    return failures == 0 ? 0 : 1;
}

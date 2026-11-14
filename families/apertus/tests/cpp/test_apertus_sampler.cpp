/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// CPU checks of the Apertus host sampler.

#include "families/apertus/runtime/sampler.h"

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAILED: " << label << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using trtmc::apertus::Sampler;
    using trtmc::apertus::SamplingParams;
    const std::vector<float> tied{0.5F, 2.0F, 2.0F, -1.0F};

    expect(Sampler(SamplingParams{}).sample(tied) == 1, "default greedy takes the first maximum");
    SamplingParams cold;
    cold.temperature = 0.0F;
    cold.top_k = 0;
    expect(Sampler(cold).sample(tied) == 1, "temperature 0 is greedy");

    SamplingParams random;
    random.top_k = 0;
    random.seed = 7;
    std::vector<std::int32_t> first, second;
    Sampler a(random), b(random);
    std::set<std::int32_t> seen;
    for (int step = 0; step < 64; ++step) {
        first.push_back(a.sample(tied));
        second.push_back(b.sample(tied));
        seen.insert(first.back());
    }
    expect(first == second, "a seeded sampler is deterministic");
    expect(seen.size() > 1, "unrestricted sampling visits more than one token");

    SamplingParams warm = random;
    warm.temperature = 5e-7F;
    Sampler small_temperature(warm);
    std::set<std::int32_t> tied_seen;
    for (int step = 0; step < 64; ++step)
        tied_seen.insert(small_temperature.sample(tied));
    expect(tied_seen == std::set<std::int32_t>{1, 2},
           "a small positive temperature samples among the tied maxima");

    SamplingParams top_two = random;
    top_two.top_k = 2;
    Sampler limited(top_two);
    bool inside = true;
    for (int step = 0; step < 64; ++step) {
        const auto token = limited.sample(tied);
        inside = inside && (token == 1 || token == 2);
    }
    expect(inside, "top-k keeps only the k most likely tokens");

    SamplingParams nucleus = random;
    nucleus.top_p = 0.1F;
    Sampler narrow(nucleus);
    expect(narrow.sample({0.0F, 5.0F, 1.0F}) == 1, "a small nucleus keeps the most likely token");

    SamplingParams floor = random;
    floor.min_p = 0.9F;
    Sampler strict(floor);
    expect(strict.sample({0.0F, 5.0F, 1.0F}) == 1, "min_p removes unlikely tokens");

    try {
        (void)Sampler(SamplingParams{}).sample({});
        expect(false, "empty logits are rejected");
    } catch (const std::invalid_argument&) {
    }
    if (failures == 0)
        std::cout << "test_apertus_sampler: all checks passed\n";
    return failures == 0 ? 0 : 1;
}

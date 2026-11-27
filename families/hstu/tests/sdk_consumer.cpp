/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sdk_fixture.h"
#include "trtmc/trtmc.hpp"

#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    HstuSdkFixture fixture;
    if (!hstu_fixture_load(argv[3], &fixture))
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::StructuredRanking>();
        task.configure_cache(0, 0.01);
        auto small = task.cache_stats();
        task.configure_cache(0, 0.05);
        auto large = task.cache_stats();
        const double factor = double(large.budget_bytes) / double(small.budget_bytes);
        if (!small.supported || factor < 4.5 || factor > 5.5)
            throw std::runtime_error("SDK cache ratio has no effect");
        task.configure_cache(134217728);
        for (const auto ratio : {-1., 1.1, std::numeric_limits<double>::infinity()}) {
            try {
                task.configure_cache(0, ratio);
                throw std::runtime_error("invalid SDK ratio accepted");
            } catch (const trtmc::Error& e) {
                if (e.code() != TRTMC_INVALID_ARGUMENT)
                    throw;
            }
        }
        std::vector<trtmc::RankingFeatureView> features;
        for (const auto& t : fixture.features)
            features.push_back({{t.name.data, t.name.size},
                                static_cast<trtmc::RankingFeatureType>(t.type),
                                {t.shape, t.rank},
                                {static_cast<const std::uint8_t*>(t.data), t.byte_size}});
        auto result = task.run({{features.data(), features.size()}});
        const auto scores = result.scores();
        if (result.kind() != TRTMC_SCORE_PROBABILITY || scores.rows != fixture.batch ||
            scores.columns != fixture.candidates)
            throw std::runtime_error("SDK ranking output schema changed");
        const auto atol = std::stof(argv[4]), rtol = std::stof(argv[5]);
        for (std::uint64_t i = 0; i < scores.count; ++i)
            if (!std::isfinite(scores.data[i]) || std::abs(scores.data[i] - fixture.expected[i]) >
                                                      atol + rtol * std::abs(fixture.expected[i]))
                throw std::runtime_error("SDK score differs from oracle");
        const auto stats = task.cache_stats();
        if (stats.budget_bytes != 134217728 || stats.used_bytes || stats.active_sequences ||
            !stats.peak_bytes)
            throw std::runtime_error("SDK cache metrics are incorrect");
        hstu_fixture_free(&fixture);
        std::cout << "HSTU C++ SDK scores, effective ratio and invalid controls: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        hstu_fixture_free(&fixture);
        std::cerr << e.what() << '\n';
        return 1;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/ranking.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class Function>
void rejects(Function fn, trtmc_status status) {
    try {
        fn();
    } catch (const trtmc::Error& e) {
        check(e.code() == status, "wrong rejection status");
        return;
    }
    throw std::runtime_error("invalid ranking operation accepted");
}
void bundle(const std::filesystem::path& file, const std::string& mode) {
    const char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string json = "{\"format\":1,\"family\":\"ranking_fixture\",\"task\":\"" + mode +
                             "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream out(file, std::ios::binary);
    out.write(magic, 8);
    for (unsigned s = 0; s < 64; s += 8)
        out.put(char((std::uint64_t(json.size()) >> s) & 255));
    out.write(json.data(), json.size());
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[1];
        auto load = [&](const char* mode) {
            auto path =
                std::filesystem::path(argv[1]) / (std::string("ranking-cpp-") + mode + ".bundle");
            bundle(path, mode);
            return trtmc::Model::load(path.string(), options);
        };
        auto model = load("structured_ranking");
        auto task = model.task<trtmc::StructuredRanking>();
        const float x[] = {.25F, .5F, .75F, 1.F};
        const std::int32_t n[] = {3, 4};
        const std::int64_t t[] = {1LL << 40, 2LL << 40};
        const std::int64_t matrix[] = {2, 2}, vector[] = {2};
        trtmc::RankingFeatureView features[] = {
            {"continuous",
             trtmc::RankingFeatureType::Float32,
             {matrix, 2},
             {reinterpret_cast<const std::uint8_t*>(x), sizeof(x)}},
            {"counts",
             trtmc::RankingFeatureType::Int32,
             {vector, 1},
             {reinterpret_cast<const std::uint8_t*>(n), sizeof(n)}},
            {"times",
             trtmc::RankingFeatureType::Int64,
             {vector, 1},
             {reinterpret_cast<const std::uint8_t*>(t), sizeof(t)}}};
        auto result = task.run({{features, 3}}, {{"scale", 2.0}});
        check(result.kind() == TRTMC_SCORE_LOGIT && result.scores().rows == 2 &&
                  result.scores().columns == 2,
              "ranking output shape/kind");
        const auto scores = result.scores();
        check(scores.data[0] == 8.5F && scores.data[3] == 14.F, "mixed feature values truncated");
        auto fields = task.config_fields();
        check(fields.size() == 1 && fields[0].name == "scale", "ranking config metadata missing");
        task.configure_cache(0, .25);
        check(task.cache_stats().budget_bytes == 250000, "cache ratio ignored");
        task.configure_cache(UINT64_MAX);
        check(task.cache_stats().budget_bytes == UINT64_MAX, "cache byte cap truncated");
        rejects([&] { task.configure_cache(1); }, TRTMC_INVALID_ARGUMENT);
        rejects([&] { task.configure_cache(1000, .1); }, TRTMC_INVALID_ARGUMENT);
        rejects([&] { task.run({{features, 3}}, {{"unknown", 1}}); }, TRTMC_INVALID_CONFIG);
        auto input_guard = load("must_not_run").task<trtmc::StructuredRanking>();
        auto bad = features[1];
        features[1].name = "continuous";
        rejects([&] { input_guard.run({{features, 3}}); }, TRTMC_INVALID_ARGUMENT);
        features[1] = bad;
        features[1].type = static_cast<trtmc::RankingFeatureType>(99);
        rejects([&] { input_guard.run({{features, 3}}); }, TRTMC_INVALID_ARGUMENT);
        features[1] = bad;
        features[1].bytes = {};
        rejects([&] { input_guard.run({{features, 3}}); }, TRTMC_INVALID_ARGUMENT);
        features[1] = bad;
        rejects([&] { input_guard.run({}); }, TRTMC_INVALID_ARGUMENT);
        for (const auto* mode : {"bad_shape", "bad_kind", "bad_scores"}) {
            auto invalid = load(mode).task<trtmc::StructuredRanking>();
            rejects([&] { invalid.run({{features, 3}}); }, TRTMC_INTERNAL_ERROR);
        }
        auto disabled = load("disabled");
        rejects([&] { disabled.task<trtmc::StructuredRanking>(); }, TRTMC_UNSUPPORTED);
        auto no_cache = load("no_cache").task<trtmc::StructuredRanking>();
        check(!no_cache.cache_stats().supported, "unsupported cache claimed");
        rejects([&] { no_cache.configure_cache(1024); }, TRTMC_UNSUPPORTED);
        model = load("disabled");
        check(result.scores().data[0] == 8.5F, "result storage lost");
        std::cout << "ranking C++ ABI, input validation, config, cache and lifetime: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

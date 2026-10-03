/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the public parser and API repeatedly without restarting the runtime.
#define main hstu_single_request_main
#include "families/hstu/runtime/runner.cpp"
#undef main

#include "trtmc/history_cache.h"

namespace {

Json cache_stats(const trtmc::HistoryCacheStats& value) {
    return {{"hits", value.hits},
            {"misses", value.misses},
            {"storage_hits", value.storage_hits},
            {"load_failures", value.load_failures},
            {"store_failures", value.store_failures},
            {"erase_failures", value.erase_failures},
            {"publications", value.publications},
            {"rejected_publications", value.rejected_publications},
            {"stale_publications", value.stale_publications},
            {"evictions", value.evictions},
            {"resident_entries", value.resident_entries},
            {"resident_bytes", value.resident_bytes},
            {"live_bytes", value.live_bytes}};
}

Json cache_output(const trtmc::RecommendationResult& result) {
    auto output = output_json(result);
    for (std::size_t index = 0; index < result.sequences.size(); ++index) {
        const auto& cache = result.sequences[index].cache;
        output["sequences"][index]["cache"] = {
            {"source", cache.source},
            {"reason", cache.reason},
            {"history_tokens", cache.history_tokens},
            {"reused_history_tokens", cache.reused_history_tokens},
            {"computed_tokens", cache.computed_tokens},
            {"published", cache.published}};
    }
    return output;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = options(argc, argv);
        std::ifstream input(args.at("--input-json"));
        Json trace;
        if (!input || !(input >> trace))
            throw std::runtime_error("cannot read HSTU cache request trace");
        const auto& settings = trace.at("cache");
        trtmc::HistoryCacheOptions cache_options;
        cache_options.max_bytes = settings.at("max_bytes").get<std::size_t>();
        cache_options.max_entries = settings.value("max_entries", std::size_t{128});
        const auto storage_bytes = settings.value("storage_max_bytes", std::size_t{0});
        if (storage_bytes) {
            cache_options.storage = std::make_shared<trtmc::InMemoryHistoryCacheStorage>(
                storage_bytes, cache_options.max_entries);
        }
        cache_options.write_through = settings.value("write_through", false);
        auto cache = std::make_shared<trtmc::HistoryCache>(std::move(cache_options));

        const bool graphs = trace.value("cuda_graphs", false);
        auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"), 0, {}, graphs);
        auto baseline_task =
            trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"), 0, {}, graphs);
        auto* model = dynamic_cast<trtmc::IRecommendation*>(task.get());
        auto* baseline = dynamic_cast<trtmc::IRecommendation*>(baseline_task.get());
        auto* consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(task.get());
        auto* baseline_consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(baseline_task.get());
        if (!model || !baseline || !consumer || !baseline_consumer)
            throw std::runtime_error("trace requires an HSTU bundle with history-cache capability");
        consumer->set_history_cache(cache);
        baseline_consumer->set_history_cache(nullptr);
        const auto artifact = consumer->history_cache_artifact_id();
        Json report = {{"artifact_id", artifact}, {"steps", Json::array()}};
        for (const auto& step : trace.at("steps")) {
            Json row = {{"name", step.at("name")}, {"stats_before", cache_stats(cache->stats())}};
            const auto action = step.value("action", std::string{});
            if (action == "clear_memory") {
                cache->clear_memory();
            } else if (action == "invalidate") {
                const auto& identity = step.at("identity");
                cache->invalidate({artifact, identity.at("feature_version").get<std::string>(),
                                   identity.at("subject_id").get<std::string>(),
                                   identity.at("history_epoch").get<std::string>()});
            } else if (!action.empty()) {
                throw std::invalid_argument("unknown cache trace action " + action);
            } else {
                trtmc::RecommendationRequest request;
                try {
                    for (const auto& sequence : step.at("request").at("sequences"))
                        request.sequences.push_back(read_sequence(sequence));
                } catch (const std::exception& error) {
                    row["error"] = error.what();
                    row["baseline_error"] = error.what();
                }
                if (!row.contains("error")) {
                    try {
                        const auto start = Clock::now();
                        const auto result = baseline->recommend(request);
                        row["baseline_ms"] =
                            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                        row["baseline"] = cache_output(result);
                    } catch (const std::exception& error) {
                        row["baseline_error"] = error.what();
                    }
                    try {
                        const auto start = Clock::now();
                        const auto result = model->recommend(request);
                        row["cached_ms"] =
                            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                        row["actual"] = cache_output(result);
                    } catch (const std::exception& error) {
                        row["error"] = error.what();
                    }
                }
            }
            row["stats_after"] = cache_stats(cache->stats());
            report["steps"].push_back(std::move(row));
        }
        report["stats"] = cache_stats(cache->stats());
        std::ofstream output(args.at("--output-json"));
        if (!output || !(output << report.dump(2) << '\n'))
            throw std::runtime_error("cannot write HSTU cache audit JSON");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hstu-cache-sequence-runner: " << error.what() << '\n';
        return 1;
    }
}

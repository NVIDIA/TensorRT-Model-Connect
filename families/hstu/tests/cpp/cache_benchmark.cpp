/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Native API microbenchmark. The input extends the session trace JSON with
// optional warmups, iterations, and storage_max_bytes settings.
#define main hstu_single_request_main
#include "families/hstu/runtime/runner.cpp"
#undef main

#include "trtmc/history_cache.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <functional>
#include <numeric>

namespace {

using Sequence = trtmc::RecommendationSequence;
using Result = trtmc::RecommendationSequenceResult;
using CacheReport = trtmc::RecommendationCacheReport;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

double timed(const std::function<void()>& operation) {
    cuda_check(cudaDeviceSynchronize(), "synchronize before measurement");
    const auto start = Clock::now();
    operation();
    cuda_check(cudaDeviceSynchronize(), "synchronize after measurement");
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Json distribution(const std::vector<double>& samples) {
    require(!samples.empty(), "empty latency sample set");
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    // Nearest-rank percentiles retain the measured observations.
    const auto percentile = [&](double fraction) {
        return ordered[static_cast<std::size_t>(std::ceil(fraction * ordered.size())) - 1];
    };
    return {{"count", samples.size()},
            {"p50_ms", percentile(0.50)},
            {"p95_ms", percentile(0.95)},
            {"p99_ms", percentile(0.99)},
            {"min_ms", ordered.front()},
            {"max_ms", ordered.back()},
            {"mean_ms", std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()},
            {"samples_ms", samples}};
}

Json cache_json(const CacheReport& value) {
    return {{"source", value.source},
            {"reason", value.reason},
            {"history_tokens", value.history_tokens},
            {"reused_history_tokens", value.reused_history_tokens},
            {"computed_tokens", value.computed_tokens},
            {"published", value.published}};
}

Json stats_json(const trtmc::HistoryCacheStats& value) {
    return {{"hits", value.hits},
            {"misses", value.misses},
            {"storage_hits", value.storage_hits},
            {"publications", value.publications},
            {"rejected_publications", value.rejected_publications},
            {"stale_publications", value.stale_publications},
            {"load_failures", value.load_failures},
            {"store_failures", value.store_failures},
            {"erase_failures", value.erase_failures},
            {"evictions", value.evictions},
            {"resident_bytes", value.resident_bytes},
            {"live_bytes", value.live_bytes}};
}

Json environment() {
    int device = 0, driver = 0, runtime = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "read CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "read CUDA device properties");
    cuda_check(cudaDriverGetVersion(&driver), "read CUDA driver version");
    cuda_check(cudaRuntimeGetVersion(&runtime), "read CUDA runtime version");
    return {{"gpu", properties.name},
            {"device", device},
            {"compute_capability",
             std::to_string(properties.major) + "." + std::to_string(properties.minor)},
            {"global_memory_bytes", properties.totalGlobalMem},
            {"cuda_driver_version", driver},
            {"cuda_runtime_version", runtime}};
}

void compare_vector(const std::vector<float>& actual, const std::vector<float>& expected,
                    double rtol, double atol, const std::string& name) {
    require(actual.size() == expected.size(), name + " shape mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double left = actual[index], right = expected[index];
        if (!std::isfinite(left) || !std::isfinite(right))
            throw std::runtime_error(name + " nonfinite result");
        if (std::abs(left - right) > atol + rtol * std::abs(right))
            throw std::runtime_error(name + " differs at index " + std::to_string(index));
    }
}

void compare(const Result& actual, const Result& expected, double rtol, double atol) {
    require(actual.candidate_item_ids == expected.candidate_item_ids, "candidate order mismatch");
    require(actual.num_candidates == expected.num_candidates, "candidate count mismatch");
    require(actual.output_dim == expected.output_dim, "output width mismatch");
    require(actual.embedding_dim == expected.embedding_dim, "embedding width mismatch");
    require(actual.sequence_length == expected.sequence_length, "sequence length mismatch");
    compare_vector(actual.logits, expected.logits, rtol, atol, "logits");
    compare_vector(actual.scores, expected.scores, rtol, atol, "scores");
    compare_vector(actual.embeddings, expected.embeddings, rtol, atol, "embeddings");
    compare_vector(actual.sequence_embeddings, expected.sequence_embeddings, rtol, atol,
                   "sequence embeddings");
}

std::int64_t greedy_item(const Result& result) {
    const auto& values = result.scores.empty() ? result.logits : result.scores;
    const auto width = result.scores.empty() ? result.output_dim : 1;
    require(width > 0 && !result.candidate_item_ids.empty() &&
                values.size() == result.candidate_item_ids.size() * width,
            "greedy loop requires candidate scores");
    std::size_t best = 0;
    for (std::size_t index = 1; index < result.candidate_item_ids.size(); ++index) {
        if (values[index * width] > values[best * width])
            best = index;
    }
    return result.candidate_item_ids[best];
}

struct Benchmark {
    const Json& settings;
    trtmc::IRecommendation& model;
    trtmc::IRecommendation& baseline;
    trtmc::IRecommendationSessionFactory& sessions;
    std::shared_ptr<trtmc::HistoryCache> cache;
    trtmc::HistoryCacheKey key;
    Sequence initial;
    std::vector<std::int64_t> candidates;
    std::vector<std::int64_t> candidate_timestamps;
    std::size_t budget, warmups, iterations, steps;
    double rtol, atol;

    Sequence request(const Sequence& history) const {
        auto value = history;
        value.candidate_item_ids = candidates;
        value.token_timestamps.insert(value.token_timestamps.end(), candidate_timestamps.begin(),
                                      candidate_timestamps.end());
        return value;
    }

    trtmc::RecommendationHistoryAppend update(std::int64_t item, std::size_t step) const {
        trtmc::RecommendationHistoryAppend value;
        value.item_ids = {item};
        if (settings.contains("append_action_id"))
            value.action_ids = {settings.at("append_action_id").get<std::int64_t>()};
        if (settings.contains("append_timestamp")) {
            const auto stamp = settings.at("append_timestamp").get<std::int64_t>() +
                               2 * static_cast<std::int64_t>(step);
            value.token_timestamps = {stamp};
            if (!value.action_ids.empty())
                value.token_timestamps.push_back(stamp + 1);
        }
        return value;
    }

    static void append(Sequence& history, const trtmc::RecommendationHistoryAppend& value) {
        history.history_item_ids.insert(history.history_item_ids.end(), value.item_ids.begin(),
                                        value.item_ids.end());
        history.history_action_ids.insert(history.history_action_ids.end(),
                                          value.action_ids.begin(), value.action_ids.end());
        history.token_timestamps.insert(history.token_timestamps.end(),
                                        value.token_timestamps.begin(),
                                        value.token_timestamps.end());
    }

    Result recommend(trtmc::IRecommendation& task, const Sequence& value) const {
        auto output = task.recommend({{value}});
        require(output.sequences.size() == 1, "benchmark expects one logical sequence");
        return std::move(output.sequences.front());
    }

    void seed() {
        cache->invalidate(key);
        const auto result = recommend(model, request(initial));
        require(result.cache.source == "miss" && result.cache.published,
                "history seed must publish a cold miss");
    }

    Json repeated(const Sequence& value, const std::function<void()>& prepare,
                  const std::function<Result()>& operation,
                  const std::function<void(const CacheReport&)>& check) {
        const auto expected = recommend(baseline, value);
        Result first, last;
        std::vector<double> samples;
        samples.reserve(iterations);
        for (std::size_t index = 0; index < warmups + iterations; ++index) {
            prepare();
            Result result;
            const double elapsed = timed([&] { result = operation(); });
            check(result.cache);
            compare(result, expected, rtol, atol);
            if (index >= warmups) {
                samples.push_back(elapsed);
                if (index == warmups)
                    first = result;
                last = std::move(result);
            }
        }
        auto output = distribution(samples);
        output["first_cache"] = cache_json(first.cache);
        output["last_cache"] = cache_json(last.cache);
        output["all_outputs_match_full_recompute"] = true;
        return output;
    }

    Json cache_cases() {
        const auto original = request(initial);
        auto history_count = initial.history_item_ids.size() + initial.history_action_ids.size();
        for (const auto& feature : initial.contextual_features)
            history_count += feature.ids.size();
        require(history_count <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                "history token count exceeds int32");
        const auto history_tokens = static_cast<std::int32_t>(history_count);
        require(history_tokens > 0, "benchmark requires nonempty history");
        Json output;
        output["full_recompute"] = repeated(
            original, [] {}, [&] { return recommend(baseline, original); },
            [](const CacheReport& value) {
                require((value.source.empty() || value.source == "disabled") &&
                            value.reused_history_tokens == 0 && !value.published,
                        "full recompute must disable cache reuse");
            });
        seed();
        output["gpu_history_hit"] = repeated(
            original, [] {}, [&] { return recommend(model, original); },
            [&](const CacheReport& value) {
                require(value.source == "memory" && value.reused_history_tokens == history_tokens,
                        "GPU hit must reuse the complete history");
                require(!value.published, "GPU exact hit must not publish a replacement");
            });
        output["host_history_reload"] = repeated(
            original, [&] { cache->clear_memory(); }, [&] { return recommend(model, original); },
            [&](const CacheReport& value) {
                require(value.source == "storage" && value.reused_history_tokens == history_tokens,
                        "host reload must reuse the complete stored history");
            });
        output["cold_miss_with_publication"] = repeated(
            original, [&] { cache->invalidate(key); }, [&] { return recommend(model, original); },
            [](const CacheReport& value) {
                require(value.source == "miss" && value.reused_history_tokens == 0 &&
                            value.published,
                        "cold miss must compute and publish the complete history");
            });
        auto extended_history = initial;
        append(extended_history, update(candidates.front(), 1));
        const auto extended = request(extended_history);
        output["append_history_with_publication"] = repeated(
            extended, [&] { seed(); }, [&] { return recommend(model, extended); },
            [&](const CacheReport& value) {
                require(value.source == "memory" && value.reused_history_tokens == history_tokens &&
                            value.history_tokens > history_tokens && value.published,
                        "append must reuse the original history and publish the extension");
            });
        return output;
    }

    struct LoopResult {
        Result final;
        std::vector<std::int64_t> selected;
        std::vector<double> step_ms;
        std::vector<CacheReport> reports;
        double setup_ms{0.0}, loop_ms{0.0};
    };

    LoopResult loop(bool local) {
        LoopResult output;
        output.selected.reserve(steps);
        output.step_ms.reserve(steps);
        output.reports.reserve(steps);
        auto history = initial;
        std::unique_ptr<trtmc::IRecommendationSession> session;
        Result result;
        output.setup_ms = timed([&] {
            if (local) {
                session = sessions.create_recommendation_session(initial, budget);
                result = session->score(candidates, candidate_timestamps);
            } else {
                result = recommend(baseline, request(history));
            }
        });
        output.loop_ms = timed([&] {
            for (std::size_t step = 1; step <= steps; ++step) {
                const auto selected = greedy_item(result);
                const auto next = update(selected, step);
                output.selected.push_back(selected);
                output.step_ms.push_back(timed([&] {
                    if (local) {
                        session->append(next);
                        result = session->score(candidates, candidate_timestamps);
                    } else {
                        append(history, next);
                        result = recommend(baseline, request(history));
                    }
                }));
                output.reports.push_back(result.cache);
            }
        });
        output.final = std::move(result);
        return output;
    }

    Json loop_cases() {
        seed();
        const auto expected = loop(false);
        Json output;
        for (const bool local : {false, true}) {
            std::vector<double> setup, totals, setup_plus_loop;
            std::vector<std::vector<double>> per_step(steps);
            Json first, last;
            const auto before = cache->stats();
            for (std::size_t iteration = 0; iteration < warmups + iterations; ++iteration) {
                const auto result = loop(local);
                require(result.selected == expected.selected,
                        "greedy choices differ from full recompute");
                compare(result.final, expected.final, rtol, atol);
                for (std::size_t step = 0; step < steps; ++step) {
                    const auto& value = result.reports[step];
                    require(!value.published, "decode session must not publish persistent history");
                    if (local) {
                        const auto appended_tokens =
                            1 + static_cast<std::int32_t>(settings.contains("append_action_id"));
                        require(value.source == "request_local" &&
                                    value.reused_history_tokens ==
                                        value.history_tokens - appended_tokens,
                                "session scoring must reuse all history before the append");
                    } else
                        require((value.source.empty() || value.source == "disabled") &&
                                    value.reused_history_tokens == 0,
                                "baseline loop must fully recompute");
                }
                if (iteration >= warmups) {
                    setup.push_back(result.setup_ms);
                    totals.push_back(result.loop_ms);
                    setup_plus_loop.push_back(result.setup_ms + result.loop_ms);
                    for (std::size_t step = 0; step < steps; ++step)
                        per_step[step].push_back(result.step_ms[step]);
                    Json reports = Json::array();
                    for (const auto& value : result.reports)
                        reports.push_back(cache_json(value));
                    last = {{"selected_item_ids", result.selected}, {"cache_by_step", reports}};
                    if (iteration == warmups)
                        first = last;
                }
            }
            const auto after = cache->stats();
            require(after.publications == before.publications,
                    "decode loops must not replace persistent history");
            Json by_step = Json::array();
            for (std::size_t step = 0; step < steps; ++step) {
                auto entry = distribution(per_step[step]);
                entry["step"] = step + 1;
                by_step.push_back(std::move(entry));
            }
            output[local ? "request_local_session" : "full_recompute"] = {
                {"setup_and_initial_score", distribution(setup)},
                {"twenty_step_loop", distribution(totals)},
                {"setup_plus_twenty_step_loop", distribution(setup_plus_loop)},
                {"scoring_calls_per_sample", steps + 1},
                {"append_and_score_by_step", by_step},
                {"first_trace", first},
                {"last_trace", last},
                {"all_final_outputs_and_greedy_choices_match_full_recompute", true},
                {"persistent_publications_unchanged", true}};
        }
        return output;
    }
};

void run(const Json& settings, const std::map<std::string, std::string>& args, Json& report) {
    const auto warmups = settings.value("warmups", std::size_t{10});
    const auto iterations = settings.value("iterations", std::size_t{100});
    require(iterations > 0, "iterations must be positive");
    const auto steps = settings.at("num_steps").get<std::size_t>();
    require(steps == 20, "benchmark requires twenty append/score steps");
    const auto budget = settings.at("cache_max_bytes").get<std::size_t>();
    const auto storage_budget = settings.value("storage_max_bytes", budget);
    require(budget > 0 && storage_budget > 0, "both cache tiers need a positive budget");
    auto cache = std::make_shared<trtmc::HistoryCache>(trtmc::HistoryCacheOptions{
        budget, 8, std::make_shared<trtmc::InMemoryHistoryCacheStorage>(storage_budget, 8), true});
    auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"));
    const auto baseline_bundle = settings.value("baseline_bundle", args.at("--bundle"));
    auto baseline_task = trtmc::load_task(baseline_bundle, args.at("--runtime-root"));
    auto* model = dynamic_cast<trtmc::IRecommendation*>(task.get());
    auto* baseline = dynamic_cast<trtmc::IRecommendation*>(baseline_task.get());
    auto* consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(task.get());
    auto* baseline_consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(baseline_task.get());
    auto* factory = dynamic_cast<trtmc::IRecommendationSessionFactory*>(task.get());
    require(model && baseline && consumer && factory,
            "benchmark requires a cache-enabled HSTU bundle and native session API");
    consumer->set_history_cache(cache);
    if (baseline_consumer)
        baseline_consumer->set_history_cache(nullptr);
    auto initial = read_sequence(settings.at("initial_history"));
    require(!initial.cache.subject_id.empty() && !initial.cache.read_only,
            "benchmark initial history needs a writable cache identity");
    require(initial.candidate_item_ids.empty(), "initial history must exclude candidates");
    const auto candidates = read_ids(settings, "candidate_item_ids", true);
    require(!candidates.empty(), "benchmark needs candidates");
    const auto artifact = consumer->history_cache_artifact_id();
    Benchmark benchmark{settings,
                        *model,
                        *baseline,
                        *factory,
                        cache,
                        {artifact, initial.cache.feature_version, initial.cache.subject_id,
                         initial.cache.history_epoch},
                        initial,
                        candidates,
                        read_ids(settings, "candidate_timestamps"),
                        budget,
                        warmups,
                        iterations,
                        steps,
                        settings.at("rtol").get<double>(),
                        settings.at("atol").get<double>()};
    require(std::isfinite(benchmark.rtol) && benchmark.rtol >= 0.0 &&
                std::isfinite(benchmark.atol) && benchmark.atol >= 0.0,
            "finite nonnegative parity tolerances are required");
    report["environment"] = environment();
    report["artifact_id"] = artifact;
    report["baseline_bundle"] = baseline_bundle;
    report["baseline"] =
        baseline_consumer
            ? "Full recomputation through a cache-enabled bundle with its cache service detached."
            : "Full recomputation through the separately built ordinary no-cache HSTU graph.";
    report["warmups"] = warmups;
    report["iterations"] = iterations;
    report["performance_sample_count_met"] = iterations >= 30;
    report["history_items"] = initial.history_item_ids.size();
    report["candidates"] = candidates.size();
    report["cache_max_bytes"] = budget;
    report["storage_max_bytes"] = storage_budget;
    report["storage_backend"] = "native InMemoryHistoryCacheStorage with write-through publication";
    report["cache_cases"] = benchmark.cache_cases();
    report["autoregressive_cases"] = benchmark.loop_cases();
    report["cache_stats"] = stats_json(cache->stats());
    const auto stats = cache->stats();
    require(stats.load_failures == 0 && stats.store_failures == 0 && stats.erase_failures == 0 &&
                stats.rejected_publications == 0 && stats.stale_publications == 0,
            "benchmark encountered a cache storage or admission failure");
}

void write_report(const std::map<std::string, std::string>& args, const Json& report) {
    std::ofstream output(args.at("--output-json"));
    if (!output || !(output << report.dump(2) << '\n'))
        throw std::runtime_error("cannot write native HSTU benchmark report");
}

} // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    Json report;
    try {
        args = options(argc, argv);
        std::ifstream input(args.at("--input-json"));
        Json settings;
        if (!input || !(input >> settings))
            throw std::runtime_error("cannot read HSTU benchmark input JSON");
        report = {{"input", settings},
                  {"bundle", args.at("--bundle")},
                  {"runtime_root", args.at("--runtime-root")},
                  {"measurement",
                   "Synchronous native API wall time with CUDA synchronization. "
                   "Includes host preparation, transfers, GPU execution, output copies, and "
                   "cache publication when applicable; excludes load, JSON I/O, numerical "
                   "checks, cache invalidation, and seed preparation."},
                  {"loop_measurement",
                   "Twenty serial greedy append/score steps, including token "
                   "selection and measurement bookkeeping. Initial session creation and initial "
                   "score are timed separately. Each step has CUDA synchronization before and "
                   "after its timer, included in the enclosing loop measurement. Total workload "
                   "samples sum paired setup and loop times for 21 scoring calls and session "
                   "setup, without a server."},
                  {"scope", "Seeded fixture microbenchmark; no serving/network latency, trained "
                            "accuracy, production throughput, or other-GPU performance claim."},
                  {"baseline",
                   "Full recomputation through the same cache-enabled bundle with "
                   "the cache service detached; not a separately optimized no-cache graph."}};
        run(settings, args, report);
        write_report(args, report);
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        if (args.count("--output-json"))
            write_report(args, report);
        std::cerr << "hstu-cache-benchmark: " << error.what() << '\n';
        return 1;
    }
}

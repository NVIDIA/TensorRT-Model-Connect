/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reuse the public JSON boundary, then exercise the native session API directly.
#define main hstu_single_request_main
#include "families/hstu/runtime/runner.cpp"
#undef main

#include "trtmc/history_cache.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>

namespace {

using Sequence = trtmc::RecommendationSequence;
using Result = trtmc::RecommendationSequenceResult;
using Session = trtmc::IRecommendationSession;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

Json stats_json(const trtmc::HistoryCacheStats& stats) {
    return {{"hits", stats.hits},
            {"misses", stats.misses},
            {"publications", stats.publications},
            {"storage_hits", stats.storage_hits},
            {"stale_publications", stats.stale_publications},
            {"rejected_publications", stats.rejected_publications},
            {"evictions", stats.evictions},
            {"live_bytes", stats.live_bytes}};
}

Json sequence_json(const Sequence& sequence) {
    Json features = Json::array();
    for (const auto& feature : sequence.contextual_features)
        features.push_back({{"name", feature.name}, {"ids", feature.ids}});
    return {{"history_item_ids", sequence.history_item_ids},
            {"history_action_ids", sequence.history_action_ids},
            {"contextual_features", features},
            {"candidate_item_ids", sequence.candidate_item_ids},
            {"token_timestamps", sequence.token_timestamps},
            {"cache",
             {{"subject_id", sequence.cache.subject_id},
              {"feature_version", sequence.cache.feature_version},
              {"history_epoch", sequence.cache.history_epoch},
              {"read_only", sequence.cache.read_only}}}};
}

void compare_vector(const std::vector<float>& actual, const std::vector<float>& expected,
                    double rtol, double atol, const std::string& name) {
    require(actual.size() == expected.size(), name + " shape mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double left = actual[index], right = expected[index];
        require(std::isfinite(left) && std::isfinite(right), name + " nonfinite result");
        if (std::abs(left - right) > atol + rtol * std::abs(right))
            throw std::runtime_error(name + " differs at index " + std::to_string(index));
    }
}

void compare_result(const Result& actual, const Result& expected, double rtol, double atol,
                    const std::string& name) {
    require(actual.candidate_item_ids == expected.candidate_item_ids, name + " candidate order");
    require(actual.num_candidates == expected.num_candidates, name + " candidate count");
    require(actual.output_dim == expected.output_dim, name + " output width");
    require(actual.embedding_dim == expected.embedding_dim, name + " embedding width");
    require(actual.sequence_length == expected.sequence_length, name + " sequence length");
    compare_vector(actual.logits, expected.logits, rtol, atol, name + "/logits");
    compare_vector(actual.scores, expected.scores, rtol, atol, name + "/scores");
    compare_vector(actual.embeddings, expected.embeddings, rtol, atol, name + "/embeddings");
    compare_vector(actual.sequence_embeddings, expected.sequence_embeddings, rtol, atol,
                   name + "/sequence_embeddings");
}

struct Trace {
    Json& report;
    trtmc::IRecommendation& baseline;
    std::shared_ptr<trtmc::HistoryCache> persistent;
    std::vector<std::int64_t> candidates;
    std::vector<std::int64_t> candidate_timestamps;
    double rtol;
    double atol;

    Result record(const std::string& name, const Sequence& request,
                  const std::function<Result()>& operation) {
        Json row = {{"name", name},
                    {"request", {{"sequences", {sequence_json(request)}}}},
                    {"persistent_before", stats_json(persistent->stats())}};
        const auto started = Clock::now();
        const auto actual = operation();
        row["session_ms"] =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        row["actual"] = output_json({{actual}});
        const auto baseline_started = Clock::now();
        const auto expected = baseline.recommend({{request}}).sequences.front();
        row["full_recompute_ms"] =
            std::chrono::duration<double, std::milli>(Clock::now() - baseline_started).count();
        row["baseline"] = output_json({{expected}});
        row["persistent_after"] = stats_json(persistent->stats());
        report["steps"].push_back(std::move(row));
        require(expected.cache.source == "disabled", "comparison task must disable its cache");
        compare_result(actual, expected, rtol, atol, name);
        return actual;
    }

    Result score(const std::string& name, Session& session, const Sequence& history) {
        auto request = history;
        request.candidate_item_ids = candidates;
        request.token_timestamps.insert(request.token_timestamps.end(),
                                        candidate_timestamps.begin(), candidate_timestamps.end());
        return record(name, request,
                      [&] { return session.score(candidates, candidate_timestamps); });
    }
};

std::int64_t greedy_item(const Result& result) {
    require(result.num_candidates > 0, "greedy session requires candidates");
    std::size_t best = 0;
    const auto& scores = result.scores.empty() ? result.logits : result.scores;
    const auto width = result.scores.empty() ? result.output_dim : 1;
    require(width > 0 && scores.size() == result.candidate_item_ids.size() * width,
            "greedy session score dimensions");
    for (std::size_t index = 1; index < result.candidate_item_ids.size(); ++index) {
        if (scores[index * width] > scores[best * width])
            best = index;
    }
    return result.candidate_item_ids[best];
}

trtmc::RecommendationHistoryAppend update_for(const Json& settings, std::int64_t item,
                                              std::size_t step) {
    trtmc::RecommendationHistoryAppend update;
    update.item_ids = {item};
    if (settings.contains("append_action_id"))
        update.action_ids = {settings.at("append_action_id").get<std::int64_t>()};
    if (settings.contains("append_timestamp")) {
        const auto timestamp = settings.at("append_timestamp").get<std::int64_t>() +
                               2 * static_cast<std::int64_t>(step);
        update.token_timestamps = {timestamp};
        if (!update.action_ids.empty())
            update.token_timestamps.push_back(timestamp + 1);
    }
    return update;
}

void append(Session& session, Sequence& history, const trtmc::RecommendationHistoryAppend& update) {
    session.append(update);
    history.history_item_ids.insert(history.history_item_ids.end(), update.item_ids.begin(),
                                    update.item_ids.end());
    history.history_action_ids.insert(history.history_action_ids.end(), update.action_ids.begin(),
                                      update.action_ids.end());
    history.token_timestamps.insert(history.token_timestamps.end(), update.token_timestamps.begin(),
                                    update.token_timestamps.end());
}

void exercise_forks(Trace& trace, Session& parent, const Sequence& history,
                    const Result& parent_result, const Json& settings) {
    const auto items = read_ids(settings, "fork_item_ids", true);
    require(items.size() == 2 && items[0] != items[1], "fork controls need different items");
    auto left = parent.branch();
    auto right = parent.branch();
    auto left_history = history, right_history = history;
    append(*left, left_history, update_for(settings, items[0], 21));
    append(*right, right_history, update_for(settings, items[1], 21));
    trace.score("fork-left", *left, left_history);
    trace.score("fork-right", *right, right_history);
    const auto unchanged = trace.score("parent-after-forks", parent, history);
    compare_result(unchanged, parent_result, trace.rtol, trace.atol, "forks preserve parent");

    auto pending_parent = parent.branch();
    auto pending_history = history;
    append(*pending_parent, pending_history, update_for(settings, items[0], 22));
    auto pending_child = pending_parent->branch();
    trace.score("fork-after-pending-append", *pending_child, pending_history);
    trace.score("pending-parent", *pending_parent, pending_history);
}

void reject_invalid_append(Trace& trace, Session& session, const Sequence& history,
                           const Result& before, const Json& settings) {
    const auto invalid = update_for(settings, std::numeric_limits<std::int64_t>::max(), 22);
    std::string error;
    try {
        session.append(invalid);
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    trace.report["invalid_append"] = {
        {"item_ids", invalid.item_ids}, {"action_ids", invalid.action_ids}, {"error", error}};
    require(!error.empty(), "invalid append must be rejected");
    const auto unchanged = trace.score("after-invalid-append", session, history);
    compare_result(unchanged, before, trace.rtol, trace.atol, "invalid append preserves parent");
}

void reject_invalid_score(Trace& trace, Session& session, const Sequence& history,
                          const Result& before) {
    std::string error;
    try {
        // Keep timestamp admission valid so a timestamp-enabled model reaches
        // the bad-ID lookup inside run_local, before any page lease exists.
        const auto timestamps = trace.candidate_timestamps.empty()
                                    ? std::vector<std::int64_t>{}
                                    : std::vector<std::int64_t>{trace.candidate_timestamps.front()};
        (void)session.score({std::numeric_limits<std::int64_t>::max()}, timestamps);
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    trace.report["invalid_score_error"] = error;
    require(!error.empty(), "invalid candidate must be rejected");
    const auto recovered = trace.score("after-invalid-score", session, history);
    compare_result(recovered, before, trace.rtol, trace.atol, "failed score preserves history");
}

void run_trace(const Json& settings, const std::map<std::string, std::string>& args, Json& report) {
    const bool graphs = settings.value("cuda_graphs", false);
    auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"), 0, {}, graphs);
    auto baseline_task =
        trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"), 0, {}, graphs);
    auto* model = dynamic_cast<trtmc::IRecommendation*>(task.get());
    auto* baseline = dynamic_cast<trtmc::IRecommendation*>(baseline_task.get());
    auto* consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(task.get());
    auto* baseline_consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(baseline_task.get());
    auto* factory = dynamic_cast<trtmc::IRecommendationSessionFactory*>(task.get());
    require(model && baseline && consumer && baseline_consumer && factory,
            "session trace requires the native recommendation session capability");
    const auto budget = settings.at("cache_max_bytes").get<std::size_t>();
    auto persistent = std::make_shared<trtmc::HistoryCache>(
        trtmc::HistoryCacheOptions{budget, 8, nullptr, false});
    consumer->set_history_cache(persistent);
    baseline_consumer->set_history_cache(nullptr);
    const auto initial = read_sequence(settings.at("initial_history"));
    require(!initial.cache.subject_id.empty(), "test must use a persistent user identity");
    require(initial.candidate_item_ids.empty(), "initial history must exclude candidates");
    const auto artifact = consumer->history_cache_artifact_id();
    report["artifact_id"] = artifact;
    report["steps"] = Json::array();
    Trace trace{report,
                *baseline,
                persistent,
                read_ids(settings, "candidate_item_ids", true),
                read_ids(settings, "candidate_timestamps"),
                settings.at("rtol").get<double>(),
                settings.at("atol").get<double>()};
    trace.record("persistent-seed", initial,
                 [&] { return model->recommend({{initial}}).sequences.front(); });
    const trtmc::HistoryCacheKey persistent_key{artifact, initial.cache.feature_version,
                                                initial.cache.subject_id,
                                                initial.cache.history_epoch};
    const auto original = persistent->lookup(persistent_key);
    require(original.value != nullptr, "persistent seed must publish real history KV");
    std::string budget_error;
    try {
        (void)factory->create_recommendation_session(initial, 0);
    } catch (const std::exception& error) {
        budget_error = error.what();
    }
    report["budget_rejection_error"] = budget_error;
    require(!budget_error.empty(), "zero session budget must be rejected");
    auto session = factory->create_recommendation_session(initial, budget);
    report["persistent_after_session_creation"] = stats_json(persistent->stats());
    auto history = initial;
    auto result = trace.score("initial", *session, history);
    const auto steps = settings.at("num_steps").get<std::size_t>();
    require(steps == 20, "session qualification requires twenty append/score steps");
    for (std::size_t step = 1; step <= steps; ++step) {
        const auto selected = greedy_item(result);
        append(*session, history, update_for(settings, selected, step));
        std::ostringstream name;
        name << "step-" << std::setfill('0') << std::setw(2) << step;
        result = trace.score(name.str(), *session, history);
        report["steps"].back()["selected_item_id"] = selected;
    }
    exercise_forks(trace, *session, history, result, settings);
    reject_invalid_append(trace, *session, history, result, settings);
    reject_invalid_score(trace, *session, history, result);
    const auto retained = persistent->lookup(persistent_key);
    require(retained.generation == original.generation && retained.value == original.value,
            "session updates must not replace persistent user history");
    report["persistent_history_unchanged"] = true;
    report["persistent_before_final_probe"] = stats_json(persistent->stats());
    auto original_request = initial;
    original_request.candidate_item_ids = trace.candidates;
    original_request.token_timestamps.insert(original_request.token_timestamps.end(),
                                             trace.candidate_timestamps.begin(),
                                             trace.candidate_timestamps.end());
    trace.record("persistent-after", original_request,
                 [&] { return model->recommend({{original_request}}).sequences.front(); });
    task.reset();
    trace.score("after-task-destruction", *session, history);
    report["session_outlived_task"] = true;
    report["persistent_final"] = stats_json(persistent->stats());
}

void write_report(const std::map<std::string, std::string>& args, const Json& report) {
    std::ofstream output(args.at("--output-json"));
    if (!output || !(output << report.dump(2) << '\n'))
        throw std::runtime_error("cannot write native session audit JSON");
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
            throw std::runtime_error("cannot read native session request trace");
        report["input"] = settings;
        run_trace(settings, args, report);
        write_report(args, report);
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        if (args.count("--output-json"))
            write_report(args, report);
        std::cerr << "hstu-session-runner: " << error.what() << '\n';
        return 1;
    }
}

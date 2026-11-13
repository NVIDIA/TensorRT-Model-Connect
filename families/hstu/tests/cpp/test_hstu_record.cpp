/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// The ranking record's parser against the bundle's static geometry: CPU only.

#include "families/hstu/runtime/record.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using trtmc::hstu::Json;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void expect_refused(const std::function<void()>& call, const std::string& fragment) {
    try {
        call();
        std::cerr << "FAIL: accepted a record that should name '" << fragment << "'\n";
        ++failures;
    } catch (const std::invalid_argument& error) {
        expect(std::string(error.what()).find(fragment) != std::string::npos,
               "refusal '" + std::string(error.what()) + "' names '" + fragment + "'");
    }
}

Json runtime() {
    return Json{{"precision", "bf16"},
                {"history_length", 2},
                {"num_candidates", 1},
                {"tokens", 2 + 2 * 2 + 1},
                {"num_tasks", 8},
                {"contextual_features",
                 Json::array({Json{{"name", "user_id"}, {"vocab_size", 10}},
                              Json{{"name", "register_days_range"}, {"vocab_size", 8}}})},
                {"item_feature", Json{{"name", "video_id"}, {"vocab_size", 100}}},
                {"action_feature", Json{{"name", "action_weights"}, {"vocab_size", 5}}}};
}

Json record() {
    return Json{{"user", Json{{"user_id", 3}, {"register_days_range", 7}}},
                {"history",
                 Json{{"video_id", Json::array({1, 99})}, {"action_weights", Json::array({0, 4})}}},
                {"candidates", Json{{"video_id", Json::array({42})}}}};
}

} // namespace

int main() {
    const auto geometry = trtmc::hstu::parse_geometry(runtime());
    expect(geometry.tokens == 7 && geometry.contextual.size() == 2, "geometry parsed");

    const auto request = trtmc::hstu::parse_request(record(), geometry);
    expect(request.contextual_ids == std::vector<std::int32_t>{3, 7}, "contextual ids in order");
    expect(request.history_item_ids == std::vector<std::int32_t>{1, 99}, "history items");
    expect(request.history_action_ids == std::vector<std::int32_t>{0, 4}, "history actions");
    expect(request.candidate_item_ids == std::vector<std::int32_t>{42}, "candidates");
    expect(request.model == "hstu", "default model name");

    auto bad_tokens = runtime();
    bad_tokens["tokens"] = 8;
    expect_refused([&] { trtmc::hstu::parse_geometry(bad_tokens); }, "tokens");
    auto bad_precision = runtime();
    bad_precision["precision"] = "fp16";
    expect_refused([&] { trtmc::hstu::parse_geometry(bad_precision); }, "precision");

    auto short_history = record();
    short_history["history"]["video_id"] = Json::array({1});
    expect_refused([&] { trtmc::hstu::parse_request(short_history, geometry); },
                   "static geometry takes exactly 2");
    auto outside = record();
    outside["candidates"]["video_id"] = Json::array({100});
    expect_refused([&] { trtmc::hstu::parse_request(outside, geometry); }, "outside its table");
    auto negative = record();
    negative["history"]["action_weights"] = Json::array({0, -1});
    expect_refused([&] { trtmc::hstu::parse_request(negative, geometry); }, "outside its table");
    auto fractional = record();
    fractional["user"]["user_id"] = 1.5;
    expect_refused([&] { trtmc::hstu::parse_request(fractional, geometry); }, "integers");
    auto unknown = record();
    unknown["extra"] = 1;
    expect_refused([&] { trtmc::hstu::parse_request(unknown, geometry); }, "unknown field");
    auto missing_user = record();
    missing_user["user"].erase("register_days_range");
    missing_user["user"]["other"] = 1;
    expect_refused([&] { trtmc::hstu::parse_request(missing_user, geometry); },
                   "lacks register_days_range");

    auto repeated = record();
    repeated["candidates"]["video_id"] = Json::array({42});
    repeated["history"]["video_id"] = Json::array({42, 42});
    expect(trtmc::hstu::parse_request(repeated, geometry).history_item_ids.size() == 2,
           "history may repeat an item");
    auto two_candidates_runtime = runtime();
    two_candidates_runtime["num_candidates"] = 2;
    two_candidates_runtime["tokens"] = 2 + 2 * 2 + 2;
    const auto two = trtmc::hstu::parse_geometry(two_candidates_runtime);
    auto duplicate = record();
    duplicate["candidates"]["video_id"] = Json::array({42, 42});
    expect_refused([&] { trtmc::hstu::parse_request(duplicate, two); },
                   "candidates repeat video_id 42");
    auto extra_history = record();
    extra_history["history"]["timestamps"] = Json::array({1, 2});
    expect_refused([&] { trtmc::hstu::parse_request(extra_history, geometry); },
                   "history must hold its item and action arrays");
    auto renamed_history = record();
    renamed_history["history"].erase("action_weights");
    renamed_history["history"]["actions"] = Json::array({0, 1});
    expect_refused([&] { trtmc::hstu::parse_request(renamed_history, geometry); },
                   "must hold a action_weights array");
    auto extra_candidates = record();
    extra_candidates["candidates"]["action_weights"] = Json::array({1});
    expect_refused([&] { trtmc::hstu::parse_request(extra_candidates, geometry); },
                   "candidates must hold one item array");
    auto numeric_model = record();
    numeric_model["model"] = 3;
    expect_refused([&] { trtmc::hstu::parse_request(numeric_model, geometry); },
                   "model must be a string");
    auto user_list = record();
    user_list["user"] = Json::array({3, 7});
    expect_refused([&] { trtmc::hstu::parse_request(user_list, geometry); },
                   "user must hold exactly the contextual features");
    expect_refused([&] { trtmc::hstu::parse_request(Json::array(), geometry); },
                   "must be a JSON object");
    auto wide_vocab = runtime();
    wide_vocab["item_feature"]["vocab_size"] = 3000000000LL;
    expect_refused([&] { trtmc::hstu::parse_geometry(wide_vocab); }, "int32 vocab");

    if (failures != 0)
        return EXIT_FAILURE;
    std::cout << "test_hstu_record: pass\n";
    return EXIT_SUCCESS;
}

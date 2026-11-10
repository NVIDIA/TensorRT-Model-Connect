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

trtmc::hstu::Request parse(const Json& document, const trtmc::hstu::Geometry& geometry) {
    return trtmc::hstu::parse_request(document.dump(), geometry);
}

trtmc::hstu::Request parse(const std::string& text, const trtmc::hstu::Geometry& geometry) {
    return trtmc::hstu::parse_request(text, geometry);
}

} // namespace

int main() {
    const auto geometry = trtmc::hstu::parse_geometry(runtime());
    expect(geometry.tokens == 7 && geometry.contextual.size() == 2, "geometry parsed");

    const auto request = parse(record(), geometry);
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
    expect_refused([&] { parse(short_history, geometry); }, "static geometry takes exactly 2");
    auto outside = record();
    outside["candidates"]["video_id"] = Json::array({100});
    expect_refused([&] { parse(outside, geometry); }, "outside its table");
    auto negative = record();
    negative["history"]["action_weights"] = Json::array({0, -1});
    expect_refused([&] { parse(negative, geometry); }, "outside its table");
    auto fractional = record();
    fractional["user"]["user_id"] = 1.5;
    expect_refused([&] { parse(fractional, geometry); }, "integers");
    auto unknown = record();
    unknown["extra"] = 1;
    expect_refused([&] { parse(unknown, geometry); }, "unknown field");
    auto missing_user = record();
    missing_user["user"].erase("register_days_range");
    missing_user["user"]["other"] = 1;
    expect_refused([&] { parse(missing_user, geometry); }, "lacks register_days_range");

    auto repeated = record();
    repeated["candidates"]["video_id"] = Json::array({42});
    repeated["history"]["video_id"] = Json::array({42, 42});
    expect(parse(repeated, geometry).history_item_ids.size() == 2, "history may repeat an item");
    auto two_candidates_runtime = runtime();
    two_candidates_runtime["num_candidates"] = 2;
    two_candidates_runtime["tokens"] = 2 + 2 * 2 + 2;
    const auto two = trtmc::hstu::parse_geometry(two_candidates_runtime);
    auto duplicate = record();
    duplicate["candidates"]["video_id"] = Json::array({42, 42});
    expect_refused([&] { parse(duplicate, two); }, "candidates repeat video_id 42");
    auto extra_history = record();
    extra_history["history"]["timestamps"] = Json::array({1, 2});
    expect_refused([&] { parse(extra_history, geometry); },
                   "history must hold its item and action arrays");
    auto renamed_history = record();
    renamed_history["history"].erase("action_weights");
    renamed_history["history"]["actions"] = Json::array({0, 1});
    expect_refused([&] { parse(renamed_history, geometry); }, "must hold a action_weights array");
    auto extra_candidates = record();
    extra_candidates["candidates"]["action_weights"] = Json::array({1});
    expect_refused([&] { parse(extra_candidates, geometry); },
                   "candidates must hold one item array");
    auto numeric_model = record();
    numeric_model["model"] = 3;
    expect_refused([&] { parse(numeric_model, geometry); }, "model must be a string");
    auto user_list = record();
    user_list["user"] = Json::array({3, 7});
    expect_refused([&] { parse(user_list, geometry); },
                   "user must hold exactly the contextual features");
    expect_refused([&] { parse(Json::array(), geometry); }, "must be a JSON object");
    auto wide_vocab = runtime();
    wide_vocab["item_feature"]["vocab_size"] = 3000000000LL;
    expect_refused([&] { trtmc::hstu::parse_geometry(wide_vocab); }, "int32 vocab");

    // What a document tree hid and the one-pass reader must still refuse.
    const std::string valid = record().dump();
    const auto with = [&](const std::string& head, const std::string& field) {
        return head + field + valid.substr(1);
    };
    expect_refused([&] { parse(with("{", "\"user\":{},"), geometry); }, "repeats the field user");
    expect_refused([&] { parse(std::string(R"({"history":{},"candidates":{}})"), geometry); },
                   "lacks user");
    auto no_history = record();
    no_history.erase("history");
    expect_refused([&] { parse(no_history, geometry); }, "lacks history");
    auto no_candidates = record();
    no_candidates.erase("candidates");
    expect_refused([&] { parse(no_candidates, geometry); }, "lacks candidates");
    auto repeated_member = std::string(
        R"({"user":{"user_id":3,"user_id":4},"history":{"video_id":[1,2],"action_weights":[0,1]},)"
        R"("candidates":{"video_id":[5]}})");
    expect_refused([&] { parse(repeated_member, geometry); }, "repeats the field user_id");
    auto nested_user = record();
    nested_user["user"]["user_id"] = Json{{"id", 3}};
    expect_refused([&] { parse(nested_user, geometry); }, "user user_id ids must be integers");
    auto nested_element = record();
    nested_element["history"]["video_id"] = Json::array({1, Json::array({2})});
    expect_refused([&] { parse(nested_element, geometry); },
                   "history video_id ids must be integers");
    auto out_then_float = record();
    out_then_float["history"]["video_id"] = Json::array({100, 1.5});
    expect_refused([&] { parse(out_then_float, geometry); }, "id 100 is outside its table");
    auto float_then_out = record();
    float_then_out["history"]["video_id"] = Json::array({1.5, 100});
    expect_refused([&] { parse(float_then_out, geometry); }, "ids must be integers");
    auto string_ids = record();
    string_ids["candidates"]["video_id"] = "42";
    expect_refused([&] { parse(string_ids, geometry); }, "must hold a video_id array");
    auto huge = record();
    huge["candidates"]["video_id"] = Json::array({18446744073709551615ULL});
    expect_refused([&] { parse(huge, geometry); }, "is outside its table");
    expect_refused([&] { parse(std::string(R"({"user":)"), geometry); }, "is not JSON");
    expect_refused([&] { parse(std::string("[1, 2]"), geometry); }, "must be a JSON object");
    expect_refused([&] { parse(with("{", "\"model\":[\"a\"],"), geometry); },
                   "model must be a string");
    expect_refused([&] { parse(with("{", "\"model\":{\"a\":[1]},"), geometry); },
                   "model must be a string");
    expect_refused([&] { parse(with("{", "\"extra\":{\"a\":[1,{\"b\":2}]},"), geometry); },
                   "unknown field: extra");
    expect_refused([&] { parse(valid + "x", geometry); }, "is not JSON");
    expect_refused([&] { parse(with("{", "\"\":1,"), geometry); }, "unknown field: ");
    expect_refused(
        [&] {
            parse(std::string(R"({"user":{"":3,"":4},"history":{"video_id":[1,2],)"
                              R"("action_weights":[0,1]},"candidates":{"video_id":[5]}})"),
                  geometry);
        },
        "repeats the field ");
    for (const auto& element : {Json("1"), Json(true), Json(nullptr), Json(1.0)}) {
        auto odd = record();
        odd["history"]["video_id"] = Json::array({element, 2});
        expect_refused([&] { parse(odd, geometry); }, "history video_id ids must be integers");
    }
    auto empty_ids = record();
    empty_ids["candidates"]["video_id"] = Json::array();
    expect_refused([&] { parse(empty_ids, geometry); }, "holds 0 ids");
    const auto named = parse(with("{", "\"model\":\"hstu-x\","), geometry);
    expect(named.model == "hstu-x" && named.candidate_item_ids == std::vector<std::int32_t>{42},
           "model and ids read in one pass");

    if (failures != 0)
        return EXIT_FAILURE;
    std::cout << "test_hstu_record: pass\n";
    return EXIT_SUCCESS;
}

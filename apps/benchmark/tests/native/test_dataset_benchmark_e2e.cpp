/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
std::string quote(const std::string& value) {
    std::string output = "'";
    for (char c : value)
        output += c == '\'' ? "'\\''" : std::string(1, c);
    return output + "'";
}
void bundle(const std::filesystem::path& path, const std::string& task,
            const std::string& family = "dataset_fixture", bool multiple = false,
            bool text = true) {
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", task},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream output(path, std::ios::binary);
    output.write("BUNDLE\x01\x00", 8);
    const std::uint64_t size = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((size >> shift) & 255));
    output << header << (!text ? "NONE" : multiple ? "BOTH" : "PLAN");
    if (!output)
        throw std::runtime_error("cannot write bundle");
}
std::vector<Json> results(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<Json> rows;
    for (std::string line; std::getline(input, line);)
        rows.push_back(Json::parse(line));
    return rows;
}
Json config(const Json& row) {
    const auto text = row.at("text").get<std::string>();
    return Json::parse(text.substr(text.rfind('\n') + 1));
}
bool error_contains(const std::filesystem::path& path, const std::string& expected) {
    std::ifstream input(path);
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return text.find(expected) != std::string::npos;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    std::string pattern =
        (std::filesystem::temp_directory_path() / "trtmc-dataset-test-XXXXXX").string();
    if (!mkdtemp(pattern.data()))
        return 2;
    const std::filesystem::path root(pattern), model = root / "model.bundle",
                                               dataset = root / "samples.jsonl",
                                               output = root / "result.jsonl";
    try {
        bundle(model, "text_continuation");
        {
            std::ofstream file(dataset);
            file << Json{{"sample_id", "A"},
                         {"prompt", "First question"},
                         {"answer", "42"},
                         {"seed_index", 3}}
                 << '\n';
            file << Json{{"sample_id", 2}, {"prompt", "Second question"}, {"answer", 42}} << '\n';
        }
        auto invoke = [&](std::initializer_list<std::string> extra) {
            std::string command = quote(argv[1]) + " " + quote(model.string()) + " " +
                                  quote(dataset.string()) + " " + quote(output.string()) +
                                  " --runtime-root " + quote(argv[2]);
            for (const auto& arg : extra)
                command += " " + quote(arg);
            command += " 2>" + quote((root / "stderr.txt").string());
            return std::system(command.c_str());
        };
        check(invoke({}) == 0, "SDK default workload succeeds");
        auto rows = results(output);
        check(rows.size() == 2, "one output per dataset sample");
        if (rows.size() == 2) {
            const auto values = config(rows[0]);
            check(values.at("max_new_tokens") == 12000 && values.at("top_k") == 1 &&
                      values.at("top_p") == 1.0 && values.at("temperature") == 1.0 &&
                      values.at("min_p") == 0.0 && values.at("use_chat_template") == false,
                  "existing benchmark defaults override different family defaults");
            check(rows[0].at("submitted_config").at("max_new_tokens") == 12000,
                  "submitted workload is visible in receipt");
            check(rows[0].at("pred_answer") == "42" && rows[0].at("gold_answer") == "42" &&
                      rows[0].at("sample_id") == "A" && rows[1].at("sample_id") == "2",
                  "answer and IDs preserved");
            check(rows[0].at("generated_token_ids") == Json::array({100, 101}) &&
                      rows[0].at("setup_ms") == 1 && rows[0].at("prefill_ms") == 2 &&
                      rows[0].at("decode_ms") == 4 && rows[0].at("tokens_per_sec") == 500,
                  "tokens and timing metrics preserved");
            check(rows[0].at("wall_ms").get<double>() >= 0, "wall time measured");
        }
        check(invoke({"--seed", "10", "--chat-template", "--no-thinking", "--stop-on-answer"}) == 0,
              "existing explicit options reach family");
        rows = results(output);
        check(rows.size() == 2 && config(rows[0]).at("seed") == 13 &&
                  config(rows[1]).at("seed") == 11,
              "seed_index and implicit row index are distinct");
        if (!rows.empty())
            check(config(rows[0]).at("use_chat_template") == true &&
                      config(rows[0]).at("enable_thinking") == false &&
                      config(rows[0]).at("stop_on_boxed_answer") == true,
                  "bool flags preserved");
        check(invoke({"--top-k", "0", "--top-p", "0.7"}) == 0,
              "semantic shorthand preserves zero and float64 precision");
        rows = results(output);
        check(!rows.empty() && config(rows[0]).at("top_k") == 0 &&
                  config(rows[0]).at("top_p") == 0.7,
              "shorthand does not truncate or replace explicit values");
        check(invoke({"--set", "seed=10", "--set", "suffix=", "--set", "ids=[0,2]", "--set",
                      "schedule=[1.0,0.0]", "--set", "labels=[\"a\",\"\"]", "--set",
                      "use_chat_template=false"}) == 0,
              "all seven Config kinds transport");
        rows = results(output);
        if (!rows.empty()) {
            const auto values = config(rows[0]);
            check(values.at("seed") == 13 && values.at("suffix") == "" &&
                      values.at("ids") == Json::array({0, 2}) &&
                      values.at("schedule") == Json::array({1.0, 0.0}) &&
                      values.at("labels") == Json::array({"a", ""}),
                  "list values and empty string survive");
        }
        for (auto args : {std::vector<std::string>{"--set", "unknown=1"},
                          {"--set", "max_new_tokens=1.5"},
                          {"--max-new-tokens", "30000"},
                          {"--set", "seed=9223372036854775807"},
                          {"--set", "suffix=a", "--set", "suffix=b"},
                          {"--top-p", "0.8", "--set", "top_p=0.9"},
                          {"--top-p", "0.7junk"},
                          {"--temperature", "nan"},
                          {"--max-new-tokens", "2.75"},
                          {"--top-p", "0.7", "--top-p", "0.8"}}) {
            std::filesystem::remove(output); // Each negative case starts without stale output.
            std::string command = quote(argv[1]) + " " + quote(model.string()) + " " +
                                  quote(dataset.string()) + " " + quote(output.string()) +
                                  " --runtime-root " + quote(argv[2]);
            for (const auto& arg : args)
                command += " " + quote(arg);
            command += " 2>" + quote((root / "stderr.txt").string());
            check(std::system(command.c_str()) != 0, "invalid explicit config rejected");
            check(results(output).empty(), "failed first sample is not reported as a result");
        }
        bundle(model, "images_text_to_text");
        for (const bool explicit_task : {false, true}) {
            std::filesystem::remove(output);
            check((explicit_task ? invoke({"--task", "text_continuation"}) : invoke({})) == 0,
                  "multimodal primary retains its bound prompt-only dataset Task");
            rows = results(output);
            check(rows.size() == 2, "secondary Task returns every dataset sample");
            if (rows.size() == 2)
                check(rows[0].at("task") == "text_continuation" &&
                          rows[0].at("bundle_task") == "images_text_to_text" &&
                          rows[0].at("generated_token_ids") == Json::array({100, 101}) &&
                          rows[0].at("pred_answer") == "42" &&
                          config(rows[0]).at("max_new_tokens") == 12000,
                      "selected secondary Task preserves full output and workload");
        }
        for (const auto* invalid_task :
             {"unknown_task", "images_text_to_text", "conditional_text_generation", ""}) {
            std::filesystem::remove(output);
            check(invoke({"--task", invalid_task}) != 0,
                  "unknown, incomplete, unbound and empty Task selections fail");
            check(results(output).empty(), "invalid selection produces no sample result");
            const std::string id = invalid_task;
            const auto message = id.empty() ? "--task requires a nonempty Task ID"
                                 : id == "conditional_text_generation"
                                     ? "model does not bind"
                                     : "dataset prompt is not a complete input";
            check(error_contains(root / "stderr.txt", message),
                  "invalid selection fails in the selector, not in a family call");
        }
        check(invoke({"--task", "text_continuation", "--task", "text_continuation"}) != 0,
              "duplicate Task selection fails");
        std::filesystem::remove(output);
        check(invoke({"--task", "text_continuation", "--max-new-tokens", "30000"}) != 0,
              "selected Task failure never retries the legacy implementation");
        check(results(output).empty(), "failed selected call has no legacy result");
        bundle(model, "images_text_to_text", "dataset_fixture", true);
        std::filesystem::remove(output);
        check(invoke({}) != 0 && results(output).empty(),
              "ambiguous prompt-only Tasks require an explicit selection");
        check(error_contains(root / "stderr.txt", "multiple dataset Tasks are available"),
              "ambiguous selection is rejected before any family call");
        check(invoke({"--task", "conditional_text_generation"}) == 0,
              "explicit selection resolves multiple bound prompt Tasks");
        rows = results(output);
        check(rows.size() == 2 && rows[0].at("task") == "conditional_text_generation" &&
                  rows[0].at("bundle_task") == "images_text_to_text" &&
                  rows[0].at("text").get<std::string>().find("conditioned:First question") == 0 &&
                  rows[0].at("generated_token_ids") == Json::array({100, 101}),
              "conditional Task receives the original prompt and preserves all token IDs");
        bundle(model, "text_continuation", "dataset_fixture", true);
        check(invoke({}) == 0, "compatible primary takes precedence over other prompt Tasks");
        rows = results(output);
        check(rows.size() == 2 && rows[0].at("task") == "text_continuation",
              "compatible primary identity is preserved");
        check(invoke({"--task", "conditional_text_generation"}) == 0,
              "explicit selection can override a compatible primary");
        rows = results(output);
        check(rows.size() == 2 && rows[0].at("task") == "conditional_text_generation" &&
                  rows[0].at("bundle_task") == "text_continuation",
              "explicit override does not change physical bundle identity");
        bundle(model, "images_text_to_text", "dataset_fixture", false, false);
        std::filesystem::remove(output);
        check(invoke({}) != 0 && results(output).empty() &&
                  error_contains(root / "stderr.txt", "model has no Task that accepts"),
              "model without a prompt-only Task is rejected before family execution");
        bundle(model, "text_translation", "dataset_fixture", false, false);
        for (const bool explicit_task : {false, true}) {
            std::filesystem::remove(output);
            check((explicit_task ? invoke({"--task", "text_translation"}) : invoke({})) == 0,
                  "translation dataset uses the family's declared language defaults");
            rows = results(output);
            check(rows.size() == 2 && rows[0].at("task") == "text_translation" &&
                      rows[0].at("bundle_task") == "text_translation" &&
                      rows[0].at("text").get<std::string>().find("translated:First question") ==
                          0 &&
                      rows[0].at("generated_token_ids") == Json::array({100, 101}) &&
                      rows[0].at("prefill_ms") == 2 && rows[0].at("decode_ms") == 4,
                  "translation preserves complete prompt/output/timing without invented languages");
        }
        bundle(model, "text_generation");
        check(invoke({"--seed", "10"}) == 0, "existing family path still succeeds");
        rows = results(output);
        check(rows.size() == 2 && config(rows[0]).at("seed") == 13 &&
                  config(rows[1]).at("seed") == 11 && config(rows[0]).at("max_new_tokens") == 12000,
              "existing workload and per-row seeding unchanged");
        check(invoke({"--set", "seed=10"}) != 0,
              "new controls cannot be silently ignored by old path");
        std::filesystem::remove(output);
        check(invoke({"--task", "text_continuation"}) != 0 && results(output).empty(),
              "explicit semantic selection never enters the legacy path");
        bundle(model, "text_continuation");
        {
            std::ofstream file(dataset);
            file << Json{{"prompt", "Seed"}, {"seed_index", 4294967296LL}} << '\n';
        }
        check(invoke({"--seed", "10"}) == 0, "int64 seed_index is not narrowed to int32");
        rows = results(output);
        check(rows.size() == 1 && config(rows[0]).at("seed") == 4294967306LL,
              "full seed_index contributes to the actual submitted seed");
        for (const auto& invalid :
             {Json(3.75), Json(true), Json("3"), Json(18446744073709551615ULL)}) {
            {
                std::ofstream file(dataset);
                file << Json{{"prompt", "Seed"}, {"seed_index", invalid}} << '\n';
            }
            check(invoke({"--seed", "10"}) != 0,
                  "invalid seed_index type/range fails without coercion");
        }
        {
            std::ofstream file(dataset);
            file << Json{{"prompt", "Seed"}, {"seed_index", 3}} << '\n';
        }
        bundle(model, "text_continuation", "api_fixture");
        check(invoke({"--max-new-tokens", "8"}) == 0,
              "limited Task uses only declared implicit controls");
        check(invoke({"--max-new-tokens", "8", "--top-k", "1"}) != 0,
              "explicit unsupported benchmark option fails even at a default value");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << error.what() << '\n';
    }
    std::filesystem::remove_all(root);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}

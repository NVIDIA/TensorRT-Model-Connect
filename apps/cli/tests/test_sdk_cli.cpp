/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "task_runtime.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path, std::string task,
                  const char* family = "api_fixture") {
    const auto header =
        nlohmann::json{{"format", 1},
                       {"family", family},
                       {"task", std::move(task)},
                       {"backend", "fake"},
                       {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
            .dump();
    std::ofstream file(path, std::ios::binary);
    file.write("BUNDLE\x01\x00", 8);
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((length >> shift) & 0xffU));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write("PLAN", 4);
    if (!file)
        throw std::runtime_error("cannot write CLI fixture bundle");
}

struct Run {
    int status;
    std::string output;
    std::string error;
};

struct FlushBuffer final : std::stringbuf {
    int flushes{0};
    int sync() override {
        ++flushes;
        return std::stringbuf::sync();
    }
};

Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    std::ostringstream output, error;
    const int status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto bundle = root / "cli-sdk.bundle";
    write_bundle(bundle, "text_continuation");
    const std::vector<std::string> base{
        "trtmc", "run", bundle.string(), "--runtime-root", root.string(), "--prompt", "Hello"};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return run(std::move(arguments));
    };
    const auto defaults = invoke({});
    check(defaults.status == 0, "semantic primary mode runs through the public SDK");
    if (defaults.status == 0) {
        const auto value = nlohmann::json::parse(defaults.output);
        check(value.at("text") == "Hello!|eos", "family string/bool defaults are not replaced");
        check(value.at("decode_ms") == 4 && value.at("prefill_ms") == 0.75,
              "family numeric defaults are not replaced by CLI defaults");
        check(value.at("segments").at(0).at("text") == std::string("seg\0tail", 8),
              "readable JSON preserves length-delimited text segments");
    } else {
        std::cerr << defaults.error;
    }

    const auto explicit_values =
        invoke({"--set", "max_new_tokens=0", "--set", "temperature=0", "--set", "emit_eos=false",
                "--set", "suffix=", "--set", "token_biases=[-3,0]", "--set", "schedule=[0.25,0]",
                "--set", "labels=[\"a\",\"b\"]"});
    check(explicit_values.status == 0, "all seven typed config kinds cross the CLI and C ABI");
    if (explicit_values.status == 0) {
        const auto value = nlohmann::json::parse(explicit_values.output);
        check(value.at("text") == "Hello|a|b|-3" && value.at("decode_ms") == 0 &&
                  value.at("prefill_ms") == 0 && value.at("setup_ms") == 0.25,
              "explicit false/zero/empty and ordered lists reach the family unchanged");
    }
    const auto empty_lists =
        invoke({"--set", "labels=[]", "--set", "schedule=[]", "--set", "token_biases=[]"});
    check(empty_lists.status == 0, "declared kind disambiguates empty arrays");
    const auto shorthand = invoke({"--max-new-tokens", "9", "--temperature", "1.25"});
    check(shorthand.status == 0, "existing knob spellings map to declared underscore names");
    if (shorthand.status == 0) {
        const auto value = nlohmann::json::parse(shorthand.output);
        check(value.at("decode_ms") == 9 && value.at("prefill_ms") == 1.25,
              "explicit short options are transported without a second default layer");
    }

    for (const auto& extra :
         std::vector<std::vector<std::string>>{{"--set", "missing=1"},
                                               {"--set", "max_new_tokens=1.5"},
                                               {"--set", "emit_eos=0"},
                                               {"--set", "labels=[1]"},
                                               {"--set", "schedule=[null]"},
                                               {"--set", "token_biases=[9223372036854775808]"},
                                               {"--set", "temperature=null"},
                                               {"--set", "temperature=3.0"},
                                               {"--set", "suffix=a", "--set", "suffix=b"},
                                               {"--temperature", "1.0", "--set", "temperature=0.5"},
                                               {"--image", "not-an-input-to-text-continuation.png"},
                                               {"--task", "unknown_task"}}) {
        auto arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        const auto result = run(std::move(arguments));
        check(result.status != 0 && result.output.empty(),
              "invalid types/keys/values/duplicates/Task inputs fail without model output");
    }
    const auto duplicate = invoke({"--set", "suffix=a", "--set", "suffix=b"});
    check(duplicate.error.find("duplicate config: suffix") != std::string::npos,
          "duplicate diagnosis originates in the family parser");
    const auto ignored_lora = invoke({"--lora-adapter", "not-loaded.bundle"});
    check(ignored_lora.status != 0 && ignored_lora.output.empty() &&
              ignored_lora.error.find("must be supplied together") != std::string::npos,
          "model-level LoRA options are handled centrally, never silently skipped by a Task group");
    const auto range = invoke({"--set", "temperature=3.0"});
    check(range.error.find("temperature must be finite and between zero and two") !=
              std::string::npos,
          "range policy originates in the family parser");
    const auto huge = invoke({"--set", "token_biases=[-9223372036854775808]"});
    check(huge.status == 0 && huge.output.find("-9223372036854775808") != std::string::npos,
          "integer config is not rounded through floating-point parsing");
    check(trtmc::app::uses_existing_task_runtime("text_generation") &&
              !trtmc::app::uses_existing_task_runtime("text_continuation"),
          "old/new selection is an explicit bundle contract decision, not exception fallback");

    const auto missing_bundle = root / "cli-sdk-unsupported.bundle";
    write_bundle(missing_bundle, "disabled");
    const auto unsupported =
        run({"trtmc", "run", missing_bundle.string(), "--runtime-root", root.string(), "--task",
             "text_continuation", "--prompt", "Hello"});
    check(unsupported.status != 0 && unsupported.output.empty() &&
              unsupported.error.find("does not support the requested Task") != std::string::npos,
          "unsupported SDK Task is reported without retrying an old interface");
    std::filesystem::remove(missing_bundle);

    const auto text_bundle = root / "cli-sdk-text.bundle";
    write_bundle(text_bundle, "text_continuation", "text_fixture");
    struct TextCase {
        const char* task;
        std::vector<std::string> inputs;
        const char* expected;
    };
    for (const auto& item : std::vector<TextCase>{
             {"text_continuation", {"--prompt", ""}, "single:!"},
             {"text_continuation", {"--token-ids", "[7,9]"}, "single:tokens:7:9!"},
             {"text_continuation", {"--token-ids", "[]"}, "single:tokens!"},
             {"conditional_text_generation", {"--prompt", "Q"}, "conditional:Q!"},
             {"conditional_text_generation", {"--token-ids", "[7,9]"}, "conditional:tokens:7:9!"},
             {"corrupted_text_reconstruction", {"--prompt", "Q"}, "reconstructed:Q!"},
             {"unconditional_text_generation", {}, "unconditional!"},
             {"text_translation", {"--prompt", "Q"}, "translation:fixed-src->en:Q!"},
             {"text_translation",
              {"--prompt", "Q", "--source-language", "de", "--target-language", "fr"},
              "translation:de->fr:Q!"},
             {"text_summarization", {"--prompt", "Q"}, "summary:Q!"},
             {"text_prefix_suffix_infilling", {"--prefix", "", "--suffix", ""}, "middle:|!"},
             {"context_question_answering", {"--prompt", "Q", "--context", "C"}, "answer:Q|C!"}}) {
        std::vector<std::string> arguments{"trtmc",          "run",         text_bundle.string(),
                                           "--runtime-root", root.string(), "--task",
                                           item.task};
        arguments.insert(arguments.end(), item.inputs.begin(), item.inputs.end());
        const auto result = run(std::move(arguments));
        check(result.status == 0, "typed text Task CLI invocation succeeds");
        if (result.status == 0)
            check(nlohmann::json::parse(result.output).at("text") == item.expected,
                  "typed text inputs arrive at the matching family method");
    }
    for (const auto& ids : {"[true]", "[1.5]", "[2147483648]", "[-2147483649]", "null"}) {
        const auto invalid = run({"trtmc", "run", text_bundle.string(), "--runtime-root",
                                  root.string(), "--token-ids", ids});
        check(invalid.status != 0 && invalid.output.empty(),
              "run rejects malformed token IDs without returning model output");
    }
    const auto ambiguous_source = run({"trtmc", "run", text_bundle.string(), "--runtime-root",
                                       root.string(), "--prompt", "", "--token-ids", "[]"});
    check(ambiguous_source.status != 0 && ambiguous_source.output.empty(),
          "run does not choose silently between text and token IDs, even when empty");
    for (const auto* task :
         {"unconditional_text_generation", "text_summarization", "text_translation",
          "corrupted_text_reconstruction", "batch_text_continuation"}) {
        const auto invalid = run({"trtmc", "run", text_bundle.string(), "--runtime-root",
                                  root.string(), "--task", task, "--token-ids", "[7]"});
        check(invalid.status != 0 && invalid.output.empty(),
              "run rejects token IDs on a different Task input contract");
    }
    const auto prompts = root / "cli-sdk-prompts.txt";
    {
        std::ofstream file(prompts);
        file << "first\n\nthird\n";
    }
    const auto batch = run({"trtmc", "run", text_bundle.string(), "--runtime-root", root.string(),
                            "--task", "batch_text_continuation", "--prompts", prompts.string()});
    check(batch.status == 0, "native batch Task accepts a file of prompts");
    if (batch.status == 0) {
        const auto values = nlohmann::json::parse(batch.output).at("results");
        check(values.size() == 3 && values.at(1).at("text") == "batch:!" &&
                  values.at(2).at("text") == "batch:third!",
              "batch preserves empty prompts, item count and order");
        check(values.at(2).at("token_ids").back() == 0,
              "batch does not call the single-request family method");
    }

    const auto stream_bundle = root / "cli-sdk-stream.bundle";
    write_bundle(stream_bundle, "streaming_text_continuation", "stream_fixture");
    std::vector<std::string> arguments{"trtmc",          "run",         stream_bundle.string(),
                                       "--runtime-root", root.string(), "--prompt",
                                       "Hello"};
    std::vector<char*> raw;
    for (auto& argument : arguments)
        raw.push_back(argument.data());
    FlushBuffer buffer;
    std::ostream output(&buffer);
    std::ostringstream error;
    const auto streamed = trtmc::cli::run(static_cast<int>(raw.size()), raw.data(), output, error);
    check(streamed == 0 && buffer.flushes >= 2,
          "each text stream event is flushed for pipe consumers");
    std::istringstream lines(buffer.str());
    std::string delta, complete;
    std::getline(lines, delta);
    std::getline(lines, complete);
    if (!delta.empty() && !complete.empty()) {
        check(nlohmann::json::parse(delta).at("event") == "delta" &&
                  nlohmann::json::parse(complete).at("event") == "complete",
              "stream delivers typed delta and complete events in order");
    } else {
        check(false, "stream did not return both events");
    }
    const auto id_stream = run({"trtmc", "run", stream_bundle.string(), "--runtime-root",
                                root.string(), "--token-ids", "[7,9]"});
    check(id_stream.status == 0, "stream accepts the same typed token-ID prefix");
    if (id_stream.status == 0) {
        std::istringstream events(id_stream.output);
        std::getline(events, delta);
        std::getline(events, complete);
        check(
            nlohmann::json::parse(delta).at("text") == "tokens!" &&
                nlohmann::json::parse(complete).at("token_ids") == nlohmann::json::array({7, 9}),
            "stream retains token IDs and its family-owned output without decoding in shared code");
    }
    std::filesystem::remove(text_bundle);
    std::filesystem::remove(stream_bundle);
    std::filesystem::remove(prompts);
    // The primary bundle is retained as a tiny CLI smoke-test input.
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}

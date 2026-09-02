/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/args.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

trtmc::cli::CliArgs parse(std::initializer_list<std::string> arguments) {
    std::vector<std::string> storage(arguments);
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& argument : storage)
        argv.push_back(argument.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
}

void test_help_exposes_only_native_h3_runtime_commands() {
    std::ostringstream output;
    std::streambuf* previous = std::cerr.rdbuf(output.rdbuf());
    trtmc::cli::print_usage();
    std::cerr.rdbuf(previous);
    const auto help = output.str();
    check(help.find("generate-video") != std::string::npos,
          "runtime help exposes native video generation");
    check(help.find("FL2VA") != std::string::npos && help.find("Ref2VA") != std::string::npos,
          "runtime help exposes all public H3 workflows");
    check(help.find("--warmup") != std::string::npos &&
              help.find("--benchmark") != std::string::npos,
          "runtime help exposes same-process video timing");
    check(help.find("trtmc build") == std::string::npos,
          "runtime help does not expose Python-backed build");
    check(help.find("hf-python") == std::string::npos,
          "runtime help has no Python interpreter option");
}

void test_runtime_command_allowlist() {
    check(!parse({"trtmc", "version"}).parse_error, "runtime accepts version");
    check(!parse({"trtmc", "inspect", "model.bundle"}).parse_error,
          "runtime accepts bundle inspection");
    check(!parse({"trtmc", "inspect", "model.bundle", "--validate-runtime"}).parse_error,
          "runtime accepts native bundle and plugin validation");
    check(!parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
                  "out.mp4", "--warmup", "1", "--benchmark", "2"})
               .parse_error,
          "runtime accepts native video generation");
    check(parse({"trtmc", "build", "checkpoint"}).parse_error,
          "runtime rejects Python-backed build");
    check(parse({"trtmc", "graph", "list"}).parse_error,
          "runtime rejects Python-backed graph tools");
    check(parse({"trtmc", "run", "model.bundle", "--prompt", "hello"}).parse_error,
          "H3 runtime rejects unrelated generic commands");
    check(parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello",
                 "--validate-runtime"})
              .parse_error,
          "runtime restricts plugin validation to inspect");
}

void test_runtime_rejects_python_escape_hatch() {
    const auto args = parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello",
                             "--output", "out.mp4", "--hf-python", "python.exe"});
    check(args.parse_error && args.error_message.find("not present") != std::string::npos,
          "runtime rejects --hf-python");
}

void test_runtime_rejects_malformed_timing_counts() {
    const auto malformed_benchmark = parse({"trtmc", "generate-video", "model.bundle", "--prompt",
                                            "hello", "--output", "out.mp4", "--benchmark", "abc"});
    check(malformed_benchmark.parse_error &&
              malformed_benchmark.error_message == "--benchmark expects an integer >= 0",
          "runtime rejects a non-integer benchmark count");

    const auto trailing_warmup = parse({"trtmc", "generate-video", "model.bundle", "--prompt",
                                        "hello", "--output", "out.mp4", "--warmup", "3junk"});
    check(trailing_warmup.parse_error &&
              trailing_warmup.error_message == "--warmup expects an integer >= 0",
          "runtime rejects a warmup count with trailing characters");

    const auto negative_benchmark = parse({"trtmc", "generate-video", "model.bundle", "--prompt",
                                           "hello", "--output", "out.mp4", "--benchmark", "-1"});
    check(negative_benchmark.parse_error &&
              negative_benchmark.error_message == "--benchmark expects an integer >= 0",
          "runtime rejects a negative benchmark count");
}

} // namespace

int main() {
    test_help_exposes_only_native_h3_runtime_commands();
    test_runtime_command_allowlist();
    test_runtime_rejects_python_escape_hatch();
    test_runtime_rejects_malformed_timing_counts();
    return failures == 0 ? 0 : 1;
}

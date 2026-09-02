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

void test_help_matches_projected_runtime_models() {
    std::ostringstream output;
    std::streambuf* previous = std::cerr.rdbuf(output.rdbuf());
    trtmc::cli::print_usage();
    std::cerr.rdbuf(previous);
    const auto help = output.str();
    check(help.find("generate-video") != std::string::npos,
          "runtime help exposes native video generation");
#if defined(TRTMC_CLI_HAS_MINIMAX_H3_VIDEO_CONTRACT)
    check(help.find("FL2VA") != std::string::npos && help.find("Ref2VA") != std::string::npos,
          "runtime help exposes all public H3 workflows");
#else
    check(help.find("MiniMax-H3") == std::string::npos && help.find("FL2VA") == std::string::npos &&
              help.find("Ref2VA") == std::string::npos,
          "non-H3 runtime help does not advertise H3 workflows or branding");
#endif
    check(help.find("--warmup") != std::string::npos &&
              help.find("--benchmark") != std::string::npos,
          "runtime help exposes same-process video timing");
    check(help.find("--output OUTPUT.mp4 is required exactly once") != std::string::npos,
          "runtime help states the single MP4 output contract");
    check(help.find("trtmc build") == std::string::npos,
          "runtime help does not expose Python-backed build");
    check(help.find("hf-python") == std::string::npos,
          "runtime help has no Python interpreter option");
}

void test_runtime_requires_one_explicit_mp4_output() {
    const auto uppercase = parse(
        {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output", "movie.MP4"});
    check(!uppercase.parse_error && uppercase.output_dir == "movie.MP4",
          "runtime accepts a case-insensitive MP4 extension");

    const auto missing = parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello"});
    check(missing.parse_error &&
              missing.error_message ==
                  "generate-video requires exactly one --output OUTPUT.mp4 argument",
          "runtime rejects a missing output during parsing");

    for (const char* output : {"frames", "frames/", "movie.mp4.tmp", "movie.png"}) {
        const auto invalid = parse(
            {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output", output});
        check(invalid.parse_error &&
                  invalid.error_message ==
                      "generate-video --output must be a file path ending in .mp4",
              "runtime rejects a directory or non-MP4 output during parsing");
    }

    const auto duplicate = parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello",
                                  "--output", "first.mp4", "-o", "second.mp4"});
    check(duplicate.parse_error &&
              duplicate.error_message ==
                  "generate-video requires exactly one --output OUTPUT.mp4 argument",
          "runtime rejects multiple output arguments during parsing");
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
    const auto unrelated = parse({"trtmc", "run", "model.bundle", "--prompt", "hello"});
    check(unrelated.parse_error, "runtime rejects unrelated generic commands");
#if defined(TRTMC_CLI_HAS_MINIMAX_H3_VIDEO_CONTRACT)
    check(unrelated.error_message.find("MiniMax-H3 runtime CLI") != std::string::npos,
          "H3 runtime command error retains H3 branding");
#else
    check(unrelated.error_message.find("runtime-only ModelConnect CLI") != std::string::npos &&
              unrelated.error_message.find("MiniMax-H3") == std::string::npos,
          "non-H3 runtime command error uses generic branding");
#endif
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

void test_nonlocked_runtime_keeps_development_loader_overrides() {
    const auto args =
        parse({"trtmc", "inspect", "model.bundle", "--backend-dir", "C:\\backends",
               "--model-plugin-dir", "C:\\models", "--kernel-bindings", "bindings.json"});
    check(!args.parse_error, "nonlocked runtime accepts development loader overrides");
    check(args.backend_search_paths.size() == 1 && args.model_plugin_search_paths.size() == 1 &&
              args.kernel_bindings_path == "bindings.json",
          "nonlocked runtime records development loader overrides");
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
    test_help_matches_projected_runtime_models();
    test_runtime_command_allowlist();
    test_runtime_requires_one_explicit_mp4_output();
    test_runtime_rejects_python_escape_hatch();
    test_nonlocked_runtime_keeps_development_loader_overrides();
    test_runtime_rejects_malformed_timing_counts();
    return failures == 0 ? 0 : 1;
}

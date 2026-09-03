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

void test_locked_help_has_no_loader_override() {
    std::ostringstream output;
    std::streambuf* previous = std::cerr.rdbuf(output.rdbuf());
    trtmc::cli::print_usage();
    std::cerr.rdbuf(previous);
    const auto help = output.str();
    check(help.find("--backend-dir") == std::string::npos, "locked help hides backend override");
    check(help.find("--model-plugin-dir") == std::string::npos,
          "locked help hides model plugin override");
    check(help.find("--kernel-bindings") == std::string::npos,
          "locked help hides kernel bindings override");
    check(help.find("--runtime-cache") != std::string::npos,
          "locked help retains native runtime cache");
}

void test_locked_parser_rejects_loader_overrides() {
    for (const char* option : {"--backend-dir", "--model-plugin-dir", "--kernel-bindings"}) {
        auto with_value = parse({"trtmc", "inspect", "model.bundle", option, "C:\\untrusted"});
        check(with_value.parse_error, "locked parser rejects override with a value");
        check(with_value.error_message.find("disabled in the locked MiniMax-H3 runtime") !=
                  std::string::npos,
              "locked parser explains rejected override");

        auto without_value = parse({"trtmc", "inspect", "model.bundle", option});
        check(without_value.parse_error, "locked parser rejects override without a value");
        check(without_value.backend_search_paths.empty() &&
                  without_value.model_plugin_search_paths.empty() &&
                  without_value.kernel_bindings_path.empty(),
              "locked parser never records an override");
    }
}

void test_locked_parser_keeps_package_commands() {
    const auto inspect = parse({"trtmc", "inspect", "model.bundle", "--validate-runtime",
                                "--list-engines", "--runtime-cache", "cache.bin"});
    check(!inspect.parse_error, "locked parser accepts package inspection");
    check(inspect.validate_runtime && inspect.list_engines,
          "locked parser retains inspection validation flags");
    check(inspect.runtime_cache == "cache.bin", "locked parser retains runtime cache");
}

void test_locked_generate_video_requires_one_mp4_output() {
    const auto valid = parse(
        {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output", "result.Mp4"});
    check(!valid.parse_error && valid.output_dir == "result.Mp4",
          "locked parser accepts one case-insensitive MP4 output");

    const auto missing = parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello"});
    check(missing.parse_error, "locked parser rejects a missing output");

    const auto directory = parse(
        {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output", "frames"});
    check(directory.parse_error, "locked parser rejects a frame-directory output");

    const auto duplicate = parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello",
                                  "--output", "first.mp4", "--output", "second.mp4"});
    check(duplicate.parse_error, "locked parser rejects multiple outputs");
}

void test_locked_generate_video_rejects_unsupported_latent_injection() {
    const auto parsed =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--initial-latents-raw", "latents.raw"});
    check(parsed.parse_error, "locked H3 parser rejects unsupported initial latents");
    check(parsed.error_message.find("not supported by the native MiniMax-H3 runtime") !=
              std::string::npos,
          "locked H3 parser explains unsupported initial latents");
    check(parsed.initial_latents_raw.empty(), "locked H3 parser never records initial latents");
}

void test_locked_generate_video_requires_one_non_negative_seed() {
    for (const char* value : {"-1", "1,2"}) {
        const auto parsed =
            parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
                   "result.mp4", "--seed", value});
        check(parsed.parse_error, "locked H3 parser rejects a non-scalar or negative seed");
        check(parsed.error_message.find("one non-negative integer") != std::string::npos,
              "locked H3 parser explains its seed contract");
    }

    const auto valid =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--seed", "17"});
    check(!valid.parse_error && valid.seed == 17,
          "locked H3 parser accepts one non-negative scalar seed");
}

} // namespace

int main() {
    test_locked_help_has_no_loader_override();
    test_locked_parser_rejects_loader_overrides();
    test_locked_parser_keeps_package_commands();
    test_locked_generate_video_requires_one_mp4_output();
    test_locked_generate_video_rejects_unsupported_latent_injection();
    test_locked_generate_video_requires_one_non_negative_seed();
    return failures == 0 ? 0 : 1;
}

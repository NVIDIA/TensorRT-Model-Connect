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
    check(help.find("--config") != std::string::npos && help.find("--set") != std::string::npos,
          "locked help exposes its schema-checked runtime configuration");
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

void test_generate_video_uses_fast_bundle_validation_policy() {
    check(!trtmc::cli::validate_bundle_payloads_for_command("generate-video"),
          "generate-video skips plan-content SHA-256 attestation");
    check(trtmc::cli::validate_bundle_payloads_for_command("inspect"),
          "inspect retains strict plan payload SHA-256 validation");
    check(trtmc::cli::validate_bundle_payloads_for_command("run"),
          "other commands retain strict plan payload SHA-256 validation");
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

void test_locked_generate_video_strict_numeric_inputs() {
    for (const char* option : {"--height", "--width"}) {
        for (const char* value : {"abc", "768junk", "0", "-1"}) {
            const auto parsed =
                parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello",
                       "--output", "result.mp4", option, value});
            check(parsed.parse_error, "locked H3 parser rejects a malformed dimension");
            check(parsed.error_message.find("expects an integer > 0") != std::string::npos,
                  "locked H3 parser explains its positive dimension contract");
        }
    }

    for (const char* value : {"abc", "4junk", "0", "-1"}) {
        const auto parsed =
            parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
                   "result.mp4", "--num-inference-steps", value});
        check(parsed.parse_error, "locked H3 parser rejects malformed inference steps");
    }

    for (const char* value : {"abc", "nan", "1junk", "0", "2"}) {
        const auto parsed =
            parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
                   "result.mp4", "--guidance-scale", value});
        check(parsed.parse_error, "locked H3 parser rejects malformed or unsupported guidance");
    }

    const auto valid = parse(
        {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
         "result.mp4", "--height", "768", "--width", "1344", "--num-inference-steps", "4",
         "--guidance-scale", "1"});
    check(!valid.parse_error && valid.diffusion_height == 768 && valid.diffusion_width == 1344 &&
              valid.num_steps == 4 && valid.guidance_scale == 1.0F,
          "locked H3 parser preserves strict valid generation numbers");
}

void test_locked_generate_video_rejects_irrelevant_known_options() {
    const std::vector<std::vector<std::string>> unsupported{
        {"--temperature", "0.5"},
        {"--top-p", "0.9"},
        {"--top-k", "4"},
        {"--max-new-tokens", "8"},
        {"--num-images", "2"},
        {"--condition-latents-raw", "condition.raw"},
        {"--condition-mask-raw", "mask.raw"},
        {"--sampling-steps-raw", "steps.raw"},
        {"--sde-noise-raw", "noise.raw"},
        {"--cfg-scale", "7.5"},
        {"--chat-template"},
        {"--cuda-graphs"},
        {"--list-engines"},
    };
    for (const auto& option : unsupported) {
        std::vector<std::string> arguments{"trtmc",         "generate-video", "model.bundle",
                                           "--prompt",      "hello",          "--output",
                                           "result.mp4"};
        arguments.insert(arguments.end(), option.begin(), option.end());
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (auto& argument : arguments)
            argv.push_back(argument.data());
        const auto parsed =
            trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
        check(parsed.parse_error, "locked H3 parser rejects an irrelevant known option");
        check(parsed.error_message.find("not supported by the locked MiniMax-H3") !=
                  std::string::npos,
              "locked H3 parser identifies its exact option allowlist");
    }

    const auto negative_prompt =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--negative-prompt", "blur"});
    check(negative_prompt.parse_error &&
              negative_prompt.error_message.find("guidance-distilled") != std::string::npos,
          "locked H3 parser rejects negative prompting with the model-specific reason");
}

void test_locked_generate_video_timing_and_runtime_configuration() {
    const auto warmup_only =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--warmup", "1"});
    check(warmup_only.parse_error &&
              warmup_only.error_message == "--warmup requires --benchmark N with N > 0",
          "locked H3 parser rejects a silently ignored warmup");

    const auto configured = parse(
        {"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
         "result.mp4", "--runtime-cache", "cache.bin", "--config", "runtime.toml", "--set",
         "minimax_h3.retain_engines=true", "--warmup", "1", "--benchmark", "2"});
    check(!configured.parse_error && configured.runtime_cache == "cache.bin" &&
              configured.config_path == "runtime.toml" && configured.set_tokens.size() == 1 &&
              configured.warmup == 1 && configured.benchmark == 2,
          "locked H3 parser preserves every supported runtime and timing option");
}

void test_locked_ref2va_requires_a_visual_reference() {
    const auto audio_only =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--reference-audio", "voice.wav"});
    check(audio_only.parse_error &&
              audio_only.error_message.find("at least one reference image or video") !=
                  std::string::npos,
          "locked H3 parser rejects audio-only Ref2VA before bundle loading");

    const auto image_and_audio =
        parse({"trtmc", "generate-video", "model.bundle", "--prompt", "hello", "--output",
               "result.mp4", "--reference-image", "subject.png", "--reference-audio",
               "voice.wav"});
    check(!image_and_audio.parse_error && image_and_audio.video_references.size() == 2,
          "locked H3 parser accepts an audio reference paired with an image");
}

} // namespace

int main() {
    test_locked_help_has_no_loader_override();
    test_locked_parser_rejects_loader_overrides();
    test_locked_parser_keeps_package_commands();
    test_generate_video_uses_fast_bundle_validation_policy();
    test_locked_generate_video_requires_one_mp4_output();
    test_locked_generate_video_rejects_unsupported_latent_injection();
    test_locked_generate_video_requires_one_non_negative_seed();
    test_locked_generate_video_strict_numeric_inputs();
    test_locked_generate_video_rejects_irrelevant_known_options();
    test_locked_generate_video_timing_and_runtime_configuration();
    test_locked_ref2va_requires_a_visual_reference();
    return failures == 0 ? 0 : 1;
}

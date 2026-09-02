/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CLI-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         CLI argument parsing for trtmc commands
// Preconditions:  None
// Postconditions: Production parser accepts supported commands and rejects
//                  malformed command lines
// =============================================================================

#include "cli/args.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void check_message_contains(const std::string& message, const std::string& needle,
                                   const char* test_name) {
    check(message.find(needle) != std::string::npos, test_name);
}

namespace {

void test_help_uses_platform_neutral_library_names() {
    std::ostringstream output;
    std::streambuf* previous = std::cerr.rdbuf(output.rdbuf());
    trtmc::cli::print_usage();
    std::cerr.rdbuf(previous);

    check(output.str().find("backend shared libraries") != std::string::npos,
          "help names backend shared libraries portably");
    check(output.str().find("model-plugin shared libraries") != std::string::npos,
          "help names model-plugin shared libraries portably");
    check(output.str().find("libtrtmc_backend_*.so") == std::string::npos,
          "help does not hard-code Linux backend filenames");
    check(output.str().find("--num-frames N") != std::string::npos,
          "help documents native video frame count");
    check(output.str().find("--first-frame IMAGE") != std::string::npos &&
              output.str().find("--reference-video VIDEO") != std::string::npos,
          "help documents native FL2VA and Ref2VA inputs");
}

trtmc::cli::CliArgs parse(std::initializer_list<std::string> args) {
    std::vector<std::string> storage(args);
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& arg : storage)
        argv.push_back(arg.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
}

void test_no_args_show_help() {
    auto args = parse({"trtmc"});
    check(args.show_help, "no args shows help");
    check(!args.parse_error, "no args not parse error");
}

void test_help_aliases_show_help() {
    check(parse({"trtmc", "help"}).show_help, "help shows help");
    check(parse({"trtmc", "--help"}).show_help, "--help shows help");
    check(parse({"trtmc", "-h"}).show_help, "-h shows help");
}

void test_version_aliases() {
    check(parse({"trtmc", "version"}).command == "version", "version command");
    check(parse({"trtmc", "--version"}).command == "version", "--version command");
    check(parse({"trtmc", "-v"}).command == "version", "-v command");
}

void test_build_forwards_args_verbatim() {
    auto args = parse(
        {"trtmc", "build", "Example/Decoder-0.6B", "-o", "out.bundle", "--precision", "fp16"});
    check(args.command == "build", "build command");
    check(args.build_args.size() == 5, "build forwards arg count");
    check(args.build_args[0] == "Example/Decoder-0.6B", "build forwards model");
    check(args.build_args[4] == "fp16", "build forwards final value");

    auto graph = parse({"trtmc", "graph", "list", "graph.json"});
    check(graph.command == "graph" && graph.build_args.size() == 2,
          "graph command forwards to Python");
}

void test_run_parses_common_flags() {
    auto args = parse({"trtmc",
                       "run",
                       "bundle.bundle",
                       "--prompt",
                       "hello",
                       "--max-new-tokens",
                       "8",
                       "--source-language-token-id",
                       "256047",
                       "--forced-bos-token-id",
                       "256057",
                       "--generation-mode",
                       "diffusion",
                       "--block-length",
                       "32",
                       "--threshold",
                       "0.9",
                       "--temperature",
                       "0.5",
                       "--top-p",
                       "0.9",
                       "--min-p",
                       "0.15",
                       "--repetition-penalty",
                       "1.05",
                       "--top-k",
                       "4",
                       "--seed",
                       "123",
                       "--greedy",
                       "--chat-template",
                       "--no-thinking",
                       "--lora-adapter",
                       "/tmp/adapter",
                       "--lora-adapter-id",
                       "adapter-1",
                       "--kv-cache-size",
                       "2GiB",
                       "--backend-dir",
                       "/tmp/lib",
                       "--model-plugin-dir",
                       "/tmp/models",
                       "--runtime-cache",
                       "/tmp/cache",
                       "--kernel-bindings",
                       "/tmp/bindings.json",
                       "--cuda-graphs"});
    check(args.command == "run", "run command");
    check(args.bundle_path == "bundle.bundle", "run bundle");
    check(args.prompt_provided, "run prompt provided");
    check(args.prompt == "hello", "run prompt");
    check(args.max_new_tokens == 8, "run max tokens");
    check(args.source_language_token_id == 256047, "run source language token");
    check(args.forced_bos_token_id == 256057, "run forced BOS token");
    check(args.generation_mode == "diffusion", "run generation mode");
    check(args.block_length == 32, "run block length");
    check(args.conf_threshold > 0.89F && args.conf_threshold < 0.91F, "run threshold");
    check(args.temperature > 0.49F && args.temperature < 0.51F, "run temperature");
    check(args.min_p > 0.14F && args.min_p < 0.16F, "run min_p");
    check(args.repetition_penalty > 1.04F && args.repetition_penalty < 1.06F,
          "run repetition penalty");
    check(args.top_k == 4, "run top_k");
    check(args.seed == 123, "run seed");
    check(args.greedy, "run greedy");
    check(args.chat_template, "run chat template");
    check(args.no_thinking, "run no thinking");
    check(args.lora_adapter_path == "/tmp/adapter", "run LoRA adapter path");
    check(args.lora_adapter_id == "adapter-1", "run LoRA adapter ID");
    check(args.kv_cache_size_bytes == 2147483648ULL, "run kv cache size");
    check(args.backend_search_paths.size() == 1 && args.backend_search_paths[0] == "/tmp/lib",
          "run backend dir");
    check(args.model_plugin_search_paths.size() == 1 &&
              args.model_plugin_search_paths[0] == "/tmp/models",
          "run model plugin dir");
    check(args.runtime_cache == "/tmp/cache", "run runtime cache");
    check(args.kernel_bindings_path == "/tmp/bindings.json", "run kernel bindings");
    check(args.cuda_graphs, "run cuda graphs");
}

void test_diffusion_flags() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--prompt", "paint", "--negative-prompt",
                       "blur", "--num-inference-steps", "20", "--height", "512", "--width", "768",
                       "--cfg-scale", "7.5", "--initial-latents-raw", "latents.raw"});
    check(args.negative_prompt == "blur", "diffusion negative prompt");
    check(args.num_steps == 20, "diffusion num steps");
    check(args.diffusion_height == 512, "diffusion height");
    check(args.diffusion_width == 768, "diffusion width");
    check(args.cfg_scale > 7.49F && args.cfg_scale < 7.51F, "diffusion cfg scale");
    check(args.initial_latents_raw == "latents.raw", "diffusion latents");
}

void test_generate_video_parses_native_output_contract() {
    auto args = parse({"trtmc", "generate-video", "bundle.bundle", "--prompt", "sports scene",
                       "--output", "video-out", "--num-inference-steps", "4", "--height", "768",
                       "--width", "1344", "--seed", "0", "--num-frames", "124"});
    check(!args.parse_error, "generate-video parses cleanly");
    check(args.command == "generate-video", "generate-video command");
    check(args.bundle_path == "bundle.bundle", "generate-video bundle");
    check(args.prompt == "sports scene" && args.prompt_provided, "generate-video prompt");
    check(args.output_dir == "video-out", "generate-video output directory");
    check(args.num_steps == 4, "generate-video inference steps");
    check(args.diffusion_height == 768 && args.diffusion_width == 1344,
          "generate-video dimensions");
    check(args.seed == 0, "generate-video seed");
    check(args.video_num_frames == 124, "generate-video 124-frame request");

    auto long_video = parse({"trtmc", "generate-video", "bundle.bundle", "--prompt", "scene",
                             "--output", "video-out", "--num-frames", "345"});
    check(!long_video.parse_error && long_video.video_num_frames == 345,
          "generate-video 345-frame request");
    check(parse({"trtmc", "generate-video", "bundle.bundle", "--prompt", "scene", "--num-frames",
                 "0"})
              .parse_error,
          "generate-video rejects zero frames");
    check(parse({"trtmc", "generate-video", "bundle.bundle", "--prompt", "scene", "--num-frames",
                 "abc"})
              .parse_error,
          "generate-video rejects malformed frame count");
}

void test_generate_video_parses_public_multimodal_modes() {
    auto fl2va = parse({"trtmc", "generate-video", "bundle.bundle", "--prompt", "animate",
                        "--first-frame", "first.png", "--last-frame", "last.jpg"});
    check(!fl2va.parse_error && fl2va.first_frame_path == "first.png" &&
              fl2va.last_frame_path == "last.jpg" && fl2va.video_references.empty(),
          "generate-video parses FL2VA first and last frames");

    auto ref2va = parse({"trtmc", "generate-video", "bundle.bundle", "--prompt",
                         "follow references", "--reference-audio", "voice.wav", "--reference-image",
                         "person.png", "--reference-video", "motion-video"});
    check(!ref2va.parse_error && ref2va.video_references.size() == 3,
          "generate-video parses Ref2VA media");
    check(ref2va.video_references.size() == 3 &&
              ref2va.video_references[0].kind == trtmc::cli::VideoReferenceArgKind::kAudio &&
              ref2va.video_references[1].kind == trtmc::cli::VideoReferenceArgKind::kImage &&
              ref2va.video_references[2].kind ==
                  trtmc::cli::VideoReferenceArgKind::kVideoDirectory &&
              ref2va.video_references[0].path == "voice.wav" &&
              ref2va.video_references[2].path == "motion-video",
          "Ref2VA command-line media order is preserved exactly");
}

void test_generate_video_rejects_invalid_public_multimodal_counts() {
    check(parse({"trtmc", "generate-video", "b.bundle", "--prompt", "x", "--first-frame",
                 "first.png", "--reference-image", "reference.png"})
              .parse_error,
          "generate-video rejects mixed FL2VA and Ref2VA inputs");
    check(parse({"trtmc", "generate-video", "b.bundle", "--prompt", "x", "--first-frame", "one.png",
                 "--first-frame", "two.png"})
              .parse_error,
          "generate-video rejects duplicate first frame");
    check(!parse({"trtmc", "generate-video", "b.bundle", "--prompt", "x", "--reference-audio",
                  "voice.wav"})
               .parse_error,
          "Ref2VA accepts audio-only conditioning");
    check(parse({"trtmc", "generate-video", "b.bundle", "--prompt", "x", "--reference-image",
                 "i.png", "--reference-video", "v1", "--reference-video", "v2", "--reference-video",
                 "v3", "--reference-video", "v4"})
              .parse_error,
          "Ref2VA rejects more than three videos");
    check(parse({"trtmc", "generate-video",    "b.bundle", "--prompt",
                 "x",     "--reference-image", "i1",       "--reference-image",
                 "i2",    "--reference-image", "i3",       "--reference-image",
                 "i4",    "--reference-image", "i5",       "--reference-image",
                 "i6",    "--reference-image", "i7",       "--reference-image",
                 "i8",    "--reference-image", "i9",       "--reference-image",
                 "i10"})
              .parse_error,
          "Ref2VA rejects more than nine images");
    check(parse({"trtmc", "generate-video", "b.bundle", "--prompt", "x", "--reference-image",
                 "i.png", "--reference-audio", "a1.wav", "--reference-audio", "a2.wav",
                 "--reference-audio", "a3.wav", "--reference-audio", "a4.wav"})
              .parse_error,
          "Ref2VA rejects more than three explicit audio files");
    check(parse({"trtmc", "generate-video",    "b.bundle", "--prompt",
                 "x",     "--reference-image", "i1",       "--reference-image",
                 "i2",    "--reference-image", "i3",       "--reference-image",
                 "i4",    "--reference-image", "i5",       "--reference-image",
                 "i6",    "--reference-image", "i7",       "--reference-image",
                 "i8",    "--reference-image", "i9",       "--reference-video",
                 "v1",    "--reference-video", "v2",       "--reference-video",
                 "v3",    "--reference-audio", "a1.wav"})
              .parse_error,
          "Ref2VA rejects more than twelve ordered files");
    check(parse({"trtmc", "run", "b.bundle", "--prompt", "x", "--reference-image", "i.png"})
              .parse_error,
          "video conditioning flags are scoped to generate-video");
}

void test_detect_parses_contract_flags() {
    auto args = parse({"trtmc", "detect", "bundle.bundle", "--image", "img.jpg", "--output-json",
                       "detections.json", "--score-threshold", "0.42"});
    check(args.command == "detect", "detect command");
    check(args.bundle_path == "bundle.bundle", "detect bundle");
    check(args.image_path == "img.jpg", "detect image");
    check(args.output_json == "detections.json", "detect output json");
    check(args.conf_threshold > 0.41F && args.conf_threshold < 0.43F, "detect threshold");
}

void test_extract_features_parses_contract_flags() {
    auto args = parse({"trtmc", "extract-features", "dinov3.bundle", "--image", "cat.png",
                       "--output-json", "features.json"});
    check(!args.parse_error, "extract-features parses cleanly");
    check(args.command == "extract-features", "extract-features command");
    check(args.bundle_path == "dinov3.bundle", "extract-features bundle");
    check(args.image_path == "cat.png", "extract-features image");
    check(args.output_json == "features.json", "extract-features output json");
}

void test_disparity_parses_stereo_images() {
    auto args = parse({"trtmc", "disparity", "bundle.bundle", "--image", "left.png",
                       "--right-image", "right.png", "--output", "disparity.f32"});
    check(args.command == "disparity", "disparity command");
    check(args.bundle_path == "bundle.bundle", "disparity bundle");
    check(args.image_path == "left.png", "disparity left image");
    check(args.right_image_path == "right.png", "disparity right image");
    check(args.output_dir == "disparity.f32", "disparity output");
}

void test_inspect_and_config_flags() {
    auto args = parse({"trtmc", "inspect", "bundle.bundle", "--list-engines", "--validate-runtime",
                       "--config", "profile.json", "--set", "audio.seed=7"});
    check(args.command == "inspect", "inspect command");
    check(args.bundle_path == "bundle.bundle", "inspect bundle");
    check(args.list_engines, "inspect list engines");
    check(args.validate_runtime, "inspect validates runtime plugin contract");
    check(args.config_path == "profile.json", "config path");
    check(args.set_tokens.size() == 1 && args.set_tokens[0] == "audio.seed=7", "set token");
}

void test_audio_and_solve_flags() {
    auto transcribe =
        parse({"trtmc", "transcribe", "bundle.bundle", "--audio", "input.wav", "--stream",
               "--chunk-ms", "80", "--att-context-size", "5,2", "--pad-and-drop-preencoded"});
    check(transcribe.audio_in == "input.wav", "transcribe audio");
    check(transcribe.audio_inputs == std::vector<std::string>({"input.wav"}),
          "transcribe audio list");
    check(transcribe.stream, "transcribe stream");
    check(transcribe.chunk_ms == 80, "transcribe chunk ms");
    check(transcribe.att_context_left == 5 && transcribe.att_context_right == 2,
          "transcribe att context");
    check(transcribe.pad_and_drop_preencoded, "transcribe pad/drop");

    auto solve =
        parse({"trtmc", "solve", "bundle.bundle", "--branch-input", "1,2", "--trunk-input", "3,4"});
    check(solve.branch_input == "1,2", "solve branch");
    check(solve.trunk_input == "3,4", "solve trunk");
}

void test_canary_transcription_flags_and_batch() {
    auto args = parse({"trtmc",
                       "transcribe",
                       "bundle.bundle",
                       "--audio",
                       "one.wav",
                       "--audio",
                       "two.wav",
                       "--beam-size",
                       "32",
                       "--beam-fallback-max-size",
                       "32",
                       "--length-penalty",
                       "0",
                       "--source-language",
                       "en",
                       "--target-language",
                       "fr",
                       "--task",
                       "translate",
                       "--no-punctuation",
                       "--timestamps",
                       "--max-input-seconds",
                       "45.5",
                       "--segment-length-seconds",
                       "30",
                       "--segment-min-seconds",
                       "20",
                       "--segment-overlap-seconds",
                       "2",
                       "--lcs-merge"});
    check(!args.parse_error, "Canary transcription controls parse");
    check(args.audio_inputs == std::vector<std::string>({"one.wav", "two.wav"}),
          "Canary repeated audio inputs form batch");
    check(args.beam_size == 32, "Canary beam size");
    check(args.beam_fallback_max_size == 32, "Canary beam fallback limit");
    check(args.length_penalty == 0.0F, "Canary beam length penalty");
    check(args.source_language == "en" && args.target_language == "fr",
          "Canary source and target languages");
    check(args.transcription_task == "translate", "Canary translation task");
    check(!args.punctuation, "Canary punctuation toggle");
    check(args.timestamps, "Canary timestamp toggle");
    check(args.max_input_seconds > 45.49F && args.max_input_seconds < 45.51F,
          "Canary maximum input seconds");
    check(args.segment_length_seconds == 30.0F, "Canary segment length seconds");
    check(args.segment_min_seconds == 20.0F, "Canary dynamic segment minimum");
    check(args.segment_overlap_seconds == 2.0F, "Canary segment overlap");
    check(args.lcs_merge, "Canary LCS merge");

    check(parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--beam-size", "0"})
              .parse_error,
          "Canary rejects zero beam size");
    check(parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--beam-size", "33"})
              .parse_error,
          "Canary rejects beam size above 32");
    check(parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--beam-fallback-max-size",
                 "33"})
              .parse_error,
          "Canary rejects beam fallback above 32");
    check(parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--length-penalty", "-1"})
              .parse_error,
          "Canary rejects negative length penalty");
    check(parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--task", "other"})
              .parse_error,
          "Canary rejects unknown task");
    check(
        parse({"trtmc", "transcribe", "b.bundle", "--audio", "a.wav", "--max-input-seconds", "nan"})
            .parse_error,
        "Canary rejects non-finite duration");
}

void test_unknown_command_fails() {
    auto args = parse({"trtmc", "missing"});
    check(args.parse_error, "unknown command parse error");
    check(args.error_message == "Unknown command: missing", "unknown command message");
}

void test_unknown_flag_fails() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--bogus"});
    check(args.parse_error, "unknown flag parse error");
    check(args.error_message == "Unknown flag: --bogus", "unknown flag message");
}

void test_missing_value_fails() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--prompt"});
    check(args.parse_error, "missing value parse error");
    check(args.error_message == "--prompt requires a value", "missing value message");

    auto bindings = parse({"trtmc", "run", "bundle.bundle", "--kernel-bindings"});
    check(bindings.parse_error, "missing kernel bindings value parse error");
    check(bindings.error_message == "--kernel-bindings requires a value",
          "missing kernel bindings value message");
}

void test_missing_prompt_is_distinct_from_empty_prompt() {
    auto missing = parse({"trtmc", "run", "bundle.bundle", "--max-new-tokens", "8"});
    check(missing.parse_error, "missing prompt parse error");
    check(missing.error_message ==
              "run requires bundle + --prompt, --prompts-file, or --initial-latents-raw",
          "missing prompt message");
    check(!missing.prompt_provided, "missing prompt not provided");
    check(missing.prompt.empty(), "missing prompt text empty");

    auto empty = parse({"trtmc", "run", "bundle.bundle", "--prompt", "", "--max-new-tokens", "8"});
    check(empty.prompt_provided, "empty prompt provided");
    check(empty.prompt.empty(), "empty prompt text empty");
    check(!empty.parse_error, "empty prompt parse ok");
}

void test_bad_kv_cache_size_fails() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--kv-cache-size=abc"});
    check(args.parse_error, "bad kv cache parse error");
    check(args.error_message.find("--kv-cache-size expects") == 0, "bad kv cache message");
}

void test_invalid_generation_sampling_values_fail() {
    auto negative_tokens =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--max-new-tokens", "-5"});
    check(negative_tokens.parse_error, "negative max tokens parse error");
    check_message_contains(negative_tokens.error_message, "--max-new-tokens expects an integer > 0",
                           "negative max tokens message");

    auto malformed_tokens =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--max-new-tokens", "abc"});
    check(malformed_tokens.parse_error, "malformed max tokens parse error");
    check_message_contains(malformed_tokens.error_message,
                           "--max-new-tokens expects an integer > 0",
                           "malformed max tokens message");

    auto negative_source_language = parse(
        {"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--source-language-token-id", "-1"});
    check(negative_source_language.parse_error, "negative source language token parse error");
    check_message_contains(negative_source_language.error_message,
                           "--source-language-token-id expects an integer >= 0",
                           "negative source language token message");

    auto malformed_forced_bos = parse(
        {"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--forced-bos-token-id", "abc"});
    check(malformed_forced_bos.parse_error, "malformed forced BOS token parse error");
    check_message_contains(malformed_forced_bos.error_message,
                           "--forced-bos-token-id expects an integer >= 0",
                           "malformed forced BOS token message");

    auto negative_temperature =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--temperature", "-1"});
    check(negative_temperature.parse_error, "negative temperature parse error");
    check_message_contains(negative_temperature.error_message,
                           "--temperature expects a finite number >= 0",
                           "negative temperature message");

    auto malformed_top_k =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--top-k", "abc"});
    check(malformed_top_k.parse_error, "malformed top-k parse error");
    check_message_contains(malformed_top_k.error_message, "--top-k expects an integer >= 0",
                           "malformed top-k message");

    auto negative_top_k =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--top-k", "-1"});
    check(negative_top_k.parse_error, "negative top-k parse error");
    check_message_contains(negative_top_k.error_message, "--top-k expects an integer >= 0",
                           "negative top-k message");

    auto out_of_range_top_p =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--top-p", "5.0"});
    check(out_of_range_top_p.parse_error, "out-of-range top-p parse error");
    check_message_contains(out_of_range_top_p.error_message,
                           "--top-p expects a finite number in [0, 1]",
                           "out-of-range top-p message");

    auto out_of_range_min_p =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--min-p", "-0.1"});
    check(out_of_range_min_p.parse_error, "out-of-range min-p parse error");
    check_message_contains(out_of_range_min_p.error_message,
                           "--min-p expects a finite number in [0, 1]",
                           "out-of-range min-p message");

    auto zero_repetition_penalty =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--repetition-penalty", "0"});
    check(zero_repetition_penalty.parse_error, "zero repetition penalty parse error");
    check_message_contains(zero_repetition_penalty.error_message,
                           "--repetition-penalty expects a finite number > 0",
                           "zero repetition penalty message");

    auto nonfinite_repetition_penalty = parse(
        {"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--repetition-penalty", "nan"});
    check(nonfinite_repetition_penalty.parse_error, "non-finite repetition penalty parse error");
}

void test_generation_sampling_boundaries_parse() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello", "--max-new-tokens",
                       "1", "--temperature", "0", "--top-p", "0", "--min-p", "1",
                       "--repetition-penalty", "0.1", "--top-k", "0"});
    check(!args.parse_error, "generation sampling boundary values parse");
    check(args.max_new_tokens == 1, "boundary max tokens");
    check(args.temperature == 0.0F, "boundary temperature");
    check(args.top_p == 0.0F, "boundary top-p");
    check(args.min_p == 1.0F, "boundary min-p");
    check(args.repetition_penalty > 0.09F && args.repetition_penalty < 0.11F,
          "boundary repetition penalty");
    check(args.top_k == 0, "boundary top-k");
}

void test_unexpected_positional_fails() {
    auto args = parse({"trtmc", "run", "one.bundle", "two.bundle"});
    check(args.parse_error, "unexpected positional parse error");
    check(args.error_message == "Unexpected positional argument: two.bundle",
          "unexpected positional message");
}

void test_num_images_zero_fails() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--num-images", "0"});
    check(args.parse_error, "num-images zero parse error");
    check(args.error_message == "--num-images must be >= 1", "num-images error message");
}

void test_seed_csv_populates_seed_list() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--prompt", "x", "--num-images", "4",
                       "--seed", "0,1,2,3"});
    check(!args.parse_error, "seed csv parses cleanly");
    check(args.num_images == 4, "num-images parsed");
    check(args.seed_list.size() == 4, "seed list size");
    check(args.seed_list[0] == 0 && args.seed_list[1] == 1 && args.seed_list[2] == 2 &&
              args.seed_list[3] == 3,
          "seed list values");
}

void test_prompt_and_prompts_file_mutually_exclusive() {
    auto args =
        parse({"trtmc", "run", "bundle.bundle", "--prompt", "hi", "--prompts-file", "prompts.txt"});
    check(args.parse_error, "prompt+prompts-file parse error");
    check(args.error_message == "--prompt and --prompts-file are mutually exclusive",
          "prompt+prompts-file error message");
}

void test_prompts_file_is_run_input_source() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--prompts-file", "prompts.txt"});
    check(!args.parse_error, "prompts-file run parses cleanly");
    check(trtmc::cli::has_run_input_source(args), "prompts-file satisfies run input guard");
    check(trtmc::cli::text_stdout_requires_jsonl(args, 1),
          "one-row prompts-file keeps stable JSONL output");

    auto prompt = parse({"trtmc", "run", "bundle.bundle", "--prompt", "hello"});
    check(!trtmc::cli::text_stdout_requires_jsonl(prompt, 1),
          "one scalar prompt keeps plain-text output");
    check(trtmc::cli::text_stdout_requires_jsonl(prompt, 2),
          "multiple scalar samples use JSONL output");
}

void test_prompts_file_rejects_single_image_input() {
    auto args = parse(
        {"trtmc", "run", "bundle.bundle", "--prompts-file", "prompts.txt", "--image", "image.png"});
    check(args.parse_error, "prompts-file plus image parse error");
    check(args.error_message == "--prompts-file cannot be combined with --image",
          "prompts-file plus image error message");
}

void test_initial_latents_are_run_input_source() {
    auto args = parse({"trtmc", "run", "bundle.bundle", "--initial-latents-raw", "latents.raw"});
    check(!args.parse_error, "initial latents run parses cleanly");
    check(trtmc::cli::has_run_input_source(args), "initial latents satisfy run input guard");
}

} // namespace

int main() {
    test_help_uses_platform_neutral_library_names();
    test_no_args_show_help();
    test_help_aliases_show_help();
    test_version_aliases();
    test_build_forwards_args_verbatim();
    test_run_parses_common_flags();
    test_diffusion_flags();
    test_generate_video_parses_native_output_contract();
    test_generate_video_parses_public_multimodal_modes();
    test_generate_video_rejects_invalid_public_multimodal_counts();
    test_detect_parses_contract_flags();
    test_extract_features_parses_contract_flags();
    test_disparity_parses_stereo_images();
    test_inspect_and_config_flags();
    test_audio_and_solve_flags();
    test_canary_transcription_flags_and_batch();
    test_unknown_command_fails();
    test_unknown_flag_fails();
    test_missing_value_fails();
    test_missing_prompt_is_distinct_from_empty_prompt();
    test_bad_kv_cache_size_fails();
    test_invalid_generation_sampling_values_fail();
    test_generation_sampling_boundaries_parse();
    test_unexpected_positional_fails();
    test_num_images_zero_fails();
    test_seed_csv_populates_seed_list();
    test_prompt_and_prompts_file_mutually_exclusive();
    test_prompts_file_is_run_input_source();
    test_prompts_file_rejects_single_image_input();
    test_initial_latents_are_run_input_source();

    if (failures) {
        std::cerr << failures << " CLI parser tests failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All CLI parser tests passed\n";
    return EXIT_SUCCESS;
}

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
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

namespace {

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
    auto args =
        parse({"trtmc", "build", "Qwen/Qwen3-0.6B", "-o", "out.trtfb", "--precision", "fp16"});
    check(args.command == "build", "build command");
    check(args.build_args.size() == 5, "build forwards arg count");
    check(args.build_args[0] == "Qwen/Qwen3-0.6B", "build forwards model");
    check(args.build_args[4] == "fp16", "build forwards final value");
}

void test_run_parses_common_flags() {
    auto args = parse({"trtmc",
                       "run",
                       "bundle.trtfb",
                       "--prompt",
                       "hello",
                       "--max-new-tokens",
                       "8",
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
                       "--top-k",
                       "4",
                       "--seed",
                       "123",
                       "--greedy",
                       "--chat-template",
                       "--no-thinking",
                       "--kv-cache-size",
                       "2GiB",
                       "--backend-dir",
                       "/tmp/lib",
                       "--runtime-cache",
                       "/tmp/cache",
                       "--cuda-graphs"});
    check(args.command == "run", "run command");
    check(args.bundle_path == "bundle.trtfb", "run bundle");
    check(args.prompt_provided, "run prompt provided");
    check(args.prompt == "hello", "run prompt");
    check(args.max_new_tokens == 8, "run max tokens");
    check(args.generation_mode == "diffusion", "run generation mode");
    check(args.block_length == 32, "run block length");
    check(args.conf_threshold > 0.89F && args.conf_threshold < 0.91F, "run threshold");
    check(args.temperature > 0.49F && args.temperature < 0.51F, "run temperature");
    check(args.top_k == 4, "run top_k");
    check(args.seed == 123, "run seed");
    check(args.greedy, "run greedy");
    check(args.chat_template, "run chat template");
    check(args.no_thinking, "run no thinking");
    check(args.kv_cache_size_bytes == 2147483648ULL, "run kv cache size");
    check(args.backend_search_paths.size() == 1 && args.backend_search_paths[0] == "/tmp/lib",
          "run backend dir");
    check(args.runtime_cache == "/tmp/cache", "run runtime cache");
    check(args.cuda_graphs, "run cuda graphs");
}

void test_diffusion_flags() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--prompt", "paint", "--negative-prompt",
                       "blur", "--num-inference-steps", "20", "--height", "512", "--width", "768",
                       "--cfg-scale", "7.5", "--initial-latents-raw", "latents.raw"});
    check(args.negative_prompt == "blur", "diffusion negative prompt");
    check(args.num_steps == 20, "diffusion num steps");
    check(args.diffusion_height == 512, "diffusion height");
    check(args.diffusion_width == 768, "diffusion width");
    check(args.cfg_scale > 7.49F && args.cfg_scale < 7.51F, "diffusion cfg scale");
    check(args.initial_latents_raw == "latents.raw", "diffusion latents");
}

void test_detect_parses_contract_flags() {
    auto args = parse({"trtmc", "detect", "bundle.trtfb", "--image", "img.jpg", "--output-json",
                       "detections.json", "--score-threshold", "0.42"});
    check(args.command == "detect", "detect command");
    check(args.bundle_path == "bundle.trtfb", "detect bundle");
    check(args.image_path == "img.jpg", "detect image");
    check(args.output_json == "detections.json", "detect output json");
    check(args.conf_threshold > 0.41F && args.conf_threshold < 0.43F, "detect threshold");
}

void test_inspect_and_config_flags() {
    auto args = parse({"trtmc", "inspect", "bundle.trtfb", "--list-engines", "--config",
                       "profile.json", "--set", "audio.seed=7"});
    check(args.command == "inspect", "inspect command");
    check(args.bundle_path == "bundle.trtfb", "inspect bundle");
    check(args.list_engines, "inspect list engines");
    check(args.config_path == "profile.json", "config path");
    check(args.set_tokens.size() == 1 && args.set_tokens[0] == "audio.seed=7", "set token");
}

void test_audio_and_solve_flags() {
    auto transcribe =
        parse({"trtmc", "transcribe", "bundle.trtfb", "--audio", "input.wav", "--stream",
               "--chunk-ms", "80", "--att-context-size", "5,2", "--pad-and-drop-preencoded"});
    check(transcribe.audio_in == "input.wav", "transcribe audio");
    check(transcribe.stream, "transcribe stream");
    check(transcribe.chunk_ms == 80, "transcribe chunk ms");
    check(transcribe.att_context_left == 5 && transcribe.att_context_right == 2,
          "transcribe att context");
    check(transcribe.pad_and_drop_preencoded, "transcribe pad/drop");

    auto solve =
        parse({"trtmc", "solve", "bundle.trtfb", "--branch-input", "1,2", "--trunk-input", "3,4"});
    check(solve.branch_input == "1,2", "solve branch");
    check(solve.trunk_input == "3,4", "solve trunk");
}

void test_unknown_command_fails() {
    auto args = parse({"trtmc", "missing"});
    check(args.parse_error, "unknown command parse error");
    check(args.error_message == "Unknown command: missing", "unknown command message");
}

void test_unknown_flag_fails() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--bogus"});
    check(args.parse_error, "unknown flag parse error");
    check(args.error_message == "Unknown flag: --bogus", "unknown flag message");
}

void test_missing_value_fails() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--prompt"});
    check(args.parse_error, "missing value parse error");
    check(args.error_message == "--prompt requires a value", "missing value message");
}

void test_missing_prompt_is_distinct_from_empty_prompt() {
    auto missing = parse({"trtmc", "run", "bundle.trtfb", "--max-new-tokens", "8"});
    check(missing.parse_error, "missing prompt parse error");
    check(missing.error_message == "run requires bundle + --prompt", "missing prompt message");
    check(!missing.prompt_provided, "missing prompt not provided");
    check(missing.prompt.empty(), "missing prompt text empty");

    auto empty = parse({"trtmc", "run", "bundle.trtfb", "--prompt", "", "--max-new-tokens", "8"});
    check(empty.prompt_provided, "empty prompt provided");
    check(empty.prompt.empty(), "empty prompt text empty");
    check(!empty.parse_error, "empty prompt parse ok");
}

void test_bad_kv_cache_size_fails() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--kv-cache-size=abc"});
    check(args.parse_error, "bad kv cache parse error");
    check(args.error_message.find("--kv-cache-size expects") == 0, "bad kv cache message");
}

void test_unexpected_positional_fails() {
    auto args = parse({"trtmc", "run", "one.trtfb", "two.trtfb"});
    check(args.parse_error, "unexpected positional parse error");
    check(args.error_message == "Unexpected positional argument: two.trtfb",
          "unexpected positional message");
}

} // namespace

int main() {
    test_no_args_show_help();
    test_help_aliases_show_help();
    test_version_aliases();
    test_build_forwards_args_verbatim();
    test_run_parses_common_flags();
    test_diffusion_flags();
    test_detect_parses_contract_flags();
    test_inspect_and_config_flags();
    test_audio_and_solve_flags();
    test_unknown_command_fails();
    test_unknown_flag_fails();
    test_missing_value_fails();
    test_missing_prompt_is_distinct_from_empty_prompt();
    test_bad_kv_cache_size_fails();
    test_unexpected_positional_fails();

    if (failures) {
        std::cerr << failures << " CLI parser tests failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All CLI parser tests passed\n";
    return EXIT_SUCCESS;
}

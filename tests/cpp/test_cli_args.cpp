// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CLI-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         CLI argument parsing for build/run/inspect/version commands
// Preconditions:  None
// Postconditions: Parsed args match expected command and options
// =============================================================================

// =============================================================================
// Test suite: CLI argument parsing for the `trtmc` command-line interface.
//
// Purpose:
//   Validates the CLI argument parser that powers the `trtmc` executable. The
//   parser handles subcommands (build, run, detect, inspect, version, help),
//   positional arguments (bundle path), and option flags (--prompt,
//   --max-new-tokens, --hf-python, detection aliases).
//
// Dependencies:
//   - trtmc/pipeline.h: only for basic type references. No GPU, TRT, or
//     filesystem access required.
//
// Approach:
//   The production CLI parser lives inside trtmc_cli.cpp, which has its own
//   main(). To test in isolation without linking two main() symbols, this file
//   replicates the CliArgs struct and parse_args() function matching the
//   simplified production code. A convenience wrapper parse(vector<const char*>)
//   converts a brace-init list to argc/argv for concise test invocations.
//
//   Each test function simulates a specific command-line invocation, parses it,
//   and asserts that the resulting CliArgs fields match expected values.
//
// Test categories:
//   - Subcommand parsing: run, detect, inspect, version, help
//   - Flag handling: --prompt, --max-new-tokens, --hf-python, detection aliases
//   - Error handling: unknown flags, unknown commands
//   - No-args: bare invocation shows help
// =============================================================================

#include "trtmc/pipeline.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
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

struct CliArgs {
    std::string command;
    std::vector<std::string> build_args;
    std::string model_or_bundle;
    std::string prompt;
    std::string hf_python;
    std::uint64_t kv_cache_size_bytes{0};
    std::string image_path;
    std::string output_path;
    std::string initial_latents_raw;
    std::string condition_latents_raw;
    std::string condition_mask_raw;
    std::string sampling_steps_raw;
    std::string sde_noise_raw;
    int max_new_tokens{0};
    int num_samples{1};
    int num_steps{-1};
    float guidance_scale{-1.0F};
    float cfg_scale{-1.0F};
    float sde_gamma{-1.0F};
    float conf_threshold{-1.0F};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
};

std::optional<std::uint64_t> parse_byte_size(const std::string& text) {
    if (text.empty())
        return std::nullopt;

    std::size_t value_end = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &value_end);
    } catch (...) {
        return std::nullopt;
    }
    if (value <= 0.0)
        return std::nullopt;

    std::string suffix = text.substr(value_end);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    long double multiplier = 1.0L;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1.0L;
    } else if (suffix == "K" || suffix == "KB") {
        multiplier = 1000.0L;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1000.0L * 1000.0L;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1000.0L * 1000.0L * 1000.0L;
    } else if (suffix == "T" || suffix == "TB") {
        multiplier = 1000.0L * 1000.0L * 1000.0L * 1000.0L;
    } else if (suffix == "KIB") {
        multiplier = 1024.0L;
    } else if (suffix == "MIB") {
        multiplier = 1024.0L * 1024.0L;
    } else if (suffix == "GIB") {
        multiplier = 1024.0L * 1024.0L * 1024.0L;
    } else if (suffix == "TIB") {
        multiplier = 1024.0L * 1024.0L * 1024.0L * 1024.0L;
    } else {
        return std::nullopt;
    }

    const long double bytes = static_cast<long double>(value) * multiplier;
    if (bytes <= 0.0L ||
        bytes > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        return std::nullopt;
    return static_cast<std::uint64_t>(bytes + 0.5L);
}

CliArgs parse_args(int argc, const char** argv) {
    CliArgs args;

    if (argc < 2) {
        args.show_help = true;
        return args;
    }

    args.command = argv[1];

    if (args.command == "version" || args.command == "--version" || args.command == "-v") {
        args.command = "version";
        return args;
    }

    if (args.command == "help" || args.command == "--help" || args.command == "-h") {
        args.show_help = true;
        return args;
    }

    if (args.command == "build") {
        for (int i = 2; i < argc; ++i)
            args.build_args.emplace_back(argv[i]);
        return args;
    }

    if (args.command != "run" && args.command != "inspect" && args.command != "detect") {
        args.parse_error = true;
        args.error_message = "Unknown command: " + args.command;
        return args;
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--prompt" || arg == "-p") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.prompt = argv[++i];
            continue;
        }
        if (arg == "--max-new-tokens") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.max_new_tokens = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--num-samples") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.num_samples = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--num-steps") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.num_steps = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--guidance-scale") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.guidance_scale = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--cfg-scale") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.cfg_scale = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--sde-gamma") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.sde_gamma = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--hf-python") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.hf_python = argv[++i];
            continue;
        }
        if (arg == "--kv-cache-size" || arg == "--kv_cache_size") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            auto parsed = parse_byte_size(argv[++i]);
            if (!parsed.has_value()) {
                args.parse_error = true;
                args.error_message = "--kv-cache-size expects a positive size";
                return args;
            }
            args.kv_cache_size_bytes = *parsed;
            continue;
        }
        if (arg.rfind("--kv-cache-size=", 0) == 0 || arg.rfind("--kv_cache_size=", 0) == 0) {
            auto parsed = parse_byte_size(arg.substr(arg.find('=') + 1));
            if (!parsed.has_value()) {
                args.parse_error = true;
                args.error_message = "--kv-cache-size expects a positive size";
                return args;
            }
            args.kv_cache_size_bytes = *parsed;
            continue;
        }
        if (arg == "--image") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.image_path = argv[++i];
            continue;
        }
        if (arg == "--output" || arg == "--output-json" || arg == "-o") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.output_path = argv[++i];
            continue;
        }
        if (arg == "--condition-latents-raw") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.condition_latents_raw = argv[++i];
            continue;
        }
        if (arg == "--initial-latents-raw") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.initial_latents_raw = argv[++i];
            continue;
        }
        if (arg == "--condition-mask-raw") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.condition_mask_raw = argv[++i];
            continue;
        }
        if (arg == "--sampling-steps-raw") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.sampling_steps_raw = argv[++i];
            continue;
        }
        if (arg == "--sde-noise-raw") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.sde_noise_raw = argv[++i];
            continue;
        }
        if (arg == "--threshold" || arg == "--score-threshold") {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = arg + " requires a value";
                return args;
            }
            args.conf_threshold = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg[0] == '-') {
            args.parse_error = true;
            args.error_message = "Unknown flag: " + arg;
            return args;
        }

        if (args.model_or_bundle.empty()) {
            args.model_or_bundle = arg;
        } else {
            args.parse_error = true;
            args.error_message = "Unexpected positional argument: " + arg;
            return args;
        }
    }

    return args;
}

CliArgs parse(std::vector<const char*> argv_vec) {
    return parse_args(static_cast<int>(argv_vec.size()), argv_vec.data());
}

} // namespace

// -----------------------------------------------------------------------------
// Intention: Verify that "trtmc run bundle.trtfb --prompt 'hello world'"
//   correctly parses the run subcommand and captures the prompt string.
// Setup: Simulated argv with "run", a bundle path, and a multi-word prompt.
// Mechanism: Calls parse(), checks command=="run", model_or_bundle=="bundle.trtfb",
//   and prompt=="hello world".
// -----------------------------------------------------------------------------
static void test_run_with_prompt() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--prompt", "hello world"});
    check(args.command == "run", "run command");
    check(args.model_or_bundle == "bundle.trtfb", "run bundle path");
    check(args.prompt == "hello world", "run prompt");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --max-new-tokens is parsed as an integer for the run
//   subcommand.
// Setup: Simulated argv with "run" + "--max-new-tokens 50".
// Mechanism: Calls parse(), checks max_new_tokens==50.
// -----------------------------------------------------------------------------
static void test_run_max_tokens() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--prompt", "hi", "--max-new-tokens", "50"});
    check(args.max_new_tokens == 50, "run max_new_tokens");
}

// -----------------------------------------------------------------------------
// Intention: Verify that --hf-python correctly captures the path to a Python
//   interpreter, used for the HuggingFace tokenizer bridge.
// Setup: Simulated argv with "run" + "--hf-python /usr/bin/python3".
// Mechanism: Calls parse(), checks no parse error and hf_python=="/usr/bin/python3".
// -----------------------------------------------------------------------------
static void test_hf_python_flag() {
    auto args = parse(
        {"trtmc", "run", "bundle.trtfb", "--prompt", "hi", "--hf-python", "/usr/bin/python3"});
    check(!args.parse_error, "hf-python no parse error");
    check(args.hf_python == "/usr/bin/python3", "hf-python value");
}

// -----------------------------------------------------------------------------
// Intention: Verify that the "inspect" subcommand correctly parses and captures
//   the positional bundle file path.
// Setup: Simulated argv: {"trtmc", "inspect", "file.trtfb"}.
// Mechanism: Calls parse(), checks command=="inspect" and
//   model_or_bundle=="file.trtfb".
// -----------------------------------------------------------------------------
static void test_inspect_subcommand() {
    auto args = parse({"trtmc", "inspect", "file.trtfb"});
    check(args.command == "inspect", "inspect command");
    check(args.model_or_bundle == "file.trtfb", "inspect file path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that invoking "trtmc" with no arguments sets show_help=true
//   (the expected behavior for a bare invocation with no subcommand).
// Setup: Simulated argv: {"trtmc"} (argc==1, no subcommand).
// Mechanism: Calls parse(), checks show_help==true.
// -----------------------------------------------------------------------------
static void test_no_args_shows_usage() {
    auto args = parse({"trtmc"});
    check(args.show_help, "no args shows help");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtmc --help" sets show_help=true.
// Setup: Simulated argv: {"trtmc", "--help"}.
// Mechanism: Calls parse(), checks show_help==true.
// -----------------------------------------------------------------------------
static void test_help_flag() {
    auto args = parse({"trtmc", "--help"});
    check(args.show_help, "--help shows help");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtmc version" parses the version subcommand.
// Setup: Simulated argv: {"trtmc", "version"}.
// Mechanism: Calls parse(), checks command=="version".
// -----------------------------------------------------------------------------
static void test_version_subcommand() {
    auto args = parse({"trtmc", "version"});
    check(args.command == "version", "version command");
}

// -----------------------------------------------------------------------------
// Intention: Verify that "trtmc build" preserves builder arguments verbatim for
//   the Python builder bridge.
// Setup: Simulated argv with build + model/output/cache flags.
// Mechanism: Calls parse(), checks command=="build" and all trailing tokens were
//   captured without validation by the C++ runtime parser.
// -----------------------------------------------------------------------------
static void test_build_forwards_args() {
    auto args = parse({"trtmc", "build", "Qwen/Qwen3-0.6B", "-o", "/tmp/qwen.trtfb",
                       "--max-cache-length", "512"});
    check(!args.parse_error, "build no parse error");
    check(args.command == "build", "build command");
    check(args.build_args.size() == 5, "build arg count");
    check(args.build_args[0] == "Qwen/Qwen3-0.6B", "build model arg");
    check(args.build_args[1] == "-o", "build output flag");
    check(args.build_args[2] == "/tmp/qwen.trtfb", "build output path");
    check(args.build_args[3] == "--max-cache-length", "build cache flag");
    check(args.build_args[4] == "512", "build cache value");
}

// -----------------------------------------------------------------------------
// Intention: Verify that an unknown flag (e.g., --bogus) is rejected with a
//   parse error whose message mentions the offending flag.
// Setup: Simulated argv with "run" + "--bogus".
// Mechanism: Calls parse(), checks parse_error==true and error_message contains
//   "--bogus".
// -----------------------------------------------------------------------------
static void test_unknown_flag_errors() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--bogus"});
    check(args.parse_error, "unknown flag causes error");
    check(args.error_message.find("--bogus") != std::string::npos, "error message mentions flag");
}

// -----------------------------------------------------------------------------
// Intention: Verify that an unknown subcommand (e.g., "foobar") is rejected
//   with a parse error whose message mentions the unknown command name.
// Setup: Simulated argv: {"trtmc", "foobar"}.
// Mechanism: Calls parse(), checks parse_error==true and error_message contains
//   "foobar".
// -----------------------------------------------------------------------------
static void test_unknown_command_errors() {
    auto args = parse({"trtmc", "foobar"});
    check(args.parse_error, "unknown command causes error");
    check(args.error_message.find("foobar") != std::string::npos, "error message mentions command");
}

// -----------------------------------------------------------------------------
// Intention: Verify that all supported run flags can be combined in a single
//   invocation without conflicts or parse errors.
// Setup: Simulated argv with "run" and every supported flag.
// Mechanism: Calls parse(), checks every field matches the expected value
//   and no parse error occurred.
// -----------------------------------------------------------------------------
static void test_all_run_flags_combined() {
    auto args = parse({"trtmc",
                       "run",
                       "bundle.trtfb",
                       "--prompt",
                       "hello",
                       "--max-new-tokens",
                       "10",
                       "--num-samples",
                       "3",
                       "--num-steps",
                       "32",
                       "--guidance-scale",
                       "3.0",
                       "--cfg-scale",
                       "2.0",
                       "--sde-gamma",
                       "1.5",
                       "--initial-latents-raw",
                       "z.bin",
                       "--condition-latents-raw",
                       "cond.bin",
                       "--condition-mask-raw",
                       "mask.bin",
                       "--sampling-steps-raw",
                       "steps.bin",
                       "--sde-noise-raw",
                       "eps.bin",
                       "--hf-python",
                       "/usr/bin/python3",
                       "--kv-cache-size",
                       "2GiB"});
    check(!args.parse_error, "combined flags no parse error");
    check(args.model_or_bundle == "bundle.trtfb", "combined bundle path");
    check(args.prompt == "hello", "combined prompt");
    check(args.max_new_tokens == 10, "combined max_new_tokens");
    check(args.num_samples == 3, "combined num_samples");
    check(args.num_steps == 32, "combined num_steps");
    check(args.guidance_scale == 3.0F, "combined guidance_scale");
    check(args.cfg_scale == 2.0F, "combined cfg_scale");
    check(args.sde_gamma == 1.5F, "combined sde_gamma");
    check(args.initial_latents_raw == "z.bin", "combined initial_latents_raw");
    check(args.condition_latents_raw == "cond.bin", "combined condition_latents_raw");
    check(args.condition_mask_raw == "mask.bin", "combined condition_mask_raw");
    check(args.sampling_steps_raw == "steps.bin", "combined sampling_steps_raw");
    check(args.sde_noise_raw == "eps.bin", "combined sde_noise_raw");
    check(args.hf_python == "/usr/bin/python3", "combined hf-python");
    check(args.kv_cache_size_bytes == (2ULL * 1024ULL * 1024ULL * 1024ULL),
          "combined kv-cache-size");
}

static void test_kv_cache_size_flag() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--kv-cache-size", "90GB"});
    check(!args.parse_error, "kv-cache-size no parse error");
    check(args.kv_cache_size_bytes == 90000000000ULL, "kv-cache-size parsed");
}

static void test_kv_cache_size_alias_flag() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--kv_cache_size", "90GB"});
    check(!args.parse_error, "kv_cache_size alias no parse error");
    check(args.kv_cache_size_bytes == 90000000000ULL, "kv_cache_size alias parsed");
}

static void test_kv_cache_size_equals_flag() {
    auto args = parse({"trtmc", "run", "bundle.trtfb", "--kv-cache-size=90GB"});
    check(!args.parse_error, "kv-cache-size equals no parse error");
    check(args.kv_cache_size_bytes == 90000000000ULL, "kv-cache-size equals parsed");
}

// -----------------------------------------------------------------------------
// Intention: Verify detect alias flags parse exactly like canonical names.
// Setup: Simulated argv with detect + --output-json + --score-threshold.
// Mechanism: Calls parse(), checks parsed command and values.
// -----------------------------------------------------------------------------
static void test_detect_alias_flags() {
    auto args = parse({"trtmc", "detect", "bundle.trtfb", "--image", "img.jpg", "--output-json",
                       "det.json", "--score-threshold", "0.25"});
    check(!args.parse_error, "detect aliases no parse error");
    check(args.command == "detect", "detect command");
    check(args.image_path == "img.jpg", "detect image path");
    check(args.output_path == "det.json", "detect output path");
    check(args.conf_threshold == 0.25F, "detect threshold");
}

// -----------------------------------------------------------------------------
// Intention: Verify strict unknown-flag behavior is preserved after adding
//   alias support for known detection flags.
// Setup: Simulated argv with detect + aliases + unknown flag.
// Mechanism: Calls parse(), checks parse_error and unknown flag message.
// -----------------------------------------------------------------------------
static void test_detect_unknown_flag_still_errors() {
    auto args = parse({"trtmc", "detect", "bundle.trtfb", "--image", "img.jpg", "--output-json",
                       "det.json", "--score-threshold", "0.25", "--not-a-real-flag"});
    check(args.parse_error, "detect unknown flag causes error");
    check(args.error_message.find("--not-a-real-flag") != std::string::npos,
          "detect unknown flag message mentions flag");
}

int main() {
    test_run_with_prompt();
    test_run_max_tokens();
    test_hf_python_flag();
    test_inspect_subcommand();
    test_no_args_shows_usage();
    test_help_flag();
    test_version_subcommand();
    test_build_forwards_args();
    test_unknown_flag_errors();
    test_unknown_command_errors();
    test_all_run_flags_combined();
    test_kv_cache_size_flag();
    test_kv_cache_size_alias_flag();
    test_kv_cache_size_equals_flag();
    test_detect_alias_flags();
    test_detect_unknown_flag_still_errors();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All cli_args tests passed.\n";
    return 0;
}

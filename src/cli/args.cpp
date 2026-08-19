/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/args.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace trtmc::cli {

namespace {

bool has_ascii_space(const char* text) {
    if (!text)
        return false;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
        if (std::isspace(*p))
            return true;
    }
    return false;
}

bool parse_strict_int(const char* text, int& out) {
    if (!text || *text == '\0' || has_ascii_space(text))
        return false;

    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value < static_cast<long>(std::numeric_limits<int>::min()) ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }

    out = static_cast<int>(value);
    return true;
}

bool parse_strict_float(const char* text, float& out) {
    if (!text || *text == '\0' || has_ascii_space(text))
        return false;

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value > static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
    }

    out = static_cast<float>(value);
    return true;
}

} // namespace

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
        bytes > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(bytes + 0.5L);
}

std::optional<std::vector<std::uint64_t>> parse_seed_csv(const std::string& text) {
    std::vector<std::uint64_t> out;
    if (text.empty())
        return out;
    std::string token;
    auto flush = [&]() -> bool {
        if (token.empty())
            return false;
        // Trim incidental whitespace ("0, 1, 2" is friendlier than "0,1,2").
        std::size_t begin = 0;
        std::size_t end = token.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(token[begin])))
            ++begin;
        while (end > begin && std::isspace(static_cast<unsigned char>(token[end - 1])))
            --end;
        if (begin == end)
            return false;
        try {
            std::size_t consumed = 0;
            const std::string slice = token.substr(begin, end - begin);
            const unsigned long long value = std::stoull(slice, &consumed, 10);
            if (consumed != slice.size())
                return false;
            out.push_back(static_cast<std::uint64_t>(value));
        } catch (...) {
            return false;
        }
        token.clear();
        return true;
    };

    for (char ch : text) {
        if (ch == ',') {
            if (!flush())
                return std::nullopt;
        } else {
            token.push_back(ch);
        }
    }
    if (!flush())
        return std::nullopt;
    return out;
}

std::vector<std::string> read_prompts_file(const std::string& path, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "failed to open prompts file: " + path;
        return {};
    }
    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(in, line)) {
        // Strip trailing CR for files written on Windows.
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        prompts.push_back(line);
    }
    // Drop a trailing blank line introduced by a final newline; keep
    // any deliberate blank prompts the user typed in the middle.
    while (!prompts.empty() && prompts.back().empty())
        prompts.pop_back();
    if (prompts.empty()) {
        error = "prompts file is empty: " + path;
        return {};
    }
    return prompts;
}

void print_usage() {
    std::cerr
        << "Usage:\n"
           "  trtmc build           <hf-model-or-dir> -o <bundle.bundle> [builder args...]\n"
           "  trtmc graph           <inspect|list|recipes|select> [args...]\n"
           "  trtmc run             <bundle.bundle> "
           "(--prompt \"text\" [--image PATH] | --prompts-file PATH) "
           "[--max-new-tokens N] [--temperature F] [--top-p F] [--min-p F] "
           "[--repetition-penalty F] "
           "[--source-language-token-id N] [--forced-bos-token-id N] "
           "[--top-k N] [--seed N] [--benchmark N] [--warmup N] [--hf-python PATH] "
           "[--lora-adapter DIR] [--lora-adapter-id ID] "
           "[--kv-cache-size SIZE] [--chat-template] [--no-thinking] "
           "[--generation-mode MODE] [--block-length N] [--threshold F] "
           "[--num-samples N] [--num-steps N] [--guidance-scale S] [--cfg-scale S] "
           "[--sde-gamma S] [--initial-latents-raw PATH] [--condition-latents-raw PATH] "
           "[--condition-mask-raw PATH] [--sampling-steps-raw PATH] [--sde-noise-raw PATH] "
           "[--output samples.jsonl]\n"
           "                        Image-generation extras: [--negative-prompt \"text\"] "
           "[--num-inference-steps N] [--height N] [--width N] "
           "[--num-images N] [--prompts-file PATH] [--seed s0,s1,...]\n"
           "  trtmc encode          <bundle.bundle> --prompt \"text\" [--hf-python PATH]\n"
           "  trtmc segment         <bundle.bundle> --image PATH --output PATH [--hf-python PATH]\n"
           "  trtmc disparity       <bundle.bundle> --image LEFT --right-image RIGHT "
           "--output PATH\n"
           "  trtmc segment-prompted <bundle.bundle> --image PATH --output DIR "
           "[--point-x F] [--point-y F] [--background] [--prompt TEXT] [--hf-python PATH]\n"
           "  trtmc classify        <bundle.bundle> --image PATH [--benchmark N] [--warmup N]\n"
           "  trtmc extract-features <bundle.bundle> --image PATH [--output-json PATH]\n"
           "  trtmc detect          <bundle.bundle> --image PATH [--output-json PATH] "
           "[--score-threshold F]\n"
           "  trtmc generate-audio  <bundle.bundle> --prompt \"text\" --output PATH "
           "[--max-new-tokens N] [--hf-python PATH]\n"
           "  trtmc serve-audio     <bundle.bundle> [--chunk-frames N] [--max-new-tokens N] "
           "[--hf-python PATH]\n"
           "                       Loads bundle once, reads prompts from stdin, streams PCM to "
           "stdout.\n"
           "  trtmc generate-video  <bundle.bundle> --prompt \"text\" --output DIR [--num-steps N] "
           "[--guidance-scale S] [--initial-latents-raw PATH]\n"
           "                        [--negative-prompt \"text\"] [--height N] [--width N]\n"
           "  trtmc embed           <bundle.bundle> --prompt \"text\" [--hf-python PATH]\n"
           "  trtmc rerank          <bundle.bundle> --prompt \"query\" --document \"text\" "
           "[--hf-python PATH]\n"
           "  trtmc solve           <bundle.bundle> --field-input CSV\n"
           "  trtmc solve           <bundle.bundle> --branch-input CSV [--trunk-input CSV]\n"
           "  trtmc transcribe      <bundle.bundle> --audio FILE.wav [--max-new-tokens N] "
           "[--beam-size N] [--beam-fallback-max-size N] [--length-penalty F] [--language TAG] "
           "[--source-language TAG] [--target-language TAG] "
           "[--task transcribe|translate] [--punctuation|--no-punctuation] [--timestamps] "
           "[--max-input-seconds F] [--segment-length-seconds F] [--segment-min-seconds F] "
           "[--segment-overlap-seconds F] [--lcs-merge] "
           "[--stream] [--chunk-ms N] [--att-context-size L,R] "
           "[--pad-and-drop-preencoded] [--hf-python PATH]\n"
           "  trtmc speak           <bundle.bundle> --audio-in INPUT.wav --audio-out OUTPUT.wav\n"
           "  trtmc inspect         <bundle.bundle> [--list-engines]\n"
           "  trtmc version\n"
           "\n"
           "Options:\n"
           "  --backend-dir PATH    Extra directory to search for libtrtmc_backend_*.so\n"
           "  --model-plugin-dir PATH\n"
           "                        Extra directory to search for libtrtmc_model_*.so\n"
           "  --runtime-cache PATH   TRT-RTX JIT kernel cache file (speeds up repeat runs)\n"
           "  --kernel-bindings PATH Bind slot-ready engines to TVM-FFI kernels at load time\n"
           "  --cuda-graphs          Enable TRT-RTX CUDA graph capture (reduces launch overhead)\n"
           "\n"
           "Build uses a sibling python3/python when installed in an environment bin "
           "directory, otherwise python3 from PATH.\n";
}

CliArgs parse_args(int argc, char** argv) {
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

    if (args.command == "build" || args.command == "graph") {
        for (int i = 2; i < argc; ++i)
            args.build_args.emplace_back(argv[i]);
        return args;
    }

    static const char* known_cmds[] = {
        "run",         "inspect",    "generate-video", "segment",          "segment-prompted",
        "disparity",   "classify",   "detect",         "extract-features", "generate-audio",
        "serve-audio", "encode",     "embed",          "rerank",           "solve",
        "speak",       "transcribe", nullptr};
    bool valid = false;
    for (const char** p = known_cmds; *p; ++p)
        if (args.command == *p) {
            valid = true;
            break;
        }
    if (!valid) {
        args.parse_error = true;
        args.error_message = "Unknown command: " + args.command;
        return args;
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value = [&](const std::string& name) -> bool {
            if (i + 1 >= argc) {
                args.parse_error = true;
                args.error_message = name + " requires a value";
                return false;
            }
            return true;
        };

        auto parse_int_value = [&](const std::string& name, const std::string& expectation) {
            int value = 0;
            if (!parse_strict_int(argv[++i], value)) {
                args.parse_error = true;
                args.error_message = name + " expects " + expectation;
                return std::optional<int>{};
            }
            return std::optional<int>{value};
        };

        auto parse_float_value = [&](const std::string& name, const std::string& expectation) {
            float value = 0.0F;
            if (!parse_strict_float(argv[++i], value)) {
                args.parse_error = true;
                args.error_message = name + " expects " + expectation;
                return std::optional<float>{};
            }
            return std::optional<float>{value};
        };

        if ((arg == "--prompt" || arg == "-p") && need_value(arg)) {
            args.prompt = argv[++i];
            args.prompt_provided = true;
            if (!args.prompts_file.empty()) {
                args.parse_error = true;
                args.error_message = "--prompt and --prompts-file are mutually exclusive";
                return args;
            }
            continue;
        }
        if (arg == "--prompts-file" && need_value(arg)) {
            args.prompts_file = argv[++i];
            if (args.prompt_provided) {
                args.parse_error = true;
                args.error_message = "--prompt and --prompts-file are mutually exclusive";
                return args;
            }
            continue;
        }
        if (arg == "--num-images" && need_value(arg)) {
            const int n = std::atoi(argv[++i]);
            if (n < 1) {
                args.parse_error = true;
                args.error_message = "--num-images must be >= 1";
                return args;
            }
            args.num_images = n;
            continue;
        }
        if (arg == "--max-new-tokens" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer > 0");
            if (!value || *value <= 0) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer > 0";
                return args;
            }
            args.max_new_tokens = *value;
            continue;
        }
        if (arg == "--source-language-token-id" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer >= 0");
            if (!value || *value < 0) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer >= 0";
                return args;
            }
            args.source_language_token_id = *value;
            continue;
        }
        if (arg == "--forced-bos-token-id" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer >= 0");
            if (!value || *value < 0) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer >= 0";
                return args;
            }
            args.forced_bos_token_id = *value;
            continue;
        }
        if (arg == "--block-length" && need_value(arg)) {
            args.block_length = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--num-samples" && need_value(arg)) {
            args.num_samples = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--benchmark" && need_value(arg)) {
            args.benchmark = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--warmup" && need_value(arg)) {
            args.warmup = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--temperature" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number >= 0");
            if (!value || *value < 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number >= 0";
                return args;
            }
            args.temperature = *value;
            continue;
        }
        if (arg == "--top-p" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number in [0, 1]");
            if (!value || *value < 0.0F || *value > 1.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number in [0, 1]";
                return args;
            }
            args.top_p = *value;
            continue;
        }
        if (arg == "--min-p" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number in [0, 1]");
            if (!value || *value < 0.0F || *value > 1.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number in [0, 1]";
                return args;
            }
            args.min_p = *value;
            continue;
        }
        if (arg == "--repetition-penalty" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number > 0");
            if (!value || *value <= 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number > 0";
                return args;
            }
            args.repetition_penalty = *value;
            continue;
        }
        if (arg == "--top-k" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer >= 0");
            if (!value || *value < 0) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer >= 0";
                return args;
            }
            args.top_k = *value;
            continue;
        }
        if (arg == "--seed" && need_value(arg)) {
            const std::string value = argv[++i];
            if (value.find(',') != std::string::npos) {
                auto parsed = parse_seed_csv(value);
                if (!parsed.has_value() || parsed->empty()) {
                    args.parse_error = true;
                    args.error_message = "--seed CSV must be a non-empty list of unsigned integers";
                    return args;
                }
                args.seed_list = std::move(*parsed);
            } else {
                args.seed = std::atoi(value.c_str());
            }
            continue;
        }
        if (arg == "--tail-frames" && need_value(arg)) {
            args.tail_frames = std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--hf-python" && need_value(arg)) {
            args.hf_python = argv[++i];
            continue;
        }
        if (arg == "--kv-cache-size" || arg == "--kv_cache_size") {
            if (!need_value(arg))
                return args;
            auto parsed = parse_byte_size(argv[++i]);
            if (!parsed.has_value()) {
                args.parse_error = true;
                args.error_message = "--kv-cache-size expects a positive size like 90GB or 90GiB";
                return args;
            }
            args.kv_cache_size_bytes = *parsed;
            continue;
        }
        if (arg.rfind("--kv-cache-size=", 0) == 0 || arg.rfind("--kv_cache_size=", 0) == 0) {
            const auto eq = arg.find('=');
            auto parsed = parse_byte_size(arg.substr(eq + 1));
            if (!parsed.has_value()) {
                args.parse_error = true;
                args.error_message = "--kv-cache-size expects a positive size like 90GB or 90GiB";
                return args;
            }
            args.kv_cache_size_bytes = *parsed;
            continue;
        }
        if (arg == "--image" && need_value(arg)) {
            args.image_path = argv[++i];
            continue;
        }
        if (arg == "--right-image" && need_value(arg)) {
            args.right_image_path = argv[++i];
            continue;
        }
        if (arg == "--lora-adapter" && need_value(arg)) {
            args.lora_adapter_path = argv[++i];
            continue;
        }
        if (arg == "--lora-adapter-id" && need_value(arg)) {
            args.lora_adapter_id = argv[++i];
            if (args.lora_adapter_id.empty()) {
                args.parse_error = true;
                args.error_message = "--lora-adapter-id must not be empty";
                return args;
            }
            continue;
        }
        if ((arg == "--output" || arg == "-o") && need_value(arg)) {
            args.output_dir = argv[++i];
            continue;
        }
        if (arg == "--output-json" && need_value(arg)) {
            args.output_json = argv[++i];
            continue;
        }
        if (arg == "--initial-latents-raw" && need_value(arg)) {
            args.initial_latents_raw = argv[++i];
            continue;
        }
        if (arg == "--condition-latents-raw" && need_value(arg)) {
            args.condition_latents_raw = argv[++i];
            continue;
        }
        if (arg == "--condition-mask-raw" && need_value(arg)) {
            args.condition_mask_raw = argv[++i];
            continue;
        }
        if (arg == "--sampling-steps-raw" && need_value(arg)) {
            args.sampling_steps_raw = argv[++i];
            continue;
        }
        if (arg == "--sde-noise-raw" && need_value(arg)) {
            args.sde_noise_raw = argv[++i];
            continue;
        }
        if ((arg == "--num-steps" || arg == "--num-inference-steps") && need_value(arg)) {
            args.num_steps = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--guidance-scale" && need_value(arg)) {
            args.guidance_scale = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--sde-gamma" && need_value(arg)) {
            args.sde_gamma = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--negative-prompt" && need_value(arg)) {
            args.negative_prompt = argv[++i];
            continue;
        }
        if (arg == "--height" && need_value(arg)) {
            args.diffusion_height = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--width" && need_value(arg)) {
            args.diffusion_width = std::atoi(argv[++i]);
            continue;
        }
        if ((arg == "--threshold" || arg == "--score-threshold") && need_value(arg)) {
            args.conf_threshold = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--generation-mode" && need_value(arg)) {
            args.generation_mode = argv[++i];
            continue;
        }
        if (arg == "--cfg-scale" && need_value(arg)) {
            args.cfg_scale = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--greedy") {
            args.greedy = true;
            continue;
        }
        if (arg == "--stream") {
            args.stream = true;
            continue;
        }
        if (arg == "--pad-and-drop-preencoded") {
            args.pad_and_drop_preencoded = true;
            continue;
        }
        if (arg == "--chunk-frames" && i + 1 < argc) {
            args.chunk_frames = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--chunk-ms" && need_value(arg)) {
            args.chunk_ms = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--att-context-size" && need_value(arg)) {
            const std::string value = argv[++i];
            const auto comma = value.find(',');
            if (comma == std::string::npos) {
                args.parse_error = true;
                args.error_message = "--att-context-size expects L,R";
                return args;
            }
            args.att_context_left = std::atoi(value.substr(0, comma).c_str());
            args.att_context_right = std::atoi(value.substr(comma + 1).c_str());
            continue;
        }
        if (arg == "--document" && need_value(arg)) {
            args.document = argv[++i];
            continue;
        }
        if (arg == "--field-input" && need_value(arg)) {
            args.field_input = argv[++i];
            continue;
        }
        if (arg == "--branch-input" && need_value(arg)) {
            args.branch_input = argv[++i];
            continue;
        }
        if (arg == "--trunk-input" && need_value(arg)) {
            args.trunk_input = argv[++i];
            continue;
        }
        if (arg == "--audio-in" && need_value(arg)) {
            args.audio_in = argv[++i];
            continue;
        }
        if (arg == "--audio-out" && need_value(arg)) {
            args.audio_out = argv[++i];
            continue;
        }
        if (arg == "--audio" && need_value(arg)) {
            args.audio_inputs.emplace_back(argv[++i]);
            if (args.audio_in.empty())
                args.audio_in = args.audio_inputs.back();
            continue;
        }
        if (arg == "--language" && need_value(arg)) {
            args.language = argv[++i];
            args.source_language = args.language;
            args.target_language = args.language;
            continue;
        }
        if (arg == "--beam-size" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer in [1, 32]");
            if (!value || *value < 1 || *value > 32) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer in [1, 32]";
                return args;
            }
            args.beam_size = *value;
            continue;
        }
        if (arg == "--length-penalty" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number >= 0");
            if (!value || *value < 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number >= 0";
                return args;
            }
            args.length_penalty = *value;
            continue;
        }
        if (arg == "--beam-fallback-max-size" && need_value(arg)) {
            auto value = parse_int_value(arg, "an integer in [1, 32]");
            if (!value || *value < 1 || *value > 32) {
                args.parse_error = true;
                args.error_message = arg + " expects an integer in [1, 32]";
                return args;
            }
            args.beam_fallback_max_size = *value;
            continue;
        }
        if (arg == "--source-language" && need_value(arg)) {
            args.source_language = argv[++i];
            continue;
        }
        if (arg == "--target-language" && need_value(arg)) {
            args.target_language = argv[++i];
            continue;
        }
        if (arg == "--task" && need_value(arg)) {
            args.transcription_task = argv[++i];
            if (args.transcription_task != "transcribe" && args.transcription_task != "translate") {
                args.parse_error = true;
                args.error_message = "--task expects transcribe or translate";
                return args;
            }
            continue;
        }
        if (arg == "--punctuation") {
            args.punctuation = true;
            continue;
        }
        if (arg == "--no-punctuation") {
            args.punctuation = false;
            continue;
        }
        if (arg == "--timestamps") {
            args.timestamps = true;
            continue;
        }
        if (arg == "--no-timestamps") {
            args.timestamps = false;
            continue;
        }
        if (arg == "--max-input-seconds" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number > 0");
            if (!value || *value <= 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number > 0";
                return args;
            }
            args.max_input_seconds = *value;
            continue;
        }
        if (arg == "--segment-length-seconds" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number > 0");
            if (!value || *value <= 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number > 0";
                return args;
            }
            args.segment_length_seconds = *value;
            continue;
        }
        if (arg == "--segment-min-seconds" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number > 0");
            if (!value || *value <= 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number > 0";
                return args;
            }
            args.segment_min_seconds = *value;
            continue;
        }
        if (arg == "--segment-overlap-seconds" && need_value(arg)) {
            auto value = parse_float_value(arg, "a finite number >= 0");
            if (!value || *value < 0.0F) {
                args.parse_error = true;
                args.error_message = arg + " expects a finite number >= 0";
                return args;
            }
            args.segment_overlap_seconds = *value;
            continue;
        }
        if (arg == "--lcs-merge") {
            args.lcs_merge = true;
            continue;
        }
        if (arg == "--point-x" && need_value(arg)) {
            args.point_x = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--point-y" && need_value(arg)) {
            args.point_y = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--chat-template") {
            args.chat_template = true;
            continue;
        }
        if (arg == "--no-thinking") {
            args.no_thinking = true;
            continue;
        }
        if (arg == "--background") {
            args.is_foreground = false;
            continue;
        }
        if (arg == "--runtime-cache" && need_value(arg)) {
            args.runtime_cache = argv[++i];
            continue;
        }
        if (arg == "--kernel-bindings" && need_value(arg)) {
            args.kernel_bindings_path = argv[++i];
            continue;
        }
        if (arg == "--backend-dir" && need_value(arg)) {
            args.backend_search_paths.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--model-plugin-dir" && need_value(arg)) {
            args.model_plugin_search_paths.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--cuda-graphs") {
            args.cuda_graphs = true;
            continue;
        }
        if (arg == "--list-engines") {
            args.list_engines = true;
            continue;
        }
        if (arg == "--config" && need_value(arg)) {
            args.config_path = argv[++i];
            continue;
        }
        if (arg == "--set" && need_value(arg)) {
            args.set_tokens.emplace_back(argv[++i]);
            continue;
        }

        if (args.parse_error)
            return args;

        if (arg[0] == '-') {
            args.parse_error = true;
            args.error_message = "Unknown flag: " + arg;
            return args;
        }

        if (args.bundle_path.empty())
            args.bundle_path = arg;
        else {
            args.parse_error = true;
            args.error_message = "Unexpected positional argument: " + arg;
            return args;
        }
    }

    if (args.command == "run" && !args.bundle_path.empty() && !has_run_input_source(args)) {
        args.parse_error = true;
        args.error_message =
            "run requires bundle + --prompt, --prompts-file, or --initial-latents-raw";
    }
    if (args.command == "run" && !args.prompts_file.empty() && !args.image_path.empty()) {
        args.parse_error = true;
        args.error_message = "--prompts-file cannot be combined with --image";
    }

    return args;
}

} // namespace trtmc::cli

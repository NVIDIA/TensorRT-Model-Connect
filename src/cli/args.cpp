#include "cli/args.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace trtmc::cli {

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
           "  trtmc build           <hf-model-or-dir> -o <bundle.trtfb> [builder args...]\n"
           "  trtmc run             <bundle.trtfb> --prompt \"text\" [--image PATH] "
           "[--max-new-tokens N] [--temperature F] [--top-p F] [--min-p F] "
           "[--top-k N] [--seed N] [--benchmark N] [--warmup N] [--hf-python PATH] "
           "[--kv-cache-size SIZE] [--chat-template] [--no-thinking] "
           "[--generation-mode MODE] [--block-length N] [--threshold F] "
           "[--num-samples N] [--num-steps N] [--guidance-scale S] [--cfg-scale S] "
           "[--sde-gamma S] [--initial-latents-raw PATH] [--condition-latents-raw PATH] "
           "[--condition-mask-raw PATH] [--sampling-steps-raw PATH] [--sde-noise-raw PATH] "
           "[--output samples.jsonl]\n"
           "                        Diffusion text-to-image extras (Qwen-Image, FLUX, "
           "Z-Image): [--negative-prompt \"text\"] "
           "[--num-inference-steps N] [--height N] [--width N] "
           "[--num-images N] [--prompts-file PATH] [--seed s0,s1,...]\n"
           "  trtmc encode          <bundle.trtfb> --prompt \"text\" [--hf-python PATH]\n"
           "  trtmc segment         <bundle.trtfb> --image PATH --output PATH [--hf-python PATH]\n"
           "  trtmc segment-sam     <bundle.trtfb> --image PATH --output DIR "
           "[--point-x F] [--point-y F] [--background] [--prompt TEXT] [--hf-python PATH]\n"
           "  trtmc classify        <bundle.trtfb> --image PATH [--benchmark N] [--warmup N]\n"
           "  trtmc detect          <bundle.trtfb> --image PATH [--output-json PATH] "
           "[--score-threshold F]\n"
           "  trtmc generate-audio  <bundle.trtfb> --prompt \"text\" --output PATH "
           "[--max-new-tokens N] [--hf-python PATH]\n"
           "  trtmc serve-audio     <bundle.trtfb> [--chunk-frames N] [--max-new-tokens N] "
           "[--hf-python PATH]\n"
           "                       Loads bundle once, reads prompts from stdin, streams PCM to "
           "stdout.\n"
           "  trtmc generate-video  <bundle.trtfb> --prompt \"text\" --output DIR [--num-steps N] "
           "[--guidance-scale S] [--initial-latents-raw PATH]\n"
           "  trtmc embed           <bundle.trtfb> --prompt \"text\" [--hf-python PATH]\n"
           "  trtmc rerank          <bundle.trtfb> --prompt \"query\" --document \"text\" "
           "[--hf-python PATH]\n"
           "  trtmc solve           <bundle.trtfb> --field-input CSV\n"
           "  trtmc solve           <bundle.trtfb> --branch-input CSV [--trunk-input CSV]\n"
           "  trtmc transcribe      <bundle.trtfb> --audio FILE.wav [--max-new-tokens N] "
           "[--stream] [--chunk-ms N] [--att-context-size L,R] "
           "[--pad-and-drop-preencoded] [--hf-python PATH]\n"
           "  trtmc speak           <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav\n"
           "  trtmc inspect         <bundle.trtfb> [--list-engines]\n"
           "  trtmc version\n"
           "\n"
           "Options:\n"
           "  --backend-dir PATH    Extra directory to search for libtrtmc_backend_*.so\n"
           "  --runtime-cache PATH   TRT-RTX JIT kernel cache file (speeds up repeat runs)\n"
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

    if (args.command == "build") {
        for (int i = 2; i < argc; ++i)
            args.build_args.emplace_back(argv[i]);
        return args;
    }

    static const char* known_cmds[] = {
        "run",    "inspect",        "generate-video", "segment", "segment-sam", "classify",
        "detect", "generate-audio", "serve-audio",    "encode",  "embed",       "rerank",
        "solve",  "speak",          "transcribe",     nullptr};
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
            args.max_new_tokens = std::atoi(argv[++i]);
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
            args.temperature = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--top-p" && need_value(arg)) {
            args.top_p = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--min-p" && need_value(arg)) {
            args.min_p = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--top-k" && need_value(arg)) {
            args.top_k = std::atoi(argv[++i]);
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
            args.audio_in = argv[++i];
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
        if (arg == "--backend-dir" && need_value(arg)) {
            args.backend_search_paths.emplace_back(argv[++i]);
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

    if (args.command == "run" && !args.bundle_path.empty() && !args.prompt_provided &&
        args.prompts_file.empty()) {
        args.parse_error = true;
        args.error_message = "run requires bundle + --prompt or --prompts-file";
    }

    return args;
}

} // namespace trtmc::cli

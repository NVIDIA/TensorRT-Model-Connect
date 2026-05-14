// trtmc CLI — command-line interface using the new C++ library API.
//
// Usage:
//   trtmc run             <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--benchmark N]
//                        [--warmup N] [--num-samples N] [--num-steps N]
//                        [--guidance-scale S] [--cfg-scale S] [--sde-gamma S]
//                        [--initial-latents-raw PATH] [--condition-latents-raw PATH]
//                        [--condition-mask-raw PATH] [--sampling-steps-raw PATH]
//                        [--sde-noise-raw PATH] [--output samples.jsonl] [--hf-python PATH]
//                        Diffusion text-to-image extras (Qwen-Image, FLUX, Z-Image):
//                        [--negative-prompt "text"] [--num-inference-steps N]
//                        [--height N] [--width N]
//   trtmc transcribe      <bundle.trtfb> --audio FILE.wav [--max-new-tokens N] [--hf-python PATH]
//   trtmc speak           <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav
//   trtmc generate-video  <bundle.trtfb> --prompt "text" --output DIR [--num-steps N]
//   trtmc classify        <bundle.trtfb> --image PATH [--benchmark N] [--warmup N]
//   trtmc inspect         <bundle.trtfb>
//   trtmc version

#include "stb_image_write.h"
#include "trtmc/bundle.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/pipeline.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    std::string command;
    std::string bundle_path;
    std::string prompt;
    std::string hf_python;
    std::uint64_t kv_cache_size_bytes{0};
    std::string image_path;
    std::string output_dir;
    std::string initial_latents_raw;
    std::string condition_latents_raw;
    std::string condition_mask_raw;
    std::string sampling_steps_raw;
    std::string sde_noise_raw;
    std::string document;
    std::string audio_in;
    std::string audio_out;
    std::string field_input;
    std::string branch_input;
    std::string trunk_input;
    int tail_frames{0};
    float point_x{0.5F};
    float point_y{0.5F};
    bool is_foreground{true};
    int max_new_tokens{0};
    int num_samples{1};
    int benchmark{0}; // >0: run N timed iterations after warmup
    int warmup{1};    // number of warmup iterations before timing
    float temperature{1.0F};
    float top_p{1.0F};
    float min_p{0.0F};
    int top_k{1};
    int seed{-1};
    int num_steps{-1};
    float guidance_scale{-1.0F};
    float sde_gamma{-1.0F};
    float conf_threshold{-1.0F};
    float cfg_scale{-1.0F};
    // Diffusion text-to-image extras (Qwen-Image, FLUX, Z-Image, ...)
    std::string negative_prompt;
    int diffusion_height{0}; // 0 = use bundle default
    int diffusion_width{0};  // 0 = use bundle default
    bool greedy{false};
    bool stream{false};
    bool pad_and_drop_preencoded{false};
    bool chat_template{false};
    bool no_thinking{false};
    int chunk_frames{32};
    int chunk_ms{160};
    int att_context_left{70};
    int att_context_right{13};
    std::string runtime_cache;
    std::vector<std::string> backend_search_paths;
    bool cuda_graphs{false};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
    // Generic config surface — see include/trtmc/config/cli_support.h.
    // New feature knobs should generally prefer these over adding flags.
    std::string config_path;
    std::vector<std::string> set_tokens;
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
        bytes > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(bytes + 0.5L);
}

trtmc::LoadOptions make_load_options(const CliArgs& args) {
    trtmc::LoadOptions options;
    options.hf_python = args.hf_python;
    options.runtime_cache_path = args.runtime_cache;
    options.cuda_graphs = args.cuda_graphs;
    options.kv_cache_size_bytes = args.kv_cache_size_bytes;
    // Forward --config/--set into the factory so ConfigBundle resolution
    // actually sees them. Without this, every --set call silently no-ops
    // because pipeline_factory only reads from LoadOptions.
    options.config_path = args.config_path;
    options.set_tokens = args.set_tokens;
    options.backend_search_paths = args.backend_search_paths;
    return options;
}

void print_usage() {
    std::cerr
        << "Usage:\n"
           "  trtmc run             <bundle.trtfb> --prompt \"text\" [--image PATH] "
           "[--max-new-tokens N] [--temperature F] [--top-p F] [--min-p F] "
           "[--top-k N] [--seed N] [--benchmark N] [--warmup N] [--hf-python PATH] "
           "[--kv-cache-size SIZE] [--chat-template] [--no-thinking] "
           "[--num-samples N] [--num-steps N] [--guidance-scale S] [--cfg-scale S] "
           "[--sde-gamma S] [--initial-latents-raw PATH] [--condition-latents-raw PATH] "
           "[--condition-mask-raw PATH] [--sampling-steps-raw PATH] [--sde-noise-raw PATH] "
           "[--output samples.jsonl]\n"
           "                        Diffusion text-to-image extras (Qwen-Image, FLUX, "
           "Z-Image): [--negative-prompt \"text\"] "
           "[--num-inference-steps N] [--height N] [--width N]\n"
           "  trtmc encode          <bundle.trtfb> --prompt \"text\" [--hf-python PATH]\n"
           "  trtmc segment         <bundle.trtfb> --image PATH --output PATH [--hf-python PATH]\n"
           "  trtmc segment-sam     <bundle.trtfb> --image PATH --output DIR "
           "[--point-x F] [--point-y F] [--background] [--hf-python PATH]\n"
           "  trtmc classify        <bundle.trtfb> --image PATH [--benchmark N] [--warmup N]\n"
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
           "  trtmc inspect         <bundle.trtfb>\n"
           "  trtmc version\n"
           "\n"
           "Options:\n"
           "  --backend-dir PATH    Extra directory to search for libtrtmc_backend_*.so\n"
           "  --runtime-cache PATH   TRT-RTX JIT kernel cache file (speeds up repeat runs)\n"
           "  --cuda-graphs          Enable TRT-RTX CUDA graph capture (reduces launch overhead)\n";
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

    static const char* known_cmds[] = {"run",         "inspect",    "generate-video", "segment",
                                       "segment-sam", "classify",   "generate-audio", "serve-audio",
                                       "encode",      "embed",      "rerank",         "solve",
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

        if ((arg == "--prompt" || arg == "-p") && need_value(arg)) {
            args.prompt = argv[++i];
            continue;
        }
        if (arg == "--max-new-tokens" && need_value(arg)) {
            args.max_new_tokens = std::atoi(argv[++i]);
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
            args.seed = std::atoi(argv[++i]);
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

    return args;
}

int cmd_version() {
    std::cout << "trtmc " << trtmc_version() << '\n';
    std::cout << "TRT support: " << (trtmc_has_trt() ? "yes" : "no") << '\n';
    return EXIT_SUCCESS;
}

void print_text_timing(const trtmc::TextResult& result) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.timing] prefill_ms=" << result.prefill_ms
         << " decode_ms=" << result.decode_ms
         << " total_ms=" << (result.prefill_ms + result.decode_ms);
    std::cerr << line.str() << '\n';
}

std::optional<std::vector<float>> read_float32_raw_file(const std::string& path,
                                                        std::string& error) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        error = "failed to open " + path;
        return std::nullopt;
    }
    const auto end_pos = in.tellg();
    if (end_pos < 0) {
        error = "failed to size " + path;
        return std::nullopt;
    }
    const auto bytes = static_cast<std::size_t>(end_pos);
    if (bytes % sizeof(float) != 0U) {
        error = path + " size is not a multiple of float32";
        return std::nullopt;
    }

    std::vector<float> values(bytes / sizeof(float));
    in.seekg(0, std::ios::beg);
    if (!values.empty() &&
        !in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes))) {
        error = "failed to read " + path;
        return std::nullopt;
    }
    return values;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (unsigned char ch : text) {
        switch (ch) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

void write_text_sample_jsonl(std::ostream& out, int32_t id, const trtmc::TextResult& result) {
    out << "{\"id\":" << id << ",\"generated\":\"" << json_escape(result.text)
        << "\",\"token_ids\":[";
    for (std::size_t i = 0; i < result.token_ids.size(); ++i) {
        if (i > 0)
            out << ',';
        out << result.token_ids[i];
    }
    out << "]}\n";
}

int cmd_run(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: run requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    if (!pipeline) {
        std::cerr << "Error: failed to load bundle\n";
        return EXIT_FAILURE;
    }

    const std::string ptype = pipeline->pipeline_type();
    const std::string prompt =
        args.prompt.empty() && ptype != "ElfFlowPipeline" ? "Hello" : args.prompt;
    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens =
        args.max_new_tokens > 0 ? args.max_new_tokens : (ptype == "ElfFlowPipeline" ? 0 : 20);
    cfg.num_samples = args.num_samples;
    cfg.num_steps = args.num_steps;
    cfg.guidance_scale = args.guidance_scale;
    cfg.cfg_scale = args.cfg_scale;
    cfg.sde_gamma = args.sde_gamma;
    cfg.use_chat_template = args.chat_template;
    cfg.enable_thinking = !args.no_thinking;
    cfg.temperature = args.greedy ? 0.0F : args.temperature;
    cfg.top_p = args.top_p;
    cfg.min_p = args.min_p;
    cfg.top_k = args.top_k;
    cfg.seed = args.seed;
    if (!args.initial_latents_raw.empty()) {
        std::string error;
        auto latents = read_float32_raw_file(args.initial_latents_raw, error);
        if (!latents) {
            std::cerr << "Error: failed to read --initial-latents-raw: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.initial_latents = std::move(*latents);
    }
    if (!args.condition_latents_raw.empty()) {
        std::string error;
        auto latents = read_float32_raw_file(args.condition_latents_raw, error);
        if (!latents) {
            std::cerr << "Error: failed to read --condition-latents-raw: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.condition_latents = std::move(*latents);
    }
    if (!args.condition_mask_raw.empty()) {
        std::string error;
        auto mask = read_float32_raw_file(args.condition_mask_raw, error);
        if (!mask) {
            std::cerr << "Error: failed to read --condition-mask-raw: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.condition_mask = std::move(*mask);
    }
    if (!args.sampling_steps_raw.empty()) {
        std::string error;
        auto steps = read_float32_raw_file(args.sampling_steps_raw, error);
        if (!steps) {
            std::cerr << "Error: failed to read --sampling-steps-raw: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.sampling_steps = std::move(*steps);
    }
    if (!args.sde_noise_raw.empty()) {
        std::string error;
        auto noise = read_float32_raw_file(args.sde_noise_raw, error);
        if (!noise) {
            std::cerr << "Error: failed to read --sde-noise-raw: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.sde_noises = std::move(*noise);
    }
    // Diffusion-only knobs. Non-diffusion pipelines ignore these.
    cfg.negative_prompt = args.negative_prompt;
    cfg.height = args.diffusion_height;
    cfg.width = args.diffusion_width;
    // --cfg-scale is an alias for --guidance-scale on the diffusion path.
    // Prefer an explicitly set --cfg-scale if the user provided one.
    if (args.cfg_scale >= 0.0F) {
        cfg.guidance_scale = args.cfg_scale;
    }

    // Detect diffusion pipelines — they use generate_image(), not generate().
    const bool is_diffusion =
        (ptype.find("Diffusion") != std::string::npos || ptype.find("Flux") != std::string::npos ||
         ptype.find("Wan") != std::string::npos || ptype.find("ZImage") != std::string::npos ||
         ptype.find("LTX") != std::string::npos || ptype.find("QwenImage") != std::string::npos);

    // Diffusion pipelines may consume shared initial latents from a raw fp32
    // file (E2E shared-latents path; mirrors the cmd_generate_video plumbing).
    if (is_diffusion && !args.initial_latents_raw.empty()) {
        std::string error;
        auto latents = read_float32_raw_file(args.initial_latents_raw, error);
        if (!latents) {
            std::cerr << "Error: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.initial_latents = std::move(*latents);
    }

    if (args.benchmark > 0) {
        // Benchmark mode: warmup, then N timed iterations.
        const int warmup_n = args.warmup > 0 ? args.warmup : 1;
        const int bench_n = args.benchmark;

        std::cerr << "[trtmc.benchmark] warmup=" << warmup_n << " iterations=" << bench_n
                  << " max_new_tokens=" << cfg.max_new_tokens << '\n';

        for (int w = 0; w < warmup_n; ++w)
            pipeline->generate(prompt, cfg);

        std::vector<double> prefill_ms_v, decode_ms_v;
        prefill_ms_v.reserve(static_cast<std::size_t>(bench_n));
        decode_ms_v.reserve(static_cast<std::size_t>(bench_n));

        for (int r = 0; r < bench_n; ++r) {
            auto result = pipeline->generate(prompt, cfg);
            prefill_ms_v.push_back(result.prefill_ms);
            decode_ms_v.push_back(result.decode_ms);
        }

        auto mean = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        };

        const double pmean = mean(prefill_ms_v);
        const double dmean = mean(decode_ms_v);
        const int ntoks = cfg.max_new_tokens;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[trtmc.benchmark] prefill_ms=" << pmean << " decode_ms=" << dmean
                  << " tokens_per_sec=" << (ntoks > 0 ? ntoks / (dmean / 1000.0) : 0.0) << '\n';

        auto last = pipeline->generate(prompt, trtmc::GenerateConfig{cfg});
        std::cout << last.text << '\n';
    } else if (is_diffusion) {
        auto result = pipeline->generate_image(prompt, cfg);
        if (result.pixels.empty()) {
            std::cerr << "Error: image generation failed\n";
            return EXIT_FAILURE;
        }

        // Save as PNG. If -o ends with .png, use as file path; otherwise
        // treat as directory and write output.png inside it.
        std::string out_path;
        if (!args.output_dir.empty() && args.output_dir.size() > 4 &&
            args.output_dir.substr(args.output_dir.size() - 4) == ".png") {
            out_path = args.output_dir;
            auto parent = std::filesystem::path(out_path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
        } else {
            const std::string out_dir =
                args.output_dir.empty() ? "/tmp/trtmc_run_output" : args.output_dir;
            std::filesystem::create_directories(out_dir);
            out_path = out_dir + "/output.png";
        }

        try {
            trtmc::io::save_png(result, out_path);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "Saved " << out_path << " (" << result.width << "x" << result.height << ")\n";
    } else if (!args.image_path.empty()) {
        // Load image using trtmc_io
        auto image = trtmc::io::read_image(args.image_path);
        if (image.pixels.empty()) {
            std::cerr << "Error: failed to load image: " << args.image_path << '\n';
            return EXIT_FAILURE;
        }

        auto result =
            pipeline->generate(prompt, image.pixels.data(), image.height, image.width, cfg);
        print_text_timing(result);
        std::cout << result.text << '\n';
    } else {
        const int samples = std::max(1, cfg.num_samples);
        if (!cfg.initial_latents.empty() && samples > 1) {
            std::cerr << "Error: --initial-latents-raw can only be used with one sample\n";
            return EXIT_FAILURE;
        }

        std::ofstream jsonl_file;
        std::ostream* jsonl_out = nullptr;
        if (!args.output_dir.empty()) {
            auto out_path = std::filesystem::path(args.output_dir);
            auto parent = out_path.parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
            jsonl_file.open(out_path, std::ios::out | std::ios::trunc);
            if (!jsonl_file) {
                std::cerr << "Error: failed to open " << args.output_dir << " for writing\n";
                return EXIT_FAILURE;
            }
            jsonl_out = &jsonl_file;
        } else if (samples > 1) {
            jsonl_out = &std::cout;
        }

        for (int i = 0; i < samples; ++i) {
            trtmc::GenerateConfig sample_cfg = cfg;
            if (cfg.seed >= 0)
                sample_cfg.seed = cfg.seed + i;
            auto result = pipeline->generate(prompt, sample_cfg);
            print_text_timing(result);
            if (jsonl_out) {
                write_text_sample_jsonl(*jsonl_out, i, result);
            } else {
                std::cout << result.text << '\n';
            }
        }
        if (jsonl_file.is_open())
            std::cout << "Saved " << args.output_dir << " (" << samples << " samples)\n";
    }
    return EXIT_SUCCESS;
}

int cmd_generate_video(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: generate-video requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }

    const std::string out_dir =
        args.output_dir.empty() ? "/tmp/trtmc_generate_video" : args.output_dir;

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    trtmc::GenerateConfig cfg;
    cfg.num_steps = args.num_steps;
    cfg.guidance_scale = args.guidance_scale;
    cfg.seed = args.seed;
    if (!args.initial_latents_raw.empty()) {
        std::string error;
        auto latents = read_float32_raw_file(args.initial_latents_raw, error);
        if (!latents) {
            std::cerr << "Error: " << error << '\n';
            return EXIT_FAILURE;
        }
        cfg.initial_latents = std::move(*latents);
    }

    auto result = pipeline->generate_image(args.prompt, cfg);
    std::cout << "Generated image: " << result.width << "x" << result.height << " ("
              << result.num_frames << " frames)\n";

    // Create output directory (including parents) if it doesn't exist.
    std::filesystem::create_directories(out_dir);

    // Each frame in result.pixels is stored as [H, W, 3] float32 in [0,1],
    // with frames stacked contiguously: total layout is [T, H, W, 3].
    const auto frame_pixels =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width) * 3;

    for (int32_t f = 0; f < result.num_frames; ++f) {
        // Convert float32 HWC [0,1] to uint8 HWC [0,255].
        std::vector<unsigned char> rgb(frame_pixels);
        const float* src = result.pixels.data() + static_cast<std::size_t>(f) * frame_pixels;
        for (std::size_t i = 0; i < frame_pixels; ++i) {
            const float v = std::max(0.0F, std::min(1.0F, src[i]));
            rgb[i] = static_cast<unsigned char>(v * 255.0F + 0.5F);
        }

        // Build filename: frame_0000.png
        std::ostringstream fname;
        fname << out_dir << "/frame_" << std::setw(4) << std::setfill('0') << f << ".png";

        const int stride = result.width * 3;
        if (!stbi_write_png(fname.str().c_str(), result.width, result.height, 3, rgb.data(),
                            stride)) {
            std::cerr << "Error: failed to write " << fname.str() << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "Saved " << fname.str() << '\n';
    }

    return EXIT_SUCCESS;
}

int cmd_segment(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: segment requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    // Load image (HWC float32 in [0,1])
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    // Preprocess: resize to 512x512, normalize with ImageNet mean/std, convert to CHW
    const int32_t target_h = 512;
    const int32_t target_w = 512;
    const float mean[3] = {0.485F, 0.456F, 0.406F};
    const float stdv[3] = {0.229F, 0.224F, 0.225F};

    std::vector<float> chw_pixels(static_cast<std::size_t>(3) * target_h * target_w);
    for (int32_t y = 0; y < target_h; ++y) {
        for (int32_t x = 0; x < target_w; ++x) {
            // Bilinear-ish nearest-neighbor resize
            const int32_t src_y =
                std::min(static_cast<int32_t>(static_cast<float>(y) * image.height / target_h),
                         image.height - 1);
            const int32_t src_x =
                std::min(static_cast<int32_t>(static_cast<float>(x) * image.width / target_w),
                         image.width - 1);
            const auto src_idx = static_cast<std::size_t>((src_y * image.width + src_x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                const float val = (image.pixels[src_idx + c] - mean[c]) / stdv[c];
                chw_pixels[static_cast<std::size_t>(c) * target_h * target_w +
                           static_cast<std::size_t>(y) * target_w + x] = val;
            }
        }
    }

    auto result = pipeline->segment(chw_pixels.data(), target_h, target_w);

    // Save class map as grayscale PNG (pixel value = class index)
    const std::string out_path = args.output_dir.empty() ? "/tmp/seg_output.png" : args.output_dir;
    const int32_t out_h = result.height > 0 ? result.height : target_h;
    const int32_t out_w = result.width > 0 ? result.width : target_w;
    const auto total_px = static_cast<std::size_t>(out_h) * out_w;
    std::vector<unsigned char> gray(total_px);
    for (std::size_t i = 0; i < total_px && i < result.mask.size(); ++i)
        gray[i] = static_cast<unsigned char>(std::max(0, std::min(255, result.mask[i])));

    if (!stbi_write_png(out_path.c_str(), out_w, out_h, 1, gray.data(), out_w)) {
        std::cerr << "Error: failed to write output PNG: " << out_path << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Segmentation saved: " << out_path << " (" << out_w << "x" << out_h << ")\n";
    return EXIT_SUCCESS;
}

std::vector<float> preprocess_classification_image(const trtmc::io::LoadedImage& image) {
    constexpr int32_t target = 224;
    constexpr float crop_pct = 0.9F;
    const int32_t resize_short = static_cast<int32_t>(static_cast<float>(target) / crop_pct + 0.5F);
    const float mean[3] = {0.5F, 0.5F, 0.5F};
    const float stdv[3] = {0.5F, 0.5F, 0.5F};

    int32_t resized_h = resize_short;
    int32_t resized_w = resize_short;
    if (image.height <= image.width) {
        resized_h = resize_short;
        resized_w =
            std::max(1, static_cast<int32_t>(
                            static_cast<float>(image.width) * resize_short / image.height + 0.5F));
    } else {
        resized_w = resize_short;
        resized_h =
            std::max(1, static_cast<int32_t>(
                            static_cast<float>(image.height) * resize_short / image.width + 0.5F));
    }

    const int32_t crop_y = std::max(0, (resized_h - target) / 2);
    const int32_t crop_x = std::max(0, (resized_w - target) / 2);
    std::vector<float> chw(static_cast<std::size_t>(3) * target * target);

    for (int32_t y = 0; y < target; ++y) {
        const int32_t ry = crop_y + y;
        const int32_t src_y =
            std::min(image.height - 1,
                     static_cast<int32_t>(static_cast<float>(ry) * image.height / resized_h));
        for (int32_t x = 0; x < target; ++x) {
            const int32_t rx = crop_x + x;
            const int32_t src_x =
                std::min(image.width - 1,
                         static_cast<int32_t>(static_cast<float>(rx) * image.width / resized_w));
            const auto src_idx = static_cast<std::size_t>((src_y * image.width + src_x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                const float val = (image.pixels[src_idx + c] - mean[c]) / stdv[c];
                chw[static_cast<std::size_t>(c) * target * target +
                    static_cast<std::size_t>(y) * target + x] = val;
            }
        }
    }
    return chw;
}

int cmd_classify(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: classify requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    auto chw_pixels = preprocess_classification_image(image);
    constexpr int32_t target = 224;

    trtmc::ClassificationResult result;
    if (args.benchmark > 0) {
        const int warmup_n = std::max(0, args.warmup);
        const int bench_n = args.benchmark;
        for (int i = 0; i < warmup_n; ++i)
            result = pipeline->classify(chw_pixels.data(), target, target);

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(bench_n));
        for (int i = 0; i < bench_n; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            result = pipeline->classify(chw_pixels.data(), target, target);
            const auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
                            static_cast<double>(std::max(1, bench_n));
        std::cerr << std::fixed << std::setprecision(6) << "[trtmc.benchmark] classify_ms=" << mean
                  << " iterations=" << bench_n << " warmup=" << warmup_n << '\n';
    } else {
        result = pipeline->classify(chw_pixels.data(), target, target);
    }

    std::cout << "{"
              << "\"top_class\":" << result.top_class << ","
              << "\"top_score\":" << std::setprecision(8) << result.top_score << ","
              << "\"num_classes\":" << result.logits.size() << "}\n";
    return EXIT_SUCCESS;
}

int write_sam_overlay(const trtmc::PromptedSegmentationResult& result,
                      const trtmc::io::LoadedImage& image, const std::string& path) {
    if (result.num_masks <= 0 || result.height <= 0 || result.width <= 0 || result.masks.empty() ||
        image.empty())
        return EXIT_FAILURE;

    const auto mask_area =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width);
    int32_t selected = 0;
    if (static_cast<int32_t>(result.iou_scores.size()) >= result.num_masks) {
        selected = static_cast<int32_t>(
            std::distance(result.iou_scores.begin(),
                          std::max_element(result.iou_scores.begin(),
                                           result.iou_scores.begin() + result.num_masks)));
    }
    const float* mask = result.masks.data() +
                        static_cast<std::size_t>(selected) * static_cast<std::size_t>(mask_area);

    std::vector<unsigned char> rgb(mask_area * 3U, 0);
    for (int32_t y = 0; y < result.height; ++y) {
        const int32_t src_y =
            std::min(image.height - 1,
                     static_cast<int32_t>(static_cast<float>(y) * image.height / result.height));
        for (int32_t x = 0; x < result.width; ++x) {
            const int32_t src_x =
                std::min(image.width - 1,
                         static_cast<int32_t>(static_cast<float>(x) * image.width / result.width));
            const auto src_idx = static_cast<std::size_t>((src_y * image.width + src_x) * 3);
            const auto dst_idx = static_cast<std::size_t>((y * result.width + x) * 3);
            const bool active = mask[static_cast<std::size_t>(y) * result.width + x] > 0.0F;
            const float alpha = active ? 0.55F : 0.0F;
            const float overlay[3] = {0.0F, 0.85F, 0.25F};
            for (int32_t c = 0; c < 3; ++c) {
                const float base = std::clamp(image.pixels[src_idx + c], 0.0F, 1.0F);
                const float mixed = base * (1.0F - alpha) + overlay[c] * alpha;
                rgb[dst_idx + c] = static_cast<unsigned char>(mixed * 255.0F + 0.5F);
            }
        }
    }

    const int stride = result.width * 3;
    return stbi_write_png(path.c_str(), result.width, result.height, 3, rgb.data(), stride)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int cmd_segment_sam(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: segment-sam requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    const std::string out_dir = args.output_dir.empty() ? "/tmp/trtmc_masks" : args.output_dir;
    std::filesystem::create_directories(out_dir);

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    auto result = pipeline->segment_prompted(image.pixels.data(), image.height, image.width,
                                             args.point_x, args.point_y, args.is_foreground);
    if (result.num_masks <= 0 || result.height <= 0 || result.width <= 0 || result.masks.empty()) {
        std::cerr << "Error: SAM produced no masks\n";
        return EXIT_FAILURE;
    }

    const auto mask_area =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width);
    if (result.masks.size() < static_cast<std::size_t>(result.num_masks) * mask_area) {
        std::cerr << "Error: SAM mask payload is incomplete\n";
        return EXIT_FAILURE;
    }

    for (int32_t mask_idx = 0; mask_idx < result.num_masks; ++mask_idx) {
        const float* src = result.masks.data() +
                           static_cast<std::size_t>(mask_idx) * static_cast<std::size_t>(mask_area);
        std::vector<unsigned char> gray(mask_area, 0);
        for (std::size_t i = 0; i < mask_area; ++i)
            gray[i] = src[i] > 0.0F ? 255 : 0;

        std::ostringstream mask_path;
        mask_path << out_dir << "/mask_" << std::setw(3) << std::setfill('0') << mask_idx << ".png";
        if (!stbi_write_png(mask_path.str().c_str(), result.width, result.height, 1, gray.data(),
                            result.width)) {
            std::cerr << "Error: failed to write " << mask_path.str() << '\n';
            return EXIT_FAILURE;
        }

        if (mask_idx < static_cast<int32_t>(result.iou_scores.size())) {
            std::ostringstream score_path;
            score_path << out_dir << "/score_" << std::setw(3) << std::setfill('0') << mask_idx
                       << ".txt";
            std::ofstream score_out(score_path.str());
            score_out << std::fixed << std::setprecision(6) << result.iou_scores[mask_idx] << '\n';
        }
    }

    const std::string overlay_path = out_dir + "/segmented.png";
    if (write_sam_overlay(result, image, overlay_path) != EXIT_SUCCESS) {
        std::cerr << "Warning: failed to write " << overlay_path << '\n';
    }

    std::cout << "SAM segmentation saved: " << out_dir << " (" << result.num_masks << " masks, "
              << result.width << "x" << result.height << ")\n";
    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// serve-audio: persistent mode — load bundle once, read prompts from stdin,
// stream PCM float32 to stdout. One prompt per line.
//
// Protocol:
//   - Each line on stdin is a text prompt
//   - For each prompt, raw PCM float32 audio is written to stdout
//   - A 4-byte zero float (0x00000000) sentinel marks end of each utterance
//   - Logging goes to stderr
//   - Empty lines are skipped
//   - EOF on stdin exits
//
// Usage:
//   echo "Hello world" | trtmc serve-audio bundle.trtfb > out.raw
//   (or pipe multiple prompts, one per line)
// ---------------------------------------------------------------------------
int cmd_serve_audio(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: serve-audio requires a bundle path\n";
        return EXIT_FAILURE;
    }

    std::cerr << "[serve-audio] Loading bundle: " << args.bundle_path << std::endl;
    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    std::cerr << "[serve-audio] Ready. Reading prompts from stdin (one per line)..." << std::endl;

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 750;
    const int32_t chunk = args.chunk_frames > 0 ? args.chunk_frames : 16;

    // Sentinel: 4-byte zero float to mark end of utterance
    const float sentinel = 0.0F;

    std::string line;
    int32_t utterance = 0;
    while (std::getline(std::cin, line)) {
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        ++utterance;
        std::cerr << "[serve-audio] Utterance " << utterance << ": \"" << line.substr(0, 80)
                  << (line.size() > 80 ? "..." : "") << "\"" << std::endl;

        pipeline->generate_audio_streaming(
            line, cfg,
            [](const float* samples, int32_t n, int32_t /*rate*/) {
                std::fwrite(samples, sizeof(float), static_cast<std::size_t>(n), stdout);
                std::fflush(stdout);
            },
            chunk);

        // Write sentinel (end of utterance marker)
        std::fwrite(&sentinel, sizeof(float), 1, stdout);
        std::fflush(stdout);

        std::cerr << "[serve-audio] Utterance " << utterance << " done." << std::endl;
    }

    std::cerr << "[serve-audio] EOF on stdin, exiting after " << utterance << " utterances."
              << std::endl;
    return EXIT_SUCCESS;
}

int cmd_generate_audio(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: generate-audio requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 0;

    if (args.stream) {
        // Streaming mode: write raw PCM float32 to output file (or stdout
        // placeholder). Codec runs on chunks during decoding for low latency.
        // Pipe output to: aplay -r 22050 -f FLOAT_LE -c 1 -t raw
        const std::string out_path =
            args.output_dir.empty() ? "/tmp/generated_audio_stream.raw" : args.output_dir;
        FILE* fp = std::fopen(out_path.c_str(), "wb");
        if (!fp) {
            std::cerr << "Error: cannot open " << out_path << " for writing\n";
            return EXIT_FAILURE;
        }

        int32_t total = pipeline->generate_audio_streaming(
            args.prompt, cfg,
            [fp](const float* samples, int32_t n, int32_t /*rate*/) {
                std::fwrite(samples, sizeof(float), static_cast<std::size_t>(n), fp);
                std::fflush(fp);
            },
            args.chunk_frames);

        std::fclose(fp);
        std::cout << "Streamed " << total << " audio samples -> " << out_path << '\n';
        return EXIT_SUCCESS;
    }

    auto result = pipeline->generate_audio(args.prompt, cfg);

    const std::string out_path =
        args.output_dir.empty() ? "/tmp/generated_audio.wav" : args.output_dir;
    trtmc::io::write_wav(result, out_path);

    std::cout << "Generated " << result.num_samples << " audio samples -> " << out_path << '\n';
    return EXIT_SUCCESS;
}

int cmd_encode(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: encode requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    auto result = pipeline->encode(args.prompt);

    std::cerr << "Hidden states dim: " << result.dim << std::endl;
    std::cout << "{\"cls_embedding\": [";
    for (int i = 0; i < result.dim; ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << result.data[static_cast<std::size_t>(i)];
    }
    std::cout << "]}\n";
    return EXIT_SUCCESS;
}

int cmd_embed(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: embed requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    auto result = pipeline->embed(args.prompt);

    std::cerr << "Embedding dim: " << result.dim << std::endl;
    std::cout << "{\"embedding\": [";
    for (int i = 0; i < result.dim; ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << result.data[static_cast<std::size_t>(i)];
    }
    std::cout << "]}\n";
    return EXIT_SUCCESS;
}

int cmd_rerank(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty() || args.document.empty()) {
        std::cerr << "Error: rerank requires bundle + --prompt + --document\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    float score = pipeline->rerank(args.prompt, args.document);
    std::cout << "Relevance score: " << score << '\n';
    return EXIT_SUCCESS;
}

std::vector<float> parse_numeric_csv(const std::string& csv) {
    std::vector<float> values;
    std::string token;
    token.reserve(csv.size());

    auto flush_token = [&]() {
        if (token.empty())
            return;
        values.push_back(std::stof(token));
        token.clear();
    };

    for (char ch : csv) {
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
            flush_token();
            continue;
        }
        token.push_back(ch);
    }
    flush_token();
    return values;
}

int cmd_solve(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: solve requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }

    const bool has_field = !args.field_input.empty();
    const bool has_branch = !args.branch_input.empty();
    const bool has_trunk = !args.trunk_input.empty();
    if (!has_field && !has_branch) {
        std::cerr << "Error: solve requires --field-input or --branch-input\n";
        return EXIT_FAILURE;
    }
    if (has_field && (has_branch || has_trunk)) {
        std::cerr << "Error: solve accepts either --field-input or --branch-input/--trunk-input\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    std::vector<float> branch = parse_numeric_csv(has_field ? args.field_input : args.branch_input);
    std::vector<float> trunk =
        has_field ? std::vector<float>{} : parse_numeric_csv(args.trunk_input);

    auto result = pipeline->solve(
        branch.empty() ? nullptr : branch.data(), static_cast<int32_t>(branch.size()),
        trunk.empty() ? nullptr : trunk.data(), static_cast<int32_t>(trunk.size()));

    // Use double precision formatting so the text round-trip preserves the
    // exact float32 value as faithfully as possible for E2E parity checks.
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "Output [" << result.dim << "]:";
    for (int32_t i = 0; i < result.dim; ++i)
        std::cout << ' ' << result.data[static_cast<std::size_t>(i)];
    std::cout << '\n';
    return EXIT_SUCCESS;
}

int cmd_transcribe(const CliArgs& args) {
    if (args.bundle_path.empty() || args.audio_in.empty()) {
        std::cerr << "Error: transcribe requires bundle + --audio\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    auto audio = trtmc::io::read_wav(args.audio_in);
    int32_t max_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 224;

    if (args.stream) {
        trtmc::TranscriptionStreamConfig cfg;
        cfg.input_sample_rate = audio.sample_rate;
        cfg.max_new_tokens = max_tokens;
        cfg.att_context_left = args.att_context_left;
        cfg.att_context_right = args.att_context_right;
        cfg.use_cache = true;
        cfg.use_feature_cache = true;
        cfg.pad_and_drop_preencoded = args.pad_and_drop_preencoded;

        auto stream = pipeline->create_transcription_stream(cfg);
        const int32_t chunk_ms = args.chunk_ms > 0 ? args.chunk_ms : 160;
        const int32_t samples_per_chunk = std::max<int32_t>(
            1, static_cast<int32_t>(static_cast<int64_t>(audio.sample_rate) * chunk_ms / 1000));
        trtmc::TranscriptionStreamResult result;
        for (std::size_t offset = 0; offset < audio.samples.size();) {
            const auto remaining = audio.samples.size() - offset;
            const auto take =
                std::min<std::size_t>(remaining, static_cast<std::size_t>(samples_per_chunk));
            const bool is_final = (offset + take) >= audio.samples.size();
            result = stream->accept_audio(audio.samples.data() + offset, static_cast<int32_t>(take),
                                          is_final);
            offset += take;
        }
        if (audio.samples.empty())
            result = stream->finish();
        std::cout << result.text << '\n';
        return EXIT_SUCCESS;
    }

    auto result =
        pipeline->transcribe(audio.samples.data(), static_cast<int32_t>(audio.samples.size()),
                             max_tokens, audio.sample_rate);
    std::cout << result.text << '\n';
    return EXIT_SUCCESS;
}

int cmd_speak(const CliArgs& args) {
    if (args.bundle_path.empty() || args.audio_in.empty()) {
        std::cerr << "Error: speak requires bundle + --audio-in\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    auto audio = trtmc::io::read_wav(args.audio_in);

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : -1;
    cfg.tail_frames = args.tail_frames;

    auto result = pipeline->speak(audio.samples.data(), static_cast<int32_t>(audio.samples.size()),
                                  cfg, audio.sample_rate);

    const std::string out_path = args.audio_out.empty() ? "/tmp/speech_output.wav" : args.audio_out;
    trtmc::io::write_wav(result, out_path);

    std::cout << "Generated " << result.num_samples << " audio samples -> " << out_path << '\n';
    return EXIT_SUCCESS;
}

int cmd_inspect(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: inspect requires a bundle file path\n";
        return EXIT_FAILURE;
    }

    if (!trtmc::IsBundle(args.bundle_path)) {
        std::cerr << "Error: not a valid .trtfb bundle: " << args.bundle_path << '\n';
        return EXIT_FAILURE;
    }

    try {
        const auto info = trtmc::InspectBundle(args.bundle_path);
        std::cout << "Model ID:           " << info.model_id << '\n';
        std::cout << "Model type:         " << info.model_type << '\n';
        std::cout << "Family:             " << info.family << '\n';
        if (!info.precision.empty())
            std::cout << "Precision:          " << info.precision << '\n';
        std::cout << "TRT version:        " << info.trt_version << '\n';
        if (!info.trt_abi.empty())
            std::cout << "TRT ABI:            " << info.trt_abi << '\n';
        std::cout << "GPU:                " << info.gpu_name << '\n';
        std::cout << "Created:            " << info.created_at << '\n';
        std::cout << "Vocab size:         " << info.vocab_size << '\n';
        std::cout << "Hidden size:        " << info.hidden_size << '\n';
        std::cout << "Layers:             " << info.num_layers << '\n';
        std::cout << "Attention heads:    " << info.num_attention_heads << '\n';
        std::cout << "KV heads:           " << info.num_key_value_heads << '\n';
        std::cout << "Max cache length:   " << info.max_cache_length << '\n';
        if (!info.runtime_strategy.empty())
            std::cout << "Runtime strategy:   " << info.runtime_strategy << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace

// Resolve --config/--set and emit effective_config.json next to the
// bundle. No-op when neither flag was used. Pre-Phase-4 (no schemas
// registered for the relevant namespaces yet) the flags are accepted
// and a clear message prints — existing invocations are unaffected.
int apply_cli_config(const CliArgs& args) {
    if (args.config_path.empty() && args.set_tokens.empty())
        return EXIT_SUCCESS;
    if (trtmc::config::SchemaRegistry::instance().registered_namespaces().empty()) {
        std::cerr << "[trtmc] --config/--set accepted but no config schemas are "
                     "registered yet; values have no effect."
                  << '\n';
        return EXIT_SUCCESS;
    }
    try {
        auto bundle = trtmc::config::resolve_cli_config(args.config_path, args.set_tokens);
        if (!args.bundle_path.empty()) {
            std::string path =
                trtmc::config::write_effective_config_next_to(bundle, args.bundle_path);
            std::cerr << "[trtmc] Wrote effective config: " << path << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error resolving config: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    const CliArgs args = parse_args(argc, argv);

    if (args.show_help) {
        print_usage();
        return EXIT_SUCCESS;
    }
    if (args.parse_error) {
        std::cerr << "Error: " << args.error_message << '\n';
        print_usage();
        return EXIT_FAILURE;
    }
    if (int rc = apply_cli_config(args); rc != EXIT_SUCCESS)
        return rc;

    try {
        if (args.command == "version")
            return cmd_version();
        if (args.command == "run")
            return cmd_run(args);
        if (args.command == "encode")
            return cmd_encode(args);
        if (args.command == "segment")
            return cmd_segment(args);
        if (args.command == "segment-sam")
            return cmd_segment_sam(args);
        if (args.command == "classify")
            return cmd_classify(args);
        if (args.command == "generate-audio")
            return cmd_generate_audio(args);
        if (args.command == "serve-audio")
            return cmd_serve_audio(args);
        if (args.command == "generate-video")
            return cmd_generate_video(args);
        if (args.command == "embed")
            return cmd_embed(args);
        if (args.command == "rerank")
            return cmd_rerank(args);
        if (args.command == "solve")
            return cmd_solve(args);
        if (args.command == "speak")
            return cmd_speak(args);
        if (args.command == "transcribe")
            return cmd_transcribe(args);
        if (args.command == "inspect")
            return cmd_inspect(args);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    print_usage();
    return EXIT_FAILURE;
}

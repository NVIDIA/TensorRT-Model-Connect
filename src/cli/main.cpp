/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// trtmc CLI — command-line interface using the new C++ library API.
//
// Usage:
//   trtmc build           <hf-model-or-dir> -o <bundle.trtfb> [builder args...]
//   trtmc run             <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--benchmark N]
//                        [--warmup N] [--generation-mode MODE] [--block-length N]
//                        [--threshold F] [--num-samples N] [--num-steps N]
//                        [--guidance-scale S] [--cfg-scale S] [--sde-gamma S]
//                        [--initial-latents-raw PATH] [--condition-latents-raw PATH]
//                        [--condition-mask-raw PATH] [--sampling-steps-raw PATH]
//                        [--sde-noise-raw PATH] [--output samples.jsonl] [--hf-python PATH]
//                        Image-generation extras:
//                        [--negative-prompt "text"] [--num-inference-steps N]
//                        [--height N] [--width N]
//   trtmc transcribe      <bundle.trtfb> --audio FILE.wav [--beam-size N]
//                        [--source-language TAG] [--target-language TAG]
//                        [--task transcribe|translate] [--timestamps]
//   trtmc speak           <bundle.trtfb> --audio-in INPUT.wav --audio-out OUTPUT.wav
//   trtmc generate-video  <bundle.trtfb> --prompt "text" --output DIR [--num-steps N]
//                        [--negative-prompt "text"] [--height N] [--width N]
//   trtmc classify        <bundle.trtfb> --image PATH [--benchmark N] [--warmup N]
//   trtmc detect          <bundle.trtfb> --image PATH [--output-json PATH]
//   trtmc inspect         <bundle.trtfb> [--list-engines]
//   trtmc version

#include "cli/args.h"
#include "stb_image_write.h"
#include "trtmc/bundle.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

using trtmc::cli::CliArgs;
using trtmc::cli::make_load_options;
using trtmc::cli::parse_args;
using trtmc::cli::print_usage;

// Build a per-sample output path. For ``total == 1`` the prefix is used as-is
// so single-image runs keep their historical filename. Otherwise an
// ``_<index>`` suffix is inserted before the extension (e.g. ``out.png`` ->
// ``out_0.png``); prefixes with no extension simply gain ``_<index>`` at the
// end.
std::string format_output_path(const std::string& prefix, int index, int total) {
    if (total <= 1)
        return prefix;
    const auto dot = prefix.find_last_of('.');
    const auto slash = prefix.find_last_of('/');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    std::ostringstream out;
    if (has_ext)
        out << prefix.substr(0, dot) << '_' << index << prefix.substr(dot);
    else
        out << prefix << '_' << index;
    return out.str();
}

std::uint32_t low32(std::uint64_t v) {
    return static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
}

std::uint32_t high32(std::uint64_t v) {
    return static_cast<std::uint32_t>((v >> 32) & 0xFFFFFFFFu);
}

std::vector<std::uint32_t> derive_image_batch_seeds(std::uint64_t global_seed, int count) {
    if (count < 1) {
        throw std::invalid_argument("count must be >= 1");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        std::seed_seq seq{
            low32(global_seed),
            high32(global_seed),
            static_cast<std::uint32_t>(i),
        };
        std::array<std::uint32_t, 1> buf{};
        seq.generate(buf.begin(), buf.end());
        out.push_back(buf[0]);
    }
    return out;
}

std::vector<std::uint32_t>
normalize_explicit_image_batch_seeds(const std::vector<std::uint64_t>& explicit_list, int count) {
    if (count < 1) {
        throw std::invalid_argument("count must be >= 1");
    }
    if (static_cast<int>(explicit_list.size()) != count) {
        throw std::invalid_argument("explicit seed list length must equal the total batch count");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (auto v : explicit_list) {
        out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
}

void preload_cli_config_schema_owner(const CliArgs& args) {
    if (args.bundle_path.empty())
        return;
    if (!trtmc::IsBundle(args.bundle_path))
        return;

    const auto info = trtmc::InspectBundle(args.bundle_path);
    std::string strategy = info.runtime_strategy;
    if (strategy.empty()) {
        auto fallback = trtmc::default_runtime_strategy();
        if (!fallback || fallback->empty())
            return;
        strategy = *fallback;
    }
    if (auto alias = trtmc::legacy_runtime_strategy_alias_target(strategy, ""))
        strategy = *alias;

    trtmc::load_model_plugin_for_strategy_with_abi_policy(
        strategy, args.model_plugin_search_paths,
        info.runtime_memory.present ? trtmc::ModelPluginAbiPolicy::kRequireCurrent
                                    : trtmc::ModelPluginAbiPolicy::kAllowLegacyUnversioned);
}

std::filesystem::path current_executable_path() {
    char buf[4096];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        return {};
    buf[len] = '\0';
    return std::filesystem::path(buf);
}

std::string build_python_executable() {
    const auto current_exe = current_executable_path();
    if (current_exe.empty())
        return "python3";

    std::error_code ec;
    const std::filesystem::path exe_path = std::filesystem::weakly_canonical(current_exe, ec);
    if (!ec && !exe_path.empty()) {
        const std::filesystem::path exe_dir = exe_path.parent_path();
        for (const char* name : {"python3", "python"}) {
            const std::filesystem::path candidate = exe_dir / name;
            std::error_code exists_ec;
            if (std::filesystem::exists(candidate, exists_ec) &&
                access(candidate.c_str(), X_OK) == 0) {
                return candidate.string();
            }
        }
    }

    return "python3";
}

std::string build_pythonpath() {
    std::string pythonpath;

#ifdef TRTMC_SOURCE_DIR
    const auto current_exe = current_executable_path();
    if (!current_exe.empty()) {
        std::error_code source_ec;
        std::error_code exe_ec;
        const auto source_root = std::filesystem::weakly_canonical(TRTMC_SOURCE_DIR, source_ec);
        const auto exe_path = std::filesystem::weakly_canonical(current_exe, exe_ec);
        std::error_code rel_ec;
        const auto rel_exe_path = std::filesystem::relative(exe_path, source_root, rel_ec);
        const auto first_component =
            rel_exe_path.empty() ? std::filesystem::path{} : *rel_exe_path.begin();
        const bool running_from_source_build = !source_ec && !exe_ec && !rel_ec &&
                                               !rel_exe_path.empty() &&
                                               first_component.string().rfind("build", 0) == 0;
        if (running_from_source_build) {
            const auto source_pkg = std::filesystem::path(TRTMC_SOURCE_DIR) / "python";
            std::error_code ec;
            if (std::filesystem::is_directory(source_pkg, ec)) {
                pythonpath = source_pkg.string();
            }
        }
    }
#endif

    const char* existing = std::getenv("PYTHONPATH");
    if (existing && existing[0] != '\0') {
        if (!pythonpath.empty())
            pythonpath += ":";
        pythonpath += existing;
    }
    return pythonpath;
}

bool configure_builder_plugin_library() {
    const char* existing = std::getenv("TRTMC_TRT_PLUGIN_LIBRARY");
    // Presence is an explicit user/qualification choice, including an empty
    // or otherwise invalid value.  Preserve it so the Python selector can
    // reject it instead of silently falling back to another build tree.
    if (existing != nullptr)
        return true;

    const auto current_exe = current_executable_path();
    if (current_exe.empty())
        return true;

    std::error_code exe_ec;
    const auto exe_path = std::filesystem::weakly_canonical(current_exe, exe_ec);
    if (exe_ec || exe_path.empty())
        return true;

    const auto candidate = exe_path.parent_path() / "libtrtmc_trt_plugins.so";
    std::error_code file_ec;
    if (!std::filesystem::is_regular_file(candidate, file_ec))
        return true;

    if (setenv("TRTMC_TRT_PLUGIN_LIBRARY", candidate.c_str(), 1) == 0)
        return true;

    std::cerr << "Error: failed to select the TensorRT plugin library adjacent to " << exe_path
              << ": " << std::strerror(errno) << '\n';
    return false;
}

int run_python_module(const std::vector<std::string>& argv) {
    if (argv.empty()) {
        std::cerr << "Error: empty Python command\n";
        return EXIT_FAILURE;
    }

    std::vector<char*> exec_argv;
    exec_argv.reserve(argv.size() + 1);
    for (const auto& arg : argv)
        exec_argv.push_back(const_cast<char*>(arg.c_str()));
    exec_argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Error: failed to start Python builder: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }

    if (pid == 0) {
        const std::string pythonpath = build_pythonpath();
        if (!pythonpath.empty())
            setenv("PYTHONPATH", pythonpath.c_str(), 1);
        if (!configure_builder_plugin_library())
            _exit(127);
        execvp(exec_argv[0], exec_argv.data());
        std::cerr << "Error: failed to execute " << argv[0] << ": " << std::strerror(errno) << '\n';
        _exit(127);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR)
            continue;
        std::cerr << "Error: failed waiting for Python builder: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) {
        const int sig = WTERMSIG(status);
        std::cerr << "Error: Python builder terminated by signal " << sig << '\n';
        return 128 + sig;
    }
    return EXIT_FAILURE;
}

int cmd_build(const CliArgs& args) {
    std::vector<std::string> command = {
        build_python_executable(),
        "-m",
        "tensorrt_model_connect",
        "build",
    };
    command.insert(command.end(), args.build_args.begin(), args.build_args.end());
    return run_python_module(command);
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
    std::cerr << std::fixed << std::setprecision(6)
              << "[trtmc.setup_timing] setup_ms=" << result.setup_ms << '\n';
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

void print_request_complete_memory_receipt(const trtmc::IPipeline& pipeline) {
    const auto* introspection =
        dynamic_cast<const trtmc::IRuntimeMemoryIntrospectionV1*>(&pipeline);
    if (introspection == nullptr || introspection->runtime_memory_api_version() != 1 ||
        introspection->runtime_kv_capacity_tokens() == 0) {
        return;
    }
    const auto receipt = introspection->runtime_memory_receipt_json();
    if (!receipt.empty())
        std::cerr << "[trtmc.memory] " << receipt << '\n';
}

int cmd_run(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: run requires a .trtfb bundle file\n";
        return EXIT_FAILURE;
    }
    if (!trtmc::cli::has_run_input_source(args)) {
        std::cerr
            << "Error: run requires bundle + --prompt, --prompts-file, or --initial-latents-raw\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    if (!pipeline) {
        std::cerr << "Error: failed to load bundle\n";
        return EXIT_FAILURE;
    }

    const std::string prompt = args.prompt;
    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens =
        args.max_new_tokens > 0 ? args.max_new_tokens : pipeline->default_max_new_tokens();
    cfg.num_samples = args.num_samples;
    cfg.block_length = args.block_length;
    cfg.confidence_threshold = args.conf_threshold;
    if (!args.generation_mode.empty())
        cfg.text_generation_mode = args.generation_mode;
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
    if (!args.lora_adapter_path.empty()) {
        pipeline->load_lora_adapter(args.lora_adapter_id, args.lora_adapter_path);
        cfg.lora_adapter_id = args.lora_adapter_id;
    }
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
    // Image-generation pipelines use generate_image(), not generate().
    const bool is_image_generation = pipeline->supports_image_generation();
    // Image diffusion historically treats --cfg-scale as an alias for
    // --guidance-scale. Text diffusion pipelines such as ELF use both values
    // independently, so preserve the separately parsed values for generate().
    if (is_image_generation && args.cfg_scale >= 0.0F) {
        cfg.guidance_scale = args.cfg_scale;
    }

    // Diffusion pipelines may consume shared initial latents from a raw fp32
    // file (E2E shared-latents path; mirrors the cmd_generate_video plumbing).
    if (is_image_generation && !args.initial_latents_raw.empty()) {
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

        std::vector<double> setup_ms_v, prefill_ms_v, decode_ms_v;
        setup_ms_v.reserve(static_cast<std::size_t>(bench_n));
        prefill_ms_v.reserve(static_cast<std::size_t>(bench_n));
        decode_ms_v.reserve(static_cast<std::size_t>(bench_n));

        for (int r = 0; r < bench_n; ++r) {
            auto result = pipeline->generate(prompt, cfg);
            setup_ms_v.push_back(result.setup_ms);
            prefill_ms_v.push_back(result.prefill_ms);
            decode_ms_v.push_back(result.decode_ms);
        }

        auto mean = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        };

        const double smean = mean(setup_ms_v);
        const double pmean = mean(prefill_ms_v);
        const double dmean = mean(decode_ms_v);
        const int ntoks = cfg.max_new_tokens;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[trtmc.benchmark] setup_ms=" << smean << " prefill_ms=" << pmean
                  << " decode_ms=" << dmean
                  << " tokens_per_sec=" << (ntoks > 0 ? ntoks / (dmean / 1000.0) : 0.0) << '\n';

        auto last = pipeline->generate(prompt, trtmc::GenerateConfig{cfg});
        std::cout << last.text << '\n';
    } else if (is_image_generation) {
        // Image-conditioned diffusion (img2img) keeps the single-image path —
        // batched img2img is out of scope for the PR-3 CLI wiring.
        if (!args.image_path.empty()) {
            auto image = trtmc::io::read_image(args.image_path);
            if (image.pixels.empty()) {
                std::cerr << "Error: failed to load image: " << args.image_path << '\n';
                return EXIT_FAILURE;
            }
            trtmc::ImageResult result = pipeline->generate_image(prompt, image.pixels.data(),
                                                                 image.height, image.width, cfg);
            if (result.pixels.empty()) {
                std::cerr << "Error: image generation failed\n";
                return EXIT_FAILURE;
            }

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
            std::cout << "Saved " << out_path << " (" << result.width << "x" << result.height
                      << ")\n";
        } else {
            // Text-to-image batch path. Build the prompt list from either
            // --prompts-file (one prompt per line) or `num_images` copies of
            // --prompt, then dispatch through generate_image_batch.
            std::vector<std::string> prompts;
            if (!args.prompts_file.empty()) {
                std::string error;
                prompts = trtmc::cli::read_prompts_file(args.prompts_file, error);
                if (prompts.empty()) {
                    std::cerr << "Error: " << error << '\n';
                    return EXIT_FAILURE;
                }
            } else {
                prompts.assign(static_cast<std::size_t>(std::max(1, args.num_images)), prompt);
            }
            const int total = static_cast<int>(prompts.size());

            // Resolve per-sample seeds: explicit CSV when --seed s0,s1,... is
            // given, else derive deterministic seeds from the scalar seed.
            std::vector<std::uint32_t> per_sample_seeds;
            try {
                if (!args.seed_list.empty()) {
                    if (static_cast<int>(args.seed_list.size()) != total) {
                        std::cerr << "Error: --seed CSV length (" << args.seed_list.size()
                                  << ") must equal the total batch count (" << total << ")\n";
                        return EXIT_FAILURE;
                    }
                    per_sample_seeds = normalize_explicit_image_batch_seeds(args.seed_list, total);
                } else {
                    const std::uint64_t global_seed =
                        args.seed >= 0 ? static_cast<std::uint64_t>(args.seed) : 0ULL;
                    per_sample_seeds = derive_image_batch_seeds(global_seed, total);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << '\n';
                return EXIT_FAILURE;
            }

            auto results = pipeline->generate_image_batch(prompts, per_sample_seeds, cfg);
            if (results.empty()) {
                std::cerr << "Error: image generation failed\n";
                return EXIT_FAILURE;
            }

            // Resolve the output prefix. ``-o foo.png`` puts files alongside
            // ``foo.png``; otherwise treat ``-o`` as a directory containing
            // ``output.png`` style outputs.
            std::string out_prefix;
            if (!args.output_dir.empty() && args.output_dir.size() > 4 &&
                args.output_dir.substr(args.output_dir.size() - 4) == ".png") {
                out_prefix = args.output_dir;
                auto parent = std::filesystem::path(out_prefix).parent_path();
                if (!parent.empty())
                    std::filesystem::create_directories(parent);
            } else {
                const std::string out_dir =
                    args.output_dir.empty() ? "/tmp/trtmc_run_output" : args.output_dir;
                std::filesystem::create_directories(out_dir);
                out_prefix = out_dir + "/output.png";
            }

            for (std::size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                if (r.pixels.empty()) {
                    std::cerr << "Error: image generation failed for sample " << i << '\n';
                    return EXIT_FAILURE;
                }
                const std::string out_path =
                    format_output_path(out_prefix, static_cast<int>(i), total);
                try {
                    trtmc::io::save_png(r, out_path);
                } catch (const std::exception& e) {
                    std::cerr << "Error: " << e.what() << '\n';
                    return EXIT_FAILURE;
                }
                std::cout << "Saved " << out_path << " (" << r.width << "x" << r.height << ")\n";
            }
        }
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
    // Dynamic-memory pipelines sampled their device-wide high-water after
    // the complete request. Emit the refreshed receipt once, outside all
    // token/prefill/decode loops, so CLI and the Python subprocess wrapper
    // observe request-lifetime rather than load-only accounting.
    print_request_complete_memory_receipt(*pipeline);
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
    cfg.negative_prompt = args.negative_prompt;
    cfg.height = args.diffusion_height;
    cfg.width = args.diffusion_width;
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

    auto result = pipeline->segment(image.pixels.data(), image.height, image.width);

    // Save class map as grayscale PNG (pixel value = class index)
    const std::string out_path = args.output_dir.empty() ? "/tmp/seg_output.png" : args.output_dir;
    const int32_t out_h = result.height > 0 ? result.height : image.height;
    const int32_t out_w = result.width > 0 ? result.width : image.width;
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

    trtmc::ClassificationResult result;
    if (args.benchmark > 0) {
        const int warmup_n = std::max(0, args.warmup);
        const int bench_n = args.benchmark;
        for (int i = 0; i < warmup_n; ++i)
            result = pipeline->classify(image.pixels.data(), image.height, image.width);

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(bench_n));
        for (int i = 0; i < bench_n; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            result = pipeline->classify(image.pixels.data(), image.height, image.width);
            const auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
                            static_cast<double>(std::max(1, bench_n));
        std::cerr << std::fixed << std::setprecision(6) << "[trtmc.benchmark] classify_ms=" << mean
                  << " iterations=" << bench_n << " warmup=" << warmup_n << '\n';
    } else {
        result = pipeline->classify(image.pixels.data(), image.height, image.width);
    }

    std::cout << "{"
              << "\"top_class\":" << result.top_class << ","
              << "\"top_score\":" << std::setprecision(8) << result.top_score << ","
              << "\"num_classes\":" << result.logits.size() << "}\n";
    return EXIT_SUCCESS;
}

int cmd_detect(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: detect requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    const float threshold = args.conf_threshold >= 0.0F ? args.conf_threshold : 0.5F;
    const std::string detections =
        pipeline->detect(image.pixels.data(), image.height, image.width, threshold);

    if (!args.output_json.empty()) {
        auto out_path = std::filesystem::path(args.output_json);
        auto parent = out_path.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream out(out_path, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "Error: failed to open " << args.output_json << " for writing\n";
            return EXIT_FAILURE;
        }
        out << detections;
        if (detections.empty() || detections.back() != '\n')
            out << '\n';
        std::cout << "Detections saved: " << args.output_json << '\n';
    } else {
        std::cout << detections;
        if (detections.empty() || detections.back() != '\n')
            std::cout << '\n';
    }
    return EXIT_SUCCESS;
}

int write_prompted_segmentation_overlay(const trtmc::PromptedSegmentationResult& result,
                                        const trtmc::io::LoadedImage& image,
                                        const std::string& path) {
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

int cmd_segment_prompted(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: segment-prompted requires bundle + --image\n";
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

    trtmc::PromptedSegmentationResult result;
    if (!args.prompt.empty()) {
        result = pipeline->segment_prompted_text(image.pixels.data(), image.height, image.width,
                                                 args.prompt);
    } else {
        result = pipeline->segment_prompted(image.pixels.data(), image.height, image.width,
                                            args.point_x, args.point_y, args.is_foreground);
    }
    if (result.num_masks <= 0 || result.height <= 0 || result.width <= 0 || result.masks.empty()) {
        std::cerr << "Error: prompted segmentation produced no masks\n";
        return EXIT_FAILURE;
    }

    const auto mask_area =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width);
    if (result.masks.size() < static_cast<std::size_t>(result.num_masks) * mask_area) {
        std::cerr << "Error: prompted segmentation mask payload is incomplete\n";
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

        const auto box_offset = static_cast<std::size_t>(mask_idx) * 4U;
        if (result.boxes.size() >= box_offset + 4U) {
            std::ostringstream box_path;
            box_path << out_dir << "/box_" << std::setw(3) << std::setfill('0') << mask_idx
                     << ".txt";
            std::ofstream box_out(box_path.str());
            box_out << std::fixed << std::setprecision(6) << result.boxes[box_offset] << ' '
                    << result.boxes[box_offset + 1U] << ' ' << result.boxes[box_offset + 2U] << ' '
                    << result.boxes[box_offset + 3U] << '\n';
        }
    }

    const std::string overlay_path = out_dir + "/segmented.png";
    if (write_prompted_segmentation_overlay(result, image, overlay_path) != EXIT_SUCCESS) {
        std::cerr << "Warning: failed to write " << overlay_path << '\n';
    }

    std::cout << "Prompted segmentation saved: " << out_dir << " (" << result.num_masks
              << " masks, " << result.width << "x" << result.height << ")\n";
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
    const std::vector<std::string> audio_paths =
        !args.audio_inputs.empty()
            ? args.audio_inputs
            : (args.audio_in.empty() ? std::vector<std::string>{}
                                     : std::vector<std::string>{args.audio_in});
    if (args.bundle_path.empty() || audio_paths.empty()) {
        std::cerr << "Error: transcribe requires bundle + --audio\n";
        return EXIT_FAILURE;
    }

    auto pipeline = trtmc::load(args.bundle_path, make_load_options(args));

    int32_t max_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 224;

    if (args.stream) {
        if (audio_paths.size() != 1) {
            std::cerr << "Error: --stream accepts exactly one --audio input\n";
            return EXIT_FAILURE;
        }
        const bool has_offline_only_controls =
            args.beam_size != 1 || args.transcription_task != "transcribe" || !args.punctuation ||
            args.timestamps || args.max_input_seconds > 0.0F ||
            args.segment_length_seconds > 0.0F ||
            (args.language.empty() &&
             (args.source_language != "en" || args.target_language != "en"));
        if (has_offline_only_controls) {
            std::cerr << "Error: decoding, duration, and segment controls are only "
                         "supported for offline transcription\n";
            return EXIT_FAILURE;
        }
        auto audio = trtmc::io::read_wav(audio_paths.front());
        trtmc::TranscriptionStreamConfig cfg;
        cfg.input_sample_rate = audio.sample_rate;
        cfg.max_new_tokens = max_tokens;
        cfg.att_context_left = args.att_context_left;
        cfg.att_context_right = args.att_context_right;
        cfg.use_cache = true;
        cfg.use_feature_cache = true;
        cfg.pad_and_drop_preencoded = args.pad_and_drop_preencoded;
        cfg.language = args.language;

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

    std::vector<trtmc::TranscriptionRequest> requests;
    requests.reserve(audio_paths.size());
    for (const auto& path : audio_paths) {
        auto audio = trtmc::io::read_wav(path);
        trtmc::TranscriptionRequest request;
        request.audio_samples = std::move(audio.samples);
        request.config.max_output_tokens = max_tokens;
        request.config.input_sample_rate = audio.sample_rate;
        request.config.beam_size = args.beam_size;
        request.config.source_language = args.source_language;
        request.config.target_language = args.target_language;
        request.config.task = args.transcription_task == "translate"
                                  ? trtmc::TranscriptionTask::kTranslate
                                  : trtmc::TranscriptionTask::kTranscribe;
        request.config.punctuation = args.punctuation;
        request.config.timestamps = args.timestamps;
        request.config.max_input_duration_seconds = args.max_input_seconds;
        request.config.segment_duration_seconds = args.segment_length_seconds;
        requests.push_back(std::move(request));
    }

    auto results = pipeline->transcribe_batch(requests);
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (args.timestamps) {
            for (const auto& segment : results[i].segments) {
                if (results.size() > 1)
                    std::cout << audio_paths[i] << '\t';
                std::cout << std::fixed << std::setprecision(3) << segment.start_seconds << '\t'
                          << segment.end_seconds << '\t' << segment.text << '\n';
            }
        } else {
            if (results.size() > 1)
                std::cout << audio_paths[i] << '\t';
            std::cout << results[i].text << '\n';
        }
    }
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

bool is_engine_section(const std::string& name) {
    return name == "engine_plan" ||
           (name.size() >= 5 && name.compare(name.size() - 5, 5, "_plan") == 0);
}

std::string engine_section_role(const std::string& name) {
    if (name == "engine_plan")
        return "primary";
    if (name.find("vision") != std::string::npos)
        return "vision";
    if (name.find("text_encoder") != std::string::npos)
        return "text_encoder";
    if (name.find("denoiser") != std::string::npos)
        return "denoiser";
    if (name.find("vae") != std::string::npos)
        return "vae";
    if (name.find("lt_") != std::string::npos ||
        name.find("local_transformer") != std::string::npos)
        return "local_transformer";
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, "_plan") == 0)
        return name.substr(0, name.size() - 5);
    return name;
}

int cmd_inspect_list_engines(const trtmc::BundleInfo& info) {
    std::vector<trtmc::BundleSectionInfo> engines;
    for (const auto& section : info.sections) {
        if (is_engine_section(section.name))
            engines.push_back(section);
    }
    if (engines.empty()) {
        std::cerr << "No engine sections found.\n";
        return EXIT_FAILURE;
    }

    std::cout << std::left << std::setw(30) << "Section" << ' ' << std::right << std::setw(10)
              << "Size"
              << " " << std::left << std::setw(16) << "Role" << '\n';
    std::cout << std::string(30, '-') << ' ' << std::string(10, '-') << ' ' << std::string(16, '-')
              << '\n';
    for (const auto& section : engines) {
        const double size_mb = static_cast<double>(section.size) / (1024.0 * 1024.0);
        std::cout << std::left << std::setw(30) << section.name << ' ' << std::right << std::setw(8)
                  << std::fixed << std::setprecision(1) << size_mb << " MB " << std::left
                  << std::setw(16) << engine_section_role(section.name) << '\n';
    }
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
        if (args.list_engines)
            return cmd_inspect_list_engines(info);

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
        if (info.runtime_memory.present) {
            const auto& memory = info.runtime_memory;
            std::cout << "Runtime KV contract version: " << memory.contract_version << '\n';
            std::cout << "Qualified model ID: " << memory.qualified_model_id << '\n';
            std::cout << "Qualified model revision: " << memory.qualified_model_revision << '\n';
            std::cout << "Qualified config fingerprint: " << memory.qualified_config_sha256 << '\n';
            std::cout << "Qualified target:   " << memory.qualified_target << '\n';
            const auto& stack = memory.qualified_runtime_stack;
            std::cout << "Qualified runtime stack: "
                      << "SM=" << stack.sm << ", TensorRT=" << stack.tensorrt
                      << ", CUDA=" << stack.cuda_runtime << ", cuDNN=" << stack.cudnn_backend
                      << ", Frontend=" << stack.cudnn_frontend_revision << ", NVRTC=" << stack.nvrtc
                      << ", driver=" << stack.driver << '\n';
            std::cout << "Native KV plugin ABI: " << memory.native_kv_plugin_abi << '\n';
            std::cout << "Model context limit: " << memory.model_context_limit << '\n';
            std::cout << "Prefill chunk limit: " << memory.prefill_chunk_limit << '\n';
            std::cout << "KV layout:          " << memory.kv_layout << '\n';
            std::cout << "KV dtype:           " << memory.kv_dtype << '\n';
            std::cout << "KV bytes per token: " << memory.kv_bytes_per_token << '\n';
            std::cout << "Active KV profile limits: ";
            for (std::size_t index = 0; index < memory.active_kv_profile_limits.size(); ++index) {
                if (index != 0)
                    std::cout << ", ";
                std::cout << memory.active_kv_profile_limits[index];
            }
            std::cout << '\n';
            std::cout << "Runtime-owned KV:   yes\n";
        } else {
            std::cout << "Max cache length:   " << info.max_cache_length << '\n';
        }
        if (!info.runtime_strategy.empty())
            std::cout << "Runtime strategy:   " << info.runtime_strategy << '\n';
        // Diffusion batch envelope (PR 1 added the field to BundleInfo;
        // defaults are 1/1/1 for legacy bundles). VAE is always sliced --
        // see Decision E in the diffusion batch-inference RFC.
        std::cout << "Max batch size:\n";
        std::cout << "  dit:          " << info.max_batch_size.dit << '\n';
        std::cout << "  text_encoder: " << info.max_batch_size.text_encoder << '\n';
        std::cout << "  vae:          " << info.max_batch_size.vae
                  << "  (always sliced -- Decision E)\n";
        if (!info.sections.empty()) {
            std::cout << "Sections:\n";
            for (const auto& section : info.sections) {
                const double size_mb = static_cast<double>(section.size) / (1024.0 * 1024.0);
                std::cout << "  " << section.name << ": " << std::fixed << std::setprecision(1)
                          << size_mb << " MB\n";
            }
        }
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
    if (args.command == "build")
        return EXIT_SUCCESS;
    if (args.config_path.empty() && args.set_tokens.empty())
        return EXIT_SUCCESS;
    if ((args.kv_cache_memory.explicitly_set || args.max_sequence_length_explicitly_set) &&
        !args.bundle_path.empty() && trtmc::IsBundle(args.bundle_path)) {
        const auto info = trtmc::InspectBundle(args.bundle_path);
        if (!info.runtime_memory.present || info.runtime_memory.contract_version != 1 ||
            !info.runtime_memory.runtime_owned) {
            // Config-schema discovery dlopens the model provider. Keep the
            // runtime-policy contract rejection ahead of that preload just as
            // PipelineFactory does on the ordinary no-config path.
            std::cerr << "Error: This bundle does not declare runtime_memory contract version 1; "
                         "runtime KV memory and max-sequence policies cannot be applied\n";
            return EXIT_FAILURE;
        }
    }
    preload_cli_config_schema_owner(args);
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
        if (args.command == "build")
            return cmd_build(args);
        if (args.command == "run")
            return cmd_run(args);
        if (args.command == "encode")
            return cmd_encode(args);
        if (args.command == "segment")
            return cmd_segment(args);
        if (args.command == "segment-prompted")
            return cmd_segment_prompted(args);
        if (args.command == "classify")
            return cmd_classify(args);
        if (args.command == "detect")
            return cmd_detect(args);
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

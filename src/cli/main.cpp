/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// trtmc CLI — command-line interface using the new C++ library API.
//
// Usage:
//   trtmc build           <hf-model-or-dir> -o <bundle.bundle> [builder args...]
//   trtmc run             <bundle.bundle> --prompt "text" [--max-new-tokens N] [--benchmark N]
//                        [--warmup N] [--generation-mode MODE] [--block-length N]
//                        [--threshold F] [--num-samples N] [--num-steps N]
//                        [--guidance-scale S] [--cfg-scale S] [--sde-gamma S]
//                        [--initial-latents-raw PATH] [--condition-latents-raw PATH]
//                        [--condition-mask-raw PATH] [--sampling-steps-raw PATH]
//                        [--sde-noise-raw PATH] [--output samples.jsonl] [--hf-python PATH]
//                        Image-generation extras:
//                        [--negative-prompt "text"] [--num-inference-steps N]
//                        [--height N] [--width N]
//   trtmc transcribe      <bundle.bundle> --audio FILE.wav [--beam-size N]
//                        [--source-language TAG] [--target-language TAG]
//                        [--task transcribe|translate] [--timestamps]
//   trtmc speak           <bundle.bundle> --audio-in INPUT.wav --audio-out OUTPUT.wav
//   trtmc generate-video  <bundle.bundle> --prompt "text" --output DIR [--num-steps N]
//                        [--num-frames N] [--negative-prompt "text"] [--height N] [--width N]
//   trtmc classify        <bundle.bundle> --image PATH [--benchmark N] [--warmup N]
//   trtmc extract-features <bundle.bundle> --image PATH [--output-json PATH]
//   trtmc geometry        <bundle.bundle> --image PATH --output DIR
//   trtmc detect          <bundle.bundle> --image PATH [--output-json PATH]
//   trtmc inspect         <bundle.bundle> [--list-engines]
//   trtmc version

#include "cli/args.h"
#include "cli/jsonl_io.h"
#include "cli/speech_session_helpers.h"
#include "cli/windows_media.h"
#if defined(_WIN32)
#include "cli/windows_utf8_argv.h"
#endif
#if defined(_WIN32) && defined(TRTMC_LOCKED_H3_RUNTIME)
#include "runtime/platform/windows_process_lockdown.h"
#endif
#include "runtime/platform/dynamic_library.h"
#if __has_include("runtime/models/moge/geometry.h")
#include "runtime/models/moge/geometry.h"
#define TRTMC_CLI_HAS_MOGE_GEOMETRY 1
#else
#define TRTMC_CLI_HAS_MOGE_GEOMETRY 0
#endif
#include "stb_image_write.h"
#include "trtmc/bundle.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/image_features.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/speech_session.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#if defined(_WIN32) && !defined(TRTMC_RUNTIME_ONLY_CLI)
#include <process.h>
#elif !defined(_WIN32) && !defined(TRTMC_RUNTIME_ONLY_CLI)
#include <sys/wait.h>
#endif
#include <system_error>
#include <thread>
#if !defined(_WIN32) && !defined(TRTMC_RUNTIME_ONLY_CLI)
#include <unistd.h>
#endif
#include <utility>
#include <vector>

namespace {

using trtmc::cli::CliArgs;
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
    const auto slash = prefix.find_last_of("/\\");
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
    options.model_plugin_search_paths = args.model_plugin_search_paths;
    return options;
}

std::unique_ptr<trtmc::IPipeline> load_pipeline(const CliArgs& args) {
    return trtmc::load(args.bundle_path, make_load_options(args), args.kernel_bindings_path);
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

    trtmc::load_model_plugin_for_strategy(strategy, args.model_plugin_search_paths);
}

#if !defined(TRTMC_RUNTIME_ONLY_CLI)
std::filesystem::path current_executable_path() {
    return trtmc::internal::current_executable_path();
}
#endif

std::string default_temp_output(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

#if !defined(TRTMC_RUNTIME_ONLY_CLI)
std::string build_python_executable() {
    if (const char* configured = std::getenv("TRTMC_PYTHON_EXECUTABLE");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }

    const auto current_exe = current_executable_path();
    if (current_exe.empty())
#if defined(_WIN32)
        return "python";
#else
        return "python3";
#endif

    std::error_code ec;
    const std::filesystem::path exe_path = std::filesystem::weakly_canonical(current_exe, ec);
    if (!ec && !exe_path.empty()) {
        const std::filesystem::path exe_dir = exe_path.parent_path();
#if defined(_WIN32)
        constexpr const char* python_names[] = {"python.exe", "python3.exe"};
#else
        constexpr const char* python_names[] = {"python3", "python"};
#endif
        for (const char* name : python_names) {
            const std::filesystem::path candidate = exe_dir / name;
            std::error_code exists_ec;
            if (std::filesystem::is_regular_file(candidate, exists_ec)
#if !defined(_WIN32)
                && access(candidate.c_str(), X_OK) == 0
#endif
            ) {
                return candidate.string();
            }
        }
    }

#if defined(_WIN32)
    return "python";
#else
    return "python3";
#endif
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
            pythonpath += trtmc::internal::path_list_separator();
        pythonpath += existing;
    }
    return pythonpath;
}

int run_python_module(const std::vector<std::string>& argv) {
    if (argv.empty()) {
        std::cerr << "Error: empty Python command\n";
        return EXIT_FAILURE;
    }

#if defined(_WIN32)
    struct EnvironmentRestore {
        std::vector<std::pair<std::string, std::optional<std::string>>> values;
        ~EnvironmentRestore() {
            for (auto it = values.rbegin(); it != values.rend(); ++it)
                (void)_putenv_s(it->first.c_str(), it->second ? it->second->c_str() : "");
        }
    } environment;
    const auto set_environment = [&](const char* name, const std::string& value) {
        const char* old = std::getenv(name);
        environment.values.emplace_back(name, old ? std::optional<std::string>(old) : std::nullopt);
        if (_putenv_s(name, value.c_str()) != 0)
            throw std::runtime_error(std::string("failed to set child environment variable ") +
                                     name);
    };
    const std::string pythonpath = build_pythonpath();
    if (!pythonpath.empty())
        set_environment("PYTHONPATH", pythonpath);
    const auto executable = current_executable_path();
    if (!executable.empty())
        set_environment("_TRTMC_INTERNAL_NATIVE_BIN_DIR", executable.parent_path().string());

    std::vector<std::wstring> wide_arguments;
    wide_arguments.reserve(argv.size());
    for (const auto& argument : argv)
        wide_arguments.push_back(std::filesystem::path(argument).wstring());
    std::vector<const wchar_t*> process_arguments;
    process_arguments.reserve(wide_arguments.size() + 1);
    for (const auto& argument : wide_arguments)
        process_arguments.push_back(argument.c_str());
    process_arguments.push_back(nullptr);

    const intptr_t result =
        _wspawnvp(_P_WAIT, wide_arguments.front().c_str(), process_arguments.data());
    if (result == -1) {
        std::cerr << "Error: failed to execute " << argv[0] << ": " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }
    return static_cast<int>(result);
#else
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
        const auto executable = current_executable_path();
        if (!executable.empty()) {
            const std::string native_bin_dir = executable.parent_path().string();
            setenv("_TRTMC_INTERNAL_NATIVE_BIN_DIR", native_bin_dir.c_str(), 1);
        }
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
#endif
}

int cmd_python(const CliArgs& args) {
    std::vector<std::string> command = {
        build_python_executable(),
        "-m",
        "tensorrt_model_connect",
        args.command,
    };
    command.insert(command.end(), args.build_args.begin(), args.build_args.end());
    return run_python_module(command);
}
#endif

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

void write_text_sample_jsonl(std::ostream& out, int32_t id, const std::string& prompt,
                             const trtmc::TextResult& result) {
    out << trtmc::cli::build_text_sample_record(id, prompt, result).dump() << '\n';
}

int cmd_run(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: run requires a .bundle artifact file\n";
        return EXIT_FAILURE;
    }
    if (!trtmc::cli::has_run_input_source(args)) {
        std::cerr
            << "Error: run requires bundle + --prompt, --prompts-file, or --initial-latents-raw\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
    if (!pipeline) {
        std::cerr << "Error: failed to load bundle\n";
        return EXIT_FAILURE;
    }

    const std::string prompt = args.prompt;
    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens =
        args.max_new_tokens > 0 ? args.max_new_tokens : pipeline->default_max_new_tokens();
    cfg.source_language_token_id = args.source_language_token_id;
    cfg.forced_bos_token_id = args.forced_bos_token_id;
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
    cfg.repetition_penalty = args.repetition_penalty;
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
        if (!args.prompts_file.empty()) {
            std::cerr << "Error: --benchmark requires one --prompt, not --prompts-file\n";
            return EXIT_FAILURE;
        }
        // Benchmark mode: warmup, then N timed iterations.
        const int warmup_n = args.warmup > 0 ? args.warmup : 1;
        const int bench_n = args.benchmark;

        std::cerr << "[trtmc.benchmark] warmup=" << warmup_n << " iterations=" << bench_n
                  << " max_new_tokens=" << cfg.max_new_tokens << '\n';

        for (int w = 0; w < warmup_n; ++w)
            pipeline->generate(prompt, cfg);

        std::vector<double> setup_ms_v, prefill_ms_v, decode_ms_v;
        std::size_t generated_tokens = 0;
        setup_ms_v.reserve(static_cast<std::size_t>(bench_n));
        prefill_ms_v.reserve(static_cast<std::size_t>(bench_n));
        decode_ms_v.reserve(static_cast<std::size_t>(bench_n));

        for (int r = 0; r < bench_n; ++r) {
            auto result = pipeline->generate(prompt, cfg);
            generated_tokens += result.token_ids.size();
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
        const double generated_tokens_mean =
            static_cast<double>(generated_tokens) / static_cast<double>(bench_n);
        const double total_decode_ms = std::accumulate(decode_ms_v.begin(), decode_ms_v.end(), 0.0);
        const double tokens_per_sec =
            total_decode_ms > 0.0
                ? static_cast<double>(generated_tokens) / (total_decode_ms / 1000.0)
                : 0.0;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[trtmc.benchmark] setup_ms=" << smean << " prefill_ms=" << pmean
                  << " decode_ms=" << dmean << " generated_tokens_mean=" << generated_tokens_mean
                  << " tokens_per_sec=" << tokens_per_sec << '\n';

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
                const std::string out_dir = args.output_dir.empty()
                                                ? default_temp_output("trtmc_run_output")
                                                : args.output_dir;
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
                const std::string out_dir = args.output_dir.empty()
                                                ? default_temp_output("trtmc_run_output")
                                                : args.output_dir;
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
        std::vector<std::string> text_prompts;
        if (!args.prompts_file.empty()) {
            std::string error;
            text_prompts = trtmc::cli::read_prompts_file(args.prompts_file, error);
            if (text_prompts.empty()) {
                std::cerr << "Error: " << error << '\n';
                return EXIT_FAILURE;
            }
        } else {
            text_prompts.push_back(prompt);
        }

        const int samples_per_prompt = std::max(1, cfg.num_samples);
        const int total_samples = static_cast<int>(text_prompts.size()) * samples_per_prompt;
        if (!cfg.initial_latents.empty() && total_samples > 1) {
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
        } else if (trtmc::cli::text_stdout_requires_jsonl(args, total_samples)) {
            jsonl_out = &std::cout;
        }

        int sample_id = 0;
        for (const auto& text_prompt : text_prompts) {
            for (int i = 0; i < samples_per_prompt; ++i, ++sample_id) {
                trtmc::GenerateConfig sample_cfg = cfg;
                if (cfg.seed >= 0)
                    sample_cfg.seed = cfg.seed + sample_id;
                auto result = pipeline->generate(text_prompt, sample_cfg);
                print_text_timing(result);
                if (jsonl_out) {
                    write_text_sample_jsonl(*jsonl_out, sample_id, text_prompt, result);
                } else {
                    std::cout << result.text << '\n';
                }
            }
        }
        if (jsonl_file.is_open())
            std::cout << "Saved " << args.output_dir << " (" << total_samples << " samples)\n";
    }
    return EXIT_SUCCESS;
}

trtmc::VideoImageInput load_video_image_input(const std::string& path) {
    auto decoded = trtmc::io::read_image(path);
    if (decoded.empty() || decoded.height <= 0 || decoded.width <= 0)
        throw std::runtime_error("failed to decode video-conditioning image: " + path);
    const auto height = static_cast<std::size_t>(decoded.height);
    const auto width = static_cast<std::size_t>(decoded.width);
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        height * width > std::numeric_limits<std::size_t>::max() / 3U ||
        decoded.pixels.size() != height * width * 3U) {
        throw std::runtime_error("invalid RGB image dimensions for video conditioning: " + path);
    }

    trtmc::VideoImageInput result;
    result.pixels = std::move(decoded.pixels);
    result.height = decoded.height;
    result.width = decoded.width;
    result.channels = 3;
    return result;
}

bool is_safe_relative_media_path(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory())
        return false;
    for (const auto& component : path) {
        if (component == "..")
            return false;
    }
    return true;
}

trtmc::VideoClipInput load_native_video_directory(const std::string& directory) {
    const std::filesystem::path root(directory);
    if (!std::filesystem::is_directory(root))
        throw std::runtime_error("reference video is not a directory: " + directory);

    const auto manifest_path = root / "manifest.json";
    std::ifstream manifest_file(manifest_path, std::ios::binary);
    if (!manifest_file)
        throw std::runtime_error("reference video is missing manifest.json: " + directory);

    nlohmann::json manifest;
    try {
        manifest_file >> manifest;
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid reference video manifest " + manifest_path.string() +
                                 ": " + e.what());
    }

    try {
        if (manifest.at("artifact_type").get<std::string>() != "trtmc.video_directory")
            throw std::runtime_error("unsupported artifact_type");
        const auto& video = manifest.at("video");
        if (video.at("frame_pattern").get<std::string>() != "frame_%04d.png")
            throw std::runtime_error("frame_pattern must be frame_%04d.png");

        trtmc::VideoClipInput result;
        result.width = video.at("width").get<int32_t>();
        result.height = video.at("height").get<int32_t>();
        result.channels = video.at("channels").get<int32_t>();
        result.num_frames = video.at("num_frames").get<int32_t>();
        if (video.contains("fps_numerator")) {
            result.fps_numerator = video.at("fps_numerator").get<int32_t>();
            result.fps_denominator = video.value("fps_denominator", 1);
        } else {
            result.fps_numerator = video.at("fps").get<int32_t>();
            result.fps_denominator = 1;
        }
        if (result.width <= 0 || result.height <= 0 || result.channels != 3 ||
            result.num_frames <= 0 || result.fps_numerator <= 0 || result.fps_denominator <= 0) {
            throw std::runtime_error("invalid video dimensions, frame count, or frame rate");
        }

        const auto height = static_cast<std::size_t>(result.height);
        const auto width = static_cast<std::size_t>(result.width);
        const auto frames = static_cast<std::size_t>(result.num_frames);
        if (height > std::numeric_limits<std::size_t>::max() / width ||
            height * width > std::numeric_limits<std::size_t>::max() / 3U ||
            frames > std::numeric_limits<std::size_t>::max() / (height * width * 3U)) {
            throw std::runtime_error("video dimensions overflow the host address space");
        }
        const auto frame_scalars = height * width * 3U;
        result.pixels.reserve(frames * frame_scalars);
        for (int32_t frame = 0; frame < result.num_frames; ++frame) {
            std::ostringstream name;
            name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".png";
            auto decoded = load_video_image_input((root / name.str()).string());
            if (decoded.width != result.width || decoded.height != result.height)
                throw std::runtime_error(
                    "reference video frame dimensions do not match manifest: " + name.str());
            result.pixels.insert(result.pixels.end(),
                                 std::make_move_iterator(decoded.pixels.begin()),
                                 std::make_move_iterator(decoded.pixels.end()));
        }

        if (manifest.contains("audio") && manifest.at("audio").is_object() &&
            manifest.at("audio").value("present", false)) {
            const auto relative_audio =
                std::filesystem::path(manifest.at("audio").at("path").get<std::string>());
            if (!is_safe_relative_media_path(relative_audio))
                throw std::runtime_error("audio path must stay within the video directory");
            result.soundtrack = trtmc::io::read_wav_interleaved((root / relative_audio).string());

            const auto& audio = manifest.at("audio");
            if (audio.contains("sample_rate") &&
                audio.at("sample_rate").get<int32_t>() != result.soundtrack.sample_rate)
                throw std::runtime_error("soundtrack sample rate does not match manifest");
            if (audio.contains("channels") &&
                audio.at("channels").get<int32_t>() != result.soundtrack.channels)
                throw std::runtime_error("soundtrack channel count does not match manifest");
            if (audio.contains("interleaved_sample_count") &&
                audio.at("interleaved_sample_count").get<std::size_t>() !=
                    result.soundtrack.samples.size())
                throw std::runtime_error("soundtrack sample count does not match manifest");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid reference video manifest " + manifest_path.string() +
                                 ": " + e.what());
    }
}

double audio_duration_seconds(const trtmc::AudioResult& audio, const std::string& label) {
    if (audio.samples.empty() || audio.sample_rate <= 0 || audio.channels <= 0 ||
        audio.samples.size() % static_cast<std::size_t>(audio.channels) != 0) {
        throw std::runtime_error(label + " has invalid interleaved audio metadata");
    }
    return static_cast<double>(audio.samples.size()) /
           (static_cast<double>(audio.sample_rate) * static_cast<double>(audio.channels));
}

void validate_reference_duration(double seconds, const std::string& label) {
    constexpr double kMinReferenceSeconds = 2.0;
    constexpr double kMaxReferenceSeconds = 15.0;
    if (!std::isfinite(seconds) || seconds < kMinReferenceSeconds ||
        seconds > kMaxReferenceSeconds) {
        std::ostringstream message;
        message << label << " duration must be in [2, 15] seconds; got " << std::fixed
                << std::setprecision(3) << seconds;
        throw std::runtime_error(message.str());
    }
}

trtmc::VideoGenerationRequest make_video_generation_request(const CliArgs& args,
                                                            trtmc::GenerateConfig config) {
    trtmc::VideoGenerationRequest request;
    request.prompt = args.prompt;
    request.config = std::move(config);

    const bool has_key_frames = !args.first_frame_path.empty() || !args.last_frame_path.empty();
    if (has_key_frames) {
        request.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
        if (!args.first_frame_path.empty())
            request.first_frame = load_video_image_input(args.first_frame_path);
        if (!args.last_frame_path.empty())
            request.last_frame = load_video_image_input(args.last_frame_path);
        return request;
    }

    if (args.video_references.empty())
        return request;

    request.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    request.references.reserve(args.video_references.size());
    std::size_t image_count = 0;
    std::size_t video_count = 0;
    std::size_t explicit_audio_count = 0;
    double total_video_seconds = 0.0;
    double total_explicit_audio_seconds = 0.0;
    for (const auto& argument : args.video_references) {
        trtmc::VideoReferenceInput reference;
        switch (argument.kind) {
        case trtmc::cli::VideoReferenceArgKind::kImage:
            reference.kind = trtmc::VideoReferenceKind::kImage;
            reference.image = load_video_image_input(argument.path);
            ++image_count;
            break;
        case trtmc::cli::VideoReferenceArgKind::kVideoDirectory: {
            reference.kind = trtmc::VideoReferenceKind::kVideo;
            reference.video = std::filesystem::is_directory(argument.path)
                                  ? load_native_video_directory(argument.path)
                                  : trtmc::cli::read_video_file(argument.path);
            ++video_count;
            const double seconds = static_cast<double>(reference.video.num_frames) *
                                   static_cast<double>(reference.video.fps_denominator) /
                                   static_cast<double>(reference.video.fps_numerator);
            validate_reference_duration(seconds, "reference video " + argument.path);
            total_video_seconds += seconds;
            if (!reference.video.soundtrack.samples.empty()) {
                // A soundtrack is attached metadata of this video reference,
                // not a separate public audio reference. Validate its decoded
                // shape/rate, but do not apply the explicit-audio 2..15 second
                // range or its aggregate/count limits.
                (void)audio_duration_seconds(reference.video.soundtrack,
                                             "reference video soundtrack " + argument.path);
            }
            break;
        }
        case trtmc::cli::VideoReferenceArgKind::kAudio: {
            reference.kind = trtmc::VideoReferenceKind::kAudio;
            reference.audio = trtmc::io::read_wav_interleaved(argument.path);
            const double seconds =
                audio_duration_seconds(reference.audio, "reference audio " + argument.path);
            validate_reference_duration(seconds, "reference audio " + argument.path);
            total_explicit_audio_seconds += seconds;
            ++explicit_audio_count;
            break;
        }
        }
        request.references.push_back(std::move(reference));
    }

    if (image_count > 9 || video_count > 3 || request.references.size() > 12)
        throw std::runtime_error("Ref2VA reference count exceeds the public H3-Base limits");
    if (explicit_audio_count > 3)
        throw std::runtime_error("Ref2VA accepts at most 3 explicit reference audio files");
    if (total_video_seconds > 15.0)
        throw std::runtime_error(
            "Ref2VA total reference-video duration must not exceed 15 seconds");
    if (total_explicit_audio_seconds > 15.0)
        throw std::runtime_error(
            "Ref2VA total explicit reference-audio duration must not exceed 15 seconds");
    return request;
}

struct VideoResultValidation {
    std::size_t num_frames{0};
    std::size_t frame_pixels{0};
    std::size_t required_pixels{0};
    std::size_t nonfinite_rgb{0};
    std::size_t nonfinite_audio{0};
    double video_seconds{0.0};
    double audio_seconds{0.0};
    bool has_audio{false};
};

bool validate_generated_video_result(const trtmc::VideoResult& result, bool require_h3_contract,
                                     int32_t expected_frames, int32_t expected_height,
                                     int32_t expected_width, VideoResultValidation& validation,
                                     std::string& error) {
    const auto& frames = result.frames;
    if (frames.height <= 0 || frames.width <= 0 || frames.num_frames <= 0 || frames.channels != 3) {
        error = "invalid frame metadata";
        return false;
    }
    if (result.fps < 0) {
        error = "negative frame rate";
        return false;
    }
    if (require_h3_contract && result.fps != 24) {
        error = "MiniMax-H3 output is not 24 fps";
        return false;
    }
    if (require_h3_contract &&
        (frames.num_frames != expected_frames || frames.height != expected_height ||
         frames.width != expected_width)) {
        std::ostringstream message;
        message << "MiniMax-H3 output geometry " << frames.width << 'x' << frames.height << 'x'
                << frames.num_frames << " does not match requested aligned geometry "
                << expected_width << 'x' << expected_height << 'x' << expected_frames;
        error = message.str();
        return false;
    }

    const auto height = static_cast<std::size_t>(frames.height);
    const auto width = static_cast<std::size_t>(frames.width);
    validation.num_frames = static_cast<std::size_t>(frames.num_frames);
    constexpr std::size_t kRgbChannels = 3;
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        height * width > std::numeric_limits<std::size_t>::max() / kRgbChannels) {
        error = "frame dimensions overflow the host address space";
        return false;
    }
    validation.frame_pixels = height * width * kRgbChannels;
    if (validation.num_frames > std::numeric_limits<std::size_t>::max() / validation.frame_pixels) {
        error = "frame count overflows the host address space";
        return false;
    }
    validation.required_pixels = validation.num_frames * validation.frame_pixels;
    if (frames.pixels.size() != validation.required_pixels) {
        error = "pixel count does not exactly match frame metadata";
        return false;
    }

    validation.has_audio = !result.audio.samples.empty();
    if (require_h3_contract && !validation.has_audio) {
        error = "MiniMax-H3 output is missing its synchronized audio track";
        return false;
    }
    if (validation.has_audio &&
        (result.audio.sample_rate <= 0 || result.audio.channels <= 0 ||
         result.audio.samples.size() % static_cast<std::size_t>(result.audio.channels) != 0 ||
         result.audio.samples.size() >
             static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) ||
         result.audio.num_samples != static_cast<int32_t>(result.audio.samples.size()))) {
        error = "invalid interleaved audio metadata";
        return false;
    }
    if (require_h3_contract && (result.audio.sample_rate != 32000 || result.audio.channels != 2)) {
        error = "MiniMax-H3 output is not stereo 32 kHz audio";
        return false;
    }
    if (result.fps > 0)
        validation.video_seconds =
            static_cast<double>(frames.num_frames) / static_cast<double>(result.fps);
    if (validation.has_audio) {
        const auto audio_frames =
            result.audio.samples.size() / static_cast<std::size_t>(result.audio.channels);
        validation.audio_seconds =
            static_cast<double>(audio_frames) / static_cast<double>(result.audio.sample_rate);
    }
    if (require_h3_contract &&
        std::abs(validation.video_seconds - validation.audio_seconds) > (1.0 / 24.0)) {
        error = "MiniMax-H3 video and audio durations differ by more than one video frame";
        return false;
    }

    validation.nonfinite_rgb =
        static_cast<std::size_t>(std::count_if(frames.pixels.begin(), frames.pixels.end(),
                                               [](float value) { return !std::isfinite(value); }));
    validation.nonfinite_audio = static_cast<std::size_t>(
        std::count_if(result.audio.samples.begin(), result.audio.samples.end(),
                      [](float value) { return !std::isfinite(value); }));
    if (validation.nonfinite_rgb != 0 || validation.nonfinite_audio != 0) {
        error = "non-finite RGB or audio values";
        return false;
    }
    return true;
}

int cmd_generate_video(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: generate-video requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }
    if (args.benchmark < 0 || args.warmup < 0) {
        std::cerr << "Error: generate-video --benchmark and --warmup must be non-negative\n";
        return EXIT_FAILURE;
    }

    std::optional<std::size_t> png_worker_override;
    if (const char* raw = std::getenv("TRTMC_PNG_WRITE_WORKERS"); raw != nullptr && *raw != '\0') {
        errno = 0;
        char* end = nullptr;
        const auto parsed = std::strtoul(raw, &end, 10);
        if (errno != 0 || end == raw || *end != '\0' || parsed < 1 || parsed > 8) {
            std::cerr << "Error: TRTMC_PNG_WRITE_WORKERS must be an integer in [1, 8]\n";
            return EXIT_FAILURE;
        }
        png_worker_override = static_cast<std::size_t>(parsed);
    }

    const std::string out_dir =
        args.output_dir.empty() ? default_temp_output("trtmc_generate_video") : args.output_dir;

    const auto total_begin = std::chrono::steady_clock::now();
    const auto load_begin = total_begin;
    auto pipeline = load_pipeline(args);
    const auto load_end = std::chrono::steady_clock::now();

    trtmc::GenerateConfig cfg;
    cfg.num_steps = args.num_steps;
    cfg.guidance_scale = args.guidance_scale;
    cfg.seed = args.seed;
    cfg.video_num_frames = args.video_num_frames;
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

    const auto input_decode_begin = std::chrono::steady_clock::now();
    auto request = make_video_generation_request(args, std::move(cfg));
    const auto input_decode_end = std::chrono::steady_clock::now();
    trtmc::VideoResult result;
    VideoResultValidation final_validation;
    const bool require_h3_contract =
        std::strcmp(pipeline->pipeline_type(), "MiniMaxH3Pipeline") == 0;
    int32_t expected_h3_frames = 0;
    int32_t expected_h3_height = 0;
    int32_t expected_h3_width = 0;
    if (require_h3_contract) {
        const int64_t requested_frames =
            request.config.video_num_frames > 0 ? request.config.video_num_frames : 124;
        const int64_t aligned_frames = requested_frames + ((5 - (requested_frames % 17) + 17) % 17);
        if (aligned_frames > std::numeric_limits<int32_t>::max()) {
            std::cerr << "Error: MiniMax-H3 aligned frame count overflows int32\n";
            return EXIT_FAILURE;
        }
        expected_h3_frames = static_cast<int32_t>(aligned_frames);
        expected_h3_height = request.config.height > 0 ? request.config.height : 768;
        expected_h3_width = request.config.width > 0 ? request.config.width : 1344;
    }
    const auto validate_iteration = [&](const char* phase, int index) {
        VideoResultValidation checked;
        std::string error;
        const bool valid =
            validate_generated_video_result(result, require_h3_contract, expected_h3_frames,
                                            expected_h3_height, expected_h3_width, checked, error);
        std::cerr << "[trtmc.video_validation] phase=" << phase << " iteration=" << index
                  << " rgb_values=" << checked.required_pixels
                  << " audio_values=" << result.audio.samples.size()
                  << " nonfinite_rgb=" << checked.nonfinite_rgb
                  << " nonfinite_audio=" << checked.nonfinite_audio
                  << " video_seconds=" << std::fixed << std::setprecision(6)
                  << checked.video_seconds << " audio_seconds=" << checked.audio_seconds
                  << " status=" << (valid ? "passed" : "failed") << '\n';
        if (!valid) {
            std::cerr << "Error: generate_video returned " << error << '\n';
            return false;
        }
        final_validation = checked;
        return true;
    };
    std::vector<double> benchmark_samples_ms;
    auto generation_begin = std::chrono::steady_clock::now();
    auto generation_end = generation_begin;
    if (args.benchmark > 0) {
        std::cerr << "[trtmc.video_benchmark] warmup=" << args.warmup
                  << " iterations=" << args.benchmark << '\n';
        for (int index = 0; index < args.warmup; ++index) {
            result = {};
            result = pipeline->generate_video(request);
            if (!validate_iteration("warmup", index))
                return EXIT_FAILURE;
        }
        benchmark_samples_ms.reserve(static_cast<std::size_t>(args.benchmark));
        for (int index = 0; index < args.benchmark; ++index) {
            // Keep destruction of the prior host result outside the public-call timer.
            result = {};
            generation_begin = std::chrono::steady_clock::now();
            result = pipeline->generate_video(request);
            generation_end = std::chrono::steady_clock::now();
            if (!validate_iteration("measured", index))
                return EXIT_FAILURE;
            const double sample_ms =
                std::chrono::duration<double, std::milli>(generation_end - generation_begin)
                    .count();
            benchmark_samples_ms.push_back(sample_ms);
            std::cerr << std::fixed << std::setprecision(3)
                      << "[trtmc.video_benchmark_sample] iteration=" << index
                      << " generation_ms=" << sample_ms << '\n';
        }
        auto sorted_samples = benchmark_samples_ms;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        const std::size_t middle = sorted_samples.size() / 2;
        const double median_ms = sorted_samples.size() % 2 == 0
                                     ? (sorted_samples[middle - 1] + sorted_samples[middle]) / 2.0
                                     : sorted_samples[middle];
        const double mean_ms = std::accumulate(sorted_samples.begin(), sorted_samples.end(), 0.0) /
                               static_cast<double>(sorted_samples.size());
        std::cerr << std::fixed << std::setprecision(3)
                  << "[trtmc.video_benchmark_summary] iterations=" << sorted_samples.size()
                  << " median_ms=" << median_ms << " mean_ms=" << mean_ms
                  << " min_ms=" << sorted_samples.front() << " max_ms=" << sorted_samples.back()
                  << '\n';
    } else {
        generation_begin = std::chrono::steady_clock::now();
        result = pipeline->generate_video(request);
        generation_end = std::chrono::steady_clock::now();
        if (!validate_iteration("generation", 0))
            return EXIT_FAILURE;
    }
    const auto& frames = result.frames;
    const auto num_frames = final_validation.num_frames;
    const auto frame_pixels = final_validation.frame_pixels;
    const bool has_audio = final_validation.has_audio;

    std::cout << "Generated video: " << frames.width << "x" << frames.height << " ("
              << frames.num_frames << " frames";
    if (result.fps > 0)
        std::cout << " at " << result.fps << " fps";
    std::cout << ")\n";

    const auto elapsed_ms = [](const auto begin, const auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    if (trtmc::cli::is_mp4_path(out_dir)) {
        const auto output_begin = std::chrono::steady_clock::now();
        try {
            trtmc::cli::write_mp4(result, out_dir);
        } catch (const std::exception& error) {
            std::cerr << "Error: failed to write native MP4: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
        const auto output_end = std::chrono::steady_clock::now();
        const auto load_ms = elapsed_ms(load_begin, load_end);
        const auto input_decode_ms = elapsed_ms(input_decode_begin, input_decode_end);
        const auto generation_ms = elapsed_ms(generation_begin, generation_end);
        const auto output_ms = elapsed_ms(output_begin, output_end);
        const auto total_ms = elapsed_ms(total_begin, output_end);
        std::cout << "Saved " << out_dir << '\n';
        std::cerr << "[trtmc.video_timing] frames=" << frames.num_frames << " fps=" << result.fps
                  << " audio=" << (has_audio ? 1 : 0) << " workers=0 load_ms=" << std::fixed
                  << std::setprecision(3) << load_ms << " input_decode_ms=" << input_decode_ms
                  << " generation_ms=" << generation_ms << " media_write_ms=" << output_ms
                  << " total_ms=" << total_ms << '\n';
        return EXIT_SUCCESS;
    }

    // Create output directory (including parents) if it doesn't exist.
    std::filesystem::create_directories(out_dir);

    // Each frame in frames.pixels is stored as [H, W, 3] float32 in [0,1],
    // with frames stacked contiguously: total layout is [T, H, W, 3].
    const auto output_begin = std::chrono::steady_clock::now();
    std::vector<std::string> frame_paths(num_frames);
    for (int32_t f = 0; f < frames.num_frames; ++f) {
        std::ostringstream fname;
        fname << out_dir << "/frame_" << std::setw(4) << std::setfill('0') << f << ".png";
        frame_paths[static_cast<std::size_t>(f)] = fname.str();
    }

    std::size_t workers_used = 0;
    if (frames.num_frames > 0) {
        const auto hardware_threads = std::max(1U, std::thread::hardware_concurrency());
        const auto automatic_workers =
            std::min<std::size_t>(8, static_cast<std::size_t>(hardware_threads));
        const auto requested_workers = png_worker_override.value_or(automatic_workers);
        const auto worker_count =
            std::min<std::size_t>(static_cast<std::size_t>(frames.num_frames), requested_workers);

        // Allocate all fallible per-worker storage before any threads start.
        std::vector<std::vector<unsigned char>> rgb_buffers;
        rgb_buffers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker)
            rgb_buffers.emplace_back(frame_pixels);

        std::vector<unsigned char> frame_status(num_frames, 0);
        std::atomic<int32_t> next_frame{0};
        const auto encode_frames = [&](std::vector<unsigned char>& rgb) noexcept {
            while (true) {
                const int32_t f = next_frame.fetch_add(1, std::memory_order_relaxed);
                if (f >= frames.num_frames)
                    return;

                try {
                    const float* src =
                        frames.pixels.data() + static_cast<std::size_t>(f) * frame_pixels;
                    for (std::size_t i = 0; i < frame_pixels; ++i) {
                        const float v = std::max(0.0F, std::min(1.0F, src[i]));
                        rgb[i] = static_cast<unsigned char>(v * 255.0F + 0.5F);
                    }

                    const auto& path = frame_paths[static_cast<std::size_t>(f)];
                    const int stride = frames.width * 3;
                    if (!stbi_write_png(path.c_str(), frames.width, frames.height, 3, rgb.data(),
                                        stride))
                        frame_status[static_cast<std::size_t>(f)] = 1;
                } catch (...) {
                    frame_status[static_cast<std::size_t>(f)] = 2;
                }
            }
        };

        // Keep the calling thread as a worker. If a background thread cannot
        // be created, safely continue with the smaller pool already started.
        std::vector<std::thread> workers;
        workers.reserve(worker_count - 1);
        for (std::size_t worker = 0; worker + 1 < worker_count; ++worker) {
            try {
                workers.emplace_back(encode_frames, std::ref(rgb_buffers[worker]));
            } catch (...) {
                break;
            }
        }
        workers_used = workers.size() + 1;
        encode_frames(rgb_buffers[workers.size()]);
        for (auto& worker : workers)
            worker.join();

        for (std::size_t f = 0; f < frame_status.size(); ++f) {
            if (frame_status[f] == 0)
                continue;
            std::cerr << "Error: failed to " << (frame_status[f] == 1 ? "write " : "encode ")
                      << frame_paths[f] << '\n';
            return EXIT_FAILURE;
        }
    }

    for (const auto& path : frame_paths) {
        std::cout << "Saved " << path << '\n';
    }

    const auto audio_path = (std::filesystem::path(out_dir) / "audio.wav").string();
    if (has_audio) {
        trtmc::io::write_wav(result.audio, audio_path);
        std::cout << "Saved " << audio_path << '\n';
    }

    const auto output_end = std::chrono::steady_clock::now();
    const auto load_ms = elapsed_ms(load_begin, load_end);
    const auto input_decode_ms = elapsed_ms(input_decode_begin, input_decode_end);
    const auto generation_ms = elapsed_ms(generation_begin, generation_end);
    const auto output_ms = elapsed_ms(output_begin, output_end);
    const auto total_ms = elapsed_ms(total_begin, output_end);

    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["artifact_type"] = "trtmc.video_directory";
    const char* request_mode = "t2va";
    if (request.mode == trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio)
        request_mode = "fl2va";
    else if (request.mode == trtmc::VideoGenerationMode::kReferenceToVideoAudio)
        request_mode = "ref2va";
    manifest["request"] = {{"mode", request_mode},
                           {"reference_count", request.references.size()},
                           {"has_first_frame", request.first_frame.has_value()},
                           {"has_last_frame", request.last_frame.has_value()}};
    manifest["video"] = {{"frame_pattern", "frame_%04d.png"}, {"width", frames.width},
                         {"height", frames.height},           {"channels", frames.channels},
                         {"num_frames", frames.num_frames},   {"fps", result.fps}};
    manifest["audio"] = {
        {"present", has_audio},
        {"path", has_audio ? nlohmann::json("audio.wav") : nlohmann::json(nullptr)},
        {"sample_rate", has_audio ? result.audio.sample_rate : 0},
        {"channels", has_audio ? result.audio.channels : 0},
        {"interleaved_sample_count", has_audio ? result.audio.samples.size() : std::size_t{0}},
        {"sample_frames",
         has_audio ? result.audio.samples.size() / static_cast<std::size_t>(result.audio.channels)
                   : std::size_t{0}}};
    manifest["timing_ms"] = {{"pipeline_load", load_ms},
                             {"input_decode", input_decode_ms},
                             {"generation", generation_ms},
                             {"media_write", output_ms},
                             {"total", total_ms}};

    const auto manifest_path = (std::filesystem::path(out_dir) / "manifest.json").string();
    std::ofstream manifest_file(manifest_path, std::ios::binary | std::ios::trunc);
    if (!manifest_file) {
        std::cerr << "Error: cannot open " << manifest_path << " for writing\n";
        return EXIT_FAILURE;
    }
    manifest_file << manifest.dump(2) << '\n';
    if (!manifest_file) {
        std::cerr << "Error: failed while writing " << manifest_path << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Saved " << manifest_path << '\n';

    std::cerr << "[trtmc.video_timing] frames=" << frames.num_frames << " fps=" << result.fps
              << " audio=" << (has_audio ? 1 : 0) << " workers=" << workers_used
              << " load_ms=" << std::fixed << std::setprecision(3) << load_ms
              << " input_decode_ms=" << input_decode_ms << " generation_ms=" << generation_ms
              << " media_write_ms=" << output_ms << " total_ms=" << total_ms << '\n';

    return EXIT_SUCCESS;
}

int cmd_segment(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: segment requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);

    // Load image (HWC float32 in [0,1])
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    auto result = pipeline->segment(image.pixels.data(), image.height, image.width);

    // Save class map as grayscale PNG (pixel value = class index)
    const std::string out_path =
        args.output_dir.empty() ? default_temp_output("seg_output.png") : args.output_dir;
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

int cmd_disparity(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty() || args.right_image_path.empty()) {
        std::cerr << "Error: disparity requires bundle + --image + --right-image\n";
        return EXIT_FAILURE;
    }

    const auto left = trtmc::io::read_image(args.image_path);
    const auto right = trtmc::io::read_image(args.right_image_path);
    if (left.empty() || right.empty()) {
        std::cerr << "Error: failed to load stereo images\n";
        return EXIT_FAILURE;
    }
    if (left.height != right.height || left.width != right.width) {
        std::cerr << "Error: stereo image dimensions must match\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
    const auto result = pipeline->estimate_disparity(left.pixels.data(), right.pixels.data(),
                                                     left.height, left.width);
    const std::string out_path =
        args.output_dir.empty() ? default_temp_output("disparity.f32") : args.output_dir;
    std::ofstream output(out_path, std::ios::binary);
    if (!output || result.disparity.empty()) {
        std::cerr << "Error: failed to create disparity output: " << out_path << '\n';
        return EXIT_FAILURE;
    }
    output.write(reinterpret_cast<const char*>(result.disparity.data()),
                 static_cast<std::streamsize>(result.disparity.size() * sizeof(float)));
    if (!output) {
        std::cerr << "Error: failed to write disparity output: " << out_path << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "{\"output\":\"" << out_path << "\",\"height\":" << result.height
              << ",\"width\":" << result.width << ",\"dtype\":\"float32\"}\n";
    return EXIT_SUCCESS;
}

template <typename Value>
void write_geometry_binary(const std::filesystem::path& path, const std::vector<Value>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create geometry output: " + path.string());
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(Value)));
    }
    output.close();
    if (!output)
        throw std::runtime_error("failed to write geometry output: " + path.string());
}

int cmd_geometry(const CliArgs& args) {
#if !TRTMC_CLI_HAS_MOGE_GEOMETRY
    (void)args;
    std::cerr << "Error: this build does not include the MoGe geometry adapter\n";
    return EXIT_FAILURE;
#else
    if (args.bundle_path.empty() || args.image_path.empty() || args.output_dir.empty()) {
        std::cerr << "Error: geometry requires bundle + --image + --output\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
    auto* estimator = dynamic_cast<trtmc::moge::IGeometryEstimator*>(pipeline.get());
    if (estimator == nullptr) {
        std::cerr << "Error: loaded pipeline does not support monocular geometry\n";
        return EXIT_FAILURE;
    }
    const auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }
    const auto result =
        estimator->estimate_geometry(image.pixels.data(), image.height, image.width);
    if (result.height <= 0 || result.width <= 0) {
        std::cerr << "Error: monocular geometry returned invalid dimensions\n";
        return EXIT_FAILURE;
    }
    const auto area = static_cast<std::size_t>(result.height) * result.width;
    if (result.points.size() != area * 3U || result.depth.size() != area ||
        result.mask.size() != area) {
        std::cerr << "Error: monocular geometry returned incomplete maps\n";
        return EXIT_FAILURE;
    }

    const std::filesystem::path directory(args.output_dir);
    std::filesystem::create_directories(directory);
    const auto points_path = directory / "points.f32";
    const auto depth_path = directory / "depth.f32";
    const auto mask_path = directory / "mask.u8";
    const auto intrinsics_path = directory / "intrinsics.json";
    write_geometry_binary(points_path, result.points);
    write_geometry_binary(depth_path, result.depth);
    write_geometry_binary(mask_path, result.mask);

    const nlohmann::json matrix = {
        {result.intrinsics[0], result.intrinsics[1], result.intrinsics[2]},
        {result.intrinsics[3], result.intrinsics[4], result.intrinsics[5]},
        {result.intrinsics[6], result.intrinsics[7], result.intrinsics[8]},
    };
    const nlohmann::json intrinsics = {
        {"height", result.height},
        {"width", result.width},
        {"intrinsics", matrix},
        {"normalized", true},
    };
    std::ofstream intrinsics_output(intrinsics_path, std::ios::out | std::ios::trunc);
    if (!intrinsics_output)
        throw std::runtime_error("failed to create geometry output: " + intrinsics_path.string());
    intrinsics_output << intrinsics.dump(2) << '\n';
    intrinsics_output.close();
    if (!intrinsics_output)
        throw std::runtime_error("failed to write geometry output: " + intrinsics_path.string());

    const nlohmann::json summary = {
        {"output", directory.string()},
        {"height", result.height},
        {"width", result.width},
    };
    std::cout << summary.dump() << '\n';
    return EXIT_SUCCESS;
#endif
}

int cmd_classify(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: classify requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
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

    std::cout << trtmc::cli::build_classify_record(result).dump() << '\n';
    return EXIT_SUCCESS;
}

void write_image_features_json(std::ostream& out, const trtmc::ImageFeaturesResult& result) {
    out << trtmc::cli::build_image_features_record(result).dump() << '\n';
}

int cmd_extract_features(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: extract-features requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
    auto image = trtmc::io::read_image(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << args.image_path << '\n';
        return EXIT_FAILURE;
    }

    auto* extractor = dynamic_cast<trtmc::IImageFeatureExtractor*>(pipeline.get());
    if (extractor == nullptr) {
        std::cerr << "Error: loaded pipeline does not support image feature extraction\n";
        return EXIT_FAILURE;
    }
    const auto result =
        extractor->extract_image_features(image.pixels.data(), image.height, image.width);
    if (args.output_json.empty()) {
        write_image_features_json(std::cout, result);
        return EXIT_SUCCESS;
    }

    const auto out_path = std::filesystem::path(args.output_json);
    const auto parent = out_path.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream out(out_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: failed to open " << args.output_json << " for writing\n";
        return EXIT_FAILURE;
    }
    write_image_features_json(out, result);
    if (!out) {
        std::cerr << "Error: failed to write " << args.output_json << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Image features saved: " << args.output_json << '\n';
    return EXIT_SUCCESS;
}

int cmd_detect(const CliArgs& args) {
    if (args.bundle_path.empty() || args.image_path.empty()) {
        std::cerr << "Error: detect requires bundle + --image\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
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

    const std::string out_dir =
        args.output_dir.empty() ? default_temp_output("trtmc_masks") : args.output_dir;
    std::filesystem::create_directories(out_dir);

    auto pipeline = load_pipeline(args);
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
//   echo "Hello world" | trtmc serve-audio bundle.bundle > out.raw
//   (or pipe multiple prompts, one per line)
// ---------------------------------------------------------------------------
int cmd_serve_audio(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: serve-audio requires a bundle path\n";
        return EXIT_FAILURE;
    }

    std::cerr << "[serve-audio] Loading bundle: " << args.bundle_path << std::endl;
    auto pipeline = load_pipeline(args);
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

    auto pipeline = load_pipeline(args);

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 0;

    if (args.stream) {
        // Streaming mode: write raw PCM float32 to output file (or stdout
        // placeholder). Codec runs on chunks during decoding for low latency.
        // Pipe output to: aplay -r 22050 -f FLOAT_LE -c 1 -t raw
        const std::string out_path = args.output_dir.empty()
                                         ? default_temp_output("generated_audio_stream.raw")
                                         : args.output_dir;
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
        args.output_dir.empty() ? default_temp_output("generated_audio.wav") : args.output_dir;
    trtmc::io::write_wav(result, out_path);

    std::cout << "Generated " << result.num_samples << " audio samples -> " << out_path << '\n';
    return EXIT_SUCCESS;
}

int cmd_encode(const CliArgs& args) {
    if (args.bundle_path.empty() || args.prompt.empty()) {
        std::cerr << "Error: encode requires bundle + --prompt\n";
        return EXIT_FAILURE;
    }

    auto pipeline = load_pipeline(args);
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

    auto pipeline = load_pipeline(args);
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

    auto pipeline = load_pipeline(args);
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
        std::cerr << "Error: solve requires a .bundle artifact file\n";
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

    auto pipeline = load_pipeline(args);
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

    auto pipeline = load_pipeline(args);

    int32_t max_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : 224;

    if (args.stream) {
        if (audio_paths.size() != 1) {
            std::cerr << "Error: --stream accepts exactly one --audio input\n";
            return EXIT_FAILURE;
        }
        const bool has_offline_only_controls =
            args.beam_size != 1 || args.length_penalty != 1.0F ||
            args.beam_fallback_max_size != 0 || args.transcription_task != "transcribe" ||
            !args.punctuation || args.timestamps || args.max_input_seconds > 0.0F ||
            args.segment_length_seconds > 0.0F || args.segment_min_seconds > 0.0F ||
            args.segment_overlap_seconds > 0.0F || args.lcs_merge ||
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
        request.config.length_penalty = args.length_penalty;
        request.config.beam_fallback_max_size = args.beam_fallback_max_size;
        request.config.source_language = args.source_language;
        request.config.target_language = args.target_language;
        request.config.task = args.transcription_task == "translate"
                                  ? trtmc::TranscriptionTask::kTranslate
                                  : trtmc::TranscriptionTask::kTranscribe;
        request.config.punctuation = args.punctuation;
        request.config.timestamps = args.timestamps;
        request.config.max_input_duration_seconds = args.max_input_seconds;
        request.config.segment_duration_seconds = args.segment_length_seconds;
        request.config.segment_min_duration_seconds = args.segment_min_seconds;
        request.config.segment_overlap_seconds = args.segment_overlap_seconds;
        request.config.lcs_merge = args.lcs_merge;
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

    auto pipeline = load_pipeline(args);

    auto audio = trtmc::io::read_wav(args.audio_in);

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = args.max_new_tokens > 0 ? args.max_new_tokens : -1;
    cfg.tail_frames = args.tail_frames;
    cfg.seed = args.seed;

    trtmc::AudioResult result;
    std::string agent_text;
    trtmc::SpeechSessionConfig session_config;
    session_config.input_sample_rate = audio.sample_rate;
    session_config.output_sample_rate = 0;
    session_config.emit_agent_audio = true;
    session_config.emit_agent_text = true;
    session_config.emit_user_transcript = false;
    session_config.enable_barge_in = false;
    session_config.seed = cfg.seed >= 0 ? cfg.seed : 0;
    // The CLI appends --tail-frames explicitly. Do not let finish_input()
    // add the session response tail a second time.
    session_config.finish_tail_frames = 0;
    auto session = trtmc::cli::create_cli_speech_session(*pipeline, session_config);
    if (session) {
        session->append_audio(audio.samples.data(), static_cast<int32_t>(audio.samples.size()));
        const int32_t tail_samples =
            trtmc::cli::speech_tail_frame_samples(session_config.input_sample_rate);
        std::vector<float> silence(static_cast<std::size_t>(tail_samples), 0.0F);
        for (int32_t frame = 0; frame < std::max(cfg.tail_frames, 0); ++frame)
            session->append_audio(silence.data(), tail_samples);
        session->finish_input();
        std::vector<trtmc::SpeechSessionEvent> completed_events;
        bool input_completed = false;
        while (!input_completed) {
            auto events = session->wait_events(-1);
            for (const auto& event : events) {
                if (event.kind == trtmc::SpeechSessionEventKind::kInputFinished)
                    input_completed = true;
                if (event.kind == trtmc::SpeechSessionEventKind::kError)
                    throw std::runtime_error(event.text.empty() ? "speech session failed"
                                                                : event.text);
                if (event.kind == trtmc::SpeechSessionEventKind::kCancelled)
                    throw std::runtime_error("speech session was cancelled");
            }
            completed_events.insert(completed_events.end(), std::make_move_iterator(events.begin()),
                                    std::make_move_iterator(events.end()));
        }
        auto aggregate = trtmc::cli::aggregate_speech_session_events(std::move(completed_events),
                                                                     audio.sample_rate);
        result = std::move(aggregate.audio);
        agent_text = std::move(aggregate.agent_text);
    } else {
        result = pipeline->speak(audio.samples.data(), static_cast<int32_t>(audio.samples.size()),
                                 cfg, audio.sample_rate);
    }

    const std::string out_path =
        args.audio_out.empty() ? default_temp_output("speech_output.wav") : args.audio_out;
    trtmc::io::write_wav(result, out_path);

    if (!agent_text.empty())
        std::cout << "Agent text: " << agent_text << '\n';
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
              << "Size" << " " << std::left << std::setw(16) << "Role" << '\n';
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
        std::cerr << "Error: not a valid .bundle artifact: " << args.bundle_path << '\n';
        return EXIT_FAILURE;
    }

    try {
        const auto info = trtmc::InspectBundle(args.bundle_path);
        if (args.validate_runtime) {
            auto pipeline = load_pipeline(args);
            if (pipeline == nullptr)
                throw std::runtime_error("runtime validation returned a null pipeline");
            std::cout << "Runtime validation:  passed (" << pipeline->pipeline_type() << ")\n";
        }
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
        std::cout << "Max cache length:   " << info.max_cache_length << '\n';
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
    preload_cli_config_schema_owner(args);
    if (trtmc::config::SchemaRegistry::instance().registered_namespaces().empty()) {
        std::cerr << "[trtmc] --config/--set accepted but no config schemas are "
                     "registered yet; values have no effect."
                  << '\n';
        return EXIT_SUCCESS;
    }
    try {
        auto bundle = trtmc::config::resolve_cli_config(args.config_path, args.set_tokens);
#if defined(TRTMC_LOCKED_H3_RUNTIME)
        // The locked runtime validates CLI configuration here, while the
        // pipeline applies it later. It must not emit effective-config
        // sidecars into the attested package or beside a user bundle.
        (void)bundle;
#else
        if (!args.bundle_path.empty()) {
            const auto sidecar =
                trtmc::config::try_write_effective_config_next_to(bundle, args.bundle_path);
            if (sidecar.path) {
                std::cerr << "[trtmc] Wrote effective config: " << *sidecar.path << '\n';
            } else {
                std::cerr << "[trtmc.config] Failed to write effective config sidecar: "
                          << sidecar.error
                          << "\n          Command will continue with resolved "
                             "runtime config.\n";
            }
        }
#endif
    } catch (const std::exception& e) {
        std::cerr << "Error resolving config: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int run_cli(int argc, char** argv) {
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
#if !defined(TRTMC_RUNTIME_ONLY_CLI)
        if (args.command == "build" || args.command == "graph")
            return cmd_python(args);
#endif
        if (args.command == "run")
            return cmd_run(args);
        if (args.command == "encode")
            return cmd_encode(args);
        if (args.command == "segment")
            return cmd_segment(args);
        if (args.command == "disparity")
            return cmd_disparity(args);
        if (args.command == "geometry")
            return cmd_geometry(args);
        if (args.command == "segment-prompted")
            return cmd_segment_prompted(args);
        if (args.command == "classify")
            return cmd_classify(args);
        if (args.command == "extract-features")
            return cmd_extract_features(args);
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

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    try {
        trtmc::internal::enforce_locked_h3_process_policy();
    } catch (const std::exception& error) {
        std::cerr << "Error: unable to enforce the locked MiniMax-H3 process policy: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
#endif
    try {
        trtmc::cli::Utf8CommandLine command_line(argc, argv);
        return run_cli(command_line.argc(), command_line.argv());
    } catch (const std::exception& error) {
        std::cerr << "Error: unable to decode the Windows command line as UTF-8: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
#else
int main(int argc, char** argv) {
    return run_cli(argc, argv);
}
#endif

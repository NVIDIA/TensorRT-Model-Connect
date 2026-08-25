/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::cli {

enum class ReferenceInputKind {
    kImage,
    kAudio,
    kVideo,
};

struct ReferenceInput {
    ReferenceInputKind kind{ReferenceInputKind::kImage};
    std::string path;
};

struct CliArgs {
    std::string command;
    std::vector<std::string> build_args;
    std::string bundle_path;
    std::string prompt;
    bool prompt_provided{false};
    std::string hf_python;
    std::uint64_t kv_cache_size_bytes{0};
    std::string image_path;
    std::string first_image_path;
    std::string last_image_path;
    // Ref2VA reference order is semantically significant. Keep all reference
    // kinds in one encounter-ordered vector rather than grouping by flag.
    std::vector<ReferenceInput> reference_inputs;
    std::string right_image_path;
    std::string lora_adapter_path;
    std::string lora_adapter_id{"default"};
    std::string output_dir;
    std::string output_json;
    std::string initial_latents_raw;
    std::string condition_latents_raw;
    std::string condition_mask_raw;
    std::string sampling_steps_raw;
    std::string sde_noise_raw;
    std::string document;
    std::string audio_in;
    std::vector<std::string> audio_inputs;
    std::string audio_out; // --audio-out/--audio-output
    std::string field_input;
    std::string branch_input;
    std::string trunk_input;
    int tail_frames{0};
    float point_x{0.5F};
    float point_y{0.5F};
    bool is_foreground{true};
    int max_new_tokens{0};
    int source_language_token_id{-1};
    int forced_bos_token_id{-1};
    int block_length{0};
    int num_samples{1};
    int benchmark{0}; // >0: run N timed iterations after warmup
    int warmup{1};    // number of warmup iterations before timing
    float temperature{1.0F};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int top_k{1};
    int seed{-1};
    int num_steps{-1};
    float guidance_scale{-1.0F};
    float sde_gamma{-1.0F};
    float conf_threshold{-1.0F};
    float cfg_scale{-1.0F};
    std::string generation_mode;
    // Image-generation extras.
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
    std::string language;
    int beam_size{1};
    float length_penalty{1.0F};
    int beam_fallback_max_size{0};
    std::string source_language{"en"};
    std::string target_language{"en"};
    std::string transcription_task{"transcribe"};
    bool punctuation{true};
    bool timestamps{false};
    float max_input_seconds{0.0F};
    float segment_length_seconds{0.0F};
    float segment_min_seconds{0.0F};
    float segment_overlap_seconds{0.0F};
    bool lcs_merge{false};
    std::string runtime_cache;
    std::string kernel_bindings_path;
    std::vector<std::string> backend_search_paths;
    std::vector<std::string> model_plugin_search_paths;
    bool cuda_graphs{false};
    bool list_engines{false};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
    // Generic config surface -- see include/trtmc/config/cli_support.h.
    // New feature knobs should generally prefer these over adding flags.
    std::string config_path;
    std::vector<std::string> set_tokens;
    // Diffusion batch-inference knobs (PR 3 — `trtmc run` batch dispatch).
    // `num_images` is the requested batch count when `prompts_file` is empty.
    // `prompts_file` (one prompt per line) is mutually exclusive with
    // `--prompt`. `seed_list`, when non-empty, supplies per-sample seeds
    // explicitly and must match the total batch count at dispatch time.
    int num_images{1};
    std::string prompts_file;
    std::vector<std::uint64_t> seed_list;
};

inline bool has_run_input_source(const CliArgs& args) {
    return args.prompt_provided || !args.prompts_file.empty() || !args.initial_latents_raw.empty();
}

inline bool text_stdout_requires_jsonl(const CliArgs& args, int total_samples) {
    return !args.prompts_file.empty() || total_samples > 1;
}

std::optional<std::uint64_t> parse_byte_size(const std::string& text);
// Parse a CSV of unsigned-64 integers (e.g. "0,1,2"). Returns nullopt when
// any token fails to parse. Empty string returns an empty vector wrapped
// in an optional (caller should treat empty CSV the same as no flag).
std::optional<std::vector<std::uint64_t>> parse_seed_csv(const std::string& text);
// Read one prompt per line from `path`. On failure returns an empty vector
// and writes a human-readable message to `error`. Trailing newlines and
// fully-blank lines are stripped; interior blank lines are kept verbatim.
std::vector<std::string> read_prompts_file(const std::string& path, std::string& error);
void print_usage();
CliArgs parse_args(int argc, char** argv);

} // namespace trtmc::cli

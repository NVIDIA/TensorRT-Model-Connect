#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::cli {

struct CliArgs {
    std::string command;
    std::vector<std::string> build_args;
    std::string bundle_path;
    std::string prompt;
    bool prompt_provided{false};
    std::string hf_python;
    std::uint64_t kv_cache_size_bytes{0};
    std::string image_path;
    std::string output_dir;
    std::string output_json;
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
    int block_length{0};
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
    std::string generation_mode;
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
    bool list_engines{false};
    bool show_help{false};
    bool parse_error{false};
    std::string error_message;
    // Generic config surface -- see include/trtmc/config/cli_support.h.
    // New feature knobs should generally prefer these over adding flags.
    std::string config_path;
    std::vector<std::string> set_tokens;
};

std::optional<std::uint64_t> parse_byte_size(const std::string& text);
void print_usage();
CliArgs parse_args(int argc, char** argv);

} // namespace trtmc::cli

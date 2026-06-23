#pragma once

#include "runtime/models/bark/bark_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct BarkCoarsePlan {
    std::vector<int32_t> remapped_semantic;
    int32_t total_steps{0};
    int32_t num_windows{0};
};

struct BarkCoarseWindowPlan {
    std::vector<int32_t> input_tokens;
    int32_t start_generated_count{0};
    int32_t generated_this_window{0};
};

struct BarkCodecPlan {
    bool use_fine_codes{false};
    int32_t frame_count{0};
};

struct BarkFinePlan {
    int32_t n_frames_raw{0};
    int32_t n_frames{0};
    int32_t actual_frames{0};
    bool should_run_trt{false};
    int32_t first_predicted_codebook{2};
    int32_t last_predicted_codebook{8};
};

inline std::vector<int32_t> remap_bark_semantic_tokens(const std::vector<int32_t>& semantic_tokens,
                                                       const BarkConfig& cfg) {
    std::vector<int32_t> remapped;
    remapped.reserve(semantic_tokens.size());
    for (const int32_t token : semantic_tokens) {
        remapped.push_back(token == cfg.semantic_pad_token ? cfg.coarse_semantic_pad_token : token);
    }
    return remapped;
}

inline int32_t compute_bark_coarse_steps(int32_t semantic_len, const BarkConfig& cfg) {
    const float scaled =
        static_cast<float>(semantic_len) * cfg.coarse_rate_hz / cfg.semantic_rate_hz;
    const int32_t frames = static_cast<int32_t>(std::floor(scaled));
    return std::max(frames * cfg.n_coarse_codebooks, 0);
}

inline std::vector<int32_t>
build_bark_coarse_input_tokens(const std::vector<int32_t>& remapped_semantic,
                               const std::vector<int32_t>& coarse_tokens, const BarkConfig& cfg) {
    const int32_t semantic_len = static_cast<int32_t>(remapped_semantic.size());
    const int32_t total_generated = static_cast<int32_t>(coarse_tokens.size());
    const int32_t max_semantic_history = static_cast<int32_t>(std::floor(
        static_cast<float>(cfg.max_coarse_history) * cfg.semantic_rate_hz / cfg.coarse_rate_hz));
    const int32_t semantic_idx = static_cast<int32_t>(std::round(
        static_cast<float>(total_generated) * cfg.semantic_rate_hz / cfg.coarse_rate_hz));
    const int32_t semantic_start = std::max(0, semantic_idx - max_semantic_history);
    const int32_t semantic_context_len =
        std::min(semantic_len - semantic_start, cfg.max_coarse_input_length);

    std::vector<int32_t> input_tokens;
    input_tokens.reserve(static_cast<std::size_t>(cfg.max_coarse_input_length) + 1U +
                         static_cast<std::size_t>(cfg.max_coarse_history));

    for (int32_t index = semantic_start; index < semantic_start + semantic_context_len; ++index) {
        input_tokens.push_back(remapped_semantic[index]);
    }
    for (int32_t index = semantic_context_len; index < cfg.max_coarse_input_length; ++index) {
        input_tokens.push_back(cfg.coarse_semantic_pad_token);
    }

    input_tokens.push_back(cfg.coarse_infer_token);

    const int32_t history_start = std::max(0, total_generated - cfg.max_coarse_history);
    for (int32_t index = history_start; index < total_generated; ++index) {
        input_tokens.push_back(coarse_tokens[index]);
    }
    return input_tokens;
}

inline BarkCoarsePlan make_bark_coarse_plan(const std::vector<int32_t>& semantic_tokens,
                                            const BarkConfig& cfg) {
    BarkCoarsePlan plan;
    plan.remapped_semantic = remap_bark_semantic_tokens(semantic_tokens, cfg);
    plan.total_steps =
        compute_bark_coarse_steps(static_cast<int32_t>(plan.remapped_semantic.size()), cfg);
    if (plan.total_steps > 0) {
        plan.num_windows = static_cast<int32_t>(
            std::ceil(static_cast<float>(plan.total_steps) / cfg.sliding_window_len));
    }
    return plan;
}

inline BarkCoarseWindowPlan
make_bark_coarse_window_plan(const BarkCoarsePlan& coarse_plan,
                             const std::vector<int32_t>& generated_tokens, const BarkConfig& cfg) {
    BarkCoarseWindowPlan plan;
    plan.start_generated_count = static_cast<int32_t>(generated_tokens.size());
    plan.generated_this_window =
        std::min(cfg.sliding_window_len, coarse_plan.total_steps - plan.start_generated_count);
    if (plan.generated_this_window <= 0) {
        plan.generated_this_window = 0;
        return plan;
    }
    plan.input_tokens =
        build_bark_coarse_input_tokens(coarse_plan.remapped_semantic, generated_tokens, cfg);
    return plan;
}

inline int32_t bark_coarse_codebook_index(int32_t total_generated, const BarkConfig& cfg) {
    return total_generated % std::max(cfg.n_coarse_codebooks, 1);
}

inline BarkCodecPlan make_bark_codec_plan(const std::vector<int32_t>& fine_codes,
                                          bool has_fine_engine,
                                          const std::vector<int32_t>& coarse_tokens,
                                          int32_t n_coarse_codebooks) {
    BarkCodecPlan plan;
    if (has_fine_engine) {
        plan.frame_count = static_cast<int32_t>(fine_codes.size()) / 8;
        plan.use_fine_codes = plan.frame_count > 0;
        if (plan.use_fine_codes) {
            return plan;
        }
    }

    plan.frame_count = n_coarse_codebooks > 0
                           ? static_cast<int32_t>(coarse_tokens.size()) / n_coarse_codebooks
                           : 0;
    return plan;
}

inline BarkFinePlan make_bark_fine_plan(const BarkConfig& cfg, std::size_t coarse_token_count,
                                        bool has_fine_engine, bool has_fine_context) {
    BarkFinePlan plan;
    plan.n_frames_raw = cfg.n_coarse_codebooks > 0
                            ? static_cast<int32_t>(coarse_token_count) / cfg.n_coarse_codebooks
                            : 0;
    plan.n_frames = std::min(plan.n_frames_raw,
                             cfg.fine_seq_length > 0 ? cfg.fine_seq_length : plan.n_frames_raw);
    plan.actual_frames = std::min(plan.n_frames, cfg.fine_seq_length);
    plan.should_run_trt =
        has_fine_engine && has_fine_context && cfg.fine_seq_length > 0 && plan.n_frames > 0;
    return plan;
}

inline std::vector<int32_t> initialize_bark_fine_codes(const std::vector<int32_t>& coarse_tokens,
                                                       int32_t n_frames, const BarkConfig& cfg) {
    std::vector<int32_t> codes(static_cast<std::size_t>(8) * n_frames, cfg.codebook_size);
    for (int32_t token_index = 0; token_index < n_frames * cfg.n_coarse_codebooks; ++token_index) {
        const int32_t codebook = token_index % cfg.n_coarse_codebooks;
        const int32_t frame = token_index / cfg.n_coarse_codebooks;
        int32_t raw =
            coarse_tokens[token_index] - cfg.semantic_vocab_size - codebook * cfg.codebook_size;
        raw = std::max(0, std::min(raw, cfg.codebook_size - 1));
        codes[static_cast<std::size_t>(codebook) * n_frames + frame] = raw;
    }
    return codes;
}

inline std::vector<int32_t> make_bark_codec_input_codes(const std::vector<int32_t>& codes_flat,
                                                        int32_t source_codebooks, int32_t n_frames,
                                                        int32_t target_codebooks,
                                                        int32_t max_frames, int32_t actual_frames) {
    std::vector<int32_t> input_codes(static_cast<std::size_t>(target_codebooks) * max_frames, 0);
    for (int32_t codebook = 0; codebook < std::min(source_codebooks, target_codebooks);
         ++codebook) {
        for (int32_t frame = 0; frame < actual_frames; ++frame) {
            input_codes[static_cast<std::size_t>(codebook) * max_frames + frame] =
                codes_flat[static_cast<std::size_t>(codebook) * n_frames + frame];
        }
    }
    return input_codes;
}

} // namespace trtmc

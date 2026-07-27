/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/bark/pipeline.h"

#include "runtime/models/bark/bark_generation_plan.h"
#include "runtime/models/bark/decode_runtime.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace trtmc {

namespace {

constexpr int32_t kBarkMaxTextLen = 256;
constexpr float kNegInf = -1e9F;

// ─── Bark helpers ───

void copy_embed_row(const float* table, int32_t token_id, int32_t hidden_size, float* out) {
    const auto offset = static_cast<std::size_t>(token_id) * static_cast<std::size_t>(hidden_size);
    std::memcpy(out, table + offset, static_cast<std::size_t>(hidden_size) * sizeof(float));
}

void sum_embed_rows(const float* a, const float* b, int32_t hidden_size, float* out) {
    for (int32_t i = 0; i < hidden_size; ++i) {
        out[i] = a[i] + b[i];
    }
}

int32_t semantic_text_token(const std::vector<int32_t>& text_ids, int32_t pos, int32_t copy_len,
                            const BarkConfig& cfg) {
    if (pos < copy_len && text_ids[pos] != 0) {
        return text_ids[pos] + cfg.text_encoding_offset;
    }
    return cfg.text_pad_token;
}

bool semantic_eos_threshold_hit(const std::vector<float>& logits, const BarkConfig& cfg) {
    if (cfg.min_eos_p <= 0.0F) {
        return false;
    }

    float max_val = *std::max_element(logits.begin(), logits.begin() + cfg.semantic_pad_token + 1);
    float sum_exp = 0.0F;
    for (int32_t i = 0; i <= cfg.semantic_pad_token; ++i) {
        sum_exp += std::exp(logits[i] - max_val);
    }
    const float eos_p =
        std::exp(logits[cfg.semantic_pad_token] - max_val) / std::max(sum_exp, 1e-10F);
    return eos_p > cfg.min_eos_p;
}

void suppress_semantic_logits(std::vector<float>& logits, int32_t semantic_pad_token) {
    for (int32_t i = semantic_pad_token + 1; i < static_cast<int32_t>(logits.size()); ++i) {
        logits[i] = kNegInf;
    }
}

void mask_coarse_logits_for_codebook(std::vector<float>& logits, int32_t codebook_idx,
                                     const BarkConfig& cfg) {
    const int32_t cb_start = cfg.semantic_vocab_size + codebook_idx * cfg.codebook_size;
    const int32_t cb_end = cb_start + cfg.codebook_size;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
        if (i < cb_start || i >= cb_end) {
            logits[i] = kNegInf;
        }
    }
}

// Dump tokens to ``<dump_path><suffix>`` when dump_path is non-empty.
// Replaces the TRTMC_BARK_DUMP env var. Value flows in through the
// ``audio_bark.dump_path`` schema field.
void maybe_dump_tokens(const std::string& dump_path, const char* suffix,
                       const std::vector<int32_t>& tokens) {
    if (dump_path.empty())
        return;
    std::ofstream dump(dump_path + suffix);
    for (int32_t token : tokens) {
        dump << token << "\n";
    }
}

// Seed the sampler RNG from the request, falling back to audio_bark.seed.
// A value of -1 (default) means "leave the RNG at its constructed state."
void maybe_seed_bark_rng(std::mt19937& rng, std::int64_t seed) {
    if (seed < 0)
        return;
    const auto converted = static_cast<std::mt19937::result_type>(seed);
    rng.seed(converted);
    std::cerr << "[trtmc] Bark: sampler seed=" << converted << std::endl;
}

std::vector<float> synthesize_simple_waveform(const std::vector<int32_t>& codes_flat,
                                              int32_t n_frames, const BarkConfig& cfg) {
    const int32_t samples_per_frame = cfg.sample_rate / cfg.coarse_rate_hz;
    const int32_t total_samples = n_frames * samples_per_frame;
    std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);
    for (int32_t f = 0; f < n_frames; ++f) {
        const float freq = 200.0F + static_cast<float>(codes_flat[f]) * 800.0F /
                                        static_cast<float>(cfg.codebook_size);
        const float amp = 0.3F;
        for (int32_t s = 0; s < samples_per_frame; ++s) {
            const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
            const float t = static_cast<float>(s) / static_cast<float>(cfg.sample_rate);
            waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
        }
    }
    return waveform;
}

void update_fine_codes_from_logits(std::vector<int32_t>& codes,
                                   const std::vector<float>& host_logits, int32_t cb_idx,
                                   int32_t n_frames, int32_t actual_frames, int32_t fine_cb_size,
                                   int32_t codebook_size) {
    const int32_t valid_range = std::min(codebook_size, fine_cb_size);
    for (int32_t frame = 0; frame < actual_frames; ++frame) {
        const float* frame_logits =
            host_logits.data() + static_cast<std::size_t>(frame) * fine_cb_size;
        int32_t best = 0;
        for (int32_t i = 1; i < valid_range; ++i) {
            if (frame_logits[i] > frame_logits[best]) {
                best = i;
            }
        }
        codes[static_cast<std::size_t>(cb_idx) * n_frames + frame] = best;
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// BarkPipeline
// ═══════════════════════════════════════════════════════════════════════════

BarkPipeline::BarkPipeline(std::unique_ptr<TrtModule> semantic, std::unique_ptr<TrtModule> coarse,
                           std::unique_ptr<BarkInferenceState> semantic_state,
                           std::unique_ptr<BarkInferenceState> coarse_state,
                           std::vector<float> semantic_embed, std::vector<float> coarse_embed,
                           BarkConfig config, cudaStream_t stream,
                           std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str)
    : semantic_(std::move(semantic)), coarse_(std::move(coarse)),
      semantic_state_(std::move(semantic_state)), coarse_state_(std::move(coarse_state)),
      semantic_embed_(std::move(semantic_embed)), coarse_embed_(std::move(coarse_embed)),
      config_(std::move(config)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)) {
    if (!semantic_ || !semantic_->ok())
        throw std::runtime_error("BarkPipeline: invalid semantic module");
    if (!coarse_ || !coarse_->ok())
        throw std::runtime_error("BarkPipeline: invalid coarse module");
    if (semantic_embed_.empty() || coarse_embed_.empty())
        throw std::runtime_error("BarkPipeline: empty embedding tables");
}

BarkPipeline::~BarkPipeline() = default;

void BarkPipeline::set_codec_module(std::unique_ptr<TrtModule> codec) {
    codec_ = std::move(codec);
}

void BarkPipeline::set_fine_module(std::unique_ptr<TrtModule> fine) {
    fine_ = std::move(fine);
}

void BarkPipeline::set_fine_embeddings(std::vector<float> embed, std::vector<float> pos_embed) {
    fine_embed_ = std::move(embed);
    fine_position_embed_ = std::move(pos_embed);
}

AudioResult BarkPipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    // Tokenize the prompt
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    // A public request seed takes precedence over the session-level
    // audio_bark.seed fallback populated by bark_plugin.cpp.
    maybe_seed_bark_rng(rng_, resolve_bark_seed(config_.seed, cfg.seed));

    std::cerr << "[trtmc] Bark: starting pipeline with " << input_ids.size()
              << " text tokens, max_semantic=" << max_tokens << (config_.greedy ? " (greedy)" : "")
              << std::endl;

    // Stage 1: Text -> Semantic tokens
    auto semantic_tokens = run_semantic(input_ids, max_tokens);
    if (semantic_tokens.empty()) {
        std::cerr << "[trtmc] Bark: semantic stage produced no tokens" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    // Stage 2: Semantic -> Coarse acoustic codes
    auto coarse_tokens = run_coarse(semantic_tokens);
    if (coarse_tokens.empty()) {
        std::cerr << "[trtmc] Bark: coarse stage produced no tokens" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    // Stage 2.5: Fine (coarse codes -> 8 codebook codes)
    auto fine_codes = run_fine(coarse_tokens);
    const BarkCodecPlan codec_plan = make_bark_codec_plan(
        fine_codes, static_cast<bool>(fine_), coarse_tokens, config_.n_coarse_codebooks);

    std::vector<float> waveform = codec_plan.use_fine_codes
                                      ? run_codec(fine_codes, codec_plan.frame_count)
                                      : run_codec(coarse_tokens);
    if (waveform.empty()) {
        std::cerr << "[trtmc] Bark: codec produced no audio" << std::endl;
        AudioResult out;
        out.sample_rate = config_.sample_rate;
        return out;
    }

    AudioResult out;
    out.samples = std::move(waveform);
    out.num_samples = static_cast<int32_t>(out.samples.size());
    out.sample_rate = config_.sample_rate;
    std::cerr << "[trtmc] Bark: generated " << out.num_samples << " samples ("
              << static_cast<float>(out.num_samples) / out.sample_rate << "s @ " << out.sample_rate
              << " Hz)" << std::endl;
    return out;
}

void BarkPipeline::run_step_with_embed(TrtModule& module, BarkInferenceState& state,
                                       const float* embed, int32_t embed_dim,
                                       std::vector<float>& logits) {
    Tensor embed_tensor;
    embed_tensor.data = const_cast<float*>(embed);
    embed_tensor.shape = {embed_dim};
    embed_tensor.dtype = DType::kFloat32;

    float use_embed = 1.0F;
    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    int32_t dummy_token = 0;
    Tensor token_tensor;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    if (module.has_input("token_id"))
        inputs["token_id"] = token_tensor;
    if (module.has_input("input_embed"))
        inputs["input_embed"] = embed_tensor;
    if (module.has_input("use_input_embed"))
        inputs["use_input_embed"] = use_embed_tensor;
    state.prepare_step(inputs);

    TensorMap outputs = module.forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("BarkPipeline: no 'logits' output");

    const auto& logits_tensor = it->second;
    logits.resize(static_cast<std::size_t>(logits_tensor.numel()));
    std::memcpy(logits.data(), logits_tensor.data, logits_tensor.numel() * sizeof(float));

    state.advance();
}

void BarkPipeline::run_step_with_token(TrtModule& module, BarkInferenceState& state,
                                       int32_t token_id, std::vector<float>& logits) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    float use_embed = 0.0F;
    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (module.has_input("use_input_embed"))
        inputs["use_input_embed"] = use_embed_tensor;
    state.prepare_step(inputs);

    TensorMap outputs = module.forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("BarkPipeline: no 'logits' output");

    const auto& logits_tensor = it->second;
    logits.resize(static_cast<std::size_t>(logits_tensor.numel()));
    std::memcpy(logits.data(), logits_tensor.data, logits_tensor.numel() * sizeof(float));

    state.advance();
}

int32_t BarkPipeline::sample_top_k(const float* logits, int32_t vocab_size, float temperature,
                                   int32_t top_k) {
    if (config_.greedy) {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > logits[best])
                best = i;
        }
        return best;
    }

    top_k = std::min(top_k, vocab_size);
    std::vector<int32_t> indices(static_cast<std::size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    if (top_k < vocab_size) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                          [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    }

    std::vector<float> probs(static_cast<std::size_t>(top_k));
    const float max_logit =
        top_k < vocab_size ? logits[indices[0]] : *std::max_element(logits, logits + vocab_size);
    float sum = 0.0F;
    for (int32_t i = 0; i < top_k; ++i) {
        probs[i] = std::exp((logits[indices[i]] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int32_t i = 0; i < top_k; ++i)
        probs[i] /= sum;

    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    float r = dist(rng_);
    float cumulative = 0.0F;
    for (int32_t i = 0; i < top_k; ++i) {
        cumulative += probs[i];
        if (r < cumulative)
            return indices[i];
    }
    return indices[top_k - 1];
}

// ---------------------------------------------------------------------------
// Stage 1: Semantic (text tokens -> semantic audio tokens)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkPipeline::run_semantic(const std::vector<int32_t>& text_ids,
                                                int32_t max_tokens) {
    const auto& cfg = config_;

    semantic_state_->reset();
    semantic_state_->bind_to(*semantic_);

    std::vector<float> logits;
    std::vector<float> embed_a(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_b(static_cast<std::size_t>(cfg.hidden_size));
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));

    // Prefill text tokens
    const auto copy_len = std::min(static_cast<int32_t>(text_ids.size()), kBarkMaxTextLen);
    for (int32_t pos = 0; pos < kBarkMaxTextLen; ++pos) {
        const int32_t text_tok = semantic_text_token(text_ids, pos, copy_len, cfg);
        copy_embed_row(semantic_embed_.data(), text_tok, cfg.hidden_size, embed_a.data());
        copy_embed_row(semantic_embed_.data(), cfg.semantic_pad_token, cfg.hidden_size,
                       embed_b.data());
        sum_embed_rows(embed_a.data(), embed_b.data(), cfg.hidden_size, embed_buf.data());
        run_step_with_embed(*semantic_, *semantic_state_, embed_buf.data(), cfg.hidden_size,
                            logits);
    }

    // Prefill infer token
    copy_embed_row(semantic_embed_.data(), cfg.semantic_infer_token, cfg.hidden_size,
                   embed_buf.data());
    run_step_with_embed(*semantic_, *semantic_state_, embed_buf.data(), cfg.hidden_size, logits);

    // Autoregressive generation
    std::vector<int32_t> semantic_tokens;
    semantic_tokens.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step) {
        if (semantic_eos_threshold_hit(logits, cfg))
            break;
        suppress_semantic_logits(logits, cfg.semantic_pad_token);

        const int32_t token = sample_top_k(logits.data(), cfg.semantic_pad_token + 1,
                                           cfg.semantic_temperature, cfg.top_k);
        if (token == cfg.semantic_pad_token)
            break;

        semantic_tokens.push_back(token);

        // During decode, use token_id path (no embedding injection)
        run_step_with_token(*semantic_, *semantic_state_, token, logits);
    }

    std::cerr << "[trtmc] Bark semantic: generated " << semantic_tokens.size() << " tokens"
              << std::endl;
    maybe_dump_tokens(config_.dump_path, ".sem_tokens", semantic_tokens);
    return semantic_tokens;
}

// ---------------------------------------------------------------------------
// Stage 2: Coarse (semantic tokens -> coarse acoustic codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkPipeline::run_coarse(const std::vector<int32_t>& semantic_tokens) {
    const auto& cfg = config_;
    const BarkCoarsePlan coarse_plan = make_bark_coarse_plan(semantic_tokens, cfg);
    const int32_t n_steps = coarse_plan.total_steps;

    if (n_steps == 0) {
        std::cerr << "[trtmc] Bark coarse: no steps to generate" << std::endl;
        return {};
    }

    std::vector<int32_t> x_coarse;
    x_coarse.reserve(static_cast<std::size_t>(n_steps));

    std::vector<float> logits;
    std::vector<float> embed_buf(static_cast<std::size_t>(cfg.hidden_size));

    for (int32_t win = 0; win < coarse_plan.num_windows; ++win) {
        const BarkCoarseWindowPlan window_plan =
            make_bark_coarse_window_plan(coarse_plan, x_coarse, cfg);
        if (window_plan.generated_this_window <= 0)
            break;

        // Reset cache for each window
        coarse_state_->reset();
        coarse_state_->bind_to(*coarse_);

        // Prefill coarse window
        for (std::size_t i = 0; i + 1 < window_plan.input_tokens.size(); ++i) {
            copy_embed_row(coarse_embed_.data(), window_plan.input_tokens[i], cfg.hidden_size,
                           embed_buf.data());
            run_step_with_embed(*coarse_, *coarse_state_, embed_buf.data(), cfg.hidden_size,
                                logits);
        }

        // Last prefill token
        copy_embed_row(coarse_embed_.data(), window_plan.input_tokens.back(), cfg.hidden_size,
                       embed_buf.data());
        run_step_with_embed(*coarse_, *coarse_state_, embed_buf.data(), cfg.hidden_size, logits);

        // Generate
        for (int32_t step = 0; step < window_plan.generated_this_window; ++step) {
            const int32_t total_generated = window_plan.start_generated_count + step;
            const int32_t codebook_idx = bark_coarse_codebook_index(total_generated, cfg);
            mask_coarse_logits_for_codebook(logits, codebook_idx, cfg);

            const int32_t token = sample_top_k(logits.data(), static_cast<int32_t>(logits.size()),
                                               cfg.coarse_temperature, cfg.top_k);
            x_coarse.push_back(token);

            if (step + 1 < window_plan.generated_this_window) {
                copy_embed_row(coarse_embed_.data(), token, cfg.hidden_size, embed_buf.data());
                run_step_with_embed(*coarse_, *coarse_state_, embed_buf.data(), cfg.hidden_size,
                                    logits);
            }
        }
    }

    std::cerr << "[trtmc] Bark coarse: generated " << x_coarse.size() << " tokens" << std::endl;
    maybe_dump_tokens(config_.dump_path, ".coarse_tokens", x_coarse);
    return x_coarse;
}

// ---------------------------------------------------------------------------
// Stage 2.5: Fine (coarse codes -> 8 codebook codes)
// ---------------------------------------------------------------------------

std::vector<int32_t> BarkPipeline::run_fine(const std::vector<int32_t>& coarse_tokens) {
    const auto& cfg = config_;
    const BarkFinePlan plan = make_bark_fine_plan(
        cfg, coarse_tokens.size(), static_cast<bool>(fine_), static_cast<bool>(fine_));
    std::vector<int32_t> codes = initialize_bark_fine_codes(coarse_tokens, plan.n_frames, cfg);

    if (!plan.should_run_trt) {
        std::cerr << "[trtmc] Bark fine: no TRT fine engine, "
                  << "codebooks 2-7 will be zero" << std::endl;
        return codes;
    }

    const int32_t fine_hidden = cfg.fine_hidden_size;
    const int32_t fine_cb_size = cfg.fine_codebook_size;
    const int32_t max_seq = cfg.fine_seq_length;

    std::vector<float> host_embeds(static_cast<std::size_t>(max_seq) * fine_hidden, 0.0F);
    std::vector<float> host_logits(static_cast<std::size_t>(max_seq) * fine_cb_size);

    for (int32_t cb_idx = plan.first_predicted_codebook; cb_idx < plan.last_predicted_codebook;
         ++cb_idx) {
        // Build input embeddings on host
        build_bark_fine_input_embeddings(host_embeds, codes, cb_idx, plan.n_frames,
                                         plan.actual_frames, max_seq, fine_hidden, fine_cb_size,
                                         cfg.codebook_size, fine_embed_, fine_position_embed_);

        // Build TensorMap
        Tensor embed_tensor;
        embed_tensor.data = host_embeds.data();
        embed_tensor.shape = {max_seq, fine_hidden};
        embed_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["input_embeds"] = embed_tensor;

        TensorMap outputs = fine_->forward(inputs);

        // Read the correct codebook head output
        const int32_t head_idx = cb_idx - 1;
        const std::string head_name = "logits_cb" + std::to_string(head_idx + 1);
        auto it = outputs.find(head_name);
        if (it == outputs.end()) {
            std::cerr << "[trtmc] Bark fine: missing output " << head_name << std::endl;
            return codes;
        }

        const auto& logits_tensor = it->second;
        const std::size_t logits_bytes =
            std::min(static_cast<std::size_t>(max_seq) * fine_cb_size * sizeof(float),
                     logits_tensor.nbytes());
        std::memcpy(host_logits.data(), logits_tensor.data, logits_bytes);

        if (bark_fine_uses_sampling(cfg)) {
            const int32_t valid_range = std::min(cfg.codebook_size, fine_cb_size);
            for (int32_t frame = 0; frame < plan.actual_frames; ++frame) {
                const float* frame_logits =
                    host_logits.data() + static_cast<std::size_t>(frame) * fine_cb_size;
                codes[static_cast<std::size_t>(cb_idx) * plan.n_frames + frame] =
                    sample_top_k(frame_logits, valid_range, cfg.fine_temperature, valid_range);
            }
        } else {
            update_fine_codes_from_logits(codes, host_logits, cb_idx, plan.n_frames,
                                          plan.actual_frames, fine_cb_size, cfg.codebook_size);
        }
    }

    std::cerr << "[trtmc] Bark fine: predicted codebooks 2-7 for " << plan.n_frames << " frames"
              << std::endl;
    return codes;
}

// ---------------------------------------------------------------------------
// Stage 3: Codec (codes -> waveform)
// ---------------------------------------------------------------------------

std::vector<float> BarkPipeline::run_codec(const std::vector<int32_t>& coarse_tokens) {
    const auto& cfg = config_;
    const int32_t n_frames = static_cast<int32_t>(coarse_tokens.size()) / cfg.n_coarse_codebooks;

    if (n_frames == 0)
        return {};

    // De-interleave coarse tokens to [n_codebooks, n_frames]
    std::vector<int32_t> codes(static_cast<std::size_t>(cfg.n_coarse_codebooks) * n_frames, 0);
    for (int32_t t = 0; t < n_frames * cfg.n_coarse_codebooks; ++t) {
        const int32_t cb = t % cfg.n_coarse_codebooks;
        const int32_t frame = t / cfg.n_coarse_codebooks;
        int32_t raw_code = coarse_tokens[t] - cfg.semantic_vocab_size - cb * cfg.codebook_size;
        raw_code = std::max(0, std::min(raw_code, cfg.codebook_size - 1));
        codes[cb * n_frames + frame] = raw_code;
    }

    if (!codec_ || !codec_->ok() || cfg.codec_seq_length <= 0) {
        std::cerr << "[trtmc] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes, n_frames, cfg);
    }
    return run_codec(codes, n_frames);
}

std::vector<float> BarkPipeline::run_codec(const std::vector<int32_t>& codes_flat,
                                           int32_t n_frames) {
    const auto& cfg = config_;

    if (n_frames <= 0)
        return {};

    if (!codec_ || !codec_->ok() || cfg.codec_seq_length <= 0) {
        std::cerr << "[trtmc] Bark codec: no TRT codec engine, "
                  << "generating simple waveform from codes" << std::endl;
        return synthesize_simple_waveform(codes_flat, n_frames, cfg);
    }

    const int32_t n_cb = cfg.codec_n_codebooks;
    const int32_t max_T = cfg.codec_seq_length;
    const int32_t upsample = cfg.codec_upsample_factor;

    if (n_frames > max_T) {
        std::cerr << "[trtmc] Bark codec: n_frames=" << n_frames
                  << " exceeds codec_seq_length=" << max_T << ", truncating" << std::endl;
    }
    const int32_t actual_frames = std::min(n_frames, max_T);

    // Determine source codebooks from codes_flat layout
    const int32_t source_codebooks =
        static_cast<int32_t>(codes_flat.size()) / std::max(n_frames, 1);
    std::vector<int32_t> input_codes = make_bark_codec_input_codes(
        codes_flat, source_codebooks, n_frames, n_cb, max_T, actual_frames);

    // Build TensorMap for codec
    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {n_cb, max_T};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["audio_codes"] = codes_tensor;

    TensorMap outputs = codec_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end()) {
        std::cerr << "[trtmc] Bark codec: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wav_tensor = it->second;
    const auto total_elems = static_cast<std::size_t>(max_T) * upsample;
    const auto trimmed = static_cast<std::size_t>(actual_frames) * upsample;
    const auto* wav_data = static_cast<const float*>(wav_tensor.data);

    std::vector<float> waveform(wav_data, wav_data + std::min(trimmed, total_elems));

    std::cerr << "[trtmc] Bark codec: TRT decode " << actual_frames << " frames -> "
              << waveform.size() << " samples" << std::endl;
    return waveform;
}

} // namespace trtmc

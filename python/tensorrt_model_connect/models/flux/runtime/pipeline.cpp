/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FluxPipeline implementation: TrtModule-based FLUX diffusion pipeline.
// Ports flux_diffusion_backend.cpp from raw TRT API to TrtModule::forward().
//
// All GPU buffer management (CudaBuffer, CudaStream, setTensorAddress,
// enqueueV3, cudaMemcpy) is removed. TrtModule::forward() handles H2D/D2H
// internally. CPU math (timestep embedding, RoPE, packing/unpacking,
// sinusoidal embedding, matmul, BN denorm) is preserved identically.

#include "pipeline.h"

#include "flux_batch_utils.h"
#include "flux_clip_helpers.h"
#include "flux_denoising_step_seam.h"
#include "flux_generation_plan.h"
#include "flux_rope_helpers.h"
#include "flux_text_helpers.h"
#include "gpu_matmul.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>

namespace trtmc {

namespace {

using diffusion::FluxPackLayout;
using diffusion::flux_scheduler::FlowMatchEulerState;

constexpr int32_t kFluxClipSeqLen = 77;
constexpr int32_t kFluxClipDim = 768;

// ---------------------------------------------------------------------------
// CPU math helpers (standalone, not methods on a base class)
// ---------------------------------------------------------------------------

void cpu_matmul_bias(const float* A, const float* B, const float* bias, float* out, int32_t M,
                     int32_t K, int32_t N) {
    // Offload to cuBLAS when the matmul is large enough to justify H2D/D2H.
    // Context embedder (512×15360×6144) and temb MLPs (1×6144×6144) hit this.
    if (int64_t(M) * K * N > 100000) {
        flux_gpu_matmul_bias(A, B, bias, out, M, K, N);
        return;
    }
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }
            if (bias != nullptr) {
                acc += static_cast<double>(bias[j]);
            }
            out[i * N + j] = static_cast<float>(acc);
        }
    }
}

void cpu_silu_inplace(float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = x / (1.0F + std::exp(-x));
    }
}

// ---------------------------------------------------------------------------
// CLIP helpers
// ---------------------------------------------------------------------------

int32_t find_first_token(const std::vector<int32_t>& ids, int32_t token_id) {
    for (int32_t i = 0; i < kFluxClipSeqLen; ++i) {
        if (ids[static_cast<std::size_t>(i)] == token_id) {
            return i;
        }
    }
    return -1;
}

int32_t find_first_max_token(const std::vector<int32_t>& ids) {
    int32_t max_token_id = ids[0];
    int32_t max_index = 0;
    for (int32_t i = 1; i < kFluxClipSeqLen; ++i) {
        if (ids[static_cast<std::size_t>(i)] > max_token_id) {
            max_token_id = ids[static_cast<std::size_t>(i)];
            max_index = i;
        }
    }
    return max_index;
}

int32_t select_clip_pool_index(const std::vector<int32_t>& padded_ids, int32_t eos_token_id) {
    int32_t pool_idx = -1;
    if (eos_token_id >= 0) {
        pool_idx = find_first_token(padded_ids, eos_token_id);
    }
    if (pool_idx < 0) {
        pool_idx = find_first_max_token(padded_ids);
    }
    pool_idx = std::max(pool_idx, 0);
    return std::min(pool_idx, kFluxClipSeqLen - 1);
}

void copy_clip_pooled_row(const std::vector<float>& clip_hidden, int32_t pool_idx,
                          std::vector<float>& pooled_output) {
    pooled_output.resize(static_cast<std::size_t>(kFluxClipDim));
    const float* pooled_src = clip_hidden.data() + static_cast<std::size_t>(pool_idx) *
                                                       static_cast<std::size_t>(kFluxClipDim);
    std::copy_n(pooled_src, static_cast<std::size_t>(kFluxClipDim), pooled_output.begin());
}

std::vector<int32_t> build_flux_clip_ids(const std::vector<int32_t>& input_ids,
                                         ITokenizer* clip_tokenizer,
                                         const std::string& raw_prompt) {
    if (clip_tokenizer != nullptr && !raw_prompt.empty()) {
        auto clip_ids = clip_tokenizer->encode(raw_prompt);
        std::cerr << "[flux] CLIP tokenized prompt (" << clip_ids.size()
                  << " tokens) from raw text\n";
        return clip_ids;
    }
    std::cerr << "[flux] Warning: no CLIP tokenizer, using T5 tokens for CLIP encoder\n";
    return input_ids;
}

template <typename RunClipFn>
bool prepare_flux_clip_conditioning(const std::vector<int32_t>& input_ids,
                                    int32_t num_text_encoders, ITokenizer* clip_tokenizer,
                                    const std::string& raw_prompt, RunClipFn&& run_clip,
                                    std::vector<float>& pooled_output) {
    if (num_text_encoders < 2) {
        pooled_output.assign(static_cast<std::size_t>(kFluxClipDim), 0.0F);
        std::cerr << "[flux] No CLIP encoder, using zero pooled output\n";
        return true;
    }

    const auto clip_ids = build_flux_clip_ids(input_ids, clip_tokenizer, raw_prompt);
    if (!run_clip(clip_ids, pooled_output)) {
        return false;
    }
    std::cerr << "[flux] CLIP encoder done\n";
    return true;
}

template <typename RunT5Fn>
bool prepare_flux_t5_conditioning(const std::vector<int32_t>& input_ids, int32_t num_text_encoders,
                                  RunT5Fn&& run_t5, std::vector<float>& text_embeddings) {
    const int32_t t5_idx = (num_text_encoders > 1) ? 1 : 0;
    if (!run_t5(t5_idx, input_ids, text_embeddings)) {
        return false;
    }
    std::cerr << "[flux] T5 encoder done\n";
    return true;
}

// ---------------------------------------------------------------------------
// Latent initialization
// ---------------------------------------------------------------------------

void initialize_flux_latents(std::vector<float>& latents, std::uint32_t seed = 42U) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents) {
        v = dist(gen);
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal embedding
// ---------------------------------------------------------------------------

void fill_flux_sinusoidal_embedding(float value, int32_t freq_dim, std::vector<float>& embedding) {
    embedding.resize(static_cast<std::size_t>(freq_dim));
    const int32_t half = freq_dim / 2;
    for (int32_t i = 0; i < half; ++i) {
        const float freq =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
        embedding[static_cast<std::size_t>(i)] = std::cos(value * freq);
        embedding[static_cast<std::size_t>(i + half)] = std::sin(value * freq);
    }
}

// ---------------------------------------------------------------------------
// Embedding combination
// ---------------------------------------------------------------------------

void combine_flux_embeddings(const std::vector<float>& timestep_proj,
                             const std::vector<float>& text_proj,
                             const std::vector<float>& guidance_proj, std::vector<float>& temb) {
    temb.resize(timestep_proj.size());
    for (std::size_t i = 0; i < timestep_proj.size(); ++i) {
        temb[i] = timestep_proj[i] + text_proj[i] + guidance_proj[i];
    }
}

void log_flux_temb_stats(float timestep, float guidance, const std::vector<float>& temb) {
    float tmin = temb[0];
    float tmax = temb[0];
    double tsum = 0.0;
    for (const auto v : temb) {
        tmin = std::min(tmin, v);
        tmax = std::max(tmax, v);
        tsum += static_cast<double>(v);
    }
    std::cerr << "[flux-temb] t=" << timestep << " g=" << guidance << " temb=[" << tmin << ","
              << tmax << ",mean=" << (tsum / static_cast<double>(temb.size())) << "]\n";
}

// ---------------------------------------------------------------------------
// FLUX.2 CHW <-> HWC packing
// ---------------------------------------------------------------------------

void pack_flux2_latents(const std::vector<float>& latents, int32_t packed_channels,
                        int32_t h_packed, int32_t w_packed, std::vector<float>& packed) {
    // FLUX.2: latents are [packed_channels, h_packed, w_packed] in CHW
    // Pack = CHW -> HWC: tokens[h*W+w, c] = latents[c, h, w]
    const auto num_tokens = static_cast<std::size_t>(h_packed) * static_cast<std::size_t>(w_packed);
    packed.resize(num_tokens * static_cast<std::size_t>(packed_channels));
    for (int32_t h = 0; h < h_packed; ++h) {
        for (int32_t w = 0; w < w_packed; ++w) {
            const int32_t tok = h * w_packed + w;
            for (int32_t c = 0; c < packed_channels; ++c) {
                const auto src =
                    static_cast<std::size_t>(c) * static_cast<std::size_t>(h_packed * w_packed) +
                    static_cast<std::size_t>(h * w_packed + w);
                const auto dst =
                    static_cast<std::size_t>(tok) * static_cast<std::size_t>(packed_channels) +
                    static_cast<std::size_t>(c);
                packed[dst] = latents[src];
            }
        }
    }
}

void unpack_flux2_velocity(const std::vector<float>& denoiser_output, int32_t packed_channels,
                           int32_t h_packed, int32_t w_packed, std::vector<float>& velocity) {
    // FLUX.2: HWC -> CHW: velocity[c, h, w] = tokens[h*W+w, c]
    const auto total = static_cast<std::size_t>(packed_channels) *
                       static_cast<std::size_t>(h_packed) * static_cast<std::size_t>(w_packed);
    velocity.resize(total);
    for (int32_t h = 0; h < h_packed; ++h) {
        for (int32_t w = 0; w < w_packed; ++w) {
            const int32_t tok = h * w_packed + w;
            for (int32_t c = 0; c < packed_channels; ++c) {
                const auto src_i =
                    static_cast<std::size_t>(tok) * static_cast<std::size_t>(packed_channels) +
                    static_cast<std::size_t>(c);
                const auto dst_i =
                    static_cast<std::size_t>(c) * static_cast<std::size_t>(h_packed * w_packed) +
                    static_cast<std::size_t>(h * w_packed + w);
                velocity[dst_i] = denoiser_output[src_i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FLUX.1 2x2 spatial packing
// ---------------------------------------------------------------------------

void pack_flux_latents(const std::vector<float>& latents, int32_t z_dim, int32_t h_lat,
                       int32_t w_lat, const FluxPackLayout& layout, std::vector<float>& packed) {
    const auto num_img_tokens =
        static_cast<std::size_t>(layout.h_packed) * static_cast<std::size_t>(layout.w_packed);
    packed.resize(num_img_tokens * static_cast<std::size_t>(layout.packed_channels));
    for (int32_t py = 0; py < layout.h_packed; ++py) {
        for (int32_t px = 0; px < layout.w_packed; ++px) {
            const int32_t tok_idx = py * layout.w_packed + px;
            float* dst = packed.data() + static_cast<std::size_t>(tok_idx) *
                                             static_cast<std::size_t>(layout.packed_channels);
            int32_t off = 0;
            for (int32_t c = 0; c < z_dim; ++c) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t y = py * layout.ph + dy;
                        const int32_t x = px * layout.pw + dx;
                        const auto src_idx =
                            static_cast<std::size_t>(c) * static_cast<std::size_t>(h_lat * w_lat) +
                            static_cast<std::size_t>(y * w_lat + x);
                        dst[off++] = latents[src_idx];
                    }
                }
            }
        }
    }
}

void unpack_flux_velocity(const std::vector<float>& denoiser_output, int32_t z_dim, int32_t h_lat,
                          int32_t w_lat, const FluxPackLayout& layout,
                          std::vector<float>& velocity) {
    velocity.resize(static_cast<std::size_t>(z_dim) * static_cast<std::size_t>(h_lat) *
                    static_cast<std::size_t>(w_lat));
    for (int32_t py = 0; py < layout.h_packed; ++py) {
        for (int32_t px = 0; px < layout.w_packed; ++px) {
            const int32_t tok_idx = py * layout.w_packed + px;
            const float* src =
                denoiser_output.data() + static_cast<std::size_t>(tok_idx) *
                                             static_cast<std::size_t>(layout.packed_channels);
            int32_t off = 0;
            for (int32_t c = 0; c < z_dim; ++c) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t y = py * layout.ph + dy;
                        const int32_t x = px * layout.pw + dx;
                        const auto dst_idx =
                            static_cast<std::size_t>(c) * static_cast<std::size_t>(h_lat * w_lat) +
                            static_cast<std::size_t>(y * w_lat + x);
                        velocity[dst_idx] = src[off++];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step logging
// ---------------------------------------------------------------------------

[[maybe_unused]] void compute_vector_stats(const std::vector<float>& values, float& min_out,
                                           float& max_out, double& mean_out) {
    min_out = values[0];
    max_out = values[0];
    double sum = 0.0;
    for (const auto v : values) {
        min_out = std::min(min_out, v);
        max_out = std::max(max_out, v);
        sum += static_cast<double>(v);
    }
    mean_out = sum / static_cast<double>(values.size());
}

void log_flux_step_stats(int32_t step, int32_t num_inference_steps,
                         const FlowMatchEulerState& scheduler,
                         const std::vector<float>& /*latents*/,
                         const std::vector<float>& /*velocity*/,
                         const std::vector<float>& /*hidden*/) {
    // Lightweight progress logging — skip expensive min/max/mean over 25M-element
    // vectors (was costing ~260ms/step = 7.2s for 28 steps).
    const auto si = static_cast<std::size_t>(step);
    std::cerr << "[flux] Step " << (step + 1) << "/" << num_inference_steps
              << " t=" << scheduler.timesteps[si] << "\n";
}

// ---------------------------------------------------------------------------
// BN denormalization (FLUX.2)
// ---------------------------------------------------------------------------

void apply_bn_denorm_inplace(std::vector<float>& data, int32_t num_channels, int32_t spatial_size,
                             const std::vector<float>& bn_mean, const std::vector<float>& bn_var,
                             float eps) {
    const int32_t bn_ch = static_cast<int32_t>(bn_mean.size());
    const auto spatial = static_cast<std::size_t>(spatial_size);
    for (int32_t c = 0; c < bn_ch && c < num_channels; ++c) {
        const float s = std::sqrt(bn_var[static_cast<std::size_t>(c)] + eps);
        const float m = bn_mean[static_cast<std::size_t>(c)];
        for (std::size_t i = 0; i < spatial; ++i) {
            const auto idx = static_cast<std::size_t>(c) * spatial + i;
            data[idx] = data[idx] * s + m;
        }
    }
}

void unpatchify_latents(const std::vector<float>& packed, const FluxPackLayout& layout,
                        int32_t z_dim, int32_t h_lat, int32_t w_lat, std::vector<float>& out) {
    const auto spatial = static_cast<std::size_t>(layout.h_packed * layout.w_packed);
    out.resize(static_cast<std::size_t>(z_dim) * static_cast<std::size_t>(h_lat) *
               static_cast<std::size_t>(w_lat));
    for (int32_t c = 0; c < z_dim; ++c) {
        for (int32_t py = 0; py < layout.h_packed; ++py) {
            for (int32_t px = 0; px < layout.w_packed; ++px) {
                for (int32_t dy = 0; dy < layout.ph; ++dy) {
                    for (int32_t dx = 0; dx < layout.pw; ++dx) {
                        const int32_t src_ch = c * layout.ph * layout.pw + dy * layout.pw + dx;
                        const auto si = static_cast<std::size_t>(src_ch) * spatial +
                                        static_cast<std::size_t>(py * layout.w_packed + px);
                        const auto di =
                            static_cast<std::size_t>(c) * static_cast<std::size_t>(h_lat * w_lat) +
                            static_cast<std::size_t>((py * layout.ph + dy) * w_lat +
                                                     px * layout.pw + dx);
                        out[di] = packed[si];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pack/unpack function factories
// ---------------------------------------------------------------------------

std::function<void(const std::vector<float>&, std::vector<float>&)>
make_flux_pack_fn(bool is_flux2, int32_t z_dim, int32_t h_lat, int32_t w_lat,
                  const FluxPackLayout& layout) {
    if (is_flux2) {
        return [&layout](const std::vector<float>& lat, std::vector<float>& packed) {
            pack_flux2_latents(lat, layout.packed_channels, layout.h_packed, layout.w_packed,
                               packed);
        };
    }
    return
        [z_dim, h_lat, w_lat, &layout](const std::vector<float>& lat, std::vector<float>& packed) {
            pack_flux_latents(lat, z_dim, h_lat, w_lat, layout, packed);
        };
}

std::function<void(const std::vector<float>&, std::vector<float>&)>
make_flux_unpack_fn(bool is_flux2, int32_t z_dim, int32_t h_lat, int32_t w_lat,
                    const FluxPackLayout& layout) {
    if (is_flux2) {
        return [&layout](const std::vector<float>& out, std::vector<float>& vel) {
            unpack_flux2_velocity(out, layout.packed_channels, layout.h_packed, layout.w_packed,
                                  vel);
        };
    }
    return [z_dim, h_lat, w_lat, &layout](const std::vector<float>& out, std::vector<float>& vel) {
        unpack_flux_velocity(out, z_dim, h_lat, w_lat, layout, vel);
    };
}

// ---------------------------------------------------------------------------
// Context embedder projection
// ---------------------------------------------------------------------------

void project_flux_encoder_hidden(const std::vector<float>& text_embeddings,
                                 const std::vector<float>& ctx_embed_w,
                                 const std::vector<float>& ctx_embed_b, int32_t text_seq,
                                 int32_t t5_dim, int32_t dit_dim,
                                 std::vector<float>& encoder_hidden) {
    if (ctx_embed_w.empty()) {
        std::cerr << "[flux] Warning: No context_embedder weights\n";
        return;
    }

    cpu_matmul_bias(text_embeddings.data(), ctx_embed_w.data(),
                    ctx_embed_b.empty() ? nullptr : ctx_embed_b.data(), encoder_hidden.data(),
                    text_seq, t5_dim, dit_dim);
    std::cerr << "[flux] Context embedder projection done\n";
}

// ---------------------------------------------------------------------------
// Scheduler logging
// ---------------------------------------------------------------------------

void log_flux_dynamic_shift(const FlowMatchEulerState& scheduler) {
    if (!scheduler.last_used_dynamic_shifting) {
        return;
    }

    std::cerr << "[flux-scheduler] Dynamic shifting: mu=" << scheduler.last_dynamic_mu
              << ", exp_mu=" << std::exp(scheduler.last_dynamic_mu)
              << ", image_seq_len=" << scheduler.image_seq_len << "\n";
}

// ---------------------------------------------------------------------------
// Hidden state embedder factory
// ---------------------------------------------------------------------------

std::function<void(const std::vector<float>&, std::vector<float>&)>
make_flux_hidden_embedder(const std::vector<float>& x_embed_w, const std::vector<float>& x_embed_b,
                          int32_t num_img_tokens, const FluxPackLayout& layout, int32_t dit_dim) {
    if (x_embed_w.empty()) {
        std::cerr << "[flux] Warning: No x_embedder weights, hidden_states are zero\n";
        return [](const std::vector<float>& /*packed*/, std::vector<float>& hidden_out) {
            std::fill(hidden_out.begin(), hidden_out.end(), 0.0F);
        };
    }

    const auto* x_embed_w_ptr = &x_embed_w;
    const auto* x_embed_b_ptr = &x_embed_b;
    return [x_embed_w_ptr, x_embed_b_ptr, num_img_tokens, layout,
            dit_dim](const std::vector<float>& packed, std::vector<float>& hidden_out) {
        cpu_matmul_bias(packed.data(), x_embed_w_ptr->data(),
                        x_embed_b_ptr->empty() ? nullptr : x_embed_b_ptr->data(), hidden_out.data(),
                        num_img_tokens, layout.packed_channels, dit_dim);
    };
}

// ---------------------------------------------------------------------------
// VAE input preparation (FLUX.2 BN denorm + unpatchify)
// ---------------------------------------------------------------------------

void prepare_flux2_vae_input(std::vector<float>& latents, const FluxPackLayout& layout,
                             int32_t z_dim, int32_t h_lat, int32_t w_lat,
                             const std::vector<float>& bn_mean, const std::vector<float>& bn_var,
                             bool is_flux2, std::vector<float>& vae_latents) {
    if (!is_flux2 || bn_mean.empty()) {
        vae_latents = latents;
        return;
    }

    apply_bn_denorm_inplace(latents, layout.packed_channels, layout.h_packed * layout.w_packed,
                            bn_mean, bn_var, 0.0001F);
    unpatchify_latents(latents, layout, z_dim, h_lat, w_lat, vae_latents);
    std::cerr << "[flux] Applied BN denorm + unpatchify (" << bn_mean.size() << " -> " << z_dim
              << " ch)\n";
}

// ---------------------------------------------------------------------------
// Latent dump (debug)
// ---------------------------------------------------------------------------

void maybe_dump_flux_latents(const std::vector<float>& latents) {
    const std::string dump_path = "/tmp/flux_final_latents.raw";
    std::ofstream dump(dump_path, std::ios::binary);
    if (!dump.is_open()) {
        return;
    }
    dump.write(reinterpret_cast<const char*>(latents.data()), latents.size() * sizeof(float));
    dump.close();
    std::cerr << "[flux] Dumped final latents (" << latents.size() << " floats) to " << dump_path
              << "\n";
}

// ---------------------------------------------------------------------------
// VAE output -> ImageResult conversion
// ---------------------------------------------------------------------------

void convert_flux_vae_output_to_image(const float* vae_output, int32_t h_out, int32_t w_out,
                                      ImageResult& result) {
    result.num_frames = 1;
    result.height = h_out;
    result.width = w_out;
    result.channels = 3;
    result.pixels.resize(static_cast<std::size_t>(h_out * w_out * 3));
    for (int32_t h = 0; h < h_out; ++h) {
        for (int32_t w = 0; w < w_out; ++w) {
            for (int32_t c = 0; c < 3; ++c) {
                const auto src =
                    static_cast<std::size_t>(c) * static_cast<std::size_t>(h_out * w_out) +
                    static_cast<std::size_t>(h * w_out + w);
                const auto dst = static_cast<std::size_t>(h * w_out * 3 + w * 3 + c);
                float v = (vae_output[src] + 1.0F) * 0.5F;
                result.pixels[dst] = std::max(0.0F, std::min(1.0F, v));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FLUX.2 prompt preparation (Mistral chat template)
// ---------------------------------------------------------------------------

std::string prepare_flux_prompt(const std::string& prompt, bool is_flux2) {
    if (is_flux2) {
        static const char* kSystemMsg =
            "You are an AI that reasons about image descriptions. "
            "You give structured responses focusing on object relationships, object\n"
            "attribution and actions without speculation.";
        return std::string("<s>[SYSTEM_PROMPT]") + kSystemMsg + "[/SYSTEM_PROMPT][INST]" + prompt +
               "[/INST]";
    }
    return prompt;
}

// ---------------------------------------------------------------------------
// CLIP tokenizer EOS/pad detection
// ---------------------------------------------------------------------------

void detect_clip_special_tokens(ITokenizer* clip_tok, int32_t& eos_token_id,
                                int32_t& pad_token_id) {
    eos_token_id = -1;
    pad_token_id = 0;

    if (!clip_tok)
        return;

    const char* kEosCandidates[] = {
        "<|endoftext|>",
        "</s>",
        "<eos>",
    };
    for (const char* tok_name : kEosCandidates) {
        try {
            const int32_t id = clip_tok->id_for_token(tok_name);
            if (id >= 0) {
                eos_token_id = id;
                break;
            }
        } catch (const std::exception&) {
            // Best-effort lookup; ignore and fall back.
        }
    }

    if (eos_token_id >= 0) {
        // OpenAI CLIP-style tokenizers use EOS as pad token.
        pad_token_id = eos_token_id;
    } else {
        const char* kPadCandidates[] = {
            "<pad>",
            "</s>",
            "<eos>",
        };
        for (const char* tok_name : kPadCandidates) {
            try {
                const int32_t id = clip_tok->id_for_token(tok_name);
                if (id >= 0) {
                    pad_token_id = id;
                    break;
                }
            } catch (const std::exception&) {
                // Best-effort lookup; ignore and keep fallback.
            }
        }
    }

    std::cerr << "[flux] CLIP tokenizer set (eos_id=" << eos_token_id << ", pad_id=" << pad_token_id
              << ")\n";
}

// ---------------------------------------------------------------------------
// Denoising loop orchestrator (uses run_flux_denoising_steps seam template)
// ---------------------------------------------------------------------------

template <typename PackFn, typename UnpackFn, typename ComputeTembFn, typename EmbedHiddenFn,
          typename RunDenoiserFn>
bool run_flux_denoising_loop(FlowMatchEulerState& scheduler, int32_t num_inference_steps,
                             std::vector<float>& latents, std::vector<float>& hidden,
                             std::vector<float>& denoiser_output, PackFn&& pack_latents,
                             UnpackFn&& unpack_velocity, ComputeTembFn&& compute_temb,
                             EmbedHiddenFn&& embed_hidden, RunDenoiserFn&& run_denoiser) {
    std::string error;
    std::vector<float> packed;
    std::vector<float> next_latents(latents.size());
    const auto prepare_hidden = [&](const std::vector<float>& current_latents,
                                    std::vector<float>& hidden_out) {
        pack_latents(current_latents, packed);
        embed_hidden(packed, hidden_out);
    };
    const auto apply_scheduler = [&](std::vector<float>& current_latents,
                                     const std::vector<float>& velocity, int32_t step) {
        scheduler.step(velocity.data(), current_latents.data(), next_latents.data(),
                       current_latents.size(), step);
        current_latents = next_latents;
    };
    const auto log_step = [&](int32_t step, const std::vector<float>& current_latents,
                              const std::vector<float>& velocity,
                              const std::vector<float>& current_hidden) {
        log_flux_step_stats(step, num_inference_steps, scheduler, current_latents, velocity,
                            current_hidden);
    };
    if (!diffusion::run_flux_denoising_steps(
            num_inference_steps, scheduler.timesteps, latents, hidden, denoiser_output, error,
            [&](float raw_timestep, std::vector<float>& temb) {
                compute_temb(raw_timestep / 1000.0F, temb);
            },
            prepare_hidden,
            [&](const std::vector<float>& hidden_in, const std::vector<float>& temb_in,
                std::vector<float>& output, std::string& err) {
                if (!run_denoiser(hidden_in, temb_in, output)) {
                    err = "FLUX denoiser step failed";
                    std::cerr << "[flux] Denoiser step failed\n";
                    return false;
                }
                return true;
            },
            unpack_velocity, apply_scheduler, log_step)) {
        std::cerr << "[flux] Denoising loop failed: " << error << "\n";
        return false;
    }
    return true;
}

} // anonymous namespace

// ===========================================================================
// FluxPipeline constructor
// ===========================================================================

FluxPipeline::FluxPipeline(std::vector<std::unique_ptr<TrtModule>> text_encoders,
                           std::unique_ptr<TrtModule> denoiser, std::unique_ptr<TrtModule> vae,
                           FluxDiffusionConfig config, FluxPreprocessorWeights weights,
                           std::shared_ptr<ITokenizer> tokenizer,
                           std::unique_ptr<ITokenizer> clip_tokenizer, std::string model_id_str,
                           std::shared_ptr<void> distributed_owner, int32_t tensor_parallel_rank,
                           int32_t tensor_parallel_size)
    : distributed_owner_(std::move(distributed_owner)), tensor_parallel_rank_(tensor_parallel_rank),
      tensor_parallel_size_(tensor_parallel_size), text_encoders_(std::move(text_encoders)),
      denoiser_(std::move(denoiser)), vae_(std::move(vae)), config_(std::move(config)),
      weights_(std::move(weights)), tokenizer_(std::move(tokenizer)),
      clip_tokenizer_(std::move(clip_tokenizer)), model_id_(std::move(model_id_str)) {
    // Compute FLUX latent layout
    h_latent_ = config_.video_height / config_.scale_factor_spatial;
    w_latent_ = config_.video_width / config_.scale_factor_spatial;

    int32_t ph = 2, pw = 2;
    if (config_.patch_size.size() >= 3) {
        ph = config_.patch_size[1];
        pw = config_.patch_size[2];
    }
    num_img_tokens_ = (h_latent_ / ph) * (w_latent_ / pw);

    std::cerr << "[flux] FluxPipeline created: img_tokens=" << num_img_tokens_
              << ", dit_dim=" << config_.dit_dim << ", h_lat=" << h_latent_
              << ", w_lat=" << w_latent_ << ", pack=" << ph << "x" << pw
              << ", text_encoders=" << text_encoders_.size()
              << ", x_embedder=" << (weights_.patch_embed_weight.empty() ? "MISSING" : "OK")
              << ", ctx_embedder=" << (weights_.context_embed_weight.empty() ? "MISSING" : "OK")
              << "\n";
    flux_gpu_matmul_init();
}

FluxPipeline::~FluxPipeline() {
    flux_gpu_matmul_shutdown();
}

// ===========================================================================
// CLIP encoder via TrtModule
// ===========================================================================

bool FluxPipeline::run_clip_encoder(const std::vector<int32_t>& input_ids,
                                    std::vector<float>& pooled_output) {
    if (text_encoders_.empty()) {
        pooled_output.assign(static_cast<std::size_t>(kFluxClipDim), 0.0F);
        return true;
    }

    auto& clip_module = text_encoders_[0];

    // Detect CLIP special tokens for pool index selection
    int32_t clip_eos_token_id = -1;
    int32_t clip_pad_token_id = 0;
    detect_clip_special_tokens(clip_tokenizer_.get(), clip_eos_token_id, clip_pad_token_id);

    const auto padded = diffusion::flux_clip::pad_and_truncate_ids(
        input_ids, static_cast<std::size_t>(kFluxClipSeqLen), clip_pad_token_id, clip_eos_token_id);

    // Build input TensorMap
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{const_cast<int32_t*>(padded.data()), {kFluxClipSeqLen}, DType::kInt32};

    auto outputs = clip_module->forward(inputs);

    // HF CLIP pools the first EOS position. The engine's legacy pooled output
    // is fixed to the last sequence row, which differs whenever EOS padding is
    // present. Prefer the full hidden states so runtime can match HF exactly.
    if (outputs.count("text_embeddings")) {
        auto& text_emb_tensor = outputs.at("text_embeddings");
        const auto* clip_hidden = static_cast<const float*>(text_emb_tensor.data);
        const auto clip_hidden_size =
            static_cast<std::size_t>(kFluxClipSeqLen) * static_cast<std::size_t>(kFluxClipDim);
        std::vector<float> clip_hidden_vec(clip_hidden, clip_hidden + clip_hidden_size);

        const int32_t pool_idx = select_clip_pool_index(padded, clip_eos_token_id);
        copy_clip_pooled_row(clip_hidden_vec, pool_idx, pooled_output);
        return true;
    }

    auto& pooled_tensor = outputs.at("pooled_output");
    const auto* data = static_cast<const float*>(pooled_tensor.data);
    pooled_output.assign(data, data + pooled_tensor.numel());
    return true;
}

// ===========================================================================
// T5 encoder via TrtModule
// ===========================================================================

bool FluxPipeline::run_t5_encoder(int32_t encoder_idx, const std::vector<int32_t>& input_ids,
                                  std::vector<float>& text_embeddings) {
    if (encoder_idx < 0 || encoder_idx >= static_cast<int32_t>(text_encoders_.size())) {
        std::cerr << "[flux] T5 encoder index " << encoder_idx << " out of range\n";
        return false;
    }

    auto& te = text_encoders_[static_cast<std::size_t>(encoder_idx)];
    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    const bool is_flux2 = !weights_.vae_bn_mean.empty();
    const int32_t pad_token_id =
        diffusion::flux_text::resolve_pad_token_id(tokenizer_.get(), is_flux2);
    auto prepared = diffusion::flux_text::prepare_inputs(input_ids, seq_len, pad_token_id);

    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{prepared.input_ids.data(), {1, static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{prepared.attention_mask.data(), {1, static_cast<int64_t>(seq_len)}, DType::kFloat32};

    auto outputs = te->forward(inputs);

    auto& emb_tensor = outputs.at("text_embeddings");
    const auto emb_size = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim);
    const auto* emb_data = static_cast<const float*>(emb_tensor.data);
    text_embeddings.assign(emb_data, emb_data + emb_size);

    // Diffusers feeds FLUX.2's full Mistral output to the denoiser, including
    // padded query rows. Preserve those rows while retaining FLUX.1's legacy
    // zero-padding contract.
    diffusion::flux_text::clear_padding_rows(text_embeddings, {prepared.valid_tokens}, seq_len,
                                             te_dim, is_flux2);

    return true;
}

// ===========================================================================
// FLUX DiT denoiser via TrtModule
// ===========================================================================

bool FluxPipeline::run_flux_denoiser(const std::vector<float>& hidden,
                                     const std::vector<float>& encoder_hidden,
                                     const std::vector<float>& temb,
                                     const std::vector<float>& cos_vals,
                                     const std::vector<float>& sin_vals,
                                     std::vector<float>& output) {
    const int32_t dit_dim = config_.dit_dim;
    const int32_t text_seq = config_.text_seq_len;
    const int32_t head_dim = dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t total_seq = text_seq + num_img_tokens_;

    // hidden_states shape depends on whether x_embedder is baked into the engine:
    // FLUX.2: [num_img_tokens, packed_channels] (x_embedder inside engine)
    // FLUX.1: [num_img_tokens, dit_dim] (x_embedder applied externally)
    const int64_t hidden_cols =
        static_cast<int64_t>(hidden.size()) / static_cast<int64_t>(num_img_tokens_);
    TensorMap inputs;
    inputs["hidden_states"] = Tensor{const_cast<float*>(hidden.data()),
                                     {static_cast<int64_t>(num_img_tokens_), hidden_cols},
                                     DType::kFloat32};
    inputs["encoder_hidden_states"] =
        Tensor{const_cast<float*>(encoder_hidden.data()),
               {static_cast<int64_t>(text_seq), static_cast<int64_t>(dit_dim)},
               DType::kFloat32};
    inputs["temb"] =
        Tensor{const_cast<float*>(temb.data()), {static_cast<int64_t>(dit_dim)}, DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);

    auto& out_tensor = outputs.at("output");
    const auto* out_data = static_cast<const float*>(out_tensor.data);
    output.assign(out_data, out_data + out_tensor.numel());

    return true;
}

// FLUX.2 denoiser: takes raw timestep/guidance scalars + raw T5 embeddings.
// Context embedder and temb MLP are baked into the TRT engine.
bool FluxPipeline::run_flux2_denoiser(const std::vector<float>& hidden,
                                      const std::vector<float>& encoder_hidden, float timestep,
                                      float guidance, const std::vector<float>& cos_vals,
                                      const std::vector<float>& sin_vals,
                                      std::vector<float>& output) {
    const int32_t text_seq = config_.text_seq_len;
    const int32_t t5_dim = config_.text_encoder_dim;
    const int32_t head_dim = config_.dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t total_seq = text_seq + num_img_tokens_;

    const int64_t hidden_cols =
        static_cast<int64_t>(hidden.size()) / static_cast<int64_t>(num_img_tokens_);
    float ts_val = timestep;
    float g_val = guidance;

    TensorMap inputs;
    inputs["hidden_states"] = Tensor{const_cast<float*>(hidden.data()),
                                     {static_cast<int64_t>(num_img_tokens_), hidden_cols},
                                     DType::kFloat32};
    inputs["encoder_hidden_states"] =
        Tensor{const_cast<float*>(encoder_hidden.data()),
               {static_cast<int64_t>(text_seq), static_cast<int64_t>(t5_dim)},
               DType::kFloat32};
    inputs["timestep"] = Tensor{&ts_val, {1}, DType::kFloat32};
    inputs["guidance"] = Tensor{&g_val, {1}, DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);
    auto& out_tensor = outputs.at("output");
    const auto* out_data = static_cast<const float*>(out_tensor.data);
    output.assign(out_data, out_data + out_tensor.numel());
    return true;
}

// ===========================================================================
// Batched T5 encoder (batch > 1 path).
// Builds [B, seq] inputs and returns [B, seq, te_dim] embeddings contiguous.
// ===========================================================================

bool FluxPipeline::run_t5_encoder_batch(int32_t encoder_idx,
                                        const std::vector<std::vector<int32_t>>& batch_input_ids,
                                        std::vector<float>& text_embeddings_batch) {
    if (encoder_idx < 0 || encoder_idx >= static_cast<int32_t>(text_encoders_.size())) {
        std::cerr << "[flux] T5 encoder index " << encoder_idx << " out of range\n";
        return false;
    }
    const auto B = static_cast<int32_t>(batch_input_ids.size());
    if (B < 1) {
        return false;
    }

    auto& te = text_encoders_[static_cast<std::size_t>(encoder_idx)];
    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    const bool is_flux2 = !weights_.vae_bn_mean.empty();
    const int32_t pad_token_id =
        diffusion::flux_text::resolve_pad_token_id(tokenizer_.get(), is_flux2);
    std::vector<int32_t> padded_ids;
    std::vector<float> mask;
    std::vector<std::size_t> valid_tokens;
    padded_ids.reserve(static_cast<std::size_t>(B) * static_cast<std::size_t>(seq_len));
    mask.reserve(static_cast<std::size_t>(B) * static_cast<std::size_t>(seq_len));
    valid_tokens.reserve(static_cast<std::size_t>(B));
    for (int32_t b = 0; b < B; ++b) {
        const auto& ids = batch_input_ids[static_cast<std::size_t>(b)];
        auto prepared = diffusion::flux_text::prepare_inputs(ids, seq_len, pad_token_id);
        padded_ids.insert(padded_ids.end(), prepared.input_ids.begin(), prepared.input_ids.end());
        mask.insert(mask.end(), prepared.attention_mask.begin(), prepared.attention_mask.end());
        valid_tokens.push_back(prepared.valid_tokens);
    }

    TensorMap inputs;
    inputs["input_ids"] = Tensor{
        padded_ids.data(), {static_cast<int64_t>(B), static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] = Tensor{
        mask.data(), {static_cast<int64_t>(B), static_cast<int64_t>(seq_len)}, DType::kFloat32};

    auto outputs = te->forward(inputs);

    auto& emb_tensor = outputs.at("text_embeddings");
    const auto emb_size = static_cast<std::size_t>(B) * static_cast<std::size_t>(seq_len) *
                          static_cast<std::size_t>(te_dim);
    const auto* emb_data = static_cast<const float*>(emb_tensor.data);
    text_embeddings_batch.assign(emb_data, emb_data + emb_size);

    diffusion::flux_text::clear_padding_rows(text_embeddings_batch, valid_tokens, seq_len, te_dim,
                                             is_flux2);
    return true;
}

// ===========================================================================
// Batched FLUX.1 DiT denoiser. All shape tuples gain a leading B dim. RoPE
// tables must also be broadcast to ``[B, total_seq, head_dim]`` (engine input
// is declared with a dynamic leading dim — see flux_dit_builder).
// ===========================================================================

bool FluxPipeline::run_flux_denoiser_batch(int32_t batch, const std::vector<float>& hidden,
                                           const std::vector<float>& encoder_hidden,
                                           const std::vector<float>& temb,
                                           const std::vector<float>& cos_vals,
                                           const std::vector<float>& sin_vals,
                                           std::vector<float>& output) {
    const int32_t dit_dim = config_.dit_dim;
    const int32_t text_seq = config_.text_seq_len;
    const int32_t head_dim = dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t total_seq = text_seq + num_img_tokens_;

    const int64_t hidden_cols =
        static_cast<int64_t>(hidden.size()) /
        (static_cast<int64_t>(batch) * static_cast<int64_t>(num_img_tokens_));
    TensorMap inputs;
    inputs["hidden_states"] =
        Tensor{const_cast<float*>(hidden.data()),
               {static_cast<int64_t>(batch), static_cast<int64_t>(num_img_tokens_), hidden_cols},
               DType::kFloat32};
    inputs["encoder_hidden_states"] =
        Tensor{const_cast<float*>(encoder_hidden.data()),
               {static_cast<int64_t>(batch), static_cast<int64_t>(text_seq),
                static_cast<int64_t>(dit_dim)},
               DType::kFloat32};
    inputs["temb"] = Tensor{const_cast<float*>(temb.data()),
                            {static_cast<int64_t>(batch), static_cast<int64_t>(dit_dim)},
                            DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(batch), static_cast<int64_t>(total_seq),
                                   static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(batch), static_cast<int64_t>(total_seq),
                                   static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);
    auto& out_tensor = outputs.at("output");
    const auto* out_data = static_cast<const float*>(out_tensor.data);
    output.assign(out_data, out_data + out_tensor.numel());
    return true;
}

// ===========================================================================
// Batched FLUX.2 denoiser. Timestep + guidance scalars are shared across the
// batch (every sample shares a denoising schedule, per design Decision B).
// ===========================================================================

bool FluxPipeline::run_flux2_denoiser_batch(int32_t batch, const std::vector<float>& hidden,
                                            const std::vector<float>& encoder_hidden,
                                            float timestep, float guidance,
                                            const std::vector<float>& cos_vals,
                                            const std::vector<float>& sin_vals,
                                            std::vector<float>& output) {
    const int32_t text_seq = config_.text_seq_len;
    const int32_t t5_dim = config_.text_encoder_dim;
    const int32_t head_dim = config_.dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t total_seq = text_seq + num_img_tokens_;

    const int64_t hidden_cols =
        static_cast<int64_t>(hidden.size()) /
        (static_cast<int64_t>(batch) * static_cast<int64_t>(num_img_tokens_));
    float ts_val = timestep;
    float g_val = guidance;

    TensorMap inputs;
    inputs["hidden_states"] =
        Tensor{const_cast<float*>(hidden.data()),
               {static_cast<int64_t>(batch), static_cast<int64_t>(num_img_tokens_), hidden_cols},
               DType::kFloat32};
    inputs["encoder_hidden_states"] = Tensor{
        const_cast<float*>(encoder_hidden.data()),
        {static_cast<int64_t>(batch), static_cast<int64_t>(text_seq), static_cast<int64_t>(t5_dim)},
        DType::kFloat32};
    inputs["timestep"] = Tensor{&ts_val, {1}, DType::kFloat32};
    inputs["guidance"] = Tensor{&g_val, {1}, DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(batch), static_cast<int64_t>(total_seq),
                                   static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(batch), static_cast<int64_t>(total_seq),
                                   static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);
    auto& out_tensor = outputs.at("output");
    const auto* out_data = static_cast<const float*>(out_tensor.data);
    output.assign(out_data, out_data + out_tensor.numel());
    return true;
}

// ===========================================================================
// Timestep embedding (CPU math, FLUX.1 only — FLUX.2 bakes this into TRT)
// ===========================================================================

void FluxPipeline::compute_flux_timestep_embedding(float timestep, float guidance,
                                                   const std::vector<float>& pooled_text,
                                                   std::vector<float>& temb) const {
    const int32_t dim = config_.dit_dim;
    const int32_t freq_dim = config_.freq_dim;

    std::vector<float> t_emb;
    fill_flux_sinusoidal_embedding(timestep * 1000.0F, freq_dim, t_emb);

    // Helper: return nullptr for empty bias vectors, valid pointer otherwise
    auto bias_or_null = [](const std::vector<float>& v) -> const float* {
        return v.empty() ? nullptr : v.data();
    };

    // timestep_embedder MLP: sinusoidal -> Linear -> SiLU -> Linear
    std::vector<float> t_proj(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_emb.data(), weights_.time_emb_0_weight.data(),
                    bias_or_null(weights_.time_emb_0_bias), t_proj.data(), 1, freq_dim, dim);
    cpu_silu_inplace(t_proj.data(), static_cast<std::size_t>(dim));

    std::vector<float> t_proj2(static_cast<std::size_t>(dim));
    cpu_matmul_bias(t_proj.data(), weights_.time_emb_2_weight.data(),
                    bias_or_null(weights_.time_emb_2_bias), t_proj2.data(), 1, dim, dim);

    // text_embedder MLP: pooled -> Linear -> SiLU -> Linear
    std::vector<float> text_proj(static_cast<std::size_t>(dim));
    if (!weights_.text_proj_weight.empty() && !pooled_text.empty()) {
        const int32_t text_in_dim = static_cast<int32_t>(pooled_text.size());
        cpu_matmul_bias(pooled_text.data(), weights_.text_proj_weight.data(),
                        bias_or_null(weights_.text_proj_bias), text_proj.data(), 1, text_in_dim,
                        dim);
        cpu_silu_inplace(text_proj.data(), static_cast<std::size_t>(dim));

        if (!weights_.text_proj_2_weight.empty()) {
            std::vector<float> text_proj2(static_cast<std::size_t>(dim));
            cpu_matmul_bias(text_proj.data(), weights_.text_proj_2_weight.data(),
                            bias_or_null(weights_.text_proj_2_bias), text_proj2.data(), 1, dim,
                            dim);
            text_proj = std::move(text_proj2);
        }
    }

    // Guidance embedding MLP (if guidance_embeds is enabled)
    std::vector<float> guidance_proj(static_cast<std::size_t>(dim), 0.0F);
    if (timestep > 0.99F) {
        std::cerr << "[flux-temb] guidance_embeds=" << config_.guidance_embeds
                  << " g_w0=" << weights_.guidance_emb_0_weight.size()
                  << " g_w2=" << weights_.guidance_emb_2_weight.size() << "\n";
    }
    if (config_.guidance_embeds && !weights_.guidance_emb_0_weight.empty()) {
        // Diffusers FLUX forward currently scales guidance by 1000 before
        // feeding it into time_text_embed (same convention as timestep).
        std::vector<float> g_emb;
        fill_flux_sinusoidal_embedding(guidance * 1000.0F, freq_dim, g_emb);

        // Linear -> SiLU -> Linear
        std::vector<float> g_proj(static_cast<std::size_t>(dim));
        cpu_matmul_bias(g_emb.data(), weights_.guidance_emb_0_weight.data(),
                        bias_or_null(weights_.guidance_emb_0_bias), g_proj.data(), 1, freq_dim,
                        dim);
        cpu_silu_inplace(g_proj.data(), static_cast<std::size_t>(dim));

        cpu_matmul_bias(g_proj.data(), weights_.guidance_emb_2_weight.data(),
                        bias_or_null(weights_.guidance_emb_2_bias), guidance_proj.data(), 1, dim,
                        dim);

        if (timestep > 0.99F) {
            float gmin = guidance_proj[0], gmax = guidance_proj[0];
            double gsum = 0.0;
            for (auto v : guidance_proj) {
                gmin = std::min(gmin, v);
                gmax = std::max(gmax, v);
                gsum += static_cast<double>(v);
            }
            std::cerr << "[flux-temb] guidance_proj=[" << gmin << "," << gmax
                      << ",mean=" << (gsum / static_cast<double>(dim)) << "]\n";
        }
    }

    combine_flux_embeddings(t_proj2, text_proj, guidance_proj, temb);
    log_flux_temb_stats(timestep, guidance, temb);
}

// ===========================================================================
// FLUX 2D RoPE (CPU math, identical to old backend)
// ===========================================================================

void FluxPipeline::compute_flux_rope(int32_t h_patches, int32_t w_patches, int32_t text_seq_len,
                                     std::vector<float>& cos_out,
                                     std::vector<float>& sin_out) const {
    const int32_t head_dim = config_.dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t num_img_tokens = h_patches * w_patches;
    const int32_t total_seq = text_seq_len + num_img_tokens;

    cos_out.resize(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 1.0F);
    sin_out.resize(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 0.0F);

    // FLUX uses multi-axis RoPE: (text_pos, h_pos, w_pos [, extra_pos])
    // FLUX.1 default axes = (16, 56, 56) => 3D, total = 128 = head_dim
    // FLUX.2 default axes = (32, 32, 32, 32) => 4D, total = 128 = head_dim
    const float theta = config_.rope_theta;

    std::vector<int32_t> axes = config_.axes_dims_rope;
    if (axes.empty()) {
        axes = {16, 56, 56}; // FLUX.1 default
    }

    auto encode_pos = [&](float* cos_row, float* sin_row, int32_t temporal_pos, int32_t h_pos,
                          int32_t w_pos, int32_t sequence_pos) {
        int32_t offset = 0;
        for (std::size_t ax = 0; ax < axes.size(); ++ax) {
            const int32_t ax_dim = axes[ax];
            const int32_t pos =
                diffusion::flux_rope::axis_position(ax, temporal_pos, h_pos, w_pos, sequence_pos);

            for (int32_t i = 0; i < ax_dim / 2; ++i) {
                const float freq = 1.0F / std::pow(theta, 2.0F * static_cast<float>(i) /
                                                              static_cast<float>(ax_dim));
                const float angle = static_cast<float>(pos) * freq;
                cos_row[offset + 2 * i] = std::cos(angle);
                cos_row[offset + 2 * i + 1] = std::cos(angle);
                sin_row[offset + 2 * i] = std::sin(angle);
                sin_row[offset + 2 * i + 1] = std::sin(angle);
            }
            offset += ax_dim;
        }
    };

    // FLUX.2 text positions are (T=0, H=0, W=0, L=token index).
    // FLUX.1 has three axes, so the sequence coordinate is ignored.
    for (int32_t t = 0; t < text_seq_len; ++t) {
        encode_pos(
            cos_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim),
            sin_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim), 0, 0,
            0, t);
    }

    // Image tokens: position (0, h, w)
    for (int32_t h = 0; h < h_patches; ++h) {
        for (int32_t w = 0; w < w_patches; ++w) {
            const int32_t idx = text_seq_len + h * w_patches + w;
            encode_pos(
                cos_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                sin_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                0, h, w, 0);
        }
    }
}

// ===========================================================================
// generate_image helpers (extracted for cyclomatic complexity)
// ===========================================================================

// Steps 1-5: Prompt prep, tokenize, plan, CLIP, T5
bool FluxPipeline::prepare_conditioning(const std::string& prompt, const GenerateConfig& cfg,
                                        diffusion::FluxGenerationPlan& plan,
                                        std::vector<float>& pooled_output,
                                        std::vector<float>& text_embeddings) {
    // Detect FLUX.2 via VAE BN weights presence
    const bool is_flux2 = !weights_.vae_bn_mean.empty();

    // 1. Prepare prompt (FLUX.2 chat template)
    const std::string prepared = prepare_flux_prompt(prompt, is_flux2);
    raw_prompt_ = prepared;

    // 2. Tokenize with primary tokenizer (T5)
    std::vector<int32_t> input_ids;
    if (tokenizer_) {
        input_ids = tokenizer_->encode(prepared);
    }

    // 3. Build generation plan
    plan =
        diffusion::make_flux_generation_plan(config_, weights_, cfg.num_steps, cfg.guidance_scale,
                                             h_latent_, w_latent_, num_img_tokens_);

    // 4. Run CLIP encoder (if available, index 0)
    auto run_clip = [this](const std::vector<int32_t>& ids, std::vector<float>& pooled) {
        return run_clip_encoder(ids, pooled);
    };
    if (!prepare_flux_clip_conditioning(input_ids, static_cast<int32_t>(text_encoders_.size()),
                                        clip_tokenizer_.get(), raw_prompt_, run_clip,
                                        pooled_output)) {
        std::cerr << "[flux] CLIP encoder failed\n";
        return false;
    }

    // 5. Run T5 encoder
    auto run_t5 = [this](int32_t idx, const std::vector<int32_t>& ids,
                         std::vector<float>& embeddings) {
        return run_t5_encoder(idx, ids, embeddings);
    };
    if (!prepare_flux_t5_conditioning(input_ids, static_cast<int32_t>(text_encoders_.size()),
                                      run_t5, text_embeddings)) {
        std::cerr << "[flux] T5 encoder failed\n";
        return false;
    }

    return true;
}

// Steps 6-8: Context projection, RoPE, latents
void FluxPipeline::prepare_denoising_state(const diffusion::FluxGenerationPlan& plan,
                                           const std::vector<float>& text_embeddings,
                                           std::vector<float>& encoder_hidden,
                                           std::vector<float>& cos_vals,
                                           std::vector<float>& sin_vals,
                                           std::vector<float>& latents) {
    using Clock = std::chrono::steady_clock;
    const int32_t dit_dim = plan.dit_dim;
    const int32_t text_seq = plan.text_seq;
    const auto& layout = plan.layout;

    // 6. Context embedder projection
    auto tp0 = Clock::now();
    const int32_t t5_dim = config_.text_encoder_dim;
    const bool is_flux2 = plan.is_flux2;
    if (is_flux2) {
        // FLUX.2: context embedder is baked into TRT engine — pass raw T5 embeddings
        encoder_hidden = text_embeddings;
    } else {
        // FLUX.1: project T5 embeddings via CPU/cuBLAS
        encoder_hidden.assign(
            static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim), 0.0F);
        project_flux_encoder_hidden(text_embeddings, weights_.context_embed_weight,
                                    weights_.context_embed_bias, text_seq, t5_dim, dit_dim,
                                    encoder_hidden);
    }
    auto tp1 = Clock::now();

    // 7. Compute RoPE
    compute_flux_rope(layout.h_packed, layout.w_packed, text_seq, cos_vals, sin_vals);
    auto tp2 = Clock::now();

    // 8. Initialize random latents
    latents.resize(plan.latent_size);
    initialize_flux_latents(latents);
    auto tp3 = Clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cerr << "[flux-perf] Denoise state: ctx_proj=" << ms(tp0, tp1)
              << "ms, RoPE=" << ms(tp1, tp2) << "ms, latent_init=" << ms(tp2, tp3) << "ms\n";
}

// Step 10: Denoising loop setup + run
bool FluxPipeline::run_denoising(const diffusion::FluxGenerationPlan& plan,
                                 const std::vector<float>& pooled_output,
                                 std::vector<float>& encoder_hidden, std::vector<float>& cos_vals,
                                 std::vector<float>& sin_vals, std::vector<float>& latents) {
    const bool is_flux2 = plan.is_flux2;
    const int32_t num_inference_steps = plan.num_inference_steps;
    const float guidance_scale = plan.guidance_scale;
    const int32_t dit_dim = plan.dit_dim;
    const int32_t z_dim = plan.z_dim;
    const auto& layout = plan.layout;

    std::cerr << "[flux] Starting denoising loop (" << num_inference_steps << " steps)"
              << " latents=[" << z_dim << "," << h_latent_ << "," << w_latent_ << "]"
              << " packed=[" << num_img_tokens_ << "," << layout.packed_channels << "] ...\n";

    // FLUX.2: x_embedder is baked into TRT engine → hidden holds packed latents.
    // FLUX.1: x_embedder is still external → hidden holds embedded dim.
    const int32_t hidden_dim = is_flux2 ? layout.packed_channels : dit_dim;
    std::vector<float> hidden(static_cast<std::size_t>(num_img_tokens_) *
                              static_cast<std::size_t>(hidden_dim));
    std::vector<float> denoiser_output;

    // FLUX.2: temb MLP is baked into TRT engine — compute_temb just stores the
    // raw timestep for run_denoiser, which passes it directly to the engine.
    // FLUX.1: temb is computed on CPU/cuBLAS and passed as a precomputed vector.
    float current_timestep = 0.0F;
    const auto compute_temb = [this, is_flux2, guidance_scale, &pooled_output,
                               &current_timestep](float t, std::vector<float>& temb_out) {
        if (is_flux2) {
            current_timestep = t;
            temb_out.resize(1); // placeholder — not used by run_flux2_denoiser
        } else {
            compute_flux_timestep_embedding(t, guidance_scale, pooled_output, temb_out);
        }
    };
    const auto run_denoiser_fn = [this, is_flux2, guidance_scale, &encoder_hidden, &cos_vals,
                                  &sin_vals, &current_timestep](const std::vector<float>& hidden_in,
                                                                const std::vector<float>& temb_in,
                                                                std::vector<float>& output) {
        if (is_flux2) {
            return run_flux2_denoiser(hidden_in, encoder_hidden, current_timestep, guidance_scale,
                                      cos_vals, sin_vals, output);
        }
        return run_flux_denoiser(hidden_in, encoder_hidden, temb_in, cos_vals, sin_vals, output);
    };

    // Pack/unpack: FLUX.2 uses simple CHW->HWC, FLUX.1 uses 2x2 spatial packing
    auto pack_latents_fn = make_flux_pack_fn(is_flux2, z_dim, h_latent_, w_latent_, layout);
    auto unpack_velocity_fn = make_flux_unpack_fn(is_flux2, z_dim, h_latent_, w_latent_, layout);

    // FLUX.2: x_embedder baked into TRT engine — just pass packed latents through.
    // FLUX.1: x_embedder still external — apply CPU/GPU matmul.
    std::function<void(const std::vector<float>&, std::vector<float>&)> embed_hidden;
    if (is_flux2) {
        embed_hidden = [](const std::vector<float>& packed, std::vector<float>& out) {
            out = packed;
        };
    } else {
        embed_hidden =
            make_flux_hidden_embedder(weights_.patch_embed_weight, weights_.patch_embed_bias,
                                      num_img_tokens_, layout, dit_dim);
    }

    FlowMatchEulerState scheduler = diffusion::make_flux_scheduler_state(plan);
    log_flux_dynamic_shift(scheduler);
    if (!run_flux_denoising_loop(scheduler, num_inference_steps, latents, hidden, denoiser_output,
                                 pack_latents_fn, unpack_velocity_fn, compute_temb, embed_hidden,
                                 run_denoiser_fn)) {
        return false;
    }

    maybe_dump_flux_latents(latents);
    return true;
}

// Steps 11-13: VAE decode and convert to ImageResult
bool FluxPipeline::decode_and_convert(const diffusion::FluxGenerationPlan& plan,
                                      std::vector<float>& latents, ImageResult& result) {
    if (tensor_parallel_size_ > 1 && tensor_parallel_rank_ != 0) {
        std::cerr << "[flux] TP rank " << tensor_parallel_rank_
                  << " skips VAE decode; rank 0 writes image artifacts\n";
        result.pixels.clear();
        result.height = 0;
        result.width = 0;
        result.num_frames = 0;
        return true;
    }

    const bool is_flux2 = plan.is_flux2;
    const int32_t z_dim = plan.z_dim;
    const auto& layout = plan.layout;

    // 11. Prepare VAE input: BN denorm + unpatchify for FLUX.2, identity for FLUX.1
    std::vector<float> vae_latents;
    prepare_flux2_vae_input(latents, layout, z_dim, h_latent_, w_latent_, weights_.vae_bn_mean,
                            weights_.vae_bn_var, is_flux2, vae_latents);

    // 12. Decode VAE via TrtModule
    std::cerr << "[flux] Decoding latents via TrtModule VAE ...\n";

    const int32_t h_out = config_.video_height;
    const int32_t w_out = config_.video_width;

    TensorMap vae_inputs;
    vae_inputs["latents"] = Tensor{vae_latents.data(),
                                   {static_cast<int64_t>(z_dim), static_cast<int64_t>(h_latent_),
                                    static_cast<int64_t>(w_latent_)},
                                   DType::kFloat32};

    auto vae_outputs = vae_->forward(vae_inputs);

    auto& image_tensor = vae_outputs.at("image");
    const auto* vae_out_data = static_cast<const float*>(image_tensor.data);

    // 13. Convert to ImageResult
    convert_flux_vae_output_to_image(vae_out_data, h_out, w_out, result);

    std::cerr << "[flux] Image generated: " << result.width << "x" << result.height << "\n";
    return true;
}

// ===========================================================================
// generate_image — thin wrapper over generate_image_batch so the two code
// paths can never diverge (Decision D).
// ===========================================================================

ImageResult FluxPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    // Preserve legacy default-seed behavior: when ``cfg.seed`` is unset
    // (``< 0``), use the historical hardcoded value (42) verbatim instead of
    // passing it through ``derive_per_sample_seeds``. That way default-seed
    // golden images are bit-identical to pre-PR-2. When the caller does pass
    // a seed, we use it directly as the per-sample seed (matches the legacy
    // intent of ``cfg.seed`` reaching the noise RNG).
    const std::uint32_t per_sample_seed =
        (cfg.seed >= 0) ? static_cast<std::uint32_t>(cfg.seed) : 42U;
    auto batch = generate_image_batch({prompt}, {per_sample_seed}, cfg);
    if (batch.empty()) {
        return ImageResult{};
    }
    return std::move(batch.front());
}

// ===========================================================================
// generate_image_batch — Decisions B/D/E
//
// - One forward per step (FLUX is guidance-distilled; no cond/uncond fusion).
// - VAE always sliced at B=1: B sequential VAE forwards per chunk.
// - When total > max_batch_size.dit we chunk silently via plan_chunks().
//
// Internally:
// - Chunk size == 1 hits the legacy single-sample path so existing engines
//   (built with max_batch_size==1, static shapes) keep working unchanged.
// - Chunk size > 1 routes through run_t5_encoder_batch / run_flux*_denoiser_batch,
//   which require the engines to be built with max_batch_size > 1
//   (dynamic leading batch dim — see PR 1 builders).
// ===========================================================================

std::vector<ImageResult>
FluxPipeline::generate_image_batch(const std::vector<std::string>& prompts,
                                   const std::vector<std::uint32_t>& per_sample_seeds,
                                   const GenerateConfig& cfg) {
    if (prompts.size() != per_sample_seeds.size()) {
        throw std::invalid_argument("FluxPipeline::generate_image_batch: prompts.size() must equal "
                                    "per_sample_seeds.size()");
    }
    if (prompts.empty()) {
        return {};
    }
    if (!cfg.initial_latents.empty() && prompts.size() != 1U) {
        throw std::invalid_argument(
            "FluxPipeline::generate_image_batch: caller initial latents require one prompt");
    }

    const auto total = static_cast<int32_t>(prompts.size());
    const int32_t cap = std::max(1, config_.max_batch_size.dit);
    const auto chunks = flux_batch::plan_chunks(total, cap);

    std::vector<ImageResult> results;
    results.reserve(prompts.size());

    std::size_t cursor = 0;
    for (int32_t chunk_size : chunks) {
        if (chunk_size == 1) {
            // Single-sample fast path — preserves legacy behavior end-to-end.
            results.push_back(
                generate_one_for_batch(prompts[cursor], per_sample_seeds[cursor], cfg));
        } else {
            auto chunk_results =
                generate_image_batch_chunk(prompts, per_sample_seeds, cursor, chunk_size, cfg);
            if (chunk_results.empty()) {
                // Sub-stage failed — preserve legacy semantics by bailing out with
                // whatever results we've accumulated dropped (matches the old
                // ``return {};`` behavior inside the chunk body).
                return {};
            }
            std::move(chunk_results.begin(), chunk_results.end(), std::back_inserter(results));
        }
        cursor += static_cast<std::size_t>(chunk_size);
    }

    return results;
}

// ===========================================================================
// Batched chunk sub-helpers — extracted from prepare_flux_batch_conditioning
// and run_flux_denoising_loop_batch to keep each method under the CCN gate.
// ===========================================================================

bool FluxPipeline::run_flux_clip_batch_for_chunk(
    const std::vector<std::vector<int32_t>>& per_sample_input_ids,
    const std::vector<std::string>& prepared_prompts, int32_t B, std::vector<float>& pooled_batch) {
    pooled_batch.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(kFluxClipDim), 0.0F);
    auto run_clip = [this](const std::vector<int32_t>& ids, std::vector<float>& p) {
        return run_clip_encoder(ids, p);
    };
    for (int32_t b = 0; b < B; ++b) {
        std::vector<float> pooled_one;
        if (!prepare_flux_clip_conditioning(
                per_sample_input_ids[static_cast<std::size_t>(b)],
                static_cast<int32_t>(text_encoders_.size()), clip_tokenizer_.get(),
                prepared_prompts[static_cast<std::size_t>(b)], run_clip, pooled_one)) {
            std::cerr << "[flux] CLIP encoder failed in batch\n";
            return false;
        }
        std::copy_n(pooled_one.begin(),
                    std::min(pooled_one.size(), static_cast<std::size_t>(kFluxClipDim)),
                    pooled_batch.begin() +
                        static_cast<std::size_t>(b) * static_cast<std::size_t>(kFluxClipDim));
    }
    return true;
}

void FluxPipeline::project_flux_context_embed_batch(const std::vector<float>& text_embeddings_batch,
                                                    int32_t B, int32_t text_seq, int32_t t5_dim,
                                                    int32_t dit_dim, bool is_flux2,
                                                    std::vector<float>& encoder_hidden_batch) {
    // FLUX.2: context embedder is baked into the TRT engine — pass raw T5
    // embeddings unchanged. FLUX.1: project per sample via CPU/cuBLAS and
    // concatenate into a [B, text_seq, dit_dim] tensor.
    if (is_flux2) {
        encoder_hidden_batch = text_embeddings_batch;
        return;
    }
    encoder_hidden_batch.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(text_seq) *
                                    static_cast<std::size_t>(dit_dim),
                                0.0F);
    const auto per_sample_in =
        static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(t5_dim);
    const auto per_sample_out =
        static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim);
    std::vector<float> one_in(per_sample_in);
    std::vector<float> one_out(per_sample_out, 0.0F);
    for (int32_t b = 0; b < B; ++b) {
        std::copy_n(text_embeddings_batch.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_in),
                    per_sample_in, one_in.begin());
        std::fill(one_out.begin(), one_out.end(), 0.0F);
        project_flux_encoder_hidden(one_in, weights_.context_embed_weight,
                                    weights_.context_embed_bias, text_seq, t5_dim, dit_dim,
                                    one_out);
        std::copy_n(one_out.begin(), per_sample_out,
                    encoder_hidden_batch.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_out));
    }
}

std::function<void(const std::vector<float>&, std::vector<float>&)>
FluxPipeline::make_flux_embed_hidden_for_batch(bool is_flux2,
                                               const diffusion::FluxPackLayout& layout,
                                               int32_t dit_dim) {
    if (is_flux2) {
        return [](const std::vector<float>& packed, std::vector<float>& out) { out = packed; };
    }
    return make_flux_hidden_embedder(weights_.patch_embed_weight, weights_.patch_embed_bias,
                                     num_img_tokens_, layout, dit_dim);
}

// ===========================================================================
// generate_image_batch_chunk — drive a single chunk (chunk_size > 1) of the
// batched pipeline. Extracted from generate_image_batch purely to satisfy the
// cyclomatic-complexity gate; preserves semantics 1:1 with the original
// inline body.
// ===========================================================================

std::vector<ImageResult> FluxPipeline::generate_image_batch_chunk(
    const std::vector<std::string>& prompts, const std::vector<std::uint32_t>& per_sample_seeds,
    std::size_t chunk_begin, int32_t B, const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    const auto t_chunk_start = Clock::now();

    diffusion::FluxGenerationPlan plan;
    std::vector<float> pooled_batch;
    std::vector<float> encoder_hidden_batch;
    std::vector<float> cos_batch;
    std::vector<float> sin_batch;
    std::vector<float> latents_batch;

    if (!prepare_flux_batch_conditioning(prompts, per_sample_seeds, chunk_begin, B, cfg, plan,
                                         pooled_batch, encoder_hidden_batch, cos_batch, sin_batch,
                                         latents_batch)) {
        return {};
    }

    if (!run_flux_denoising_loop_batch(B, plan, pooled_batch, encoder_hidden_batch, cos_batch,
                                       sin_batch, latents_batch)) {
        return {};
    }

    std::vector<ImageResult> chunk_results;
    chunk_results.reserve(static_cast<std::size_t>(B));
    decode_flux_vae_per_sample(B, plan, latents_batch, chunk_results);

    const auto t_chunk_end = Clock::now();
    const double chunk_ms =
        std::chrono::duration<double, std::milli>(t_chunk_end - t_chunk_start).count();
    std::cerr << "[flux-batch] Chunk B=" << B << " done in " << chunk_ms << " ms (" << chunk_ms / B
              << " ms/sample)\n";

    return chunk_results;
}

// ===========================================================================
// prepare_flux_batch_conditioning — Steps 1-8 batched for one chunk.
// ===========================================================================

bool FluxPipeline::prepare_flux_batch_conditioning(
    const std::vector<std::string>& prompts, const std::vector<std::uint32_t>& per_sample_seeds,
    std::size_t chunk_begin, int32_t B, const GenerateConfig& cfg,
    diffusion::FluxGenerationPlan& plan, std::vector<float>& pooled_batch,
    std::vector<float>& encoder_hidden_batch, std::vector<float>& cos_batch,
    std::vector<float>& sin_batch, std::vector<float>& latents_batch) {
    const bool is_flux2 = !weights_.vae_bn_mean.empty();
    const auto chunk_end = chunk_begin + static_cast<std::size_t>(B);

    // 1-3. Prompt prep + tokenize + generation plan.
    std::vector<std::vector<int32_t>> per_sample_input_ids;
    per_sample_input_ids.reserve(static_cast<std::size_t>(B));
    std::vector<std::string> prepared_prompts;
    prepared_prompts.reserve(static_cast<std::size_t>(B));
    for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
        const std::string prepared = prepare_flux_prompt(prompts[i], is_flux2);
        prepared_prompts.push_back(prepared);
        std::vector<int32_t> ids;
        if (tokenizer_) {
            ids = tokenizer_->encode(prepared);
        }
        per_sample_input_ids.push_back(std::move(ids));
    }
    raw_prompt_ = prepared_prompts.front();

    plan =
        diffusion::make_flux_generation_plan(config_, weights_, cfg.num_steps, cfg.guidance_scale,
                                             h_latent_, w_latent_, num_img_tokens_);

    // 4. Per-sample CLIP (cheap, ~77 tokens).
    if (!run_flux_clip_batch_for_chunk(per_sample_input_ids, prepared_prompts, B, pooled_batch)) {
        return false;
    }

    // 5. Batched T5: returns [B, seq, te_dim] contiguous.
    const int32_t t5_idx = (static_cast<int32_t>(text_encoders_.size()) > 1) ? 1 : 0;
    std::vector<float> text_embeddings_batch;
    if (!run_t5_encoder_batch(t5_idx, per_sample_input_ids, text_embeddings_batch)) {
        std::cerr << "[flux] Batched T5 encoder failed\n";
        return false;
    }

    // 6. Context embedder projection (FLUX.1 only).
    const int32_t dit_dim = plan.dit_dim;
    const int32_t text_seq = plan.text_seq;
    const int32_t t5_dim = config_.text_encoder_dim;
    project_flux_context_embed_batch(text_embeddings_batch, B, text_seq, t5_dim, dit_dim, is_flux2,
                                     encoder_hidden_batch);

    // 7. RoPE: shared across the batch positionally — replicate B times so
    //    the DiT engine receives [B, total_seq, head_dim] as built (PR 1).
    std::vector<float> cos_one, sin_one;
    compute_flux_rope(plan.layout.h_packed, plan.layout.w_packed, text_seq, cos_one, sin_one);
    cos_batch.assign(static_cast<std::size_t>(B) * cos_one.size(), 0.0F);
    sin_batch.assign(static_cast<std::size_t>(B) * sin_one.size(), 0.0F);
    for (int32_t b = 0; b < B; ++b) {
        std::copy_n(cos_one.begin(), cos_one.size(),
                    cos_batch.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * cos_one.size()));
        std::copy_n(sin_one.begin(), sin_one.size(),
                    sin_batch.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * sin_one.size()));
    }

    // 8. Per-sample initial latents (per-sample RNG; see Diffusers
    //    randn_tensor convention — research brief §1).
    const auto per_sample_latent = plan.latent_size;
    latents_batch.assign(static_cast<std::size_t>(B) * per_sample_latent, 0.0F);
    for (int32_t b = 0; b < B; ++b) {
        std::vector<float> one(per_sample_latent);
        initialize_flux_latents(one, per_sample_seeds[chunk_begin + static_cast<std::size_t>(b)]);
        std::copy_n(one.begin(), per_sample_latent,
                    latents_batch.begin() + static_cast<std::ptrdiff_t>(
                                                static_cast<std::size_t>(b) * per_sample_latent));
    }

    return true;
}

// ===========================================================================
// run_flux_denoising_loop_batch — Step 10 batched for one chunk.
// ===========================================================================

bool FluxPipeline::run_flux_denoising_loop_batch(
    int32_t B, const diffusion::FluxGenerationPlan& plan, const std::vector<float>& pooled_batch,
    const std::vector<float>& encoder_hidden_batch, const std::vector<float>& cos_batch,
    const std::vector<float>& sin_batch, std::vector<float>& latents_batch) {
    const bool is_flux2 = plan.is_flux2;
    const int32_t z_dim = plan.z_dim;
    const auto& layout = plan.layout;
    const int32_t dit_dim = plan.dit_dim;
    const int32_t hidden_dim = is_flux2 ? layout.packed_channels : dit_dim;
    const auto per_sample_latent = plan.latent_size;
    const auto per_sample_hidden =
        static_cast<std::size_t>(num_img_tokens_) * static_cast<std::size_t>(hidden_dim);

    std::vector<float> hidden_batch(static_cast<std::size_t>(B) * per_sample_hidden, 0.0F);
    std::vector<float> denoiser_output_batch;
    std::vector<float> velocity_batch(latents_batch.size());
    std::vector<float> next_latents_batch(latents_batch.size());

    auto pack_fn = make_flux_pack_fn(is_flux2, z_dim, h_latent_, w_latent_, layout);
    auto unpack_fn = make_flux_unpack_fn(is_flux2, z_dim, h_latent_, w_latent_, layout);
    auto embed_hidden = make_flux_embed_hidden_for_batch(is_flux2, layout, dit_dim);

    FlowMatchEulerState scheduler = diffusion::make_flux_scheduler_state(plan);
    log_flux_dynamic_shift(scheduler);
    // Per-sample temb for FLUX.1 (depends on pooled_output). For FLUX.2 we
    // just pass the raw scalar timestep; the engine carries the temb MLP.
    const float guidance_scale = plan.guidance_scale;
    const int32_t num_inference_steps = plan.num_inference_steps;

    std::vector<float> temb_batch;
    if (!is_flux2) {
        temb_batch.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(dit_dim), 0.0F);
    }

    std::vector<float> packed_one(static_cast<std::size_t>(num_img_tokens_) *
                                      static_cast<std::size_t>(layout.packed_channels),
                                  0.0F);
    std::vector<float> hidden_one(per_sample_hidden, 0.0F);
    std::vector<float> latent_one(per_sample_latent, 0.0F);
    std::vector<float> pooled_one(static_cast<std::size_t>(kFluxClipDim), 0.0F);
    std::vector<float> temb_one(static_cast<std::size_t>(dit_dim), 0.0F);
    std::vector<float> velocity_one(per_sample_latent, 0.0F);

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float timestep_raw = scheduler.timesteps[static_cast<std::size_t>(step)];
        const float timestep_norm = timestep_raw / 1000.0F;

        // Per-sample temb (FLUX.1 only).
        if (!is_flux2) {
            for (int32_t b = 0; b < B; ++b) {
                std::copy_n(pooled_batch.begin() +
                                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) *
                                                            static_cast<std::size_t>(kFluxClipDim)),
                            static_cast<std::size_t>(kFluxClipDim), pooled_one.begin());
                compute_flux_timestep_embedding(timestep_norm, guidance_scale, pooled_one,
                                                temb_one);
                std::copy_n(temb_one.begin(), static_cast<std::size_t>(dit_dim),
                            temb_batch.begin() +
                                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) *
                                                            static_cast<std::size_t>(dit_dim)));
            }
        }

        // Pack + embed_hidden per sample.
        for (int32_t b = 0; b < B; ++b) {
            std::copy_n(
                latents_batch.begin() +
                    static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_latent),
                per_sample_latent, latent_one.begin());
            pack_fn(latent_one, packed_one);
            embed_hidden(packed_one, hidden_one);
            std::copy_n(
                hidden_one.begin(), per_sample_hidden,
                hidden_batch.begin() +
                    static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_hidden));
        }

        // Batched DiT forward.
        const bool ok =
            is_flux2 ? run_flux2_denoiser_batch(B, hidden_batch, encoder_hidden_batch, timestep_raw,
                                                guidance_scale, cos_batch, sin_batch,
                                                denoiser_output_batch)
                     : run_flux_denoiser_batch(B, hidden_batch, encoder_hidden_batch, temb_batch,
                                               cos_batch, sin_batch, denoiser_output_batch);
        if (!ok) {
            std::cerr << "[flux] Batched denoiser step " << step << " failed\n";
            return false;
        }

        // Unpack velocity per sample, then step the scheduler over the
        // contiguous [B, latent] buffer in one shot — the step kernel is
        // elementwise so a batched call is identical to B independent calls.
        const std::size_t per_sample_denoiser =
            denoiser_output_batch.size() / static_cast<std::size_t>(B);
        std::vector<float> dout_one(per_sample_denoiser);
        for (int32_t b = 0; b < B; ++b) {
            std::copy_n(
                denoiser_output_batch.begin() +
                    static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_denoiser),
                per_sample_denoiser, dout_one.begin());
            unpack_fn(dout_one, velocity_one);
            std::copy_n(
                velocity_one.begin(), per_sample_latent,
                velocity_batch.begin() +
                    static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * per_sample_latent));
        }
        scheduler.step(velocity_batch.data(), latents_batch.data(), next_latents_batch.data(),
                       latents_batch.size(), step);
        latents_batch.swap(next_latents_batch);

        std::cerr << "[flux-batch] Step " << (step + 1) << "/" << num_inference_steps
                  << " t=" << scheduler.timesteps[static_cast<std::size_t>(step)] << " B=" << B
                  << "\n";
    }

    return true;
}

// ===========================================================================
// decode_flux_vae_per_sample — Steps 11-13 batched (B sequential decodes).
// ===========================================================================

void FluxPipeline::decode_flux_vae_per_sample(int32_t B, const diffusion::FluxGenerationPlan& plan,
                                              const std::vector<float>& latents_batch,
                                              std::vector<ImageResult>& out) {
    const bool is_flux2 = plan.is_flux2;
    const int32_t z_dim = plan.z_dim;
    const auto& layout = plan.layout;
    const auto per_sample_latent = plan.latent_size;
    const bool is_rank0 = (tensor_parallel_size_ <= 1 || tensor_parallel_rank_ == 0);
    const int32_t h_out = config_.video_height;
    const int32_t w_out = config_.video_width;

    for (int32_t b = 0; b < B; ++b) {
        ImageResult one;
        if (!is_rank0) {
            out.push_back(std::move(one));
            continue;
        }
        std::vector<float> one_latent(per_sample_latent);
        std::copy_n(latents_batch.begin() + static_cast<std::ptrdiff_t>(
                                                static_cast<std::size_t>(b) * per_sample_latent),
                    per_sample_latent, one_latent.begin());
        std::vector<float> vae_latents;
        prepare_flux2_vae_input(one_latent, layout, z_dim, h_latent_, w_latent_,
                                weights_.vae_bn_mean, weights_.vae_bn_var, is_flux2, vae_latents);

        TensorMap vae_inputs;
        vae_inputs["latents"] =
            Tensor{vae_latents.data(),
                   {static_cast<int64_t>(z_dim), static_cast<int64_t>(h_latent_),
                    static_cast<int64_t>(w_latent_)},
                   DType::kFloat32};
        auto vae_outputs = vae_->forward(vae_inputs);
        auto& image_tensor = vae_outputs.at("image");
        const auto* vae_out_data = static_cast<const float*>(image_tensor.data);
        convert_flux_vae_output_to_image(vae_out_data, h_out, w_out, one);
        out.push_back(std::move(one));
    }
}

// ===========================================================================
// generate_one_for_batch — legacy single-sample path, used by both the
// public generate_image wrapper and the chunk-size-1 branch in
// generate_image_batch. Mirrors the original generate_image body with one
// change: the per-sample seed overrides the hardcoded 42 in
// initialize_flux_latents (matches the per-sample seed contract).
// ===========================================================================

ImageResult FluxPipeline::generate_one_for_batch(const std::string& prompt,
                                                 std::uint32_t per_sample_seed,
                                                 const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();

    ImageResult result;

    // Steps 1-5: Prompt prep, tokenize, plan, CLIP, T5
    diffusion::FluxGenerationPlan plan;
    std::vector<float> pooled_output;
    std::vector<float> text_embeddings;
    if (!prepare_conditioning(prompt, cfg, plan, pooled_output, text_embeddings)) {
        return result;
    }
    const auto t_cond = Clock::now();

    // Steps 6-8: Context projection, RoPE, latents (latents will be re-seeded below).
    std::vector<float> encoder_hidden;
    std::vector<float> cos_vals, sin_vals;
    std::vector<float> latents;
    prepare_denoising_state(plan, text_embeddings, encoder_hidden, cos_vals, sin_vals, latents);
    // Override latent init with the per-sample seed; this is the single
    // behavior change vs. the legacy code (which hardcoded seed=42).
    initialize_flux_latents(latents, per_sample_seed);
    std::string latent_error;
    if (!diffusion::apply_flux_initial_latents(plan.latent_size, cfg.initial_latents, latents,
                                               latent_error)) {
        std::cerr << "[flux] " << latent_error << "\n";
        return result;
    }
    const auto t_prep = Clock::now();

    // Step 10: Denoising loop
    if (!run_denoising(plan, pooled_output, encoder_hidden, cos_vals, sin_vals, latents)) {
        return result;
    }
    const auto t_denoise = Clock::now();

    // Steps 11-13: VAE decode and convert
    decode_and_convert(plan, latents, result);
    const auto t_vae = Clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const double total_ms = ms(t_start, t_vae);
    std::cerr << "\n[flux-perf] ===== Timing Summary =====\n"
              << "[flux-perf] Text encoding (CLIP+T5): " << ms(t_start, t_cond) << " ms\n"
              << "[flux-perf] Denoise prep (proj+RoPE): " << ms(t_cond, t_prep) << " ms\n"
              << "[flux-perf] Denoising (" << plan.num_inference_steps
              << " steps): " << ms(t_prep, t_denoise) << " ms ("
              << ms(t_prep, t_denoise) / plan.num_inference_steps << " ms/step)\n"
              << "[flux-perf] VAE decode:              " << ms(t_denoise, t_vae) << " ms\n"
              << "[flux-perf] Total E2E:               " << total_ms << " ms\n"
              << "[flux-perf] ===========================\n";

    return result;
}

} // namespace trtmc

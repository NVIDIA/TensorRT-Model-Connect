// FluxPipeline implementation: TrtModule-based FLUX diffusion pipeline.
// Ports flux_diffusion_backend.cpp from raw TRT API to TrtModule::forward().
//
// All GPU buffer management (CudaBuffer, CudaStream, setTensorAddress,
// enqueueV3, cudaMemcpy) is removed. TrtModule::forward() handles H2D/D2H
// internally. CPU math (timestep embedding, RoPE, packing/unpacking,
// sinusoidal embedding, matmul, BN denorm) is preserved identically.

#include "runtime/models/flux/pipeline.h"

#include "runtime/core/gpu_matmul.h"
#include "runtime/domains/diffusion/diffusion_denoising_step_seam.h"
#include "runtime/domains/diffusion/diffusion_generation_plan.h"
#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

namespace trtmc {

namespace {

using diffusion::FlowMatchEulerState;
using diffusion::FluxPackLayout;

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
        gpu_matmul_bias(A, B, bias, out, M, K, N);
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

std::vector<int32_t> make_clip_padded_ids(const std::vector<int32_t>& input_ids,
                                          int32_t pad_token_id) {
    std::vector<int32_t> padded(static_cast<std::size_t>(kFluxClipSeqLen),
                                std::max(pad_token_id, 0));
    const auto copy_len = std::min(static_cast<std::size_t>(kFluxClipSeqLen), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded.begin());
    return padded;
}

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

void initialize_flux_latents(std::vector<float>& latents) {
    std::mt19937 gen(42);
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
                           DiffusionConfig config, PreprocessorWeights weights,
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
    gpu_matmul_init();
}

FluxPipeline::~FluxPipeline() {
    gpu_matmul_shutdown();
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

    const auto padded = make_clip_padded_ids(input_ids, clip_pad_token_id);

    // Build input TensorMap
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{const_cast<int32_t*>(padded.data()), {kFluxClipSeqLen}, DType::kInt32};

    auto outputs = clip_module->forward(inputs);

    // Check if pooled_output is directly available from the engine
    if (outputs.count("pooled_output")) {
        auto& pooled_tensor = outputs.at("pooled_output");
        const auto* data = static_cast<const float*>(pooled_tensor.data);
        pooled_output.assign(data, data + pooled_tensor.numel());
        return true;
    }

    // Fallback: manually extract pooled row from text_embeddings at EOS position
    auto& text_emb_tensor = outputs.at("text_embeddings");
    const auto* clip_hidden = static_cast<const float*>(text_emb_tensor.data);
    const auto clip_hidden_size =
        static_cast<std::size_t>(kFluxClipSeqLen) * static_cast<std::size_t>(kFluxClipDim);
    std::vector<float> clip_hidden_vec(clip_hidden, clip_hidden + clip_hidden_size);

    const int32_t pool_idx = select_clip_pool_index(padded, clip_eos_token_id);
    copy_clip_pooled_row(clip_hidden_vec, pool_idx, pooled_output);
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

    // Pad input_ids to seq_len
    std::vector<int32_t> padded_ids(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded_ids.begin());

    // Build attention mask: 0.0 for real tokens, -1e9 for padding
    std::vector<float> mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] != 0) {
            mask[static_cast<std::size_t>(i)] = 0.0F;
        }
    }

    TensorMap inputs;
    inputs["input_ids"] = Tensor{padded_ids.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{mask.data(), {static_cast<int64_t>(seq_len)}, DType::kFloat32};

    auto outputs = te->forward(inputs);

    auto& emb_tensor = outputs.at("text_embeddings");
    const auto emb_size = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim);
    const auto* emb_data = static_cast<const float*>(emb_tensor.data);
    text_embeddings.assign(emb_data, emb_data + emb_size);

    // Zero out padding token embeddings
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] == 0) {
            float* row = text_embeddings.data() +
                         static_cast<std::size_t>(i) * static_cast<std::size_t>(te_dim);
            std::fill_n(row, static_cast<std::size_t>(te_dim), 0.0F);
        }
    }

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

    auto encode_pos = [&](float* cos_row, float* sin_row, int32_t text_pos, int32_t h_pos,
                          int32_t w_pos) {
        int32_t offset = 0;
        for (std::size_t ax = 0; ax < axes.size(); ++ax) {
            const int32_t ax_dim = axes[ax];
            // Determine position value for this axis
            int32_t pos = 0;
            if (ax == 0)
                pos = text_pos;
            else if (ax == 1)
                pos = h_pos;
            else if (ax == 2)
                pos = w_pos;
            // axes[3+] default to position 0 (identity rotation for image tokens)

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

    // Text tokens: ALL positions are (0, 0, 0) -- identity rotation
    for (int32_t t = 0; t < text_seq_len; ++t) {
        encode_pos(
            cos_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim),
            sin_out.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(head_dim), 0, 0,
            0);
    }

    // Image tokens: position (0, h, w)
    for (int32_t h = 0; h < h_patches; ++h) {
        for (int32_t w = 0; w < w_patches; ++w) {
            const int32_t idx = text_seq_len + h * w_patches + w;
            encode_pos(
                cos_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                sin_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                0, h, w);
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
// generate_image — Full FLUX pipeline
// ===========================================================================

ImageResult FluxPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
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

    // Steps 6-8: Context projection, RoPE, latents
    std::vector<float> encoder_hidden;
    std::vector<float> cos_vals, sin_vals;
    std::vector<float> latents;
    prepare_denoising_state(plan, text_embeddings, encoder_hidden, cos_vals, sin_vals, latents);
    const auto t_prep = Clock::now();

    // Step 10: Denoising loop
    if (!run_denoising(plan, pooled_output, encoder_hidden, cos_vals, sin_vals, latents)) {
        return result;
    }
    const auto t_denoise = Clock::now();

    // Steps 11-13: VAE decode and convert
    decode_and_convert(plan, latents, result);
    const auto t_vae = Clock::now();

    // Timing summary
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

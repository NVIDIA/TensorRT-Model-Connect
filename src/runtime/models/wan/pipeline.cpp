/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// WanPipeline — TrtModule-based Wan2.1 diffusion pipeline.
// Ports WanDiffusionBackend to use TrtModule::forward() for all GPU work.
// Large preprocessing projections use cuBLAS with the exact CPU implementation
// retained as a fallback. Raw TRT calls use TrtModule::forward(TensorMap).

#include "runtime/models/wan/pipeline.h"

#include "runtime/models/wan/gpu_matmul.h"
#include "runtime/models/wan/wan_denoising_step_seam.h"
#include "runtime/models/wan/wan_generation_conditioning.h"
#include "runtime/models/wan/wan_generation_plan.h"
#include "runtime/models/wan/wan_matmul_policy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace trtmc {

namespace {

using diffusion::WanLayout;
using diffusion::wan_scheduler::FlowMatchEulerState;
using diffusion::wan_scheduler::UniPCFlowState;

// ---------------------------------------------------------------------------
// DDIM Scheduler (epsilon-prediction models like PixArt)
// ---------------------------------------------------------------------------

struct DDIMState {
    std::vector<double> sigmas; // [num_steps + 1]
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    std::vector<double> prev_x0;
    double prev_lambda_src{0.0};
    bool has_prev{false};

    void set_timesteps(int32_t num_steps, double beta_start = 0.0001, double beta_end = 0.02) {
        const int32_t T = num_train_timesteps;
        std::vector<double> alpha_cumprod(static_cast<std::size_t>(T));
        double cum = 1.0;
        for (int32_t i = 0; i < T; ++i) {
            double beta = beta_start + static_cast<double>(i) / static_cast<double>(T - 1) *
                                           (beta_end - beta_start);
            cum *= (1.0 - beta);
            alpha_cumprod[static_cast<std::size_t>(i)] = cum;
        }
        timesteps.resize(static_cast<std::size_t>(num_steps));
        for (int32_t i = 0; i < num_steps; ++i) {
            double frac = static_cast<double>(i) / static_cast<double>(num_steps);
            timesteps[static_cast<std::size_t>(i)] =
                static_cast<float>(std::round((1.0 - frac) * (T - 1)));
        }
        sigmas.resize(static_cast<std::size_t>(num_steps) + 1);
        for (int32_t i = 0; i < num_steps; ++i) {
            const int32_t t =
                static_cast<int32_t>(std::round(timesteps[static_cast<std::size_t>(i)]));
            const double acp =
                alpha_cumprod[static_cast<std::size_t>(std::max(0, std::min(t, T - 1)))];
            sigmas[static_cast<std::size_t>(i)] = std::sqrt((1.0 - acp) / acp);
        }
        sigmas[static_cast<std::size_t>(num_steps)] = 0.0;
    }

    void step(const float* eps_pred, const float* x_t, float* x_out, std::size_t count,
              int32_t step_index) {
        const auto si = static_cast<std::size_t>(step_index);

        const double raw_src = sigmas[si];
        const double raw_tgt = sigmas[si + 1];
        const double alp_src = 1.0 / std::sqrt(1.0 + raw_src * raw_src);
        const double sig_src = raw_src / std::sqrt(1.0 + raw_src * raw_src);
        const double alp_tgt = 1.0 / std::sqrt(1.0 + raw_tgt * raw_tgt);
        const double sig_tgt = raw_tgt / std::sqrt(1.0 + raw_tgt * raw_tgt);

        double lam_src = std::log(alp_src / sig_src);
        double lam_tgt = std::log(alp_tgt / sig_tgt);
        double h = lam_tgt - lam_src;

        double ratio = sig_tgt / sig_src;
        double coeff = -alp_tgt * std::expm1(-h);

        std::vector<double> x0(count);
        for (std::size_t i = 0; i < count; ++i) {
            x0[i] = (static_cast<double>(x_t[i]) - sig_src * static_cast<double>(eps_pred[i])) /
                    alp_src;
        }

        for (std::size_t i = 0; i < count; ++i) {
            x_out[i] = static_cast<float>(ratio * static_cast<double>(x_t[i]) + coeff * x0[i]);
        }
    }
};

// ---------------------------------------------------------------------------
// CPU math helpers (standalone — replaces DiffusionBackendBase methods)
// ---------------------------------------------------------------------------

void cpu_matmul_bias(const float* A, const float* B, const float* bias, float* out, int32_t M,
                     int32_t K, int32_t N) {
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

void cpu_gelu_tanh_inplace(float* data, std::size_t count) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    constexpr float kCoeff = 0.044715F;
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        data[i] = 0.5F * x * (1.0F + std::tanh(inner));
    }
}

// ---------------------------------------------------------------------------
// CHW -> HWC conversion helpers
// ---------------------------------------------------------------------------

float clamp_unit(float value) {
    return std::max(0.0F, std::min(1.0F, value));
}

void convert_wan_chw_to_hwc(const std::vector<float>& raw, int32_t h_out, int32_t w_out,
                            WanVideoResult& result) {
    result.height = h_out;
    result.width = w_out;
    result.num_frames = 1;
    result.frames.resize(static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out) * 3);
    const auto hw = static_cast<std::size_t>(h_out * w_out);
    for (int32_t y = 0; y < h_out; ++y) {
        for (int32_t x = 0; x < w_out; ++x) {
            for (int32_t ch = 0; ch < 3; ++ch) {
                const auto src =
                    static_cast<std::size_t>(ch) * hw + static_cast<std::size_t>(y * w_out + x);
                const auto dst = static_cast<std::size_t>(y) * static_cast<std::size_t>(w_out) * 3 +
                                 static_cast<std::size_t>(x) * 3 + static_cast<std::size_t>(ch);
                result.frames[dst] = clamp_unit((raw[src] + 1.0F) * 0.5F);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VAE latent preparation helpers
// ---------------------------------------------------------------------------

std::vector<float> prepare_wan_vae_2d_input(const std::vector<float>& latents,
                                            const WanDiffusionConfig& config,
                                            std::size_t input_size) {
    std::vector<float> scaled_latents(latents.begin(),
                                      latents.begin() + static_cast<std::ptrdiff_t>(input_size));
    if (config.latents_mean.empty() && config.vae_scaling_factor > 0.0F) {
        const float inv_sf = 1.0F / config.vae_scaling_factor;
        for (auto& v : scaled_latents) {
            v *= inv_sf;
        }
    }
    return scaled_latents;
}

void extract_wan_latent_frame(const std::vector<float>& latents, int32_t c, int32_t t_lat,
                              int32_t h_lat, int32_t w_lat, int32_t t,
                              std::vector<float>& frame_buf) {
    const auto spatial = static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    frame_buf.resize(static_cast<std::size_t>(c) * spatial);
    for (int32_t ci = 0; ci < c; ++ci) {
        const float* ch_src =
            latents.data() +
            static_cast<std::size_t>(ci) * static_cast<std::size_t>(t_lat) * spatial +
            static_cast<std::size_t>(t) * spatial;
        std::memcpy(frame_buf.data() + static_cast<std::size_t>(ci) * spatial, ch_src,
                    spatial * sizeof(float));
    }
}

void initialize_wan_video_result(int32_t num_frames, int32_t h_out, int32_t w_out,
                                 WanVideoResult& result) {
    result.num_frames = num_frames;
    result.height = h_out;
    result.width = w_out;
    result.frames.resize(static_cast<std::size_t>(num_frames) * static_cast<std::size_t>(h_out) *
                         static_cast<std::size_t>(w_out) * 3);
}

void copy_wan_vae_output_frame(const float* raw_base, int32_t chunk_t, int32_t sub_t,
                               int32_t final_t, int32_t h_out, int32_t w_out,
                               WanVideoResult& result) {
    const auto per_frame_spatial =
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    for (int32_t fh = 0; fh < h_out; ++fh) {
        for (int32_t fw = 0; fw < w_out; ++fw) {
            for (int32_t fc = 0; fc < 3; ++fc) {
                const auto s_idx = static_cast<std::size_t>(fc) *
                                       static_cast<std::size_t>(chunk_t) * per_frame_spatial +
                                   static_cast<std::size_t>(sub_t) * per_frame_spatial +
                                   static_cast<std::size_t>(fh) * static_cast<std::size_t>(w_out) +
                                   static_cast<std::size_t>(fw);
                const auto d_idx =
                    static_cast<std::size_t>(final_t) * per_frame_spatial * 3 +
                    static_cast<std::size_t>(fh) * static_cast<std::size_t>(w_out) * 3 +
                    static_cast<std::size_t>(fw) * 3 + static_cast<std::size_t>(fc);
                result.frames[d_idx] = clamp_unit((raw_base[s_idx] + 1.0F) * 0.5F);
            }
        }
    }
}

void compose_wan_vae_first_frame_chunks(const std::vector<float>& all_raw_frames, int32_t t_lat,
                                        int32_t t_out_per_frame, int32_t first_t_out, int32_t h_out,
                                        int32_t w_out, int32_t max_video_frames,
                                        WanVideoResult& result) {
    const int32_t total_out_frames = first_t_out + std::max(t_lat - 1, 0) * t_out_per_frame;
    initialize_wan_video_result(std::min(total_out_frames, max_video_frames), h_out, w_out, result);
    const auto per_frame_spatial =
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    std::size_t raw_offset = 0;
    int32_t final_t = 0;
    for (int32_t input_t = 0; input_t < t_lat && final_t < result.num_frames; ++input_t) {
        const int32_t chunk_t = (input_t == 0) ? first_t_out : t_out_per_frame;
        const float* raw_base = all_raw_frames.data() + raw_offset;
        for (int32_t sub_t = 0; sub_t < chunk_t && final_t < result.num_frames;
             ++sub_t, ++final_t) {
            copy_wan_vae_output_frame(raw_base, chunk_t, sub_t, final_t, h_out, w_out, result);
        }
        raw_offset +=
            static_cast<std::size_t>(3) * static_cast<std::size_t>(chunk_t) * per_frame_spatial;
    }
}

void compose_wan_vae_uniform_chunks(const std::vector<float>& all_raw_frames, int32_t t_lat,
                                    int32_t t_out_per_frame, int32_t h_out, int32_t w_out,
                                    int32_t scale_factor_temporal, int32_t max_video_frames,
                                    WanVideoResult& result) {
    const int32_t total_out_frames = t_lat * t_out_per_frame;
    const int32_t trim = scale_factor_temporal - 1;
    const int32_t t_final = total_out_frames - trim;
    initialize_wan_video_result(std::min(t_final, max_video_frames), h_out, w_out, result);
    const auto per_frame_spatial =
        static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    const auto out_frame_floats =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(t_out_per_frame) * per_frame_spatial;
    for (int32_t input_t = 0; input_t < t_lat; ++input_t) {
        const float* raw_base =
            all_raw_frames.data() + static_cast<std::size_t>(input_t) * out_frame_floats;
        for (int32_t sub_t = 0; sub_t < t_out_per_frame; ++sub_t) {
            const int32_t global_t = input_t * t_out_per_frame + sub_t;
            const int32_t final_t = global_t - trim;
            if (global_t < trim || final_t >= result.num_frames) {
                continue;
            }
            copy_wan_vae_output_frame(raw_base, t_out_per_frame, sub_t, final_t, h_out, w_out,
                                      result);
        }
    }
}

void compose_wan_vae_video_frames(const std::vector<float>& all_raw_frames, int32_t t_lat,
                                  int32_t t_out_per_frame, int32_t first_t_out, int32_t h_out,
                                  int32_t w_out, int32_t scale_factor_temporal,
                                  int32_t max_video_frames, WanVideoResult& result) {
    if (first_t_out != t_out_per_frame) {
        compose_wan_vae_first_frame_chunks(all_raw_frames, t_lat, t_out_per_frame, first_t_out,
                                           h_out, w_out, max_video_frames, result);
        return;
    }
    compose_wan_vae_uniform_chunks(all_raw_frames, t_lat, t_out_per_frame, h_out, w_out,
                                   scale_factor_temporal, max_video_frames, result);
}

// ---------------------------------------------------------------------------
// Positional embedding
// ---------------------------------------------------------------------------

void compute_wan_pos_embed_2d(int32_t nh_p, int32_t nw_p, int32_t dim,
                              std::vector<float>& pos_embed_2d) {
    const int32_t half_dim = dim / 2;
    const int32_t quarter_dim = half_dim / 2;
    const float interp_scale = 2.0F;
    pos_embed_2d.assign(static_cast<std::size_t>(nh_p * nw_p) * static_cast<std::size_t>(dim),
                        0.0F);

    std::vector<double> omega(static_cast<std::size_t>(quarter_dim));
    for (int32_t i = 0; i < quarter_dim; ++i) {
        omega[static_cast<std::size_t>(i)] =
            1.0 / std::pow(10000.0, static_cast<double>(i) / static_cast<double>(quarter_dim));
    }

    for (int32_t hi = 0; hi < nh_p; ++hi) {
        for (int32_t wi = 0; wi < nw_p; ++wi) {
            const int32_t patch_idx = hi * nw_p + wi;
            float* row = pos_embed_2d.data() +
                         static_cast<std::size_t>(patch_idx) * static_cast<std::size_t>(dim);
            const double h_pos = static_cast<double>(hi) / static_cast<double>(interp_scale);
            const double w_pos = static_cast<double>(wi) / static_cast<double>(interp_scale);
            for (int32_t d = 0; d < quarter_dim; ++d) {
                const double angle_w = w_pos * omega[static_cast<std::size_t>(d)];
                row[d] = static_cast<float>(std::sin(angle_w));
                row[quarter_dim + d] = static_cast<float>(std::cos(angle_w));
            }
            for (int32_t d = 0; d < quarter_dim; ++d) {
                const double angle_h = h_pos * omega[static_cast<std::size_t>(d)];
                row[half_dim + d] = static_cast<float>(std::sin(angle_h));
                row[half_dim + quarter_dim + d] = static_cast<float>(std::cos(angle_h));
            }
        }
    }
}

void add_wan_positional_embedding(std::vector<float>& hidden,
                                  const std::vector<float>& pos_embed_2d) {
    if (pos_embed_2d.empty()) {
        return;
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += pos_embed_2d[i];
    }
}

// ---------------------------------------------------------------------------
// CFG noise prediction
// ---------------------------------------------------------------------------

template <typename RunDenoiserFn>
bool predict_wan_noise(const std::vector<float>& hidden, const std::vector<float>& temb_6d,
                       const std::vector<float>& time_embed,
                       const std::vector<float>& text_projected,
                       const std::vector<float>& null_text,
                       const std::vector<float>& encoder_attn_mask, float guidance_scale,
                       std::vector<float>& denoiser_output, std::string& error,
                       RunDenoiserFn&& run_denoiser) {
    if (guidance_scale > 1.0F) {
        std::vector<float> cond_pred;
        std::vector<float> uncond_pred;
        std::vector<float> null_mask;
        if (!encoder_attn_mask.empty()) {
            null_mask.assign(encoder_attn_mask.size(), 0.0F);
        }

        if (!run_denoiser(hidden, temb_6d, time_embed, text_projected, encoder_attn_mask, cond_pred,
                          error)) {
            return false;
        }
        if (!run_denoiser(hidden, temb_6d, time_embed, null_text, null_mask, uncond_pred, error)) {
            return false;
        }

        denoiser_output.resize(cond_pred.size());
        for (std::size_t i = 0; i < cond_pred.size(); ++i) {
            denoiser_output[i] = uncond_pred[i] + guidance_scale * (cond_pred[i] - uncond_pred[i]);
        }
        return true;
    }

    return run_denoiser(hidden, temb_6d, time_embed, text_projected, encoder_attn_mask,
                        denoiser_output, error);
}

// ---------------------------------------------------------------------------
// Output truncation (for models with out_channels=2*z_dim)
// ---------------------------------------------------------------------------

void maybe_truncate_wan_output(std::vector<float>& denoiser_output, int32_t num_patches,
                               int32_t z_dim, int32_t pt, int32_t ph, int32_t pw) {
    const int32_t expected_patch_out = z_dim * pt * ph * pw;
    const auto actual_patch_out =
        static_cast<int32_t>(denoiser_output.size() / static_cast<std::size_t>(num_patches));
    if (actual_patch_out <= expected_patch_out) {
        return;
    }

    const int32_t c_out = actual_patch_out / (pt * ph * pw);
    std::vector<float> truncated(static_cast<std::size_t>(num_patches) *
                                 static_cast<std::size_t>(expected_patch_out));
    for (int32_t pi = 0; pi < num_patches; ++pi) {
        const float* src = denoiser_output.data() + static_cast<std::size_t>(pi) *
                                                        static_cast<std::size_t>(actual_patch_out);
        float* dst = truncated.data() +
                     static_cast<std::size_t>(pi) * static_cast<std::size_t>(expected_patch_out);
        int32_t di = 0;
        for (int32_t pti = 0; pti < pt; ++pti) {
            for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                for (int32_t pwi = 0; pwi < pw; ++pwi) {
                    const int32_t base = ((pti * ph + phi_) * pw + pwi) * c_out;
                    for (int32_t ci = 0; ci < z_dim; ++ci) {
                        dst[di++] = src[base + ci];
                    }
                }
            }
        }
    }
    denoiser_output = std::move(truncated);
}

// ---------------------------------------------------------------------------
// Scheduler step dispatch
// ---------------------------------------------------------------------------

void apply_wan_scheduler_step(bool use_ddim, bool use_unipc, DDIMState& ddim_scheduler,
                              UniPCFlowState& unipc_scheduler, FlowMatchEulerState& fm_scheduler,
                              const std::vector<float>& noise_pred_spatial,
                              std::vector<float>& latents, std::size_t latent_count, int32_t step) {
    if (use_ddim) {
        ddim_scheduler.step(noise_pred_spatial.data(), latents.data(), latents.data(), latent_count,
                            step);
        return;
    }
    if (use_unipc) {
        unipc_scheduler.step(noise_pred_spatial.data(), latents.data(), latents.data(),
                             latent_count, step);
        return;
    }
    fm_scheduler.step(noise_pred_spatial.data(), latents.data(), latents.data(), latent_count,
                      step);
}

// ---------------------------------------------------------------------------
// Step logging
// ---------------------------------------------------------------------------

void maybe_log_wan_step(int32_t step, int32_t num_inference_steps, float timestep,
                        const std::vector<float>& latents) {
    if (step % 5 != 0 && step != num_inference_steps - 1) {
        return;
    }
    double lat_sq = 0.0;
    for (const auto v : latents) {
        lat_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double lat_std = std::sqrt(lat_sq / static_cast<double>(latents.size()));
    std::cerr << "  Step " << (step + 1) << "/" << num_inference_steps << " t=" << timestep
              << " lat_std=" << lat_std << "\n";
}

// ---------------------------------------------------------------------------
// Layout logging
// ---------------------------------------------------------------------------

void log_wan_layout(const WanLayout& layout) {
    std::cerr << "[diffusion] Latent shape: " << layout.z_dim << "x" << layout.t_lat << "x"
              << layout.h_lat << "x" << layout.w_lat << " (patches=" << layout.num_patches << ")\n";
}

// ---------------------------------------------------------------------------
// Denoising loop orchestrator
// ---------------------------------------------------------------------------

template <typename ComputeTembFn, typename PatchifyFn, typename EmbedHiddenFn,
          typename UnpatchifyFn, typename RunDenoiserFn>
bool run_wan_denoising_loop(
    int32_t num_inference_steps, bool use_ddim, bool use_unipc, float guidance_scale,
    const WanLayout& layout, const std::vector<float>& step_timesteps,
    const std::vector<float>& pos_embed_2d, const std::vector<float>& text_projected,
    const std::vector<float>& null_text, const std::vector<float>& encoder_attn_mask,
    DDIMState& ddim_scheduler, UniPCFlowState& unipc_scheduler, FlowMatchEulerState& fm_scheduler,
    std::vector<float>& latents, std::string& error, ComputeTembFn&& compute_temb,
    PatchifyFn&& patchify, EmbedHiddenFn&& embed_hidden, UnpatchifyFn&& unpatchify,
    RunDenoiserFn&& run_denoiser) {
    std::vector<float> patches;
    return wan_denoising::run_wan_video_denoising_steps(
        num_inference_steps, step_timesteps, latents, error, compute_temb,
        [&](const std::vector<float>& current_latents, std::vector<float>& hidden) {
            patchify(current_latents, patches);
            hidden.resize(static_cast<std::size_t>(layout.num_patches) *
                          static_cast<std::size_t>(layout.dim));
            embed_hidden(patches, hidden);
            add_wan_positional_embedding(hidden, pos_embed_2d);
        },
        [&](const std::vector<float>& hidden, const std::vector<float>& temb_6d,
            const std::vector<float>& time_embed, std::vector<float>& denoiser_output,
            std::string& err) {
            return predict_wan_noise(hidden, temb_6d, time_embed, text_projected, null_text,
                                     encoder_attn_mask, guidance_scale, denoiser_output, err,
                                     run_denoiser);
        },
        [&](std::vector<float>& denoiser_output, std::vector<float>& noise_pred_spatial) {
            maybe_truncate_wan_output(denoiser_output, layout.num_patches, layout.z_dim, layout.pt,
                                      layout.ph, layout.pw);
            unpatchify(denoiser_output, noise_pred_spatial);
        },
        [&](const std::vector<float>& noise_pred_spatial, std::vector<float>& current_latents,
            int32_t step) {
            apply_wan_scheduler_step(use_ddim, use_unipc, ddim_scheduler, unipc_scheduler,
                                     fm_scheduler, noise_pred_spatial, current_latents,
                                     current_latents.size(), step);
        },
        [&](int32_t step, float timestep, const std::vector<float>& current_latents) {
            maybe_log_wan_step(step, num_inference_steps, timestep, current_latents);
        });
}

// ---------------------------------------------------------------------------
// Latent denormalization
// ---------------------------------------------------------------------------

void denormalize_wan_latents(const WanDiffusionConfig& config, int32_t z_dim, int32_t t_lat,
                             int32_t h_lat, int32_t w_lat, std::vector<float>& latents) {
    if (config.latents_mean.empty() || config.latents_std.empty()) {
        return;
    }
    const auto channel_size = static_cast<std::size_t>(t_lat * h_lat * w_lat);
    for (int32_t ci = 0; ci < z_dim; ++ci) {
        const float mean = config.latents_mean[static_cast<std::size_t>(ci)];
        const float std_val = config.latents_std[static_cast<std::size_t>(ci)];
        float* ch = latents.data() + static_cast<std::size_t>(ci) * channel_size;
        for (std::size_t i = 0; i < channel_size; ++i) {
            ch[i] = ch[i] * std_val + mean;
        }
    }
}

// ---------------------------------------------------------------------------
// WanVideoResult -> ImageResult conversion
// ---------------------------------------------------------------------------

ImageResult video_to_image(const WanVideoResult& vr, int32_t default_h, int32_t default_w) {
    ImageResult out;
    out.pixels = vr.frames;
    out.height = (vr.height > 0) ? vr.height : default_h;
    out.width = (vr.width > 0) ? vr.width : default_w;
    out.channels = 3;
    out.num_frames = vr.num_frames;
    return out;
}

} // anonymous namespace

// ===========================================================================
// WanPipeline implementation
// ===========================================================================

WanPipeline::WanPipeline(std::unique_ptr<TrtModule> text_encoder,
                         std::unique_ptr<TrtModule> denoiser, std::unique_ptr<TrtModule> vae,
                         WanDiffusionConfig config, WanPreprocessorWeights weights,
                         std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                         std::shared_ptr<void> distributed_owner, int32_t distributed_rank,
                         int32_t distributed_world_size, std::unique_ptr<TrtModule> vae_first_frame)
    : distributed_owner_(std::move(distributed_owner)), distributed_rank_(distributed_rank),
      distributed_world_size_(distributed_world_size), text_encoder_(std::move(text_encoder)),
      denoiser_(std::move(denoiser)), vae_(std::move(vae)),
      vae_first_frame_(std::move(vae_first_frame)), config_(std::move(config)),
      weights_(std::move(weights)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), gpu_matmul_(std::make_unique<WanGpuMatmul>()) {}

WanPipeline::~WanPipeline() = default;

void WanPipeline::matmul_bias(const float* lhs, const float* rhs, const float* bias, float* output,
                              int32_t rows, int32_t inner, int32_t columns) const {
    if (wan_should_use_gpu_matmul(rows, inner, columns) && gpu_matmul_ != nullptr &&
        gpu_matmul_->run(lhs, rhs, bias, output, rows, inner, columns)) {
        return;
    }
    cpu_matmul_bias(lhs, rhs, bias, output, rows, inner, columns);
}

// ---------------------------------------------------------------------------
// T5 encoder via TrtModule::forward()
// ---------------------------------------------------------------------------

bool WanPipeline::run_t5_encoder(const std::vector<int32_t>& input_ids,
                                 std::vector<float>& text_embeddings) {
    if (!text_encoder_ || !text_encoder_->ok()) {
        return false;
    }

    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    // Pad/truncate input_ids to seq_len
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

    // Build TensorMap inputs
    TensorMap inputs;
    inputs["input_ids"] = Tensor{padded_ids.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{mask.data(), {static_cast<int64_t>(seq_len)}, DType::kFloat32};

    // Forward through T5 encoder
    TensorMap outputs = text_encoder_->forward(inputs);

    // Copy output embeddings
    const auto emb_size = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim);
    text_embeddings.resize(emb_size);
    auto* emb_data = static_cast<float*>(outputs["text_embeddings"].data);
    std::copy_n(emb_data, emb_size, text_embeddings.data());

    // Zero out padding positions
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] == 0) {
            float* row = text_embeddings.data() +
                         static_cast<std::size_t>(i) * static_cast<std::size_t>(te_dim);
            std::fill_n(row, static_cast<std::size_t>(te_dim), 0.0F);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// DiT denoiser via TrtModule::forward()
// ---------------------------------------------------------------------------

bool WanPipeline::run_denoiser(const std::vector<float>& hidden, const std::vector<float>& temb_6d,
                               const std::vector<float>& time_embed,
                               const std::vector<float>& encoder_hidden,
                               const std::vector<float>& cos_vals,
                               const std::vector<float>& sin_vals, std::vector<float>& output,
                               const std::vector<float>& encoder_attn_mask) {
    if (!denoiser_ || !denoiser_->ok()) {
        return false;
    }

    const int32_t dit_dim = config_.dit_dim;
    const int32_t num_patches =
        static_cast<int32_t>(hidden.size() / static_cast<std::size_t>(dit_dim));

    TensorMap inputs;
    inputs["hidden_states"] =
        Tensor{const_cast<float*>(hidden.data()),
               {static_cast<int64_t>(num_patches), static_cast<int64_t>(dit_dim)},
               DType::kFloat32};
    inputs["timestep_embedding"] = Tensor{
        const_cast<float*>(temb_6d.data()), {6, static_cast<int64_t>(dit_dim)}, DType::kFloat32};
    inputs["time_embed"] = Tensor{
        const_cast<float*>(time_embed.data()), {static_cast<int64_t>(dit_dim)}, DType::kFloat32};
    inputs["encoder_hidden_states"] =
        Tensor{const_cast<float*>(encoder_hidden.data()),
               {static_cast<int64_t>(encoder_hidden.size() / static_cast<std::size_t>(dit_dim)),
                static_cast<int64_t>(dit_dim)},
               DType::kFloat32};

    if (config_.use_rope && !cos_vals.empty()) {
        const int32_t head_dim = dit_dim / std::max(config_.dit_num_heads, 1);
        inputs["rotary_cos"] =
            Tensor{const_cast<float*>(cos_vals.data()),
                   {static_cast<int64_t>(num_patches), static_cast<int64_t>(head_dim)},
                   DType::kFloat32};
        inputs["rotary_sin"] =
            Tensor{const_cast<float*>(sin_vals.data()),
                   {static_cast<int64_t>(num_patches), static_cast<int64_t>(head_dim)},
                   DType::kFloat32};
    }

    if (!encoder_attn_mask.empty() && !config_.use_rope) {
        inputs["encoder_attention_mask"] = Tensor{const_cast<float*>(encoder_attn_mask.data()),
                                                  {static_cast<int64_t>(encoder_attn_mask.size())},
                                                  DType::kFloat32};
    }

    TensorMap outputs = denoiser_->forward(inputs);

    auto* out_data = static_cast<float*>(outputs["output"].data);
    const auto out_numel = outputs["output"].numel();
    output.resize(out_numel);
    std::copy_n(out_data, out_numel, output.data());

    return true;
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

void WanPipeline::compute_timestep_embedding(float timestep, std::vector<float>& temb_6d,
                                             std::vector<float>& time_embed) const {
    const int32_t dim = config_.dit_dim;
    const int32_t freq_dim = config_.freq_dim;
    const int32_t half = freq_dim / 2;

    std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
    for (int32_t i = 0; i < half; ++i) {
        const double freq =
            std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
        const double angle = static_cast<double>(timestep) * freq;
        sinusoidal[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(angle));
        sinusoidal[static_cast<std::size_t>(i + half)] = static_cast<float>(std::sin(angle));
    }

    std::vector<float> hidden_1(static_cast<std::size_t>(dim));
    matmul_bias(sinusoidal.data(), weights_.time_emb_0_weight.data(),
                weights_.time_emb_0_bias.data(), hidden_1.data(), 1, freq_dim, dim);
    cpu_silu_inplace(hidden_1.data(), static_cast<std::size_t>(dim));

    time_embed.resize(static_cast<std::size_t>(dim));
    matmul_bias(hidden_1.data(), weights_.time_emb_2_weight.data(), weights_.time_emb_2_bias.data(),
                time_embed.data(), 1, dim, dim);

    std::vector<float> silu_te(time_embed.begin(), time_embed.end());
    cpu_silu_inplace(silu_te.data(), static_cast<std::size_t>(dim));

    temb_6d.resize(static_cast<std::size_t>(6 * dim));
    matmul_bias(silu_te.data(), weights_.time_proj_weight.data(), weights_.time_proj_bias.data(),
                temb_6d.data(), 1, dim, 6 * dim);
}

// ---------------------------------------------------------------------------
// Text projection
// ---------------------------------------------------------------------------

void WanPipeline::project_text(const std::vector<float>& in, int32_t seq_len,
                               std::vector<float>& out) const {
    const int32_t te_dim = config_.text_encoder_dim;
    const int32_t dim = config_.dit_dim;

    out.resize(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
    matmul_bias(in.data(), weights_.text_proj_weight.data(), weights_.text_proj_bias.data(),
                out.data(), seq_len, te_dim, dim);

    if (!weights_.text_proj_2_weight.empty()) {
        cpu_gelu_tanh_inplace(out.data(),
                              static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(dim));
        std::vector<float> tmp(out.size());
        matmul_bias(out.data(), weights_.text_proj_2_weight.data(),
                    weights_.text_proj_2_bias.data(), tmp.data(), seq_len, dim, dim);
        out = std::move(tmp);
    }
}

// ---------------------------------------------------------------------------
// Patchify (CPU — identical to old backend)
// ---------------------------------------------------------------------------

void WanPipeline::patchify(const std::vector<float>& latents, int32_t c, int32_t t, int32_t h,
                           int32_t w, std::vector<float>& patches) const {
    int32_t pt = 1, ph = 2, pw = 2;
    if (config_.patch_size.size() >= 3) {
        pt = config_.patch_size[0];
        ph = config_.patch_size[1];
        pw = config_.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;
    const int32_t num_patches = nt * nh * nw;

    patches.resize(static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(patch_dim));

    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t ci = 0; ci < c; ++ci) {
                    for (int32_t pti = 0; pti < pt; ++pti) {
                        for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                            for (int32_t pwi = 0; pwi < pw; ++pwi) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto src_idx =
                                    static_cast<std::size_t>(ci) *
                                        static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                patches[static_cast<std::size_t>(patch_idx) *
                                            static_cast<std::size_t>(patch_dim) +
                                        static_cast<std::size_t>(elem)] = latents[src_idx];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unpatchify (CPU — identical to old backend)
// ---------------------------------------------------------------------------

void WanPipeline::unpatchify(const std::vector<float>& patches, int32_t c, int32_t t, int32_t h,
                             int32_t w, std::vector<float>& output) const {
    int32_t pt = 1, ph = 2, pw = 2;
    if (config_.patch_size.size() >= 3) {
        pt = config_.patch_size[0];
        ph = config_.patch_size[1];
        pw = config_.patch_size[2];
    }
    const int32_t nt = t / pt, nh = h / ph, nw = w / pw;
    const int32_t patch_dim = c * pt * ph * pw;

    output.resize(static_cast<std::size_t>(c * t * h * w));

    int32_t patch_idx = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                int32_t elem = 0;
                for (int32_t pti = 0; pti < pt; ++pti) {
                    for (int32_t phi_ = 0; phi_ < ph; ++phi_) {
                        for (int32_t pwi = 0; pwi < pw; ++pwi) {
                            for (int32_t ci = 0; ci < c; ++ci) {
                                const int32_t tt = ti * pt + pti;
                                const int32_t hh = hi * ph + phi_;
                                const int32_t ww = wi * pw + pwi;
                                const auto dst_idx =
                                    static_cast<std::size_t>(ci) *
                                        static_cast<std::size_t>(t * h * w) +
                                    static_cast<std::size_t>(tt) * static_cast<std::size_t>(h * w) +
                                    static_cast<std::size_t>(hh) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(ww);
                                output[dst_idx] = patches[static_cast<std::size_t>(patch_idx) *
                                                              static_cast<std::size_t>(patch_dim) +
                                                          static_cast<std::size_t>(elem)];
                                ++elem;
                            }
                        }
                    }
                }
                ++patch_idx;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3D RoPE (CPU — identical to old backend)
// ---------------------------------------------------------------------------

void WanPipeline::compute_3d_rope(int32_t nt, int32_t nh, int32_t nw, std::vector<float>& cos_out,
                                  std::vector<float>& sin_out) const {
    const int32_t dim = config_.dit_dim;
    const int32_t num_heads = config_.dit_num_heads;
    const int32_t head_dim = dim / std::max(num_heads, 1);
    const int32_t num_patches = nt * nh * nw;
    const double theta = 10000.0;

    // Wan uses: h_dim = w_dim = 2*(head_dim//6), t_dim = head_dim - h_dim - w_dim
    const int32_t h_dim = 2 * (head_dim / 6);
    const int32_t w_dim = h_dim;
    const int32_t t_dim = head_dim - h_dim - w_dim;

    auto get_1d_rope = [&](int32_t rdim, int32_t max_len,
                           std::vector<std::vector<float>>& cos_table,
                           std::vector<std::vector<float>>& sin_table) {
        const int32_t half_r = rdim / 2;
        cos_table.resize(static_cast<std::size_t>(max_len));
        sin_table.resize(static_cast<std::size_t>(max_len));

        for (int32_t pos = 0; pos < max_len; ++pos) {
            auto& c = cos_table[static_cast<std::size_t>(pos)];
            auto& s = sin_table[static_cast<std::size_t>(pos)];
            c.resize(static_cast<std::size_t>(rdim));
            s.resize(static_cast<std::size_t>(rdim));

            for (int32_t i = 0; i < half_r; ++i) {
                const double freq =
                    1.0 / std::pow(theta, static_cast<double>(i) / static_cast<double>(half_r));
                const double angle = static_cast<double>(pos) * freq;
                const auto cv = static_cast<float>(std::cos(angle));
                const auto sv = static_cast<float>(std::sin(angle));
                c[static_cast<std::size_t>(2 * i)] = cv;
                c[static_cast<std::size_t>(2 * i + 1)] = cv;
                s[static_cast<std::size_t>(2 * i)] = sv;
                s[static_cast<std::size_t>(2 * i + 1)] = sv;
            }
        }
    };

    std::vector<std::vector<float>> t_cos, t_sin, h_cos, h_sin, w_cos, w_sin;
    get_1d_rope(t_dim, std::max(nt, 1024), t_cos, t_sin);
    get_1d_rope(h_dim, std::max(nh, 1024), h_cos, h_sin);
    get_1d_rope(w_dim, std::max(nw, 1024), w_cos, w_sin);

    cos_out.resize(static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(head_dim));
    sin_out.resize(static_cast<std::size_t>(num_patches) * static_cast<std::size_t>(head_dim));

    int32_t p = 0;
    for (int32_t ti = 0; ti < nt; ++ti) {
        for (int32_t hi = 0; hi < nh; ++hi) {
            for (int32_t wi = 0; wi < nw; ++wi) {
                float* c_row = cos_out.data() +
                               static_cast<std::size_t>(p) * static_cast<std::size_t>(head_dim);
                float* s_row = sin_out.data() +
                               static_cast<std::size_t>(p) * static_cast<std::size_t>(head_dim);

                int32_t off = 0;
                std::memcpy(c_row + off, t_cos[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                std::memcpy(s_row + off, t_sin[static_cast<std::size_t>(ti)].data(),
                            static_cast<std::size_t>(t_dim) * sizeof(float));
                off += t_dim;

                std::memcpy(c_row + off, h_cos[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                std::memcpy(s_row + off, h_sin[static_cast<std::size_t>(hi)].data(),
                            static_cast<std::size_t>(h_dim) * sizeof(float));
                off += h_dim;

                std::memcpy(c_row + off, w_cos[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));
                std::memcpy(s_row + off, w_sin[static_cast<std::size_t>(wi)].data(),
                            static_cast<std::size_t>(w_dim) * sizeof(float));

                ++p;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VAE 2D decode via TrtModule::forward()
// ---------------------------------------------------------------------------

bool WanPipeline::decode_vae_2d(const std::vector<float>& latents, int32_t c, int32_t h_lat,
                                int32_t w_lat, WanVideoResult& result) {
    if (!vae_ || !vae_->ok()) {
        return false;
    }

    const int32_t h_out = h_lat * config_.scale_factor_spatial;
    const int32_t w_out = w_lat * config_.scale_factor_spatial;
    const auto input_size = static_cast<std::size_t>(c) * static_cast<std::size_t>(h_lat) *
                            static_cast<std::size_t>(w_lat);
    auto scaled_latents = prepare_wan_vae_2d_input(latents, config_, input_size);

    // Forward through VAE: latent_input -> decoder_output
    TensorMap inputs;
    inputs["latent_input"] = Tensor{
        scaled_latents.data(),
        {1, static_cast<int64_t>(c), static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)},
        DType::kFloat32};

    TensorMap outputs = vae_->forward(inputs);

    const auto out_size = static_cast<std::size_t>(3) * static_cast<std::size_t>(h_out) *
                          static_cast<std::size_t>(w_out);
    auto* raw_data = static_cast<float*>(outputs["decoder_output"].data);
    std::vector<float> raw(raw_data, raw_data + out_size);

    convert_wan_chw_to_hwc(raw, h_out, w_out, result);
    return true;
}

// ---------------------------------------------------------------------------
// VAE 3D helpers (extracted from decode_vae_3d for cyclomatic complexity)
// ---------------------------------------------------------------------------

int32_t WanPipeline::query_vae_output_temporal_dim(const TrtModule& module) {
    auto out_infos = module.output_info();
    for (const auto& info : out_infos) {
        if (info.name == "video_frame" && info.shape.size() >= 3) {
            return std::max(static_cast<int32_t>(info.shape[2]), 1);
        }
    }
    return 1;
}

bool WanPipeline::has_first_frame_vae() const {
    return vae_first_frame_ && vae_first_frame_->ok();
}

void WanPipeline::init_vae_caches() {
    const int32_t num_caches = config_.num_vae_caches;
    auto out_infos = vae_->output_info();

    // We need a stream for DeviceTensor operations — get it from the VAE module
    // by querying the device pointer (which implicitly gives us a usable device).
    // DeviceTensor needs a cudaStream_t; we create a dedicated one.
    cudaStream_t cache_stream = nullptr;
    cudaStreamCreate(&cache_stream);

    for (int32_t ci = 0; ci < num_caches; ++ci) {
        const std::string cache_out_name = "cache_out_" + std::to_string(ci);
        const std::string cache_in_name = "cache_" + std::to_string(ci);

        // Find cache shape from output_info
        std::vector<int64_t> cache_shape;
        DType cache_dtype = DType::kFloat32;
        for (const auto& info : out_infos) {
            if (info.name == cache_out_name) {
                cache_shape = info.shape;
                cache_dtype = info.dtype;
                break;
            }
        }

        if (cache_shape.empty()) {
            // Cache not found — skip
            continue;
        }

        // Create DeviceTensor pairs for cache_in and cache_out
        vae_cache_in_.emplace_back(cache_shape, cache_dtype, cache_stream);
        vae_cache_out_.emplace_back(cache_shape, cache_dtype, cache_stream);

        // Bind external device memory to VAE module's cache tensors
        vae_->bind_external(cache_in_name, vae_cache_in_.back().data());
        vae_->bind_external(cache_out_name, vae_cache_out_.back().data());
        if (vae_first_frame_ && vae_first_frame_->ok()) {
            vae_first_frame_->bind_external(cache_in_name, vae_cache_in_.back().data());
            vae_first_frame_->bind_external(cache_out_name, vae_cache_out_.back().data());
        }
    }

    vae_caches_initialized_ = true;

    std::cerr << "[diffusion] VAE caches initialized: " << vae_cache_in_.size() << " cache pairs\n";

    // Clean up the stream (DeviceTensors store a copy)
    // Note: DeviceTensor operations are async on their stored stream.
    // We keep the stream alive since DeviceTensors reference it.
    // Actually, DeviceTensors keep a copy of the stream handle, so we
    // must NOT destroy it. Store it as a shared_ptr via keep_alive.
    auto stream_deleter = [](void* s) { cudaStreamDestroy(static_cast<cudaStream_t>(s)); };
    vae_->keep_alive(std::shared_ptr<void>(cache_stream, stream_deleter));
}

void WanPipeline::zero_vae_caches() {
    const auto num_cache_pairs = static_cast<int32_t>(vae_cache_in_.size());
    for (int32_t ci = 0; ci < num_cache_pairs; ++ci) {
        auto& cache_in = vae_cache_in_[static_cast<std::size_t>(ci)];
        if (cache_in.ok()) {
            cudaMemsetAsync(cache_in.data(), 0, cache_in.nbytes(), cache_in.stream());
        }
    }
}

void WanPipeline::decode_vae_single_frame(const std::vector<float>& latents, int32_t c,
                                          int32_t t_lat, int32_t h_lat, int32_t w_lat, int32_t t,
                                          std::size_t out_frame_floats, TrtModule& module,
                                          std::vector<float>& all_raw_frames) {
    // Extract single latent frame [c, h, w] from [c, t, h, w]
    std::vector<float> frame_buf;
    extract_wan_latent_frame(latents, c, t_lat, h_lat, w_lat, t, frame_buf);

    // Forward through VAE with latent_frame input
    // Cache tensors are already bound via bind_external
    TensorMap inputs;
    inputs["latent_frame"] = Tensor{
        frame_buf.data(),
        {1, static_cast<int64_t>(c), static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)},
        DType::kFloat32};

    TensorMap outputs = module.forward(inputs);

    // Copy output frame to host
    auto* frame_data = static_cast<float*>(outputs["video_frame"].data);
    std::vector<float> out_buf(out_frame_floats);
    std::copy_n(frame_data, out_frame_floats, out_buf.data());
    all_raw_frames.insert(all_raw_frames.end(), out_buf.begin(), out_buf.end());

    // D2D copy: cache_out -> cache_in for next frame
    const auto num_cache_pairs = static_cast<int32_t>(vae_cache_in_.size());
    for (int32_t ci = 0; ci < num_cache_pairs; ++ci) {
        auto& cache_in = vae_cache_in_[static_cast<std::size_t>(ci)];
        auto& cache_out = vae_cache_out_[static_cast<std::size_t>(ci)];
        if (cache_in.ok() && cache_out.ok()) {
            cache_in.copy_from(cache_out);
        }
    }

    if (t % 2 == 0) {
        std::cerr << "  VAE frame " << (t + 1) << "/" << t_lat << "\n";
    }
}

// ---------------------------------------------------------------------------
// VAE 3D decode via TrtModule + DeviceTensor cache swap
// ---------------------------------------------------------------------------

bool WanPipeline::decode_vae_3d(const std::vector<float>& latents, int32_t c, int32_t t_lat,
                                int32_t h_lat, int32_t w_lat, WanVideoResult& result) {
    if (!vae_ || !vae_->ok()) {
        return false;
    }

    const int32_t h_out = config_.video_height;
    const int32_t w_out = config_.video_width;
    const int32_t vae_output_t = query_vae_output_temporal_dim(*vae_);
    const bool has_first_frame_engine = has_first_frame_vae();
    const int32_t first_vae_output_t =
        has_first_frame_engine ? query_vae_output_temporal_dim(*vae_first_frame_) : vae_output_t;

    const auto output_spatial = static_cast<std::size_t>(3) * static_cast<std::size_t>(h_out) *
                                static_cast<std::size_t>(w_out);

    // Initialize VAE caches on first call
    if (!vae_caches_initialized_ && config_.num_vae_caches > 0) {
        init_vae_caches();
    }

    // Zero-initialize cache_in tensors for the first frame
    zero_vae_caches();

    // Decode each latent frame
    std::vector<float> all_raw_frames;
    all_raw_frames.reserve((static_cast<std::size_t>(first_vae_output_t) +
                            static_cast<std::size_t>(std::max(t_lat - 1, 0) * vae_output_t)) *
                           output_spatial);

    for (int32_t t = 0; t < t_lat; ++t) {
        const bool use_first_frame_engine = (t == 0 && has_first_frame_engine);
        TrtModule& module = use_first_frame_engine ? *vae_first_frame_ : *vae_;
        const int32_t chunk_t = use_first_frame_engine ? first_vae_output_t : vae_output_t;
        decode_vae_single_frame(latents, c, t_lat, h_lat, w_lat, t,
                                static_cast<std::size_t>(chunk_t) * output_spatial, module,
                                all_raw_frames);
    }

    compose_wan_vae_video_frames(all_raw_frames, t_lat, vae_output_t, first_vae_output_t, h_out,
                                 w_out, config_.scale_factor_temporal, config_.video_num_frames,
                                 result);
    return true;
}

// ---------------------------------------------------------------------------
// generate_image helpers (extracted for cyclomatic complexity)
// ---------------------------------------------------------------------------

bool WanPipeline::run_wan_text_conditioning(const std::vector<int32_t>& input_ids, int32_t seq_len,
                                            std::vector<float>& text_projected,
                                            std::vector<float>& null_text, std::string& error) {
    // Build conditioning inputs (null IDs needed for null-text encoding)
    const diffusion::WanConditioningInputs conditioning_inputs =
        diffusion::make_wan_conditioning_inputs(config_, diffusion::make_wan_layout(config_),
                                                input_ids);

    std::cerr << "[diffusion] Encoding text (" << input_ids.size() << " tokens) ...\n";

    diffusion::WanTextConditioning text_conditioning;
    if (!diffusion::build_wan_text_conditioning(
            input_ids, conditioning_inputs, seq_len, error,
            [this](const std::vector<int32_t>& ids, std::vector<float>& embeddings,
                   std::string& /*encoder_error*/) { return run_t5_encoder(ids, embeddings); },
            [this](const std::vector<float>& embeddings, int32_t sl,
                   std::vector<float>& projected) { project_text(embeddings, sl, projected); },
            text_conditioning)) {
        return false;
    }

    text_projected = std::move(text_conditioning.text_projected);
    null_text = std::move(text_conditioning.null_text);

    std::cerr << "[diffusion] T5 conditioning done (" << text_projected.size() << " floats)\n";
    return true;
}

bool WanPipeline::run_wan_vae_decode(int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
                                     std::vector<float>& latents, WanVideoResult& result) {
    std::cerr << "[diffusion] Decoding video ...\n";
    if (config_.num_vae_caches <= 0) {
        if (!decode_vae_2d(latents, z_dim, h_lat, w_lat, result)) {
            std::cerr << "[diffusion] VAE 2D decode failed\n";
            return false;
        }
    } else {
        if (!decode_vae_3d(latents, z_dim, t_lat, h_lat, w_lat, result)) {
            std::cerr << "[diffusion] VAE 3D decode failed\n";
            return false;
        }
    }
    return true;
}

ImageResult WanPipeline::finish_wan_generation(int32_t z_dim, int32_t t_lat, int32_t h_lat,
                                               int32_t w_lat, std::vector<float>& latents,
                                               WanVideoResult& result) {
    denormalize_wan_latents(config_, z_dim, t_lat, h_lat, w_lat, latents);

    if (distributed_world_size_ > 1 && distributed_rank_ != 0) {
        std::cerr << "[wan-t2v] Distributed rank " << distributed_rank_
                  << " skips VAE decode; rank 0 writes video artifacts\n";
        ImageResult empty;
        empty.num_frames = 0;
        return empty;
    }

    if (!run_wan_vae_decode(z_dim, t_lat, h_lat, w_lat, latents, result)) {
        return video_to_image(result, config_.video_height, config_.video_width);
    }

    result.num_frames = config_.video_num_frames;
    std::cerr << "[diffusion] Video generation complete: " << result.num_frames << " frames, "
              << result.height << "x" << result.width << "\n";

    return video_to_image(result, config_.video_height, config_.video_width);
}

// ---------------------------------------------------------------------------
// Main generation pipeline
// ---------------------------------------------------------------------------

std::vector<int32_t> WanPipeline::tokenize_wan_prompt(const std::string& prompt) const {
    if (!tokenizer_) {
        return {};
    }
    return diffusion::normalize_wan_t5_token_ids(tokenizer_->encode(prompt), config_.text_seq_len,
                                                 config_.tokenizer_add_special_tokens);
}

ImageResult WanPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    // Tokenize prompt
    const std::vector<int32_t> input_ids = tokenize_wan_prompt(prompt);

    const int32_t requested_steps = (cfg.num_steps > 0) ? cfg.num_steps : -1;
    const float requested_guidance = (cfg.guidance_scale >= 0.0f) ? cfg.guidance_scale : -1.0f;

    const auto plan =
        diffusion::make_wan_generation_plan(config_, requested_steps, requested_guidance);

    WanVideoResult result;
    result.height = config_.video_height;
    result.width = config_.video_width;

    if (!weights_.valid) {
        std::cerr << "[diffusion] ERROR: preprocessor weights not loaded\n";
        return video_to_image(result, config_.video_height, config_.video_width);
    }

    const WanLayout& layout = plan.layout;
    log_wan_layout(layout);

    // Encode text: run T5 for both real and null prompts, project both
    std::string error;
    std::vector<float> text_projected, null_text;
    if (!run_wan_text_conditioning(input_ids, layout.seq_len, text_projected, null_text, error)) {
        std::cerr << "[diffusion] T5 conditioning failed: " << error << "\n";
        return video_to_image(result, config_.video_height, config_.video_width);
    }

    // Build conditioning inputs for denoising (need encoder_attn_mask)
    const diffusion::WanConditioningInputs conditioning_inputs =
        diffusion::make_wan_conditioning_inputs(config_, layout, input_ids);

    // Compute positional embeddings (RoPE 3D or 2D)
    std::vector<float> rope_cos, rope_sin, pos_embed_2d;
    if (config_.use_rope) {
        compute_3d_rope(layout.nt, layout.nh_p, layout.nw_p, rope_cos, rope_sin);
    } else {
        compute_wan_pos_embed_2d(layout.nh_p, layout.nw_p, layout.dim, pos_embed_2d);
    }

    // Initialize from the caller-provided parity tensor, or honor the requested seed.
    std::vector<float> latents;
    if (!diffusion::resolve_wan_initial_latents(plan.latent_count, cfg.initial_latents, cfg.seed,
                                                latents, error)) {
        std::cerr << "[diffusion] Invalid initial latents: " << error << "\n";
        return video_to_image(result, config_.video_height, config_.video_width);
    }

    // Set up scheduler
    FlowMatchEulerState fm_scheduler;
    UniPCFlowState unipc_scheduler;
    DDIMState ddim_scheduler;
    std::vector<float> step_timesteps;
    if (plan.use_ddim) {
        ddim_scheduler.num_train_timesteps = 1000;
        ddim_scheduler.set_timesteps(plan.num_inference_steps);
        step_timesteps = ddim_scheduler.timesteps;
    } else if (plan.use_unipc) {
        unipc_scheduler = diffusion::make_wan_unipc_scheduler(config_, plan);
        step_timesteps = unipc_scheduler.timesteps;
    } else {
        fm_scheduler = diffusion::make_wan_flow_match_scheduler(plan);
        step_timesteps = fm_scheduler.timesteps;
    }

    // Build lambda closures for the denoising loop
    const auto compute_temb = [this](float timestep, std::vector<float>& temb_6d,
                                     std::vector<float>& time_embed) {
        compute_timestep_embedding(timestep, temb_6d, time_embed);
    };
    const auto patchify_fn = [this, &layout](const std::vector<float>& src_latents,
                                             std::vector<float>& patches) {
        patchify(src_latents, layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, patches);
    };
    const auto embed_hidden = [this, &layout](const std::vector<float>& patches,
                                              std::vector<float>& hidden) {
        matmul_bias(patches.data(), weights_.patch_embed_weight.data(),
                    weights_.patch_embed_bias.data(), hidden.data(), layout.num_patches,
                    layout.patch_dim, layout.dim);
    };
    const auto unpatchify_fn = [this, &layout](const std::vector<float>& patches,
                                               std::vector<float>& out) {
        unpatchify(patches, layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, out);
    };
    const auto run_denoiser_fn =
        [this, &rope_cos,
         &rope_sin](const std::vector<float>& hidden, const std::vector<float>& temb_6d,
                    const std::vector<float>& time_embed, const std::vector<float>& encoder_hidden,
                    const std::vector<float>& encoder_mask, std::vector<float>& output,
                    std::string& /*err*/) {
            return this->run_denoiser(hidden, temb_6d, time_embed, encoder_hidden, rope_cos,
                                      rope_sin, output, encoder_mask);
        };

    // Run denoising loop
    if (!run_wan_denoising_loop(plan.num_inference_steps, plan.use_ddim, plan.use_unipc,
                                plan.guidance_scale, layout, step_timesteps, pos_embed_2d,
                                text_projected, null_text, conditioning_inputs.encoder_attn_mask,
                                ddim_scheduler, unipc_scheduler, fm_scheduler, latents, error,
                                compute_temb, patchify_fn, embed_hidden, unpatchify_fn,
                                run_denoiser_fn)) {
        std::cerr << "[diffusion] Denoiser failed: " << error << "\n";
        return video_to_image(result, config_.video_height, config_.video_width);
    }

    return finish_wan_generation(layout.z_dim, layout.t_lat, layout.h_lat, layout.w_lat, latents,
                                 result);
}

} // namespace trtmc

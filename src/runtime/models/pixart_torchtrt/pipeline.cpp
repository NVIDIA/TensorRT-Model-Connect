// PixArtTorchTrtPipeline — simplified diffusion for torch-trt PixArt engines.
//
// The torch-trt engines include all preprocessing internally (patch embedding,
// timestep embedding, caption projection), so the pipeline is simpler than
// PixArtPipeline: no CPU patchify/unpatchify, no preprocessor weights.
//
// Flow: tokenize -> T5 encoder -> denoising loop -> VAE decode -> image

#include "runtime/models/pixart_torchtrt/pipeline.h"

#include "runtime/domains/diffusion/diffusion_generation_plan.h"
#include "runtime/domains/diffusion/wan_generation_conditioning.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace trtmc {

// ─── fp32 ↔ fp16 conversion helpers ─────────────────────────────────────
// TRT engines compiled with use_explicit_typing=True keep their original
// dtypes (fp16 for diffusion). We must convert our fp32 pipeline data to
// fp16 before feeding engines and back after receiving output.

namespace {

using half_t = uint16_t;

inline half_t fp32_to_fp16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0)
        return static_cast<half_t>(sign); // flush to zero
    if (exp >= 31)
        return static_cast<half_t>(sign | 0x7C00); // inf
    return static_cast<half_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float fp16_to_fp32(half_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        float f;
        uint32_t bits = sign;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    uint32_t bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline half_t fp32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<half_t>(bits >> 16U);
}

inline float bf16_to_fp32(half_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<half_t> convert_float_to_16(const std::vector<float>& src, DType dtype) {
    std::vector<half_t> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = (dtype == DType::kBFloat16) ? fp32_to_bf16(src[i]) : fp32_to_fp16(src[i]);
    }
    return dst;
}

Tensor make_float_tensor(const std::vector<float>& values, const std::vector<int64_t>& shape) {
    return Tensor{const_cast<float*>(values.data()), shape, DType::kFloat32};
}

Tensor make_model_tensor(const std::vector<float>& values, std::vector<half_t>& scratch16,
                         DType dtype, const std::vector<int64_t>& shape) {
    if (dtype == DType::kFloat32) {
        return make_float_tensor(values, shape);
    }
    scratch16 = convert_float_to_16(values, dtype);
    return Tensor{scratch16.data(), shape, dtype};
}

DType input_dtype_or(const TrtModule* module, const std::string& name, DType fallback) {
    if (!module) {
        return fallback;
    }
    for (const auto& info : module->input_info()) {
        if (info.name == name) {
            return info.dtype;
        }
    }
    return fallback;
}

bool tensor_to_float_vector(const Tensor& tensor, std::size_t count, std::vector<float>& dst,
                            const char* label) {
    if (!tensor.data) {
        std::cerr << "[torchtrt_diffusion] " << label << ": empty output tensor\n";
        return false;
    }
    if (tensor.numel() < count) {
        std::cerr << "[torchtrt_diffusion] " << label << ": output too small (got "
                  << tensor.numel() << ", need " << count << ")\n";
        return false;
    }

    dst.assign(count, 0.0F);
    if (tensor.dtype == DType::kFloat32) {
        const auto* src = static_cast<const float*>(tensor.data);
        std::copy_n(src, count, dst.data());
        return true;
    }
    if (tensor.dtype == DType::kFloat16) {
        const auto* src = static_cast<const half_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = fp16_to_fp32(src[i]);
        }
        return true;
    }
    if (tensor.dtype == DType::kBFloat16) {
        const auto* src = static_cast<const half_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = bf16_to_fp32(src[i]);
        }
        return true;
    }

    std::cerr << "[torchtrt_diffusion] " << label << ": unsupported output dtype\n";
    return false;
}

} // anonymous namespace

// ─── DPM-Solver++ Multistep Scheduler ────────────────────────────────────────
// Matches HF diffusers DPMSolverMultistepScheduler:
//   algorithm_type = "dpmsolver++"
//   solver_order = 2
//   solver_type = "midpoint"
//   prediction_type = "epsilon"
//   beta_schedule = "linear"
//   lower_order_final = true
//   timestep_spacing = "linspace"
//
// Variable naming follows HF convention:
//   s0 = current timestep (where we have the sample)
//   s1 = previous model-output timestep (one iteration ago)
//   t  = target timestep (where we want to go)

namespace {

double rms(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (const auto v : values) {
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

// Convert epsilon prediction to x0 prediction.
void eps_to_x0(const DpmSolverState& s, const float* eps, const float* x_s, float* x0,
               std::size_t count, int32_t t) {
    const auto ti = static_cast<std::size_t>(std::clamp(t, 0, s.num_train_timesteps - 1));
    const double a = s.alpha_t[ti];
    const double sig = s.sigma_t[ti];
    for (std::size_t i = 0; i < count; ++i) {
        x0[i] = static_cast<float>(
            (static_cast<double>(x_s[i]) - sig * static_cast<double>(eps[i])) / a);
    }
}

void first_order_update(const DpmSolverState& s, const float* m0, const float* sample, float* x_out,
                        std::size_t count, int32_t t_s0, int32_t t_t) {
    const auto i_s0 = static_cast<std::size_t>(std::clamp(t_s0, 0, s.num_train_timesteps - 1));
    const auto i_t = static_cast<std::size_t>(std::clamp(t_t, 0, s.num_train_timesteps - 1));
    const double h = s.lambda_t[i_t] - s.lambda_t[i_s0];
    for (std::size_t i = 0; i < count; ++i) {
        x_out[i] =
            static_cast<float>((s.sigma_t[i_t] / s.sigma_t[i_s0]) * static_cast<double>(sample[i]) -
                               s.alpha_t[i_t] * (std::exp(-h) - 1.0) * static_cast<double>(m0[i]));
    }
}

void second_order_update(const DpmSolverState& s, const float* m0, const float* m1,
                         const float* sample, float* x_out, std::size_t count, int32_t t_s0,
                         int32_t t_s1, int32_t t_t) {
    const auto i_s0 = static_cast<std::size_t>(std::clamp(t_s0, 0, s.num_train_timesteps - 1));
    const auto i_s1 = static_cast<std::size_t>(std::clamp(t_s1, 0, s.num_train_timesteps - 1));
    const auto i_t = static_cast<std::size_t>(std::clamp(t_t, 0, s.num_train_timesteps - 1));
    const double h = s.lambda_t[i_t] - s.lambda_t[i_s0];
    const double h_0 = s.lambda_t[i_s0] - s.lambda_t[i_s1];
    const double r0 = h_0 / h;
    const double exp_neg_h = std::exp(-h);
    const double base_coeff = s.alpha_t[i_t] * (exp_neg_h - 1.0);
    for (std::size_t i = 0; i < count; ++i) {
        const double d0 = static_cast<double>(m0[i]);
        const double d1 = (1.0 / r0) * (static_cast<double>(m0[i]) - static_cast<double>(m1[i]));
        x_out[i] =
            static_cast<float>((s.sigma_t[i_t] / s.sigma_t[i_s0]) * static_cast<double>(sample[i]) -
                               base_coeff * d0 - 0.5 * base_coeff * d1);
    }
}

} // anonymous namespace

void DpmSolverState::set_timesteps(int32_t num_steps, double beta_start, double beta_end) {
    const int32_t T = num_train_timesteps;
    alpha_t.resize(static_cast<std::size_t>(T));
    sigma_t.resize(static_cast<std::size_t>(T));
    lambda_t.resize(static_cast<std::size_t>(T));

    double cum = 1.0;
    for (int32_t i = 0; i < T; ++i) {
        double beta = beta_start +
                      static_cast<double>(i) / static_cast<double>(T - 1) * (beta_end - beta_start);
        cum *= (1.0 - beta);
        const auto si = static_cast<std::size_t>(i);
        alpha_t[si] = std::sqrt(cum);
        sigma_t[si] = std::sqrt(1.0 - cum);
        lambda_t[si] = std::log(alpha_t[si] / sigma_t[si]);
    }

    timesteps.resize(static_cast<std::size_t>(num_steps));
    for (int32_t i = 1; i <= num_steps; ++i) {
        double val =
            static_cast<double>(i) / static_cast<double>(num_steps) * static_cast<double>(T - 1);
        timesteps[static_cast<std::size_t>(num_steps - i)] = static_cast<float>(std::round(val));
    }

    model_outputs.clear();
    lower_order_nums = 0;
}

void DpmSolverState::step(const float* eps_pred, const float* sample, float* x_out,
                          std::size_t count, int32_t step_index, int32_t num_steps) {
    const auto si = static_cast<std::size_t>(step_index);
    const int32_t t_s0 = static_cast<int32_t>(std::round(timesteps[si]));
    const int32_t t_t =
        (si + 1 < timesteps.size()) ? static_cast<int32_t>(std::round(timesteps[si + 1])) : 0;

    std::vector<float> x0(count);
    eps_to_x0(*this, eps_pred, sample, x0.data(), count, t_s0);

    model_outputs.push_back(x0);
    if (model_outputs.size() > 2) {
        model_outputs.erase(model_outputs.begin());
    }

    int32_t order = 2;
    if (lower_order_nums < 1 || step_index == num_steps - 1) {
        order = 1;
    }

    if (order == 1 || model_outputs.size() < 2) {
        first_order_update(*this, model_outputs.back().data(), sample, x_out, count, t_s0, t_t);
    } else {
        const int32_t t_s1 = static_cast<int32_t>(std::round(timesteps[si - 1]));
        second_order_update(*this, model_outputs.back().data(),
                            model_outputs[model_outputs.size() - 2].data(), sample, x_out, count,
                            t_s0, t_s1, t_t);
    }

    if (lower_order_nums < 2) {
        ++lower_order_nums;
    }
}

// CHW float → HWC float [0,1]  (VAE output is in [-1,1] for AutoencoderKL)
namespace {
void chw_to_hwc(const float* src, int32_t h, int32_t w, std::vector<float>& out) {
    const auto hw = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
    out.resize(hw * 3);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            for (int32_t ch = 0; ch < 3; ++ch) {
                const auto src_idx =
                    static_cast<std::size_t>(ch) * hw + static_cast<std::size_t>(y * w + x);
                const auto dst_idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(w) * 3 +
                                     static_cast<std::size_t>(x) * 3 + static_cast<std::size_t>(ch);
                float v = (src[src_idx] + 1.0F) * 0.5F;
                out[dst_idx] = std::max(0.0F, std::min(1.0F, v));
            }
        }
    }
}
} // anonymous namespace

// ─── Construction / destruction ──────────────────────────────────────────

PixArtTorchTrtPipeline::PixArtTorchTrtPipeline(std::unique_ptr<TrtModule> text_encoder,
                                               std::unique_ptr<TrtModule> denoiser,
                                               std::unique_ptr<TrtModule> vae,
                                               DiffusionConfig config,
                                               std::shared_ptr<ITokenizer> tokenizer,
                                               std::string model_id_str)
    : text_encoder_(std::move(text_encoder)), denoiser_(std::move(denoiser)), vae_(std::move(vae)),
      config_(std::move(config)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)) {}

PixArtTorchTrtPipeline::~PixArtTorchTrtPipeline() = default;

// ─── T5 text encoder ────────────────────────────────────────────────────

bool PixArtTorchTrtPipeline::run_t5_encoder(const std::vector<int32_t>& input_ids,
                                            std::vector<float>& text_embeddings,
                                            bool zero_padding) {
    if (!text_encoder_ || !text_encoder_->ok())
        return false;

    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    // Pad/truncate to seq_len
    std::vector<int32_t> padded(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded.begin());

    // Build attention mask: 1 for real tokens, 0 for padding.
    // For null-text encoding (zero_padding=false), all tokens are padding
    // but we pass all-ones so T5 attends to them (matching HF behavior
    // for unconditional CFG embeddings).
    std::vector<int32_t> attention_mask(static_cast<std::size_t>(seq_len), zero_padding ? 0 : 1);
    if (zero_padding) {
        for (std::size_t i = 0; i < copy_len; ++i) {
            if (padded[i] != 0) {
                attention_mask[i] = 1;
            }
        }
    }

    // Torch-trt T5 engine expects:
    //   input_ids: int32 [1, seq_len]
    //   attention_mask: int32 [1, seq_len]
    // Output: "output0" float32 [1, seq_len, te_dim]
    TensorMap inputs;
    inputs["input_ids"] = Tensor{padded.data(), {1, static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attention_mask.data(), {1, static_cast<int64_t>(seq_len)}, DType::kInt32};

    TensorMap outputs = text_encoder_->forward(inputs);

    // Find the output tensor (torch-trt names it "output0")
    auto it = outputs.find("output0");
    if (it == outputs.end()) {
        // Fallback: try "text_embeddings" (raw TRT naming)
        it = outputs.find("text_embeddings");
        if (it == outputs.end()) {
            std::cerr << "[torchtrt_diffusion] T5 encoder: no output found\n";
            return false;
        }
    }

    const auto emb_size = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim);
    // Output is [1, seq_len, te_dim]; the helper copies the flattened payload.
    return tensor_to_float_vector(it->second, emb_size, text_embeddings, "T5 encoder");
}

// ─── DiT denoiser ───────────────────────────────────────────────────────

bool PixArtTorchTrtPipeline::run_denoiser(const std::vector<float>& latent,
                                          const std::vector<float>& text_embeddings,
                                          int32_t num_real_tokens, float timestep,
                                          std::vector<float>& output) {
    if (!denoiser_ || !denoiser_->ok())
        return false;

    const int32_t z_dim = config_.z_dim;
    const int32_t h_lat = config_.video_height / config_.scale_factor_spatial;
    const int32_t w_lat = config_.video_width / config_.scale_factor_spatial;
    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    // Torch-trt DiT engine inputs:
    //   sample: [1, z_dim, h_lat, w_lat]
    //   encoder_hidden_states: [1, seq_len, te_dim]
    //   timestep: [1]
    //   encoder_attention_mask: [1, seq_len] (1=real, 0=padding)
    //
    // The engine's PixArtDiTWrapper converts the {0,1} mask to additive
    // attention bias ({0, -10000}) and passes it as 3D to the model's
    // cross-attention layers (using TRT-safe matmul+softmax, not SDPA).
    //
    // Pipeline works in fp32; pack inputs to each TensorRT engine's actual
    // dtype, then convert outputs back to fp32.
    const DType sample_dtype = input_dtype_or(denoiser_.get(), "sample", DType::kFloat16);
    const DType text_dtype = input_dtype_or(denoiser_.get(), "encoder_hidden_states", sample_dtype);
    const DType timestep_dtype = input_dtype_or(denoiser_.get(), "timestep", sample_dtype);
    const DType mask_dtype =
        input_dtype_or(denoiser_.get(), "encoder_attention_mask", sample_dtype);

    std::vector<half_t> sample_scratch16;
    std::vector<half_t> text_scratch16;
    std::vector<half_t> timestep_scratch16;
    std::vector<half_t> mask_scratch16;
    std::vector<float> timestep_vec = {timestep};

    // Build encoder attention mask
    const auto mask_len = static_cast<std::size_t>(seq_len);
    std::vector<float> enc_mask(mask_len, 0.0F);
    const auto real = std::min(static_cast<std::size_t>(num_real_tokens), mask_len);
    for (std::size_t i = 0; i < real; ++i) {
        enc_mask[i] = 1.0F;
    }

    TensorMap inputs;
    inputs["sample"] = make_model_tensor(
        latent, sample_scratch16, sample_dtype,
        {1, static_cast<int64_t>(z_dim), static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)});
    inputs["encoder_hidden_states"] =
        make_model_tensor(text_embeddings, text_scratch16, text_dtype,
                          {1, static_cast<int64_t>(seq_len), static_cast<int64_t>(te_dim)});
    inputs["timestep"] = make_model_tensor(timestep_vec, timestep_scratch16, timestep_dtype, {1});
    inputs["encoder_attention_mask"] =
        make_model_tensor(enc_mask, mask_scratch16, mask_dtype, {1, static_cast<int64_t>(seq_len)});

    TensorMap outputs_map = denoiser_->forward(inputs);

    auto it = outputs_map.find("output0");
    if (it == outputs_map.end()) {
        it = outputs_map.find("output");
        if (it == outputs_map.end()) {
            std::cerr << "[torchtrt_diffusion] DiT: no output found\n";
            return false;
        }
    }

    // Output is [1, out_channels, h_lat, w_lat]. Torch-TensorRT may keep this
    // in fp16 or promote it to fp32 depending on the compiled graph.
    // PixArt-Sigma outputs 8 channels (learned sigma); take first z_dim channels
    const auto out_per_channel = static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const auto out_total = static_cast<std::size_t>(z_dim) * out_per_channel;
    return tensor_to_float_vector(it->second, out_total, output, "DiT");
}

// ─── VAE decode ─────────────────────────────────────────────────────────

bool PixArtTorchTrtPipeline::decode_vae(const std::vector<float>& latent, int32_t h_lat,
                                        int32_t w_lat, VideoResult& result) {
    if (!vae_ || !vae_->ok())
        return false;

    const int32_t z_dim = config_.z_dim;
    const int32_t h_out = h_lat * config_.scale_factor_spatial;
    const int32_t w_out = w_lat * config_.scale_factor_spatial;

    // Torch-trt VAE engine: "latent" [1, z_dim, h_lat, w_lat]
    //                       → "output0" fp32 [1, 3, h, w]
    // VAE scaling is already applied inside the VAEDecoderWrapper.
    const DType latent_dtype = input_dtype_or(vae_.get(), "latent", DType::kFloat16);
    std::vector<half_t> latent_scratch16;

    TensorMap inputs;
    inputs["latent"] = make_model_tensor(
        latent, latent_scratch16, latent_dtype,
        {1, static_cast<int64_t>(z_dim), static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)});

    TensorMap outputs_map = vae_->forward(inputs);

    auto it = outputs_map.find("output0");
    if (it == outputs_map.end()) {
        it = outputs_map.find("decoder_output");
        if (it == outputs_map.end()) {
            std::cerr << "[torchtrt_diffusion] VAE: no output found\n";
            return false;
        }
    }

    auto* raw = static_cast<float*>(it->second.data);
    chw_to_hwc(raw, h_out, w_out, result.frames);
    result.height = h_out;
    result.width = w_out;
    result.num_frames = 1;
    return true;
}

// ─── Tokenize + T5 encode ───────────────────────────────────────────────

bool PixArtTorchTrtPipeline::encode_prompt(const std::string& prompt,
                                           std::vector<int32_t>& input_ids,
                                           std::vector<float>& text_embeddings,
                                           std::vector<float>& null_text) {
    if (!tokenizer_) {
        std::cerr << "[torchtrt_diffusion] No tokenizer available\n";
        return false;
    }
    input_ids = tokenizer_->encode(prompt);
    if (input_ids.empty() || input_ids.back() != 1)
        input_ids.push_back(1); // T5 EOS
    std::cerr << "[torchtrt_diffusion] Tokenized: " << input_ids.size() << " tokens (with EOS)\n";

    if (!run_t5_encoder(input_ids, text_embeddings)) {
        std::cerr << "[torchtrt_diffusion] T5 encoding failed\n";
        return false;
    }
    // Null-text for CFG: just [EOS=1], matching HF's unconditional embedding
    std::vector<int32_t> null_ids = {1};
    if (!run_t5_encoder(null_ids, null_text)) {
        std::cerr << "[torchtrt_diffusion] T5 null encoding failed\n";
        return false;
    }
    return true;
}

// ─── Denoising loop ─────────────────────────────────────────────────────

bool PixArtTorchTrtPipeline::denoise_loop(std::vector<float>& latents,
                                          const std::vector<float>& text_embeddings,
                                          int32_t num_real_tokens,
                                          const std::vector<float>& null_text, float guidance_scale,
                                          DpmSolverState& scheduler, int32_t num_steps) {
    const auto latent_count = latents.size();
    std::vector<float> noise_pred(latent_count);
    std::vector<float> noise_uncond(latent_count);
    std::vector<float> eps_combined(latent_count);

    for (int32_t step = 0; step < num_steps; ++step) {
        const float t = scheduler.timesteps[static_cast<std::size_t>(step)];

        if (!run_denoiser(latents, text_embeddings, num_real_tokens, t, noise_pred))
            return false;

        if (!apply_cfg(latents, null_text, t, guidance_scale, noise_pred, noise_uncond,
                       eps_combined))
            return false;

        std::vector<float> new_latents(latent_count);
        scheduler.step(eps_combined.data(), latents.data(), new_latents.data(), latent_count, step,
                       num_steps);
        latents = std::move(new_latents);

        if ((step + 1) % 5 == 0 || step == num_steps - 1)
            std::cerr << "[torchtrt_diffusion] Step " << (step + 1) << "/" << num_steps
                      << " t=" << t << " eps_std=" << rms(eps_combined)
                      << " lat_std=" << rms(latents) << "\n";
    }
    return true;
}

// ─── CFG helper ─────────────────────────────────────────────────────────

bool PixArtTorchTrtPipeline::apply_cfg(const std::vector<float>& latents,
                                       const std::vector<float>& null_text, float timestep,
                                       float guidance_scale, const std::vector<float>& noise_pred,
                                       std::vector<float>& noise_uncond,
                                       std::vector<float>& eps_out) {
    if (guidance_scale <= 1.0F) {
        eps_out = noise_pred;
        return true;
    }
    if (!run_denoiser(latents, null_text, 1, timestep, noise_uncond))
        return false;
    for (std::size_t i = 0; i < noise_pred.size(); ++i) {
        eps_out[i] = noise_uncond[i] + guidance_scale * (noise_pred[i] - noise_uncond[i]);
    }
    return true;
}

// ─── Main generation ────────────────────────────────────────────────────

ImageResult PixArtTorchTrtPipeline::generate_image(const std::string& prompt,
                                                   const GenerateConfig& cfg) {
    ImageResult result;
    const int32_t z_dim = config_.z_dim;
    const int32_t h_lat = config_.video_height / config_.scale_factor_spatial;
    const int32_t w_lat = config_.video_width / config_.scale_factor_spatial;
    const auto latent_count = static_cast<std::size_t>(z_dim) * static_cast<std::size_t>(h_lat) *
                              static_cast<std::size_t>(w_lat);

    const int32_t num_steps = (cfg.num_steps > 0) ? cfg.num_steps : config_.num_inference_steps;
    const float guidance_scale =
        (cfg.guidance_scale > 0.0F) ? cfg.guidance_scale : config_.guidance_scale;

    std::cerr << "[torchtrt_diffusion] Generating image " << config_.video_height << "x"
              << config_.video_width << " (" << num_steps << " steps, guidance=" << guidance_scale
              << ")\n";

    // 1. Tokenize and run T5 encoder (conditioned + null text)
    std::vector<int32_t> input_ids;
    std::vector<float> text_embeddings, null_text;
    if (!encode_prompt(prompt, input_ids, text_embeddings, null_text)) {
        result.pixels = {};
        return result;
    }

    // 2. Set up scheduler (DPM-Solver++, matching the HF PixArt pipeline)
    DpmSolverState scheduler;
    scheduler.set_timesteps(num_steps);

    // 3. Initialize latents with the same deterministic Box-Muller path used
    // by the native PixArt/WAN runtime and HF reference harness.
    const uint32_t seed = (cfg.seed >= 0) ? static_cast<uint32_t>(cfg.seed) : 42U;
    std::vector<float> latents = diffusion::make_wan_initial_latents(latent_count, seed);
    std::cerr << "[torchtrt_diffusion] Initial latent std=" << rms(latents) << "\n";

    // 4. Denoising loop with CFG
    if (!denoise_loop(latents, text_embeddings, static_cast<int32_t>(input_ids.size()), null_text,
                      guidance_scale, scheduler, num_steps)) {
        std::cerr << "[torchtrt_diffusion] Denoising failed\n";
        result.pixels = {};
        return result;
    }

    // 5. VAE decode
    VideoResult vr;
    if (!decode_vae(latents, h_lat, w_lat, vr)) {
        std::cerr << "[torchtrt_diffusion] VAE decode failed\n";
        result.pixels = {};
        return result;
    }

    result.pixels = std::move(vr.frames);
    result.height = vr.height;
    result.width = vr.width;
    result.channels = 3;
    result.num_frames = 1;

    std::cerr << "[torchtrt_diffusion] Image generated: " << result.height << "x" << result.width
              << "\n";
    return result;
}

} // namespace trtmc

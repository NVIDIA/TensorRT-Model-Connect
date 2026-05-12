#include "runtime/models/z_image/pipeline.h"

#include "runtime/domains/diffusion/diffusion_math.h"
#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace trtmc {

using diffusion::FlowMatchEulerState;
using diffusion::resolve_requested_guidance;
using diffusion::resolve_requested_steps;
using diffusion_math::cpu_matmul_bias;
using diffusion_math::cpu_silu_inplace;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace {

constexpr int32_t kPadTokenId = 151643;
constexpr int32_t kSeqMultipleOf = 32;
constexpr float kVaeScalingFactor = 0.3611F;
constexpr float kVaeShiftFactor = 0.1159F;
constexpr float kRopeTheta = 256.0F;

// Z-Image RoPE axes dimensions
constexpr int32_t kRopeDimT = 32;
constexpr int32_t kRopeDimH = 48;
constexpr int32_t kRopeDimW = 48;

// ---------------------------------------------------------------------------
// Layout helper
// ---------------------------------------------------------------------------

struct ZImageLayout {
    int32_t dit_dim{0};
    int32_t text_seq{0};
    int32_t z_dim{0};
    int32_t h_lat{0};
    int32_t w_lat{0};
    int32_t ph{2};
    int32_t pw{2};
    int32_t nh{0};
    int32_t nw{0};
    int32_t num_patches{0};
    int32_t patch_dim{0};
    int32_t head_dim{0};
};

ZImageLayout make_layout(const DiffusionConfig& config) {
    ZImageLayout layout;
    layout.dit_dim = config.dit_dim;
    layout.text_seq = config.text_seq_len;
    layout.z_dim = config.z_dim;

    const int32_t vae_scale = config.scale_factor_spatial;
    layout.h_lat = 2 * (config.video_height / (vae_scale * 2));
    layout.w_lat = 2 * (config.video_width / (vae_scale * 2));

    if (config.patch_size.size() >= 3) {
        layout.ph = config.patch_size[1];
        layout.pw = config.patch_size[2];
    }
    layout.nh = layout.h_lat / layout.ph;
    layout.nw = layout.w_lat / layout.pw;
    layout.num_patches = layout.nh * layout.nw;
    layout.patch_dim = layout.ph * layout.pw * layout.z_dim;
    layout.head_dim = layout.dit_dim / std::max(config.dit_num_heads, 1);
    return layout;
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

std::string apply_chat_template(const std::string& prompt) {
    // HF ZImagePipeline._encode_prompt wraps the prompt in Qwen3 chat template.
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

// ---------------------------------------------------------------------------
// Token counting helpers
// ---------------------------------------------------------------------------

int32_t count_non_pad_tokens(const std::vector<int32_t>& input_ids) {
    int32_t count = 0;
    for (const auto id : input_ids) {
        if (id != kPadTokenId) {
            ++count;
        }
    }
    return count;
}

int32_t pad_to_next_multiple(int32_t value, int32_t multiple) {
    const int32_t rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

// ---------------------------------------------------------------------------
// Latent initialization
// ---------------------------------------------------------------------------

void initialize_latents(std::vector<float>& latents) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents) {
        v = dist(gen);
    }
}

// ---------------------------------------------------------------------------
// Negate in-place (Z-Image: noise_pred = -output)
// ---------------------------------------------------------------------------

void negate_inplace(std::vector<float>& values) {
    for (auto& v : values) {
        v = -v;
    }
}

// ---------------------------------------------------------------------------
// Latent denormalization before VAE
// ---------------------------------------------------------------------------

void denormalize_latents(std::vector<float>& latents) {
    const float inv_scale = 1.0F / kVaeScalingFactor;
    for (auto& v : latents) {
        v = v * inv_scale + kVaeShiftFactor;
    }
}

// ---------------------------------------------------------------------------
// VAE output conversion: CHW -> HWC, (pixel+1)*0.5, clamp [0,1]
// ---------------------------------------------------------------------------

ImageResult convert_vae_output(const float* raw, int32_t h_out, int32_t w_out) {
    ImageResult result;
    result.height = h_out;
    result.width = w_out;
    result.channels = 3;
    result.num_frames = 1;
    result.pixels.resize(static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out) * 3);

    for (int32_t y = 0; y < h_out; ++y) {
        for (int32_t x = 0; x < w_out; ++x) {
            for (int32_t ch = 0; ch < 3; ++ch) {
                const auto src_idx =
                    static_cast<std::size_t>(ch) * static_cast<std::size_t>(h_out * w_out) +
                    static_cast<std::size_t>(y * w_out + x);
                const auto dst_idx =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(w_out * 3) +
                    static_cast<std::size_t>(x * 3 + ch);
                const float v = (raw[src_idx] + 1.0F) * 0.5F;
                result.pixels[dst_idx] = std::max(0.0F, std::min(1.0F, v));
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Step logging
// ---------------------------------------------------------------------------

void log_step_stats(int32_t step, int32_t num_inference_steps, float raw_timestep,
                    const std::vector<float>& latents) {
    float lat_min = latents[0];
    float lat_max = latents[0];
    double lat_sum = 0.0;
    for (const auto v : latents) {
        lat_min = std::min(lat_min, v);
        lat_max = std::max(lat_max, v);
        lat_sum += static_cast<double>(v);
    }
    std::cerr << "  Step " << (step + 1) << "/" << num_inference_steps << " (t=" << raw_timestep
              << ") lat=[" << lat_min << ", " << lat_max
              << "] mean=" << (lat_sum / static_cast<double>(latents.size())) << "\n";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ZImagePipeline constructor / destructor
// ---------------------------------------------------------------------------

ZImagePipeline::ZImagePipeline(std::unique_ptr<TrtModule> text_encoder,
                               std::unique_ptr<TrtModule> denoiser, std::unique_ptr<TrtModule> vae,
                               DiffusionConfig config, PreprocessorWeights weights,
                               ZImagePreprocessorWeights z_weights,
                               std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                               std::string bundle_path)
    : text_encoder_(std::move(text_encoder)), denoiser_(std::move(denoiser)), vae_(std::move(vae)),
      config_(std::move(config)), weights_(std::move(weights)), z_weights_(std::move(z_weights)),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      bundle_path_(std::move(bundle_path)) {
    std::cerr << "[z-image] ZImagePipeline initialized"
              << " (height=" << config_.video_height << ", width=" << config_.video_width
              << ", steps=" << config_.num_inference_steps << ", cfg=" << config_.guidance_scale
              << ")\n";
}

ZImagePipeline::~ZImagePipeline() = default;

// ---------------------------------------------------------------------------
// Text encoder: Qwen3 (non-autoregressive, hidden_states[-2])
// ---------------------------------------------------------------------------

bool ZImagePipeline::run_text_encoder(const std::vector<int32_t>& input_ids,
                                      std::vector<float>& text_embeddings) {
    if (!text_encoder_) {
        std::cerr << "[z-image] No text encoder module\n";
        return false;
    }

    const int32_t seq_len = config_.text_seq_len;
    const int32_t te_dim = config_.text_encoder_dim;

    // Pad/truncate input_ids to text_seq_len
    std::vector<int32_t> padded_ids(static_cast<std::size_t>(seq_len), 0);
    const auto copy_len = std::min(static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), copy_len, padded_ids.begin());

    // Attention mask: 0 for real tokens, -1e9 for padding
    std::vector<float> mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < seq_len; ++i) {
        if (padded_ids[static_cast<std::size_t>(i)] != 0) {
            mask[static_cast<std::size_t>(i)] = 0.0F;
        }
    }

    // Build TensorMap for TrtModule::forward()
    TensorMap inputs;
    inputs["input_ids"] = Tensor{padded_ids.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{mask.data(), {static_cast<int64_t>(seq_len)}, DType::kFloat32};

    auto outputs = text_encoder_->forward(inputs);

    // Copy text_embeddings from output
    const auto& te_out = outputs["text_embeddings"];
    const auto emb_size = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(te_dim);
    text_embeddings.resize(emb_size);
    std::memcpy(text_embeddings.data(), te_out.data, emb_size * sizeof(float));

    // Zero out embeddings for padding positions
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
// Denoiser: Z-Image DiT (unified attention)
// ---------------------------------------------------------------------------

bool ZImagePipeline::run_denoiser(const std::vector<float>& hidden,
                                  const std::vector<float>& encoder_hidden,
                                  const std::vector<float>& temb,
                                  const std::vector<float>& cos_vals,
                                  const std::vector<float>& sin_vals, std::vector<float>& output) {
    if (!denoiser_) {
        std::cerr << "[z-image] No denoiser module\n";
        return false;
    }

    // Build TensorMap — all const_cast because Tensor::data is void*
    // but TrtModule::forward copies data in (H2D), so the source is not modified.
    TensorMap inputs;
    inputs["hidden_states"] = Tensor{
        const_cast<float*>(hidden.data()), {static_cast<int64_t>(hidden.size())}, DType::kFloat32};
    inputs["encoder_hidden_states"] = Tensor{const_cast<float*>(encoder_hidden.data()),
                                             {static_cast<int64_t>(encoder_hidden.size())},
                                             DType::kFloat32};
    inputs["timestep_embedding"] = Tensor{
        const_cast<float*>(temb.data()), {static_cast<int64_t>(temb.size())}, DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(cos_vals.size())},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(sin_vals.size())},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);

    const auto& dit_out = outputs["output"];
    const auto out_numel = dit_out.numel();
    output.resize(out_numel);
    std::memcpy(output.data(), dit_out.data, out_numel * sizeof(float));

    return true;
}

// ---------------------------------------------------------------------------
// Caption projection: RMSNorm + Linear(cap_dim -> dit_dim) + pad fill
// ---------------------------------------------------------------------------

void ZImagePipeline::project_caption(const std::vector<float>& text_emb, int32_t actual_len,
                                     int32_t padded_len, std::vector<float>& projected) const {
    const int32_t te_dim = config_.text_encoder_dim;
    const int32_t dit_dim = config_.dit_dim;
    const int32_t text_seq = config_.text_seq_len;

    // RMSNorm(text_embeddings) using cap_norm_weight
    std::vector<float> normed(text_emb.size());
    for (int32_t s = 0; s < text_seq; ++s) {
        const float* row =
            text_emb.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(te_dim);
        float* out_row =
            normed.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(te_dim);

        double sum_sq = 0.0;
        for (int32_t d = 0; d < te_dim; ++d) {
            sum_sq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        }
        const float rms =
            std::sqrt(static_cast<float>(sum_sq / static_cast<double>(te_dim)) + 1e-5F);
        const float inv_rms = 1.0F / rms;

        for (int32_t d = 0; d < te_dim; ++d) {
            out_row[d] = row[d] * inv_rms * z_weights_.cap_norm_weight[static_cast<std::size_t>(d)];
        }
    }

    // Linear(te_dim, dit_dim) + bias
    projected.resize(static_cast<std::size_t>(text_seq) * static_cast<std::size_t>(dit_dim));
    cpu_matmul_bias(normed.data(), z_weights_.cap_proj_weight.data(),
                    z_weights_.cap_proj_bias.data(), projected.data(), text_seq, te_dim, dit_dim);

    // Fill padding positions (actual_len..text_seq) with cap_pad_token
    if (!z_weights_.cap_pad_token.empty()) {
        for (int32_t t = actual_len; t < text_seq; ++t) {
            float* row =
                projected.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(dit_dim);
            for (int32_t d = 0; d < dit_dim; ++d) {
                row[d] = z_weights_.cap_pad_token[static_cast<std::size_t>(
                    d % static_cast<int32_t>(z_weights_.cap_pad_token.size()))];
            }
        }
    }

    (void)padded_len; // padded_len used for RoPE, not projection
}

// ---------------------------------------------------------------------------
// 3-axis RoPE (time, height, width) with theta=256
// ---------------------------------------------------------------------------

void ZImagePipeline::compute_3d_rope(int32_t cap_padded_len, int32_t num_patches, int32_t nh,
                                     int32_t nw, std::vector<float>& cos_out,
                                     std::vector<float>& sin_out) const {
    const int32_t head_dim = config_.dit_dim / std::max(config_.dit_num_heads, 1);
    const int32_t text_seq = config_.text_seq_len;
    const int32_t total_seq = num_patches + text_seq;

    // Initialize cos=1, sin=0 (identity RoPE)
    cos_out.assign(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 1.0F);
    sin_out.assign(static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(head_dim), 0.0F);

    // HF RoPE uses complex numbers: freqs = 1/(theta^(2i/d)) for i in 0..d/2
    // Applied as x_complex * freqs_cis (rotate-half with interleaved pairs):
    //   cos_row[2*i] = cos(angle), cos_row[2*i+1] = cos(angle)
    //   sin_row[2*i] = sin(angle), sin_row[2*i+1] = sin(angle)
    auto encode_pos = [&](float* cos_row, float* sin_row, int32_t t_pos, int32_t h_pos,
                          int32_t w_pos) {
        int32_t offset = 0;

        // Time dimension (kRopeDimT/2 pairs)
        for (int32_t i = 0; i < kRopeDimT / 2; ++i) {
            const float freq = 1.0F / std::pow(kRopeTheta, 2.0F * static_cast<float>(i) /
                                                               static_cast<float>(kRopeDimT));
            const float angle = static_cast<float>(t_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += kRopeDimT;

        // Height dimension (kRopeDimH/2 pairs)
        for (int32_t i = 0; i < kRopeDimH / 2; ++i) {
            const float freq = 1.0F / std::pow(kRopeTheta, 2.0F * static_cast<float>(i) /
                                                               static_cast<float>(kRopeDimH));
            const float angle = static_cast<float>(h_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
        offset += kRopeDimH;

        // Width dimension (kRopeDimW/2 pairs)
        for (int32_t i = 0; i < kRopeDimW / 2; ++i) {
            const float freq = 1.0F / std::pow(kRopeTheta, 2.0F * static_cast<float>(i) /
                                                               static_cast<float>(kRopeDimW));
            const float angle = static_cast<float>(w_pos) * freq;
            cos_row[offset + 2 * i] = std::cos(angle);
            cos_row[offset + 2 * i + 1] = std::cos(angle);
            sin_row[offset + 2 * i] = std::sin(angle);
            sin_row[offset + 2 * i + 1] = std::sin(angle);
        }
    };

    // Noise token positions: image_ori_pos_ids start at (cap_padded_len + 1, 0, 0)
    const int32_t noise_t_start = cap_padded_len + 1;

    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t idx = hy * nw + wx;
            encode_pos(
                cos_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                sin_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
                noise_t_start, hy, wx);
        }
    }

    // Caption token positions: start=(1, 0, 0), stepping in time only
    for (int32_t t = 0; t < cap_padded_len; ++t) {
        const int32_t idx = num_patches + t;
        encode_pos(
            cos_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
            sin_out.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(head_dim),
            t + 1, 0, 0); // t starts at 1
    }
    // Remaining positions (cap_padded_len..text_seq) keep identity (cos=1, sin=0)
}

// ---------------------------------------------------------------------------
// Patchify 2D: [C, H, W] -> [num_patches, patch_dim]
// HF order: "c 1 1 h ph w pw -> (h w) (ph pw c)"
// ---------------------------------------------------------------------------

void ZImagePipeline::patchify_2d(const std::vector<float>& latents, int32_t c, int32_t h, int32_t w,
                                 std::vector<float>& patches) const {
    int32_t ph = 2, pw = 2;
    if (config_.patch_size.size() >= 3) {
        ph = config_.patch_size[1];
        pw = config_.patch_size[2];
    }
    const int32_t nh = h / ph;
    const int32_t nw = w / pw;
    const int32_t patch_dim = ph * pw * c;
    const int32_t num_patches_val = nh * nw;

    patches.resize(static_cast<std::size_t>(num_patches_val) * static_cast<std::size_t>(patch_dim));

    // latents layout: [C, H, W]
    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t patch_idx = hy * nw + wx;
            float* dst = patches.data() +
                         static_cast<std::size_t>(patch_idx) * static_cast<std::size_t>(patch_dim);

            // HF order: (pf ph pw c) -> iterate dy, dx, channel
            int32_t offset = 0;
            for (int32_t dy = 0; dy < ph; ++dy) {
                for (int32_t dx = 0; dx < pw; ++dx) {
                    for (int32_t ci = 0; ci < c; ++ci) {
                        const int32_t y = hy * ph + dy;
                        const int32_t x = wx * pw + dx;
                        const auto src_idx =
                            static_cast<std::size_t>(ci) * static_cast<std::size_t>(h * w) +
                            static_cast<std::size_t>(y * w + x);
                        dst[offset++] = latents[src_idx];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unpatchify 2D: [num_patches, patch_dim] -> [C, H, W]
// ---------------------------------------------------------------------------

void ZImagePipeline::unpatchify_2d(const std::vector<float>& patches, int32_t c, int32_t h,
                                   int32_t w, std::vector<float>& output) const {
    int32_t ph = 2, pw = 2;
    if (config_.patch_size.size() >= 3) {
        ph = config_.patch_size[1];
        pw = config_.patch_size[2];
    }
    const int32_t nh = h / ph;
    const int32_t nw = w / pw;
    const int32_t patch_dim = ph * pw * c;

    output.resize(static_cast<std::size_t>(c) * static_cast<std::size_t>(h) *
                  static_cast<std::size_t>(w));

    for (int32_t hy = 0; hy < nh; ++hy) {
        for (int32_t wx = 0; wx < nw; ++wx) {
            const int32_t patch_idx = hy * nw + wx;
            const float* src = patches.data() + static_cast<std::size_t>(patch_idx) *
                                                    static_cast<std::size_t>(patch_dim);

            int32_t offset = 0;
            for (int32_t dy = 0; dy < ph; ++dy) {
                for (int32_t dx = 0; dx < pw; ++dx) {
                    for (int32_t ci = 0; ci < c; ++ci) {
                        const int32_t y = hy * ph + dy;
                        const int32_t x = wx * pw + dx;
                        const auto dst_idx =
                            static_cast<std::size_t>(ci) * static_cast<std::size_t>(h * w) +
                            static_cast<std::size_t>(y * w + x);
                        output[dst_idx] = src[offset++];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// generate_image — full Z-Image pipeline
// ---------------------------------------------------------------------------

ImageResult ZImagePipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    ImageResult result;
    result.height = config_.video_height;
    result.width = config_.video_width;
    result.channels = 3;
    result.num_frames = 1;

    const int32_t num_inference_steps =
        resolve_requested_steps(cfg.num_steps, config_.num_inference_steps, true);
    const float guidance_scale =
        resolve_requested_guidance(cfg.guidance_scale, config_.guidance_scale);
    (void)guidance_scale; // Z-Image does not use CFG

    const ZImageLayout layout = make_layout(config_);
    std::cerr << "[z-image] Latent: " << layout.h_lat << "x" << layout.w_lat
              << ", patches: " << layout.num_patches << " (" << layout.nh << "x" << layout.nw
              << ")\n";

    if (!z_weights_.valid) {
        std::cerr << "[z-image] WARNING: Z-Image preprocessor weights not loaded.\n";
        return result;
    }

    // ── 1. Apply chat template and tokenize ──
    const std::string prepared = apply_chat_template(prompt);
    if (!tokenizer_) {
        std::cerr << "[z-image] No tokenizer available\n";
        return result;
    }
    std::vector<int32_t> input_ids = tokenizer_->encode(prepared);

    // ── 2. Run Qwen3 text encoder ──
    std::cerr << "[z-image] Running text encoder ...\n";
    std::vector<float> text_embeddings;
    if (!run_text_encoder(input_ids, text_embeddings)) {
        std::cerr << "[z-image] Text encoder failed\n";
        return result;
    }
    std::cerr << "[z-image] Text encoder done\n";

    // ── 3. Count actual tokens and compute padded caption length ──
    const int32_t cap_ori_len = count_non_pad_tokens(input_ids);
    const int32_t cap_padded_len = pad_to_next_multiple(cap_ori_len, kSeqMultipleOf);

    std::cerr << "[z-image] Caption: " << cap_ori_len << " actual tokens, " << cap_padded_len
              << " padded (SEQ_MULTI_OF=" << kSeqMultipleOf << ")\n";

    // ── 4. Project caption: RMSNorm + Linear + pad fill ──
    std::vector<float> caption_projected;
    project_caption(text_embeddings, cap_ori_len, cap_padded_len, caption_projected);

    // ── 5. Compute 3D RoPE ──
    std::vector<float> rope_cos, rope_sin;
    compute_3d_rope(cap_padded_len, layout.num_patches, layout.nh, layout.nw, rope_cos, rope_sin);

    // ── 6. Initialize random latents ──
    const auto latent_size = static_cast<std::size_t>(layout.z_dim) *
                             static_cast<std::size_t>(layout.h_lat) *
                             static_cast<std::size_t>(layout.w_lat);
    std::vector<float> latents(latent_size);
    initialize_latents(latents);

    // ── 7. Create FlowMatchEuler scheduler ──
    FlowMatchEulerState scheduler;
    scheduler.shift = config_.flow_shift;
    scheduler.use_zero_sigma_min = true;
    scheduler.set_timesteps(num_inference_steps);

    std::cerr << "[z-image] Scheduler: shift=" << scheduler.shift << ", timesteps=[";
    for (int32_t i = 0; i < num_inference_steps; ++i) {
        if (i > 0) {
            std::cerr << ", ";
        }
        std::cerr << scheduler.timesteps[static_cast<std::size_t>(i)];
    }
    std::cerr << "]\n";

    // ── 8. Denoising loop ──
    std::cerr << "[z-image] Starting denoising loop (" << num_inference_steps << " steps) ...\n";

    const int32_t freq_dim = config_.freq_dim;

    std::vector<float> temb;
    std::vector<float> patches;
    std::vector<float> hidden;
    std::vector<float> denoiser_output;
    std::vector<float> noise_pred;

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float raw_timestep = scheduler.timesteps[static_cast<std::size_t>(step)];

        // (a) Compute timestep embedding via MLP (uses 1000-t)
        const float t_for_embedding = 1000.0F - raw_timestep;
        {
            // Sinusoidal embedding
            const int32_t half = freq_dim / 2;
            std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
            for (int32_t i = 0; i < half; ++i) {
                const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                                            static_cast<float>(half));
                sinusoidal[static_cast<std::size_t>(i)] = std::cos(t_for_embedding * freq);
                sinusoidal[static_cast<std::size_t>(i + half)] = std::sin(t_for_embedding * freq);
            }

            // MLP: Linear(freq_dim, mid_dim) -> SiLU -> Linear(mid_dim, freq_dim)
            const int32_t mid_dim = static_cast<int32_t>(z_weights_.t_embedder_mlp_0_bias.size());
            std::vector<float> h1(static_cast<std::size_t>(mid_dim));
            cpu_matmul_bias(sinusoidal.data(), z_weights_.t_embedder_mlp_0_weight.data(),
                            z_weights_.t_embedder_mlp_0_bias.data(), h1.data(), 1, freq_dim,
                            mid_dim);
            cpu_silu_inplace(h1.data(), static_cast<std::size_t>(mid_dim));

            temb.resize(static_cast<std::size_t>(freq_dim));
            cpu_matmul_bias(h1.data(), z_weights_.t_embedder_mlp_2_weight.data(),
                            z_weights_.t_embedder_mlp_2_bias.data(), temb.data(), 1, mid_dim,
                            freq_dim);
        }

        // (b) Patchify 2D latents
        patchify_2d(latents, layout.z_dim, layout.h_lat, layout.w_lat, patches);

        // (c) Embed patches via x_embedder linear
        hidden.resize(static_cast<std::size_t>(layout.num_patches) *
                      static_cast<std::size_t>(layout.dit_dim));
        cpu_matmul_bias(patches.data(), z_weights_.x_embed_weight.data(),
                        z_weights_.x_embed_bias.data(), hidden.data(), layout.num_patches,
                        layout.patch_dim, layout.dit_dim);

        // (d) Run denoiser via TrtModule::forward()
        if (!run_denoiser(hidden, caption_projected, temb, rope_cos, rope_sin, denoiser_output)) {
            std::cerr << "[z-image] DiT failed at step " << step << "\n";
            return result;
        }

        // (e) Unpatchify
        unpatchify_2d(denoiser_output, layout.z_dim, layout.h_lat, layout.w_lat, noise_pred);

        // (f) NEGATE output (Z-Image: noise_pred = -output)
        negate_inplace(noise_pred);

        // (g) Scheduler step
        scheduler.step(noise_pred.data(), latents.data(), latents.data(), latents.size(), step);

        log_step_stats(step, num_inference_steps, raw_timestep, latents);
    }

    // ── 9. Denormalize latents ──
    denormalize_latents(latents);

    // ── 10. Run VAE decode via TrtModule::forward() ──
    std::cerr << "[z-image] Decoding latents via VAE ...\n";
    if (!vae_) {
        std::cerr << "[z-image] No VAE decoder module\n";
        return result;
    }

    {
        const int32_t h_out = layout.h_lat * 8;
        const int32_t w_out = layout.w_lat * 8;

        TensorMap vae_inputs;
        vae_inputs["latent_input"] =
            Tensor{latents.data(),
                   {1, static_cast<int64_t>(layout.z_dim), static_cast<int64_t>(layout.h_lat),
                    static_cast<int64_t>(layout.w_lat)},
                   DType::kFloat32};

        auto vae_outputs = vae_->forward(vae_inputs);

        const auto& vae_out = vae_outputs["decoder_output"];
        const auto* raw_pixels = static_cast<const float*>(vae_out.data);

        result = convert_vae_output(raw_pixels, h_out, w_out);
    }

    std::cerr << "[z-image] Image generated: " << result.width << "x" << result.height << "\n";
    return result;
}

} // namespace trtmc

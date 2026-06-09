#include "runtime/models/z_image/pipeline.h"

#include "runtime/domains/diffusion/batch_utils.h"
#include "runtime/domains/diffusion/diffusion_math.h"
#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
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

void initialize_latents_data(float* data, std::size_t count, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
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
                               std::string bundle_path, std::shared_ptr<void> distributed_owner,
                               int32_t tensor_parallel_rank, int32_t tensor_parallel_size)
    : distributed_owner_(std::move(distributed_owner)), tensor_parallel_rank_(tensor_parallel_rank),
      tensor_parallel_size_(tensor_parallel_size), text_encoder_(std::move(text_encoder)),
      denoiser_(std::move(denoiser)), vae_(std::move(vae)), config_(std::move(config)),
      weights_(std::move(weights)), z_weights_(std::move(z_weights)),
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
// Batched denoiser forward. Every input carries a leading batch dim of
// ``batch_size``. The output is contiguous ``[B, num_patches, patch_dim]``.
// ---------------------------------------------------------------------------

bool ZImagePipeline::run_denoiser_batched(const std::vector<float>& hidden,
                                          const std::vector<float>& encoder_hidden,
                                          const std::vector<float>& temb,
                                          const std::vector<float>& cos_vals,
                                          const std::vector<float>& sin_vals, int32_t batch_size,
                                          int32_t num_patches, int32_t dit_dim, int32_t text_seq,
                                          int32_t freq_dim, int32_t total_seq, int32_t head_dim,
                                          int32_t patch_dim, std::vector<float>& output) {
    if (!denoiser_) {
        std::cerr << "[z-image] No denoiser module\n";
        return false;
    }
    if (batch_size < 1) {
        std::cerr << "[z-image] Invalid denoiser batch size: " << batch_size << "\n";
        return false;
    }

    TensorMap inputs;
    inputs["hidden_states"] =
        Tensor{const_cast<float*>(hidden.data()),
               {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_patches),
                static_cast<int64_t>(dit_dim)},
               DType::kFloat32};
    inputs["encoder_hidden_states"] =
        Tensor{const_cast<float*>(encoder_hidden.data()),
               {static_cast<int64_t>(batch_size), static_cast<int64_t>(text_seq),
                static_cast<int64_t>(dit_dim)},
               DType::kFloat32};
    inputs["timestep_embedding"] =
        Tensor{const_cast<float*>(temb.data()),
               {static_cast<int64_t>(batch_size), static_cast<int64_t>(freq_dim)},
               DType::kFloat32};
    inputs["rotary_cos"] = Tensor{const_cast<float*>(cos_vals.data()),
                                  {static_cast<int64_t>(batch_size),
                                   static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};
    inputs["rotary_sin"] = Tensor{const_cast<float*>(sin_vals.data()),
                                  {static_cast<int64_t>(batch_size),
                                   static_cast<int64_t>(total_seq), static_cast<int64_t>(head_dim)},
                                  DType::kFloat32};

    auto outputs = denoiser_->forward(inputs);

    const auto& dit_out = outputs["output"];
    const auto expected = static_cast<std::size_t>(batch_size) *
                          static_cast<std::size_t>(num_patches) *
                          static_cast<std::size_t>(patch_dim);
    if (dit_out.numel() < expected) {
        std::cerr << "[z-image] Batched DiT output too small: got " << dit_out.numel()
                  << " floats, expected at least " << expected << "\n";
        return false;
    }
    output.resize(expected);
    std::memcpy(output.data(), dit_out.data, expected * sizeof(float));

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

ImageResult ZImagePipeline::decode_z_image_result(int32_t z_dim, int32_t h_lat, int32_t w_lat,
                                                  std::vector<float>& latents, ImageResult result) {
    denormalize_latents(latents);

    if (tensor_parallel_size_ > 1 && tensor_parallel_rank_ != 0) {
        std::cerr << "[z-image] TP rank " << tensor_parallel_rank_
                  << " skips VAE decode; rank 0 writes image artifacts\n";
        ImageResult empty;
        empty.num_frames = 0;
        return empty;
    }

    std::cerr << "[z-image] Decoding latents via VAE ...\n";
    if (!vae_) {
        std::cerr << "[z-image] No VAE decoder module\n";
        return result;
    }

    const int32_t h_out = h_lat * 8;
    const int32_t w_out = w_lat * 8;

    TensorMap vae_inputs;
    vae_inputs["latent_input"] = Tensor{
        latents.data(),
        {1, static_cast<int64_t>(z_dim), static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)},
        DType::kFloat32};

    auto vae_outputs = vae_->forward(vae_inputs);

    const auto& vae_out = vae_outputs["decoder_output"];
    const auto* raw_pixels = static_cast<const float*>(vae_out.data);

    result = convert_vae_output(raw_pixels, h_out, w_out);
    std::cerr << "[z-image] Image generated: " << result.width << "x" << result.height << "\n";
    return result;
}

// ---------------------------------------------------------------------------
// Helpers private to the batched generate path.
// ---------------------------------------------------------------------------

namespace {

void compute_timestep_embedding(const ZImagePreprocessorWeights& z_weights, int32_t freq_dim,
                                float raw_timestep, std::vector<float>& temb_out) {
    const float t_for_embedding = 1000.0F - raw_timestep;
    const int32_t half = freq_dim / 2;
    std::vector<float> sinusoidal(static_cast<std::size_t>(freq_dim));
    for (int32_t i = 0; i < half; ++i) {
        const float freq =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
        sinusoidal[static_cast<std::size_t>(i)] = std::cos(t_for_embedding * freq);
        sinusoidal[static_cast<std::size_t>(i + half)] = std::sin(t_for_embedding * freq);
    }

    const int32_t mid_dim = static_cast<int32_t>(z_weights.t_embedder_mlp_0_bias.size());
    std::vector<float> h1(static_cast<std::size_t>(mid_dim));
    cpu_matmul_bias(sinusoidal.data(), z_weights.t_embedder_mlp_0_weight.data(),
                    z_weights.t_embedder_mlp_0_bias.data(), h1.data(), 1, freq_dim, mid_dim);
    cpu_silu_inplace(h1.data(), static_cast<std::size_t>(mid_dim));

    temb_out.resize(static_cast<std::size_t>(freq_dim));
    cpu_matmul_bias(h1.data(), z_weights.t_embedder_mlp_2_weight.data(),
                    z_weights.t_embedder_mlp_2_bias.data(), temb_out.data(), 1, mid_dim, freq_dim);
}

std::vector<std::uint32_t> resolve_batch_seeds(const std::vector<std::string>& prompts,
                                               const std::vector<std::uint32_t>& per_sample_seeds,
                                               int32_t cfg_seed) {
    if (!per_sample_seeds.empty()) {
        if (per_sample_seeds.size() != prompts.size()) {
            throw std::invalid_argument(
                "generate_image_batch: per_sample_seeds size must match prompts size");
        }
        return per_sample_seeds;
    }
    if (prompts.size() == 1U) {
        return {static_cast<std::uint32_t>(cfg_seed >= 0 ? cfg_seed : static_cast<int32_t>(42))};
    }
    const std::uint64_t global_seed = cfg_seed >= 0 ? static_cast<std::uint64_t>(cfg_seed) : 42ULL;
    return diffusion::derive_per_sample_seeds(global_seed, static_cast<int>(prompts.size()));
}

} // namespace

ImageResult ZImagePipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    const std::uint32_t seed =
        static_cast<std::uint32_t>(cfg.seed >= 0 ? cfg.seed : static_cast<int32_t>(42));
    auto outs = generate_image_batch({prompt}, {seed}, cfg);
    if (outs.empty()) {
        ImageResult empty;
        empty.height = config_.video_height;
        empty.width = config_.video_width;
        empty.channels = 3;
        empty.num_frames = 0;
        return empty;
    }
    return std::move(outs.front());
}

// ---------------------------------------------------------------------------
// generate_image_batch — full Z-Image pipeline, batched.
// Decisions B/D/E (design doc 2026-05-11):
//   B: one DiT forward per step, no CFG branch (guidance_scale ignored).
//   D: reproducible per-sample seeds derived from cfg.seed.
//   E: VAE decode always slices at B=1 (one decode per sample).
// Chunking: per-call batch is split into chunks of at most
// ``config_.max_batch_size.dit`` (further clamped by the engine profile's
// kMAX for "hidden_states"). When the DiT engine is the legacy static
// (rank-2) build, the path collapses to one-prompt-at-a-time invocations
// of the same step body for correctness without an override.
// ---------------------------------------------------------------------------

std::vector<ImageResult>
ZImagePipeline::generate_image_batch(const std::vector<std::string>& prompts,
                                     const std::vector<std::uint32_t>& per_sample_seeds,
                                     const GenerateConfig& cfg) {
    if (prompts.empty())
        return {};

    const std::vector<std::uint32_t> resolved_seeds =
        resolve_batch_seeds(prompts, per_sample_seeds, cfg.seed);

    const int32_t num_inference_steps =
        resolve_requested_steps(cfg.num_steps, config_.num_inference_steps, true);
    const float guidance_scale =
        resolve_requested_guidance(cfg.guidance_scale, config_.guidance_scale);
    (void)guidance_scale; // Z-Image does not use CFG even under B > 1.

    const ZImageLayout layout = make_layout(config_);
    std::cerr << "[z-image] Latent: " << layout.h_lat << "x" << layout.w_lat
              << ", patches: " << layout.num_patches << " (" << layout.nh << "x" << layout.nw
              << "), batch=" << prompts.size() << "\n";

    if (!z_weights_.valid) {
        std::cerr << "[z-image] WARNING: Z-Image preprocessor weights not loaded.\n";
        return {};
    }
    if (!tokenizer_) {
        std::cerr << "[z-image] No tokenizer available\n";
        return {};
    }
    if (!vae_) {
        std::cerr << "[z-image] No VAE decoder module\n";
        return {};
    }

    int32_t cap = std::max(config_.max_batch_size.dit, 1);
    const bool engine_is_batched = denoiser_ && denoiser_->input_rank("hidden_states") == 3;
    if (engine_is_batched) {
        const auto profile_max = denoiser_->input_profile_shape(
            "hidden_states", denoiser_->profile_idx(), ProfileShapeSelector::kMax);
        if (!profile_max.empty() && profile_max[0] > 0) {
            cap = std::min(cap, static_cast<int32_t>(profile_max[0]));
        }
    } else {
        cap = 1;
    }
    cap = std::max(cap, 1);

    const auto chunks = diffusion::plan_chunks(static_cast<int>(prompts.size()), cap);
    std::vector<ImageResult> results;
    results.reserve(prompts.size());

    const auto latent_size = static_cast<std::size_t>(layout.z_dim) *
                             static_cast<std::size_t>(layout.h_lat) *
                             static_cast<std::size_t>(layout.w_lat);
    const auto patch_size =
        static_cast<std::size_t>(layout.num_patches) * static_cast<std::size_t>(layout.patch_dim);
    const auto hidden_size =
        static_cast<std::size_t>(layout.num_patches) * static_cast<std::size_t>(layout.dit_dim);
    const auto caption_size =
        static_cast<std::size_t>(layout.text_seq) * static_cast<std::size_t>(layout.dit_dim);
    const int32_t total_seq = layout.num_patches + layout.text_seq;
    const auto rope_size =
        static_cast<std::size_t>(total_seq) * static_cast<std::size_t>(layout.head_dim);
    const int32_t freq_dim = config_.freq_dim;

    std::size_t prompt_offset = 0;
    for (int32_t chunk_size : chunks) {
        const int32_t batch = chunk_size;
        std::cerr << "[z-image] Chunk B=" << batch << "/" << cap << ", prompts [" << prompt_offset
                  << ".." << (prompt_offset + static_cast<std::size_t>(batch) - 1U) << "]\n";

        std::vector<float> caption_projected_b(static_cast<std::size_t>(batch) * caption_size);
        std::vector<float> rope_cos_b(static_cast<std::size_t>(batch) * rope_size);
        std::vector<float> rope_sin_b(static_cast<std::size_t>(batch) * rope_size);
        std::vector<float> latents(static_cast<std::size_t>(batch) * latent_size);

        for (int32_t b = 0; b < batch; ++b) {
            const std::size_t sample_idx = prompt_offset + static_cast<std::size_t>(b);
            const std::string prepared = apply_chat_template(prompts[sample_idx]);
            const std::vector<int32_t> input_ids = tokenizer_->encode(prepared);

            std::vector<float> text_embeddings;
            if (!run_text_encoder(input_ids, text_embeddings)) {
                std::cerr << "[z-image] Text encoder failed for sample " << sample_idx << "\n";
                return {};
            }

            const int32_t cap_ori_len = count_non_pad_tokens(input_ids);
            const int32_t cap_padded_len = pad_to_next_multiple(cap_ori_len, kSeqMultipleOf);

            std::vector<float> caption_projected;
            project_caption(text_embeddings, cap_ori_len, cap_padded_len, caption_projected);
            std::copy(caption_projected.begin(), caption_projected.end(),
                      caption_projected_b.begin() + static_cast<std::ptrdiff_t>(b) *
                                                        static_cast<std::ptrdiff_t>(caption_size));

            std::vector<float> rope_cos_one, rope_sin_one;
            compute_3d_rope(cap_padded_len, layout.num_patches, layout.nh, layout.nw, rope_cos_one,
                            rope_sin_one);
            std::copy(rope_cos_one.begin(), rope_cos_one.end(),
                      rope_cos_b.begin() +
                          static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(rope_size));
            std::copy(rope_sin_one.begin(), rope_sin_one.end(),
                      rope_sin_b.begin() +
                          static_cast<std::ptrdiff_t>(b) * static_cast<std::ptrdiff_t>(rope_size));

            initialize_latents_data(latents.data() + static_cast<std::size_t>(b) * latent_size,
                                    latent_size, resolved_seeds[sample_idx]);
        }

        // FlowMatchEuler is stateless per-sample, so one scheduler suffices.
        FlowMatchEulerState scheduler;
        scheduler.shift = config_.flow_shift;
        scheduler.use_zero_sigma_min = true;
        scheduler.set_timesteps(num_inference_steps);

        std::vector<float> temb_one;
        std::vector<float> temb_b(static_cast<std::size_t>(batch) *
                                  static_cast<std::size_t>(freq_dim));
        std::vector<float> sample_latents;
        std::vector<float> patches;
        std::vector<float> hidden_b(static_cast<std::size_t>(batch) * hidden_size);
        std::vector<float> hidden_one;
        std::vector<float> denoiser_output;
        std::vector<float> sample_noise_pred;
        std::vector<float> noise_pred(static_cast<std::size_t>(batch) * latent_size);

        for (int32_t step = 0; step < num_inference_steps; ++step) {
            const float raw_timestep = scheduler.timesteps[static_cast<std::size_t>(step)];

            compute_timestep_embedding(z_weights_, freq_dim, raw_timestep, temb_one);
            for (int32_t b = 0; b < batch; ++b) {
                std::copy(temb_one.begin(), temb_one.end(),
                          temb_b.begin() + static_cast<std::ptrdiff_t>(b) *
                                               static_cast<std::ptrdiff_t>(freq_dim));
            }

            for (int32_t b = 0; b < batch; ++b) {
                const auto* sample_latents_ptr =
                    latents.data() + static_cast<std::size_t>(b) * latent_size;
                sample_latents.assign(sample_latents_ptr, sample_latents_ptr + latent_size);
                patchify_2d(sample_latents, layout.z_dim, layout.h_lat, layout.w_lat, patches);
                cpu_matmul_bias(patches.data(), z_weights_.x_embed_weight.data(),
                                z_weights_.x_embed_bias.data(),
                                hidden_b.data() + static_cast<std::size_t>(b) * hidden_size,
                                layout.num_patches, layout.patch_dim, layout.dit_dim);
            }

            if (engine_is_batched) {
                if (!run_denoiser_batched(hidden_b, caption_projected_b, temb_b, rope_cos_b,
                                          rope_sin_b, batch, layout.num_patches, layout.dit_dim,
                                          layout.text_seq, freq_dim, total_seq, layout.head_dim,
                                          layout.patch_dim, denoiser_output)) {
                    std::cerr << "[z-image] Batched DiT failed at step " << step << "\n";
                    return {};
                }
            } else {
                denoiser_output.resize(static_cast<std::size_t>(batch) * patch_size);
                for (int32_t b = 0; b < batch; ++b) {
                    const auto* h_ptr = hidden_b.data() + static_cast<std::size_t>(b) * hidden_size;
                    const auto* cap_ptr =
                        caption_projected_b.data() + static_cast<std::size_t>(b) * caption_size;
                    const auto* cos_ptr =
                        rope_cos_b.data() + static_cast<std::size_t>(b) * rope_size;
                    const auto* sin_ptr =
                        rope_sin_b.data() + static_cast<std::size_t>(b) * rope_size;
                    hidden_one.assign(h_ptr, h_ptr + hidden_size);
                    std::vector<float> cap_one(cap_ptr, cap_ptr + caption_size);
                    std::vector<float> cos_one(cos_ptr, cos_ptr + rope_size);
                    std::vector<float> sin_one(sin_ptr, sin_ptr + rope_size);
                    std::vector<float> out_one;
                    if (!run_denoiser(hidden_one, cap_one, temb_one, cos_one, sin_one, out_one)) {
                        std::cerr << "[z-image] DiT failed at step " << step << " sample "
                                  << (prompt_offset + static_cast<std::size_t>(b)) << "\n";
                        return {};
                    }
                    std::copy(out_one.begin(), out_one.end(),
                              denoiser_output.begin() +
                                  static_cast<std::ptrdiff_t>(b) *
                                      static_cast<std::ptrdiff_t>(patch_size));
                }
            }

            for (int32_t b = 0; b < batch; ++b) {
                const auto* sample_patches_ptr =
                    denoiser_output.data() + static_cast<std::size_t>(b) * patch_size;
                std::vector<float> sample_patches(sample_patches_ptr,
                                                  sample_patches_ptr + patch_size);
                unpatchify_2d(sample_patches, layout.z_dim, layout.h_lat, layout.w_lat,
                              sample_noise_pred);
                std::copy(sample_noise_pred.begin(), sample_noise_pred.end(),
                          noise_pred.begin() + static_cast<std::ptrdiff_t>(b) *
                                                   static_cast<std::ptrdiff_t>(latent_size));
            }
            negate_inplace(noise_pred);
            scheduler.step(noise_pred.data(), latents.data(), latents.data(), latents.size(), step);
            log_step_stats(step, num_inference_steps, raw_timestep, latents);
        }

        // VAE decode — per Decision E, always B=1. Route through
        // decode_z_image_result so non-rank-0 ranks return empty results.
        for (int32_t b = 0; b < batch; ++b) {
            const auto* sample_latents_ptr =
                latents.data() + static_cast<std::size_t>(b) * latent_size;
            sample_latents.assign(sample_latents_ptr, sample_latents_ptr + latent_size);

            ImageResult slot;
            slot.height = config_.video_height;
            slot.width = config_.video_width;
            slot.channels = 3;
            slot.num_frames = 1;
            results.push_back(decode_z_image_result(layout.z_dim, layout.h_lat, layout.w_lat,
                                                    sample_latents, slot));
        }

        prompt_offset += static_cast<std::size_t>(batch);
    }

    return results;
}

} // namespace trtmc

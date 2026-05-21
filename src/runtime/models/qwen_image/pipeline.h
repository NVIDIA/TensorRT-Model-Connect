#pragma once

// =============================================================================
// qwen_image_pipeline.h — Qwen-Image diffusion pipeline (T2I + Edit).
// =============================================================================
//
// Mirrors the Python QwenImageDebugRunner in
// tensorrt_model_connect.debug_runner — same forward steps:
//   tokenize -> text encoder (pos + neg) -> seeded latents ->
//   N-step flow-match Euler denoise with true-CFG -> unpatchify ->
//   un-normalize -> VAE decode -> PNG.
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include "runtime/domains/diffusion/qwen_image_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// QwenImagePipeline: Qwen-Image diffusion pipeline with Qwen2.5-VL text
// encoder, MMDiT denoiser, and AutoencoderKLQwenImage VAE. Currently
// supports T2I; Edit-mode engines (vision + VAE encoder) are accepted in
// Construction but remain null for T2I bundles.
class QwenImagePipeline final : public IPipeline {
  public:
    // Construction-time dependencies, populated by the plugin factory.
    // Edit-only engines (vision_engine, vae_encoder) may be null for T2I
    // bundles.
    struct Construction {
        std::unique_ptr<TrtModule> text_engine;
        std::unique_ptr<TrtModule> denoiser_engine;
        std::unique_ptr<TrtModule> vae_decoder_engine;
        std::unique_ptr<TrtModule> vision_engine;      // Edit mode only.
        std::unique_ptr<TrtModule> vae_encoder_engine; // Edit mode only.
        std::shared_ptr<ITokenizer> tokenizer;
        QwenImageConfig config;
        QwenImagePreprocessorWeights preprocessor;
        std::string model_id;
        std::string bundle_path;
    };

    explicit QwenImagePipeline(Construction c);
    ~QwenImagePipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    ImageResult generate_image(const std::string& prompt, const float* image_pixels,
                               int32_t image_height, int32_t image_width,
                               const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "QwenImagePipeline"; }

    // -------------------------------------------------------------------------
    // Math helpers: exposed publicly so unit tests can verify them without
    // needing engine bytes. Consumed by the full generate_image()
    // implementation.
    // -------------------------------------------------------------------------

    // Latent + packed-token dimensions derived from a target image size,
    // using vae.spatial_scale_factor and denoiser.patch_size from config.
    struct LatentShape {
        int latent_h{0};     // height / vae_scale_factor
        int latent_w{0};     // width / vae_scale_factor
        int packed_h{0};     // latent_h / patch_size
        int packed_w{0};     // latent_w / patch_size
        int n_img_tokens{0}; // packed_h * packed_w
    };

    // Compute latent + packed dims for a target image size. Reads vae and
    // denoiser config to derive vae_scale_factor and patch_size. Throws
    // std::runtime_error if either factor is non-positive.
    LatentShape compute_latent_shape(int height, int width) const;

    struct EditImagePlan {
        int output_height{0};
        int output_width{0};
        int condition_height{0}; // Qwen2.5-VL processor image height.
        int condition_width{0};  // Qwen2.5-VL processor image width.
        int vae_height{0};       // Input image height before VAE encode.
        int vae_width{0};        // Input image width before VAE encode.
        LatentShape output_tokens;
        LatentShape condition_tokens;
        int scheduler_image_tokens{0}; // Output-image tokens only.
        int denoiser_image_tokens{0};  // Output + condition tokens.
    };

    // Resolve Edit output/condition image sizes and packed token counts.
    // cfg.height/width override output size only.
    EditImagePlan compute_edit_image_plan(int image_height, int image_width,
                                          const GenerateConfig& cfg = {}) const;

    // Normalize a raw timestep value to the [0, 1] range the engine expects.
    // Qwen-Image scheduler convention: divide by num_train_timesteps (1000).
    // The engine has the full sinusoidal + 2-layer SiLU MLP baked in, so the
    // pipeline only feeds the normalized scalar.
    float normalize_timestep(float scalar_t) const;

    // Seeded host-side standard normal sampling. Produces a flat row-major
    // [1, n_channels, h_lat, w_lat] fp32 buffer (C, H, W). Deterministic for
    // a given seed (std::mt19937 + std::normal_distribution<float>). Does NOT
    // attempt byte-for-byte parity with torch.randn — that requires
    // serializing torch's RNG state and is left for a future task.
    std::vector<float> prepare_initial_latents(int h_lat, int w_lat, int n_channels,
                                               uint64_t seed) const;

    // -------------------------------------------------------------------------
    // Engine-bound methods. Wrap ITrtModule::forward() with the
    // Qwen-Image-specific shapes and the diffusers hardcoded prompt template.
    // -------------------------------------------------------------------------

    // Encoded prompt ready to feed the denoiser. hidden_states is padded to
    // [max_text_tokens, text_embed_dim] row-major; rows past valid_text_len
    // are zero. attention_mask is 1 for valid token rows, 0 for padding.
    struct EncodedPrompt {
        std::vector<float> hidden_states;    // [max_text_tokens * text_embed_dim] fp32
        std::vector<int32_t> attention_mask; // [max_text_tokens]
        int valid_text_len = 0;              // tokens AFTER drop_idx removal
    };

    // Apply the diffusers T2I hardcoded prompt_template_encode, tokenize,
    // pad to text_encoder.max_seq_len, run the text encoder, drop the first
    // prompt_template_drop_idx hidden_state rows, and zero-pad back to
    // denoiser.max_text_tokens × text_embed_dim. Mirrors
    // QwenImageDebugRunner._encode_prompt in debug_runner.py.
    //
    // Throws std::runtime_error on missing engine/tokenizer or when the
    // tokenized template+prompt has ≤ drop_idx valid tokens.
    EncodedPrompt encode_text(const std::string& prompt) const;

    // One denoiser forward pass. latents_packed is [1, n_img, in_channels=64]
    // row-major fp32; hidden_states is the EncodedPrompt.hidden_states blob
    // ([max_text_tokens, text_embed_dim] flat fp32); normalized_t is the
    // already-divided-by-1000 scalar. Returns the predicted noise as
    // [1, n_img, out_channels * patch_size^2 = 64] row-major fp32.
    //
    // attention_mask is accepted for API parity with the Python runner but
    // currently unused: the denoiser engine was baked WITHOUT an
    // encoder_hidden_states_mask input so zero-padded text positions
    // participate in attention (documented deviation).
    std::vector<float> run_denoiser_once(const std::vector<float>& latents_packed,
                                         float normalized_t,
                                         const std::vector<float>& hidden_states,
                                         const std::vector<int32_t>& attention_mask) const;

    // -------------------------------------------------------------------------
    // Denoise loop. Drives the scheduler over `num_steps`, calls
    // run_denoiser_once twice per step when cfg_scale > 1.0 (cond + uncond),
    // combines via Qwen-Image-flavored true-CFG (per-token L2 renormalization),
    // and applies the Euler step in-place. Returns the final packed latents.
    //
    // The latents_packed buffer is consumed by value so the caller does not
    // need its pre-loop state; n_img is passed explicitly because the scheduler
    // needs it for dynamic-mu shifting (image_seq_len argument).
    //
    // Mirrors QwenImageDebugRunner._generate's main loop (debug_runner.py).
    // Caveat: the Python runner ALSO performs per-token L2 renormalization
    //   noise = comb * (||pos|| / max(||comb||, 1e-8))
    // where the norm is taken over the channel axis of each token. We mirror
    // that behavior verbatim — it's a Qwen-Image-specific CFG variant.
    // -------------------------------------------------------------------------
    std::vector<float> denoise_loop_with_cfg(std::vector<float> latents_packed,
                                             const EncodedPrompt& pos, const EncodedPrompt& neg,
                                             int n_img, int num_steps, float cfg_scale) const;

    // -------------------------------------------------------------------------
    // VAE decode. Unpatchify packed latents back to [1, C, 1, h_lat, w_lat],
    // apply per-channel un-normalization (z = z * raw_std + mean), run the
    // VAE decoder engine, and convert the [-1, 1] image into the HWC float32
    // [0, 1] layout used by ImageResult (matches FLUX / Z-Image).
    //
    // Mirrors QwenImageDebugRunner._vae_decode (debug_runner.py).
    // -------------------------------------------------------------------------
    struct DecodedImage {
        std::vector<float> pixels; // [H * W * 3] HWC float32 in [0, 1]
        int height{0};
        int width{0};
    };

    DecodedImage vae_decode(const std::vector<float>& latents_packed, int n_img, int h_lat,
                            int w_lat) const;

  private:
    struct EditInputTensors {
        EditImagePlan plan;
        std::vector<float> condition_pixels_hwc;
        std::vector<float> vae_pixels_ncthw;
    };

    EditInputTensors preprocess_edit_input_image(const float* image_pixels, int32_t image_height,
                                                 int32_t image_width,
                                                 const GenerateConfig& cfg = {}) const;
    EncodedPrompt
    encode_text_with_image_conditioning(const std::string& prompt,
                                        const std::vector<float>& image_features) const;
    std::vector<float> vision_encode_edit_condition(const EditInputTensors& edit_inputs) const;
    std::vector<float> denoise_loop_with_cfg_image_conditioning(
        std::vector<float> latents_packed, const std::vector<float>& condition_latents_packed,
        const EncodedPrompt& pos, const EncodedPrompt& neg, int n_img, int n_condition_img,
        int num_steps, float cfg_scale) const;
    std::vector<float> vae_encode_edit_condition(const EditInputTensors& edit_inputs) const;

    // Patchify a 4D latent tensor [1, C, H, W] (row-major C, H, W) into the
    // packed denoiser-input layout [1, n_img, C * patch_size * patch_size],
    // mirroring diffusers' QwenImagePipeline._pack_latents.
    //
    // Conceptually:
    //   v = latents.reshape(B, C, H/p, p, W/p, p)
    //   v = v.transpose(0, 2, 4, 1, 3, 5)
    //   packed = v.reshape(B, (H/p) * (W/p), C * p * p)
    //
    // Throws std::runtime_error if dimensions aren't divisible by patch_size
    // or buffer sizes don't match.
    static std::vector<float> patchify_latents(const std::vector<float>& latents,
                                               int latent_channels, int h_lat, int w_lat,
                                               int patch_size);

    std::unique_ptr<TrtModule> text_engine_;
    std::unique_ptr<TrtModule> denoiser_engine_;
    std::unique_ptr<TrtModule> vae_decoder_engine_;
    std::unique_ptr<TrtModule> vision_engine_;
    std::unique_ptr<TrtModule> vae_encoder_engine_;
    std::shared_ptr<ITokenizer> tokenizer_;
    QwenImageConfig config_;
    QwenImagePreprocessorWeights preprocessor_;
    std::string model_id_;
    std::string bundle_path_;
};

} // namespace trtmc

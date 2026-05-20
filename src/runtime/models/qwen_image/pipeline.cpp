// =============================================================================
// qwen_image_pipeline.cpp — Qwen-Image diffusion pipeline implementation.
// =============================================================================
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include "runtime/models/qwen_image/pipeline.h"

#include "runtime/domains/diffusion/batch_utils.h"
#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"
#include "trtmc/runtime/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

// Diffusers QwenImagePipeline.prompt_template_encode (T2I path).
// Source: references/diffusers/.../pipeline_qwenimage.py:176.
// The "{}" marker is the user-prompt insertion slot. We split into prefix
// and suffix so the prompt slots in without an std::format call.
inline const char* kPromptTemplatePrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects "
    "and background:<|im_end|>\n<|im_start|>user\n";
inline const char* kPromptTemplateSuffix = "<|im_end|>\n<|im_start|>assistant\n";

} // namespace

QwenImagePipeline::QwenImagePipeline(Construction c)
    : text_engine_(std::move(c.text_engine)), denoiser_engine_(std::move(c.denoiser_engine)),
      vae_decoder_engine_(std::move(c.vae_decoder_engine)),
      vision_engine_(std::move(c.vision_engine)),
      vae_encoder_engine_(std::move(c.vae_encoder_engine)), tokenizer_(std::move(c.tokenizer)),
      config_(std::move(c.config)), preprocessor_(std::move(c.preprocessor)),
      max_dit_batch_size_(std::max(c.max_dit_batch_size, 1)),
      model_id_(std::move(c.model_id)), bundle_path_(std::move(c.bundle_path)) {}

QwenImagePipeline::~QwenImagePipeline() = default;

namespace {

// Bundle of runtime knobs resolved for one generate_image() call. All fields
// have been finalized — either from `cfg` overrides or bundle defaults.
struct GenerateKnobs {
    int num_steps;
    float cfg_scale;
    int height;
    int width;
    uint64_t seed;
    std::string negative;
};

// Resolve runtime knobs from cfg, falling back to bundle defaults.
GenerateKnobs resolve_generate_knobs(const GenerateConfig& cfg, const QwenImageConfig& bundle_cfg) {
    GenerateKnobs k;
    k.num_steps = diffusion::resolve_requested_steps(
        cfg.num_steps, bundle_cfg.diffusion.default_num_inference_steps, true);
    k.cfg_scale = diffusion::resolve_requested_guidance(cfg.guidance_scale,
                                                        bundle_cfg.diffusion.default_cfg_scale);
    k.height = cfg.height > 0 ? cfg.height : bundle_cfg.image.default_height;
    k.width = cfg.width > 0 ? cfg.width : bundle_cfg.image.default_width;
    // mirrors debug_runner default seed=42
    k.seed = cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed) : 42ULL;
    // Negative prompt: empty in cfg means "use bundle default".
    k.negative = cfg.negative_prompt.empty() ? bundle_cfg.diffusion.default_negative_prompt
                                             : cfg.negative_prompt;
    return k;
}

// Validate that all engines required for generate_image() are loaded.
// Throws if any engine is null.
void validate_generate_image_engines(bool has_text, bool has_denoiser, bool has_vae) {
    if (!has_text || !has_denoiser || !has_vae) {
        throw std::runtime_error("QwenImagePipeline::generate_image: required engines (text, "
                                 "denoiser, vae_decoder) must all be loaded");
    }
}

// Validate that the latent shape derived from height/width is positive in
// all dimensions. Throws on any non-positive value.
void validate_generate_image_shape(int latent_h, int latent_w, int n_img_tokens) {
    if (latent_h <= 0 || latent_w <= 0 || n_img_tokens <= 0) {
        throw std::runtime_error("QwenImagePipeline::generate_image: invalid latent shape derived "
                                 "from height/width");
    }
}

// Validate that caller-supplied initial_latents (when non-empty) match the
// expected unpacked [C, h_lat, w_lat] size. Throws on mismatch; no-op for
// the empty case (caller will sample fresh latents).
void validate_caller_initial_latents(const std::vector<float>& initial_latents, int latent_channels,
                                     int latent_h, int latent_w) {
    if (initial_latents.empty()) {
        return;
    }
    const std::size_t expected_unpacked = static_cast<std::size_t>(latent_channels) *
                                          static_cast<std::size_t>(latent_h) *
                                          static_cast<std::size_t>(latent_w);
    if (initial_latents.size() != expected_unpacked) {
        throw std::runtime_error("QwenImagePipeline::generate_image: cfg.initial_latents size " +
                                 std::to_string(initial_latents.size()) +
                                 " does not match expected [1, " + std::to_string(latent_channels) +
                                 ", " + std::to_string(latent_h) + ", " + std::to_string(latent_w) +
                                 "] = " + std::to_string(expected_unpacked));
    }
}

FlowMatchEulerScheduler build_scheduler(const QwenImageDiffusionConfig& dc, int num_steps,
                                        int n_img);
void combine_cfg_with_renorm(const std::vector<float>& noise_pos,
                             const std::vector<float>& noise_neg, float cfg_scale, int n_img,
                             std::size_t channels, std::vector<float>& out);

} // namespace

ImageResult QwenImagePipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    validate_generate_image_engines(text_engine_ != nullptr, denoiser_engine_ != nullptr,
                                    vae_decoder_engine_ != nullptr);

    // 1) Resolve runtime knobs from cfg, falling back to bundle defaults.
    const GenerateKnobs k = resolve_generate_knobs(cfg, config_);

    // 2) Encode positive + negative prompts via the text encoder engine.
    auto pos = encode_text(prompt);
    auto neg = encode_text(k.negative);

    // 3) Derive latent / packed shapes from target image size.
    auto shape = compute_latent_shape(k.height, k.width);
    validate_generate_image_shape(shape.latent_h, shape.latent_w, shape.n_img_tokens);

    // 4) Seed initial latents [C, h_lat, w_lat] then patchify to
    //    [1, n_img, in_channels=64].
    //
    // When the caller supplies cfg.initial_latents (E2E shared-latents path),
    // those bytes are used verbatim instead of the std::mt19937 sample so the
    // C++ and HF reference subprocesses see byte-identical noise. The buffer
    // is the UNPACKED [1, C, h_lat, w_lat] (row-major C, H, W) layout that
    // matches diffusers' randn_tensor((1, 1, C, h_lat, w_lat)) reshape; we
    // patchify it locally just like the seeded path does.
    validate_caller_initial_latents(cfg.initial_latents, config_.vae.latent_channels,
                                    shape.latent_h, shape.latent_w);
    std::vector<float> latents = cfg.initial_latents.empty()
                                     ? prepare_initial_latents(shape.latent_h, shape.latent_w,
                                                               config_.vae.latent_channels, k.seed)
                                     : cfg.initial_latents;
    auto latents_packed = patchify_latents(latents, config_.vae.latent_channels, shape.latent_h,
                                           shape.latent_w, config_.denoiser.patch_size);

    // 5) Run the N-step denoise loop with true-CFG.
    auto denoised = denoise_loop_with_cfg(std::move(latents_packed), pos, neg, shape.n_img_tokens,
                                          k.num_steps, k.cfg_scale);

    // 6) VAE decode -> HWC float pixels in [0, 1].
    auto image = vae_decode(denoised, shape.n_img_tokens, shape.latent_h, shape.latent_w);

    // 7) Package into ImageResult (PNG write is the caller / CLI's job).
    ImageResult result;
    result.height = image.height;
    result.width = image.width;
    result.channels = 3;
    result.num_frames = 1;
    result.pixels = std::move(image.pixels);
    return result;
}

std::vector<ImageResult> QwenImagePipeline::generate_images(
    const std::vector<std::string>& prompts, const std::vector<int32_t>& per_sample_seeds,
    const GenerateConfig& cfg) {
    if (prompts.empty())
        return {};
    if (!per_sample_seeds.empty() && per_sample_seeds.size() != prompts.size()) {
        throw std::invalid_argument("per_sample_seeds size must match prompts size");
    }

    validate_generate_image_engines(text_engine_ != nullptr, denoiser_engine_ != nullptr,
                                    vae_decoder_engine_ != nullptr);
    const GenerateKnobs k = resolve_generate_knobs(cfg, config_);

    std::vector<int32_t> resolved_seeds;
    if (!per_sample_seeds.empty()) {
        resolved_seeds = per_sample_seeds;
    } else if (prompts.size() == 1U) {
        resolved_seeds = {cfg.seed};
    } else {
        resolved_seeds =
            diffusion::derive_per_sample_seeds(k.seed, static_cast<int>(prompts.size()));
    }

    if (!cfg.initial_latents.empty() || max_dit_batch_size_ <= 1 ||
        denoiser_engine_->input_rank("img_patched") != 3) {
        return IPipeline::generate_images(prompts, resolved_seeds, cfg);
    }

    int32_t cap = std::max(max_dit_batch_size_, 1);
    const auto profile_max = denoiser_engine_->input_profile_shape(
        "img_patched", denoiser_engine_->profile_idx(), ProfileShapeSelector::kMax);
    if (!profile_max.empty() && profile_max[0] > 0) {
        cap = std::min(cap, static_cast<int32_t>(profile_max[0]));
    }
    if (cap <= 1) {
        return IPipeline::generate_images(prompts, resolved_seeds, cfg);
    }

    auto shape = compute_latent_shape(k.height, k.width);
    validate_generate_image_shape(shape.latent_h, shape.latent_w, shape.n_img_tokens);
    const int latent_channels = config_.vae.latent_channels;
    const int in_channels = config_.denoiser.in_channels;
    const int max_text_tokens = config_.denoiser.max_text_tokens;
    const int text_embed_dim = config_.denoiser.text_embed_dim;
    const std::size_t packed_size = static_cast<std::size_t>(shape.n_img_tokens) *
                                    static_cast<std::size_t>(in_channels);
    const std::size_t hidden_size = static_cast<std::size_t>(max_text_tokens) *
                                    static_cast<std::size_t>(text_embed_dim);

    const EncodedPrompt neg = encode_text(k.negative);
    std::vector<float> neg_mask(static_cast<std::size_t>(max_text_tokens));
    for (int i = 0; i < max_text_tokens; ++i) {
        neg_mask[static_cast<std::size_t>(i)] =
            neg.attention_mask[static_cast<std::size_t>(i)] != 0 ? 1.0F : 0.0F;
    }
    const bool do_cfg = k.cfg_scale > 1.0F;
    const auto chunks = diffusion::plan_chunks(static_cast<int>(prompts.size()), cap);
    std::vector<ImageResult> results;
    results.reserve(prompts.size());

    auto run_denoiser_batched = [&](const std::vector<float>& latents_packed,
                                    const std::vector<float>& hidden_states,
                                    const std::vector<float>& text_mask,
                                    float normalized_t, int32_t batch) {
        std::vector<float> timestep_buf(static_cast<std::size_t>(batch), normalized_t);
        TensorMap inputs;
        inputs["img_patched"] =
            Tensor{const_cast<float*>(latents_packed.data()),
                   {static_cast<int64_t>(batch), static_cast<int64_t>(shape.n_img_tokens),
                    static_cast<int64_t>(in_channels)},
                   DType::kFloat32};
        inputs["txt_hidden"] =
            Tensor{const_cast<float*>(hidden_states.data()),
                   {static_cast<int64_t>(batch), static_cast<int64_t>(max_text_tokens),
                    static_cast<int64_t>(text_embed_dim)},
                   DType::kFloat32};
        if (denoiser_engine_->has_input("encoder_hidden_states_mask")) {
            inputs["encoder_hidden_states_mask"] =
                Tensor{const_cast<float*>(text_mask.data()),
                       {static_cast<int64_t>(batch), static_cast<int64_t>(max_text_tokens)},
                       DType::kFloat32};
        }
        inputs["timestep"] = Tensor{timestep_buf.data(), {static_cast<int64_t>(batch)},
                                    DType::kFloat32};

        auto outputs = denoiser_engine_->forward(inputs);
        const auto& noise = outputs["noise_patched"];
        const std::size_t expected = static_cast<std::size_t>(batch) * packed_size;
        if (noise.numel() < expected) {
            throw std::runtime_error(
                "QwenImagePipeline::generate_images: batched denoiser output size " +
                std::to_string(noise.numel()) + " does not match expected " +
                std::to_string(expected));
        }
        std::vector<float> out(expected);
        std::memcpy(out.data(), noise.data, expected * sizeof(float));
        return out;
    };

    std::size_t prompt_offset = 0;
    for (int32_t chunk_size : chunks) {
        const int32_t batch = chunk_size;
        std::vector<float> latents_packed(static_cast<std::size_t>(batch) * packed_size);
        std::vector<float> pos_hidden(static_cast<std::size_t>(batch) * hidden_size);
        std::vector<float> neg_hidden(static_cast<std::size_t>(batch) * hidden_size);
        std::vector<float> pos_mask(static_cast<std::size_t>(batch) *
                                    static_cast<std::size_t>(max_text_tokens));
        std::vector<float> neg_mask_batched(static_cast<std::size_t>(batch) *
                                            static_cast<std::size_t>(max_text_tokens));

        for (int32_t b = 0; b < batch; ++b) {
            const std::size_t sample_idx = prompt_offset + static_cast<std::size_t>(b);
            const auto pos = encode_text(prompts[sample_idx]);
            const auto mask_offset = static_cast<std::size_t>(b) *
                                     static_cast<std::size_t>(max_text_tokens);
            std::copy(pos.hidden_states.begin(), pos.hidden_states.end(),
                      pos_hidden.begin() +
                          static_cast<std::ptrdiff_t>(b) *
                              static_cast<std::ptrdiff_t>(hidden_size));
            std::copy(neg.hidden_states.begin(), neg.hidden_states.end(),
                      neg_hidden.begin() +
                          static_cast<std::ptrdiff_t>(b) *
                              static_cast<std::ptrdiff_t>(hidden_size));
            for (int i = 0; i < max_text_tokens; ++i) {
                pos_mask[mask_offset + static_cast<std::size_t>(i)] =
                    pos.attention_mask[static_cast<std::size_t>(i)] != 0 ? 1.0F : 0.0F;
            }
            std::copy(neg_mask.begin(), neg_mask.end(),
                      neg_mask_batched.begin() + static_cast<std::ptrdiff_t>(mask_offset));

            const uint64_t seed = resolved_seeds[sample_idx] >= 0
                                      ? static_cast<uint64_t>(resolved_seeds[sample_idx])
                                      : 42ULL;
            auto latents =
                prepare_initial_latents(shape.latent_h, shape.latent_w, latent_channels, seed);
            auto packed = patchify_latents(latents, latent_channels, shape.latent_h,
                                           shape.latent_w, config_.denoiser.patch_size);
            std::copy(packed.begin(), packed.end(),
                      latents_packed.begin() +
                          static_cast<std::ptrdiff_t>(b) *
                              static_cast<std::ptrdiff_t>(packed_size));
        }

        auto scheduler = build_scheduler(config_.diffusion, k.num_steps, shape.n_img_tokens);
        const auto& timesteps = scheduler.timesteps();
        std::vector<float> noise_pred(static_cast<std::size_t>(batch) * packed_size);

        for (int step = 0; step < k.num_steps; ++step) {
            const float norm_t = normalize_timestep(timesteps[static_cast<std::size_t>(step)]);
            auto noise_pos =
                run_denoiser_batched(latents_packed, pos_hidden, pos_mask, norm_t, batch);
            if (do_cfg) {
                auto noise_neg =
                    run_denoiser_batched(latents_packed, neg_hidden, neg_mask_batched, norm_t,
                                         batch);
                combine_cfg_with_renorm(noise_pos, noise_neg, k.cfg_scale,
                                        batch * shape.n_img_tokens,
                                        static_cast<std::size_t>(in_channels), noise_pred);
            } else {
                noise_pred = std::move(noise_pos);
            }
            scheduler.step(latents_packed.data(), noise_pred.data(),
                           static_cast<int32_t>(latents_packed.size()), step);
        }

        for (int32_t b = 0; b < batch; ++b) {
            const auto* sample_ptr = latents_packed.data() + static_cast<std::size_t>(b) * packed_size;
            std::vector<float> sample(sample_ptr, sample_ptr + packed_size);
            auto image = vae_decode(sample, shape.n_img_tokens, shape.latent_h, shape.latent_w);
            ImageResult result;
            result.height = image.height;
            result.width = image.width;
            result.channels = 3;
            result.num_frames = 1;
            result.pixels = std::move(image.pixels);
            results.push_back(std::move(result));
        }

        prompt_offset += static_cast<std::size_t>(batch);
    }

    return results;
}

// -----------------------------------------------------------------------------
// Math helpers. Exposed publicly for unit-testability; called by the full
// generate_image() implementation.
// -----------------------------------------------------------------------------

QwenImagePipeline::LatentShape QwenImagePipeline::compute_latent_shape(int height,
                                                                       int width) const {
    const int vae_scale = config_.vae.spatial_scale_factor;
    const int patch = config_.denoiser.patch_size;
    if (vae_scale <= 0 || patch <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_latent_shape: invalid vae_scale_factor "
            "or patch_size in config");
    }
    LatentShape s;
    s.latent_h = height / vae_scale;
    s.latent_w = width / vae_scale;
    s.packed_h = s.latent_h / patch;
    s.packed_w = s.latent_w / patch;
    s.n_img_tokens = s.packed_h * s.packed_w;
    return s;
}

float QwenImagePipeline::normalize_timestep(float scalar_t) const {
    // Qwen-Image scheduler: timesteps live in [0, num_train_timesteps]; the
    // engine's baked-in MLP expects the normalized scalar t / 1000.
    const int train_ts = config_.diffusion.num_train_timesteps;
    const float denom = train_ts > 0 ? static_cast<float>(train_ts) : 1000.0F;
    return scalar_t / denom;
}

std::vector<float> QwenImagePipeline::prepare_initial_latents(int h_lat, int w_lat, int n_channels,
                                                              uint64_t seed) const {
    if (h_lat <= 0 || w_lat <= 0 || n_channels <= 0) {
        throw std::runtime_error("QwenImagePipeline::prepare_initial_latents: invalid latent "
                                 "dimensions (h_lat, w_lat, n_channels must all be > 0)");
    }
    const std::size_t total = static_cast<std::size_t>(n_channels) *
                              static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    std::vector<float> out(total);
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = dist(gen);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Engine-bound methods.
// -----------------------------------------------------------------------------

namespace {

// Validate pre-tokenize inputs to encode_text. Throws on any failure.
void validate_encode_text_inputs(bool has_engine, bool has_tokenizer, int max_seq_len, int drop_idx,
                                 int max_text_tokens, int text_embed_dim) {
    if (!has_engine) {
        throw std::runtime_error("QwenImagePipeline::encode_text: text_engine_ is null");
    }
    if (!has_tokenizer) {
        throw std::runtime_error("QwenImagePipeline::encode_text: tokenizer_ is null");
    }
    if (max_seq_len <= 0 || drop_idx < 0 || max_text_tokens <= 0 || text_embed_dim <= 0) {
        throw std::runtime_error("QwenImagePipeline::encode_text: invalid config dimensions");
    }
}

} // namespace

QwenImagePipeline::EncodedPrompt QwenImagePipeline::encode_text(const std::string& prompt) const {
    const int max_seq_len = config_.text_encoder.max_seq_len;
    const int drop_idx = config_.tokenizer.prompt_template_drop_idx;
    const int max_text_tokens = config_.denoiser.max_text_tokens;
    const int text_embed_dim = config_.denoiser.text_embed_dim;

    validate_encode_text_inputs(text_engine_ != nullptr, tokenizer_ != nullptr, max_seq_len,
                                drop_idx, max_text_tokens, text_embed_dim);

    // 1. Wrap the user prompt in the diffusers T2I hardcoded template.
    const std::string templated =
        std::string(kPromptTemplatePrefix) + prompt + std::string(kPromptTemplateSuffix);

    // 2. Tokenize, then pad/truncate to max_seq_len.
    std::vector<int32_t> input_ids = tokenizer_->encode(templated);
    const int raw_token_count = static_cast<int>(input_ids.size());
    if (raw_token_count <= drop_idx) {
        throw std::runtime_error("QwenImagePipeline::encode_text: tokenized prompt has " +
                                 std::to_string(raw_token_count) + " tokens, but drop_idx=" +
                                 std::to_string(drop_idx) + " requires more");
    }

    std::vector<int32_t> padded_ids(static_cast<std::size_t>(max_seq_len), 0);
    const int real_len = std::min(raw_token_count, max_seq_len);
    std::copy_n(input_ids.begin(), real_len, padded_ids.begin());

    // Build the additive attention mask the text encoder engine expects
    // (matches Z-Image / Qwen2.5-VL convention: 0 valid, -1e9 pad).
    std::vector<float> attn_mask_additive(static_cast<std::size_t>(max_seq_len), -1.0e9F);
    for (int i = 0; i < real_len; ++i) {
        attn_mask_additive[static_cast<std::size_t>(i)] = 0.0F;
    }

    // 3. Run text encoder engine.
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{padded_ids.data(), {static_cast<int64_t>(max_seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attn_mask_additive.data(), {static_cast<int64_t>(max_seq_len)}, DType::kFloat32};
    auto outputs = text_engine_->forward(inputs);

    const auto& last_hidden = outputs["last_hidden_state"];
    const auto raw_embed_size =
        static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(text_embed_dim);
    if (last_hidden.numel() != raw_embed_size) {
        throw std::runtime_error("QwenImagePipeline::encode_text: text engine output size " +
                                 std::to_string(last_hidden.numel()) +
                                 " does not match expected max_seq_len * text_embed_dim = " +
                                 std::to_string(raw_embed_size));
    }

    // 4. Drop the first drop_idx rows from the valid prefix; zero-pad to
    //    [max_text_tokens, text_embed_dim].
    EncodedPrompt out;
    out.hidden_states.assign(
        static_cast<std::size_t>(max_text_tokens) * static_cast<std::size_t>(text_embed_dim), 0.0F);
    out.attention_mask.assign(static_cast<std::size_t>(max_text_tokens), 0);

    const int valid_after_drop = std::min(real_len - drop_idx, max_text_tokens);
    out.valid_text_len = valid_after_drop;
    if (valid_after_drop > 0) {
        const float* src =
            static_cast<const float*>(last_hidden.data) +
            static_cast<std::size_t>(drop_idx) * static_cast<std::size_t>(text_embed_dim);
        std::memcpy(out.hidden_states.data(), src,
                    static_cast<std::size_t>(valid_after_drop) *
                        static_cast<std::size_t>(text_embed_dim) * sizeof(float));
        for (int i = 0; i < valid_after_drop; ++i) {
            out.attention_mask[static_cast<std::size_t>(i)] = 1;
        }
    }
    return out;
}

std::vector<float>
QwenImagePipeline::run_denoiser_once(const std::vector<float>& latents_packed, float normalized_t,
                                     const std::vector<float>& hidden_states,
                                     const std::vector<int32_t>& attention_mask) const {
    if (!denoiser_engine_) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: denoiser_engine_ is null");
    }
    (void)attention_mask; // Denoiser engine has no mask input; documented.

    const int in_channels = config_.denoiser.in_channels;
    const int max_text_tokens = config_.denoiser.max_text_tokens;
    const int text_embed_dim = config_.denoiser.text_embed_dim;
    if (in_channels <= 0 || max_text_tokens <= 0 || text_embed_dim <= 0) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: invalid denoiser config");
    }

    if (latents_packed.size() % static_cast<std::size_t>(in_channels) != 0) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: latents_packed size " +
                                 std::to_string(latents_packed.size()) +
                                 " is not divisible by in_channels=" + std::to_string(in_channels));
    }
    const std::size_t expected_hidden_size =
        static_cast<std::size_t>(max_text_tokens) * static_cast<std::size_t>(text_embed_dim);
    if (hidden_states.size() != expected_hidden_size) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: hidden_states size " +
                                 std::to_string(hidden_states.size()) +
                                 " does not match max_text_tokens * text_embed_dim = " +
                                 std::to_string(expected_hidden_size));
    }
    if (attention_mask.size() != static_cast<std::size_t>(max_text_tokens)) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: attention_mask size " +
                                 std::to_string(attention_mask.size()) +
                                 " does not match max_text_tokens = " +
                                 std::to_string(max_text_tokens));
    }

    const int64_t n_img =
        static_cast<int64_t>(latents_packed.size() / static_cast<std::size_t>(in_channels));
    float timestep_buf = normalized_t;
    std::vector<float> encoder_mask(static_cast<std::size_t>(max_text_tokens));
    for (int i = 0; i < max_text_tokens; ++i) {
        encoder_mask[static_cast<std::size_t>(i)] =
            attention_mask[static_cast<std::size_t>(i)] != 0 ? 1.0F : 0.0F;
    }

    TensorMap inputs;
    inputs["img_patched"] = Tensor{const_cast<float*>(latents_packed.data()),
                                   {1, n_img, static_cast<int64_t>(in_channels)},
                                   DType::kFloat32};
    inputs["txt_hidden"] =
        Tensor{const_cast<float*>(hidden_states.data()),
               {1, static_cast<int64_t>(max_text_tokens), static_cast<int64_t>(text_embed_dim)},
               DType::kFloat32};
    if (denoiser_engine_->has_input("encoder_hidden_states_mask")) {
        inputs["encoder_hidden_states_mask"] =
            Tensor{encoder_mask.data(), {1, static_cast<int64_t>(max_text_tokens)},
                   DType::kFloat32};
    }
    inputs["timestep"] = Tensor{&timestep_buf, {1}, DType::kFloat32};

    auto outputs = denoiser_engine_->forward(inputs);
    const auto& noise = outputs["noise_patched"];
    const std::size_t expected = latents_packed.size();
    if (noise.numel() < expected) {
        throw std::runtime_error("QwenImagePipeline::run_denoiser_once: denoiser output size " +
                                 std::to_string(noise.numel()) + " does not match expected " +
                                 std::to_string(expected));
    }
    std::vector<float> result(expected);
    std::memcpy(result.data(), noise.data, expected * sizeof(float));
    return result;
}

// -----------------------------------------------------------------------------
// Denoise loop. Mirrors QwenImageDebugRunner._generate verbatim.
// -----------------------------------------------------------------------------

namespace {

// Validate inputs to denoise_loop_with_cfg. Throws on any failure.
void validate_denoise_loop_inputs(std::size_t latents_size, int n_img, int num_steps,
                                  int in_channels) {
    if (num_steps <= 0) {
        throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: num_steps must be > 0");
    }
    if (n_img <= 0) {
        throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: n_img must be > 0");
    }
    if (in_channels <= 0) {
        throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: invalid in_channels");
    }
    const auto expected = static_cast<std::size_t>(n_img) * static_cast<std::size_t>(in_channels);
    if (latents_size != expected) {
        throw std::runtime_error(
            "QwenImagePipeline::denoise_loop_with_cfg: latents_packed size " +
            std::to_string(latents_size) +
            " does not match n_img * in_channels = " + std::to_string(expected));
    }
}

// Build a FlowMatchEulerScheduler from the bundle's diffusion config and
// set up its timestep schedule for `num_steps` and `n_img` tokens.
FlowMatchEulerScheduler build_scheduler(const QwenImageDiffusionConfig& dc, int num_steps,
                                        int n_img) {
    FlowMatchEulerConfig sc_cfg;
    sc_cfg.num_train_timesteps = dc.num_train_timesteps;
    sc_cfg.shift = dc.shift;
    sc_cfg.use_dynamic_shifting = dc.use_dynamic_shifting;
    sc_cfg.base_shift = dc.base_shift;
    sc_cfg.max_shift = dc.max_shift;
    sc_cfg.base_image_seq_len = dc.base_image_seq_len;
    sc_cfg.max_image_seq_len = dc.max_image_seq_len;
    sc_cfg.shift_terminal = dc.shift_terminal;
    sc_cfg.time_shift_type = dc.time_shift_type;
    FlowMatchEulerScheduler scheduler(sc_cfg);
    scheduler.set_timesteps(num_steps, n_img);
    if (static_cast<int>(scheduler.timesteps().size()) != num_steps) {
        throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: scheduler produced " +
                                 std::to_string(scheduler.timesteps().size()) +
                                 " timesteps, expected " + std::to_string(num_steps));
    }
    return scheduler;
}

// Combine pos + neg noise predictions via true-CFG and apply Qwen-Image's
// per-token L2 renormalization. Matches debug_runner.py exactly:
//   comb = neg + cfg*(pos - neg)
//   noise = comb * (||pos|| / max(||comb||, 1e-8))   per-token
// Writes the result into `out`.
void combine_cfg_with_renorm(const std::vector<float>& noise_pos,
                             const std::vector<float>& noise_neg, float cfg_scale, int n_img,
                             std::size_t channels, std::vector<float>& out) {
    for (int tok = 0; tok < n_img; ++tok) {
        const std::size_t base = static_cast<std::size_t>(tok) * channels;
        double pos_sq = 0.0;
        double comb_sq = 0.0;
        for (std::size_t c = 0; c < channels; ++c) {
            const float p = noise_pos[base + c];
            const float ng = noise_neg[base + c];
            const float comb = ng + cfg_scale * (p - ng);
            out[base + c] = comb;
            pos_sq += static_cast<double>(p) * static_cast<double>(p);
            comb_sq += static_cast<double>(comb) * static_cast<double>(comb);
        }
        const double scale = std::sqrt(pos_sq) / std::max(std::sqrt(comb_sq), 1e-8);
        const float scale_f = static_cast<float>(scale);
        for (std::size_t c = 0; c < channels; ++c) {
            out[base + c] *= scale_f;
        }
    }
}

} // namespace

std::vector<float> QwenImagePipeline::denoise_loop_with_cfg(std::vector<float> latents_packed,
                                                            const EncodedPrompt& pos,
                                                            const EncodedPrompt& neg, int n_img,
                                                            int num_steps, float cfg_scale) const {
    const int in_channels = config_.denoiser.in_channels;
    validate_denoise_loop_inputs(latents_packed.size(), n_img, num_steps, in_channels);

    auto scheduler = build_scheduler(config_.diffusion, num_steps, n_img);
    const auto& timesteps = scheduler.timesteps();

    const bool do_cfg = (cfg_scale > 1.0F);
    const std::size_t numel = latents_packed.size();
    const std::size_t channels = static_cast<std::size_t>(in_channels);
    std::vector<float> noise_pred(numel);

    for (int step = 0; step < num_steps; ++step) {
        const float t = timesteps[static_cast<std::size_t>(step)];
        const float norm_t = normalize_timestep(t);

        auto noise_pos =
            run_denoiser_once(latents_packed, norm_t, pos.hidden_states, pos.attention_mask);
        if (noise_pos.size() != numel) {
            throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: denoiser "
                                     "output size mismatch");
        }

        if (do_cfg) {
            auto noise_neg =
                run_denoiser_once(latents_packed, norm_t, neg.hidden_states, neg.attention_mask);
            if (noise_neg.size() != numel) {
                throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: denoiser "
                                         "output size mismatch");
            }
            combine_cfg_with_renorm(noise_pos, noise_neg, cfg_scale, n_img, channels, noise_pred);
        } else {
            noise_pred = std::move(noise_pos);
        }

        // Euler step in-place: latents += (sigma_next - sigma) * noise.
        scheduler.step(latents_packed.data(), noise_pred.data(),
                       static_cast<int32_t>(latents_packed.size()), step);
    }

    return latents_packed;
}

namespace {

// Validate inputs to patchify_latents. Throws on any failure.
void validate_patchify_inputs(std::size_t latents_size, int latent_channels, int h_lat, int w_lat,
                              int patch_size) {
    if (latent_channels <= 0 || h_lat <= 0 || w_lat <= 0 || patch_size <= 0) {
        throw std::runtime_error("QwenImagePipeline::patchify_latents: invalid dims (channels, "
                                 "h_lat, w_lat, patch_size must all be > 0)");
    }
    if (h_lat % patch_size != 0 || w_lat % patch_size != 0) {
        throw std::runtime_error("QwenImagePipeline::patchify_latents: latent dims " +
                                 std::to_string(h_lat) + "x" + std::to_string(w_lat) +
                                 " not divisible by patch_size=" + std::to_string(patch_size));
    }
    const std::size_t expected = static_cast<std::size_t>(latent_channels) *
                                 static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    if (latents_size != expected) {
        throw std::runtime_error(
            "QwenImagePipeline::patchify_latents: input size " + std::to_string(latents_size) +
            " does not match channels * h_lat * w_lat = " + std::to_string(expected));
    }
}

// Pack a single (ph, pw) tile: copy the [C, p, p] block from `latents`
// (row-major C, H, W) into the packed channel axis at token `tok`.
void pack_one_tile(const std::vector<float>& latents, std::vector<float>& packed, int ph, int pw,
                   int latent_channels, int p, std::size_t stride_c, std::size_t stride_h,
                   std::size_t tok, std::size_t out_channels) {
    for (int c = 0; c < latent_channels; ++c) {
        for (int p1 = 0; p1 < p; ++p1) {
            for (int p2 = 0; p2 < p; ++p2) {
                const std::size_t src = static_cast<std::size_t>(c) * stride_c +
                                        static_cast<std::size_t>(ph * p + p1) * stride_h +
                                        static_cast<std::size_t>(pw * p + p2);
                const std::size_t dst =
                    tok * out_channels + static_cast<std::size_t>(c * p * p + p1 * p + p2);
                packed[dst] = latents[src];
            }
        }
    }
}

} // namespace

std::vector<float> QwenImagePipeline::patchify_latents(const std::vector<float>& latents,
                                                       int latent_channels, int h_lat, int w_lat,
                                                       int patch_size) {
    validate_patchify_inputs(latents.size(), latent_channels, h_lat, w_lat, patch_size);

    const int p = patch_size;
    const int packed_h = h_lat / p;
    const int packed_w = w_lat / p;
    const std::size_t out_channels = static_cast<std::size_t>(latent_channels) *
                                     static_cast<std::size_t>(p) * static_cast<std::size_t>(p);
    const std::size_t n_img =
        static_cast<std::size_t>(packed_h) * static_cast<std::size_t>(packed_w);

    std::vector<float> packed(n_img * out_channels);

    // Source layout: latents[c, h, w] with row-major C, H, W strides.
    //   stride_c = h_lat * w_lat
    //   stride_h = w_lat
    //   stride_w = 1
    // Target layout: packed[ph, pw, c, p1, p2] = latents[c, ph*p+p1, pw*p+p2].
    //   With (ph * packed_w + pw) collapsing to the token axis, and
    //   (c * p * p + p1 * p + p2) collapsing to the channel axis.
    const std::size_t stride_c = static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const std::size_t stride_h = static_cast<std::size_t>(w_lat);
    for (int ph = 0; ph < packed_h; ++ph) {
        for (int pw = 0; pw < packed_w; ++pw) {
            const std::size_t tok =
                static_cast<std::size_t>(ph) * static_cast<std::size_t>(packed_w) +
                static_cast<std::size_t>(pw);
            pack_one_tile(latents, packed, ph, pw, latent_channels, p, stride_c, stride_h, tok,
                          out_channels);
        }
    }
    return packed;
}

namespace {

// Validate the scalar dims/config used by vae_decode. Throws on any failure.
void validate_vae_decode_dims(bool has_engine, int latent_channels, int patch, int vae_scale,
                              int n_img, int h_lat, int w_lat, int packed_h, int packed_w) {
    if (!has_engine) {
        throw std::runtime_error("QwenImagePipeline::vae_decode: vae_decoder_engine_ is null");
    }
    if (latent_channels <= 0 || patch <= 0 || vae_scale <= 0 || n_img <= 0 || h_lat <= 0 ||
        w_lat <= 0) {
        throw std::runtime_error("QwenImagePipeline::vae_decode: invalid dims in config or "
                                 "arguments (all must be > 0)");
    }
    if (packed_h * packed_w != n_img) {
        throw std::runtime_error(
            "QwenImagePipeline::vae_decode: n_img=" + std::to_string(n_img) +
            " does not match packed_h * packed_w = " + std::to_string(packed_h * packed_w));
    }
}

// Validate the buffer sizes used by vae_decode (latents_packed and the
// preprocessor latents_mean/std vectors). Throws on any mismatch.
void validate_vae_decode_buffers(int latent_channels, int patch, int n_img,
                                 std::size_t latents_packed_size, std::size_t latents_mean_size,
                                 std::size_t latents_std_size) {
    const int ch_packed = latent_channels * patch * patch;
    const std::size_t expected_in =
        static_cast<std::size_t>(n_img) * static_cast<std::size_t>(ch_packed);
    if (latents_packed_size != expected_in) {
        throw std::runtime_error("QwenImagePipeline::vae_decode: latents_packed size " +
                                 std::to_string(latents_packed_size) +
                                 " does not match n_img * latent_channels * patch_size^2 = " +
                                 std::to_string(expected_in));
    }
    if (static_cast<int>(latents_mean_size) != latent_channels ||
        static_cast<int>(latents_std_size) != latent_channels) {
        throw std::runtime_error("QwenImagePipeline::vae_decode: preprocessor latents_mean/std "
                                 "missing or wrong size (need " +
                                 std::to_string(latent_channels) + " entries each)");
    }
}

// Unpatchify packed latents [n_img, c * p * p] -> dense [C, H, W] (T=1
// collapses since the source has B=1 and the VAE engine accepts the byte
// layout interchangeably). Inverse of patchify_latents.
//
// Stored row-major: latent_chw[c, h, w] with strides
//   stride_c = h_lat * w_lat, stride_h = w_lat, stride_w = 1.
std::vector<float> unpatchify_latents(const std::vector<float>& packed, int latent_channels,
                                      int patch, int packed_h, int packed_w, int h_lat, int w_lat) {
    const std::size_t per_channel =
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    const int ch_packed = latent_channels * patch * patch;
    std::vector<float> out(static_cast<std::size_t>(latent_channels) * per_channel);
    const std::size_t stride_c = per_channel;
    const std::size_t stride_h = static_cast<std::size_t>(w_lat);
    for (int ph = 0; ph < packed_h; ++ph) {
        for (int pw = 0; pw < packed_w; ++pw) {
            const std::size_t tok =
                static_cast<std::size_t>(ph) * static_cast<std::size_t>(packed_w) +
                static_cast<std::size_t>(pw);
            for (int c = 0; c < latent_channels; ++c) {
                for (int p1 = 0; p1 < patch; ++p1) {
                    for (int p2 = 0; p2 < patch; ++p2) {
                        const std::size_t src =
                            tok * static_cast<std::size_t>(ch_packed) +
                            static_cast<std::size_t>(c * patch * patch + p1 * patch + p2);
                        const std::size_t dst =
                            static_cast<std::size_t>(c) * stride_c +
                            static_cast<std::size_t>(ph * patch + p1) * stride_h +
                            static_cast<std::size_t>(pw * patch + p2);
                        out[dst] = packed[src];
                    }
                }
            }
        }
    }
    return out;
}

// In-place per-channel affine: z[c, *] = z[c, *] * std[c] + mean[c].
// `data` is laid out [C, H, W] row-major; per_channel = H * W.
void unnormalize_latents_inplace(float* data, int latent_channels, std::size_t per_channel,
                                 const std::vector<float>& latents_mean,
                                 const std::vector<float>& latents_std) {
    for (int c = 0; c < latent_channels; ++c) {
        const float m = latents_mean[static_cast<std::size_t>(c)];
        const float s = latents_std[static_cast<std::size_t>(c)];
        float* base = data + static_cast<std::size_t>(c) * per_channel;
        for (std::size_t i = 0; i < per_channel; ++i) {
            base[i] = base[i] * s + m;
        }
    }
}

// Convert VAE output [3, H, W] in [-1, 1] CHW -> [H, W, 3] in [0, 1] HWC,
// clamped to [0, 1]. Matches FluxPipeline / ZImagePipeline conventions.
QwenImagePipeline::DecodedImage chw_to_hwc_unit_range(const float* raw, int h_out, int w_out) {
    QwenImagePipeline::DecodedImage out;
    out.height = h_out;
    out.width = w_out;
    out.pixels.resize(3UL * static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out));
    const std::size_t plane = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    for (int y = 0; y < h_out; ++y) {
        for (int x = 0; x < w_out; ++x) {
            for (int c = 0; c < 3; ++c) {
                const std::size_t src =
                    static_cast<std::size_t>(c) * plane +
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(w_out) +
                    static_cast<std::size_t>(x);
                const std::size_t dst =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(w_out * 3) +
                    static_cast<std::size_t>(x * 3 + c);
                const float v = (raw[src] + 1.0F) * 0.5F;
                out.pixels[dst] = std::max(0.0F, std::min(1.0F, v));
            }
        }
    }
    return out;
}

} // namespace

QwenImagePipeline::DecodedImage
QwenImagePipeline::vae_decode(const std::vector<float>& latents_packed, int n_img, int h_lat,
                              int w_lat) const {
    const int latent_channels = config_.vae.latent_channels;
    const int patch = config_.denoiser.patch_size;
    const int vae_scale = config_.vae.spatial_scale_factor;
    const int packed_h = (patch > 0) ? h_lat / patch : 0;
    const int packed_w = (patch > 0) ? w_lat / patch : 0;
    validate_vae_decode_dims(vae_decoder_engine_ != nullptr, latent_channels, patch, vae_scale,
                             n_img, h_lat, w_lat, packed_h, packed_w);
    validate_vae_decode_buffers(latent_channels, patch, n_img, latents_packed.size(),
                                preprocessor_.latents_mean.size(),
                                preprocessor_.latents_std.size());

    // 1) Unpatchify packed -> dense [C, H, W].
    auto latent_chw = unpatchify_latents(latents_packed, latent_channels, patch, packed_h, packed_w,
                                         h_lat, w_lat);

    // 2) Per-channel un-normalize: z = z * raw_std + mean.
    //    Matches QwenImageDebugRunner._vae_decode: the bundle stores the raw
    //    vae.config.latents_std/mean; diffusers internally inverts to
    //    1/raw_std, so the multiplicative pass collapses to z * raw_std.
    const std::size_t per_channel =
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    unnormalize_latents_inplace(latent_chw.data(), latent_channels, per_channel,
                                preprocessor_.latents_mean, preprocessor_.latents_std);

    // 3) Run VAE decode. Input name is "latent" with shape [1, C, 1, H, W],
    //    matching tensorrt_model_connect.qwen_image_vae_builder. Byte layout
    //    of latent_chw matches NCTHW with T=1 (T axis is contiguous and size 1).
    TensorMap inputs;
    inputs["latent"] = Tensor{latent_chw.data(),
                              {1, static_cast<int64_t>(latent_channels), 1,
                               static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)},
                              DType::kFloat32};
    auto outputs = vae_decoder_engine_->forward(inputs);
    const auto& image_tensor = outputs.at("image");

    // Expected image shape: [1, 3, 1, H, W] or [1, 3, H, W] — numel must be
    // 3 * H * W either way.
    const int h_out = h_lat * vae_scale;
    const int w_out = w_lat * vae_scale;
    const std::size_t expected_out =
        3UL * static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    if (image_tensor.numel() != expected_out) {
        throw std::runtime_error(
            "QwenImagePipeline::vae_decode: VAE output size " +
            std::to_string(image_tensor.numel()) +
            " does not match expected 3 * H * W = " + std::to_string(expected_out));
    }

    // 4) Convert [-1, 1] CHW -> [0, 1] HWC, with clamp.
    return chw_to_hwc_unit_range(static_cast<const float*>(image_tensor.data), h_out, w_out);
}

} // namespace trtmc

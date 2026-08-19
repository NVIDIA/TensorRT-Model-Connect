/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// qwen_image_pipeline.cpp — Qwen-Image diffusion pipeline implementation.
// =============================================================================
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include "pipeline.h"

#include "qwen_image_batch_utils.h"
#include "qwen_image_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
inline const char* kEditPromptTemplatePrefix =
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with "
    "the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
    "Picture 1: <|vision_start|><|image_pad|><|vision_end|>";
inline const char* kEditPromptTemplateSuffix = "<|im_end|>\n<|im_start|>assistant\n";

// Additive-mask sentinel for padded text-encoder positions: large negative so
// softmax drives the attention weight to ~0.
inline constexpr float kAdditiveMaskPad = -1.0e9F;

// Floor used when dividing by the L2 norm of the per-token CFG-combined noise.
inline constexpr double kCfgRenormFloor = 1e-8;

struct ImageSize {
    int height{0};
    int width{0};
};

int floor_to_multiple(int value, int multiple) {
    if (multiple <= 0) {
        throw std::runtime_error("QwenImagePipeline: invalid alignment multiple");
    }
    return (value / multiple) * multiple;
}

ImageSize calculate_aspect_size_from_area(int target_side, int image_height, int image_width,
                                          int alignment) {
    if (target_side <= 0 || image_height <= 0 || image_width <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: image dimensions and target sizes must "
            "be positive");
    }
    if (alignment <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: image alignment must be positive");
    }
    const double ratio = static_cast<double>(image_width) / static_cast<double>(image_height);
    const double target_area = static_cast<double>(target_side) * static_cast<double>(target_side);
    const double raw_width = std::sqrt(target_area * ratio);
    const double raw_height = raw_width / ratio;
    ImageSize out;
    out.width =
        std::max(alignment, static_cast<int>(std::round(raw_width / alignment)) * alignment);
    out.height =
        std::max(alignment, static_cast<int>(std::round(raw_height / alignment)) * alignment);
    return out;
}

int positive_or_default(int value, int fallback) {
    if (value > 0) {
        return value;
    }
    return fallback;
}

void validate_compute_edit_image_plan_inputs(int image_height, int image_width) {
    if (image_height <= 0 || image_width <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: input image dimensions must be positive");
    }
}

void validate_edit_plan_config(const QwenImageConfig& config) {
    if (config.image_conditioning.max_input_images != 1) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: only one input image is supported");
    }
    if (config.image_conditioning.vae_concat_axis != "sequence") {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: only sequence-axis VAE condition "
            "concatenation is supported");
    }
    if (config.vae.spatial_scale_factor <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: invalid vae_scale_factor");
    }
    if (config.denoiser.patch_size <= 0) {
        throw std::runtime_error("QwenImagePipeline::compute_edit_image_plan: invalid patch_size");
    }
}

ImageSize resolve_edit_output_size(const QwenImageConfig& config, const GenerateConfig& cfg,
                                   int image_height, int image_width, int alignment) {
    const auto default_output = calculate_aspect_size_from_area(
        config.image_conditioning.vae_image_size, image_height, image_width, 32);
    ImageSize output;
    output.height = positive_or_default(cfg.height, default_output.height);
    output.width = positive_or_default(cfg.width, default_output.width);
    output.height = floor_to_multiple(output.height, alignment);
    output.width = floor_to_multiple(output.width, alignment);
    if (output.height <= 0 || output.width <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::compute_edit_image_plan: output dimensions align to zero");
    }
    return output;
}

ImageSize resolve_edit_condition_size(const QwenImageConfig& config, int image_height,
                                      int image_width) {
    if (config.vision_encoder.image_height > 0 && config.vision_encoder.image_width > 0) {
        return ImageSize{config.vision_encoder.image_height, config.vision_encoder.image_width};
    }
    const int vl_alignment = config.vision_encoder.patch_size * config.vision_encoder.merge_size;
    return calculate_aspect_size_from_area(config.image_conditioning.vl_image_size, image_height,
                                           image_width, vl_alignment);
}

ImageSize resolve_edit_vae_size(const QwenImageConfig& config) {
    ImageSize vae;
    vae.height = positive_or_default(config.image_conditioning.vae_image_height,
                                     config.image_conditioning.vae_image_size);
    vae.width = positive_or_default(config.image_conditioning.vae_image_width,
                                    config.image_conditioning.vae_image_size);
    return vae;
}

void validate_text_to_image_overload_call(const float* image_pixels, int32_t image_height,
                                          int32_t image_width) {
    if (image_pixels == nullptr && image_height <= 0 && image_width <= 0) {
        return;
    }
    throw std::runtime_error(
        "QwenImagePipeline::generate_image: text-to-image Qwen-Image bundles do not "
        "accept an input image");
}

void validate_edit_generate_input_image(const float* image_pixels, int32_t image_height,
                                        int32_t image_width) {
    if (image_pixels == nullptr) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image: Qwen-Image Edit requires a non-empty input image");
    }
    if (image_height <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image: Qwen-Image Edit requires a non-empty input image");
    }
    if (image_width <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image: Qwen-Image Edit requires a non-empty input image");
    }
}

void validate_edit_runtime_complete(bool has_text, bool has_denoiser, bool has_vae_decoder,
                                    bool has_vision, bool has_vae_encoder) {
    if (has_text && has_denoiser && has_vae_decoder && has_vision && has_vae_encoder) {
        return;
    }
    throw std::runtime_error(
        "QwenImagePipeline::generate_image: Qwen-Image Edit runtime is not complete; "
        "required engines are text, vision, denoiser, vae_encoder, and vae_decoder");
}

} // namespace

QwenImagePipeline::QwenImagePipeline(Construction c)
    : text_engine_(std::move(c.text_engine)), denoiser_engine_(std::move(c.denoiser_engine)),
      vae_decoder_engine_(std::move(c.vae_decoder_engine)),
      vision_engine_(std::move(c.vision_engine)),
      vae_encoder_engine_(std::move(c.vae_encoder_engine)), tokenizer_(std::move(c.tokenizer)),
      config_(std::move(c.config)), preprocessor_(std::move(c.preprocessor)),
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

int resolve_qwen_image_requested_steps(int requested, int fallback) {
    if (requested < 0) {
        return fallback;
    }
    if (requested == 0) {
        return fallback;
    }
    return requested;
}

float resolve_qwen_image_requested_guidance(float requested, float fallback) {
    return requested < 0.0F ? fallback : requested;
}

// Resolve runtime knobs from cfg, falling back to bundle defaults.
GenerateKnobs resolve_generate_knobs(const GenerateConfig& cfg, const QwenImageConfig& bundle_cfg) {
    GenerateKnobs k;
    k.num_steps = resolve_qwen_image_requested_steps(
        cfg.num_steps, bundle_cfg.diffusion.default_num_inference_steps);
    k.cfg_scale = resolve_qwen_image_requested_guidance(cfg.guidance_scale,
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

} // namespace

ImageResult QwenImagePipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    if (config_.task_mode == QwenImageTaskMode::Edit) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image: Edit bundles require an input image; "
            "call generate_image(prompt, image_pixels, image_height, image_width, cfg)");
    }

    // Preserve legacy default-seed behaviour: when ``cfg.seed`` is unset
    // (``< 0``), pass the historical default (42) verbatim instead of routing
    // through ``derive_per_sample_seeds``. That way default-seed golden
    // images are bit-identical to pre-PR-2. The single-sample call is
    // delegated to ``generate_image_batch`` so the single-prompt and batched
    // codepaths can never drift (Decision D).
    const std::uint32_t per_sample_seed =
        (cfg.seed >= 0) ? static_cast<std::uint32_t>(cfg.seed) : 42U;
    auto batch = generate_image_batch({prompt}, {per_sample_seed}, cfg);
    if (batch.empty()) {
        return ImageResult{};
    }
    return std::move(batch.front());
}

ImageResult QwenImagePipeline::generate_image(const std::string& prompt, const float* image_pixels,
                                              int32_t image_height, int32_t image_width,
                                              const GenerateConfig& cfg) {
    if (config_.task_mode != QwenImageTaskMode::Edit) {
        validate_text_to_image_overload_call(image_pixels, image_height, image_width);
        return generate_image(prompt, cfg);
    }

    validate_edit_generate_input_image(image_pixels, image_height, image_width);
    validate_edit_runtime_complete(text_engine_ != nullptr, denoiser_engine_ != nullptr,
                                   vae_decoder_engine_ != nullptr, vision_engine_ != nullptr,
                                   vae_encoder_engine_ != nullptr);

    const EditInputTensors edit_inputs =
        preprocess_edit_input_image(image_pixels, image_height, image_width, cfg);
    const auto condition_latents_packed = vae_encode_edit_condition(edit_inputs);
    const auto image_features = vision_encode_edit_condition(edit_inputs);

    const GenerateKnobs k = resolve_generate_knobs(cfg, config_);
    auto pos = encode_text_with_image_conditioning(prompt, image_features);
    auto neg = encode_text_with_image_conditioning(k.negative, image_features);

    auto shape = compute_latent_shape(k.height, k.width);
    validate_generate_image_shape(shape.latent_h, shape.latent_w, shape.n_img_tokens);
    validate_caller_initial_latents(cfg.initial_latents, config_.vae.latent_channels,
                                    shape.latent_h, shape.latent_w);

    std::vector<float> latents = cfg.initial_latents.empty()
                                     ? prepare_initial_latents(shape.latent_h, shape.latent_w,
                                                               config_.vae.latent_channels, k.seed)
                                     : cfg.initial_latents;
    auto latents_packed = patchify_latents(latents, config_.vae.latent_channels, shape.latent_h,
                                           shape.latent_w, config_.denoiser.patch_size);

    auto denoised = denoise_loop_with_cfg(std::move(latents_packed), pos, neg, shape.n_img_tokens,
                                          k.num_steps, k.cfg_scale, condition_latents_packed,
                                          edit_inputs.plan.condition_tokens.n_img_tokens);

    auto image = vae_decode(denoised, shape.n_img_tokens, shape.latent_h, shape.latent_w);

    ImageResult result;
    result.height = image.height;
    result.width = image.width;
    result.channels = 3;
    result.num_frames = 1;
    result.pixels = std::move(image.pixels);
    return result;
}

// ===========================================================================
// generate_image_batch — Decisions B/D/E (RFC 2026-05-11)
//
// Decision B: TWO denoiser forwards per step (cond + uncond), combined by
//             the existing per-token L2-renorm helper. No fusion in Phase 1.
// Decision D: silently chunk against ``config_.max_batch_size.dit`` via
//             ``qwen_image_batch::plan_chunks``.
// Decision E: VAE always slices at B=1 — one sequential decode per sample.
//
// Per-sample seeds: the caller hands us one ``per_sample_seeds`` entry per
// prompt. We never broadcast a single RNG stream across samples (lesson #174
// from AnimateDiff: shared noise leads to mode collapse). The hot-path
// within a chunk seeds each sample independently via
// ``prepare_initial_latents``.
//
// Encoder: PR-2 keeps the encoder sequential — the Qwen2.5-VL encoder
// builder remains static-batch in PR-1. The DiT batched path stacks the
// per-sample encoded prompts into one contiguous [B, max_text, D] blob just
// before the denoise loop. Batched-encoder upgrade is a Phase 1.5 follow-up.
//
// Edit mode: out of scope for PR 2.
// ===========================================================================

namespace {

// Stack per-sample encoded prompt hidden_states into one contiguous
// [B * max_text_tokens * text_embed_dim] buffer. The DiT engine (when built
// with max_batch_size > 1) reads this as [B, max_text_tokens, text_embed_dim].
std::vector<float>
stack_encoded_prompts(const std::vector<QwenImagePipeline::EncodedPrompt>& encoded,
                      int max_text_tokens, int text_embed_dim) {
    const std::size_t per_sample =
        static_cast<std::size_t>(max_text_tokens) * static_cast<std::size_t>(text_embed_dim);
    std::vector<float> out(encoded.size() * per_sample);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i].hidden_states.size() != per_sample) {
            throw std::runtime_error(
                "QwenImagePipeline::generate_image_batch: per-sample hidden_states size "
                "does not match max_text_tokens * text_embed_dim");
        }
        std::memcpy(out.data() + i * per_sample, encoded[i].hidden_states.data(),
                    per_sample * sizeof(float));
    }
    return out;
}

// Bundle of shape constants needed by the batched denoise loop. Computed
// once per chunk so the per-step helpers only see derived quantities.
struct BatchedChunkShape {
    int B;
    int n_img;
    int max_text;
    int txt_d;
    int in_ch;
    int out_ch_packed;
    std::size_t per_sample_packed;
    std::size_t batched_numel;
};

BatchedChunkShape make_batched_chunk_shape(const QwenImageConfig& config, int n_img, int B) {
    BatchedChunkShape s{};
    s.B = B;
    s.n_img = n_img;
    s.max_text = config.denoiser.max_text_tokens;
    s.txt_d = config.denoiser.text_embed_dim;
    s.in_ch = config.denoiser.in_channels;
    s.out_ch_packed =
        config.denoiser.out_channels * config.denoiser.patch_size * config.denoiser.patch_size;
    s.per_sample_packed = static_cast<std::size_t>(n_img) * static_cast<std::size_t>(s.in_ch);
    s.batched_numel = static_cast<std::size_t>(B) * static_cast<std::size_t>(n_img) *
                      static_cast<std::size_t>(s.out_ch_packed);
    return s;
}

// Build the flow-match-Euler scheduler used by the batched denoise loop.
// Mirrors the single-sample ``denoise_loop_with_cfg`` helper but is local
// to the batched path because ``build_scheduler`` is defined further down
// in this translation unit.
FlowMatchEulerScheduler build_batched_scheduler(const QwenImageDiffusionConfig& dc, int num_steps,
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
    return scheduler;
}

// Run one batched DiT forward (cond OR uncond depending on which hidden_batch
// is passed). ``out_buf`` is sized to ``batched_numel`` by the caller and
// written in-place.
void dit_forward_batched(TrtModule& engine, float* latents_batch, float* hidden_batch,
                         float* timestep_buf, const BatchedChunkShape& s,
                         std::vector<float>& out_buf, const char* pass_label) {
    TensorMap inputs;
    inputs["img_patched"] = Tensor{
        latents_batch,
        {static_cast<int64_t>(s.B), static_cast<int64_t>(s.n_img), static_cast<int64_t>(s.in_ch)},
        DType::kFloat32};
    inputs["txt_hidden"] = Tensor{hidden_batch,
                                  {static_cast<int64_t>(s.B), static_cast<int64_t>(s.max_text),
                                   static_cast<int64_t>(s.txt_d)},
                                  DType::kFloat32};
    inputs["timestep"] = Tensor{timestep_buf, {static_cast<int64_t>(s.B)}, DType::kFloat32};
    auto outputs = engine.forward(inputs);
    const auto& noise = outputs.at("noise_patched");
    if (noise.numel() != s.batched_numel) {
        throw std::runtime_error(std::string("QwenImagePipeline::generate_image_batch: ") +
                                 pass_label + " denoiser output size " +
                                 std::to_string(noise.numel()) +
                                 " != B*n_img*out_ch_packed = " + std::to_string(s.batched_numel));
    }
    std::memcpy(out_buf.data(), noise.data, s.batched_numel * sizeof(float));
}

// Apply one per-sample Euler step across the batch. The scheduler is
// shape-agnostic, so we slice the contiguous per-sample views and call
// ``step`` once per slot.
void apply_per_sample_euler_step(FlowMatchEulerScheduler& scheduler, float* latents_batch,
                                 const float* noise_for_scheduler, std::size_t per_sample_packed,
                                 int B, int step_idx) {
    for (int b = 0; b < B; ++b) {
        float* per_latents = latents_batch + static_cast<std::size_t>(b) * per_sample_packed;
        const float* per_noise =
            noise_for_scheduler + static_cast<std::size_t>(b) * per_sample_packed;
        scheduler.step(per_latents, per_noise, static_cast<int32_t>(per_sample_packed), step_idx);
    }
}

// Wrap a single packed-latent slot as an ``ImageResult``. Extracted so both
// the single-sample and batched paths produce results the same way.
ImageResult wrap_decoded_as_image_result(QwenImagePipeline::DecodedImage&& image) {
    ImageResult one;
    one.height = image.height;
    one.width = image.width;
    one.channels = 3;
    one.num_frames = 1;
    one.pixels = std::move(image.pixels);
    return one;
}

} // namespace

ImageResult QwenImagePipeline::generate_image_batch_single_sample_chunk(
    const std::string& prompt, std::uint32_t seed, const GenerateConfig& cfg,
    const LatentShape& shape, int num_steps, float cfg_scale, const std::string& negative_prompt) {
    // Single-sample fast path — preserves the legacy byte-for-byte
    // behaviour. The caller-supplied ``cfg`` is forwarded so that the
    // optional ``cfg.initial_latents`` override flows through.
    auto pos = encode_text(prompt);
    auto neg = encode_text(negative_prompt);

    validate_caller_initial_latents(cfg.initial_latents, config_.vae.latent_channels,
                                    shape.latent_h, shape.latent_w);
    std::vector<float> latents =
        cfg.initial_latents.empty()
            ? prepare_initial_latents(shape.latent_h, shape.latent_w, config_.vae.latent_channels,
                                      static_cast<std::uint64_t>(seed))
            : cfg.initial_latents;
    auto latents_packed = patchify_latents(latents, config_.vae.latent_channels, shape.latent_h,
                                           shape.latent_w, config_.denoiser.patch_size);

    auto denoised = denoise_loop_with_cfg(std::move(latents_packed), pos, neg, shape.n_img_tokens,
                                          num_steps, cfg_scale);
    auto image = vae_decode(denoised, shape.n_img_tokens, shape.latent_h, shape.latent_w);
    return wrap_decoded_as_image_result(std::move(image));
}

std::vector<ImageResult> QwenImagePipeline::generate_image_batch_chunk(
    const std::vector<std::string>& prompts, std::size_t chunk_begin, int chunk_size,
    const std::vector<std::uint32_t>& per_sample_seeds, const LatentShape& shape, int num_steps,
    float cfg_scale, const std::string& negative_prompt) {
    // ----------------- Batched path (chunk_size > 1) -----------------
    //
    // Requires denoiser_engine_ built with max_batch_size > 1 (PR-1
    // builder path via add_dynamic_batch_profile).
    const BatchedChunkShape s = make_batched_chunk_shape(config_, shape.n_img_tokens, chunk_size);
    const std::size_t chunk_end = chunk_begin + static_cast<std::size_t>(chunk_size);

    // ---- Encode positive prompts per sample, then tile the negative.
    std::vector<EncodedPrompt> pos_encoded;
    pos_encoded.reserve(static_cast<std::size_t>(s.B));
    for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
        pos_encoded.push_back(encode_text(prompts[i]));
    }
    // Single negative prompt tiled B-times (matches existing pipeline.cpp
    // behavior — ``encode_text(k.negative)`` once).
    const EncodedPrompt neg_template = encode_text(negative_prompt);
    std::vector<EncodedPrompt> neg_encoded(static_cast<std::size_t>(s.B), neg_template);

    std::vector<float> pos_hidden_batch = stack_encoded_prompts(pos_encoded, s.max_text, s.txt_d);
    std::vector<float> neg_hidden_batch = stack_encoded_prompts(neg_encoded, s.max_text, s.txt_d);

    // ---- Per-sample initial latents (per-sample RNG).
    std::vector<float> latents_packed_batch(static_cast<std::size_t>(s.B) * s.per_sample_packed,
                                            0.0F);
    for (int b = 0; b < s.B; ++b) {
        auto one_unpacked = prepare_initial_latents(
            shape.latent_h, shape.latent_w, config_.vae.latent_channels,
            static_cast<std::uint64_t>(
                per_sample_seeds[chunk_begin + static_cast<std::size_t>(b)]));
        auto one_packed =
            patchify_latents(one_unpacked, config_.vae.latent_channels, shape.latent_h,
                             shape.latent_w, config_.denoiser.patch_size);
        if (one_packed.size() != s.per_sample_packed) {
            throw std::runtime_error(
                "QwenImagePipeline::generate_image_batch: patchify produced unexpected "
                "per-sample size");
        }
        std::memcpy(latents_packed_batch.data() + static_cast<std::size_t>(b) * s.per_sample_packed,
                    one_packed.data(), s.per_sample_packed * sizeof(float));
    }

    // ---- Scheduler state: timesteps are batch-invariant (read-only after
    // set_timesteps), so one scheduler stepped on per-sample views is
    // sufficient.
    FlowMatchEulerScheduler flow_scheduler =
        build_batched_scheduler(config_.diffusion, num_steps, s.n_img);

    const bool do_cfg = (cfg_scale > 1.0F);
    std::vector<float> noise_pos_buf(s.batched_numel);
    std::vector<float> noise_neg_buf;
    std::vector<float> noise_pred_buf;
    if (do_cfg) {
        noise_neg_buf.resize(s.batched_numel);
        noise_pred_buf.resize(s.batched_numel);
    }
    std::vector<float> timestep_buf(static_cast<std::size_t>(s.B));

    // ---- Denoise loop: two DiT forwards per step + per-token L2 renorm.
    const auto& timesteps = flow_scheduler.timesteps();
    for (int step = 0; step < num_steps; ++step) {
        const float norm_t = normalize_timestep(timesteps[static_cast<std::size_t>(step)]);
        std::fill(timestep_buf.begin(), timestep_buf.end(), norm_t);

        dit_forward_batched(*denoiser_engine_, latents_packed_batch.data(), pos_hidden_batch.data(),
                            timestep_buf.data(), s, noise_pos_buf, "positive");

        const float* noise_for_scheduler = noise_pos_buf.data();
        if (do_cfg) {
            dit_forward_batched(*denoiser_engine_, latents_packed_batch.data(),
                                neg_hidden_batch.data(), timestep_buf.data(), s, noise_neg_buf,
                                "negative");
            // Per-token L2 renorm — reduction is strictly per-token over
            // the channel axis (see ``combine_cfg_with_renorm`` audit
            // comment). Passing ``B * n_img`` as ``n_tokens`` gives one
            // independent renorm per (sample, token), so sample i never
            // pulls signal from sample j.
            combine_cfg_with_renorm(noise_pos_buf, noise_neg_buf, cfg_scale, s.B * s.n_img,
                                    static_cast<std::size_t>(s.out_ch_packed), noise_pred_buf);
            noise_for_scheduler = noise_pred_buf.data();
        }

        apply_per_sample_euler_step(flow_scheduler, latents_packed_batch.data(),
                                    noise_for_scheduler, s.per_sample_packed, s.B, step);
    }

    // ---- VAE always slices at B=1 (Decision E).
    std::vector<ImageResult> results;
    results.reserve(static_cast<std::size_t>(s.B));
    for (int b = 0; b < s.B; ++b) {
        std::vector<float> one_packed(s.per_sample_packed);
        std::memcpy(one_packed.data(),
                    latents_packed_batch.data() + static_cast<std::size_t>(b) * s.per_sample_packed,
                    s.per_sample_packed * sizeof(float));
        auto image = vae_decode(one_packed, shape.n_img_tokens, shape.latent_h, shape.latent_w);
        results.push_back(wrap_decoded_as_image_result(std::move(image)));
    }
    return results;
}

std::vector<ImageResult>
QwenImagePipeline::generate_image_batch(const std::vector<std::string>& prompts,
                                        const std::vector<std::uint32_t>& per_sample_seeds,
                                        const GenerateConfig& cfg) {
    if (prompts.size() != per_sample_seeds.size()) {
        throw std::invalid_argument(
            "QwenImagePipeline::generate_image_batch: prompts.size() must equal "
            "per_sample_seeds.size()");
    }
    if (prompts.empty()) {
        return {};
    }
    if (config_.task_mode == QwenImageTaskMode::Edit) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image_batch: Edit bundles require an input image; "
            "batched Edit generation is not supported in PR 2");
    }
    // ``cfg.initial_latents`` is reserved for the single-prompt
    // golden-parity harness (one buffer, [1, C, h_lat, w_lat]). Reject if
    // present alongside a multi-sample call — shared-noise batches are a
    // Phase 1.5 follow-up.
    if (!cfg.initial_latents.empty() && prompts.size() > 1) {
        throw std::runtime_error(
            "QwenImagePipeline::generate_image_batch: cfg.initial_latents is only supported "
            "for single-sample calls (PR 2 scope)");
    }

    validate_generate_image_engines(text_engine_ != nullptr, denoiser_engine_ != nullptr,
                                    vae_decoder_engine_ != nullptr);

    const GenerateKnobs k_template = resolve_generate_knobs(cfg, config_);
    const auto shape = compute_latent_shape(k_template.height, k_template.width);
    validate_generate_image_shape(shape.latent_h, shape.latent_w, shape.n_img_tokens);

    const int32_t total = static_cast<int32_t>(prompts.size());
    const int32_t cap = std::max(1, config_.max_batch_size.dit);
    const auto chunks = qwen_image_batch::plan_chunks(total, cap);

    std::vector<ImageResult> results;
    results.reserve(prompts.size());

    std::size_t cursor = 0;
    for (int chunk_size : chunks) {
        if (chunk_size == 1) {
            results.push_back(generate_image_batch_single_sample_chunk(
                prompts[cursor], per_sample_seeds[cursor], cfg, shape, k_template.num_steps,
                k_template.cfg_scale, k_template.negative));
        } else {
            auto chunk_results = generate_image_batch_chunk(
                prompts, cursor, chunk_size, per_sample_seeds, shape, k_template.num_steps,
                k_template.cfg_scale, k_template.negative);
            for (auto& r : chunk_results) {
                results.push_back(std::move(r));
            }
        }
        cursor += static_cast<std::size_t>(chunk_size);
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

QwenImagePipeline::EditImagePlan
QwenImagePipeline::compute_edit_image_plan(int image_height, int image_width,
                                           const GenerateConfig& cfg) const {
    validate_compute_edit_image_plan_inputs(image_height, image_width);
    validate_edit_plan_config(config_);

    const int vae_scale = config_.vae.spatial_scale_factor;
    const int patch = config_.denoiser.patch_size;
    const int output_alignment = vae_scale * patch;

    const auto output =
        resolve_edit_output_size(config_, cfg, image_height, image_width, output_alignment);
    const auto condition = resolve_edit_condition_size(config_, image_height, image_width);
    const auto vae = resolve_edit_vae_size(config_);

    EditImagePlan plan;
    plan.output_height = output.height;
    plan.output_width = output.width;
    plan.condition_height = condition.height;
    plan.condition_width = condition.width;
    plan.vae_height = vae.height;
    plan.vae_width = vae.width;
    plan.output_tokens = compute_latent_shape(plan.output_height, plan.output_width);
    plan.condition_tokens = compute_latent_shape(plan.vae_height, plan.vae_width);
    plan.scheduler_image_tokens = plan.output_tokens.n_img_tokens;
    plan.denoiser_image_tokens =
        plan.output_tokens.n_img_tokens + plan.condition_tokens.n_img_tokens;
    return plan;
}

namespace {

void validate_edit_preprocess_inputs(const float* image_pixels, int32_t image_height,
                                     int32_t image_width) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::runtime_error("QwenImagePipeline::preprocess_edit_input_image: requires a "
                                 "non-empty input image");
    }
}

void validate_resize_target(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::runtime_error(
            "QwenImagePipeline::preprocess_edit_input_image: target dimensions must be positive");
    }
}

std::size_t hwc_index(int y, int x, int c, int width) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)) *
               3UL +
           static_cast<std::size_t>(c);
}

float clamp_unit_image_value(float value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "QwenImagePipeline::preprocess_edit_input_image: image pixels must be finite");
    }
    return std::max(0.0F, std::min(1.0F, value));
}

float sample_hwc_bilinear_unit(const float* pixels, int src_h, int src_w, int dst_y, int dst_x,
                               int dst_h, int dst_w, int channel) {
    const double src_y = (static_cast<double>(dst_y) + 0.5) * static_cast<double>(src_h) /
                             static_cast<double>(dst_h) -
                         0.5;
    const double src_x = (static_cast<double>(dst_x) + 0.5) * static_cast<double>(src_w) /
                             static_cast<double>(dst_w) -
                         0.5;
    const double y = std::max(0.0, std::min(src_y, static_cast<double>(src_h - 1)));
    const double x = std::max(0.0, std::min(src_x, static_cast<double>(src_w - 1)));
    const int y0 = static_cast<int>(std::floor(y));
    const int x0 = static_cast<int>(std::floor(x));
    const int y1 = std::min(y0 + 1, src_h - 1);
    const int x1 = std::min(x0 + 1, src_w - 1);
    const double wy = y - static_cast<double>(y0);
    const double wx = x - static_cast<double>(x0);

    const float p00 = clamp_unit_image_value(pixels[hwc_index(y0, x0, channel, src_w)]);
    const float p01 = clamp_unit_image_value(pixels[hwc_index(y0, x1, channel, src_w)]);
    const float p10 = clamp_unit_image_value(pixels[hwc_index(y1, x0, channel, src_w)]);
    const float p11 = clamp_unit_image_value(pixels[hwc_index(y1, x1, channel, src_w)]);

    const double top = static_cast<double>(p00) * (1.0 - wx) + static_cast<double>(p01) * wx;
    const double bottom = static_cast<double>(p10) * (1.0 - wx) + static_cast<double>(p11) * wx;
    return static_cast<float>(top * (1.0 - wy) + bottom * wy);
}

std::vector<float> resize_hwc_unit(const float* pixels, int src_h, int src_w, int dst_h,
                                   int dst_w) {
    validate_resize_target(dst_h, dst_w);
    std::vector<float> out(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w) * 3UL);
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            for (int c = 0; c < 3; ++c) {
                out[hwc_index(y, x, c, dst_w)] =
                    sample_hwc_bilinear_unit(pixels, src_h, src_w, y, x, dst_h, dst_w, c);
            }
        }
    }
    return out;
}

std::vector<float> resize_hwc_unit_to_ncthw_minus1_1(const float* pixels, int src_h, int src_w,
                                                     int dst_h, int dst_w) {
    validate_resize_target(dst_h, dst_w);
    const std::size_t plane = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
    std::vector<float> out(3UL * plane);
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const std::size_t pixel_index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                static_cast<std::size_t>(x);
            for (int c = 0; c < 3; ++c) {
                const float unit =
                    sample_hwc_bilinear_unit(pixels, src_h, src_w, y, x, dst_h, dst_w, c);
                out[static_cast<std::size_t>(c) * plane + pixel_index] = unit * 2.0F - 1.0F;
            }
        }
    }
    return out;
}

} // namespace

QwenImagePipeline::EditInputTensors
QwenImagePipeline::preprocess_edit_input_image(const float* image_pixels, int32_t image_height,
                                               int32_t image_width,
                                               const GenerateConfig& cfg) const {
    validate_edit_preprocess_inputs(image_pixels, image_height, image_width);

    EditInputTensors out;
    out.plan = compute_edit_image_plan(image_height, image_width, cfg);
    out.condition_pixels_hwc = resize_hwc_unit(image_pixels, image_height, image_width,
                                               out.plan.condition_height, out.plan.condition_width);
    out.vae_pixels_ncthw = resize_hwc_unit_to_ncthw_minus1_1(
        image_pixels, image_height, image_width, out.plan.vae_height, out.plan.vae_width);
    return out;
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

std::vector<float> make_additive_attention_mask(int max_seq_len, int valid_len) {
    std::vector<float> mask(static_cast<std::size_t>(max_seq_len), kAdditiveMaskPad);
    for (int i = 0; i < valid_len; ++i) {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }
    return mask;
}

QwenImagePipeline::EncodedPrompt
pack_prompt_hidden_after_drop(const Tensor& last_hidden, int valid_token_count, int drop_idx,
                              int max_text_tokens, int text_embed_dim, const char* caller) {
    const auto raw_embed_size =
        static_cast<std::size_t>(last_hidden.shape.empty() ? 0 : last_hidden.shape[0]) *
        static_cast<std::size_t>(text_embed_dim);
    if (last_hidden.numel() != raw_embed_size) {
        throw std::runtime_error(std::string(caller) + ": text engine output size " +
                                 std::to_string(last_hidden.numel()) +
                                 " does not match expected max_seq_len * text_embed_dim = " +
                                 std::to_string(raw_embed_size));
    }

    QwenImagePipeline::EncodedPrompt out;
    out.hidden_states.assign(
        static_cast<std::size_t>(max_text_tokens) * static_cast<std::size_t>(text_embed_dim), 0.0F);
    out.attention_mask.assign(static_cast<std::size_t>(max_text_tokens), 0);

    const int valid_after_drop = std::min(valid_token_count - drop_idx, max_text_tokens);
    out.valid_text_len = valid_after_drop;
    if (valid_after_drop > 0) {
        const float* src =
            static_cast<const float*>(last_hidden.data) +
            static_cast<std::size_t>(drop_idx) * static_cast<std::size_t>(text_embed_dim);
        std::memcpy(out.hidden_states.data(), src,
                    static_cast<std::size_t>(valid_after_drop) *
                        static_cast<std::size_t>(text_embed_dim) * sizeof(float));
        std::fill_n(out.attention_mask.begin(), valid_after_drop, 1);
    }
    return out;
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
    auto attn_mask_additive = make_additive_attention_mask(max_seq_len, real_len);

    // 3. Run text encoder engine.
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{padded_ids.data(), {static_cast<int64_t>(max_seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attn_mask_additive.data(), {static_cast<int64_t>(max_seq_len)}, DType::kFloat32};
    auto outputs = text_engine_->forward(inputs);

    return pack_prompt_hidden_after_drop(outputs.at("last_hidden_state"), real_len, drop_idx,
                                         max_text_tokens, text_embed_dim,
                                         "QwenImagePipeline::encode_text");
}

namespace {

int32_t resolve_required_token_id(const ITokenizer& tokenizer, std::string_view token,
                                  const char* caller) {
    try {
        const int32_t direct = tokenizer.id_for_token(token);
        if (direct >= 0) {
            return direct;
        }
    } catch (const std::exception&) {
    }
    const auto ids = tokenizer.encode(std::string(token));
    if (ids.size() == 1) {
        return ids[0];
    }
    throw std::runtime_error(std::string(caller) + ": tokenizer cannot resolve token " +
                             std::string(token));
}

std::vector<int32_t> expand_single_image_pad_token(const std::vector<int32_t>& ids,
                                                   int32_t image_pad_id, int image_feature_tokens,
                                                   std::size_t& image_start) {
    const auto it = std::find(ids.begin(), ids.end(), image_pad_id);
    if (it == ids.end()) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: prompt template did not "
            "contain <|image_pad|>");
    }
    image_start = static_cast<std::size_t>(std::distance(ids.begin(), it));
    std::vector<int32_t> out;
    out.reserve(ids.size() + static_cast<std::size_t>(std::max(image_feature_tokens - 1, 0)));
    for (auto cur = ids.begin(); cur != ids.end(); ++cur) {
        if (cur == it) {
            out.insert(out.end(), static_cast<std::size_t>(image_feature_tokens), image_pad_id);
        } else {
            out.push_back(*cur);
        }
    }
    return out;
}

} // namespace

QwenImagePipeline::EncodedPrompt QwenImagePipeline::encode_text_with_image_conditioning(
    const std::string& prompt, const std::vector<float>& image_features) const {
    const int max_seq_len = config_.text_encoder.max_seq_len;
    const int drop_idx = config_.tokenizer.prompt_template_drop_idx;
    const int max_text_tokens = config_.denoiser.max_text_tokens;
    const int text_embed_dim = config_.denoiser.text_embed_dim;

    validate_encode_text_inputs(text_engine_ != nullptr, tokenizer_ != nullptr, max_seq_len,
                                drop_idx, max_text_tokens, text_embed_dim);
    if (!text_engine_->has_input("image_hidden") || !text_engine_->has_input("image_mask")) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: text engine is missing "
            "Edit image_hidden/image_mask inputs");
    }
    if (image_features.empty() ||
        image_features.size() % static_cast<std::size_t>(text_embed_dim) != 0) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: image_features size must be "
            "a positive multiple of text_embed_dim");
    }
    const int image_feature_tokens =
        static_cast<int>(image_features.size() / static_cast<std::size_t>(text_embed_dim));

    const std::string templated =
        std::string(kEditPromptTemplatePrefix) + prompt + std::string(kEditPromptTemplateSuffix);
    const std::vector<int32_t> template_ids = tokenizer_->encode(templated);
    const int32_t image_pad_id = resolve_required_token_id(
        *tokenizer_, "<|image_pad|>", "QwenImagePipeline::encode_text_with_image_conditioning");
    std::size_t image_start = 0;
    std::vector<int32_t> input_ids = expand_single_image_pad_token(
        template_ids, image_pad_id, image_feature_tokens, image_start);

    const int raw_token_count = static_cast<int>(input_ids.size());
    if (raw_token_count <= drop_idx) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: tokenized prompt has " +
            std::to_string(raw_token_count) + " tokens, but drop_idx=" + std::to_string(drop_idx) +
            " requires more");
    }
    if (raw_token_count > max_seq_len) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: tokenized prompt has " +
            std::to_string(raw_token_count) +
            " tokens, exceeding text max_seq_len=" + std::to_string(max_seq_len));
    }

    std::vector<int32_t> padded_ids(static_cast<std::size_t>(max_seq_len), 0);
    std::copy(input_ids.begin(), input_ids.end(), padded_ids.begin());

    auto attn_mask_additive = make_additive_attention_mask(max_seq_len, raw_token_count);

    std::vector<float> image_hidden(
        static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(text_embed_dim), 0.0F);
    std::vector<float> image_mask(static_cast<std::size_t>(max_seq_len), 0.0F);
    if (image_start + static_cast<std::size_t>(image_feature_tokens) >
        static_cast<std::size_t>(max_seq_len)) {
        throw std::runtime_error(
            "QwenImagePipeline::encode_text_with_image_conditioning: image features exceed "
            "text max_seq_len");
    }
    // image_features and the destination slice of image_hidden are both
    // contiguous (image_feature_tokens consecutive rows of text_embed_dim
    // floats), so the splice collapses to one memcpy plus a fill for the mask.
    std::memcpy(image_hidden.data() + image_start * static_cast<std::size_t>(text_embed_dim),
                image_features.data(),
                static_cast<std::size_t>(image_feature_tokens) *
                    static_cast<std::size_t>(text_embed_dim) * sizeof(float));
    std::fill_n(image_mask.begin() + static_cast<std::ptrdiff_t>(image_start), image_feature_tokens,
                1.0F);

    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{padded_ids.data(), {static_cast<int64_t>(max_seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attn_mask_additive.data(), {static_cast<int64_t>(max_seq_len)}, DType::kFloat32};
    inputs["image_hidden"] =
        Tensor{image_hidden.data(),
               {static_cast<int64_t>(max_seq_len), static_cast<int64_t>(text_embed_dim)},
               DType::kFloat32};
    inputs["image_mask"] =
        Tensor{image_mask.data(), {static_cast<int64_t>(max_seq_len)}, DType::kFloat32};
    auto outputs = text_engine_->forward(inputs);

    return pack_prompt_hidden_after_drop(outputs.at("last_hidden_state"), raw_token_count, drop_idx,
                                         max_text_tokens, text_embed_dim,
                                         "QwenImagePipeline::encode_text_with_image_conditioning");
}

namespace {

std::vector<float> resize_hwc_unit_to_qwen_vl_pixel_values(const std::vector<float>& pixels,
                                                           int src_h, int src_w, int dst_h,
                                                           int dst_w) {
    validate_resize_target(dst_h, dst_w);
    if (pixels.size() != static_cast<std::size_t>(src_h) * static_cast<std::size_t>(src_w) * 3UL) {
        throw std::runtime_error(
            "QwenImagePipeline::vision_encode_edit_condition: condition pixel buffer size "
            "does not match plan dimensions");
    }
    constexpr float kMean[3] = {0.48145466F, 0.4578275F, 0.40821073F};
    constexpr float kStd[3] = {0.26862954F, 0.26130258F, 0.27577711F};
    const std::size_t plane = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
    std::vector<float> out(6UL * plane);
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const std::size_t pixel_index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                static_cast<std::size_t>(x);
            for (int c = 0; c < 3; ++c) {
                const float unit =
                    sample_hwc_bilinear_unit(pixels.data(), src_h, src_w, y, x, dst_h, dst_w, c);
                const float normalized = (unit - kMean[c]) / kStd[c];
                const std::size_t base = static_cast<std::size_t>(c) * 2UL * plane;
                out[base + pixel_index] = normalized;
                out[base + plane + pixel_index] = normalized;
            }
        }
    }
    return out;
}

struct VisionEncodeDims {
    int height{0};
    int width{0};
    int patch{0};
    int merge{0};
    int hidden{0};
};

VisionEncodeDims resolve_vision_encode_dims(const QwenImageConfig& config) {
    VisionEncodeDims dims;
    dims.height =
        positive_or_default(config.vision_encoder.image_height, config.vision_encoder.image_size);
    dims.width =
        positive_or_default(config.vision_encoder.image_width, config.vision_encoder.image_size);
    dims.patch = config.vision_encoder.patch_size;
    dims.merge = config.vision_encoder.merge_size;
    dims.hidden = config.vision_encoder.out_hidden_size;
    return dims;
}

void validate_vision_encode_dims(const VisionEncodeDims& dims) {
    constexpr const char* kFn = "QwenImagePipeline::vision_encode_edit_condition: ";
    if (dims.height <= 0 || dims.width <= 0 || dims.patch <= 0 || dims.merge <= 0 ||
        dims.hidden <= 0) {
        throw std::runtime_error(std::string(kFn) +
                                 "invalid vision dims (height, width, patch_size, merge_size, "
                                 "hidden_size must all be > 0)");
    }
    const int divisor = dims.patch * dims.merge;
    if (dims.height % divisor != 0 || dims.width % divisor != 0) {
        throw std::runtime_error(std::string(kFn) +
                                 "vision image dims must be divisible by patch_size * merge_size");
    }
}

std::size_t expected_vision_feature_numel(const VisionEncodeDims& dims) {
    const int grid_h = dims.height / dims.patch;
    const int grid_w = dims.width / dims.patch;
    const int merged_grid_h = grid_h / dims.merge;
    const int merged_grid_w = grid_w / dims.merge;
    return static_cast<std::size_t>(merged_grid_h) * static_cast<std::size_t>(merged_grid_w) *
           static_cast<std::size_t>(dims.hidden);
}

void validate_vision_feature_numel(std::size_t actual, std::size_t expected) {
    if (actual != expected) {
        throw std::runtime_error(
            "QwenImagePipeline::vision_encode_edit_condition: vision output size " +
            std::to_string(actual) +
            " does not match expected merged tokens * hidden = " + std::to_string(expected));
    }
}

} // namespace

std::vector<float>
QwenImagePipeline::vision_encode_edit_condition(const EditInputTensors& edit_inputs) const {
    if (!vision_engine_) {
        throw std::runtime_error(
            "QwenImagePipeline::vision_encode_edit_condition: vision_engine_ is null");
    }
    const auto dims = resolve_vision_encode_dims(config_);
    validate_vision_encode_dims(dims);

    auto pixel_values = resize_hwc_unit_to_qwen_vl_pixel_values(
        edit_inputs.condition_pixels_hwc, edit_inputs.plan.condition_height,
        edit_inputs.plan.condition_width, dims.height, dims.width);

    TensorMap inputs;
    inputs["pixel_values"] =
        Tensor{pixel_values.data(),
               {6, static_cast<int64_t>(dims.height), static_cast<int64_t>(dims.width)},
               DType::kFloat32};
    auto outputs = vision_engine_->forward(inputs);
    const auto& features = outputs.at("image_features");

    validate_vision_feature_numel(features.numel(), expected_vision_feature_numel(dims));

    std::vector<float> result(features.numel());
    std::memcpy(result.data(), features.data, result.size() * sizeof(float));
    return result;
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

    std::vector<float> result;
    run_denoiser_into(latents_packed, normalized_t, hidden_states, result);
    return result;
}

void QwenImagePipeline::run_denoiser_into(const std::vector<float>& latents_packed,
                                          float normalized_t,
                                          const std::vector<float>& hidden_states,
                                          std::vector<float>& out_noise) const {
    const int in_channels = config_.denoiser.in_channels;
    const int max_text_tokens = config_.denoiser.max_text_tokens;
    const int text_embed_dim = config_.denoiser.text_embed_dim;
    const int64_t n_img =
        static_cast<int64_t>(latents_packed.size() / static_cast<std::size_t>(in_channels));
    float timestep_buf = normalized_t;

    TensorMap inputs;
    inputs["img_patched"] = Tensor{const_cast<float*>(latents_packed.data()),
                                   {1, n_img, static_cast<int64_t>(in_channels)},
                                   DType::kFloat32};
    inputs["txt_hidden"] =
        Tensor{const_cast<float*>(hidden_states.data()),
               {1, static_cast<int64_t>(max_text_tokens), static_cast<int64_t>(text_embed_dim)},
               DType::kFloat32};
    inputs["timestep"] = Tensor{&timestep_buf, {1}, DType::kFloat32};

    auto outputs = denoiser_engine_->forward(inputs);
    const auto& noise = outputs["noise_patched"];
    if (out_noise.size() != noise.numel()) {
        out_noise.resize(noise.numel());
    }
    std::memcpy(out_noise.data(), noise.data, out_noise.size() * sizeof(float));
}

// -----------------------------------------------------------------------------
// Denoise loop. Mirrors QwenImageDebugRunner._generate verbatim.
// -----------------------------------------------------------------------------

namespace {

// Validate inputs to denoise_loop_with_cfg. Throws on any failure.
// condition_size and n_condition_img are checked only when condition latents
// are provided (Edit mode); pass condition_size=0, n_condition_img=0 for T2I.
void validate_denoise_loop_inputs(std::size_t latents_size, std::size_t condition_size, int n_img,
                                  int n_condition_img, int num_steps, int in_channels) {
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
    if (condition_size > 0 || n_condition_img > 0) {
        if (n_condition_img <= 0) {
            throw std::runtime_error("QwenImagePipeline::denoise_loop_with_cfg: "
                                     "n_condition_img must be > 0 when condition latents provided");
        }
        const auto expected_condition =
            static_cast<std::size_t>(n_condition_img) * static_cast<std::size_t>(in_channels);
        if (condition_size != expected_condition) {
            throw std::runtime_error(
                "QwenImagePipeline::denoise_loop_with_cfg: condition_latents_packed size " +
                std::to_string(condition_size) +
                " does not match n_condition_img * in_channels = " +
                std::to_string(expected_condition));
        }
    }
}

// Validate denoiser runtime invariants used by denoise_loop_with_cfg. These
// don't change across steps, so checking once before the loop lets the hot
// path skip per-call re-validation.
void validate_denoise_loop_runtime(bool has_engine, int max_text_tokens, int text_embed_dim,
                                   std::size_t pos_hidden_size, std::size_t neg_hidden_size) {
    constexpr const char* kFn = "QwenImagePipeline::denoise_loop_with_cfg: ";
    if (!has_engine) {
        throw std::runtime_error(std::string(kFn) + "denoiser_engine_ is null");
    }
    if (max_text_tokens <= 0 || text_embed_dim <= 0) {
        throw std::runtime_error(std::string(kFn) + "invalid denoiser config");
    }
    const std::size_t expected =
        static_cast<std::size_t>(max_text_tokens) * static_cast<std::size_t>(text_embed_dim);
    if (pos_hidden_size != expected || neg_hidden_size != expected) {
        throw std::runtime_error(std::string(kFn) +
                                 "hidden_states size does not match max_text_tokens * "
                                 "text_embed_dim = " +
                                 std::to_string(expected));
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

} // namespace

// Combine pos + neg noise predictions via true-CFG and apply Qwen-Image's
// per-token L2 renormalization. Matches models/qwen_image/debug_runner.py exactly:
//   comb = neg + cfg*(pos - neg)
//   noise = comb * (||pos|| / max(||comb||, 1e-8))   per-token
//
// Per-sample-correctness audit (PR 2 of diffusion batch-inference):
//   - Reduction axis: ``channels`` only (inner loop ``c < channels``).
//   - Outer loop walks ``n_tokens`` independently — NO cross-token pooling.
//   - Caller passes ``n_tokens = B * n_img`` for a batched buffer that lays
//     out samples contiguously (sample 0 tokens, then sample 1 tokens, ...).
//   - Because reductions never sum across the outer loop, sample i's output
//     cannot depend on sample j's values — the renorm is strictly per-token,
//     which is also per-sample. See tests/cpp/test_qwen_image_cfg_renorm.cpp.
void QwenImagePipeline::combine_cfg_with_renorm(const std::vector<float>& noise_pos,
                                                const std::vector<float>& noise_neg,
                                                float cfg_scale, int n_tokens, std::size_t channels,
                                                std::vector<float>& out) {
    // Per-token L2 norms over `channels` (=64) fp32 values stay well below
    // 1e4, so a float accumulator is numerically fine and lets the compiler
    // SIMD-vectorize the reductions.
    constexpr float kCfgRenormFloorF = static_cast<float>(kCfgRenormFloor);
    const std::size_t required = static_cast<std::size_t>(n_tokens) * channels;
    if (out.size() != required) {
        out.resize(required);
    }
    for (int tok = 0; tok < n_tokens; ++tok) {
        const std::size_t base = static_cast<std::size_t>(tok) * channels;
        float pos_sq = 0.0F;
        float comb_sq = 0.0F;
        for (std::size_t c = 0; c < channels; ++c) {
            const float p = noise_pos[base + c];
            const float ng = noise_neg[base + c];
            const float comb = ng + cfg_scale * (p - ng);
            out[base + c] = comb;
            pos_sq += p * p;
            comb_sq += comb * comb;
        }
        const float scale = std::sqrt(pos_sq) / std::max(std::sqrt(comb_sq), kCfgRenormFloorF);
        for (std::size_t c = 0; c < channels; ++c) {
            out[base + c] *= scale;
        }
    }
}

std::vector<float> QwenImagePipeline::denoise_loop_with_cfg(
    std::vector<float> latents_packed, const EncodedPrompt& pos, const EncodedPrompt& neg,
    int n_img, int num_steps, float cfg_scale, const std::vector<float>& condition_latents_packed,
    int n_condition_img) const {
    const int in_channels = config_.denoiser.in_channels;
    const bool has_condition = !condition_latents_packed.empty();
    const int effective_condition_img = has_condition ? n_condition_img : 0;
    validate_denoise_loop_inputs(latents_packed.size(), condition_latents_packed.size(), n_img,
                                 effective_condition_img, num_steps, in_channels);
    // Per-step invariants validated once here so the hot loop can call
    // run_denoiser_into directly without re-checking dimensions.
    validate_denoise_loop_runtime(denoiser_engine_ != nullptr, config_.denoiser.max_text_tokens,
                                  config_.denoiser.text_embed_dim, pos.hidden_states.size(),
                                  neg.hidden_states.size());

    auto scheduler = build_scheduler(config_.diffusion, num_steps, n_img);
    const auto& timesteps = scheduler.timesteps();

    const bool do_cfg = (cfg_scale > 1.0F);
    const std::size_t generated_numel = latents_packed.size();
    const std::size_t total_numel = generated_numel + condition_latents_packed.size();
    const std::size_t channels = static_cast<std::size_t>(in_channels);

    // Pre-allocate hot-loop buffers. The denoiser engine produces total_numel
    // floats per forward (= generated_numel in T2I, = generated + condition in
    // Edit). Reusing these across steps + CFG variants means the loop does no
    // per-iteration allocation.
    std::vector<float> noise_pos_buf(total_numel);
    std::vector<float> noise_neg_buf;
    std::vector<float> noise_pred_buf;
    if (do_cfg) {
        noise_neg_buf.resize(total_numel);
        noise_pred_buf.resize(generated_numel);
    }

    // Edit mode concatenates condition latents along the sequence axis on each
    // step (the engine is baked for the fixed total length, latents_packed[0]
    // varies per step). Copy the condition tail once; refresh the head each
    // step from the changing latents_packed.
    std::vector<float> model_buffer;
    if (has_condition) {
        model_buffer.resize(total_numel);
        std::memcpy(model_buffer.data() + generated_numel, condition_latents_packed.data(),
                    condition_latents_packed.size() * sizeof(float));
    }

    for (int step = 0; step < num_steps; ++step) {
        const float t = timesteps[static_cast<std::size_t>(step)];
        const float norm_t = normalize_timestep(t);
        const std::vector<float>* model_latents = &latents_packed;
        if (has_condition) {
            std::memcpy(model_buffer.data(), latents_packed.data(),
                        latents_packed.size() * sizeof(float));
            model_latents = &model_buffer;
        }

        run_denoiser_into(*model_latents, norm_t, pos.hidden_states, noise_pos_buf);
        const float* noise_for_scheduler = noise_pos_buf.data();
        if (do_cfg) {
            run_denoiser_into(*model_latents, norm_t, neg.hidden_states, noise_neg_buf);
            // combine_cfg_with_renorm only touches the first n_img tokens, so
            // the trailing condition portion of the buffers is ignored.
            QwenImagePipeline::combine_cfg_with_renorm(noise_pos_buf, noise_neg_buf, cfg_scale,
                                                       n_img, channels, noise_pred_buf);
            noise_for_scheduler = noise_pred_buf.data();
        }

        // Euler step in-place over the output-image tokens only.
        scheduler.step(latents_packed.data(), noise_for_scheduler,
                       static_cast<int32_t>(latents_packed.size()), step);
    }

    return latents_packed;
}

namespace {

constexpr const char* kVaeEncodeFn = "QwenImagePipeline::vae_encode_edit_condition: ";

void validate_vae_encode_image_geometry(int latent_channels, int patch, int vae_scale, int image_h,
                                        int image_w) {
    if (latent_channels <= 0 || patch <= 0 || vae_scale <= 0 || image_h <= 0 || image_w <= 0) {
        throw std::runtime_error(std::string(kVaeEncodeFn) +
                                 "invalid dims (latent_channels, patch_size, vae spatial scale, "
                                 "VAE image height/width must all be > 0)");
    }
    if (image_h % vae_scale != 0 || image_w % vae_scale != 0) {
        throw std::runtime_error(std::string(kVaeEncodeFn) +
                                 "VAE image dims must be divisible by vae spatial scale");
    }
}

void validate_vae_encode_buffer_sizes(int latent_channels, int image_h, int image_w,
                                      std::size_t vae_pixels_size, std::size_t latents_mean_size,
                                      std::size_t latents_std_size) {
    const std::size_t expected_pixels =
        3UL * static_cast<std::size_t>(image_h) * static_cast<std::size_t>(image_w);
    if (vae_pixels_size != expected_pixels) {
        throw std::runtime_error(std::string(kVaeEncodeFn) + "vae_pixels_ncthw size " +
                                 std::to_string(vae_pixels_size) +
                                 " does not match 3 * H * W = " + std::to_string(expected_pixels));
    }
    if (static_cast<int>(latents_mean_size) != latent_channels ||
        static_cast<int>(latents_std_size) != latent_channels) {
        throw std::runtime_error(std::string(kVaeEncodeFn) +
                                 "preprocessor latents_mean/std missing or wrong size (need " +
                                 std::to_string(latent_channels) + " entries each)");
    }
}

void validate_vae_encode_edit_dims(bool has_engine, int latent_channels, int patch, int vae_scale,
                                   int image_h, int image_w, std::size_t vae_pixels_size,
                                   std::size_t latents_mean_size, std::size_t latents_std_size) {
    if (!has_engine) {
        throw std::runtime_error(std::string(kVaeEncodeFn) + "vae_encoder_engine_ is null");
    }
    validate_vae_encode_image_geometry(latent_channels, patch, vae_scale, image_h, image_w);
    validate_vae_encode_buffer_sizes(latent_channels, image_h, image_w, vae_pixels_size,
                                     latents_mean_size, latents_std_size);
}

void normalize_encoded_latents_inplace(float* data, int latent_channels, std::size_t per_channel,
                                       const std::vector<float>& latents_mean,
                                       const std::vector<float>& latents_std) {
    for (int c = 0; c < latent_channels; ++c) {
        const float mean = latents_mean[static_cast<std::size_t>(c)];
        const float stdv = latents_std[static_cast<std::size_t>(c)];
        if (stdv == 0.0F) {
            throw std::runtime_error(
                "QwenImagePipeline::vae_encode_edit_condition: latents_std contains zero");
        }
        float* base = data + static_cast<std::size_t>(c) * per_channel;
        for (std::size_t i = 0; i < per_channel; ++i) {
            base[i] = (base[i] - mean) / stdv;
        }
    }
}

} // namespace

std::vector<float>
QwenImagePipeline::vae_encode_edit_condition(const EditInputTensors& edit_inputs) const {
    const int latent_channels = config_.vae.latent_channels;
    const int patch = config_.denoiser.patch_size;
    const int vae_scale = config_.vae.spatial_scale_factor;
    const int image_h = edit_inputs.plan.vae_height;
    const int image_w = edit_inputs.plan.vae_width;
    validate_vae_encode_edit_dims(vae_encoder_engine_ != nullptr, latent_channels, patch, vae_scale,
                                  image_h, image_w, edit_inputs.vae_pixels_ncthw.size(),
                                  preprocessor_.latents_mean.size(),
                                  preprocessor_.latents_std.size());

    const int h_lat = image_h / vae_scale;
    const int w_lat = image_w / vae_scale;

    TensorMap inputs;
    inputs["image"] =
        Tensor{const_cast<float*>(edit_inputs.vae_pixels_ncthw.data()),
               {1, 3, 1, static_cast<int64_t>(image_h), static_cast<int64_t>(image_w)},
               DType::kFloat32};
    auto outputs = vae_encoder_engine_->forward(inputs);
    const auto& latent_tensor = outputs.at("latent");

    const std::size_t expected_latent = static_cast<std::size_t>(latent_channels) *
                                        static_cast<std::size_t>(h_lat) *
                                        static_cast<std::size_t>(w_lat);
    if (latent_tensor.numel() != expected_latent) {
        throw std::runtime_error(
            "QwenImagePipeline::vae_encode_edit_condition: VAE encoder output size " +
            std::to_string(latent_tensor.numel()) +
            " does not match latent_channels * H * W = " + std::to_string(expected_latent));
    }

    std::vector<float> latent_chw(expected_latent);
    std::memcpy(latent_chw.data(), latent_tensor.data, latent_chw.size() * sizeof(float));
    normalize_encoded_latents_inplace(latent_chw.data(), latent_channels,
                                      static_cast<std::size_t>(h_lat) *
                                          static_cast<std::size_t>(w_lat),
                                      preprocessor_.latents_mean, preprocessor_.latents_std);
    return patchify_latents(latent_chw, latent_channels, h_lat, w_lat, patch);
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

    auto latent_chw = unpatchify_latents(latents_packed, latent_channels, patch, packed_h, packed_w,
                                         h_lat, w_lat);

    // Bundle stores raw vae.config.latents_std/mean; diffusers internally
    // inverts to 1/raw_std before multiplying, so the un-normalize collapses
    // to z = z * raw_std + mean.
    const std::size_t per_channel =
        static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat);
    unnormalize_latents_inplace(latent_chw.data(), latent_channels, per_channel,
                                preprocessor_.latents_mean, preprocessor_.latents_std);

    // VAE engine expects [1, C, 1, H, W] NCTHW; the [C, H, W] byte layout above
    // matches it because the T axis has size 1.
    TensorMap inputs;
    inputs["latent"] = Tensor{latent_chw.data(),
                              {1, static_cast<int64_t>(latent_channels), 1,
                               static_cast<int64_t>(h_lat), static_cast<int64_t>(w_lat)},
                              DType::kFloat32};
    auto outputs = vae_decoder_engine_->forward(inputs);
    const auto& image_tensor = outputs.at("image");

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

    return chw_to_hwc_unit_range(static_cast<const float*>(image_tensor.data), h_out, w_out);
}

} // namespace trtmc

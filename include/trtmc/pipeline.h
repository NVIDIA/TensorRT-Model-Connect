/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// trtmc public C++ API — the only header users need.
//
// Usage:
//   auto pipe = trtmc::load("model.bundle");
//   auto result = pipe->generate("Hello", {.max_new_tokens = 20});
//   std::cout << result.text << std::endl;

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

// --- Result types (all value types, user owns the data) ---

struct TranscriptionSegment {
    double start_seconds{0.0};
    double end_seconds{0.0};
    std::string text;
    std::vector<int32_t> token_ids;
};

struct TextResult {
    TextResult() = default;
    TextResult(std::string result_text, std::vector<int32_t> result_token_ids,
               double result_prefill_ms = 0.0, double result_decode_ms = 0.0,
               std::vector<TranscriptionSegment> result_segments = {})
        : text(std::move(result_text)), token_ids(std::move(result_token_ids)),
          prefill_ms(result_prefill_ms), decode_ms(result_decode_ms),
          segments(std::move(result_segments)) {}

    std::string text;
    std::vector<int32_t> token_ids;
    // Time spent resetting per-request logical state and preparing reusable
    // runtime objects before prefill begins.
    double setup_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    // Populated by transcription pipelines when timestamp intervals are
    // requested and supported.
    std::vector<TranscriptionSegment> segments;
};

enum class TranscriptionTask {
    kTranscribe,
    kTranslate,
};

struct TranscriptionConfig {
    // Maximum number of decoder output tokens. The valid upper bound is
    // model-specific and excludes the decoder prompt tokens.
    int32_t max_output_tokens{224};
    // Source sample rate in Hz. 0 means the samples are already at the model
    // rate. A positive value is resampled when necessary.
    int32_t input_sample_rate{0};
    // 1 preserves greedy decoding. Values greater than 1 select beam search.
    int32_t beam_size{1};
    // Beam-search length normalization exponent. 0 ranks hypotheses by their
    // accumulated log probability; 1 ranks by mean log probability.
    float length_penalty{1.0F};
    // 0 disables automatic fallback. Otherwise, an unterminated decode is
    // retried with doubled beam sizes up to this value.
    int32_t beam_fallback_max_size{0};
    std::string source_language{"en"};
    std::string target_language{"en"};
    TranscriptionTask task{TranscriptionTask::kTranscribe};
    bool punctuation{true};
    bool timestamps{false};
    // 0 disables the caller-specified duration limit. A positive value is a
    // hard input limit in seconds; longer inputs are rejected.
    float max_input_duration_seconds{0.0F};
    // 0 processes one model-sized segment. A positive value splits the input
    // into segments no longer than this many seconds and joins their text.
    float segment_duration_seconds{0.0F};
    // 0 keeps fixed-size segmentation. A positive value enables dynamic
    // segmentation: long inputs are covered by approximately equal windows
    // between this minimum and segment_duration_seconds, minimizing short
    // padded tail windows.
    float segment_min_duration_seconds{0.0F};
    // Requested overlap between adjacent segments. Dynamic segmentation can
    // increase the overlap when needed to keep every window at least
    // segment_min_duration_seconds long.
    float segment_overlap_seconds{0.0F};
    // Merge overlapping segment token sequences with a boundary-constrained
    // longest common subsequence instead of concatenating their text.
    bool lcs_merge{false};
};

struct TranscriptionRequest {
    std::vector<float> audio_samples;
    TranscriptionConfig config;
};

struct ImageResult {
    std::vector<float> pixels; // [C, H, W] float32 in [0,1]
    int32_t height{0};
    int32_t width{0};
    int32_t channels{3};
    int32_t num_frames{1}; // >1 for video
};

struct AudioResult {
    std::vector<float> samples; // mono float32 [-1,1]
    int32_t num_samples{0};
    int32_t sample_rate{24000};
};

struct TranscriptionStreamConfig {
    // Cache-aware streaming transcription contract:
    // att_context_size=[left,right], measured in 80 ms encoder frames.
    // Common right-context presets {0, 1, 6, 13} give chunk sizes of
    // 80 ms, 160 ms, 560 ms, and 1120 ms respectively.
    int32_t input_sample_rate{16000};
    int32_t max_new_tokens{224};
    int32_t att_context_left{70};
    int32_t att_context_right{13};
    bool use_cache{true};         // Reuse encoder attention/conv caches between chunks.
    bool use_feature_cache{true}; // Reuse mel/pre-encoder overlap between chunks.
    bool emit_partial_results{true};
    bool online_normalization{false};
    bool pad_and_drop_preencoded{false};

    // Multilingual ASR: a language tag from the bundle's prompt_dictionary
    // (e.g. "en-US", "es-ES", "auto"). Empty selects prompt index 0.
    // Ignored when the bundle's has_prompt_kernel is false.
    std::string language;
};

struct TranscriptionStreamResult {
    std::string text;
    std::vector<int32_t> token_ids;
    bool is_final{false};
    int32_t chunk_index{0};
    int64_t accepted_samples{0};
    int32_t sample_rate{16000};
};

struct EmbeddingResult {
    std::vector<float> data;
    int32_t dim{0};
};

struct SegmentResult {
    std::vector<int32_t> mask; // class indices [H, W]
    int32_t height{0};
    int32_t width{0};
};

struct StereoDisparityResult {
    std::vector<float> disparity; // [H, W] non-negative disparity in pixels
    int32_t height{0};
    int32_t width{0};
};

struct PromptedSegmentationResult {
    std::vector<float> masks;      // [num_masks, H, W], logits after postprocess
    std::vector<float> iou_scores; // [num_masks]
    std::vector<float> boxes;      // [num_masks, 4], xyxy absolute pixel coordinates
    int32_t num_masks{0};
    int32_t height{0};
    int32_t width{0};
};

struct VideoFrameView {
    // Borrowed RGB HWC float32 storage in [0,1]. It must remain valid for the
    // complete IVideoTrackingPipeline::track_video() call.
    const float* pixels{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct VideoFrame {
    // Owned RGB HWC float32 storage in [0,1]. Model capabilities may use a
    // family-specific decoder when source-framework pixel parity requires it.
    std::vector<float> pixels;
    int32_t height{0};
    int32_t width{0};

    bool empty() const { return pixels.empty(); }
    VideoFrameView view() const { return {pixels.data(), height, width}; }
};

struct ClassificationResult {
    std::vector<float> logits; // [num_classes]
    int32_t top_class{-1};
    float top_score{0.0F};
};

struct TextEmbedding {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct GenerateConfig {
    int32_t max_new_tokens{128};
    // Multilingual encoder-decoder controls. A non-negative source token is
    // placed after the encoder EOS token. A non-negative forced BOS token is
    // emitted as the first decoder token after decoder_start_token_id.
    int32_t source_language_token_id{-1};
    int32_t forced_bos_token_id{-1};
    int32_t num_samples{1}; // non-AR generators: number of independent samples to emit
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled, 0.0 = greedy, (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};
    float guidance_scale{-1.0f};          // diffusion; ELF uses this as self-conditioning CFG scale
    float cfg_scale{-1.0f};               // conditional CFG scale; <0 uses model default
    int32_t num_steps{-1};                // diffusion
    float sde_gamma{-1.0f};               // diffusion/flow matching; <0 uses model default
    std::vector<float> initial_latents;   // diffusion: optional packed initial latents
    std::vector<float> condition_latents; // ELF: [max_length, text_encoder_dim] cond seq
    std::vector<float> condition_mask;    // ELF: [max_length], >0 marks fixed cond tokens
    std::vector<float> sampling_steps;    // ELF: optional upstream t_steps [num_steps + 1]
    std::vector<float> sde_noises;        // ELF: optional scaled eps [num_steps - 1, L, D]
    // Diffusion (text-to-image): optional override for the negative prompt.
    // Empty means "use the bundle's default negative prompt".
    std::string negative_prompt;
    // Diffusion (text-to-image): optional output image size override. <=0
    // means "use the bundle's default height/width".
    int32_t height{0};
    int32_t width{0};
    int32_t eos_token_id{-1};
    // Text diffusion / speculative decoding modes. Empty/"auto" lets a
    // runtime choose its model-default mode; causal decoder runtimes ignore it.
    std::string text_generation_mode{
        "auto"};                       // "ar", "diffusion", "linear_spec", "linear_spec_lora"
    int32_t block_length{0};           // <=0 uses bundle/model default
    float confidence_threshold{-1.0f}; // <0 uses mode default
    int32_t tail_frames{0};            // speech-to-speech: extra frames after input
    bool use_chat_template{false};     ///< Apply chat template before tokenization
    bool enable_thinking{true};        ///< If false, disable reasoning/thinking mode
    bool stop_on_boxed_answer{false};  ///< Stop once generated text contains a full \boxed{...}
    int32_t stop_check_interval{16};   ///< Token interval for answer-stop checks
    // Empty selects the base model. A non-empty ID selects a LoRA adapter
    // registered with a LoRA-capable runtime before generation.
    std::string lora_adapter_id;
};

class ITranscriptionStream {
  public:
    virtual ~ITranscriptionStream() = default;

    // Append one mono float32 audio chunk. Set is_final=true for the last
    // chunk, or call finish() after the final accept_audio().
    virtual TranscriptionStreamResult accept_audio(const float* audio_samples, int32_t num_samples,
                                                   bool is_final = false) = 0;

    // Flush pending right-context/audio tail and return the final transcript.
    virtual TranscriptionStreamResult finish() = 0;

    // Clear hypotheses, encoder caches, feature cache, and accepted audio.
    virtual void reset() = 0;

    virtual TranscriptionStreamConfig config() const = 0;
};

class IVideoTrackingPipeline {
  public:
    virtual ~IVideoTrackingPipeline() = default;

    // Decode one frame according to the owning model's source-framework
    // contract. Keeping this on the capability avoids model checks in shared
    // CLI code while allowing exact family-local decoding where required.
    virtual VideoFrame load_video_frame(const std::string& path) = 0;

    // The owning model defines the JSON schema and any per-frame assets written
    // below output_assets_dir. Pass both output paths empty to discard outputs,
    // for example during an in-process benchmark. Supplying exactly one empty
    // output path is invalid. The return value is the number of produced frames.
    // The call is synchronous: all capability work and requested output writes
    // complete before return, and the borrowed frame storage is no longer used.
    virtual int32_t track_video(const std::vector<VideoFrameView>& frames,
                                const std::string& output_json,
                                const std::string& output_assets_dir) = 0;
};

// Optional capability for pipelines that can decode an ordered frame batch.
// Keeping this separate from IVideoTrackingPipeline preserves that interface's
// vtable while allowing the CLI to cross-cast across the core and model DSOs.
class IVideoFrameBatchLoader {
  public:
    virtual ~IVideoFrameBatchLoader();

    // Results correspond one-for-one with paths and retain the exact input
    // order. Implementations must finish all work before returning or throwing.
    virtual std::vector<VideoFrame> load_video_frames(const std::vector<std::string>& paths) = 0;

    // Maximum number of frame decodes that load_video_frames() may execute at
    // once. The CLI records this as request-loading provenance.
    virtual std::size_t max_video_frame_load_concurrency() const noexcept = 0;
};

// --- Pipeline interface ---

class IPipeline {
  public:
    virtual ~IPipeline() = default;

    virtual int32_t default_max_new_tokens() const { return 20; }

    // -- Text generation --
    virtual TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support generate()");
    }

    // -- Text generation with image --
    virtual TextResult generate(const std::string& prompt, const float* image_pixels,
                                int32_t image_height, int32_t image_width,
                                const GenerateConfig& cfg = {}) {
        (void)image_pixels;
        (void)image_height;
        (void)image_width;
        return generate(prompt, cfg);
    }

    // -- Text encoding (reusable embeddings for diffusion) --
    virtual TextEmbedding encode_text(const std::string& prompt) {
        (void)prompt;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support encode_text()");
    }

    // -- Image generation (diffusion) --
    virtual bool supports_image_generation() const { return false; }

    virtual ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_image()");
    }

    virtual ImageResult generate_image(const std::string& prompt, const float* image_pixels,
                                       int32_t image_height, int32_t image_width,
                                       const GenerateConfig& cfg = {}) {
        (void)image_pixels;
        (void)image_height;
        (void)image_width;
        return generate_image(prompt, cfg);
    }

    virtual ImageResult generate_image(const TextEmbedding& emb, const GenerateConfig& cfg = {}) {
        (void)emb;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_image(TextEmbedding)");
    }

    // -- Image batch generation (diffusion) --
    //
    // ``prompts.size() == per_sample_seeds.size()`` is the total per-call batch.
    // Implementations may chunk internally if the engine cap is below that
    // size — see the diffusion batch-inference RFC (Decisions A/D).
    //
    // Default implementation is a sequential ``generate_image`` loop with
    // ``cfg.seed`` overridden per sample. Pipelines that can actually batch
    // override this for the speed win; pipelines that can't get correctness
    // for free without an override.
    virtual std::vector<ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts,
                         const std::vector<std::uint32_t>& per_sample_seeds,
                         const GenerateConfig& cfg = {}) {
        if (prompts.size() != per_sample_seeds.size()) {
            throw std::invalid_argument(
                "generate_image_batch: prompts.size() must equal per_sample_seeds.size()");
        }
        std::vector<ImageResult> out;
        out.reserve(prompts.size());
        GenerateConfig per_sample_cfg = cfg;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            per_sample_cfg.seed = static_cast<int32_t>(per_sample_seeds[i]);
            out.push_back(generate_image(prompts[i], per_sample_cfg));
        }
        return out;
    }

    // -- Audio generation --
    virtual AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_audio()");
    }

    // -- Streaming audio generation --
    // Callback receives (pcm_samples, num_samples, sample_rate) per chunk.
    using AudioChunkCallback = std::function<void(const float*, int32_t, int32_t)>;
    virtual int32_t generate_audio_streaming(const std::string& prompt, const GenerateConfig& cfg,
                                             AudioChunkCallback callback,
                                             int32_t chunk_frames = 32) {
        (void)prompt;
        (void)cfg;
        (void)callback;
        (void)chunk_frames;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support streaming");
    }

    // -- Transcription --
    // input_sample_rate: source audio sample rate. 0 = assume already at model rate.
    // When non-zero and different from the model's expected rate, the pipeline
    // resamples the audio before mel extraction.
    virtual TextResult transcribe(const float* audio_samples, int32_t num_samples,
                                  int32_t max_tokens = 224, int32_t input_sample_rate = 0) {
        (void)audio_samples;
        (void)num_samples;
        (void)max_tokens;
        (void)input_sample_rate;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support transcribe()");
    }

    // Typed transcription configuration. The default forwards to the legacy
    // overload so existing speech pipelines remain compatible.
    virtual TextResult transcribe(const float* audio_samples, int32_t num_samples,
                                  const TranscriptionConfig& cfg) {
        return transcribe(audio_samples, num_samples, cfg.max_output_tokens, cfg.input_sample_rate);
    }

    // Each request owns its samples and its complete per-input configuration.
    // Pipelines may override this for true batching; the default preserves the
    // request order and executes sequentially.
    virtual std::vector<TextResult>
    transcribe_batch(const std::vector<TranscriptionRequest>& requests) {
        std::vector<TextResult> results;
        results.reserve(requests.size());
        for (const auto& request : requests) {
            results.push_back(transcribe(request.audio_samples.data(),
                                         static_cast<int32_t>(request.audio_samples.size()),
                                         request.config));
        }
        return results;
    }

    // -- Streaming transcription (cache-aware ASR) --
    virtual std::unique_ptr<ITranscriptionStream>
    create_transcription_stream(const TranscriptionStreamConfig& cfg = {}) {
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support streaming transcription");
    }

    using TranscriptionChunkCallback = std::function<void(const TranscriptionStreamResult&)>;
    virtual TextResult transcribe_streaming(const float* audio_samples, int32_t num_samples,
                                            const TranscriptionStreamConfig& cfg,
                                            TranscriptionChunkCallback callback = nullptr) {
        auto stream = create_transcription_stream(cfg);
        auto chunk = stream->accept_audio(audio_samples, num_samples, false);
        if (callback && (!chunk.text.empty() || !chunk.token_ids.empty()))
            callback(chunk);
        auto final = stream->finish();
        if (callback)
            callback(final);
        TextResult out;
        out.text = std::move(final.text);
        out.token_ids = std::move(final.token_ids);
        return out;
    }

    // -- Speech to speech --
    virtual AudioResult speak(const float* audio_in, int32_t num_samples,
                              const GenerateConfig& cfg = {}, int32_t input_sample_rate = 0) {
        (void)audio_in;
        (void)num_samples;
        (void)cfg;
        (void)input_sample_rate;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support speak()");
    }

    // -- Embedding --
    virtual EmbeddingResult embed(const std::string& text) {
        (void)text;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support embed()");
    }

    // -- Reranking --
    virtual float rerank(const std::string& query, const std::string& document) {
        (void)query;
        (void)document;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support rerank()");
    }

    // -- Segmentation --
    // Image pixels are RGB HWC float values in [0, 1]. The owning model family
    // applies its bundle-defined resize, normalization, and layout transform.
    virtual SegmentResult segment(const float* pixels, int32_t height, int32_t width) {
        (void)pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support segment()");
    }

    // -- Stereo disparity --
    // Images are rectified RGB HWC float values in [0, 1], with matching
    // dimensions. The result preserves the input height and width.
    virtual StereoDisparityResult estimate_disparity(const float* left_pixels,
                                                     const float* right_pixels, int32_t height,
                                                     int32_t width) {
        (void)left_pixels;
        (void)right_pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support estimate_disparity()");
    }

    virtual PromptedSegmentationResult segment_prompted(const float* image_pixels,
                                                        int32_t image_height, int32_t image_width,
                                                        float point_x = 0.5F, float point_y = 0.5F,
                                                        bool is_foreground = true) {
        (void)image_pixels;
        (void)image_height;
        (void)image_width;
        (void)point_x;
        (void)point_y;
        (void)is_foreground;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support segment_prompted()");
    }

    virtual PromptedSegmentationResult segment_prompted_text(const float* image_pixels,
                                                             int32_t image_height,
                                                             int32_t image_width,
                                                             const std::string& text_prompt) {
        (void)image_pixels;
        (void)image_height;
        (void)image_width;
        (void)text_prompt;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support segment_prompted_text()");
    }

    // -- Image classification --
    // Image pixels are RGB HWC float values in [0, 1]. The owning model family
    // applies its bundle-defined resize, crop, normalization, and layout transform.
    virtual ClassificationResult classify(const float* pixels, int32_t height, int32_t width) {
        (void)pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support classify()");
    }

    // -- Encoder-only hidden states --
    virtual EmbeddingResult encode(const std::string& text) {
        (void)text;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support encode()");
    }

    // -- Neural operator --
    virtual EmbeddingResult solve(const float* branch_input, int32_t branch_len,
                                  const float* trunk_input, int32_t trunk_len) {
        (void)branch_input;
        (void)branch_len;
        (void)trunk_input;
        (void)trunk_len;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support solve()");
    }

    // -- Object detection --
    virtual std::string detect(const float* pixels, int32_t height, int32_t width,
                               float conf_threshold = 0.5f) {
        (void)pixels;
        (void)height;
        (void)width;
        (void)conf_threshold;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support detect()");
    }

    // -- Dynamic LoRA adapters --
    // Adapter files remain external to the base bundle. Implementations load
    // and cache them by ID; GenerateConfig::lora_adapter_id selects one for a
    // generation request. An empty ID always selects the base model.
    virtual bool supports_lora_adapters() const { return false; }

    virtual void load_lora_adapter(const std::string& adapter_id, const std::string& adapter_path) {
        (void)adapter_id;
        (void)adapter_path;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support dynamic LoRA adapters");
    }

    virtual void unload_lora_adapter(const std::string& adapter_id) {
        (void)adapter_id;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support dynamic LoRA adapters");
    }

    virtual std::vector<std::string> loaded_lora_adapters() const { return {}; }

    // -- Metadata --
    virtual const char* model_id() const = 0;
    virtual const char* pipeline_type() const = 0;
};

// --- Factory ---
// LoadOptions bundles every knob the factory understands. Users who only want
// the defaults can still call the positional overload below.
struct LoadOptions {
    std::string hf_python;
    std::string runtime_cache_path;
    bool cuda_graphs{false};
    std::uint64_t kv_cache_size_bytes{0};               // 0 = use bundle's max_cache_length
    std::string config_path;                            // --config <file> (empty = none)
    std::vector<std::string> set_tokens;                // --set ns.field=value (repeatable)
    std::vector<std::string> backend_search_paths;      // Extra directories for backend DSOs
    std::vector<std::string> model_plugin_search_paths; // Extra dirs for libtrtmc_model_*.so
};

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python = "",
                                const std::string& runtime_cache_path = "",
                                bool cuda_graphs = false);
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options);
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options,
                                const std::string& kernel_bindings_path);

} // namespace trtmc

// --- C ABI ---

extern "C" {

struct TrtmcPipelineOptions {
    int max_new_tokens;        // 0 = use model default
    const char* hf_python;     // nullptr = auto-detect
    const char* image_path;    // nullptr = text-only
    const char* runtime_cache; // nullptr = no RTX cache
    int cuda_graphs;           // 0 = disabled
};

// --- C ABI error codes ---
//
// Returned by C-ABI functions that yield an int (e.g. trtmc_generate_batch).
// On any non-zero return, callers may inspect trtmc_last_error() for a
// descriptive message. Codes are stable and additive — new codes get new
// non-zero integers; existing callers must treat unknown codes as a generic
// failure.
#define TRTMC_OK 0
#define TRTMC_ERR_INVALID_ARG 1
#define TRTMC_ERR_RUNTIME 2

// --- C ABI image result ---
//
// Plain-old-data image result returned by trtmc_generate_batch. The caller
// owns the trtmc_image_result_t array (typically a fixed-size stack/heap
// buffer of N entries) and is responsible for releasing the per-result
// `pixels` buffer via trtmc_image_result_free.
struct trtmc_image_result_t {
    float* pixels;            // malloc'd [C*H*W] float32 in [0,1]. nullptr on error.
    int32_t height;           // image height in pixels
    int32_t width;            // image width in pixels
    int32_t channels;         // number of channels (typically 3)
    int32_t num_frames;       // >1 for video
    std::uint64_t num_pixels; // total floats in `pixels` (channels*height*width*num_frames)
};

// Opaque handle alias used by future C-ABI generation functions. Today the
// C ABI hands users a `trtmc::IPipeline*` directly (a C++ type with a
// trivial vtable); language bindings that don't want the C++ type can use
// `trtmc_pipeline_t` for documentation purposes.
typedef trtmc::IPipeline* trtmc_pipeline_t;

trtmc::IPipeline* trtmc_create_pipeline(const char* bundle_path, int flags);
trtmc::IPipeline* trtmc_create_pipeline_ex(const char* bundle_path,
                                           const TrtmcPipelineOptions* options);
const char* trtmc_last_error(void);
const char* trtmc_version(void);
int trtmc_has_trt(void);

// Release the `pixels` buffer in a single trtmc_image_result_t entry. Safe
// on a zero-initialized entry (no-op when pixels is null). Sets pixels to
// nullptr after free.
void trtmc_image_result_free(trtmc_image_result_t* result);

// Generate a batch of images. `prompts` is an array of `num_prompts`
// null-terminated C strings; `seeds` is an array of `num_seeds` uint32_t
// per-sample seeds. `num_prompts` must equal `num_seeds`.
// `out_results` is a caller-owned array of `num_prompts`
// trtmc_image_result_t that the function fills; each result's `pixels`
// buffer is malloc'd and must be released with trtmc_image_result_free.
// Returns TRTMC_OK on success, a non-zero TRTMC_ERR_* code on failure.
int trtmc_generate_batch(trtmc_pipeline_t handle, const char* const* prompts, int num_prompts,
                         const std::uint32_t* seeds, int num_seeds, int num_inference_steps,
                         float guidance_scale, trtmc_image_result_t* out_results);
}

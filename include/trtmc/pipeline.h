#pragma once

// trtmc public C++ API — the only header users need.
//
// Usage:
//   auto pipe = trtmc::load("model.trtfb");
//   auto result = pipe->generate("Hello", {.max_new_tokens = 20});
//   std::cout << result.text << std::endl;

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

// --- Result types (all value types, user owns the data) ---

struct TextResult {
    std::string text;
    std::vector<int32_t> token_ids;
    double prefill_ms{0.0};
    double decode_ms{0.0};
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
    // NeMo cache-aware streaming contract for FastConformer-RNNT:
    // att_context_size=[left,right], measured in 80 ms encoder frames.
    // Supported Nemotron right contexts are {0, 1, 6, 13}, giving
    // chunk sizes of 80 ms, 160 ms, 560 ms, and 1120 ms respectively.
    int32_t input_sample_rate{16000};
    int32_t max_new_tokens{224};
    int32_t att_context_left{70};
    int32_t att_context_right{13};
    bool use_cache{true};         // Reuse encoder attention/conv caches between chunks.
    bool use_feature_cache{true}; // Reuse mel/pre-encoder overlap between chunks.
    bool emit_partial_results{true};
    bool online_normalization{false};
    bool pad_and_drop_preencoded{false};
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

struct PromptedSegmentationResult {
    std::vector<float> masks;      // [num_masks, H, W], logits after postprocess
    std::vector<float> iou_scores; // [num_masks]
    int32_t num_masks{0};
    int32_t height{0};
    int32_t width{0};
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
    int32_t tail_frames{0};           // speech-to-speech: extra frames after input
    bool use_chat_template{false};    ///< Apply chat template before tokenization
    bool enable_thinking{true};       ///< Qwen3: if false, disable thinking mode
    bool stop_on_boxed_answer{false}; ///< Stop once generated text contains a full \boxed{...}
    int32_t stop_check_interval{16};  ///< Token interval for answer-stop checks
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

// --- Pipeline interface ---

class IPipeline {
  public:
    virtual ~IPipeline() = default;

    // -- Text generation (decoder, mamba, rwkv, VL) --
    virtual TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support generate()");
    }

    // -- Text generation with image (VL models) --
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

    // -- Audio generation (bark, magpie) --
    virtual AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) {
        (void)prompt;
        (void)cfg;
        throw std::runtime_error(std::string(pipeline_type()) +
                                 " does not support generate_audio()");
    }

    // -- Streaming audio generation (magpie) --
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

    // -- Transcription (whisper, canary) --
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
    virtual SegmentResult segment(const float* pixels, int32_t height, int32_t width) {
        (void)pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support segment()");
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

    // -- Image classification --
    virtual ClassificationResult classify(const float* pixels, int32_t height, int32_t width) {
        (void)pixels;
        (void)height;
        (void)width;
        throw std::runtime_error(std::string(pipeline_type()) + " does not support classify()");
    }

    // -- Encoder-only hidden states (BERT) --
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
    std::uint64_t kv_cache_size_bytes{0};          // 0 = use bundle's max_cache_length
    std::string config_path;                       // --config <file> (empty = none)
    std::vector<std::string> set_tokens;           // --set ns.field=value (repeatable)
    std::vector<std::string> backend_search_paths; // Extra directories for backend DSOs
};

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python = "",
                                const std::string& runtime_cache_path = "",
                                bool cuda_graphs = false);
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options);

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

trtmc::IPipeline* trtmc_create_pipeline(const char* bundle_path, int flags);
trtmc::IPipeline* trtmc_create_pipeline_ex(const char* bundle_path,
                                           const TrtmcPipelineOptions* options);
const char* trtmc_last_error(void);
const char* trtmc_version(void);
int trtmc_has_trt(void);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

template <typename T>
class Span {
  public:
    constexpr Span() noexcept = default;
    constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

    template <std::size_t Size>
    constexpr Span(T (&data)[Size]) noexcept : data_(data), size_(Size) {}

    constexpr T* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr T* begin() const noexcept { return data_; }
    constexpr T* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }
    constexpr T& operator[](std::size_t index) const noexcept { return data_[index]; }

  private:
    T* data_{nullptr};
    std::size_t size_{0};
};

class ITask {
  public:
    virtual ~ITask() = default;
    virtual const char* task() const noexcept = 0;
};

struct TranscriptionSegment {
    double start_seconds{0.0};
    double end_seconds{0.0};
    std::string text;
    std::vector<std::int32_t> token_ids;
};

struct TextResult {
    TextResult() = default;
    TextResult(std::string result_text, std::vector<std::int32_t> result_token_ids,
               double result_prefill_ms = 0.0, double result_decode_ms = 0.0,
               std::vector<TranscriptionSegment> result_segments = {})
        : text(std::move(result_text)), token_ids(std::move(result_token_ids)),
          prefill_ms(result_prefill_ms), decode_ms(result_decode_ms),
          segments(std::move(result_segments)) {}

    std::string text;
    std::vector<std::int32_t> token_ids;
    double setup_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::vector<TranscriptionSegment> segments;
};

enum class TranscriptionTask {
    kTranscribe,
    kTranslate,
};

struct TranscriptionConfig {
    std::int32_t max_output_tokens{224};
    std::int32_t input_sample_rate{0};
    std::int32_t beam_size{1};
    float length_penalty{1.0F};
    std::string source_language{"en"};
    std::string target_language{"en"};
    TranscriptionTask task{TranscriptionTask::kTranscribe};
    bool punctuation{true};
    bool timestamps{false};
    float max_input_duration_seconds{0.0F};
    float segment_duration_seconds{0.0F};
    float segment_min_duration_seconds{0.0F};
    float segment_overlap_seconds{0.0F};
    bool lcs_merge{false};
};

struct TranscriptionRequest {
    std::vector<float> audio_samples;
    TranscriptionConfig config;
};

struct ImageResult {
    // Contiguous row-major HWC for images and THWC when num_frames > 1.
    std::vector<float> pixels;
    std::int32_t height{0};
    std::int32_t width{0};
    std::int32_t channels{3};
    std::int32_t num_frames{1};
};

struct AudioResult {
    std::vector<float> samples;
    std::int32_t num_samples{0};
    std::int32_t sample_rate{24000};
};

struct TranscriptionStreamConfig {
    std::int32_t input_sample_rate{16000};
    std::int32_t max_new_tokens{224};
    std::int32_t att_context_left{70};
    std::int32_t att_context_right{13};
    bool use_cache{true};
    bool use_feature_cache{true};
    bool online_normalization{false};
    bool pad_and_drop_preencoded{false};
    std::string language;
};

struct TranscriptionStreamResult {
    std::string text;
    std::vector<std::int32_t> token_ids;
    bool is_final{false};
    std::int32_t chunk_index{0};
    std::int64_t accepted_samples{0};
    std::int32_t sample_rate{16000};
};

struct EmbeddingResult {
    std::vector<float> data;
    std::int32_t dim{0};
};

struct SegmentResult {
    std::vector<std::int32_t> mask;
    std::int32_t height{0};
    std::int32_t width{0};
};

struct StereoDisparityResult {
    std::vector<float> disparity;
    std::int32_t height{0};
    std::int32_t width{0};
};

struct GeometryResult {
    std::vector<float> points;
    std::vector<float> depth;
    std::vector<std::uint8_t> mask;
    std::array<float, 9> intrinsics{};
    std::int32_t height{0};
    std::int32_t width{0};
};

struct PromptedSegmentationResult {
    std::vector<float> masks;
    std::vector<float> iou_scores;
    std::vector<float> boxes;
    std::int32_t num_masks{0};
    std::int32_t height{0};
    std::int32_t width{0};
};

struct ClassificationResult {
    std::vector<float> logits;
    std::int32_t top_class{-1};
    float top_score{0.0F};
};

enum class PoseCropStage {
    kRefinement,
    kScoring,
};

struct PoseCropBatch {
    std::vector<float> rendered_features;
    std::vector<float> observed_features;
    std::int32_t num_hypotheses{0};
    std::int32_t height{0};
    std::int32_t width{0};
    std::int32_t channels{0};
};

using PoseCropProvider =
    std::function<PoseCropBatch(const std::vector<float>& current_object_to_camera_poses,
                                PoseCropStage stage, std::int32_t refinement_iteration)>;

struct PoseEstimationRequest {
    // Row-major [num_hypotheses, 4, 4] object-to-camera rigid transforms.
    std::vector<float> candidate_poses;
    std::int32_t num_hypotheses{0};
    float mesh_diameter{0.0F};
    std::int32_t refinement_iterations{1};
    bool score_hypotheses{true};
    bool use_tracked_pose{false};
    bool update_tracking_state{true};
    PoseCropProvider crop_provider;
};

struct PoseEstimationResult {
    std::vector<float> refined_poses;
    std::vector<float> scores;
    std::int32_t num_hypotheses{0};
    std::int32_t best_index{-1};
    bool all_poses_rigid{false};
    double refinement_ms{0.0};
    double scoring_ms{0.0};
};

struct ImageFeaturesResult {
    std::vector<float> last_hidden_state;
    std::vector<std::int64_t> last_hidden_state_shape;
    std::vector<float> pooler_output;
    std::vector<std::int64_t> pooler_output_shape;
};

struct RobotObservation {
    Span<const float> image_pixels;
    std::int32_t image_height{0};
    std::int32_t image_width{0};
    std::int32_t image_channels{3};
    Span<const float> state;
};

struct RobotActionChunk {
    std::vector<float> actions;
    std::int32_t num_actions{0};
    std::int32_t action_dim{0};
    bool within_training_bounds{false};
    double inference_ms{0.0};
};

struct RobotAction {
    std::vector<float> values;
    std::int32_t action_dim{0};
    bool within_training_bounds{false};
    bool started_new_chunk{false};
    double inference_ms{0.0};
};

enum class VideoFrameFormat {
    kRgb8,
    kRgbFloat32,
};

struct VideoFrameView {
    const void* pixels{nullptr};
    std::size_t element_count{0};
    std::int32_t height{0};
    std::int32_t width{0};
    VideoFrameFormat format{VideoFrameFormat::kRgb8};
};

struct VideoSegmentationRequest {
    std::vector<VideoFrameView> frames;
    std::string text_prompt;
};

struct VideoSegmentationFrameResult {
    std::vector<std::uint8_t> masks;
    std::vector<std::int32_t> object_ids;
    std::vector<float> detection_scores;
    std::vector<float> tracking_scores;
    std::vector<float> boxes;
    std::int32_t num_objects{0};
    std::int32_t height{0};
    std::int32_t width{0};
};

struct VideoSegmentationResult {
    std::vector<VideoSegmentationFrameResult> frames;
};

class IVideoSegmentationSession {
  public:
    virtual ~IVideoSegmentationSession() = default;
    virtual VideoSegmentationResult segment(const VideoSegmentationRequest& request) = 0;
};

struct TextGenerationConfig {
    std::int32_t max_new_tokens{128};
    std::int32_t source_language_token_id{-1};
    std::int32_t forced_bos_token_id{-1};
    float temperature{1.0F};
    std::int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    std::int32_t seed{-1};
    float guidance_scale{-1.0F};
    float cfg_scale{-1.0F};
    std::int32_t num_steps{-1};
    float sde_gamma{-1.0F};
    std::vector<float> initial_latents;
    std::vector<float> condition_latents;
    std::vector<float> condition_mask;
    std::vector<float> sampling_steps;
    std::vector<float> sde_noises;
    std::int32_t eos_token_id{-1};
    std::string text_generation_mode{"auto"};
    std::int32_t block_length{0};
    float confidence_threshold{-1.0F};
    bool use_chat_template{false};
    bool enable_thinking{true};
    bool stop_on_boxed_answer{false};
    std::int32_t stop_check_interval{16};
    std::string lora_adapter_id;
    float repetition_penalty{1.0F};
};

struct ImageGenerationConfig {
    std::int32_t num_samples{1};
    std::int32_t seed{-1};
    float guidance_scale{-1.0F};
    float cfg_scale{-1.0F};
    std::int32_t num_steps{-1};
    float sde_gamma{-1.0F};
    std::vector<float> initial_latents;
    std::string negative_prompt;
    std::int32_t height{0};
    std::int32_t width{0};
};

struct WorldModelRequest {
    std::string prompt;
    // Contiguous row-major HWC float pixels for the conditioning frame.
    std::vector<float> image;
    std::int32_t image_height{0};
    std::int32_t image_width{0};
    std::int32_t image_channels{3};
    std::string action;
    // Either fx,fy,cx,cy; one row-major 3x3 matrix; or one 3x3 matrix per frame.
    std::vector<float> camera_intrinsics;
    std::int32_t num_frames{0};
    ImageGenerationConfig generation;
};

struct AudioGenerationConfig {
    std::int32_t max_new_tokens{128};
    std::int32_t talker_max_new_tokens{0};
    std::int32_t seed{-1};
};

using AudioChunkCallback =
    std::function<void(const float* samples, std::int32_t num_samples, std::int32_t sample_rate)>;

struct SpeechToSpeechConfig {
    std::int32_t max_new_tokens{128};
    std::int32_t seed{-1};
    std::int32_t tail_frames{0};
};

class ITranscriptionStream {
  public:
    virtual ~ITranscriptionStream() = default;
    virtual TranscriptionStreamResult
    accept_audio(const float* audio_samples, std::int32_t num_samples, bool is_final = false) = 0;
    virtual TranscriptionStreamResult finish() = 0;
    virtual void reset() = 0;
    virtual TranscriptionStreamConfig config() const = 0;
};

enum class SpeechSessionEventKind {
    kAgentAudio,
    kAgentText,
    kUserTranscript,
    kTurnStarted,
    kTurnFinished,
    kYielded,
    kCancelled,
    kReset,
    kError,
    kInputFinished,
    kUserSpeechStarted,
    kUserSpeechStopped,
    kFunctionCall,
    kFunctionCallStarted,
    kFunctionResponseFinished,
    kInputCleared,
};

struct SpeechSessionEvent {
    SpeechSessionEventKind kind{SpeechSessionEventKind::kError};
    std::uint64_t epoch{0};
    std::uint64_t sequence{0};
    std::vector<float> audio_samples;
    std::int32_t sample_rate{0};
    std::int64_t media_start_sample{-1};
    std::int64_t media_end_sample{-1};
    std::int64_t frame_index{-1};
    std::string text;
    bool is_final{false};
};

struct SpeechSessionConfig {
    std::int32_t input_sample_rate{16000};
    std::int32_t output_sample_rate{0};
    std::string system_prompt;
    bool emit_agent_audio{true};
    bool emit_agent_text{true};
    bool emit_user_transcript{true};
    bool enable_barge_in{true};
    std::int32_t seed{0};
    std::int32_t finish_tail_frames{-1};
};

struct SpeechToolSessionConfig {
    std::string tools_json;
    std::string on_hold_messages_json;
};

class ISpeechSession {
  public:
    virtual ~ISpeechSession() = default;
    virtual void append_audio(const float* audio_samples, std::int32_t num_samples) = 0;
    virtual void finish_input() = 0;
    virtual std::vector<SpeechSessionEvent> take_events() = 0;
    virtual std::vector<SpeechSessionEvent> wait_events(std::int32_t timeout_ms) = 0;
    virtual void cancel() = 0;
    virtual void reset() = 0;
    virtual SpeechSessionConfig config() const = 0;
};

class ISpeechRealtimeControl {
  public:
    virtual ~ISpeechRealtimeControl() = default;
    virtual void commit_input_turn(bool create_response = true) = 0;
    virtual void create_response() = 0;
    virtual void clear_pending_input() = 0;
    virtual void cancel_response() = 0;
    virtual void truncate_response(std::uint64_t epoch, std::int64_t played_output_samples) = 0;
};

class ISpeechToolSession {
  public:
    virtual ~ISpeechToolSession() = default;
    virtual void submit_tool_response(std::uint64_t epoch, const std::string& call_id,
                                      const std::string& output) = 0;
};

class ITextGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "text_generation";
    const char* task() const noexcept override { return kTask; }
    virtual std::int32_t default_max_new_tokens() const = 0;
    virtual TextResult generate(const std::string& prompt,
                                const TextGenerationConfig& config = {}) = 0;
};

class IVisionLanguageGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "vision_language_generation";
    const char* task() const noexcept override { return kTask; }
    virtual TextResult generate(const std::string& prompt, const float* image_pixels,
                                std::int32_t image_height, std::int32_t image_width,
                                const TextGenerationConfig& config = {}) = 0;
};

class IImageGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "image_generation";
    const char* task() const noexcept override { return kTask; }
    virtual ImageResult generate_image(const std::string& prompt,
                                       const ImageGenerationConfig& config = {}) = 0;
};

class IImageEditing : public virtual ITask {
  public:
    static constexpr const char* kTask = "image_edit";
    const char* task() const noexcept override { return kTask; }
    virtual ImageResult generate_image(const std::string& prompt, const float* image_pixels,
                                       std::int32_t image_height, std::int32_t image_width,
                                       const ImageGenerationConfig& config = {}) = 0;
};

class IImageBatchGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "image_generation_batch";
    const char* task() const noexcept override { return kTask; }
    virtual std::vector<ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts,
                         const std::vector<std::uint32_t>& per_sample_seeds,
                         const ImageGenerationConfig& config = {}) = 0;
};

class IWorldModelGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "world_model_generation";
    const char* task() const noexcept override { return kTask; }
    virtual ImageResult generate_world(const WorldModelRequest& request) = 0;
};

class IAudioGeneration : public virtual ITask {
  public:
    static constexpr const char* kTask = "audio_generation";
    const char* task() const noexcept override { return kTask; }
    virtual AudioResult generate_audio(const std::string& prompt,
                                       const AudioGenerationConfig& config = {}) = 0;
};

// Optional task capability for families that produce audio before an utterance
// is complete. It is separate from IAudioGeneration so non-streaming families
// never need an adapter or a fake implementation.
class IStreamingAudioGeneration {
  public:
    static constexpr const char* kTask = IAudioGeneration::kTask;
    virtual ~IStreamingAudioGeneration() = default;
    virtual std::int32_t generate_audio_streaming(const std::string& prompt,
                                                  const AudioGenerationConfig& config,
                                                  AudioChunkCallback callback,
                                                  std::int32_t chunk_frames) = 0;
};

class ITranscription : public virtual ITask {
  public:
    static constexpr const char* kTask = "transcription";
    const char* task() const noexcept override { return kTask; }
    virtual TextResult transcribe(const float* audio_samples, std::int32_t num_samples,
                                  const TranscriptionConfig& config = {}) = 0;
};

class IBatchTranscription : public virtual ITask {
  public:
    static constexpr const char* kTask = "transcription_batch";
    const char* task() const noexcept override { return kTask; }
    virtual std::vector<TextResult>
    transcribe_batch(const std::vector<TranscriptionRequest>& requests) = 0;
};

class IStreamingTranscription : public virtual ITask {
  public:
    static constexpr const char* kTask = "transcription_streaming";
    const char* task() const noexcept override { return kTask; }
    virtual std::unique_ptr<ITranscriptionStream>
    create_transcription_stream(const TranscriptionStreamConfig& config = {}) = 0;
};

class ISpeechToSpeech : public virtual ITask {
  public:
    static constexpr const char* kTask = "speech_to_speech";
    const char* task() const noexcept override { return kTask; }
    virtual AudioResult speak(const float* audio_samples, std::int32_t num_samples,
                              const SpeechToSpeechConfig& config = {},
                              std::int32_t input_sample_rate = 0) = 0;
};

class ISpeechSessionProvider : public virtual ITask {
  public:
    static constexpr const char* kTask = "speech_session";
    const char* task() const noexcept override { return kTask; }
    virtual std::unique_ptr<ISpeechSession>
    create_speech_session(const SpeechSessionConfig& config = {}) = 0;
};

class ISpeechBatchSessionProvider : public virtual ITask {
  public:
    static constexpr const char* kTask = "speech_batch_session";
    const char* task() const noexcept override { return kTask; }
    virtual std::unique_ptr<ISpeechSession>
    create_batch_speech_session(const SpeechSessionConfig& config = {}) = 0;
};

class ISpeechToolSessionProvider : public virtual ITask {
  public:
    static constexpr const char* kTask = "speech_tool_session";
    const char* task() const noexcept override { return kTask; }
    virtual std::unique_ptr<ISpeechSession>
    create_tool_speech_session(const SpeechSessionConfig& session_config,
                               const SpeechToolSessionConfig& tool_config) = 0;
};

class IEmbedding : public virtual ITask {
  public:
    static constexpr const char* kTask = "embedding";
    const char* task() const noexcept override { return kTask; }
    virtual EmbeddingResult embed(const std::string& text) = 0;
};

class IEncoding : public virtual ITask {
  public:
    static constexpr const char* kTask = "encoding";
    const char* task() const noexcept override { return kTask; }
    virtual EmbeddingResult encode(const std::string& text) = 0;
};

class IReranking : public virtual ITask {
  public:
    static constexpr const char* kTask = "reranking";
    const char* task() const noexcept override { return kTask; }
    virtual float rerank(const std::string& query, const std::string& document) = 0;
    virtual std::vector<float> rerank_batch(const std::string& query,
                                            const std::vector<std::string>& documents) = 0;
};

class ISegmentation : public virtual ITask {
  public:
    static constexpr const char* kTask = "segmentation";
    const char* task() const noexcept override { return kTask; }
    virtual SegmentResult segment(const float* pixels, std::int32_t height, std::int32_t width) = 0;
};

class IPointPromptedSegmentation : public virtual ITask {
  public:
    static constexpr const char* kTask = "prompted_segmentation";
    const char* task() const noexcept override { return kTask; }
    virtual PromptedSegmentationResult
    segment_prompted(const float* image_pixels, std::int32_t image_height, std::int32_t image_width,
                     float point_x = 0.5F, float point_y = 0.5F, bool is_foreground = true) = 0;
};

class ITextPromptedSegmentation : public virtual ITask {
  public:
    static constexpr const char* kTask = "text_prompted_segmentation";
    const char* task() const noexcept override { return kTask; }
    virtual PromptedSegmentationResult segment_prompted_text(const float* image_pixels,
                                                             std::int32_t image_height,
                                                             std::int32_t image_width,
                                                             const std::string& text_prompt) = 0;
};

class IStereoDisparity : public virtual ITask {
  public:
    static constexpr const char* kTask = "stereo_disparity";
    const char* task() const noexcept override { return kTask; }
    virtual StereoDisparityResult estimate_disparity(const float* left_pixels,
                                                     const float* right_pixels, std::int32_t height,
                                                     std::int32_t width) = 0;
};

class IMonocularGeometry : public virtual ITask {
  public:
    static constexpr const char* kTask = "monocular_geometry";
    const char* task() const noexcept override { return kTask; }
    virtual GeometryResult estimate_geometry(const float* pixels, std::int32_t height,
                                             std::int32_t width) = 0;
};

class IImageClassification : public virtual ITask {
  public:
    static constexpr const char* kTask = "classification";
    const char* task() const noexcept override { return kTask; }
    virtual ClassificationResult classify(const float* pixels, std::int32_t height,
                                          std::int32_t width) = 0;
};

class IPoseHypothesisRefinement : public virtual ITask {
  public:
    static constexpr const char* kTask = "pose_hypothesis_refinement";
    const char* task() const noexcept override { return kTask; }
    virtual PoseEstimationResult estimate_pose_hypotheses(const PoseEstimationRequest& request) = 0;
    virtual void reset_pose_tracking() = 0;
};

class IImageFeatureExtractor : public virtual ITask {
  public:
    static constexpr const char* kTask = "image_features";
    const char* task() const noexcept override { return kTask; }
    virtual ImageFeaturesResult extract_image_features(const float* pixels, std::int32_t height,
                                                       std::int32_t width) = 0;
};

class IVideoSegmentation : public virtual ITask {
  public:
    static constexpr const char* kTask = "video_segmentation";
    const char* task() const noexcept override { return kTask; }
    virtual std::unique_ptr<IVideoSegmentationSession> create_video_segmentation_session() = 0;
};

class INeuralOperator : public virtual ITask {
  public:
    static constexpr const char* kTask = "neural_operator";
    const char* task() const noexcept override { return kTask; }
    virtual EmbeddingResult solve(const float* branch_input, std::int32_t branch_length,
                                  const float* trunk_input, std::int32_t trunk_length) = 0;
};

struct ForecastRequest {
    Span<const float> past_values;
    // Empty means every supplied past value is observed.
    Span<const float> observed_mask;
    // Zero is the default category; families that do not model frequency reject nonzero values.
    std::int32_t frequency{0};
};

struct ForecastResult {
    std::vector<float> values;
    std::vector<std::int64_t> shape;
};

class ITimeSeriesForecast : public virtual ITask {
  public:
    static constexpr const char* kTask = "time_series_forecast";

    const char* task() const noexcept override { return kTask; }
    virtual ForecastResult forecast(const ForecastRequest& request) = 0;
};

class IRobotControl : public virtual ITask {
  public:
    static constexpr const char* kTask = "robot_control";

    const char* task() const noexcept override { return kTask; }
    virtual RobotActionChunk predict_action_chunk(const RobotObservation& observation) = 0;
    virtual RobotAction act(const RobotObservation& observation) = 0;
    virtual void reset() = 0;
};

class ILoraAdapterManager {
  public:
    virtual ~ILoraAdapterManager() = default;
    virtual void load_lora_adapter(const std::string& adapter_id,
                                   const std::string& adapter_path) = 0;
    virtual void unload_lora_adapter(const std::string& adapter_id) = 0;
    virtual std::vector<std::string> loaded_lora_adapters() const = 0;
};

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-PIP-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         C API pipeline creation via trtmc_create_pipeline_ex
// Preconditions:  TRT runtime available
// Postconditions: Pipeline created or appropriate error returned
// =============================================================================

// =============================================================================
// Test suite: Pipeline C ABI -- IPipeline virtual interface via trtmc_create_pipeline
// =============================================================================
//
// Purpose:
//   Validates the public C ABI entry point trtmc_create_pipeline() and the
//   IPipeline virtual interface it returns. Tests cover null/invalid input
//   handling, version queries, and ABI stability guarantees.
//
// Dependencies:
//   - trtmc/pipeline.h (IPipeline, trtmc_create_pipeline, trtmc_last_error,
//     trtmc_version, trtmc_has_trt)
//   - No TRT, GPU, or model files required.
// =============================================================================

#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "trtmc/speech_session.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Minimal DummyPipeline implementing only the two required pure virtuals.
class DummyPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "dummy-model"; }
    const char* pipeline_type() const override { return "DummyPipeline"; }
};

class RecordingVideoPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "recording-video"; }
    const char* pipeline_type() const override { return "RecordingVideoPipeline"; }

    trtmc::ImageResult generate_image(const std::string& prompt,
                                      const trtmc::GenerateConfig& cfg) override {
        observed_prompt = prompt;
        observed_config = cfg;
        trtmc::ImageResult result;
        result.pixels = {0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
        result.height = 1;
        result.width = 1;
        result.channels = 3;
        result.num_frames = 2;
        return result;
    }

    std::string observed_prompt;
    trtmc::GenerateConfig observed_config;
};

class RecordingStructuredVideoPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "recording-structured-video"; }
    const char* pipeline_type() const override { return "RecordingStructuredVideoPipeline"; }

    trtmc::VideoResult generate_video(const trtmc::VideoGenerationRequest& request) override {
        observed_request = request;
        trtmc::VideoResult result;
        result.fps = 24;
        return result;
    }

    trtmc::VideoGenerationRequest observed_request;
};

class RecordingTranscriptionPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "recording"; }
    const char* pipeline_type() const override { return "RecordingTranscriptionPipeline"; }

    trtmc::TextResult transcribe(const float* audio, int32_t count,
                                 const trtmc::TranscriptionConfig& cfg) override {
        observed.push_back(cfg);
        return {std::to_string(count) + ":" + std::to_string(audio == nullptr ? -1 : audio[0]),
                {cfg.beam_size}};
    }

    std::vector<trtmc::TranscriptionConfig> observed;
};

class RecordingSpeechSession final : public trtmc::ISpeechSession,
                                     public trtmc::ISpeechRealtimeControl,
                                     public trtmc::ISpeechToolSession {
  public:
    explicit RecordingSpeechSession(trtmc::SpeechSessionConfig cfg) : cfg_(std::move(cfg)) {}

    void append_audio(const float* audio_samples, int32_t num_samples) override {
        ++append_calls;
        if (audio_samples != nullptr && num_samples > 0)
            accepted_audio.insert(accepted_audio.end(), audio_samples, audio_samples + num_samples);
        trtmc::SpeechSessionEvent event;
        event.kind = trtmc::SpeechSessionEventKind::kAgentAudio;
        event.epoch = epoch;
        event.sequence = next_sequence++;
        event.audio_samples = {0.25F, -0.25F};
        event.sample_rate = cfg_.output_sample_rate == 0 ? 22050 : cfg_.output_sample_rate;
        event.media_start_sample = 0;
        event.media_end_sample = 2;
        event.frame_index = 0;
        events.push_back(std::move(event));
    }

    void finish_input() override {
        finished = true;
        trtmc::SpeechSessionEvent event;
        event.kind = trtmc::SpeechSessionEventKind::kTurnFinished;
        event.epoch = epoch;
        event.sequence = next_sequence++;
        event.is_final = true;
        events.push_back(std::move(event));
    }

    std::vector<trtmc::SpeechSessionEvent> take_events() override {
        auto result = std::move(events);
        events.clear();
        return result;
    }

    void cancel() override {
        cancelled = true;
        ++epoch;
        trtmc::SpeechSessionEvent event;
        event.kind = trtmc::SpeechSessionEventKind::kCancelled;
        event.epoch = epoch;
        event.sequence = 0;
        event.is_final = true;
        events.push_back(std::move(event));
        next_sequence = 1;
    }

    void reset() override {
        cancelled = false;
        finished = false;
        accepted_audio.clear();
        ++epoch;
        trtmc::SpeechSessionEvent event;
        event.kind = trtmc::SpeechSessionEventKind::kReset;
        event.epoch = epoch;
        event.sequence = 0;
        event.is_final = true;
        events.push_back(std::move(event));
        next_sequence = 1;
    }

    trtmc::SpeechSessionConfig config() const override { return cfg_; }

    void commit_input_turn(bool create_response) override {
        ++commit_input_turn_calls;
        commit_create_response = create_response;
    }

    void create_response() override { ++create_response_calls; }

    void clear_pending_input() override {
        ++clear_pending_input_calls;
        trtmc::SpeechSessionEvent event;
        event.kind = trtmc::SpeechSessionEventKind::kInputCleared;
        event.epoch = epoch;
        event.sequence = next_sequence++;
        event.is_final = true;
        events.push_back(std::move(event));
    }

    void cancel_response() override { ++cancel_response_calls; }

    void truncate_response(std::uint64_t response_epoch,
                           std::int64_t played_output_samples) override {
        truncate_response_epoch = response_epoch;
        truncate_played_output_samples = played_output_samples;
    }

    void submit_tool_response(std::uint64_t response_epoch, const std::string& call_id,
                              const std::string& output) override {
        tool_response_epoch = response_epoch;
        tool_response_call_id = call_id;
        tool_response_output = output;
    }

    trtmc::SpeechSessionConfig cfg_;
    std::vector<float> accepted_audio;
    std::vector<trtmc::SpeechSessionEvent> events;
    std::uint64_t epoch{1};
    std::uint64_t next_sequence{0};
    int append_calls{0};
    bool finished{false};
    bool cancelled{false};
    int commit_input_turn_calls{0};
    int create_response_calls{0};
    bool commit_create_response{false};
    int clear_pending_input_calls{0};
    int cancel_response_calls{0};
    std::uint64_t truncate_response_epoch{0};
    std::int64_t truncate_played_output_samples{0};
    std::uint64_t tool_response_epoch{0};
    std::string tool_response_call_id;
    std::string tool_response_output;
};

class RecordingSpeechPipeline final : public trtmc::IPipeline,
                                      public trtmc::ISpeechSessionProvider,
                                      public trtmc::ISpeechBatchSessionProvider,
                                      public trtmc::ISpeechToolSessionProvider {
  public:
    const char* model_id() const override { return "recording-speech"; }
    const char* pipeline_type() const override { return "RecordingSpeechPipeline"; }
    std::unique_ptr<trtmc::ISpeechSession>
    create_speech_session(const trtmc::SpeechSessionConfig& cfg) override {
        ++live_factory_calls;
        observed_config = cfg;
        auto result = std::make_unique<RecordingSpeechSession>(cfg);
        last_session = result.get();
        return result;
    }

    std::unique_ptr<trtmc::ISpeechSession>
    create_batch_speech_session(const trtmc::SpeechSessionConfig& cfg) override {
        ++batch_factory_calls;
        observed_config = cfg;
        auto result = std::make_unique<RecordingSpeechSession>(cfg);
        last_session = result.get();
        return result;
    }

    std::unique_ptr<trtmc::ISpeechSession>
    create_tool_speech_session(const trtmc::SpeechSessionConfig& cfg,
                               const trtmc::SpeechToolSessionConfig& tool_cfg) override {
        observed_tool_config = tool_cfg;
        return create_speech_session(cfg);
    }

    trtmc::SpeechSessionConfig observed_config;
    trtmc::SpeechToolSessionConfig observed_tool_config;
    RecordingSpeechSession* last_session{nullptr};
    int live_factory_calls{0};
    int batch_factory_calls{0};
};

class PoolTestPipeline final : public trtmc::IPipeline {
  public:
    explicit PoolTestPipeline(std::string id) : id_(std::move(id)) {}

    const char* model_id() const override { return id_.c_str(); }
    const char* pipeline_type() const override { return "PoolTestPipeline"; }
    bool supports_lora_adapters() const override { return true; }

    void load_lora_adapter(const std::string& adapter_id, const std::string&) override {
        adapters_.insert(adapter_id);
    }

    void unload_lora_adapter(const std::string& adapter_id) override {
        if (adapters_.erase(adapter_id) == 0)
            throw std::invalid_argument("unknown adapter");
    }

    std::vector<std::string> loaded_lora_adapters() const override {
        return {adapters_.begin(), adapters_.end()};
    }

  private:
    std::string id_;
    std::set<std::string> adapters_;
};

static void test_null_input_returns_null() {
    auto* p = trtmc_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after null input");
}

static void test_invalid_path_returns_null() {
    auto* p = trtmc_create_pipeline("/nonexistent/path/to/bundle.bundle", 0);
    check(p == nullptr, "invalid path returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after invalid path");
}

static void test_version_available() {
    const char* ver = trtmc_version();
    check(ver != nullptr, "version is non-null");
    check(std::strlen(ver) > 0, "version is non-empty");
}

static void test_has_trt_returns_bool() {
    const int val = trtmc_has_trt();
    check(val == 0 || val == 1, "trtmc_has_trt returns 0 or 1");
}

static void test_sizeof_ipipeline_is_vtable() {
    check(sizeof(trtmc::IPipeline) == sizeof(void*),
          "sizeof(IPipeline) equals vtable pointer size");
}

static void test_delete_null_safe() {
    trtmc::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

// Exercise default virtual methods -- they should throw with descriptive messages.
static void test_ipipeline_default_virtuals() {
    DummyPipeline pipeline;

    check(std::string(pipeline.model_id()) == "dummy-model", "model_id");
    check(std::string(pipeline.pipeline_type()) == "DummyPipeline", "pipeline_type");
    trtmc::IPipeline* base = &pipeline;
    check(dynamic_cast<trtmc::ISpeechSessionProvider*>(base) == nullptr,
          "speech session capability is absent by default");
    check(dynamic_cast<trtmc::ISpeechBatchSessionProvider*>(base) == nullptr,
          "speech batch-session capability is absent by default");
    check(dynamic_cast<trtmc::ISpeechRealtimeControl*>(base) == nullptr,
          "speech realtime-control capability is absent by default");
    check(dynamic_cast<trtmc::ISpeechToolSessionProvider*>(base) == nullptr,
          "speech tool-session capability is absent by default");

    // Default generate(string) should throw
    bool threw = false;
    try {
        pipeline.generate("hello");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate(string) throws");

    // Default generate(string, image) should throw
    threw = false;
    try {
        pipeline.generate("hello", nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate(string,image) throws");

    // Default generate_image should throw
    threw = false;
    try {
        pipeline.generate_image("prompt");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_image throws");

    // Default generate_image(string, image) should throw via the text-only
    // image-generation overload.
    threw = false;
    try {
        pipeline.generate_image("prompt", nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_image(string,image) throws");

    // Default generate_audio should throw
    threw = false;
    try {
        pipeline.generate_audio("hello");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default generate_audio throws");

    // Default transcribe should throw
    threw = false;
    try {
        pipeline.transcribe(nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default transcribe throws");

    // Default streaming transcription should throw
    threw = false;
    try {
        pipeline.create_transcription_stream();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default create_transcription_stream throws");

    threw = false;
    try {
        trtmc::TranscriptionStreamConfig cfg;
        pipeline.transcribe_streaming(nullptr, 0, cfg);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default transcribe_streaming throws");

    // Default speak should throw
    threw = false;
    try {
        pipeline.speak(nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default speak throws");

    // Default embed should throw
    threw = false;
    try {
        pipeline.embed("text");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default embed throws");

    // Default rerank should throw
    threw = false;
    try {
        pipeline.rerank("q", "doc");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default rerank throws");

    // Default segment should throw
    threw = false;
    try {
        pipeline.segment(nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default segment throws");

    // Default encode should throw
    threw = false;
    try {
        pipeline.encode("text");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default encode throws");

    // Default solve should throw
    threw = false;
    try {
        pipeline.solve(nullptr, 0, nullptr, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default solve throws");

    // Default detect should throw
    threw = false;
    try {
        pipeline.detect(nullptr, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "default detect throws");
}

static void test_video_result_and_legacy_generation_contract() {
    trtmc::GenerateConfig defaults;
    check(defaults.video_num_frames == 0 && defaults.height == 0 && defaults.width == 0,
          "video geometry defaults to model selection");

    RecordingVideoPipeline pipeline;
    trtmc::GenerateConfig cfg;
    cfg.seed = 17;
    cfg.video_num_frames = 345;
    const auto result = pipeline.generate_video("a native video", cfg);

    check(pipeline.observed_prompt == "a native video" && pipeline.observed_config.seed == 17 &&
              pipeline.observed_config.video_num_frames == 345,
          "default generate_video delegates prompt and config to generate_image");
    check(result.frames.width == 1 && result.frames.height == 1 && result.frames.num_frames == 2 &&
              result.frames.pixels.size() == 6,
          "default generate_video returns legacy video frames");
    check(result.audio.samples.empty() && result.audio.channels == 1 && result.fps == 0,
          "legacy video generation defaults to no audio and unknown fps");

    trtmc::VideoGenerationRequest t2va;
    t2va.prompt = "structured text only";
    t2va.config.height = 768;
    t2va.config.width = 1344;
    t2va.config.video_num_frames = 124;
    const auto structured_result = pipeline.generate_video(t2va);
    check(pipeline.observed_prompt == "structured text only" &&
              pipeline.observed_config.height == 768 && pipeline.observed_config.width == 1344 &&
              pipeline.observed_config.video_num_frames == 124 &&
              structured_result.frames.num_frames == 2,
          "structured T2VA defaults to the legacy video overload");

    t2va.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    t2va.first_frame = trtmc::VideoImageInput{{0.0F, 0.5F, 1.0F}, 1, 1, 3};
    bool rejected_multimodal = false;
    try {
        (void)pipeline.generate_video(t2va);
    } catch (const std::runtime_error&) {
        rejected_multimodal = true;
    }
    check(rejected_multimodal, "legacy video overload rejects multimodal requests explicitly");

    trtmc::VideoResult synchronized;
    synchronized.audio.samples = {0.25F, -0.25F, 0.5F, -0.5F};
    synchronized.audio.num_samples = 4;
    synchronized.audio.sample_rate = 32000;
    synchronized.audio.channels = 2;
    synchronized.fps = 24;
    check(synchronized.audio.samples.size() == 4 && synchronized.audio.num_samples == 4 &&
              synchronized.audio.sample_rate == 32000 && synchronized.audio.channels == 2 &&
              synchronized.fps == 24,
          "VideoResult carries 32 kHz interleaved stereo audio and fps");
}

static void test_structured_video_generation_request_contract() {
    trtmc::VideoGenerationRequest request;
    request.prompt = "use these references in order";
    request.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    request.config.height = 768;
    request.config.width = 1344;
    request.config.video_num_frames = 345;

    trtmc::VideoReferenceInput image;
    image.kind = trtmc::VideoReferenceKind::kImage;
    image.image = {{0.1F, 0.2F, 0.3F}, 1, 1, 3};
    request.references.push_back(image);

    trtmc::VideoReferenceInput video;
    video.kind = trtmc::VideoReferenceKind::kVideo;
    video.video.pixels = {0.4F, 0.5F, 0.6F};
    video.video.num_frames = 1;
    video.video.height = 1;
    video.video.width = 1;
    video.video.channels = 3;
    video.video.fps_numerator = 30000;
    video.video.fps_denominator = 1001;
    video.video.soundtrack.samples = {0.25F, -0.25F};
    video.video.soundtrack.num_samples = 2;
    video.video.soundtrack.sample_rate = 32000;
    video.video.soundtrack.channels = 2;
    request.references.push_back(video);

    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    audio.audio.samples = {0.75F, -0.75F};
    audio.audio.num_samples = 2;
    audio.audio.sample_rate = 44100;
    audio.audio.channels = 1;
    request.references.push_back(audio);

    RecordingStructuredVideoPipeline pipeline;
    trtmc::IPipeline* base = &pipeline;
    const auto result = base->generate_video(request);
    const auto& observed = pipeline.observed_request;
    check(result.fps == 24 && observed.mode == trtmc::VideoGenerationMode::kReferenceToVideoAudio &&
              observed.config.height == 768 && observed.config.width == 1344 &&
              observed.config.video_num_frames == 345,
          "structured video request dispatches through the appended virtual overload");
    check(observed.references.size() == 3 &&
              observed.references[0].kind == trtmc::VideoReferenceKind::kImage &&
              observed.references[1].kind == trtmc::VideoReferenceKind::kVideo &&
              observed.references[2].kind == trtmc::VideoReferenceKind::kAudio,
          "Ref2VA preserves ordered heterogeneous references");
    check(observed.references[1].video.fps_numerator == 30000 &&
              observed.references[1].video.fps_denominator == 1001 &&
              observed.references[1].video.soundtrack.channels == 2 &&
              observed.references[2].audio.sample_rate == 44100,
          "decoded video and audio reference metadata survives API dispatch");
}

static void test_speech_batch_session_optional_interface() {
    RecordingSpeechPipeline pipeline;
    auto* provider = dynamic_cast<trtmc::ISpeechBatchSessionProvider*>(&pipeline);
    check(provider != nullptr, "speech batch-session capability is explicit");

    trtmc::SpeechSessionConfig cfg;
    cfg.input_sample_rate = 24000;
    cfg.enable_barge_in = true;
    cfg.seed = 31;
    auto session = provider->create_batch_speech_session(cfg);
    check(session != nullptr && pipeline.batch_factory_calls == 1 &&
              pipeline.live_factory_calls == 0,
          "batch speech factory is independent from the live factory");
    check(session->config().input_sample_rate == 24000 && session->config().enable_barge_in &&
              session->config().seed == 31,
          "batch speech factory preserves the public session config");
}

static void test_speech_session_value_contract() {
    const trtmc::SpeechSessionConfig defaults;
    check(defaults.input_sample_rate == 16000, "speech session default input sample rate");
    check(defaults.output_sample_rate == 0, "speech session default output uses model rate");
    check(defaults.emit_agent_audio && defaults.emit_agent_text && defaults.emit_user_transcript,
          "speech session optional event streams default enabled");
    check(defaults.enable_barge_in, "live speech sessions enable barge-in by default");
    check(defaults.seed == 0, "native speech sessions default to deterministic seed zero");
    check(defaults.finish_tail_frames == -1,
          "live speech sessions use the model-owned bounded response tail by default");

    trtmc::SpeechSessionEvent event;
    event.kind = trtmc::SpeechSessionEventKind::kAgentText;
    event.epoch = 7;
    event.sequence = 12;
    event.sample_rate = 22050;
    event.media_start_sample = 1764;
    event.media_end_sample = 3528;
    event.frame_index = 1;
    event.text = "one moment";
    event.audio_samples = {0.5F};
    event.is_final = true;

    check(event.is_final && event.epoch == 7 && event.sequence == 12,
          "speech session event carries final and ordering state");
    check(event.audio_samples == std::vector<float>{0.5F} && event.sample_rate == 22050,
          "speech session event carries audio");
    check(event.text == "one moment" && event.media_start_sample == 1764 &&
              event.media_end_sample == 3528 && event.frame_index == 1,
          "speech session event carries text and media position");
}

static void test_speech_session_virtual_interface() {
    RecordingSpeechPipeline pipeline;
    auto* provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(&pipeline);
    check(provider != nullptr, "speech session capability is explicit");
    trtmc::SpeechSessionConfig cfg;
    cfg.input_sample_rate = 44100;
    cfg.output_sample_rate = 48000;
    cfg.system_prompt = "Speak briefly";
    cfg.enable_barge_in = false;
    cfg.seed = 17;

    auto session = provider->create_speech_session(cfg);
    check(session != nullptr && pipeline.last_session != nullptr,
          "speech session factory returns a session");
    check(pipeline.observed_config.input_sample_rate == 44100 &&
              session->config().output_sample_rate == 48000 && !session->config().enable_barge_in &&
              session->config().seed == 17,
          "speech session factory preserves arbitrary sample-rate config");

    const float audio[] = {1.0F, 0.5F, -0.5F};
    session->append_audio(audio, 3);
    check(pipeline.last_session->append_calls == 1 &&
              pipeline.last_session->accepted_audio == std::vector<float>({1.0F, 0.5F, -0.5F}),
          "speech session append preserves a persistent input stream");
    auto events = session->take_events();
    check(events.size() == 1 && events[0].kind == trtmc::SpeechSessionEventKind::kAgentAudio &&
              events[0].sample_rate == 48000,
          "speech session drains currently available agent output");
    check(session->take_events().empty(),
          "speech session event drain is non-blocking and consuming");
    check(session->wait_events(0).empty(),
          "appended speech wait API preserves synchronous session behavior");

    session->finish_input();
    events = session->take_events();
    check(pipeline.last_session->finished && events.size() == 1 && events[0].is_final &&
              events[0].kind == trtmc::SpeechSessionEventKind::kTurnFinished,
          "speech session finish is explicit and observable");

    const auto epoch_before_cancel = pipeline.last_session->epoch;
    session->cancel();
    events = session->take_events();
    check(pipeline.last_session->cancelled && pipeline.last_session->epoch > epoch_before_cancel &&
              events.size() == 1 && events[0].kind == trtmc::SpeechSessionEventKind::kCancelled,
          "speech session cancel invalidates outstanding output");

    const auto epoch_before_reset = pipeline.last_session->epoch;
    session->reset();
    events = session->take_events();
    check(!pipeline.last_session->cancelled && !pipeline.last_session->finished &&
              pipeline.last_session->epoch > epoch_before_reset && events.size() == 1 &&
              events[0].kind == trtmc::SpeechSessionEventKind::kReset,
          "speech session reset starts a fresh conversation epoch");

    auto* realtime_control = dynamic_cast<trtmc::ISpeechRealtimeControl*>(session.get());
    check(realtime_control != nullptr, "speech realtime-control capability is explicit");

    realtime_control->commit_input_turn(false);
    check(pipeline.last_session->commit_input_turn_calls == 1 &&
              !pipeline.last_session->commit_create_response,
          "speech realtime control preserves explicit response creation");
    realtime_control->commit_input_turn();
    check(pipeline.last_session->commit_input_turn_calls == 2 &&
              pipeline.last_session->commit_create_response,
          "speech realtime control defaults to creating a response");
    realtime_control->create_response();
    check(pipeline.last_session->create_response_calls == 1,
          "speech realtime control can create a response after a commit-only turn");

    realtime_control->clear_pending_input();
    events = session->take_events();
    check(events.size() == 1 && events[0].kind == trtmc::SpeechSessionEventKind::kInputCleared,
          "speech input clear completion is observable without changing the control ABI");
    realtime_control->cancel_response();
    realtime_control->truncate_response(23, 3528);
    check(pipeline.last_session->clear_pending_input_calls == 1 &&
              pipeline.last_session->cancel_response_calls == 1 &&
              pipeline.last_session->truncate_response_epoch == 23 &&
              pipeline.last_session->truncate_played_output_samples == 3528,
          "speech realtime control preserves lifecycle and truncation arguments");
}

static void test_speech_tool_session_optional_interface() {
    RecordingSpeechPipeline pipeline;
    auto* provider = dynamic_cast<trtmc::ISpeechToolSessionProvider*>(&pipeline);
    check(provider != nullptr, "speech tool-session capability is explicit");

    trtmc::SpeechToolSessionConfig tool_config;
    tool_config.tools_json =
        R"([{"type":"function","name":"get_weather","parameters":{"type":"object"}}])";
    tool_config.on_hold_messages_json = R"({"get_weather":"One moment."})";
    auto session = provider->create_tool_speech_session({}, tool_config);
    check(session != nullptr &&
              pipeline.observed_tool_config.tools_json == tool_config.tools_json &&
              pipeline.observed_tool_config.on_hold_messages_json ==
                  tool_config.on_hold_messages_json,
          "tool-session factory preserves tool configuration");

    auto* tool_session = dynamic_cast<trtmc::ISpeechToolSession*>(session.get());
    check(tool_session != nullptr, "tool-session response capability is explicit");
    tool_session->submit_tool_response(9, "call-9-1", R"({"temperature":72})");
    check(pipeline.last_session->tool_response_epoch == 9 &&
              pipeline.last_session->tool_response_call_id == "call-9-1" &&
              pipeline.last_session->tool_response_output == R"({"temperature":72})",
          "tool response preserves epoch, call id, and output");

    trtmc::SpeechSessionEvent event;
    event.kind = trtmc::SpeechSessionEventKind::kFunctionCall;
    event.epoch = 9;
    event.sequence = 1;
    event.text = R"({"call_id":"call-9-1","name":"get_weather","arguments":{}})";
    check(event.kind == trtmc::SpeechSessionEventKind::kFunctionCall && event.epoch == 9 &&
              event.sequence == 1 && event.text.find("call-9-1") != std::string::npos,
          "function-call event reuses the ABI-stable text and ordering fields");
}

static void test_transcription_batch_preserves_per_request_config() {
    RecordingTranscriptionPipeline pipeline;
    trtmc::TranscriptionRequest first;
    first.audio_samples = {1.0F, 2.0F};
    first.config.beam_size = 1;
    first.config.source_language = "en";
    trtmc::TranscriptionRequest second;
    second.audio_samples = {3.0F};
    second.config.beam_size = 4;
    second.config.source_language = "fr";

    const auto results = pipeline.transcribe_batch({first, second});
    check(results.size() == 2, "transcription batch result count");
    check(results[0].token_ids == std::vector<int32_t>({1}) &&
              results[1].token_ids == std::vector<int32_t>({4}),
          "transcription batch preserves request order");
    check(pipeline.observed.size() == 2 && pipeline.observed[0].source_language == "en" &&
              pipeline.observed[1].source_language == "fr",
          "transcription batch preserves per-request config");
}

static void test_pipeline_pool_leases_and_adapter_maintenance() {
    std::vector<std::unique_ptr<trtmc::IPipeline>> pipelines;
    pipelines.push_back(std::make_unique<PoolTestPipeline>("lane-0"));
    pipelines.push_back(std::make_unique<PoolTestPipeline>("lane-1"));
    trtmc::PipelinePool pool(std::move(pipelines));

    check(pool.size() == 2 && pool.available() == 2, "pipeline pool initial capacity");
    auto first = pool.acquire();
    auto second = pool.acquire();
    check(pool.available() == 0, "pipeline pool leases are exclusive");
    check(std::string(first->model_id()) != std::string(second->model_id()),
          "pipeline pool leases distinct lanes");
    check(!pool.try_acquire().has_value(), "pipeline pool reports exhaustion");

    const std::string released_lane = first->model_id();
    auto waiter = std::async(std::launch::async, [&pool] {
        auto lease = pool.acquire();
        return std::string(lease->model_id());
    });
    check(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "pipeline pool waits when all lanes are busy");
    first = {};
    check(waiter.get() == released_lane, "pipeline pool reuses released lane");
    second = {};

    pool.load_lora_adapter("adapter-a", "/synthetic/adapter");
    check(pool.supports_lora_adapters(), "pipeline pool reports shared adapter capability");
    check(pool.loaded_lora_adapters() == std::vector<std::string>{"adapter-a"},
          "pipeline pool registers adapter across lanes");
    auto lane_a = pool.acquire();
    auto lane_b = pool.acquire();
    check(lane_a->loaded_lora_adapters() == std::vector<std::string>{"adapter-a"} &&
              lane_b->loaded_lora_adapters() == std::vector<std::string>{"adapter-a"},
          "pipeline pool keeps lane adapter registries consistent");
    lane_a = {};
    lane_b = {};
    pool.unload_lora_adapter("adapter-a");
    check(pool.loaded_lora_adapters().empty(), "pipeline pool unloads adapter across lanes");
}

int main() {
    test_null_input_returns_null();
    test_invalid_path_returns_null();
    test_version_available();
    test_has_trt_returns_bool();
    test_sizeof_ipipeline_is_vtable();
    test_delete_null_safe();
    test_ipipeline_default_virtuals();
    test_video_result_and_legacy_generation_contract();
    test_structured_video_generation_request_contract();
    test_speech_session_value_contract();
    test_speech_batch_session_optional_interface();
    test_speech_session_virtual_interface();
    test_speech_tool_session_optional_interface();
    test_transcription_batch_preserves_per_request_config();
    test_pipeline_pool_leases_and_adapter_maintenance();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}

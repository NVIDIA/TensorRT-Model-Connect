/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/audio.hpp"
#include "trtmc/speech.h"
#include "trtmc/tools.hpp"

#include <exception>

namespace trtmc {

struct SpeechInputFormat {
    std::optional<std::uint32_t> sample_rate{};
    std::uint32_t channels{0};
};
struct SpeechAudioFormat {
    std::uint32_t sample_rate{0}, channels{0};
};
struct SpeechSessionInfo {
    SpeechAudioFormat input;
    std::optional<SpeechAudioFormat> output{};
    std::optional<std::string> source_language{}, system_prompt{};
};
struct StreamingSpeechTranscriptionRequest {
    SpeechInputFormat input;
    std::optional<std::string> source_language{};
};
struct SpeechDialogueRequest {
    SpeechInputFormat input;
    std::optional<std::string> system_prompt{};
};
struct ToolAcknowledgement {
    std::string tool_name;
    std::vector<std::string> messages;
};
struct ToolSpeechDialogueRequest {
    SpeechDialogueRequest dialogue;
    std::vector<ToolDefinition> tools;
    std::vector<ToolAcknowledgement> acknowledgements;
    std::optional<std::vector<std::string>> default_acknowledgements{};
};

namespace detail {
inline trtmc_speech_input_format_v1 c_speech_format(const SpeechInputFormat& input) noexcept {
    return {input.sample_rate ? 1U : 0U, input.sample_rate.value_or(0), input.channels};
}
inline trtmc_speech_dialogue_request_v1 c_dialogue(const SpeechDialogueRequest& input) noexcept {
    return {c_speech_format(input.input), input.system_prompt ? 1U : 0U,
            input.system_prompt ? c_string(*input.system_prompt) : trtmc_string_view{}};
}
inline SpeechSessionInfo copy_speech_info(const trtmc_speech_session_info_v1& input) {
    SpeechSessionInfo info{{input.input_sample_rate, input.input_channels}, {}, {}, {}};
    if (input.has_output_format)
        info.output = SpeechAudioFormat{input.output_sample_rate, input.output_channels};
    if (input.has_source_language)
        info.source_language = std::string(string_view(input.source_language));
    if (input.has_system_prompt)
        info.system_prompt = std::string(string_view(input.system_prompt));
    return info;
}
inline Config copy_config(const trtmc_config_view_v1& input) {
    Config result;
    for (std::uint64_t i = 0; i < input.count; ++i)
        result.add(std::string(string_view(input.entries[i].name)),
                   copy_value(input.entries[i].value));
    return result;
}
template <class Table>
void validate_speech_table(const trtmc_api_header* table) {
    if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible speech API table");
}
} // namespace detail

class SpeechTranscriptUpdate {
  public:
    SpeechTranscriptUpdate(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    SpeechTranscriptUpdate(const SpeechTranscriptUpdate&) = delete;
    SpeechTranscriptUpdate& operator=(const SpeechTranscriptUpdate&) = delete;
    SpeechTranscriptUpdate(SpeechTranscriptUpdate&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    SpeechTranscriptUpdate& operator=(SpeechTranscriptUpdate&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    TextResultView transcript() const { return detail::text_result_view(view_.transcript); }
    bool is_final() const noexcept { return view_.is_final != 0; }
    std::uint64_t chunk_index() const noexcept { return view_.chunk_index; }
    std::uint64_t accepted_samples() const noexcept { return view_.accepted_samples; }
    SpeechAudioFormat input_format() const noexcept { return {view_.sample_rate, view_.channels}; }
    trtmc_speech_transcript_update_v1& wire_view() noexcept { return view_; }

  private:
    detail::ResultOwner owner_;
    trtmc_speech_transcript_update_v1 view_{};
};

class SpeechTranscriptionStream {
  public:
    ~SpeechTranscriptionStream() { close(); }
    SpeechTranscriptionStream(const SpeechTranscriptionStream&) = delete;
    SpeechTranscriptionStream& operator=(const SpeechTranscriptionStream&) = delete;
    SpeechTranscriptionStream(SpeechTranscriptionStream&& other) noexcept
        : state_(std::move(other.state_)), api_(other.api_),
          handle_(std::exchange(other.handle_, nullptr)) {}
    SpeechTranscriptionStream& operator=(SpeechTranscriptionStream&& other) noexcept {
        if (this != &other) {
            close();
            state_ = std::move(other.state_);
            api_ = other.api_;
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    SpeechTranscriptUpdate accept_audio(Span<const float> audio, bool final = false) {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            api_->accept_audio(handle_, audio.data(), audio.size(), final ? 1U : 0U, &raw, &error);
        return update(status, raw, error);
    }
    SpeechTranscriptUpdate finish() {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->finish(handle_, &raw, &error);
        return update(status, raw, error);
    }
    void reset() {
        trtmc_error* error = nullptr;
        const auto status = api_->reset(handle_, &error);
        detail::check(state_->api, status, error);
    }
    SpeechSessionInfo info() const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->info(handle_, &raw, &error);
        detail::ResultOwner owner(state_, raw);
        detail::check(state_->api, status, error);
        trtmc_speech_session_info_v1 view{};
        error = nullptr;
        status = api_->info_view(raw, &view, &error);
        detail::check(state_->api, status, error);
        return detail::copy_speech_info(view);
    }
    Config config() const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->config(handle_, &raw, &error);
        detail::ResultOwner owner(state_, raw);
        detail::check(state_->api, status, error);
        trtmc_config_view_v1 view{};
        error = nullptr;
        status = api_->config_view(raw, &view, &error);
        detail::check(state_->api, status, error);
        return detail::copy_config(view);
    }
    void close() noexcept {
        if (handle_)
            api_->release(std::exchange(handle_, nullptr));
    }

  private:
    friend class StreamingSpeechTranscription;
    SpeechTranscriptionStream(std::shared_ptr<detail::ModelState> state,
                              const trtmc_streaming_speech_transcription_api_v1* api,
                              trtmc_asr_stream* handle) noexcept
        : state_(std::move(state)), api_(api), handle_(handle) {}
    SpeechTranscriptUpdate update(trtmc_status status, trtmc_result* raw, trtmc_error* error) {
        SpeechTranscriptUpdate result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->update_view(raw, &result.wire_view(), &error);
        detail::check(state_->api, status, error);
        return result;
    }
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_speech_transcription_api_v1* api_;
    trtmc_asr_stream* handle_;
};
class StreamingSpeechTranscription {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STREAMING_SPEECH_TRANSCRIPTION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechTranscriptionStream create(const StreamingSpeechTranscriptionRequest& input,
                                     const Config& config = {}) const {
        const trtmc_streaming_speech_transcription_request_v1 request{
            detail::c_speech_format(input.input), input.source_language ? 1U : 0U,
            input.source_language ? detail::c_string(*input.source_language) : trtmc_string_view{}};
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_asr_stream* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->create(state_->handle, &request, &options, &raw, &error);
        SpeechTranscriptionStream result(state_, api_, raw);
        detail::check(state_->api, status, error);
        return result;
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_speech_table<trtmc_streaming_speech_transcription_api_v1>(table);
    }

  private:
    friend class Model;
    StreamingSpeechTranscription(std::shared_ptr<detail::ModelState> state,
                                 const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_streaming_speech_transcription_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_speech_transcription_api_v1* api_;
};

enum class AudioDeliveryOutcome : std::uint32_t {
    Complete = TRTMC_AUDIO_DELIVERY_COMPLETE,
    Stopped = TRTMC_AUDIO_DELIVERY_STOPPED
};
struct StreamingAudioSummary {
    std::uint64_t emitted_sample_count, emitted_frame_count;
    SpeechAudioFormat output;
    AudioDeliveryOutcome outcome;
    double setup_ms, inference_ms;
};
namespace detail {
template <class Callback>
struct AudioCallbackContext {
    Callback* callback;
    std::exception_ptr exception;
};
template <class Callback>
trtmc_status TRTMC_CALL audio_callback(void* context, const trtmc_audio_view_v1* input,
                                       trtmc_audio_chunk_reply_v1* reply) noexcept {
    if (!context || !input || !reply)
        return TRTMC_INVALID_ARGUMENT;
    *reply = {};
    auto& state = *static_cast<AudioCallbackContext<Callback>*>(context);
    try {
        const AudioView audio{{input->samples, static_cast<std::size_t>(input->sample_count)},
                              input->has_sample_rate
                                  ? std::optional<std::uint32_t>{input->sample_rate}
                                  : std::nullopt,
                              input->channels};
        using Return = std::invoke_result_t<Callback&, const AudioView&>;
        static_assert(std::is_void_v<Return> || std::is_same_v<Return, bool>,
                      "audio callback must return void or bool");
        if constexpr (std::is_void_v<Return>) {
            (*state.callback)(audio);
            return TRTMC_OK;
        } else
            return (*state.callback)(audio) ? TRTMC_OK : TRTMC_END;
    } catch (const Error& error) {
        state.exception = std::current_exception();
        reply->error_message = c_string(error.what());
        return error.code();
    } catch (const std::bad_alloc&) {
        state.exception = std::current_exception();
        reply->error_message = c_string("audio callback allocation failed");
        return TRTMC_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        state.exception = std::current_exception();
        reply->error_message = c_string(error.what());
        return TRTMC_INTERNAL_ERROR;
    } catch (...) {
        state.exception = std::current_exception();
        reply->error_message = c_string("audio callback failed");
        return TRTMC_INTERNAL_ERROR;
    }
}
} // namespace detail
class StreamingTextToSpeech {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STREAMING_TEXT_TO_SPEECH;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    template <class Callback>
    StreamingAudioSummary run(const TextToSpeechRequest& input, Callback&& callback,
                              const Config& config = {}) const {
        using Function = std::remove_reference_t<Callback>;
        detail::AudioCallbackContext<Function> context{&callback, {}};
        const trtmc_text_to_speech_request_v1 request{
            detail::c_string(input.text), input.language ? 1U : 0U,
            input.language ? detail::c_string(*input.language) : trtmc_string_view{}};
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &request, &options,
                                detail::audio_callback<Function>, &context, &raw, &error);
        detail::ResultOwner owner(state_, raw);
        if (context.exception) {
            state_->api.error_release(error);
            std::rethrow_exception(context.exception);
        }
        detail::check(state_->api, status, error);
        trtmc_streaming_audio_summary_v1 summary{};
        error = nullptr;
        status = api_->summary_view(raw, &summary, &error);
        detail::check(state_->api, status, error);
        return {summary.emitted_sample_count,
                summary.emitted_frame_count,
                {summary.sample_rate, summary.channels},
                static_cast<AudioDeliveryOutcome>(summary.outcome),
                summary.setup_ms,
                summary.inference_ms};
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_speech_table<trtmc_streaming_text_to_speech_api_v1>(table);
    }

  private:
    friend class Model;
    StreamingTextToSpeech(std::shared_ptr<detail::ModelState> state,
                          const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_streaming_text_to_speech_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_streaming_text_to_speech_api_v1* api_;
};

enum class SpeechEventKind : std::uint32_t {
    AgentAudio = 1,
    AgentText,
    UserTranscript,
    TurnStarted,
    TurnFinished,
    Yielded,
    Cancelled,
    Reset,
    Error,
    InputFinished,
    UserSpeechStarted,
    UserSpeechStopped,
    FunctionCall,
    FunctionCallStarted,
    FunctionResponseFinished,
    InputCleared
};
enum class SpeechReadState : std::uint32_t { Active = 1, EpochEnded = 2, Failed = 3 };
struct SpeechDialogueEventView {
    SpeechEventKind kind;
    std::uint64_t epoch, sequence;
    AudioView audio;
    std::int64_t media_start_sample, media_end_sample, frame_index;
    std::string_view text;
    bool is_final;
    std::optional<ToolCall> tool_call;
};
class SpeechEvents {
  public:
    SpeechEvents(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    SpeechEvents(const SpeechEvents&) = delete;
    SpeechEvents& operator=(const SpeechEvents&) = delete;
    SpeechEvents(SpeechEvents&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    SpeechEvents& operator=(SpeechEvents&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    std::size_t size() const noexcept { return static_cast<std::size_t>(view_.count); }
    SpeechReadState state() const noexcept { return static_cast<SpeechReadState>(view_.state); }
    // Views borrow this batch owner. ToolCall strings are copied for convenience.
    SpeechDialogueEventView at(std::size_t index) const {
        if (index >= size())
            throw std::out_of_range("speech event index");
        const auto& event = view_.events[index];
        SpeechDialogueEventView result{
            static_cast<SpeechEventKind>(event.kind),
            event.epoch,
            event.sequence,
            {{event.audio.samples, static_cast<std::size_t>(event.audio.sample_count)},
             event.audio.has_sample_rate ? std::optional<std::uint32_t>{event.audio.sample_rate}
                                         : std::nullopt,
             event.audio.channels},
            event.media_start_sample,
            event.media_end_sample,
            event.frame_index,
            detail::string_view(event.text),
            event.is_final != 0,
            {}};
        if (event.has_tool_call)
            result.tool_call = detail::copy_tool_call(event.tool_call);
        return result;
    }
    trtmc_speech_event_batch_view_v1& wire_view() noexcept { return view_; }

  private:
    detail::ResultOwner owner_;
    trtmc_speech_event_batch_view_v1 view_{};
};
enum class SpeechPollStatus { Events, Timeout, EpochEnd };
struct SpeechReadResult {
    SpeechPollStatus status;
    std::optional<SpeechEvents> events;
};

namespace detail {
struct SpeechSessionState {
    SpeechSessionState(std::shared_ptr<ModelState> model, const trtmc_speech_session_api_v1* api,
                       trtmc_speech_session* handle) noexcept
        : model(std::move(model)), api(api), handle(handle) {}
    ~SpeechSessionState() { close(); }
    void close() noexcept {
        if (handle)
            api->release(std::exchange(handle, nullptr));
    }
    std::shared_ptr<ModelState> model;
    const trtmc_speech_session_api_v1* api;
    trtmc_speech_session* handle;
};
} // namespace detail
class SpeechRealtimeControl {
  public:
    void commit_input_turn(bool create = true) {
        trtmc_error* e = nullptr;
        auto s = api_->commit_input_turn(state_->handle, create, &e);
        detail::check(state_->model->api, s, e);
    }
    void create_response() {
        trtmc_error* e = nullptr;
        auto s = api_->create_response(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    void clear_pending_input() {
        trtmc_error* e = nullptr;
        auto s = api_->clear_pending_input(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    void cancel_response() {
        trtmc_error* e = nullptr;
        auto s = api_->cancel_response(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    void truncate_response(std::uint64_t epoch, std::int64_t played_samples) {
        trtmc_error* e = nullptr;
        auto s = api_->truncate_response(state_->handle, epoch, played_samples, &e);
        detail::check(state_->model->api, s, e);
    }

  private:
    friend class SpeechSession;
    SpeechRealtimeControl(std::shared_ptr<detail::SpeechSessionState> state,
                          const trtmc_speech_realtime_api_v1* api) noexcept
        : state_(std::move(state)), api_(api) {}
    std::shared_ptr<detail::SpeechSessionState> state_;
    const trtmc_speech_realtime_api_v1* api_;
};
class SpeechToolControl {
  public:
    void submit_tool_result(std::uint64_t epoch, const ToolResult& value) {
        const auto result = detail::c_tool_result(value);
        trtmc_error* error = nullptr;
        const auto status = api_->submit_tool_result(state_->handle, epoch, &result, &error);
        detail::check(state_->model->api, status, error);
    }

  private:
    friend class SpeechSession;
    SpeechToolControl(std::shared_ptr<detail::SpeechSessionState> state,
                      const trtmc_speech_tools_api_v1* api) noexcept
        : state_(std::move(state)), api_(api) {}
    std::shared_ptr<detail::SpeechSessionState> state_;
    const trtmc_speech_tools_api_v1* api_;
};
class SpeechSession {
  public:
    SpeechSession(const SpeechSession&) = delete;
    SpeechSession& operator=(const SpeechSession&) = delete;
    SpeechSession(SpeechSession&&) noexcept = default;
    SpeechSession& operator=(SpeechSession&&) noexcept = default;
    [[nodiscard]] bool append_audio(Span<const float> input) {
        trtmc_error* error = nullptr;
        const auto status =
            state_->api->append_audio(state_->handle, input.data(), input.size(), &error);
        if (status == TRTMC_AGAIN) {
            state_->model->api.error_release(error);
            return false;
        }
        detail::check(state_->model->api, status, error);
        return true;
    }
    void finish_input() {
        trtmc_error* e = nullptr;
        auto s = state_->api->finish_input(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    void cancel() {
        trtmc_error* e = nullptr;
        auto s = state_->api->cancel(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    void reset() {
        trtmc_error* e = nullptr;
        auto s = state_->api->reset(state_->handle, &e);
        detail::check(state_->model->api, s, e);
    }
    SpeechReadResult read_events(std::int64_t timeout_ms = 0) {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = state_->api->read_events(state_->handle, timeout_ms, &raw, &error);
        SpeechEvents events(state_->model, raw);
        if (status == TRTMC_AGAIN || status == TRTMC_END) {
            state_->model->api.error_release(error);
            return {status == TRTMC_AGAIN ? SpeechPollStatus::Timeout : SpeechPollStatus::EpochEnd,
                    {}};
        }
        detail::check(state_->model->api, status, error);
        error = nullptr;
        status = state_->api->events_view(raw, &events.wire_view(), &error);
        detail::check(state_->model->api, status, error);
        return {SpeechPollStatus::Events, std::move(events)};
    }
    SpeechReadResult take_events() { return read_events(0); }
    SpeechReadResult wait_events(std::int64_t timeout_ms = -1) { return read_events(timeout_ms); }
    SpeechSessionInfo info() const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = state_->api->info(state_->handle, &raw, &error);
        detail::ResultOwner owner(state_->model, raw);
        detail::check(state_->model->api, status, error);
        trtmc_speech_session_info_v1 view{};
        error = nullptr;
        status = state_->api->info_view(raw, &view, &error);
        detail::check(state_->model->api, status, error);
        return detail::copy_speech_info(view);
    }
    Config config() const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = state_->api->config(state_->handle, &raw, &error);
        detail::ResultOwner owner(state_->model, raw);
        detail::check(state_->model->api, status, error);
        trtmc_config_view_v1 view{};
        error = nullptr;
        status = state_->api->config_view(raw, &view, &error);
        detail::check(state_->model->api, status, error);
        return detail::copy_config(view);
    }
    SpeechRealtimeControl realtime() {
        const trtmc_speech_realtime_api_v1* api = nullptr;
        trtmc_error* error = nullptr;
        const auto status = state_->api->get_realtime_api(state_->handle, 1, 0, &api, &error);
        detail::check(state_->model->api, status, error);
        detail::validate_speech_table<trtmc_speech_realtime_api_v1>(&api->header);
        return SpeechRealtimeControl(state_, api);
    }
    SpeechToolControl tools() {
        const trtmc_speech_tools_api_v1* api = nullptr;
        trtmc_error* error = nullptr;
        const auto status = state_->api->get_tools_api(state_->handle, 1, 0, &api, &error);
        detail::check(state_->model->api, status, error);
        detail::validate_speech_table<trtmc_speech_tools_api_v1>(&api->header);
        return SpeechToolControl(state_, api);
    }
    void close() noexcept {
        if (state_)
            state_->close();
    }
    // Used by provider wrappers; adoption keeps cleanup safe if allocation fails.
    SpeechSession(std::shared_ptr<detail::ModelState> model, const trtmc_speech_session_api_v1* api,
                  trtmc_speech_session* handle) {
        const auto release = [api](trtmc_speech_session* value) {
            if (value)
                api->release(value);
        };
        std::unique_ptr<trtmc_speech_session, decltype(release)> guard(handle, release);
        state_ = std::make_shared<detail::SpeechSessionState>(std::move(model), api, handle);
        guard.release();
    }

  private:
    std::shared_ptr<detail::SpeechSessionState> state_;
};

class DuplexSpeechDialogue {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_DUPLEX_SPEECH_DIALOGUE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechSession create(const SpeechDialogueRequest& input, const Config& config = {}) const {
        auto request = detail::c_dialogue(input);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_speech_session* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->create(state_->handle, &request, &options, &raw, &error);
        if (status != TRTMC_OK) {
            if (raw)
                api_->session_api->release(raw);
            detail::check(state_->api, status, error);
        }
        detail::check(state_->api, status, error);
        return SpeechSession(state_, api_->session_api, raw);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_speech_table<trtmc_duplex_speech_dialogue_api_v1>(table);
    }

  private:
    friend class Model;
    DuplexSpeechDialogue(std::shared_ptr<detail::ModelState> state,
                         const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_duplex_speech_dialogue_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_duplex_speech_dialogue_api_v1* api_;
};
class OfflineSpeechDialogue {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_OFFLINE_SPEECH_DIALOGUE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechSession create(const SpeechDialogueRequest& input, const Config& config = {}) const {
        auto request = detail::c_dialogue(input);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_speech_session* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->create(state_->handle, &request, &options, &raw, &error);
        if (status != TRTMC_OK) {
            if (raw)
                api_->session_api->release(raw);
            detail::check(state_->api, status, error);
        }
        detail::check(state_->api, status, error);
        return SpeechSession(state_, api_->session_api, raw);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_speech_table<trtmc_offline_speech_dialogue_api_v1>(table);
    }

  private:
    friend class Model;
    OfflineSpeechDialogue(std::shared_ptr<detail::ModelState> state,
                          const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_offline_speech_dialogue_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_offline_speech_dialogue_api_v1* api_;
};
class ToolSpeechDialogue {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TOOL_SPEECH_DIALOGUE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    SpeechSession create(const ToolSpeechDialogueRequest& input, const Config& config = {}) const {
        std::vector<trtmc_tool_definition_v1> tools;
        for (const auto& tool : input.tools)
            tools.push_back(detail::c_tool_definition(tool));
        std::vector<std::vector<trtmc_string_view>> lists(input.acknowledgements.size());
        std::vector<trtmc_tool_acknowledgement_v1> acks;
        for (std::size_t i = 0; i < input.acknowledgements.size(); ++i) {
            for (const auto& message : input.acknowledgements[i].messages)
                lists[i].push_back(detail::c_string(message));
            acks.push_back({detail::c_string(input.acknowledgements[i].tool_name),
                            {lists[i].data(), lists[i].size()}});
        }
        std::vector<trtmc_string_view> defaults;
        if (input.default_acknowledgements)
            for (const auto& message : *input.default_acknowledgements)
                defaults.push_back(detail::c_string(message));
        const trtmc_tool_speech_dialogue_request_v1 request{detail::c_dialogue(input.dialogue),
                                                            tools.data(),
                                                            tools.size(),
                                                            acks.data(),
                                                            acks.size(),
                                                            input.default_acknowledgements ? 1U
                                                                                           : 0U,
                                                            {defaults.data(), defaults.size()}};
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_speech_session* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->create(state_->handle, &request, &options, &raw, &error);
        if (status != TRTMC_OK) {
            if (raw)
                api_->session_api->release(raw);
            detail::check(state_->api, status, error);
        }
        detail::check(state_->api, status, error);
        return SpeechSession(state_, api_->session_api, raw);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_speech_table<trtmc_tool_speech_dialogue_api_v1>(table);
    }

  private:
    friend class Model;
    ToolSpeechDialogue(std::shared_ptr<detail::ModelState> state,
                       const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_tool_speech_dialogue_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_tool_speech_dialogue_api_v1* api_;
};

} // namespace trtmc

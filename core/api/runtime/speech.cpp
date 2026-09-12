/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/speech.h"

#include "api_internal.h"
#include "trtmc/speech.h"

#include <limits>

namespace trtmc::api {
namespace {

void valid_output(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
std::optional<std::string_view> optional_text(std::uint32_t present, trtmc_string_view text,
                                              bool empty_allowed = false) {
    require(present <= 1, "presence must be zero or one");
    if (!present)
        return std::nullopt;
    auto value = string_view(text);
    require(empty_allowed || !value.empty(), "present language must not be empty");
    return value;
}
internal::SpeechInputFormat input_format(const trtmc_speech_input_format_v1& input) {
    require(input.channels > 0, "input channels must be positive");
    require(input.has_sample_rate <= 1, "sample-rate presence must be zero or one");
    std::optional<std::uint32_t> rate;
    if (input.has_sample_rate) {
        require(input.sample_rate > 0, "present input rate must be positive");
        rate = input.sample_rate;
    }
    return {rate, input.channels};
}
void validate_info(const internal::SpeechSessionInfo& info) {
    valid_output(info.input.sample_rate && info.input.channels,
                 "family did not resolve input format");
    if (info.output)
        valid_output(info.output->sample_rate && info.output->channels,
                     "family returned invalid output format");
}
struct InfoStorage final : ResultStorage {
    explicit InfoStorage(internal::SpeechSessionInfo value) : value(std::move(value)) {
        validate_info(this->value);
    }
    internal::SpeechSessionInfo value;
};
trtmc_status TRTMC_CALL info_view(const trtmc_result* result, trtmc_speech_session_info_v1* out,
                                  trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "info output is null");
        const auto& info = require_result<InfoStorage>(result).value;
        *out = {info.input.sample_rate,
                info.input.channels,
                info.output ? 1U : 0U,
                info.output ? info.output->sample_rate : 0U,
                info.output ? info.output->channels : 0U,
                info.source_language ? 1U : 0U,
                info.source_language ? borrowed_string(*info.source_language) : trtmc_string_view{},
                info.system_prompt ? 1U : 0U,
                info.system_prompt ? borrowed_string(*info.system_prompt) : trtmc_string_view{}};
    });
}
struct AsrRun {
    std::unique_ptr<ModelSession> model;
    std::unique_ptr<internal::ISpeechTranscriptionStream> stream;
    internal::SpeechSessionInfo info;
};
struct DialogueRun {
    ~DialogueRun() {
        if (session)
            session->stop();
    }
    DialogueRun(std::unique_ptr<ModelSession> owner,
                std::unique_ptr<internal::ISpeechDialogueSession> implementation,
                internal::SpeechSessionInfo configuration)
        : model(std::move(owner)), session(std::move(implementation)),
          info(std::move(configuration)) {}
    std::unique_ptr<ModelSession> model;
    std::unique_ptr<internal::ISpeechDialogueSession> session;
    internal::SpeechSessionInfo info;
};
struct UpdateStorage final : ResultStorage {
    explicit UpdateStorage(internal::SpeechTranscriptUpdate update)
        : text(std::move(update.transcript)), is_final(update.is_final),
          chunk_index(update.chunk_index), accepted(update.accepted_samples), input(update.input) {
        valid_output(input.sample_rate && input.channels, "ASR update has invalid input format");
    }
    TextResultStorage text;
    bool is_final;
    std::uint64_t chunk_index, accepted;
    internal::SpeechAudioFormat input;
};
struct SummaryStorage final : ResultStorage {
    explicit SummaryStorage(internal::StreamingAudioSummary value) : value(std::move(value)) {}
    internal::StreamingAudioSummary value;
};

struct EventsStorage final : ResultStorage {
    explicit EventsStorage(internal::SpeechEventBatch batch) : batch(std::move(batch)) {
        const auto state = static_cast<std::uint32_t>(this->batch.state);
        valid_output(state >= TRTMC_SPEECH_ACTIVE && state <= TRTMC_SPEECH_FAILED,
                     "unknown speech read state");
        views.reserve(this->batch.events.size());
        for (const auto& event : this->batch.events) {
            const auto kind = static_cast<std::uint32_t>(event.kind);
            valid_output(kind >= TRTMC_SPEECH_AGENT_AUDIO && kind <= TRTMC_SPEECH_INPUT_CLEARED,
                         "unknown speech event kind");
            const auto& audio = event.audio;
            if (event.kind == internal::SpeechEventKind::AgentAudio)
                valid_output(audio.sample_rate && audio.channels &&
                                 audio.samples.size() % audio.channels == 0,
                             "speech event has invalid PCM format");
            valid_output(event.media_start_sample >= -1 && event.media_end_sample >= -1 &&
                             event.frame_index >= -1,
                         "speech media positions are invalid");
            if (event.media_start_sample >= 0 && event.media_end_sample >= 0)
                valid_output(event.media_end_sample >= event.media_start_sample,
                             "speech media bounds are reversed");
            valid_output((event.kind == internal::SpeechEventKind::FunctionCall) ==
                             event.tool_call.has_value(),
                         "function-call event must contain exactly its typed call");
            trtmc_tool_call_v1 call{};
            if (event.tool_call) {
                valid_output(!event.tool_call->call_id.empty() && !event.tool_call->name.empty(),
                             "empty tool call identity");
                const auto call_state = static_cast<std::uint32_t>(event.tool_call->state);
                valid_output(call_state <= TRTMC_TOOL_CALL_MALFORMED, "unknown tool-call state");
                call = {borrowed_string(event.tool_call->call_id),
                        borrowed_string(event.tool_call->name),
                        borrowed_string(event.tool_call->arguments_json), call_state};
            }
            views.push_back({kind,
                             event.epoch,
                             event.sequence,
                             {audio.samples.data(), audio.samples.size(),
                              audio.sample_rate ? 1U : 0U, audio.sample_rate, audio.channels},
                             event.media_start_sample,
                             event.media_end_sample,
                             event.frame_index,
                             borrowed_string(event.text),
                             event.is_final ? 1U : 0U,
                             event.tool_call ? 1U : 0U,
                             call});
        }
    }
    internal::SpeechEventBatch batch;
    std::vector<trtmc_speech_event_view_v1> views;
};

bool fatal(trtmc_status status) noexcept {
    return status == TRTMC_INTERNAL_ERROR || status == TRTMC_OUT_OF_MEMORY;
}

} // namespace
} // namespace trtmc::api

struct trtmc_asr_stream {
    std::mutex mutex;
    std::shared_ptr<trtmc::api::AsrRun> run;
};
struct trtmc_speech_session {
    std::mutex state_mutex, reader_mutex;
    std::shared_ptr<trtmc::api::DialogueRun> run;
};

namespace trtmc::api {
namespace {

template <class Function>
trtmc_status asr_call(trtmc_asr_stream* stream, trtmc_error** error, bool consuming,
                      Function function) noexcept {
    std::shared_ptr<AsrRun> run;
    std::unique_lock<std::mutex> lock;
    const auto status = guarded(error, [&] {
        require(stream, "ASR stream is null");
        lock = std::unique_lock<std::mutex>(stream->mutex);
        run = stream->run;
        require(run != nullptr, "ASR stream is closed");
        function(*run);
    });
    if (run && consuming && fatal(status))
        stream->run.reset();
    run.reset(); // Error copied first; model stays owned until family cleanup.
    return status;
}
trtmc_status TRTMC_CALL asr_create(trtmc_model* model,
                                   const trtmc_streaming_speech_transcription_request_v1* request,
                                   const trtmc_config_view_v1* config, trtmc_asr_stream** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    std::unique_ptr<ModelSession> owner;
    std::unique_ptr<internal::ISpeechTranscriptionStream> implementation;
    std::shared_ptr<AsrRun> run;
    return guarded(error, [&] {
        require(request && out, "ASR request or output is null");
        const internal::StreamingSpeechTranscriptionRequest input{
            input_format(request->input),
            optional_text(request->has_source_language, request->source_language)};
        const ConvertedConfig options(config);
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IStreamingSpeechTranscription>(
                model, internal::IStreamingSpeechTranscription::kTask);
            validate_task_config(model_owner(model),
                                 internal::contract_key<internal::IStreamingSpeechTranscription>(),
                                 options.view());
            owner = std::make_unique<ModelSession>(model);
            implementation = family.create(input, options.view());
        }
        valid_output(implementation != nullptr, "family returned a null ASR stream");
        auto info = implementation->info();
        validate_info(info);
        run = std::make_shared<AsrRun>(
            AsrRun{std::move(owner), std::move(implementation), std::move(info)});
        auto handle = std::make_unique<trtmc_asr_stream>();
        handle->run = run;
        *out = handle.release();
    });
}
trtmc_status TRTMC_CALL asr_accept(trtmc_asr_stream* stream, const float* samples,
                                   std::uint64_t count, std::uint32_t final, trtmc_result** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return asr_call(stream, error, true, [&](AsrRun& run) {
        require(out && final <= 1, "ASR result or final flag is invalid");
        const auto audio = checked_span(samples, count);
        require(audio.size() % run.info.input.channels == 0, "ASR input has an incomplete frame");
        *out = make_result<UpdateStorage>(run.stream->accept_audio(audio, final != 0));
    });
}
trtmc_status TRTMC_CALL asr_finish(trtmc_asr_stream* stream, trtmc_result** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return asr_call(stream, error, true, [&](AsrRun& run) {
        require(out, "ASR result is null");
        *out = make_result<UpdateStorage>(run.stream->finish());
    });
}
trtmc_status TRTMC_CALL asr_reset(trtmc_asr_stream* stream, trtmc_error** error) noexcept {
    return asr_call(stream, error, true, [](AsrRun& run) { run.stream->reset(); });
}
trtmc_status TRTMC_CALL asr_info(trtmc_asr_stream* stream, trtmc_result** out,
                                 trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return asr_call(stream, error, false, [&](AsrRun& run) {
        require(out, "ASR info output is null");
        *out = make_result<InfoStorage>(run.stream->info());
    });
}
trtmc_status TRTMC_CALL asr_config(trtmc_asr_stream* stream, trtmc_result** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return asr_call(stream, error, false, [&](AsrRun& run) {
        require(out, "ASR config output is null");
        *out = make_config_snapshot(run.stream->effective_config());
    });
}
trtmc_status TRTMC_CALL update_view(const trtmc_result* result,
                                    trtmc_speech_transcript_update_v1* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "ASR update view is null");
        const auto& value = require_result<UpdateStorage>(result);
        fill_text_result_view(value.text, &out->transcript);
        out->is_final = value.is_final;
        out->chunk_index = value.chunk_index;
        out->accepted_samples = value.accepted;
        out->sample_rate = value.input.sample_rate;
        out->channels = value.input.channels;
    });
}
void TRTMC_CALL asr_release(trtmc_asr_stream* stream) noexcept {
    delete stream;
}

struct ChunkBridge {
    trtmc_audio_chunk_callback_v1 callback;
    void* context;
    std::uint64_t samples{0};
    std::uint32_t sample_rate{0}, channels{0};
    bool stopped{false};
    std::optional<OwnedApiFailure> failure;
    bool emit(const internal::AudioView& chunk) {
        valid_output(!stopped && !failure, "family delivered audio after callback stopped");
        valid_output(chunk.sample_rate && *chunk.sample_rate > 0 && chunk.channels > 0 &&
                         chunk.samples.size() % chunk.channels == 0,
                     "family callback PCM format is invalid");
        if (sample_rate)
            valid_output(sample_rate == *chunk.sample_rate && channels == chunk.channels,
                         "callback audio format changed");
        sample_rate = *chunk.sample_rate;
        channels = chunk.channels;
        valid_output(chunk.samples.size() <= std::numeric_limits<std::uint64_t>::max() - samples,
                     "callback sample count overflow");
        const trtmc_audio_view_v1 input{chunk.samples.data(), chunk.samples.size(), 1, sample_rate,
                                        channels};
        trtmc_audio_chunk_reply_v1 reply{};
        const auto status = callback(context, &input, &reply);
        samples += chunk.samples.size();
        if (status == TRTMC_OK)
            return true;
        if (status == TRTMC_END) {
            stopped = true;
            return false;
        }
        switch (status) {
        case TRTMC_INVALID_ARGUMENT:
        case TRTMC_INVALID_CONFIG:
        case TRTMC_UNSUPPORTED:
        case TRTMC_VERSION_MISMATCH:
        case TRTMC_OUT_OF_MEMORY:
        case TRTMC_INTERNAL_ERROR:
        case TRTMC_BUSY:
            break;
        default:
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "audio callback returned an invalid status"};
        }
        const auto message = string_view(reply.error_message);
        failure = OwnedApiFailure{status,
                                  message.empty() ? "audio callback failed" : std::string(message)};
        throw *failure;
    }
};
trtmc_status TRTMC_CALL tts_run(trtmc_model* model, const trtmc_text_to_speech_request_v1* request,
                                const trtmc_config_view_v1* config,
                                trtmc_audio_chunk_callback_v1 callback, void* context,
                                trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    std::unique_ptr<ModelSession> execution;
    return guarded(error, [&] {
        require(request && out && callback, "TTS request, callback and output are required");
        internal::TextToSpeechRequest input{
            string_view(request->text), optional_text(request->has_language, request->language)};
        const ConvertedConfig options(config);
        internal::IStreamingTextToSpeech* family;
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            family = &require_interface<internal::IStreamingTextToSpeech>(
                model, internal::IStreamingTextToSpeech::kTask);
            validate_task_config(model_owner(model),
                                 internal::contract_key<internal::IStreamingTextToSpeech>(),
                                 options.view());
            execution = std::make_unique<ModelSession>(model);
        }
        ChunkBridge bridge{callback, context, 0, 0, 0, false, std::nullopt};
        auto summary = family->run(input, options.view(), [&](const internal::AudioView& chunk) {
            return bridge.emit(chunk);
        });
        if (bridge.failure)
            throw *bridge.failure;
        valid_output(summary.output.sample_rate && summary.output.channels,
                     "TTS summary has invalid format");
        valid_output(summary.emitted_sample_count == bridge.samples &&
                         summary.emitted_frame_count ==
                             summary.emitted_sample_count / summary.output.channels &&
                         summary.emitted_sample_count % summary.output.channels == 0,
                     "TTS summary count is inconsistent");
        if (bridge.sample_rate)
            valid_output(summary.output.sample_rate == bridge.sample_rate &&
                             summary.output.channels == bridge.channels,
                         "TTS summary format differs from chunks");
        const auto expected = bridge.stopped ? internal::AudioDeliveryOutcome::Stopped
                                             : internal::AudioDeliveryOutcome::Complete;
        valid_output(summary.outcome == expected, "family did not honor callback stop outcome");
        *out = make_result<SummaryStorage>(std::move(summary));
    });
}
trtmc_status TRTMC_CALL summary_view(const trtmc_result* result,
                                     trtmc_streaming_audio_summary_v1* out,
                                     trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "TTS summary view is null");
        const auto& value = require_result<SummaryStorage>(result).value;
        *out = {value.emitted_sample_count,
                value.emitted_frame_count,
                value.output.sample_rate,
                value.output.channels,
                static_cast<std::uint32_t>(value.outcome),
                value.setup_ms,
                value.inference_ms};
    });
}

std::shared_ptr<DialogueRun> current(trtmc_speech_session* session) {
    require(session, "speech session is null");
    const std::lock_guard<std::mutex> lock(session->state_mutex);
    return session->run;
}
void detach(trtmc_speech_session* session, const std::shared_ptr<DialogueRun>& run) {
    const std::lock_guard<std::mutex> lock(session->state_mutex);
    if (session->run == run)
        session->run.reset();
}
template <class Function>
trtmc_status session_call(trtmc_speech_session* session, trtmc_error** error, bool consuming,
                          Function function) noexcept {
    std::shared_ptr<DialogueRun> run;
    const auto status = guarded(error, [&] {
        run = current(session);
        require(run != nullptr, "speech session is closed");
        function(*run);
    });
    if (run && consuming && fatal(status)) {
        run->session->stop();
        detach(session, run);
    }
    run.reset();
    return status;
}
internal::SpeechDialogueRequest dialogue_request(const trtmc_speech_dialogue_request_v1& value) {
    return {input_format(value.input),
            optional_text(value.has_system_prompt, value.system_prompt, true)};
}

template <class Provider, auto Create>
trtmc_status create_dialogue(trtmc_model* model, const trtmc_speech_dialogue_request_v1* request,
                             const trtmc_config_view_v1* config, trtmc_speech_session** out,
                             trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    std::unique_ptr<ModelSession> owner;
    std::unique_ptr<internal::ISpeechDialogueSession> implementation;
    std::shared_ptr<DialogueRun> run;
    return guarded(error, [&] {
        require(request && out, "dialogue request or output is null");
        const auto input = dialogue_request(*request);
        const ConvertedConfig options(config);
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<Provider>(model, Provider::kTask);
            validate_task_config(model_owner(model), internal::contract_key<Provider>(),
                                 options.view());
            owner = std::make_unique<ModelSession>(model);
            implementation = (family.*Create)(input, options.view());
        }
        valid_output(implementation != nullptr, "family returned null speech session");
        auto info = implementation->info();
        validate_info(info);
        run = std::make_shared<DialogueRun>(std::move(owner), std::move(implementation),
                                            std::move(info));
        auto handle = std::make_unique<trtmc_speech_session>();
        handle->run = run;
        *out = handle.release();
    });
}

trtmc_status TRTMC_CALL tool_create(trtmc_model* model,
                                    const trtmc_tool_speech_dialogue_request_v1* request,
                                    const trtmc_config_view_v1* config, trtmc_speech_session** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    std::unique_ptr<ModelSession> owner;
    std::unique_ptr<internal::ISpeechDialogueSession> implementation;
    std::shared_ptr<DialogueRun> run;
    return guarded(error, [&] {
        require(request && out, "tool dialogue request or output is null");
        const auto tool_inputs = checked_span(request->tools, request->tool_count);
        require(!tool_inputs.empty(), "tool dialogue requires nonempty tool definitions");
        std::vector<internal::ToolDefinitionView> tools;
        for (const auto& tool : tool_inputs)
            tools.push_back({string_view(tool.name), string_view(tool.description),
                             string_view(tool.parameters_schema_json)});
        const auto ack_inputs =
            checked_span(request->acknowledgements, request->acknowledgement_count);
        std::vector<std::vector<std::string_view>> messages(ack_inputs.size());
        std::vector<internal::ToolAcknowledgementView> acks;
        for (std::size_t i = 0; i < ack_inputs.size(); ++i) {
            for (const auto message :
                 checked_span(ack_inputs[i].messages.data, ack_inputs[i].messages.size))
                messages[i].push_back(string_view(message));
            acks.push_back(
                {string_view(ack_inputs[i].tool_name), {messages[i].data(), messages[i].size()}});
        }
        require(request->has_default_acknowledgements <= 1,
                "default acknowledgement presence is invalid");
        std::vector<std::string_view> defaults;
        std::optional<Span<const std::string_view>> default_view;
        if (request->has_default_acknowledgements) {
            for (const auto message : checked_span(request->default_acknowledgements.data,
                                                   request->default_acknowledgements.size))
                defaults.push_back(string_view(message));
            default_view = Span<const std::string_view>{defaults.data(), defaults.size()};
        }
        const internal::ToolSpeechDialogueRequest input{dialogue_request(request->dialogue),
                                                        {tools.data(), tools.size()},
                                                        {acks.data(), acks.size()},
                                                        default_view};
        const ConvertedConfig options(config);
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IToolSpeechDialogue>(
                model, internal::IToolSpeechDialogue::kTask);
            validate_task_config(model_owner(model),
                                 internal::contract_key<internal::IToolSpeechDialogue>(),
                                 options.view());
            owner = std::make_unique<ModelSession>(model);
            implementation = family.create_tool_session(input, options.view());
        }
        valid_output(implementation && implementation->tool_control(),
                     "tool session has no typed tool control");
        auto info = implementation->info();
        validate_info(info);
        run = std::make_shared<DialogueRun>(std::move(owner), std::move(implementation),
                                            std::move(info));
        auto handle = std::make_unique<trtmc_speech_session>();
        handle->run = run;
        *out = handle.release();
    });
}

trtmc_status TRTMC_CALL append_audio(trtmc_speech_session* session, const float* samples,
                                     std::uint64_t count, trtmc_error** error) noexcept {
    bool accepted = true;
    const auto status = session_call(session, error, true, [&](DialogueRun& run) {
        const auto input = checked_span(samples, count);
        require(input.size() % run.info.input.channels == 0,
                "dialogue input has an incomplete frame");
        accepted = run.session->append_audio(input);
    });
    return status == TRTMC_OK && !accepted ? TRTMC_AGAIN : status;
}
trtmc_status TRTMC_CALL finish_input(trtmc_speech_session* session, trtmc_error** error) noexcept {
    return session_call(session, error, true,
                        [](DialogueRun& run) { run.session->finish_input(); });
}
trtmc_status TRTMC_CALL cancel(trtmc_speech_session* session, trtmc_error** error) noexcept {
    return session_call(session, error, true, [](DialogueRun& run) { run.session->cancel(); });
}
trtmc_status TRTMC_CALL reset(trtmc_speech_session* session, trtmc_error** error) noexcept {
    return session_call(session, error, true, [](DialogueRun& run) { run.session->reset(); });
}
trtmc_status TRTMC_CALL session_info(trtmc_speech_session* session, trtmc_result** out,
                                     trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return session_call(session, error, false, [&](DialogueRun& run) {
        require(out, "session info output is null");
        *out = make_result<InfoStorage>(run.session->info());
    });
}
trtmc_status TRTMC_CALL session_config(trtmc_speech_session* session, trtmc_result** out,
                                       trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return session_call(session, error, false, [&](DialogueRun& run) {
        require(out, "session config output is null");
        *out = make_config_snapshot(run.session->effective_config());
    });
}
trtmc_status TRTMC_CALL read_events(trtmc_speech_session* session, std::int64_t timeout,
                                    trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    std::shared_ptr<DialogueRun> run;
    std::unique_lock<std::mutex> reader;
    bool entered = false, failed = false;
    trtmc_status disposition = TRTMC_OK;
    const auto status = guarded(error, [&] {
        require(session && out && timeout >= -1, "session, output or read timeout is invalid");
        reader = std::unique_lock<std::mutex>(session->reader_mutex, std::try_to_lock);
        if (!reader.owns_lock())
            throw ApiFailure{TRTMC_BUSY, "another speech event read is active"};
        run = current(session);
        if (!run) {
            disposition = TRTMC_END;
            return;
        }
        entered = true;
        auto batch = run->session->read_events(timeout);
        failed = batch.state == internal::SpeechReadState::Failed;
        if (batch.events.empty() && batch.state == internal::SpeechReadState::Active) {
            valid_output(timeout != -1, "blocking speech read returned no events");
            disposition = TRTMC_AGAIN;
            return;
        }
        if (batch.events.empty() && batch.state == internal::SpeechReadState::EpochEnded) {
            disposition = TRTMC_END;
            return;
        }
        *out = make_result<EventsStorage>(std::move(batch));
    });
    if (run && (failed || (entered && status != TRTMC_OK))) {
        run->session->stop();
        detach(session, run);
    }
    run.reset();
    return status == TRTMC_OK ? disposition : status;
}
trtmc_status TRTMC_CALL events_view(const trtmc_result* result,
                                    trtmc_speech_event_batch_view_v1* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "speech event view output is null");
        const auto& storage = require_result<EventsStorage>(result);
        *out = {storage.views.data(), storage.views.size(),
                static_cast<std::uint32_t>(storage.batch.state)};
    });
}

internal::ISpeechRealtimeControl& realtime(DialogueRun& run) {
    auto* control = run.session->realtime_control();
    if (!control)
        throw ApiFailure{TRTMC_UNSUPPORTED, "session has no realtime control"};
    return *control;
}
internal::ISpeechToolControl& tools(DialogueRun& run) {
    auto* control = run.session->tool_control();
    if (!control)
        throw ApiFailure{TRTMC_UNSUPPORTED, "session has no tool control"};
    return *control;
}
trtmc_status TRTMC_CALL commit(trtmc_speech_session* session, std::uint32_t create,
                               trtmc_error** error) noexcept {
    return session_call(session, error, true, [&](DialogueRun& run) {
        require(create <= 1, "create-response flag is invalid");
        realtime(run).commit_input_turn(create != 0);
    });
}
trtmc_status TRTMC_CALL create_response(trtmc_speech_session* session,
                                        trtmc_error** error) noexcept {
    return session_call(session, error, true,
                        [](DialogueRun& run) { realtime(run).create_response(); });
}
trtmc_status TRTMC_CALL clear_input(trtmc_speech_session* session, trtmc_error** error) noexcept {
    return session_call(session, error, true,
                        [](DialogueRun& run) { realtime(run).clear_pending_input(); });
}
trtmc_status TRTMC_CALL cancel_response(trtmc_speech_session* session,
                                        trtmc_error** error) noexcept {
    return session_call(session, error, true,
                        [](DialogueRun& run) { realtime(run).cancel_response(); });
}
trtmc_status TRTMC_CALL truncate(trtmc_speech_session* session, std::uint64_t epoch,
                                 std::int64_t samples, trtmc_error** error) noexcept {
    return session_call(session, error, true, [&](DialogueRun& run) {
        require(samples >= 0, "playback position is negative");
        realtime(run).truncate_response(epoch, samples);
    });
}
trtmc_status TRTMC_CALL submit_tool(trtmc_speech_session* session, std::uint64_t epoch,
                                    const trtmc_tool_result_v1* result,
                                    trtmc_error** error) noexcept {
    return session_call(session, error, true, [&](DialogueRun& run) {
        require(result && result->is_error <= 1, "tool result or error flag is invalid");
        tools(run).submit_tool_result(epoch,
                                      {string_view(result->call_id),
                                       string_view(result->content_text), result->is_error != 0});
    });
}
const trtmc_speech_realtime_api_v1 realtime_api{
    {1, 0, sizeof(realtime_api)}, commit, create_response, clear_input, cancel_response, truncate};
const trtmc_speech_tools_api_v1 tools_api{{1, 0, sizeof(tools_api)}, submit_tool};
trtmc_status TRTMC_CALL get_realtime(trtmc_speech_session* session, std::uint32_t major,
                                     std::uint32_t minor, const trtmc_speech_realtime_api_v1** out,
                                     trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return session_call(session, error, false, [&](DialogueRun& run) {
        require(out, "realtime table output is null");
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "unsupported realtime version"};
        (void)realtime(run);
        *out = &realtime_api;
    });
}
trtmc_status TRTMC_CALL get_tools(trtmc_speech_session* session, std::uint32_t major,
                                  std::uint32_t minor, const trtmc_speech_tools_api_v1** out,
                                  trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return session_call(session, error, false, [&](DialogueRun& run) {
        require(out, "tools table output is null");
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "unsupported tools version"};
        (void)tools(run);
        *out = &tools_api;
    });
}
void TRTMC_CALL session_release(trtmc_speech_session* session) noexcept {
    delete session;
}

const trtmc_streaming_speech_transcription_api_v1 asr_api{{1, 0, sizeof(asr_api)},
                                                          asr_create,
                                                          asr_accept,
                                                          asr_finish,
                                                          update_view,
                                                          asr_reset,
                                                          asr_info,
                                                          info_view,
                                                          asr_config,
                                                          config_snapshot_view,
                                                          asr_release};
const trtmc_streaming_text_to_speech_api_v1 tts_api{{1, 0, sizeof(tts_api)}, tts_run, summary_view};
const trtmc_speech_session_api_v1 session_api{{1, 0, sizeof(session_api)},
                                              append_audio,
                                              finish_input,
                                              read_events,
                                              events_view,
                                              cancel,
                                              reset,
                                              session_info,
                                              info_view,
                                              session_config,
                                              config_snapshot_view,
                                              get_realtime,
                                              get_tools,
                                              session_release};
const trtmc_duplex_speech_dialogue_api_v1 duplex_api{
    {1, 0, sizeof(duplex_api)},
    create_dialogue<internal::IDuplexSpeechDialogue,
                    &internal::IDuplexSpeechDialogue::create_duplex_session>,
    &session_api};
const trtmc_offline_speech_dialogue_api_v1 offline_api{
    {1, 0, sizeof(offline_api)},
    create_dialogue<internal::IOfflineSpeechDialogue,
                    &internal::IOfflineSpeechDialogue::create_offline_session>,
    &session_api};
const trtmc_tool_speech_dialogue_api_v1 tool_api{
    {1, 0, sizeof(tool_api)}, tool_create, &session_api};
const TaskBinding bindings[] = {
    {internal::IStreamingSpeechTranscription::kTask, 1, 0, &asr_api.header},
    {internal::IStreamingTextToSpeech::kTask, 1, 0, &tts_api.header},
    {internal::IDuplexSpeechDialogue::kTask, 1, 0, &duplex_api.header},
    {internal::IOfflineSpeechDialogue::kTask, 1, 0, &offline_api.header},
    {internal::IToolSpeechDialogue::kTask, 1, 0, &tool_api.header},
};

} // namespace
Span<const TaskBinding> speech_task_bindings() noexcept {
    return bindings;
}
} // namespace trtmc::api

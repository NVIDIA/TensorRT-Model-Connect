/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/speech.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>

namespace {
using namespace trtmc::internal;
using trtmc::Span;
std::atomic<int> waiting_readers{0};
std::atomic<int> live_sessions{0};
std::atomic<int> example_streaming_loads{0};

std::vector<ConfigField> fields() {
    static const std::int64_t integers[]{0, 9007199254740993LL};
    static const double numbers[]{0.5, 0};
    static const std::string_view strings[]{"first", "second"};
    return {
        {"i64", ConfigKind::I64, ConfigValue{std::int64_t{0}}, "Protocol integer"},
        {"f64", ConfigKind::F64, ConfigValue{0.0}, "Protocol real"},
        {"bool", ConfigKind::Bool, ConfigValue{false}, "Protocol flag"},
        {"text", ConfigKind::String, ConfigValue{std::string_view{}}, "Protocol text"},
        {"ints", ConfigKind::I64List, ConfigValue{Span<const std::int64_t>{integers}},
         "Protocol integers"},
        {"floats", ConfigKind::F64List, ConfigValue{Span<const double>{numbers}}, "Protocol reals"},
        {"strings", ConfigKind::StringList, ConfigValue{Span<const std::string_view>{strings}},
         "Protocol strings"}};
}
struct ConfigState {
    std::int64_t integer{0};
    double real{0};
    bool flag{false};
    std::string text;
    std::vector<std::int64_t> integers;
    std::vector<double> numbers;
    std::vector<std::string> strings;
    std::vector<std::string_view> string_views;
    std::vector<ConfigEntry> entries;
    explicit ConfigState(ConfigView supplied) {
        const auto rules = fields();
        for (const auto& field : rules)
            assign(field.name, *field.default_value);
        std::set<std::string_view> seen;
        for (const auto& entry : supplied) {
            const auto field = std::find_if(rules.begin(), rules.end(), [&](const ConfigField& f) {
                return f.name == entry.name;
            });
            if (field == rules.end() || !seen.insert(entry.name).second ||
                field->kind != config_kind(entry.value))
                throw ConfigError("invalid speech fixture option");
            assign(entry.name, entry.value);
        }
        for (const auto& value : strings)
            string_views.push_back(value);
        entries = {
            {"i64", integer},
            {"f64", real},
            {"bool", flag},
            {"text", std::string_view{text}},
            {"ints", Span<const std::int64_t>{integers.data(), integers.size()}},
            {"floats", Span<const double>{numbers.data(), numbers.size()}},
            {"strings", Span<const std::string_view>{string_views.data(), string_views.size()}}};
    }
    void assign(std::string_view key, const ConfigValue& value) {
        if (key == "i64")
            integer = std::get<std::int64_t>(value);
        else if (key == "f64")
            real = std::get<double>(value);
        else if (key == "bool")
            flag = std::get<bool>(value);
        else if (key == "text")
            text = std::get<std::string_view>(value);
        else if (key == "ints") {
            const auto v = std::get<Span<const std::int64_t>>(value);
            integers.clear();
            for (auto x : v)
                integers.push_back(x);
        } else if (key == "floats") {
            const auto v = std::get<Span<const double>>(value);
            numbers.clear();
            for (auto x : v)
                numbers.push_back(x);
        } else if (key == "strings") {
            strings.clear();
            for (auto x : std::get<Span<const std::string_view>>(value))
                strings.emplace_back(x);
        }
    }
    ConfigView view() const { return {entries.data(), entries.size()}; }
};

SpeechSessionInfo info_for(const SpeechInputFormat& input, bool output) {
    SpeechSessionInfo info;
    info.input = {input.sample_rate.value_or(16000), input.channels};
    if (output)
        info.output = SpeechAudioFormat{24000, input.channels};
    return info;
}
class Asr final : public ISpeechTranscriptionStream {
  public:
    Asr(const StreamingSpeechTranscriptionRequest& input, ConfigView config)
        : info_(info_for(input.input, false)), config_(std::make_unique<ConfigState>(config)) {
        if (input.source_language)
            info_.source_language = std::string(*input.source_language);
        ++live_sessions;
    }
    ~Asr() override { --live_sessions; }
    SpeechTranscriptUpdate accept_audio(Span<const float> audio, bool final) override {
        if (config_->text == "benchmark-fail")
            throw std::runtime_error("benchmark ASR provider failed");
        if (finished_)
            throw std::invalid_argument("ASR epoch already finished; reset first");
        accepted_ += audio.size() / info_.input.channels;
        ++chunks_;
        if (final)
            return finish();
        return update(false);
    }
    SpeechTranscriptUpdate finish() override {
        if (config_->text == "benchmark-unfinished")
            return update(false);
        if (!finished_) {
            finished_ = true;
            final_ = update(true);
        }
        return final_;
    }
    void reset() override {
        accepted_ = 0;
        chunks_ = 0;
        finished_ = false;
        final_ = {};
    }
    SpeechSessionInfo info() const override { return info_; }
    ConfigView effective_config() const override { return config_->view(); }

  private:
    SpeechTranscriptUpdate update(bool final) const {
        TextResult text;
        text.text = config_->text + "frames:" + std::to_string(accepted_);
        text.token_ids = {static_cast<std::int32_t>(accepted_)};
        text.segments.push_back({0, static_cast<double>(accepted_) / info_.input.sample_rate,
                                 text.text, text.token_ids});
        return {std::move(text), final, chunks_, accepted_, info_.input};
    }
    SpeechSessionInfo info_;
    std::unique_ptr<ConfigState> config_;
    std::uint64_t accepted_{0}, chunks_{0};
    bool finished_{false};
    SpeechTranscriptUpdate final_;
};

class Dialogue final : public ISpeechDialogueSession,
                       public ISpeechRealtimeControl,
                       public ISpeechToolControl {
  public:
    Dialogue(const SpeechDialogueRequest& input, ConfigView config, bool offline,
             std::string behavior, const ToolSpeechDialogueRequest* tools = nullptr)
        : info_(info_for(input.input, true)), config_(std::make_unique<ConfigState>(config)),
          offline_(offline), behavior_(std::move(behavior)), tools_(tools != nullptr) {
        info_.system_prompt =
            input.system_prompt ? std::string(*input.system_prompt) : std::string("fixture prompt");
        if (tools) {
            std::set<std::string_view> names;
            for (const auto& tool : tools->tools) {
                if (tool.name.empty() || !names.insert(tool.name).second)
                    throw std::invalid_argument("invalid tool identity");
                tool_defs_.push_back({std::string(tool.name), std::string(tool.description),
                                      std::string(tool.parameters_schema_json)});
            }
            for (const auto& ack : tools->acknowledgements) {
                if (ack.messages.empty() || names.count(ack.tool_name) == 0)
                    throw std::invalid_argument("invalid tool acknowledgement");
                std::vector<std::string> messages;
                for (auto message : ack.messages)
                    messages.emplace_back(message);
                acks_.push_back({std::string(ack.tool_name), std::move(messages)});
            }
            if (tools->default_acknowledgements) {
                if (tools->default_acknowledgements->empty())
                    throw std::invalid_argument("empty default acknowledgement list");
                for (auto message : *tools->default_acknowledgements)
                    defaults_.emplace_back(message);
            }
        }
        ++live_sessions;
    }
    ~Dialogue() override {
        stop();
        --live_sessions;
    }
    void stop() noexcept override {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }
    bool append_audio(Span<const float> input) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_active();
        if (behavior_ == "fatal_append")
            throw std::logic_error("unexpected fixture invariant failure");
        if (input.size() > 16 - audio_.size())
            return false;
        if (audio_.empty() && !input.empty())
            emit(SpeechEventKind::UserSpeechStarted, {}, false);
        for (float value : input)
            audio_.push_back(value);
        ++frame_;
        emit(SpeechEventKind::UserTranscript, "input:" + std::to_string(audio_.size()), false);
        cv_.notify_all();
        return true;
    }
    void finish_input() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SpeechReadState::EpochEnded)
            return;
        require_active();
        emit(SpeechEventKind::UserSpeechStopped, {}, true);
        emit(SpeechEventKind::UserTranscript, "input:" + std::to_string(audio_.size()), true);
        if (!responding_)
            response();
        emit(SpeechEventKind::TurnFinished, {}, true);
        responding_ = false;
        emit(SpeechEventKind::InputFinished, {}, true);
        state_ = SpeechReadState::EpochEnded;
        cv_.notify_all();
    }
    SpeechEventBatch read_events(std::int64_t timeout) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (behavior_ == "throw_read")
            throw std::logic_error("unexpected read invariant");
        if (behavior_ == "bad_event") {
            SpeechDialogueEvent event;
            event.kind = static_cast<SpeechEventKind>(999);
            return {{std::move(event)}, SpeechReadState::Active};
        }
        const auto ready = [&] {
            return stopped_ || !events_.empty() || state_ != SpeechReadState::Active;
        };
        if (!ready() && timeout != 0) {
            ++waiting_readers;
            if (timeout < 0)
                cv_.wait(lock, ready);
            else
                (void)cv_.wait_for(lock, std::chrono::milliseconds(timeout), ready);
            --waiting_readers;
        }
        if (stopped_)
            return {{}, SpeechReadState::Failed};
        SpeechEventBatch result{std::move(events_), state_};
        events_.clear();
        return result;
    }
    void cancel() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
            throw std::invalid_argument("session stopped");
        ++epoch_;
        emit(SpeechEventKind::Cancelled, {}, true);
        state_ = SpeechReadState::EpochEnded;
        pending_.clear();
        cv_.notify_all();
    }
    void reset() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
            throw std::invalid_argument("session stopped");
        ++epoch_;
        state_ = SpeechReadState::Active;
        events_.clear();
        audio_.clear();
        pending_.clear();
        responding_ = committed_ = false;
        frame_ = 0;
        emit(SpeechEventKind::Reset, {}, true);
        cv_.notify_all();
    }
    SpeechSessionInfo info() const override { return info_; }
    ConfigView effective_config() const override { return config_->view(); }
    ISpeechRealtimeControl* realtime_control() noexcept override {
        return offline_ ? nullptr : this;
    }
    ISpeechToolControl* tool_control() noexcept override { return tools_ ? this : nullptr; }
    void commit_input_turn(bool create) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_live();
        committed_ = true;
        if (create)
            response();
        cv_.notify_all();
    }
    void create_response() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_live();
        if (!committed_)
            throw std::invalid_argument("commit input first");
        response();
        cv_.notify_all();
    }
    void clear_pending_input() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_live();
        if (responding_)
            throw std::invalid_argument("cannot clear during response");
        audio_.clear();
        committed_ = false;
        emit(SpeechEventKind::InputCleared, {}, true);
        cv_.notify_all();
    }
    void cancel_response() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_live();
        if (!responding_)
            throw std::invalid_argument("no response to cancel");
        responding_ = false;
        emit(SpeechEventKind::Yielded, {}, true);
        emit(SpeechEventKind::TurnFinished, {}, true);
        cv_.notify_all();
    }
    void truncate_response(std::uint64_t epoch, std::int64_t samples) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_live();
        if (!responding_ || epoch != epoch_ || samples < 0 ||
            static_cast<std::uint64_t>(samples) > audio_.size() / info_.input.channels)
            throw std::invalid_argument("stale response epoch or invalid playback cursor");
        responding_ = false;
        emit(SpeechEventKind::Yielded, "truncated:" + std::to_string(samples), true);
        events_.back().media_end_sample = samples;
        emit(SpeechEventKind::TurnFinished, {}, true);
        cv_.notify_all();
    }
    void submit_tool_result(std::uint64_t epoch, const ToolResultView& result) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        require_active();
        if (!tools_ || epoch != epoch_)
            throw std::invalid_argument("stale tool epoch");
        const auto call = std::find(pending_.begin(), pending_.end(), result.call_id);
        if (call == pending_.end())
            throw std::invalid_argument("unknown or duplicate tool result");
        pending_.erase(call);
        emit(result.is_error ? SpeechEventKind::Error : SpeechEventKind::AgentText,
             std::string(result.content_text), true);
        if (pending_.empty())
            emit(SpeechEventKind::FunctionResponseFinished, {}, true);
        cv_.notify_all();
    }

  private:
    void require_active() const {
        if (stopped_ || state_ != SpeechReadState::Active)
            throw std::invalid_argument("epoch ended; reset first");
    }
    void require_live() const {
        require_active();
        if (offline_)
            throw std::invalid_argument("offline session has no realtime control");
    }
    void emit(SpeechEventKind kind, std::string text, bool final) {
        SpeechDialogueEvent event;
        event.kind = kind;
        event.epoch = epoch_;
        event.sequence = sequence_++;
        event.frame_index = frame_;
        event.text = std::move(text);
        event.is_final = final;
        if (kind == SpeechEventKind::UserSpeechStarted ||
            kind == SpeechEventKind::UserSpeechStopped) {
            event.audio.sample_rate = info_.input.sample_rate;
            event.audio.channels = info_.input.channels;
            event.media_start_sample = 0;
            event.media_end_sample =
                static_cast<std::int64_t>(audio_.size() / info_.input.channels);
        }
        events_.push_back(std::move(event));
    }
    void response() {
        responding_ = true;
        ++epoch_;
        emit(SpeechEventKind::TurnStarted, {}, false);
        emit(SpeechEventKind::AgentText, offline_ ? "offline reply" : "live reply", false);
        emit(SpeechEventKind::AgentAudio, {}, false);
        auto& audio = events_.back();
        audio.audio = {audio_, info_.output->sample_rate, info_.output->channels, 0, 0};
        audio.media_start_sample = 0;
        audio.media_end_sample = static_cast<std::int64_t>(audio_.size() / info_.output->channels);
        if (!tools_)
            return;
        emit(SpeechEventKind::FunctionCallStarted, {}, false);
        for (std::size_t i = 0; i < tool_defs_.size(); ++i) {
            const auto& definition = tool_defs_[i];
            const std::string id = "call_" + std::to_string(epoch_) + "_" + std::to_string(i);
            pending_.push_back(id);
            emit(SpeechEventKind::FunctionCall, {}, true);
            events_.back().tool_call = ToolCall{id, definition.name, "{}"};
            auto found = std::find_if(acks_.begin(), acks_.end(), [&](const auto& ack) {
                return ack.first == definition.name;
            });
            if (found != acks_.end())
                emit(SpeechEventKind::AgentText, found->second.back(), true);
            else if (!defaults_.empty())
                emit(SpeechEventKind::AgentText, defaults_.back(), true);
        }
    }
    SpeechSessionInfo info_;
    std::unique_ptr<ConfigState> config_;
    bool offline_;
    std::string behavior_;
    bool tools_, stopped_{false}, responding_{false}, committed_{false};
    SpeechReadState state_{SpeechReadState::Active};
    std::uint64_t epoch_{1}, sequence_{0};
    std::int64_t frame_{0};
    std::vector<float> audio_;
    std::vector<SpeechDialogueEvent> events_;
    std::vector<ToolDefinition> tool_defs_;
    std::vector<std::pair<std::string, std::vector<std::string>>> acks_;
    std::vector<std::string> defaults_, pending_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

std::vector<ConfigField> example_voice_fields() {
    return {
        {"output_sample_rate", ConfigKind::I64, ConfigValue{std::int64_t{48000}}, "Playback rate."},
        {"seed", ConfigKind::I64, ConfigValue{std::int64_t{0}}, "Speech seed."},
        {"emit_agent_audio", ConfigKind::Bool, ConfigValue{true}, "Audio output."},
        {"emit_agent_text", ConfigKind::Bool, ConfigValue{true}, "Agent transcript."},
        {"emit_user_transcript", ConfigKind::Bool, ConfigValue{true}, "User transcript."},
        {"enable_barge_in", ConfigKind::Bool, ConfigValue{true}, "Interruption handling."}};
}
class ExampleVoiceDialogue final : public ISpeechDialogueSession {
  public:
    ExampleVoiceDialogue(const SpeechDialogueRequest& input, ConfigView config, std::string mode)
        : mode_(std::move(mode)) {
        const auto rules = example_voice_fields();
        std::set<std::string_view> seen;
        for (const auto& entry : config) {
            const auto found =
                std::find_if(rules.begin(), rules.end(),
                             [&](const ConfigField& field) { return field.name == entry.name; });
            if (found == rules.end() || !seen.insert(entry.name).second ||
                config_kind(entry.value) != found->kind)
                throw ConfigError("invalid VoiceChat example option");
            if (entry.name == "output_sample_rate") {
                const auto rate = std::get<std::int64_t>(entry.value);
                if (rate < 8000 || rate > 192000)
                    throw ConfigError("invalid example output rate");
                output_rate_ = static_cast<std::uint32_t>(rate);
            } else if (entry.name == "seed") {
                if (std::get<std::int64_t>(entry.value) < 0)
                    throw ConfigError("invalid example seed");
            } else if (!std::get<bool>(entry.value))
                throw ConfigError("VoiceChat example requires each output and barge-in option");
            // Only scalar values are accepted by this test mode.
            config_.push_back(entry);
        }
        if (seen.size() != rules.size() || !input.input.sample_rate || input.input.channels != 1)
            throw ConfigError("VoiceChat example omitted a required input or option");
        // The names above reference caller storage; retain model-independent
        // static declaration names before the create call returns.
        for (auto& entry : config_)
            entry.name = std::find_if(rules.begin(), rules.end(), [&](const ConfigField& field) {
                             return field.name == entry.name;
                         })->name;
        info_.input = {*input.input.sample_rate, 1};
        info_.output = SpeechAudioFormat{
            output_rate_ + (mode_ == "example_voicechat_bad_format" ? 1U : 0U), 1};
        if (input.system_prompt)
            info_.system_prompt = std::string(*input.system_prompt);
        if (mode_ == "example_voicechat_failed")
            state_ = SpeechReadState::Failed;
        ++live_sessions;
    }
    ~ExampleVoiceDialogue() override {
        stop();
        --live_sessions;
    }
    void stop() noexcept override {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        condition_.notify_all();
    }
    bool append_audio(Span<const float> input) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ || state_ != SpeechReadState::Active)
            throw std::invalid_argument("example speech epoch is not active");
        ++attempts_;
        if (attempts_ == 1) {
            rejected_.assign(input.begin(), input.end());
            return false;
        }
        if (attempts_ == 2 && (input.size() != rejected_.size() ||
                               !std::equal(input.begin(), input.end(), rejected_.begin())))
            throw std::invalid_argument("capture dropped or changed a backpressured audio chunk");
        emit(SpeechEventKind::UserTranscript,
             "heard " + std::to_string(input.size()) + " samples after " +
                 std::to_string(attempts_) + " attempts",
             true);
        emit(SpeechEventKind::AgentText, "reply", false);
        emit(SpeechEventKind::AgentText, "reply", true);
        emit(SpeechEventKind::AgentAudio, {}, false);
        auto& audio = events_.back().audio;
        audio.samples.assign(input.begin(), input.end());
        audio.channels = 1;
        audio.sample_rate = output_rate_ + (mode_ == "example_voicechat_bad_audio_rate" ? 1U : 0U);
        if (mode_ == "example_voicechat_yield")
            emit(SpeechEventKind::Yielded, "interrupted", true);
        condition_.notify_all();
        return true;
    }
    void finish_input() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        emit(SpeechEventKind::InputFinished, {}, true);
        state_ = SpeechReadState::EpochEnded;
        condition_.notify_all();
    }
    SpeechEventBatch read_events(std::int64_t timeout_ms) override {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto ready = [&] {
            return stopped_ || !events_.empty() || state_ != SpeechReadState::Active;
        };
        if (timeout_ms < 0)
            condition_.wait(lock, ready);
        else if (timeout_ms > 0)
            condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
        SpeechEventBatch result{std::move(events_),
                                stopped_ ? SpeechReadState::EpochEnded : state_};
        events_.clear();
        return result;
    }
    void cancel() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SpeechReadState::EpochEnded)
            return;
        emit(SpeechEventKind::Cancelled, {}, true);
        state_ = SpeechReadState::EpochEnded;
        condition_.notify_all();
    }
    void reset() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        state_ = SpeechReadState::Active;
        attempts_ = 0;
        rejected_.clear();
        emit(SpeechEventKind::Reset, {}, true);
        condition_.notify_all();
    }
    SpeechSessionInfo info() const override { return info_; }
    ConfigView effective_config() const override { return {config_.data(), config_.size()}; }

  private:
    void emit(SpeechEventKind kind, std::string text, bool final) {
        SpeechDialogueEvent event;
        event.kind = kind;
        event.text = std::move(text);
        event.is_final = final;
        event.epoch = 1;
        event.sequence = sequence_++;
        events_.push_back(std::move(event));
    }
    std::string mode_;
    SpeechSessionInfo info_;
    std::vector<ConfigEntry> config_;
    std::vector<float> rejected_;
    std::vector<SpeechDialogueEvent> events_;
    std::mutex mutex_;
    std::condition_variable condition_;
    SpeechReadState state_{SpeechReadState::Active};
    std::uint32_t output_rate_{48000};
    std::uint64_t attempts_{0}, sequence_{0};
    bool stopped_{false};
};
class Family final : public IModel,
                     public ITextContinuation,
                     public IStreamingSpeechTranscription,
                     public IStreamingTextToSpeech,
                     public IDuplexSpeechDialogue,
                     public IOfflineSpeechDialogue,
                     public IToolSpeechDialogue {
  public:
    explicit Family(std::string mode) : mode_(std::move(mode)) {
        if (mode_.find("example_streaming") == 0 && ++example_streaming_loads != 1)
            throw std::logic_error("persistent streaming example loaded its model more than once");
    }
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "example_streaming_unsupported")
            return {bind<ITextContinuation>(*this, fields_for(ITextContinuation::kTask))};
        return {bind<ITextContinuation>(*this, fields_for(ITextContinuation::kTask)),
                bind<IStreamingSpeechTranscription>(
                    *this, fields_for(IStreamingSpeechTranscription::kTask)),
                bind<IStreamingTextToSpeech>(*this, fields_for(IStreamingTextToSpeech::kTask)),
                bind<IDuplexSpeechDialogue>(*this, fields_for(IDuplexSpeechDialogue::kTask)),
                bind<IOfflineSpeechDialogue>(*this, fields_for(IOfflineSpeechDialogue::kTask)),
                bind<IToolSpeechDialogue>(*this, fields_for(IToolSpeechDialogue::kTask))};
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view task) const {
        if (mode_.find("example_voicechat") == 0 && task == IDuplexSpeechDialogue::kTask) {
            static const auto declared = example_voice_fields();
            return {declared.data(), declared.size()};
        }
        if (mode_.find("example_streaming") == 0 && task == IStreamingTextToSpeech::kTask) {
            static const ConfigField declared[] = {
                {"max_new_tokens", ConfigKind::I64, ConfigValue{std::int64_t{750}}, "Token limit."},
                {"chunk_frames", ConfigKind::I64, ConfigValue{std::int64_t{16}}, "Chunk frames."}};
            return declared;
        }
        if (task == ITextContinuation::kTask)
            return {};
        static const auto declared = fields();
        return {declared.data(), declared.size()};
    }
    TextResult run(const TextContinuationRequest&, ConfigView) override { return {"sync", {7}}; }
    std::unique_ptr<ISpeechTranscriptionStream>
    create(const StreamingSpeechTranscriptionRequest& input, ConfigView config) override {
        return std::make_unique<Asr>(input, config);
    }
    StreamingAudioSummary run(const TextToSpeechRequest& input, ConfigView config,
                              const AudioChunkSink& sink) override {
        if (mode_.find("example_streaming") == 0)
            return run_example_streaming(input, config, sink);
        ConfigState values(config);
        StreamingAudioSummary summary;
        if (input.text == "benchmark-fail")
            throw std::runtime_error("benchmark TTS provider failed");
        summary.output = {24000, 2};
        summary.inference_ms = 1.5;
        for (int i = 0; i < 3; ++i) {
            const float values[]{static_cast<float>(i), 0.25F, 0.5F, 0.75F};
            const bool keep_going = sink({{values, 4}, std::uint32_t{24000}, 2});
            summary.emitted_sample_count += 4;
            summary.emitted_frame_count += 2;
            if (!keep_going) {
                summary.outcome = AudioDeliveryOutcome::Stopped;
                break;
            }
        }
        if (input.text == "benchmark-stopped")
            summary.outcome = AudioDeliveryOutcome::Stopped;
        if (input.text == "benchmark-count-mismatch")
            summary.emitted_sample_count += 2;
        return summary;
    }
    std::unique_ptr<ISpeechDialogueSession>
    create_duplex_session(const SpeechDialogueRequest& input, ConfigView config) override {
        if (mode_.find("example_voicechat") == 0)
            return std::make_unique<ExampleVoiceDialogue>(input, config, mode_);
        return std::make_unique<Dialogue>(input, config, false, mode_);
    }
    std::unique_ptr<ISpeechDialogueSession>
    create_offline_session(const SpeechDialogueRequest& input, ConfigView config) override {
        return std::make_unique<Dialogue>(input, config, true, mode_);
    }
    std::unique_ptr<ISpeechDialogueSession>
    create_tool_session(const ToolSpeechDialogueRequest& input, ConfigView config) override {
        return std::make_unique<Dialogue>(input.dialogue, config, false, mode_, &input);
    }

  private:
    StreamingAudioSummary run_example_streaming(const TextToSpeechRequest& input, ConfigView config,
                                                const AudioChunkSink& sink) {
        const auto expected_tokens = mode_ == "example_streaming_defaults" ? 750 : 7;
        const auto expected_frames = mode_ == "example_streaming_defaults" ? 16 : 2;
        bool tokens = false, frames = false;
        for (const auto& entry : config) {
            if (!std::holds_alternative<std::int64_t>(entry.value))
                throw ConfigError("streaming example options must be integers");
            const auto value = std::get<std::int64_t>(entry.value);
            if (entry.name == "max_new_tokens" && !tokens && value == expected_tokens)
                tokens = true;
            else if (entry.name == "chunk_frames" && !frames && value == expected_frames)
                frames = true;
            else
                throw ConfigError("unexpected, duplicate or incorrect streaming example option");
        }
        if (!tokens || !frames || input.language)
            throw ConfigError("streaming example omitted its options or supplied a language");
        if (mode_ == "example_streaming_fail")
            throw std::runtime_error("streaming example provider failed");
        StreamingAudioSummary summary;
        summary.output = {22050, 1};
        if (mode_ == "example_streaming_empty")
            return summary;
        std::vector<std::vector<float>> chunks;
        if (input.text == "first prompt" && utterances_ == 0)
            chunks = {{0.25F, -0.5F}, {0.75F}};
        else if (input.text == "second prompt" && utterances_ == 1)
            chunks = {{-1.0F, 0.5F}};
        else
            throw std::invalid_argument("unexpected streaming example prompt or request order");
        ++utterances_;
        if (mode_ == "example_streaming_stereo") {
            summary.output.channels = 2;
            chunks = {{0.25F, -0.5F}};
        }
        if (mode_ == "example_streaming_bad_rate")
            summary.output.sample_rate = 0;
        for (const auto& chunk : chunks) {
            const bool keep_going = sink({{chunk.data(), chunk.size()},
                                          summary.output.sample_rate,
                                          summary.output.channels});
            summary.emitted_sample_count += chunk.size();
            summary.emitted_frame_count += chunk.size() / summary.output.channels;
            if (!keep_going) {
                summary.outcome = AudioDeliveryOutcome::Stopped;
                break;
            }
        }
        return summary;
    }
    std::string mode_;
    std::uint64_t utterances_{0};
};
} // namespace

extern "C" int trtmc_test_speech_waiting_readers() {
    return waiting_readers.load();
}
extern "C" int trtmc_test_speech_live_sessions() {
    return live_sessions.load();
}
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "speech_fixture")
        throw std::invalid_argument("wrong speech fixture");
    return new Family(context.reader.info().task);
}

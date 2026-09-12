/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/stream.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace {
using namespace trtmc::internal;
std::atomic<int> live_models{0};
std::atomic<int> waiting_readers{0};
using AllocationHook = void (*)();
std::atomic<AllocationHook> allocation_hook{nullptr};

class Stream final : public ITextStream {
  public:
    Stream(std::string prefix, std::vector<std::int32_t> tokens, std::string suffix, bool wait,
           std::string failure)
        : text_(std::move(prefix) + suffix), tokens_(std::move(tokens)), wait_(wait),
          failure_(std::move(failure)) {}
    std::optional<TextStreamEvent> next(std::int64_t timeout_ms) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (terminal_emitted_)
            throw std::logic_error("fixture terminal event was already consumed");
        if (failure_ == "throw")
            throw std::runtime_error("fixture next failed");
        if (failure_ == "blocking_timeout")
            return std::nullopt;
        if (failure_ == "malformed") {
            terminal_emitted_ = true;
            return TextStreamEvent{StreamEventKind::Complete, {}, {}, std::nullopt};
        }
        if (failure_ == "packing_oom") {
            terminal_emitted_ = true;
            std::optional<TextStreamEvent> event{
                TextStreamEvent{StreamEventKind::Complete, {}, {}, TextResult{text_, tokens_}}};
            const auto hook = allocation_hook.load();
            if (!hook)
                throw std::logic_error("packing fault hook is not installed");
            hook(); // Fail the C API's allocation AFTER this event is consumed.
            return event;
        }
        if (wait_ && !cancelled_) {
            ++waiting_readers;
            bool ready = false;
            if (timeout_ms < 0) {
                cv_.wait(lock, [&] { return cancelled_; });
                ready = true;
            } else
                ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                     [&] { return cancelled_; });
            --waiting_readers;
            if (!ready)
                return std::nullopt;
        }
        if (cancelled_) {
            terminal_emitted_ = true;
            return TextStreamEvent{StreamEventKind::Cancelled, {}, {}, std::nullopt};
        }
        if (timeout_ms == 0 && !polled_) {
            polled_ = true;
            return std::nullopt;
        }
        if (!emitted_) {
            emitted_ = true;
            return TextStreamEvent{StreamEventKind::Delta, text_, tokens_, std::nullopt};
        }
        TextResult result{text_, tokens_, 0.25, 0.5};
        result.setup_ms = 0.125;
        terminal_emitted_ = true;
        return TextStreamEvent{StreamEventKind::Complete, {}, {}, std::move(result)};
    }
    void cancel() noexcept override {
        const std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        cv_.notify_all();
    }

  private:
    std::string text_;
    std::vector<std::int32_t> tokens_;
    bool wait_;
    std::string failure_;
    bool cancelled_{false}, polled_{false}, emitted_{false};
    bool terminal_emitted_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

class MediaStream final : public ITextStream {
  public:
    MediaStream(std::string before, float pixel, std::string after, std::string kind = "image",
                std::string detail = {})
        : before_(std::move(before)), pixel_(pixel), after_(std::move(after)),
          kind_(std::move(kind)), detail_(std::move(detail)) {}
    std::optional<TextStreamEvent> next(std::int64_t) override {
        if (terminal_)
            throw std::logic_error("fixture media terminal already consumed");
        if (cancelled_.load()) {
            terminal_ = true;
            return TextStreamEvent{StreamEventKind::Cancelled, {}, {}, std::nullopt};
        }
        if (stage_++ == 0)
            return TextStreamEvent{StreamEventKind::Delta, before_, {70}, std::nullopt};
        if (stage_ == 2) {
            tail_ = "|" + kind_ + ":" + std::to_string(pixel_) + detail_ + "|" + after_;
            return TextStreamEvent{StreamEventKind::Delta, tail_, {71}, std::nullopt};
        }
        terminal_ = true;
        return TextStreamEvent{
            StreamEventKind::Complete, {}, {}, TextResult{before_ + tail_, {70, 71}}};
    }
    void cancel() noexcept override { cancelled_.store(true); }

  private:
    std::string before_;
    float pixel_;
    std::string after_, tail_, kind_, detail_;
    unsigned stage_{0};
    bool terminal_{false};
    std::atomic<bool> cancelled_{false};
};
float image_marker(const ImageView& image) {
    if (image.format != ImageFormat::Float32)
        throw std::invalid_argument("fixture requires float image");
    return static_cast<const float*>(image.data)[0];
}
std::string video_marker(const VideoView& video) {
    if (video.frames.empty() || video.timestamps_seconds.size() != video.frames.size())
        throw std::invalid_argument("fixture video requires exact frame timestamps");
    std::string result = ":V[";
    for (std::size_t i = 0; i < video.frames.size(); ++i)
        result += std::to_string(image_marker(video.frames[i])) + "@" +
                  std::to_string(video.timestamps_seconds[i]) + ";";
    return result + "]";
}
template <class Part>
struct MediaConversationData {
    explicit MediaConversationData(trtmc::Span<const MediaMessage<Part>> source) {
        bodies.reserve(source.size());
        messages.reserve(source.size());
        for (const auto& message : source) {
            auto& body = bodies.emplace_back();
            body.reserve(message.parts.size());
            for (const auto& part : message.parts)
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, ImageView>)
                            probe += ":I[" + std::to_string(image_marker(value)) + "]";
                        else if constexpr (std::is_same_v<T, VideoView>)
                            probe += video_marker(value);
                        else
                            body.emplace_back(value);
                    },
                    part);
            messages.push_back({message.role, {body.data(), body.size()}});
        }
    }
    TextConversationRequest text(trtmc::Span<const ToolDefinitionView> tools) const {
        return {{messages.data(), messages.size()}, tools};
    }
    std::vector<std::vector<ConversationPartView>> bodies;
    std::vector<ConversationMessageView> messages;
    std::string probe;
};
ConversationResult conversation_response(const TextConversationRequest& input,
                                         std::string_view suffix, std::string_view fault) {
    if (input.messages.size() != 4 || input.tools.size() != 2 ||
        input.messages[0].role != MessageRole::System ||
        input.messages[1].role != MessageRole::Assistant ||
        input.messages[2].role != MessageRole::Tool || input.messages[3].role != MessageRole::User)
        throw std::invalid_argument(
            "fixture requires structured system/assistant/tool/user history and two tools");
    const auto& assistant = input.messages[1].parts;
    if (assistant.size() != 2 || input.messages[0].parts.size() != 1 ||
        input.messages[2].parts.size() != 1 || input.messages[3].parts.size() != 1)
        throw std::invalid_argument("fixture history part counts");
    const auto* reasoning = std::get_if<ReasoningPartView>(&assistant[0]);
    const auto* previous_call = std::get_if<ToolCallView>(&assistant[1]);
    const auto* previous_result = std::get_if<ToolResultView>(&input.messages[2].parts[0]);
    const auto* question = std::get_if<TextPartView>(&input.messages[3].parts[0]);
    if (!reasoning || !previous_call || !previous_result || !question ||
        previous_call->call_id != previous_result->call_id ||
        previous_call->state != ToolCallState::Unknown || input.tools[0].name != "lookup" ||
        input.tools[1].name != "search")
        throw std::invalid_argument("fixture requires typed reasoning, correlated tool result and "
                                    "unchanged unknown call state");
    ConversationResult result;
    const auto state = fault == "unknown" ? ToolCallState::Unknown : ToolCallState::Complete;
    result.parts = {ReasoningPart{std::string(reasoning->text) + " -> thinking"},
                    TextPart{std::string(question->text) + " -> answer" + std::string(suffix)},
                    ToolCall{"call-a", std::string(input.tools[0].name), R"({"a":"x\"y"})", state},
                    ToolCall{"call-b", std::string(input.tools[1].name), R"({"b":2})", state}};
    result.token_ids = {31, 32};
    result.finish_reason = FinishReason::ToolCalls;
    result.usage = {11, 7, 18};
    result.setup_ms = 0.125;
    result.prefill_ms = 0.25;
    result.decode_ms = 0.5;
    if (fault == "partial") {
        auto& call = std::get<ToolCall>(result.parts[2]);
        call.arguments_json = R"({"a":)";
        call.state = ToolCallState::Incomplete;
        result.finish_reason = FinishReason::Length;
    } else if (fault == "malformed_tool") {
        auto& call = std::get<ToolCall>(result.parts[2]);
        call.arguments_json = "{]";
        call.state = ToolCallState::Malformed;
        result.finish_reason = FinishReason::Stop;
    } else if (fault == "unknown") {
        result.finish_reason = FinishReason::Unknown;
        result.usage = {};
    } else if (fault == "finish_other") {
        result.finish_reason = FinishReason::Other;
        result.other_finish_reason = "provider_exit";
    } else if (fault == "zero_usage") {
        result.usage = {0, 0, 0};
    }
    return result;
}
class ConversationStream final : public IConversationStream {
  public:
    ConversationStream(ConversationResult result, bool wait, std::string fault)
        : result_(std::move(result)), wait_(wait), fault_(std::move(fault)) {}
    std::optional<ConversationStreamEvent> next(std::int64_t timeout) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (terminal_)
            throw std::logic_error("fixture conversation terminal already consumed");
        if (fault_ == "throw")
            throw std::runtime_error("fixture conversation read failed");
        if (fault_ == "blocking_timeout")
            return std::nullopt;
        if (wait_ && !cancelled_) {
            ++waiting_readers;
            bool ready = false;
            if (timeout < 0) {
                cv_.wait(lock, [&] { return cancelled_; });
                ready = true;
            } else
                ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout),
                                     [&] { return cancelled_; });
            --waiting_readers;
            if (!ready)
                return std::nullopt;
        }
        if (cancelled_) {
            terminal_ = true;
            return ConversationCancelled{};
        }
        if (timeout == 0 && !polled_) {
            polled_ = true;
            return std::nullopt;
        }
        const auto& first = std::get<ToolCall>(result_.parts[2]);
        const auto& second = std::get<ToolCall>(result_.parts[3]);
        const bool partial = fault_ == "partial" || fault_ == "malformed_tool";
        switch (stage_++) {
        case 0:
            return ConversationReasoningDelta{fault_ == "bad_index" ? 2U : 0U,
                                              std::get<ReasoningPart>(result_.parts[0]).text};
        case 1:
            return ConversationTextDelta{1, std::get<TextPart>(result_.parts[1]).text};
        case 2:
            return ConversationToolDelta{2, std::nullopt, "look",
                                         partial ? first.arguments_json : R"({"a":)"};
        case 3:
            return ConversationToolDelta{3, fault_ == "duplicate_id" ? "call-a" : "call-b",
                                         "search", R"({"b":)"};
        case 4:
            return ConversationToolDelta{2, "call-a", "up", partial ? "" : R"("x\")"};
        case 5:
            return ConversationToolDelta{3, std::nullopt, "", "2}"};
        case 6:
            return ConversationToolEnd{3, second.state};
        case 7:
            if (fault_ == "after_end")
                return ConversationToolDelta{3, std::nullopt, "", "!"};
            return ConversationToolDelta{
                2, fault_ == "changed_id" ? std::optional<std::string>{"changed"} : std::nullopt,
                "", partial ? "" : R"(y"})"};
        case 8:
            return ConversationToolEnd{2, first.state};
        case 9:
            return ConversationTokenDelta{{31, 32}};
        default: {
            terminal_ = true;
            if (fault_ == "bad_final")
                result_.parts.emplace_back(TextPart{"unannounced"});
            std::optional<ConversationStreamEvent> event{ConversationComplete{std::move(result_)}};
            if (fault_ == "packing_oom") {
                const auto hook = allocation_hook.load();
                if (!hook)
                    throw std::logic_error("conversation allocation hook missing");
                hook();
            }
            return event;
        }
        }
    }
    void cancel() noexcept override {
        const std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        cv_.notify_all();
    }

  private:
    ConversationResult result_;
    bool wait_;
    std::string fault_;
    std::size_t stage_{0};
    bool cancelled_{false}, terminal_{false}, polled_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};
class Model final : public IModel,
                    public ITextContinuation,
                    public IStreamingTextContinuation,
                    public IStreamingImagesTextToText,
                    public ITextConversation,
                    public IStreamingTextConversation,
                    public IStreamingVideoTextToText,
                    public IStreamingImageVideoTextToText,
                    public IImagesTextConversation,
                    public IVideoTextConversation,
                    public IStreamingImagesTextConversation,
                    public IStreamingVideoTextConversation,
                    public trtmc::ILoraAdapterManager {
  public:
    explicit Model(std::string mode) : mode_(std::move(mode)) { ++live_models; }
    ~Model() override { --live_models; }
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "disabled")
            return {bind<ITextContinuation>(*this, fields_for(ITextContinuation::kTask))};
        return {
            bind<ITextContinuation>(*this, fields_for(ITextContinuation::kTask)),
            bind<IStreamingTextContinuation>(*this, fields_for(IStreamingTextContinuation::kTask)),
            bind<IStreamingImagesTextToText>(*this, fields_for(IStreamingImagesTextToText::kTask)),
            bind<ITextConversation>(*this, fields_for(ITextConversation::kTask)),
            bind<IStreamingTextConversation>(*this, fields_for(IStreamingTextConversation::kTask)),
            bind<IStreamingVideoTextToText>(*this, fields_for(IStreamingVideoTextToText::kTask)),
            bind<IStreamingImageVideoTextToText>(*this,
                                                 fields_for(IStreamingImageVideoTextToText::kTask)),
            bind<IImagesTextConversation>(*this, fields_for(IImagesTextConversation::kTask)),
            bind<IVideoTextConversation>(*this, fields_for(IVideoTextConversation::kTask)),
            bind<IStreamingImagesTextConversation>(
                *this, fields_for(IStreamingImagesTextConversation::kTask)),
            bind<IStreamingVideoTextConversation>(
                *this, fields_for(IStreamingVideoTextConversation::kTask))};
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view task) const {
        if (task == ITextContinuation::kTask)
            return {};
        if ((task == IStreamingImagesTextToText::kTask ||
             task == IStreamingVideoTextToText::kTask ||
             task == IStreamingImageVideoTextToText::kTask) &&
            mode_ != "disabled")
            return {};
        if ((task != IStreamingTextContinuation::kTask &&
             task != IStreamingTextConversation::kTask && task != ITextConversation::kTask &&
             task != IImagesTextConversation::kTask && task != IVideoTextConversation::kTask &&
             task != IStreamingImagesTextConversation::kTask &&
             task != IStreamingVideoTextConversation::kTask) ||
            mode_ == "disabled")
            throw UnsupportedTask("stream is disabled");
        static const ConfigField declared[] = {
            {"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}}, "Text suffix"},
            {"wait_for_cancel", ConfigKind::Bool, ConfigValue{false}, "Wait until cancelled"},
            {"failure", ConfigKind::String, ConfigValue{std::string_view{}},
             "Protocol fault fixture"}};
        return declared;
    }
    TextResult run(const TextContinuationRequest&, ConfigView config) override {
        if (!config.empty())
            throw ConfigError("sync fixture has no options");
        return {"sync", {7}};
    }
    std::unique_ptr<ITextStream> start(const TextContinuationRequest& input,
                                       ConfigView config) override {
        const auto values = stream_options(IStreamingTextContinuation::kTask, config);
        std::string prefix;
        std::vector<std::int32_t> tokens;
        if (const auto* text = std::get_if<std::string_view>(&input.prefix)) {
            prefix = *text;
            tokens = {42};
        } else {
            prefix = "tokens";
            const auto view = std::get<trtmc::Span<const std::int32_t>>(input.prefix);
            if (!view.empty())
                tokens.assign(view.begin(), view.end());
        }
        // These copies are family-owned and complete before start returns.
        return std::make_unique<Stream>(std::move(prefix), std::move(tokens),
                                        std::string(config_value_as<std::string_view>(values[0])),
                                        config_value_as<bool>(values[1]),
                                        std::string(config_value_as<std::string_view>(values[2])));
    }
    std::unique_ptr<ITextStream> start_images_text(const ImagesTextToTextRequest& input,
                                                   ConfigView config) override {
        if (!input.tools.empty())
            throw std::invalid_argument("plain image fixture does not consume tool declarations");
        if (!config.empty())
            throw ConfigError("image stream fixture accepts no config");
        if (input.messages.size() != 1 || input.messages[0].role != MessageRole::User ||
            input.messages[0].parts.size() != 3)
            throw std::invalid_argument("fixture needs one user message with text/image/text");
        const auto& parts = input.messages[0].parts;
        const auto* before = std::get_if<TextPartView>(&parts[0]);
        const auto* image = std::get_if<ImageView>(&parts[1]);
        const auto* after = std::get_if<TextPartView>(&parts[2]);
        if (!before || !image || !after || image->format != ImageFormat::Float32)
            throw std::invalid_argument("fixture preserves exact text/image/text order");
        return std::make_unique<MediaStream>(std::string(before->text),
                                             static_cast<const float*>(image->data)[0],
                                             std::string(after->text));
    }
    std::unique_ptr<ITextStream> start_video_text(const VideoTextToTextRequest& input,
                                                  ConfigView config) override {
        if (!config.empty())
            throw ConfigError("plain video fixture accepts no config");
        if (!input.tools.empty() || input.messages.size() != 1 ||
            input.messages[0].role != MessageRole::User || input.messages[0].parts.size() != 3)
            throw std::invalid_argument("fixture requires text/video/text only");
        const auto& parts = input.messages[0].parts;
        const auto* before = std::get_if<TextPartView>(&parts[0]);
        const auto* video = std::get_if<VideoView>(&parts[1]);
        const auto* after = std::get_if<TextPartView>(&parts[2]);
        if (!before || !video || !after)
            throw std::invalid_argument("fixture requires exact text/video/text order");
        const auto marker = video_marker(*video);
        return std::make_unique<MediaStream>(std::string(before->text),
                                             image_marker(video->frames[0]),
                                             std::string(after->text), "video", marker);
    }
    std::unique_ptr<ITextStream> start_image_video_text(const ImageVideoTextToTextRequest& input,
                                                        ConfigView config) override {
        if (!config.empty())
            throw ConfigError("plain mixed fixture accepts no config");
        if (input.messages.size() != 1 || input.messages[0].role != MessageRole::User ||
            input.messages[0].parts.size() != 4)
            throw std::invalid_argument("fixture requires text/image/video/text");
        const auto& parts = input.messages[0].parts;
        const auto* before = std::get_if<TextPartView>(&parts[0]);
        const auto* image = std::get_if<ImageView>(&parts[1]);
        const auto* video = std::get_if<VideoView>(&parts[2]);
        const auto* after = std::get_if<TextPartView>(&parts[3]);
        if (!before || !image || !video || !after)
            throw std::invalid_argument("fixture requires exact mixed input order");
        return std::make_unique<MediaStream>(std::string(before->text), image_marker(*image),
                                             std::string(after->text), "image_video",
                                             video_marker(*video));
    }
    ConversationResult run(const TextConversationRequest& input, ConfigView config) override {
        const auto values = stream_options(ITextConversation::kTask, config);
        if (config_value_as<bool>(values[1]))
            throw ConfigError("one-shot fixture cannot wait for stream cancellation");
        return conversation_response(input, config_value_as<std::string_view>(values[0]),
                                     config_value_as<std::string_view>(values[2]));
    }
    std::unique_ptr<IConversationStream>
    start_text_conversation(const TextConversationRequest& input, ConfigView config) override {
        const auto values = stream_options(IStreamingTextConversation::kTask, config);
        const auto failure = config_value_as<std::string_view>(values[2]);
        return std::make_unique<ConversationStream>(
            conversation_response(input, config_value_as<std::string_view>(values[0]), failure),
            config_value_as<bool>(values[1]), std::string(failure));
    }
    ConversationResult run_conversation(const ImagesTextToTextRequest& input,
                                        ConfigView config) override {
        return media_response(input, stream_options(IImagesTextConversation::kTask, config), true);
    }
    ConversationResult run_conversation(const VideoTextToTextRequest& input,
                                        ConfigView config) override {
        return media_response(input, stream_options(IVideoTextConversation::kTask, config), true);
    }
    std::unique_ptr<IConversationStream>
    start_images_conversation(const ImagesTextToTextRequest& input, ConfigView config) override {
        const auto values = stream_options(IStreamingImagesTextConversation::kTask, config);
        return std::make_unique<ConversationStream>(
            media_response(input, values, false), config_value_as<bool>(values[1]),
            std::string(config_value_as<std::string_view>(values[2])));
    }
    std::unique_ptr<IConversationStream>
    start_video_conversation(const VideoTextToTextRequest& input, ConfigView config) override {
        const auto values = stream_options(IStreamingVideoTextConversation::kTask, config);
        return std::make_unique<ConversationStream>(
            media_response(input, values, false), config_value_as<bool>(values[1]),
            std::string(config_value_as<std::string_view>(values[2])));
    }
    trtmc::ILoraAdapterManager* lora_adapters() noexcept override { return this; }
    void load_lora_adapter(const std::string& id, const std::string&) override {
        adapters_.push_back(id);
    }
    void unload_lora_adapter(const std::string&) override { adapters_.clear(); }
    std::vector<std::string> loaded_lora_adapters() const override { return adapters_; }

  private:
    template <class Request>
    static ConversationResult
    media_response(const Request& input, const std::vector<ConfigValue>& values, bool one_shot) {
        if (one_shot && config_value_as<bool>(values[1]))
            throw ConfigError("one-shot cannot wait for cancellation");
        using Part = std::decay_t<decltype(input.messages[0].parts[0])>;
        MediaConversationData<Part> data(input.messages);
        auto result = conversation_response(data.text(input.tools),
                                            config_value_as<std::string_view>(values[0]),
                                            config_value_as<std::string_view>(values[2]));
        std::get<TextPart>(result.parts[1]).text += data.probe;
        return result;
    }
    std::vector<ConfigValue> stream_options(std::string_view task, ConfigView config) const {
        const auto fields = fields_for(task);
        std::vector<ConfigValue> values;
        for (const auto& field : fields)
            values.push_back(*field.default_value);
        std::vector<bool> seen(fields.size(), false);
        for (const auto& entry : config) {
            auto found = std::find_if(fields.begin(), fields.end(), [&](const ConfigField& field) {
                return field.name == entry.name;
            });
            if (found == fields.end())
                throw ConfigError("unknown stream option");
            const auto index = static_cast<std::size_t>(found - fields.begin());
            if (seen[index] || config_kind(entry.value) != found->kind)
                throw ConfigError("invalid stream option");
            seen[index] = true;
            values[index] = entry.value;
        }
        return values;
    }

    std::string mode_;
    std::vector<std::string> adapters_;
};
} // namespace

extern "C" int trtmc_test_stream_live_models() {
    return live_models.load();
}
extern "C" int trtmc_test_stream_waiting_readers() {
    return waiting_readers.load();
}
extern "C" void trtmc_test_stream_allocation_hook(AllocationHook hook) {
    allocation_hook.store(hook);
}
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "stream_fixture")
        throw std::runtime_error("wrong stream fixture family");
    return new Model(context.reader.info().task);
}

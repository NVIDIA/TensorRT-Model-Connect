/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/language.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function function, trtmc_status expected, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == expected;
    }
    check(rejected, message);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"language_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
trtmc::Model load(const std::filesystem::path& root, const std::string& mode) {
    const auto path = root / ("language_cpp_" + mode + ".bundle");
    bundle(path, mode);
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    return trtmc::Model::load(path.string(), options);
}
void contains(std::string_view text, std::string_view expected, const char* message) {
    check(text.find(expected) != std::string_view::npos, message);
}
void marker(const trtmc::TextContinuationResult& result, int expected) {
    check(result.token_ids().size() == 2 && result.token_ids()[0] == expected,
          "each multimodal Task calls its distinct family method");
    check(result.setup_ms() == 0.25 && result.prefill_ms() == 0.5 && result.decode_ms() == 0.75,
          "text timing fields survive language transport");
}
void batch_marker(const trtmc::ConversationBatchResult& result, int expected, uint64_t count) {
    check(result.size() == count, "batch returns one result for every independent request");
    for (uint64_t index = 0; index < count; ++index) {
        const auto ids = result[index].token_ids();
        check(ids.size() == 3 && ids[0] == expected && ids[1] == result[0].token_ids()[1] &&
                  ids[2] == static_cast<int32_t>(index),
              "each batch signature uses one distinct family method and preserves order");
    }
}
std::string batch_text(const trtmc::ConversationBatchResult& result, uint64_t index) {
    return std::get<trtmc::TextPart>(result[index].message().parts[1]).text;
}
void exercise(const trtmc::Model& model, const std::filesystem::path& root) {
    using namespace trtmc;
    const uint8_t pixels_a[] = {4, 5, 6}, pixels_b[] = {8, 9, 10};
    const ImageInput a({pixels_a, 3}, 1, 1), b({pixels_b, 3}, 1, 1);
    const VideoInput video{{a, b}, {0.0, 0.125}};
    const float samples[] = {0.25F, -0.25F, 0.5F, -0.5F};
    const AudioView audio{{samples, 4}, 16000, 2};
    const AlignedAudioVideoInput aligned{video, audio, -0.025};
    check(model.tasks().size() == 19,
          "all 19 independently implemented language signatures are discoverable");

    auto image_task = model.task<ImagesTextToText>();
    const ImagesTextToTextRequest image_input{
        {{MessageRole::System, {TextPart{"rules"}}},
         {MessageRole::User, {TextPart{"before"}, a, TextPart{"between"}, b, TextPart{"after"}}}}};
    auto images = image_task.run(image_input);
    marker(images, 1);
    check(images.text() == "1:[1:T:rules;][3:T:before;I:4;T:between;I:8;T:after;]!",
          "multi-image, interleaving, message roles and family default survive exactly");
    const auto fields = image_task.config_fields();
    check(fields.size() == 1 && fields[0].name == "suffix" && fields[0].default_value &&
              fields[0].default_value->get<std::string>() == "!",
          "family config is discoverable per Task");
    const ImagesTextToTextRequest batch_second{{{MessageRole::System, {TextPart{"second-system"}}},
                                                {MessageRole::User, {b, TextPart{"second"}, a}}},
                                               {{"second-tool", "second-description", "{}"}}};
    BatchImagesTextConversationRequest batch_request{
        {{image_input, {{"suffix", "first"}}}, {batch_second, {}}}};
    auto batch_task = model.task<BatchImagesTextConversation>();
    auto batch_bad = batch_request;
    batch_bad.items[1].config = {{"suffix", false}};
    rejects([&] { batch_task.run(batch_bad); }, TRTMC_INVALID_CONFIG,
            "invalid batch item config rejects before the family execution marker");
    batch_bad = batch_request;
    batch_bad.items[1].input.messages.clear();
    rejects([&] { batch_task.run(batch_bad); }, TRTMC_INVALID_ARGUMENT,
            "invalid second input cannot be synthesized from another item");
    auto batch_result = batch_task.run(batch_request);
    check(batch_result.size() == 2 && batch_result[0].token_ids()[0] == 13 &&
              batch_result[0].token_ids()[1] == 1 && batch_result[1].token_ids()[1] == 1 &&
              batch_result[1].token_ids()[2] == 1,
          "two independent conversations use one family batch method, preserving order");
    const auto first_reply = batch_result[0].message();
    const auto second_reply = batch_result[1].message();
    const auto& first_text = std::get<TextPart>(first_reply.parts[1]).text;
    const auto& second_text = std::get<TextPart>(second_reply.parts[1]).text;
    check(first_text == "[1:T:rules;][3:T:before;I:4;T:between;I:8;T:after;]first" &&
              second_text ==
                  "D:second-tool:second-description:{};[1:T:second-system;][3:I:8;T:second;I:4;]!",
          "batch item history, image count, tools and independent defaults are not merged");
    check(batch_result[0].finish_reason() == FinishReason::Unknown &&
              !batch_result[0].usage().total_tokens,
          "batch transport does not invent finish reason or token usage");
    rejects([&] { (void)batch_result.item(2); }, TRTMC_INVALID_ARGUMENT,
            "batch output index is checked");
    rejects([&] { batch_task.run({{}}); }, TRTMC_INVALID_ARGUMENT,
            "empty batch is not a scalar call");
    rejects(
        [&] {
            load(root, "bad_batch_count").task<BatchImagesTextConversation>().run(batch_request);
        },
        TRTMC_INTERNAL_ERROR, "provider output count mismatch cannot return a partial handle");
    auto detached_batch = [&] {
        auto local = load(root, "all");
        return local.task<BatchImagesTextConversation>().run(batch_request);
    }();
    auto moved_batch = std::move(detached_batch);
    check(detached_batch.size() == 0 && moved_batch[1].token_ids()[2] == 1,
          "batch result survives model release and clears moved-from size");

    TextConversationRequest batch_text_input{
        {{MessageRole::System, {TextPart{"text-system"}}},
         {MessageRole::User, {TextPart{"text-request"}}},
         {MessageRole::Assistant, {ReasoningPart{"thinking"}, ToolCall{"t1", "text-tool", "{}"}}},
         {MessageRole::Tool, {ToolResult{"t1", "text-tool-output", false}}}},
        {{"text-tool", "text-description", "{}"}}};
    TextConversationRequest second_text_input{{{MessageRole::User, {TextPart{"independent"}}}}, {}};
    auto text_batch = model.task<BatchTextConversation>().run(
        {{{batch_text_input, {}}, {second_text_input, {{"suffix", "?"}}}}});
    batch_marker(text_batch, 14, 2);
    contains(batch_text(text_batch, 0), "C:t1:text-tool:{};",
             "text batch retains typed call history");
    check(batch_text(text_batch, 1) == "[3:T:independent;]?",
          "per-item tool declarations and generation options do not leak into another request");
    const VideoTextToTextRequest batch_video_input{
        {{MessageRole::User, {TextPart{"video-request"}, video}}}, {}};
    auto video_batch = model.task<BatchVideoTextConversation>().run(
        {{{batch_video_input, {}}, {batch_video_input, {{"suffix", ""}}}}});
    batch_marker(video_batch, 15, 2);
    contains(batch_text(video_batch, 0), "V:2:I:4;I:8;t0.125000;",
             "independent video batch retains frame ordering and timeline");

    const AudioTextConversationRequest batch_audio_input{
        {{MessageRole::System, {TextPart{"audio-system"}}},
         {MessageRole::User, {audio, TextPart{"audio-request"}}},
         {MessageRole::Assistant,
          {ReasoningPart{"audio-thinking"}, ToolCall{"a1", "audio-tool", "{}"}}},
         {MessageRole::Tool, {ToolResult{"a1", "audio-tool-output", false}}}},
        {{"audio-tool", "audio-description", "{}"}}};
    auto audio_batch = model.task<BatchAudioTextConversation>().run(
        {{{batch_audio_input, {}}, {batch_audio_input, {{"suffix", "?"}}}}});
    batch_marker(audio_batch, 16, 2);
    contains(batch_text(audio_batch, 0), "A:16000:2:4:0.250000;",
             "audio batch retains PCM channels and explicitly supplied sample rate");
    contains(batch_text(audio_batch, 1), "C:a1:audio-tool:{};",
             "complete audio DTO retains tools and reasoning rather than flattening them");
    const ImageAudioTextConversationRequest batch_compound_input{
        {{MessageRole::User,
          {b, TextPart{"compound-before"}, audio, a, TextPart{"compound-after"}}}},
        {{"compound-tool", "compound-description", "{}"}}};
    auto compound_batch = model.task<BatchImageAudioTextConversation>().run(
        {{{batch_compound_input, {}}, {batch_compound_input, {}}}});
    batch_marker(compound_batch, 17, 2);
    contains(batch_text(compound_batch, 0), "I:8;T:compound-before;A:16000:2:4:0.250000;I:4;",
             "one compound request preserves interleaved images, audio and text");
    auto mixed_visual = model.task<BatchTextImagesVideoConversations>().run(
        {{{image_input, {}}, {batch_video_input, {}}, {batch_second, {}}, {batch_text_input, {}}}});
    batch_marker(mixed_visual, 18, 4);
    contains(batch_text(mixed_visual, 1),
             "V:2:", "mixed visual batch retains the video's request domain");
    contains(batch_text(mixed_visual, 3), "text-tool",
             "mixed visual text item retains its complete history");
    auto mixed_audio =
        model.task<BatchTextImagesAudioConversations>().run({{{image_input, {}},
                                                              {batch_audio_input, {}},
                                                              {batch_text_input, {}},
                                                              {batch_compound_input, {}}}});
    batch_marker(mixed_audio, 19, 4);
    contains(batch_text(mixed_audio, 1), "audio-tool", "mixed audio item has its own tools");
    contains(batch_text(mixed_audio, 2), "text-tool", "mixed text item has independent tools");
    contains(batch_text(mixed_audio, 3), "compound-before",
             "mixed compound item is not split into two requests");

    const Config all_kinds{{"suffix", ""},
                           {"fixture.flag", false},
                           {"fixture.number", int64_t{0}},
                           {"fixture.scale", 0.0},
                           {"fixture.numbers", std::vector<int64_t>{0, 7}},
                           {"fixture.scales", std::vector<double>{0.0, 0.5}},
                           {"fixture.labels", std::vector<std::string>{"", "last"}}};
    auto kinds = model.task<BatchAudioTextConversation>().run({{{batch_audio_input, all_kinds}}});
    contains(batch_text(kinds, 0), "fixture.flag=false;fixture.number=0;fixture.scale=0.000000;",
             "batch config preserves explicit false and numeric zero");
    contains(batch_text(kinds, 0),
             "fixture.numbers=[0,7,];fixture.scales=[0.000000,0.500000,];fixture.labels=[,last,];",
             "all typed config arrays and nested empty strings survive item-local conversion");
    check(model.task<BatchAudioTextConversation>().config_fields().size() == 7,
          "batch discovery describes family-owned item options for the selected Task");
    rejects(
        [&] {
            model.task<BatchTextConversation>().run(
                {{{batch_text_input, {{"suffix", "a"}, {"suffix", "b"}}}}});
        },
        TRTMC_INVALID_CONFIG, "duplicate item config is not overwritten by a map");
    rejects(
        [&] {
            model.task<BatchAudioTextConversation>().run(
                {{{AudioTextConversationRequest::from_parts({audio}), {}}}});
        },
        TRTMC_INVALID_ARGUMENT, "audio-only input cannot satisfy audio-and-text batch contract");
    rejects(
        [&] {
            model.task<BatchImageAudioTextConversation>().run(
                {{{ImageAudioTextConversationRequest::from_parts(
                       {audio, TextPart{"missing-image"}}),
                   {}}}});
        },
        TRTMC_INVALID_ARGUMENT, "compound input must include its required image");
    auto missing_rate = audio;
    missing_rate.sample_rate.reset();
    rejects(
        [&] {
            model.task<BatchAudioTextConversation>().run(
                {{{AudioTextConversationRequest::from_parts({missing_rate, TextPart{"a"}}), {}}}});
        },
        TRTMC_INVALID_ARGUMENT,
        "family can reject omitted audio rate instead of shared inventing one");
    auto invalid_video = video;
    invalid_video.timestamps_seconds[1] = std::numeric_limits<double>::quiet_NaN();
    rejects(
        [&] {
            model.task<BatchVideoTextConversation>().run(
                {{{VideoTextToTextRequest::from_parts({invalid_video, TextPart{"v"}}), {}}}});
        },
        TRTMC_INVALID_ARGUMENT, "batch video clock uses existing typed validation");
    rejects(
        [&] {
            load(root, "equal_batch_config").task<BatchImagesTextConversation>().run(batch_request);
        },
        TRTMC_INVALID_CONFIG,
        "provider batch option limits reject incompatible per-item overrides");
    rejects(
        [&] {
            load(root, "batch_failure")
                .task<BatchTextConversation>()
                .run({{{batch_text_input, {}}}});
        },
        TRTMC_INTERNAL_ERROR, "execution failure does not return a partial batch");
    auto structured_batch = load(root, "batch_structured")
                                .task<BatchTextConversation>()
                                .run({{{batch_text_input, {}}, {second_text_input, {}}}});
    check(structured_batch[0].finish_reason() == FinishReason::Length &&
              structured_batch[0].usage().output_tokens == 0 &&
              !structured_batch[0].usage().input_tokens &&
              structured_batch[0].c_view().parts[2].content.tool_call.state ==
                  TRTMC_TOOL_CALL_MALFORMED,
          "batch results preserve malformed calls and explicit zero versus absent usage");
    check(std::get<ToolCall>(structured_batch[1].message().parts[2]).arguments_json == "{broken",
          "malformed tool arguments remain caller-interpretable raw bytes");
    auto empty_suffix = image_task.run(image_input, {{"suffix", ""}});
    check(empty_suffix.text().back() == ']',
          "explicit empty config string overrides nonempty family default");
    rejects([&] { image_task.run(image_input, {{"suffix", false}}); }, TRTMC_INVALID_CONFIG,
            "family rejects config type mismatch instead of coercion");
    rejects([&] { image_task.run(image_input, {{"suffix", "a"}, {"suffix", "b"}}); },
            TRTMC_INVALID_CONFIG, "duplicate configs reach and are rejected by family parser");
    rejects([&] { image_task.run(image_input, {{"unknown", 1}}); }, TRTMC_INVALID_CONFIG,
            "unknown key is not dropped");
    const std::string nul("a\0b", 3);
    auto binary_text = image_task.run(ImagesTextToTextRequest::from_parts({a, TextPart{nul}}));
    contains(binary_text.text(), std::string_view(nul),
             "length-delimited text preserves embedded NUL");
    auto videos = model.task<VideoTextToText>().run(
        VideoTextToTextRequest::from_parts({TextPart{"v"}, video}));
    marker(videos, 2);
    contains(videos.text(), "V:2:I:4;I:8;t0.125000;",
             "video frame order and nonuniform clock pass through");
    auto mixed = model.task<ImageVideoTextToText>().run(
        ImageVideoTextToTextRequest::from_parts({a, TextPart{"m"}, video, b, video}));
    marker(mixed, 3);
    contains(mixed.text(), "I:4;T:m;V:2:I:4;I:8;t0.125000;I:8;V:2:",
             "nested video descriptor lifetimes and mixed ordering survive");
    auto heard = model.task<AudioTextToText>().run(
        AudioTextToTextRequest::from_parts({audio, TextPart{"a"}}));
    marker(heard, 4);
    contains(heard.text(), "A:16000:2:4:0.250000;",
             "PCM data, scalar count, channel layout and rate remain explicit");
    auto image_audio =
        model.task<ImageAudioToText>().run(ImageAudioToTextRequest::from_parts({b, audio}));
    marker(image_audio, 5);
    auto av =
        model.task<AudioVideoTextToText>().run(AudioVideoTextToTextRequest::from_parts({aligned}));
    marker(av, 6);
    contains(av.text(), ":-0.025000;",
             "aligned audio/video retains its shared clock offset without requiring text");
    auto three = model.task<ImageAudioTextToText>().run(
        ImageAudioTextToTextRequest::from_parts({a, TextPart{"all"}, audio}));
    marker(three, 7);
    auto spoken = model.task<ImageAudioTextToTextSpeechResponse>().run(
        ImageAudioTextToTextSpeechResponseRequest::from_parts({a, audio, TextPart{"speak"}}));
    contains(spoken.text(), "8:[3:I:4;A:", "paired text/speech calls its own interface");
    check(spoken.speech().channels == 2 && spoken.speech().sample_rate == 16000 &&
              spoken.speech().samples.size() == 4 && spoken.speech().samples[2] == 0.5F,
          "paired speech owns actual PCM output without inventing word alignment");
    auto moved_speech = std::move(spoken);
    check(spoken.view().speech.audio.sample_count == 0 && moved_speech.speech().samples.size() == 4,
          "moved result clears the source view and retains the destination");
    auto no_rate = audio;
    no_rate.sample_rate.reset();
    rejects(
        [&] {
            model.task<AudioTextToText>().run(
                AudioTextToTextRequest::from_parts({no_rate, TextPart{"a"}}));
        },
        TRTMC_INVALID_ARGUMENT, "omitted rate is not silently invented by shared transport");
    auto no_clock = aligned;
    no_clock.audio_start_seconds.reset();
    rejects(
        [&] {
            model.task<AudioVideoTextToText>().run(
                AudioVideoTextToTextRequest::from_parts({no_clock}));
        },
        TRTMC_INVALID_ARGUMENT, "aligned timing omission is subject to an actual family default");
    rejects(
        [&] {
            model.task<ImageAudioTextToText>().run(
                ImageAudioTextToTextRequest::from_parts({a, audio}));
        },
        TRTMC_INVALID_ARGUMENT, "text-required and text-optional signatures remain distinct");

    TextConversationRequest conversation{
        {{MessageRole::System, {TextPart{"system"}}},
         {MessageRole::Developer, {TextPart{"developer"}}},
         {MessageRole::User, {TextPart{"question"}}},
         {MessageRole::Assistant,
          {ReasoningPart{"thinking"}, ToolCall{"call-1", "lookup", R"({"x":1})"}}},
         {MessageRole::Tool, {ToolResult{"call-1", "result-text", false}}}},
        {{"lookup", "Find an item", R"({"type":"object"})"}}};
    auto reply = model.task<TextConversation>().run(conversation);
    const auto reply_view = reply.view();
    check(reply_view.part_count == 3 && reply_view.token_ids.size == 2 &&
              reply_view.token_ids.data[0] == 9 && reply_view.setup_ms == 0.25 &&
              reply_view.prefill_ms == 0.5 && reply_view.decode_ms == 0.75,
          "conversation structured output, tokens and timings remain intact");
    check(reply.finish_reason() == FinishReason::Unknown && !reply.usage().input_tokens &&
              !reply.usage().output_tokens && !reply.usage().total_tokens &&
              reply_view.parts[2].content.tool_call.state == TRTMC_TOOL_CALL_UNKNOWN,
          "one-shot transport does not infer tool completion, finish reason or usage");
    auto assistant = reply.message();
    check(assistant.role == MessageRole::Assistant &&
              std::get<ReasoningPart>(assistant.parts[0]).text == "private reasoning marker",
          "reasoning is distinguishable from user-facing text");
    const auto& response_text = std::get<TextPart>(assistant.parts[1]).text;
    contains(response_text, R"(D:lookup:Find an item:{"type":"object"};)",
             "tool schema and description reach family without generic JSON reinterpretation");
    contains(response_text, R"(C:call-1:lookup:{"x":1};)", "tool call arguments and ID preserved");
    contains(response_text, "[5:O:call-1:result-text:ok;]",
             "tool output remains correlated and false is preserved");
    check(std::get<ToolCall>(assistant.parts[2]).call_id == "next-1",
          "structured generated call is not auto-executed");
    conversation.messages.push_back(assistant);
    conversation.messages.push_back(
        {MessageRole::Tool, {ToolResult{"next-1", "next-output", true}}});
    auto next_reply = model.task<TextConversation>().run(conversation);
    contains(std::get<TextPart>(next_reply.message().parts[1]).text, "O:next-1:next-output:error;",
             "caller can append structured assistant output and correlated tool response");
    auto bad_conversation = conversation;
    std::get<ToolResult>(bad_conversation.messages.back().parts[0]).call_id = "unmatched";
    rejects([&] { model.task<TextConversation>().run(bad_conversation); }, TRTMC_INVALID_ARGUMENT,
            "family owns semantic tool-response correlation validation");

    auto label = model.task<TextLabelClassification>().run({"excellent"});
    check(label.raw_label() == "positive" && label.label_index() == 1 &&
              label.vocabulary()[0] == "negative" && label.view().token_ids.data[0] == 10,
          "generated labels carry vocabulary/index, not invented class probabilities");
    auto invalid = model.task<TextLabelClassification>().run({"unknown"});
    check(invalid.raw_label() == "not a label" && invalid.label_index() == -1,
          "invalid generated label is preserved as raw text and sentinel index");
    auto pair = model.task<TextPairLabelClassification>().run({"first", "second"});
    check(pair.raw_label() == "related" && pair.view().token_ids.data[0] == 11 &&
              pair.view().token_ids.data[1] == 5 && pair.view().token_ids.data[2] == 6,
          "pair classification retains both input roles instead of concatenating them");

    int32_t source[] = {4, 0, 6}, decoder[] = {8, 9};
    const uint8_t source_mask[] = {1, 0, 1};
    auto states = model.task<TextEncoderDecoderHiddenStates>().run(
        {{{source, 3}, {source_mask, 3}}, {{decoder, 2}, {}}});
    source[0] = 999;
    decoder[0] = 999;
    const auto values = states.view();
    check(values.encoder.last_hidden_state.rows == 3 &&
              values.encoder.last_hidden_state.columns == 2 &&
              values.decoder.last_hidden_state.rows == 2 &&
              values.decoder.last_hidden_state.columns == 3,
          "encoder and decoder have independent token axes and hidden widths");
    check(values.encoder.token_ids.data[0] == 4 && values.decoder.token_ids.data[0] == 8 &&
              values.encoder.attention_mask.data[1] == 0 &&
              values.decoder.attention_mask.data[1] == 1 &&
              values.encoder.last_hidden_state.data[2] == 1000,
          "result snapshots original tokens, preserves pad rows and materializes only documented "
          "all-valid mask");
    rejects(
        [&] {
            load(root, "bad_axis")
                .task<TextEncoderDecoderHiddenStates>()
                .run({{{source, 3}, {}}, {{decoder, 2}, {}}});
        },
        TRTMC_INTERNAL_ERROR, "wrong model result axes are reported as internal failure");
    rejects([&] { load(root, "bad_label").task<TextLabelClassification>().run({"x"}); },
            TRTMC_INTERNAL_ERROR, "out-of-vocabulary model label index is not exposed as valid");
    rejects([&] { load(root, "bad_calls").task<TextConversation>().run(conversation); },
            TRTMC_INTERNAL_ERROR, "duplicate generated call IDs cannot break response correlation");

    auto detached = [&] {
        auto local = load(root, "all");
        return local.task<TextEncoderDecoderHiddenStates>().run(
            {{{source, 3}, {}}, {{decoder, 2}, {}}});
    }();
    check(detached.view().decoder.last_hidden_state.rows == 2,
          "C++ result outlives local model and task wrapper");
    auto none = load(root, "none");
    check(none.tasks().empty() && !none.supports<ImagesTextToText>(),
          "declaration belongs to each model, not just its DSO class");
    auto single_only = load(root, "single_only");
    check(single_only.supports<ImagesTextToText>() &&
              !single_only.supports<BatchImagesTextConversation>(),
          "scalar support does not imply independent-request batching");
    rejects([&] { (void)single_only.task<BatchImagesTextConversation>(); }, TRTMC_UNSUPPORTED,
            "single-only model cannot receive a shared scalar-loop batch fallback");
    auto missing = load(root, "missing");
    check(missing.tasks().empty() && !missing.supports<ImagesTextToText>(),
          "a loaded variant without an implementation binding advertises no task");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        exercise(load(root, "all"), root);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return failures ? 1 : 0;
}

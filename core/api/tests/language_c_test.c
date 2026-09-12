/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/language.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <string.h>

static int failures;
static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static trtmc_string_view str(const char* value) {
    trtmc_string_view output = {value, strlen(value)};
    return output;
}
static int equal(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) && memcmp(value.data, expected, (size_t)value.size) == 0;
}
static int contains(trtmc_string_view value, const char* expected) {
    const size_t length = strlen(expected);
    for (uint64_t index = 0; index <= value.size && value.size - index >= length; ++index)
        if (memcmp(value.data + index, expected, length) == 0)
            return 1;
    return 0;
}
static void release_error(const trtmc_core_api_v1* core, trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}
static int bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    const int count =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"language_fixture\",\"task\":\"%s\",\"backend\":"
                 "\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 mode);
    if (count < 0 || (size_t)count >= sizeof(header))
        return 0;
    FILE* out = fopen(path, "wb");
    if (!out)
        return 0;
    int ok = fwrite(magic, 1, sizeof(magic), out) == sizeof(magic);
    for (unsigned shift = 0; shift < 64; shift += 8)
        ok = (fputc((int)(((uint64_t)count >> shift) & 255U), out) != EOF) && ok;
    ok = (fwrite(header, 1, (size_t)count, out) == (size_t)count) && ok;
    ok = (fwrite("PLAN", 1, 4, out) == 4) && ok;
    return fclose(out) == 0 && ok;
}
#define GET(name, id)                                                                              \
    if (core->model_get_task_api(model, str(id), 1, 0, &header, &error) != TRTMC_OK)               \
        return 2;                                                                                  \
    const trtmc_##name##_api_v1* name = (const trtmc_##name##_api_v1*)header

#define TEXT_RUN(name, input, marker)                                                              \
    do {                                                                                           \
        trtmc_result* output = NULL;                                                               \
        trtmc_text_result_view_v1 view = {0};                                                      \
        check(name->run(model, &input, NULL, &output, &error) == TRTMC_OK,                         \
              "C typed multimodal run succeeds");                                                  \
        check(name->result_view(output, &view, &error) == TRTMC_OK && view.token_ids.size == 2 &&  \
                  view.token_ids.data[0] == marker,                                                \
              "C distinct task table reaches matching family method");                             \
        core->result_release(output);                                                              \
        release_error(core, &error);                                                               \
    } while (0)

#define BATCH_RUN(name, input, marker, expected_count)                                             \
    do {                                                                                           \
        trtmc_result* output = NULL;                                                               \
        trtmc_conversation_result_view_v1 view = {0};                                              \
        uint64_t count = 0;                                                                        \
        int32_t call = 0;                                                                          \
        check(name->run(model, &input, &output, &error) == TRTMC_OK,                               \
              "C independent conversation batch succeeds");                                        \
        check(name->result_count(output, &count, &error) == TRTMC_OK && count == expected_count,   \
              "C batch count is independent-request count");                                       \
        for (uint64_t index = 0; index < count; ++index) {                                         \
            check(name->result_item_view(output, index, &view, &error) == TRTMC_OK &&              \
                      view.token_ids.size == 3 && view.token_ids.data[0] == marker &&              \
                      view.token_ids.data[2] == (int32_t)index,                                    \
                  "C batch table reaches its distinct family method with stable order");           \
            if (index == 0)                                                                        \
                call = view.token_ids.data[1];                                                     \
            check(view.token_ids.data[1] == call, "C results share one family execution marker");  \
        }                                                                                          \
        check(name->result_count(batch_result, &count, &error) == TRTMC_INVALID_ARGUMENT &&        \
                  count == 0,                                                                      \
              "another Task's conversation batch handle is rejected");                             \
        release_error(core, &error);                                                               \
        check(name->run(disabled, &input, &rejected, &error) == TRTMC_UNSUPPORTED &&               \
                  rejected == NULL,                                                                \
              "a foreign model cannot borrow batch support from the acquired table");              \
        release_error(core, &error);                                                               \
        core->result_release(output);                                                              \
    } while (0)

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    char path[4096], disabled_path[4096];
    if (snprintf(path, sizeof(path), "%s/language_c_all.bundle", argv[1]) >= (int)sizeof(path) ||
        snprintf(disabled_path, sizeof(disabled_path), "%s/language_c_none.bundle", argv[1]) >=
            (int)sizeof(disabled_path))
        return 2;
    if (!bundle(path, "all") || !bundle(disabled_path, "none"))
        return 2;
    const trtmc_core_api_v1* core = NULL;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    trtmc_model *model = NULL, *disabled = NULL;
    if (core->model_load(str(path), &options, &model, &error) != TRTMC_OK ||
        core->model_load(str(disabled_path), &options, &disabled, &error) != TRTMC_OK)
        return 2;
    const trtmc_api_header* header = NULL;
    uint64_t task_count = 0;
    check(core->model_task_count(model, &task_count, &error) == TRTMC_OK && task_count == 19,
          "C metadata exposes exactly the 19 actually implemented language Tasks");
    GET(images_text_to_text, TRTMC_TASK_IMAGES_TEXT_TO_TEXT);
    const uint8_t pixels[] = {4, 5, 6};
    const trtmc_image_input_v1 image = {pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_UINT8};
    trtmc_images_text_part_v1 image_parts[3] = {{0}};
    image_parts[0].kind = TRTMC_MEDIA_TEXT;
    image_parts[0].content.text = str("before");
    image_parts[1].kind = TRTMC_MEDIA_IMAGE;
    image_parts[1].content.image = image;
    image_parts[2].kind = TRTMC_MEDIA_TEXT;
    image_parts[2].content.text = str("after");
    trtmc_images_text_message_v1 image_message = {TRTMC_ROLE_USER, image_parts, 3};
    trtmc_images_text_to_text_request_v1 image_request = {&image_message, 1, NULL, 0};
    trtmc_result *image_result = NULL, *rejected = NULL;
    check(images_text_to_text->run(model, &image_request, NULL, &image_result, &error) == TRTMC_OK,
          "C image/text input succeeds");
    trtmc_text_result_view_v1 text_view = {0};
    check(images_text_to_text->result_view(image_result, &text_view, &error) == TRTMC_OK &&
              equal(text_view.text, "1:[3:T:before;I:4;T:after;]!"),
          "C input order and family default are preserved");
    check(images_text_to_text->run(disabled, &image_request, NULL, &rejected, &error) ==
                  TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "C table acquired from one model cannot bypass another model's declarations");
    release_error(core, &error);
    image_parts[1].kind = TRTMC_MEDIA_AUDIO;
    check(images_text_to_text->run(model, &image_request, NULL, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "closed image/text union rejects another modality before interpreting payload");
    release_error(core, &error);
    image_parts[1].kind = TRTMC_MEDIA_IMAGE;
    image_message.role = TRTMC_ROLE_TOOL;
    check(images_text_to_text->run(model, &image_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "media Task cannot smuggle a tool message");
    release_error(core, &error);
    image_message.role = TRTMC_ROLE_USER;
    image_request.messages = NULL;
    check(images_text_to_text->run(model, &image_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "null pointer with nonzero message count fails");
    release_error(core, &error);
    image_request.messages = &image_message;
    image_message.part_count = 1;
    check(images_text_to_text->run(model, &image_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "required image cannot be replaced by text-only input");
    release_error(core, &error);
    image_message.part_count = 3;

    GET(batch_images_text_conversation, TRTMC_TASK_BATCH_IMAGES_TEXT_CONVERSATION);
    const trtmc_api_header* wrong_version = (const trtmc_api_header*)1;
    check(core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGES_TEXT_CONVERSATION), 2, 0,
                                   &wrong_version, &error) == TRTMC_VERSION_MISMATCH &&
              wrong_version == NULL,
          "batch Task version negotiation clears rejected table output");
    release_error(core, &error);
    trtmc_images_text_part_v1 second_parts[4] = {{0}};
    second_parts[0] = image_parts[1];
    second_parts[1] = image_parts[0];
    second_parts[2] = image_parts[1];
    second_parts[3] = image_parts[2];
    trtmc_images_text_message_v1 second_message = {TRTMC_ROLE_USER, second_parts, 4};
    trtmc_tool_definition_v1 batch_tool = {str("lookup"), str("description"), str("{}")};
    trtmc_config_entry_v1 batch_suffix = {0};
    batch_suffix.name = str("suffix");
    batch_suffix.value.kind = TRTMC_CONFIG_STRING;
    batch_suffix.value.as.string = str("");
    trtmc_batch_images_text_conversation_item_v1 batch_items[2] = {
        {image_request, {&batch_suffix, 1}}, {{&second_message, 1, &batch_tool, 1}, {NULL, 0}}};
    trtmc_batch_images_text_conversation_request_v1 batch_request = {batch_items, 2};
    trtmc_result* batch_result = NULL;
    trtmc_conversation_result_view_v1 batch_view = {0};
    trtmc_config_entry_v1 bad_batch_option = {0};
    bad_batch_option.name = str("suffix");
    bad_batch_option.value.kind = TRTMC_CONFIG_BOOL;
    bad_batch_option.value.as.boolean = 2;
    batch_items[1].config.entries = &bad_batch_option;
    batch_items[1].config.count = 1;
    check(batch_images_text_conversation->run(model, &batch_request, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL && error && contains(core->error_message(error), "item[1]"),
          "C invalid wire config reports the item index before family execution");
    release_error(core, &error);
    batch_items[1].config.entries = NULL;
    batch_items[1].config.count = 0;
    check(batch_images_text_conversation->run(model, &batch_request, &batch_result, &error) ==
              TRTMC_OK,
          "C image conversations batch calls the family batch interface");
    uint64_t batch_count = 0;
    check(batch_images_text_conversation->result_count(batch_result, &batch_count, &error) ==
                  TRTMC_OK &&
              batch_count == 2,
          "C output count is independent request count, not image count");
    check(batch_images_text_conversation->result_item_view(batch_result, 0, &batch_view, &error) ==
                  TRTMC_OK &&
              batch_view.token_ids.data[0] == 13 && batch_view.token_ids.data[1] == 1 &&
              equal(batch_view.parts[1].content.text, "[3:T:before;I:4;T:after;]"),
          "C first item preserves explicit empty config and execution marker");
    check(batch_images_text_conversation->result_item_view(batch_result, 1, &batch_view, &error) ==
                  TRTMC_OK &&
              batch_view.token_ids.data[1] == 1 && batch_view.token_ids.data[2] == 1 &&
              equal(batch_view.parts[1].content.text,
                    "D:lookup:description:{};[3:I:4;T:before;I:4;T:after;]!"),
          "C second item preserves own tool declarations, images and family default");
    check(batch_images_text_conversation->result_item_view(image_result, 0, &batch_view, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              batch_view.part_count == 0,
          "scalar text result cannot be viewed as a conversation batch");
    release_error(core, &error);
    check(batch_images_text_conversation->run(disabled, &batch_request, &rejected, &error) ==
                  TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "batch table cannot bypass the target model's declarations");
    release_error(core, &error);

    const trtmc_image_input_v1 frames[] = {image, image};
    const double times[] = {0, 0.125};
    const trtmc_video_view_v1 video = {frames, 2, {times, 2}};
    GET(video_text_to_text, TRTMC_TASK_VIDEO_TEXT_TO_TEXT);
    trtmc_video_text_part_v1 video_parts[2] = {{0}};
    video_parts[0].kind = TRTMC_MEDIA_VIDEO;
    video_parts[0].content.video = video;
    video_parts[1].kind = TRTMC_MEDIA_TEXT;
    video_parts[1].content.text = str("video");
    trtmc_video_text_message_v1 video_message = {TRTMC_ROLE_USER, video_parts, 2};
    trtmc_video_text_to_text_request_v1 video_request = {&video_message, 1, NULL, 0};
    TEXT_RUN(video_text_to_text, video_request, 2);
    GET(image_video_text_to_text, TRTMC_TASK_IMAGE_VIDEO_TEXT_TO_TEXT);
    trtmc_image_video_text_part_v1 mixed_parts[3] = {{0}};
    mixed_parts[0].kind = TRTMC_MEDIA_IMAGE;
    mixed_parts[0].content.image = image;
    mixed_parts[1].kind = TRTMC_MEDIA_TEXT;
    mixed_parts[1].content.text = str("mixed");
    mixed_parts[2].kind = TRTMC_MEDIA_VIDEO;
    mixed_parts[2].content.video = video;
    trtmc_image_video_text_message_v1 mixed_message = {TRTMC_ROLE_USER, mixed_parts, 3};
    trtmc_image_video_text_to_text_request_v1 mixed_request = {&mixed_message, 1};
    TEXT_RUN(image_video_text_to_text, mixed_request, 3);
    const float samples[] = {0.25F, -0.25F, 0.5F, -0.5F};
    const trtmc_audio_view_v1 audio = {samples, 4, 1, 16000, 2};
    GET(audio_text_to_text, TRTMC_TASK_AUDIO_TEXT_TO_TEXT);
    trtmc_audio_text_part_v1 audio_parts[2] = {{0}};
    audio_parts[0].kind = TRTMC_MEDIA_AUDIO;
    audio_parts[0].content.audio = audio;
    audio_parts[1].kind = TRTMC_MEDIA_TEXT;
    audio_parts[1].content.text = str("audio");
    trtmc_audio_text_message_v1 audio_message = {TRTMC_ROLE_USER, audio_parts, 2};
    trtmc_audio_text_to_text_request_v1 audio_request = {&audio_message, 1};
    TEXT_RUN(audio_text_to_text, audio_request, 4);
    GET(image_audio_to_text, TRTMC_TASK_IMAGE_AUDIO_TO_TEXT);
    trtmc_image_audio_text_part_v1 ia_parts[3] = {{0}};
    ia_parts[0].kind = TRTMC_MEDIA_IMAGE;
    ia_parts[0].content.image = image;
    ia_parts[1].kind = TRTMC_MEDIA_AUDIO;
    ia_parts[1].content.audio = audio;
    ia_parts[2].kind = TRTMC_MEDIA_TEXT;
    ia_parts[2].content.text = str("paired");
    trtmc_image_audio_text_message_v1 ia_message = {TRTMC_ROLE_USER, ia_parts, 2};
    trtmc_image_audio_to_text_request_v1 ia_request = {&ia_message, 1};
    TEXT_RUN(image_audio_to_text, ia_request, 5);
    GET(audio_video_text_to_text, TRTMC_TASK_AUDIO_VIDEO_TEXT_TO_TEXT);
    trtmc_audio_video_text_part_v1 av_part = {0};
    av_part.kind = TRTMC_MEDIA_ALIGNED_AUDIO_VIDEO;
    av_part.content.aligned = (trtmc_aligned_audio_video_input_v1){video, audio, 1, -0.025};
    trtmc_audio_video_text_message_v1 av_message = {TRTMC_ROLE_USER, &av_part, 1};
    trtmc_audio_video_text_to_text_request_v1 av_request = {&av_message, 1};
    TEXT_RUN(audio_video_text_to_text, av_request, 6);
    av_part.content.aligned.has_audio_start = 2;
    check(audio_video_text_to_text->run(model, &av_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "audio/video origin presence is a checked C bool");
    release_error(core, &error);
    av_part.content.aligned.has_audio_start = 1;
    GET(image_audio_text_to_text, TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT);
    ia_message.part_count = 3;
    trtmc_image_audio_text_to_text_request_v1 iat_request = {&ia_message, 1};
    TEXT_RUN(image_audio_text_to_text, iat_request, 7);
    GET(image_audio_text_to_text_speech_response,
        TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT_SPEECH_RESPONSE);
    trtmc_image_audio_text_to_text_speech_response_request_v1 spoken_request = {&ia_message, 1};
    trtmc_result* spoken_result = NULL;
    check(image_audio_text_to_text_speech_response->run(model, &spoken_request, NULL,
                                                        &spoken_result, &error) == TRTMC_OK,
          "C paired text/speech response succeeds");
    trtmc_text_speech_result_view_v1 spoken_view = {0};
    check(image_audio_text_to_text_speech_response->result_view(spoken_result, &spoken_view,
                                                                &error) == TRTMC_OK &&
              spoken_view.text.token_ids.data[0] == 8 &&
              spoken_view.speech.audio.sample_rate == 16000 &&
              spoken_view.speech.audio.channels == 2 && spoken_view.speech.audio.samples[2] == 0.5F,
          "C paired response has separate interpretable text and PCM views");
    check(image_audio_text_to_text_speech_response->result_view(image_result, &spoken_view,
                                                                &error) == TRTMC_INVALID_ARGUMENT &&
              spoken_view.speech.audio.sample_count == 0,
          "wrong result type cannot masquerade as paired speech");
    release_error(core, &error);

    GET(text_conversation, TRTMC_TASK_TEXT_CONVERSATION);
    const trtmc_tool_definition_v1 tool = {str("lookup"), str("Lookup"),
                                           str("{\"type\":\"object\"}")};
    trtmc_conversation_part_v1 history[3] = {{0}};
    history[0].kind = TRTMC_CONVERSATION_TEXT;
    history[0].content.text = str("question");
    history[1].kind = TRTMC_CONVERSATION_TOOL_CALL;
    history[1].content.tool_call =
        (trtmc_tool_call_v1){str("c1"), str("lookup"), str("{\"x\":1}"), TRTMC_TOOL_CALL_UNKNOWN};
    history[2].kind = TRTMC_CONVERSATION_TOOL_RESULT;
    history[2].content.tool_result = (trtmc_tool_result_v1){str("c1"), str("ok"), 0};
    trtmc_conversation_message_v1 conversation_messages[] = {{TRTMC_ROLE_USER, &history[0], 1},
                                                             {TRTMC_ROLE_ASSISTANT, &history[1], 1},
                                                             {TRTMC_ROLE_TOOL, &history[2], 1}};
    trtmc_text_conversation_request_v1 conversation_request = {conversation_messages, 3, &tool, 1};
    trtmc_result* conversation_result = NULL;
    check(text_conversation->run(model, &conversation_request, NULL, &conversation_result,
                                 &error) == TRTMC_OK,
          "C conversation carries declared tools, tool call and correlated tool response");
    trtmc_conversation_result_view_v1 conversation_view = {0};
    check(text_conversation->result_view(conversation_result, &conversation_view, &error) ==
                  TRTMC_OK &&
              conversation_view.part_count == 3 &&
              conversation_view.parts[0].kind == TRTMC_CONVERSATION_REASONING &&
              conversation_view.parts[2].kind == TRTMC_CONVERSATION_TOOL_CALL &&
              equal(conversation_view.parts[2].content.tool_call.call_id, "next-1"),
          "C assistant result preserves separate reasoning, text and correlated next call");
    check(conversation_view.finish_reason == TRTMC_FINISH_UNKNOWN &&
              !conversation_view.usage.has_input_tokens &&
              !conversation_view.usage.has_output_tokens &&
              !conversation_view.usage.has_total_tokens &&
              conversation_view.parts[2].content.tool_call.state == TRTMC_TOOL_CALL_UNKNOWN,
          "C populated tool fields do not imply Complete and absent usage stays absent");
    history[2].content.tool_result.is_error = 2;
    check(text_conversation->run(model, &conversation_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "invalid C tool-result bool is rejected without coercion");
    release_error(core, &error);
    history[2].content.tool_result.is_error = 0;
    conversation_messages[0].parts = &history[1];
    check(text_conversation->run(model, &conversation_request, NULL, &rejected, &error) ==
              TRTMC_INVALID_ARGUMENT,
          "a user role cannot inject an assistant tool-call part");
    release_error(core, &error);
    conversation_messages[0].parts = &history[0];

    GET(batch_text_conversation, TRTMC_TASK_BATCH_TEXT_CONVERSATION);
    trtmc_batch_text_conversation_item_v1 text_batch_items[] = {
        {conversation_request, {NULL, 0}}, {conversation_request, {&batch_suffix, 1}}};
    trtmc_batch_text_conversation_request_v1 text_batch_request = {text_batch_items, 2};
    BATCH_RUN(batch_text_conversation, text_batch_request, 14, 2);
    GET(batch_video_text_conversation, TRTMC_TASK_BATCH_VIDEO_TEXT_CONVERSATION);
    trtmc_batch_video_text_conversation_item_v1 video_batch_items[] = {
        {video_request, {NULL, 0}}, {video_request, {&batch_suffix, 1}}};
    trtmc_batch_video_text_conversation_request_v1 video_batch_request = {video_batch_items, 2};
    BATCH_RUN(batch_video_text_conversation, video_batch_request, 15, 2);
    GET(batch_audio_text_conversation, TRTMC_TASK_BATCH_AUDIO_TEXT_CONVERSATION);
    trtmc_audio_text_conversation_part_v1 complete_audio_parts[5] = {{0}};
    complete_audio_parts[0].kind = TRTMC_MEDIA_AUDIO;
    complete_audio_parts[0].content.audio = audio;
    complete_audio_parts[1].kind = TRTMC_MEDIA_TEXT;
    complete_audio_parts[1].content.text = str("hear");
    complete_audio_parts[2].kind = TRTMC_MEDIA_REASONING;
    complete_audio_parts[2].content.text = str("thinking");
    complete_audio_parts[3].kind = TRTMC_MEDIA_TOOL_CALL;
    complete_audio_parts[3].content.tool_call = history[1].content.tool_call;
    complete_audio_parts[4].kind = TRTMC_MEDIA_TOOL_RESULT;
    complete_audio_parts[4].content.tool_result = history[2].content.tool_result;
    trtmc_audio_text_conversation_message_v1 complete_audio_messages[] = {
        {TRTMC_ROLE_USER, complete_audio_parts, 2},
        {TRTMC_ROLE_ASSISTANT, complete_audio_parts + 2, 2},
        {TRTMC_ROLE_TOOL, complete_audio_parts + 4, 1}};
    trtmc_audio_text_conversation_request_v1 complete_audio = {complete_audio_messages, 3, &tool,
                                                               1};
    trtmc_batch_audio_text_conversation_item_v1 audio_batch_items[] = {
        {complete_audio, {NULL, 0}}, {complete_audio, {&batch_suffix, 1}}};
    trtmc_batch_audio_text_conversation_request_v1 audio_batch_request = {audio_batch_items, 2};
    BATCH_RUN(batch_audio_text_conversation, audio_batch_request, 16, 2);
    GET(batch_image_audio_text_conversation, TRTMC_TASK_BATCH_IMAGE_AUDIO_TEXT_CONVERSATION);
    trtmc_image_audio_text_conversation_part_v1 complete_compound_parts[3] = {{0}};
    complete_compound_parts[0].kind = TRTMC_MEDIA_IMAGE;
    complete_compound_parts[0].content.image = image;
    complete_compound_parts[1].kind = TRTMC_MEDIA_AUDIO;
    complete_compound_parts[1].content.audio = audio;
    complete_compound_parts[2].kind = TRTMC_MEDIA_TEXT;
    complete_compound_parts[2].content.text = str("both");
    trtmc_image_audio_text_conversation_message_v1 complete_compound_message = {
        TRTMC_ROLE_USER, complete_compound_parts, 3};
    trtmc_image_audio_text_conversation_request_v1 complete_compound = {&complete_compound_message,
                                                                        1, NULL, 0};
    trtmc_batch_image_audio_text_conversation_item_v1 compound_batch_items[] = {
        {complete_compound, {NULL, 0}}, {complete_compound, {&batch_suffix, 1}}};
    trtmc_batch_image_audio_text_conversation_request_v1 compound_batch_request = {
        compound_batch_items, 2};
    BATCH_RUN(batch_image_audio_text_conversation, compound_batch_request, 17, 2);
    GET(batch_text_images_video_conversations, TRTMC_TASK_BATCH_TEXT_IMAGES_VIDEO_CONVERSATIONS);
    trtmc_batch_text_images_video_conversations_item_v1 visual_batch_items[4] = {0};
    visual_batch_items[0].input.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
    visual_batch_items[0].input.input.images_text = image_request;
    visual_batch_items[1].input.kind = TRTMC_BATCH_CONVERSATION_VIDEO_TEXT;
    visual_batch_items[1].input.input.video_text = video_request;
    visual_batch_items[2].input.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
    visual_batch_items[2].input.input.images_text = batch_items[1].input;
    visual_batch_items[3].input.kind = TRTMC_BATCH_CONVERSATION_TEXT;
    visual_batch_items[3].input.input.text = conversation_request;
    trtmc_batch_text_images_video_conversations_request_v1 visual_batch_request = {
        visual_batch_items, 4};
    BATCH_RUN(batch_text_images_video_conversations, visual_batch_request, 18, 4);
    GET(batch_text_images_audio_conversations, TRTMC_TASK_BATCH_TEXT_IMAGES_AUDIO_CONVERSATIONS);
    trtmc_batch_text_images_audio_conversations_item_v1 mixed_audio_items[4] = {0};
    mixed_audio_items[0].input.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
    mixed_audio_items[0].input.input.images_text = image_request;
    mixed_audio_items[1].input.kind = TRTMC_BATCH_CONVERSATION_AUDIO_TEXT;
    mixed_audio_items[1].input.input.audio_text = complete_audio;
    mixed_audio_items[2].input.kind = TRTMC_BATCH_CONVERSATION_TEXT;
    mixed_audio_items[2].input.input.text = conversation_request;
    mixed_audio_items[3].input.kind = TRTMC_BATCH_CONVERSATION_IMAGE_AUDIO_TEXT;
    mixed_audio_items[3].input.input.image_audio_text = complete_compound;
    trtmc_batch_text_images_audio_conversations_request_v1 mixed_audio_request = {mixed_audio_items,
                                                                                  4};
    BATCH_RUN(batch_text_images_audio_conversations, mixed_audio_request, 19, 4);
    mixed_audio_items[1].input.kind = TRTMC_BATCH_CONVERSATION_VIDEO_TEXT;
    check(batch_text_images_audio_conversations->run(model, &mixed_audio_request, &rejected,
                                                     &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL && error && contains(core->error_message(error), "item[1]"),
          "closed mixed audio batch rejects a video tag before interpreting the union");
    release_error(core, &error);
    mixed_audio_items[1].input.kind = TRTMC_BATCH_CONVERSATION_AUDIO_TEXT;
    visual_batch_items[0].input.kind = 999;
    check(batch_text_images_video_conversations->run(model, &visual_batch_request, &rejected,
                                                     &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "unknown mixed request tag is not a generic extension payload");
    release_error(core, &error);
    visual_batch_items[0].input.kind = TRTMC_BATCH_CONVERSATION_IMAGES_TEXT;
    complete_audio_parts[4].content.tool_result.is_error = 2;
    check(batch_audio_text_conversation->run(model, &audio_batch_request, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "complete audio history validates tool booleans using existing rules");
    release_error(core, &error);
    complete_audio_parts[4].content.tool_result.is_error = 0;
    complete_audio_parts[0].content.audio.has_sample_rate = 2;
    check(batch_audio_text_conversation->run(model, &audio_batch_request, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "audio-rate presence is checked without silently choosing a rate");
    release_error(core, &error);
    complete_audio_parts[0].content.audio.has_sample_rate = 1;

    GET(text_label_classification, TRTMC_TASK_TEXT_LABEL_CLASSIFICATION);
    trtmc_text_label_classification_request_v1 label_request = {str("unknown")};
    trtmc_result* label_result = NULL;
    check(text_label_classification->run(model, &label_request, NULL, &label_result, &error) ==
              TRTMC_OK,
          "C generated label call succeeds even if model label is unmapped");
    trtmc_generated_label_result_view_v1 label_view = {0};
    check(text_label_classification->result_view(label_result, &label_view, &error) == TRTMC_OK &&
              label_view.label_index == -1 && equal(label_view.raw_label, "not a label") &&
              label_view.vocabulary.size == 2,
          "invalid generated label remains interpretable instead of being replaced");
    GET(text_pair_label_classification, TRTMC_TASK_TEXT_PAIR_LABEL_CLASSIFICATION);
    trtmc_text_pair_label_classification_request_v1 pair_request = {str("abc"), str("12345")};
    trtmc_result* pair_result = NULL;
    check(text_pair_label_classification->run(model, &pair_request, NULL, &pair_result, &error) ==
              TRTMC_OK,
          "C paired labels have a distinct request");
    check(text_pair_label_classification->result_view(pair_result, &label_view, &error) ==
                  TRTMC_OK &&
              label_view.token_ids.size == 3 && label_view.token_ids.data[0] == 11 &&
              label_view.token_ids.data[1] == 3 && label_view.token_ids.data[2] == 5,
          "both pair operands retain their own roles");

    GET(text_encoder_decoder_hidden_states, TRTMC_TASK_TEXT_ENCODER_DECODER_HIDDEN_STATES);
    int32_t source[] = {4, 0, 6}, decoder[] = {8, 9};
    uint8_t mask[] = {1, 0, 1};
    trtmc_text_encoder_decoder_hidden_states_request_v1 states_request = {
        {{source, 3}, {mask, 3}}, {{decoder, 2}, {NULL, 0}}};
    trtmc_result* states_result = NULL;
    check(text_encoder_decoder_hidden_states->run(model, &states_request, NULL, &states_result,
                                                  &error) == TRTMC_OK,
          "C encoder/decoder states accept exact IDs and effective masks");
    source[0] = 99;
    decoder[0] = 99;
    mask[0] = 0;
    trtmc_encoder_decoder_states_result_view_v1 states_view = {0};
    check(text_encoder_decoder_hidden_states->result_view(states_result, &states_view, &error) ==
                  TRTMC_OK &&
              states_view.encoder.last_hidden_state.rows == 3 &&
              states_view.encoder.last_hidden_state.columns == 2 &&
              states_view.decoder.last_hidden_state.rows == 2 &&
              states_view.decoder.last_hidden_state.columns == 3 &&
              states_view.encoder.token_ids.data[0] == 4 &&
              states_view.decoder.token_ids.data[0] == 8 &&
              states_view.encoder.attention_mask.data[0] == 1 &&
              states_view.encoder.attention_mask.data[1] == 0 &&
              states_view.decoder.attention_mask.data[1] == 1,
          "independent token axes and masks are result-owned snapshots");
    states_request.source.attention_mask.count = 2;
    check(text_encoder_decoder_hidden_states->run(model, &states_request, NULL, &rejected,
                                                  &error) == TRTMC_INVALID_ARGUMENT,
          "mismatched source attention mask is rejected");
    release_error(core, &error);
    states_request.source.attention_mask.count = 3;
    mask[0] = 2;
    check(text_encoder_decoder_hidden_states->run(model, &states_request, NULL, &rejected,
                                                  &error) == TRTMC_INVALID_ARGUMENT,
          "mask values are C booleans rather than arbitrary attention weights");
    release_error(core, &error);
    mask[0] = 1;
    check(text_encoder_decoder_hidden_states->result_view(label_result, &states_view, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              states_view.encoder.last_hidden_state.count == 0,
          "classification result cannot be viewed as token states");
    release_error(core, &error);

    core->model_release(model);
    core->model_release(disabled);
    check(batch_images_text_conversation->result_item_view(batch_result, 1, &batch_view, &error) ==
                  TRTMC_OK &&
              equal(batch_view.parts[1].content.text,
                    "D:lookup:description:{};[3:I:4;T:before;I:4;T:after;]!"),
          "C batch owns nested conversation output after model release");
    core->result_release(batch_result);
    check(images_text_to_text->result_view(image_result, &text_view, &error) == TRTMC_OK &&
              text_view.token_ids.data[0] == 1,
          "C text owns its bytes after model release");
    check(image_audio_text_to_text_speech_response->result_view(spoken_result, &spoken_view,
                                                                &error) == TRTMC_OK &&
              spoken_view.speech.audio.samples[2] == 0.5F,
          "C paired audio owns samples after model release");
    check(text_conversation->result_view(conversation_result, &conversation_view, &error) ==
                  TRTMC_OK &&
              equal(conversation_view.parts[2].content.tool_call.arguments_json, "{\"x\":2}"),
          "C tool call owns its JSON bytes after model release");
    check(text_label_classification->result_view(label_result, &label_view, &error) == TRTMC_OK &&
              equal(label_view.raw_label, "not a label"),
          "C generated label owns vocabulary and raw text after model release");
    check(text_encoder_decoder_hidden_states->result_view(states_result, &states_view, &error) ==
                  TRTMC_OK &&
              states_view.encoder.token_ids.data[0] == 4,
          "C states retain token mapping after model and request mutation");
    core->result_release(image_result);
    core->result_release(spoken_result);
    core->result_release(conversation_result);
    core->result_release(label_result);
    core->result_release(pair_result);
    core->result_release(states_result);
    release_error(core, &error);
    return failures ? 1 : 0;
}

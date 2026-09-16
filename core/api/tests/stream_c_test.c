/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/stream.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <string.h>

static int failures;
static const trtmc_core_api_v1* core;
static void check(int condition, const char* label) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static trtmc_string_view string(const char* value) {
    trtmc_string_view out = {value, strlen(value)};
    return out;
}
static int equals(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) && memcmp(value.data, expected, (size_t)value.size) == 0;
}
static void clear(trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}
static void image_stream(const char* path, const trtmc_load_options_v1* options) {
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_streaming_images_text_to_text_api_v1* factory = NULL;
    const trtmc_text_stream_api_v1* api = NULL;
    trtmc_text_stream* stream = NULL;
    trtmc_result *first = NULL, *second = NULL, *final = NULL, *unexpected = NULL;
    trtmc_text_stream_event_view_v1 view = {0};
    float pixel[3] = {0.25F, 0, 0};
    char before[] = "before", after[] = "after";
    trtmc_images_text_part_v1 parts[3] = {0};
    trtmc_images_text_message_v1 message = {TRTMC_ROLE_USER, parts, 3};
    trtmc_images_text_to_text_request_v1 request = {&message, 1, NULL, 0};
    check(core->model_load(string(path), options, &model, &error) == TRTMC_OK,
          "load C image-stream model");
    clear(&error);
    if (!model)
        goto cleanup;
    check(core->model_get_task_api(model, string(TRTMC_TASK_STREAMING_IMAGES_TEXT_TO_TEXT), 1, 0,
                                   &header, &error) == TRTMC_OK,
          "C image stream has typed input table");
    clear(&error);
    factory = (const trtmc_streaming_images_text_to_text_api_v1*)header;
    if (!factory)
        goto cleanup;
    api = factory->stream_api;
    parts[0].kind = TRTMC_MEDIA_TEXT;
    parts[0].content.text = string(before);
    parts[1].kind = TRTMC_MEDIA_IMAGE;
    parts[1].content.image =
        (trtmc_image_input_v1){pixel, sizeof(pixel), 1, 1, 3, TRTMC_IMAGE_FLOAT32};
    parts[2].kind = TRTMC_MEDIA_TEXT;
    parts[2].content.text = string(after);
    check(factory->start(model, &request, NULL, &stream, &error) == TRTMC_OK,
          "C starts with typed ordered media");
    clear(&error);
    if (!stream)
        goto cleanup;
    pixel[0] = 0.75F;
    before[0] = 'X';
    after[0] = 'Y';
    core->model_release(model);
    model = NULL;
    check(api->next(stream, -1, &first, &error) == TRTMC_OK, "C image first partial event");
    clear(&error);
    check(api->event_view(first, &view, &error) == TRTMC_OK && equals(view.text_delta, "before"),
          "C start retains original text and role structure");
    clear(&error);
    check(api->next(stream, -1, &second, &error) == TRTMC_OK, "C image second partial event");
    clear(&error);
    check(api->event_view(second, &view, &error) == TRTMC_OK &&
              equals(view.text_delta, "|image:0.250000|after"),
          "C family retains actual image value and trailing text before start returns");
    clear(&error);
    check(api->next(stream, -1, &final, &error) == TRTMC_OK, "C image terminal event");
    clear(&error);
    check(api->event_view(final, &view, &error) == TRTMC_OK && view.kind == TRTMC_STREAM_COMPLETE &&
              equals(view.final_result.text, "before|image:0.250000|after"),
          "C image stream reuses final TextResult schema");
    clear(&error);
    check(api->next(stream, -1, &unexpected, &error) == TRTMC_END && !unexpected && !error,
          "C image stream has one terminal then END");
    api->release(stream);
    stream = NULL;
    check(api->event_view(first, &view, &error) == TRTMC_OK && equals(view.text_delta, "before"),
          "C owned image stream event outlives model and stream");
    clear(&error);
cleanup:
    if (api)
        api->release(stream);
    core->model_release(model);
    core->result_release(first);
    core->result_release(second);
    core->result_release(final);
    core->result_release(unexpected);
}

static void append_fragment(char* output, size_t capacity, trtmc_string_view fragment) {
    const size_t used = strlen(output);
    check(fragment.size < capacity - used, "fixture fragment buffer fits");
    if (fragment.size >= capacity - used)
        return;
    memcpy(output + used, fragment.data, (size_t)fragment.size);
    output[used + (size_t)fragment.size] = '\0';
}
static trtmc_result* conversation_final(const trtmc_conversation_stream_api_v1* api,
                                        trtmc_conversation_stream* stream,
                                        trtmc_conversation_result_view_v1* final_view) {
    trtmc_result *result = NULL, *final = NULL;
    trtmc_error* error = NULL;
    int count = 0;
    while (++count < 32) {
        const trtmc_status status = api->next(stream, -1, &result, &error);
        if (status == TRTMC_END)
            break;
        check(status == TRTMC_OK, "C media conversation next");
        clear(&error);
        if (status != TRTMC_OK)
            break;
        trtmc_conversation_stream_event_view_v1 view = {0};
        check(api->event_view(result, &view, &error) == TRTMC_OK,
              "C media conversation event view");
        clear(&error);
        if (view.kind == TRTMC_CONVERSATION_STREAM_COMPLETE) {
            check(!final, "C media conversation terminal is unique");
            core->result_release(final);
            final = result;
            result = NULL;
            *final_view = view.content.complete;
        }
        core->result_release(result);
        result = NULL;
    }
    check(count < 32 && final, "C media conversation terminates");
    core->result_release(result);
    return final;
}
static trtmc_result* plain_final(const trtmc_text_stream_api_v1* api, trtmc_text_stream* stream,
                                 trtmc_text_result_view_v1* final_view) {
    trtmc_result *result = NULL, *final = NULL;
    trtmc_error* error = NULL;
    int count = 0;
    while (++count < 16) {
        const trtmc_status status = api->next(stream, -1, &result, &error);
        if (status == TRTMC_END)
            break;
        check(status == TRTMC_OK, "C plain media next");
        clear(&error);
        if (status != TRTMC_OK)
            break;
        trtmc_text_stream_event_view_v1 view = {0};
        check(api->event_view(result, &view, &error) == TRTMC_OK, "C plain media event view");
        clear(&error);
        if (view.kind == TRTMC_STREAM_COMPLETE) {
            check(!final, "C plain media unique terminal");
            core->result_release(final);
            final = result;
            result = NULL;
            *final_view = view.final_result;
        }
        core->result_release(result);
        result = NULL;
    }
    check(count < 16 && final, "C plain media terminates");
    core->result_release(result);
    return final;
}
static void media_streams(trtmc_model* model, const trtmc_text_conversation_request_v1* source) {
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    trtmc_images_text_part_v1 image_parts[4][3];
    trtmc_video_text_part_v1 video_parts[4][3];
    trtmc_images_text_message_v1 image_messages[4];
    trtmc_video_text_message_v1 video_messages[4];
    float a[3] = {0.25F, 0, 0}, b[3] = {0.5F, 0, 0};
    double times[2] = {2.0, 4.5};
    trtmc_image_input_v1 frames[2] = {{a, sizeof(a), 1, 1, 3, TRTMC_IMAGE_FLOAT32},
                                      {b, sizeof(b), 1, 1, 3, TRTMC_IMAGE_FLOAT32}};
    trtmc_video_view_v1 video = {frames, 2, {times, 2}};
    trtmc_images_text_to_text_request_v1 images = {image_messages, 4, source->tools,
                                                   source->tool_count};
    trtmc_video_text_to_text_request_v1 videos = {video_messages, 4, source->tools,
                                                  source->tool_count};
    trtmc_result *image_result = NULL, *video_result = NULL, *sync_image = NULL, *sync_video = NULL,
                 *plain_result = NULL, *mixed_result = NULL;
    trtmc_conversation_stream *image_stream = NULL, *video_stream = NULL;
    trtmc_text_stream *plain_stream = NULL, *mixed_stream = NULL;
    trtmc_conversation_result_view_v1 image_view = {0}, video_view = {0}, sync_view = {0};
    trtmc_text_result_view_v1 plain_view = {0}, mixed_view = {0};
    const trtmc_streaming_images_text_conversation_api_v1* image_api = NULL;
    const trtmc_streaming_video_text_conversation_api_v1* video_api = NULL;
    const trtmc_images_text_conversation_api_v1* image_sync = NULL;
    const trtmc_video_text_conversation_api_v1* video_sync = NULL;
    const trtmc_streaming_video_text_to_text_api_v1* plain_api = NULL;
    const trtmc_streaming_image_video_text_to_text_api_v1* mixed_api = NULL;
    size_t i, j;
    memset(image_parts, 0, sizeof(image_parts));
    memset(video_parts, 0, sizeof(video_parts));
    for (i = 0; i < 4; ++i) {
        image_messages[i] = (trtmc_images_text_message_v1){source->messages[i].role, image_parts[i],
                                                           source->messages[i].part_count};
        video_messages[i] = (trtmc_video_text_message_v1){source->messages[i].role, video_parts[i],
                                                          source->messages[i].part_count};
        for (j = 0; j < source->messages[i].part_count; ++j) {
            const trtmc_conversation_part_v1* part = &source->messages[i].parts[j];
            uint32_t kind = 0;
            if (part->kind == TRTMC_CONVERSATION_TEXT ||
                part->kind == TRTMC_CONVERSATION_REASONING) {
                kind = part->kind == TRTMC_CONVERSATION_TEXT ? TRTMC_MEDIA_TEXT
                                                             : TRTMC_MEDIA_REASONING;
                image_parts[i][j].content.text = part->content.text;
                video_parts[i][j].content.text = part->content.text;
            } else if (part->kind == TRTMC_CONVERSATION_TOOL_CALL) {
                kind = TRTMC_MEDIA_TOOL_CALL;
                image_parts[i][j].content.tool_call = part->content.tool_call;
                video_parts[i][j].content.tool_call = part->content.tool_call;
            } else {
                kind = TRTMC_MEDIA_TOOL_RESULT;
                image_parts[i][j].content.tool_result = part->content.tool_result;
                video_parts[i][j].content.tool_result = part->content.tool_result;
            }
            image_parts[i][j].kind = kind;
            video_parts[i][j].kind = kind;
        }
    }
    image_parts[3][1].kind = TRTMC_MEDIA_IMAGE;
    image_parts[3][1].content.image = frames[0];
    image_messages[3].part_count = 2;
    video_parts[3][1].kind = TRTMC_MEDIA_VIDEO;
    video_parts[3][1].content.video = video;
    video_messages[3].part_count = 2;
#define MEDIA_TABLE(Name, Type, Id)                                                                \
    check(core->model_get_task_api(model, string(Id), 1, 0, &header, &error) == TRTMC_OK,          \
          "C media table " #Name);                                                                 \
    clear(&error);                                                                                 \
    Name = (const Type*)header;                                                                    \
    if (!Name)                                                                                     \
        goto cleanup;
    MEDIA_TABLE(image_api, trtmc_streaming_images_text_conversation_api_v1,
                TRTMC_TASK_STREAMING_IMAGES_TEXT_CONVERSATION)
    MEDIA_TABLE(video_api, trtmc_streaming_video_text_conversation_api_v1,
                TRTMC_TASK_STREAMING_VIDEO_TEXT_CONVERSATION)
    MEDIA_TABLE(image_sync, trtmc_images_text_conversation_api_v1,
                TRTMC_TASK_IMAGES_TEXT_CONVERSATION)
    MEDIA_TABLE(video_sync, trtmc_video_text_conversation_api_v1,
                TRTMC_TASK_VIDEO_TEXT_CONVERSATION)
    MEDIA_TABLE(plain_api, trtmc_streaming_video_text_to_text_api_v1,
                TRTMC_TASK_STREAMING_VIDEO_TEXT_TO_TEXT)
    MEDIA_TABLE(mixed_api, trtmc_streaming_image_video_text_to_text_api_v1,
                TRTMC_TASK_STREAMING_IMAGE_VIDEO_TEXT_TO_TEXT)
#undef MEDIA_TABLE
    check(image_sync->run(model, &images, NULL, &sync_image, &error) == TRTMC_OK,
          "C image one-shot full conversation");
    clear(&error);
    check(image_api->start(model, &images, NULL, &image_stream, &error) == TRTMC_OK,
          "C image structured stream full history");
    clear(&error);
    if (!image_stream)
        goto cleanup;
    a[0] = 0.75F;
    image_result = conversation_final(image_api->stream_api, image_stream, &image_view);
    check(image_sync->result_view(sync_image, &sync_view, &error) == TRTMC_OK,
          "C image one-shot typed snapshot");
    clear(&error);
    if (!image_result || !sync_image)
        goto cleanup;
    check(image_view.parts[1].content.text.size == sync_view.parts[1].content.text.size &&
              memcmp(image_view.parts[1].content.text.data, sync_view.parts[1].content.text.data,
                     (size_t)image_view.parts[1].content.text.size) == 0 &&
              equals(image_view.parts[1].content.text, "question -> answer!:I[0.250000]"),
          "C image stream and one-shot preserve same original media and structured history");
    image_api->stream_api->release(image_stream);
    image_stream = NULL;
    a[0] = 0.25F;
    check(video_sync->run(model, &videos, NULL, &sync_video, &error) == TRTMC_OK,
          "C video one-shot full conversation");
    clear(&error);
    check(video_api->start(model, &videos, NULL, &video_stream, &error) == TRTMC_OK,
          "C video structured stream full history");
    clear(&error);
    if (!video_stream)
        goto cleanup;
    a[0] = 0.75F;
    times[1] = 9.0;
    video_result = conversation_final(video_api->stream_api, video_stream, &video_view);
    if (!video_result || !sync_video)
        goto cleanup;
    check(video_sync->result_view(sync_video, &sync_view, &error) == TRTMC_OK,
          "C video one-shot snapshot");
    clear(&error);
    check(equals(video_view.parts[1].content.text,
                 "question -> answer!:V[0.250000@2.000000;0.500000@4.500000;]") &&
              video_view.parts[2].content.tool_call.state == TRTMC_TOOL_CALL_COMPLETE,
          "C video structured stream owns frame/time values and typed tool completion");
    video_api->stream_api->release(video_stream);
    video_stream = NULL;
    a[0] = 0.25F;
    times[1] = 4.5;
    {
        trtmc_video_text_part_v1 parts[3] = {0};
        trtmc_video_text_message_v1 message = {TRTMC_ROLE_USER, parts, 3};
        trtmc_video_text_to_text_request_v1 request = {&message, 1, NULL, 0};
        parts[0].kind = TRTMC_MEDIA_TEXT;
        parts[0].content.text = string("before");
        parts[1].kind = TRTMC_MEDIA_VIDEO;
        parts[1].content.video = video;
        parts[2].kind = TRTMC_MEDIA_TEXT;
        parts[2].content.text = string("after");
        check(plain_api->start(model, &request, NULL, &plain_stream, &error) == TRTMC_OK,
              "C plain video stream");
        clear(&error);
        if (!plain_stream)
            goto cleanup;
        plain_result = plain_final(plain_api->stream_api, plain_stream, &plain_view);
        check(equals(plain_view.text,
                     "before|video:0.250000:V[0.250000@2.000000;0.500000@4.500000;]|after"),
              "C plain video retains required temporal input");
        plain_api->stream_api->release(plain_stream);
        plain_stream = NULL;
    }
    {
        trtmc_image_video_text_part_v1 parts[4] = {0};
        trtmc_image_video_text_message_v1 message = {TRTMC_ROLE_USER, parts, 4};
        trtmc_image_video_text_to_text_request_v1 request = {&message, 1};
        parts[0].kind = TRTMC_MEDIA_TEXT;
        parts[0].content.text = string("before");
        parts[1].kind = TRTMC_MEDIA_IMAGE;
        parts[1].content.image = frames[1];
        parts[2].kind = TRTMC_MEDIA_VIDEO;
        parts[2].content.video = video;
        parts[3].kind = TRTMC_MEDIA_TEXT;
        parts[3].content.text = string("after");
        check(mixed_api->start(model, &request, NULL, &mixed_stream, &error) == TRTMC_OK,
              "C mixed required image/video stream");
        clear(&error);
        if (!mixed_stream)
            goto cleanup;
        mixed_result = plain_final(mixed_api->stream_api, mixed_stream, &mixed_view);
        check(equals(mixed_view.text,
                     "before|image_video:0.500000:V[0.250000@2.000000;0.500000@4.500000;]|after"),
              "C mixed route retains independent image and timed clip");
    }
cleanup:
    if (image_api)
        image_api->stream_api->release(image_stream);
    if (video_api)
        video_api->stream_api->release(video_stream);
    if (plain_api)
        plain_api->stream_api->release(plain_stream);
    if (mixed_api)
        mixed_api->stream_api->release(mixed_stream);
    core->result_release(image_result);
    core->result_release(video_result);
    core->result_release(sync_image);
    core->result_release(sync_video);
    core->result_release(plain_result);
    core->result_release(mixed_result);
}
static void conversation_stream(const char* path, const trtmc_load_options_v1* options) {
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_streaming_text_conversation_api_v1* factory = NULL;
    const trtmc_text_conversation_api_v1* sync = NULL;
    const trtmc_conversation_stream_api_v1* api = NULL;
    trtmc_conversation_stream* stream = NULL;
    trtmc_result *event = NULL, *first = NULL, *final = NULL, *baseline = NULL;
    trtmc_conversation_stream_event_view_v1 view = {0};
    trtmc_conversation_result_view_v1 complete = {0};
    trtmc_conversation_part_v1 system = {0}, assistant[2] = {{0}}, tool_result = {0}, user = {0};
    trtmc_conversation_message_v1 messages[4] = {{TRTMC_ROLE_SYSTEM, &system, 1},
                                                 {TRTMC_ROLE_ASSISTANT, assistant, 2},
                                                 {TRTMC_ROLE_TOOL, &tool_result, 1},
                                                 {TRTMC_ROLE_USER, &user, 1}};
    trtmc_tool_definition_v1 tools[2];
    trtmc_text_conversation_request_v1 input = {messages, 4, tools, 2};
    char prior[] = "prior", question[] = "question";
    char arguments[4][64] = {{0}}, names[4][32] = {{0}};
    int count = 0, delayed_id = 0, tool_ends = 0;
    system.kind = TRTMC_CONVERSATION_TEXT;
    system.content.text = string("rules");
    assistant[0].kind = TRTMC_CONVERSATION_REASONING;
    assistant[0].content.text = string(prior);
    assistant[1].kind = TRTMC_CONVERSATION_TOOL_CALL;
    assistant[1].content.tool_call = (trtmc_tool_call_v1){string("old"), string("lookup"),
                                                          string("{}"), TRTMC_TOOL_CALL_UNKNOWN};
    tool_result.kind = TRTMC_CONVERSATION_TOOL_RESULT;
    tool_result.content.tool_result = (trtmc_tool_result_v1){string("old"), string("done"), 0};
    user.kind = TRTMC_CONVERSATION_TEXT;
    user.content.text = string(question);
    tools[0] = (trtmc_tool_definition_v1){string("lookup"), string("first"),
                                          string("{\"type\":\"object\"}")};
    tools[1] = (trtmc_tool_definition_v1){string("search"), string("second"),
                                          string("{\"type\":\"object\"}")};
    check(core->model_load(string(path), options, &model, &error) == TRTMC_OK,
          "C structured stream model loads");
    clear(&error);
    if (!model)
        goto cleanup;
    check(core->model_get_task_api(model, string(TRTMC_TASK_TEXT_CONVERSATION), 1, 0, &header,
                                   &error) == TRTMC_OK,
          "C matching one-shot conversation contract");
    clear(&error);
    sync = (const trtmc_text_conversation_api_v1*)header;
    if (!sync)
        goto cleanup;
    check(sync->run(model, &input, NULL, &baseline, &error) == TRTMC_OK,
          "C one-shot shares structured request");
    clear(&error);
    check(core->model_get_task_api(model, string(TRTMC_TASK_STREAMING_TEXT_CONVERSATION), 1, 0,
                                   &header, &error) == TRTMC_OK,
          "C structured text stream table");
    clear(&error);
    factory = (const trtmc_streaming_text_conversation_api_v1*)header;
    if (!factory)
        goto cleanup;
    api = factory->stream_api;
    check(factory->start(model, &input, NULL, &stream, &error) == TRTMC_OK,
          "C structured stream retains typed history");
    clear(&error);
    if (!stream)
        goto cleanup;
    prior[0] = 'X';
    question[0] = 'Y';
    check(api->next(stream, 0, &event, &error) == TRTMC_AGAIN && !event && !error,
          "C structured timeout is AGAIN");
    while (1) {
        const trtmc_status status = api->next(stream, -1, &event, &error);
        if (status == TRTMC_END) {
            check(!event && !error, "C structured END is empty");
            break;
        }
        check(status == TRTMC_OK, "C structured event status");
        clear(&error);
        if (status != TRTMC_OK)
            break;
        ++count;
        check(api->event_view(event, &view, &error) == TRTMC_OK,
              "C structured event discriminated view");
        clear(&error);
        if (view.kind == TRTMC_CONVERSATION_STREAM_REASONING) {
            check(equals(view.content.reasoning.text_delta, "prior -> thinking"),
                  "C typed reasoning is retained separately");
            first = event;
            event = NULL;
        } else if (view.kind == TRTMC_CONVERSATION_STREAM_TEXT)
            check(equals(view.content.text.text_delta, "question -> answer!"),
                  "C answer does not contain reasoning");
        else if (view.kind == TRTMC_CONVERSATION_STREAM_TOOL_DELTA) {
            const trtmc_conversation_tool_delta_v1* delta = &view.content.tool_delta;
            check(delta->part_index < 4, "C stable tool part index");
            if (delta->part_index < 4) {
                append_fragment(names[delta->part_index], sizeof(names[0]), delta->name_delta);
                append_fragment(arguments[delta->part_index], sizeof(arguments[0]),
                                delta->arguments_delta);
            }
            if (delta->part_index == 2 && !delta->has_call_id)
                delayed_id = 1;
            if (delta->has_call_id)
                check(equals(delta->call_id, delta->part_index == 2 ? "call-a" : "call-b"),
                      "C immutable correlated tool ID");
        } else if (view.kind == TRTMC_CONVERSATION_STREAM_TOOL_END) {
            ++tool_ends;
            check(view.content.tool_end.state == TRTMC_TOOL_CALL_COMPLETE,
                  "C family-reported complete tool state");
        } else if (view.kind == TRTMC_CONVERSATION_STREAM_TOKENS)
            check(view.content.token_ids.size == 2 && view.content.token_ids.data[0] == 31,
                  "C raw token delta separate from parts");
        else if (view.kind == TRTMC_CONVERSATION_STREAM_COMPLETE) {
            final = event;
            event = NULL;
            complete = view.content.complete;
        }
        core->result_release(event);
        event = NULL;
    }
    check(count == 11 && delayed_id && tool_ends == 2, "C one terminal and indexed tool lifecycle");
    check(strcmp(names[2], "lookup") == 0 && strcmp(arguments[2], "{\"a\":\"x\\\"y\"}") == 0 &&
              strcmp(arguments[3], "{\"b\":2}") == 0,
          "C partial JSON escapes and interleaved calls are not repaired or flattened");
    check(complete.part_count == 4 && complete.finish_reason == TRTMC_FINISH_TOOL_CALLS &&
              complete.usage.has_input_tokens && complete.usage.input_tokens == 11 &&
              complete.usage.has_output_tokens && complete.usage.output_tokens == 7 &&
              complete.usage.has_total_tokens && complete.usage.total_tokens == 18,
          "C reported finish and usage do not derive from two token IDs");
    check(equals(complete.parts[2].content.tool_call.arguments_json, arguments[2]),
          "C final tool snapshot matches deltas");
    check(api->event_view(baseline, &view, &error) == TRTMC_INVALID_ARGUMENT,
          "C event reader rejects one-shot result handle");
    clear(&error);
    api->release(stream);
    stream = NULL;
    check(api->event_view(first, &view, &error) == TRTMC_OK &&
              equals(view.content.reasoning.text_delta, "prior -> thinking"),
          "C owned structured event outlives session");
    clear(&error);
    {
        trtmc_config_entry_v1 entry = {string("failure"),
                                       {TRTMC_CONFIG_STRING, {.string = {"unknown", 7}}}};
        trtmc_config_view_v1 config = {&entry, 1};
        check(factory->start(model, &input, &config, &stream, &error) == TRTMC_OK,
              "C unknown metadata stream");
        clear(&error);
        while (stream) {
            const trtmc_status status = api->next(stream, -1, &event, &error);
            if (status == TRTMC_END)
                break;
            check(status == TRTMC_OK, "C unknown metadata event");
            clear(&error);
            if (status != TRTMC_OK)
                break;
            check(api->event_view(event, &view, &error) == TRTMC_OK, "C unknown metadata view");
            clear(&error);
            if (view.kind == TRTMC_CONVERSATION_STREAM_COMPLETE)
                check(view.content.complete.finish_reason == TRTMC_FINISH_UNKNOWN &&
                          !view.content.complete.usage.has_input_tokens &&
                          !view.content.complete.usage.has_output_tokens &&
                          !view.content.complete.usage.has_total_tokens &&
                          view.content.complete.parts[2].content.tool_call.state ==
                              TRTMC_TOOL_CALL_UNKNOWN,
                      "C unknown fields remain unknown/absent despite populated tool arguments");
            core->result_release(event);
            event = NULL;
        }
    }
    api->release(stream);
    stream = NULL;
    prior[0] = 'p';
    question[0] = 'q';
    media_streams(model, &input);
cleanup:
    if (api)
        api->release(stream);
    core->model_release(model);
    core->result_release(event);
    core->result_release(first);
    core->result_release(final);
    core->result_release(baseline);
}
int main(int argc, char** argv) {
    static const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    static const char header[] = "{\"format\":1,\"family\":\"stream_fixture\",\"task\":\"enabled\","
                                 "\"backend\":\"fake\",\"sections\":{}}";
    char path[4096];
    unsigned shift;
    FILE* file;
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    const trtmc_api_header* table = NULL;
    const trtmc_streaming_text_continuation_api_v1* factory;
    const trtmc_text_stream_api_v1* stream_api;
    const trtmc_text_continuation_api_v1* text_api;
    trtmc_text_stream* stream = NULL;
    trtmc_result *event = NULL, *final = NULL, *output = NULL;
    trtmc_error* error = NULL;
    trtmc_text_stream_event_view_v1 view = {0};
    char prompt[] = "copy", suffix[] = "!";
    trtmc_text_continuation_request_v1 input = {0};
    trtmc_config_entry_v1 entry = {{"suffix", 6}, {TRTMC_CONFIG_STRING, {.string = {suffix, 1}}}};
    trtmc_config_view_v1 config = {&entry, 1};
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    if (snprintf(path, sizeof(path), "%s/c-stream.bundle", argv[1]) >= (int)sizeof(path))
        return 2;
    file = fopen(path, "wb");
    if (!file)
        return 2;
    fwrite(magic, 1, sizeof(magic), file);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)(sizeof(header) - 1) >> shift) & 255), file);
    fwrite(header, 1, sizeof(header) - 1, file);
    if (fclose(file))
        return 2;
    options.struct_size = sizeof(options);
    options.runtime_root = string(argv[1]);
    check(core->model_load(string(path), &options, &model, &error) == TRTMC_OK,
          "load stream model");
    clear(&error);
    if (!model)
        return 2;
    check(core->model_get_task_api(model, string(TRTMC_TASK_STREAMING_TEXT_CONTINUATION), 1, 0,
                                   &table, &error) == TRTMC_OK,
          "stream is a separately selected typed Task");
    clear(&error);
    factory = (const trtmc_streaming_text_continuation_api_v1*)table;
    if (!factory)
        return 2;
    stream_api = factory->stream_api;
    check(core->model_get_task_api(model, string(TRTMC_TASK_TEXT_CONTINUATION), 1, 0, &table,
                                   &error) == TRTMC_OK,
          "sync Task stays independently available");
    clear(&error);
    text_api = (const trtmc_text_continuation_api_v1*)table;
    input.prefix.kind = TRTMC_TEXT_UTF8;
    input.prefix.as.text = string(prompt);
    check(factory->start(model, &input, &config, &stream, &error) == TRTMC_OK,
          "start with borrowed input/config");
    clear(&error);
    if (!stream || !text_api)
        return 2;
    prompt[0] = 'X';
    suffix[0] = '?';
    check(text_api->run(model, &input, NULL, &output, &error) == TRTMC_BUSY && output == NULL,
          "active stream excludes concurrent sync inference");
    clear(&error);
    check(stream_api->next(stream, -2, &event, &error) == TRTMC_INVALID_ARGUMENT && !event,
          "invalid timeout does not destroy a valid stream");
    clear(&error);
    check(stream_api->next(stream, 0, &event, &error) == TRTMC_AGAIN && !event && !error,
          "poll timeout has no fabricated event/error");
    core->model_release(model);
    model = NULL;
    check(stream_api->next(stream, -1, &event, &error) == TRTMC_OK,
          "stream retains model after C owner release");
    clear(&error);
    check(stream_api->event_view(event, &view, &error) == TRTMC_OK &&
              view.kind == TRTMC_STREAM_DELTA && equals(view.text_delta, "copy!"),
          "family owns copies of input and config");
    clear(&error);
    check(text_api->result_view(event, &view.final_result, &error) == TRTMC_INVALID_ARGUMENT,
          "ordinary text-result view rejects a stream-event owner");
    clear(&error);
    check(stream_api->next(stream, -1, &final, &error) == TRTMC_OK, "terminal event");
    clear(&error);
    check(stream_api->event_view(final, &view, &error) == TRTMC_OK &&
              view.kind == TRTMC_STREAM_COMPLETE && equals(view.final_result.text, "copy!"),
          "complete event has final typed result");
    clear(&error);
    check(stream_api->next(stream, -1, &output, &error) == TRTMC_END && !output && !error,
          "END follows exactly one terminal event");
    check(stream_api->cancel(stream, &error) == TRTMC_OK, "cancel is harmless after completion");
    clear(&error);
    stream_api->release(stream);
    check(stream_api->event_view(event, &view, &error) == TRTMC_OK &&
              equals(view.text_delta, "copy!"),
          "owned event survives stream and model release");
    clear(&error);
    core->result_release(event);
    core->result_release(final);
    stream_api->release(NULL);
    image_stream(path, &options);
    conversation_stream(path, &options);
    remove(path);
    fprintf(stderr, failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}

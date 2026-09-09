/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/text.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(offsetof(trtmc_conditional_text_generation_api_v1, header) == 0,
               "real first header");
_Static_assert(offsetof(trtmc_batch_text_continuation_api_v1, header) == 0, "real first header");

static int failures;
static const trtmc_core_api_v1* core;

static trtmc_string_view string_view(const char* value) {
    trtmc_string_view view = {value, strlen(value)};
    return view;
}

static void check(int condition, const char* label) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

static void status_is(trtmc_status actual, trtmc_status expected, trtmc_error* error,
                      const char* label) {
    check(actual == expected, label);
    core->error_release(error);
}

static int write_bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    const int length =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"text_fixture\",\"task\":\"%s\","
                 "\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 mode);
    if (length < 0 || (size_t)length >= sizeof(header))
        return 0;
    FILE* file = fopen(path, "wb");
    if (file == NULL)
        return 0;
    fwrite(magic, 1, sizeof(magic), file);
    for (unsigned shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255U), file);
    fwrite(header, 1, (size_t)length, file);
    fwrite("PLAN", 1, 4, file);
    const int good = !ferror(file);
    return fclose(file) == 0 && good;
}

static const trtmc_api_header* task_table(trtmc_model* model, const char* task) {
    const trtmc_api_header* header = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = core->model_get_task_api(model, string_view(task), 1, 0, &header, &error);
    status_is(status, TRTMC_OK, error, "C consumer discovers typed task table");
    return header;
}

static void test_tables(trtmc_model* model, trtmc_model* restricted) {
    const trtmc_conditional_text_generation_api_v1* conditional =
        (const trtmc_conditional_text_generation_api_v1*)task_table(
            model, TRTMC_TASK_CONDITIONAL_TEXT_GENERATION);
    const trtmc_text_translation_api_v1* translation =
        (const trtmc_text_translation_api_v1*)task_table(model, TRTMC_TASK_TEXT_TRANSLATION);
    const trtmc_batch_text_continuation_api_v1* batch =
        (const trtmc_batch_text_continuation_api_v1*)task_table(model,
                                                                TRTMC_TASK_BATCH_TEXT_CONTINUATION);
    if (conditional == NULL || translation == NULL || batch == NULL)
        return;

    trtmc_conditional_text_generation_request_v1 source = {0};
    source.source.kind = TRTMC_TEXT_UTF8;
    source.source.as.text = string_view("source");
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = conditional->run(restricted, &source, NULL, &result, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "table from full model cannot bypass restricted model support");
    check(result == NULL, "unsupported call clears result output");

    status = conditional->run(model, &source, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "C conditional call accepts absent config");
    uint64_t count = 100;
    status = batch->result_count(result, &count, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "single text result is not a batch result");
    check(count == 0, "failed result count clears output");
    core->result_release(result);

    source.source.kind = 99;
    status = conditional->run(model, &source, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "unknown text representation rejected before family call");
    check(result == NULL, "malformed input clears result");

    trtmc_text_translation_request_v1 request = {0};
    request.source_text = string_view("source");
    status = translation->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "C caller can omit a family-defaulted target language");
    trtmc_text_result_view_v1 translated = {0};
    status = translation->result_view(result, &translated, &error);
    status_is(status, TRTMC_OK, error, "defaulted translation result readable");
    const char expected_translation[] = "translation:fixed-src->en:source!";
    check(translated.text.size == sizeof(expected_translation) - 1 &&
              memcmp(translated.text.data, expected_translation,
                     sizeof(expected_translation) - 1) == 0,
          "family alone selects the default target language");
    core->result_release(result);
    request.has_target_language = 1;
    status = translation->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "explicit empty target rejected");
    request.has_target_language = 2;
    status = translation->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "invalid target-language presence rejected");
    request.has_target_language = 1;
    request.target_language = string_view("en");
    request.has_source_language = 2;
    status = translation->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "invalid optional-language presence rejected");

    trtmc_batch_text_continuation_item_v1 item = {0};
    item.input.prefix.kind = TRTMC_TEXT_UTF8;
    item.input.prefix.as.text = string_view("one");
    trtmc_batch_text_continuation_request_v1 batch_request = {&item, UINT64_MAX};
    status = batch->run_batch(model, &batch_request, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "batch count overflow rejected without reading items");
    batch_request.count = 1;
    status = batch->run_batch(model, &batch_request, &result, &error);
    status_is(status, TRTMC_OK, error, "C native batch call succeeds");
    status = batch->result_count(result, &count, &error);
    status_is(status, TRTMC_OK, error, "batch result count available");
    check(count == 1, "batch count equals input count");
    trtmc_text_result_view_v1 view = {0};
    status = batch->result_item_view(result, 1, &view, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "out-of-range batch view rejected");
    check(view.text.data == NULL && view.segment_count == 0, "failed item view is empty");
    status = batch->result_item_view(result, 0, &view, &error);
    status_is(status, TRTMC_OK, error, "C batch item view succeeds");
    check(view.text.size == 10 && memcmp(view.text.data, "batch:one!", 10) == 0,
          "C native batch result is human-readable");
    core->result_release(result);
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || core == NULL)
        return 2;
    char full_path[4096], restricted_path[4096];
    int full_length = snprintf(full_path, sizeof(full_path), "%s/text-c-full.bundle", argv[1]);
    int restricted_length =
        snprintf(restricted_path, sizeof(restricted_path), "%s/text-c-restricted.bundle", argv[1]);
    if (full_length < 0 || (size_t)full_length >= sizeof(full_path) || restricted_length < 0 ||
        (size_t)restricted_length >= sizeof(restricted_path))
        return 2;
    if (!write_bundle(full_path, "text_all") || !write_bundle(restricted_path, "translation_only"))
        return 2;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = string_view(argv[1]);
    trtmc_model *model = NULL, *restricted = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = core->model_load(string_view(full_path), &options, &model, &error);
    status_is(status, TRTMC_OK, error, "C full fixture loads");
    status = core->model_load(string_view(restricted_path), &options, &restricted, &error);
    status_is(status, TRTMC_OK, error, "C restricted fixture loads");
    if (model && restricted)
        test_tables(model, restricted);
    core->model_release(model);
    core->model_release(restricted);
    remove(full_path);
    remove(restricted_path);
    fprintf(stderr, "%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures;
}

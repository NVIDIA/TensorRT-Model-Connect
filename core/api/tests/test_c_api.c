/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(offsetof(trtmc_core_api_v1, header) == 0, "core header is the first member");
_Static_assert(offsetof(trtmc_text_continuation_api_v1, header) == 0,
               "task header is the first member");

static int failures;
static const trtmc_core_api_v1* core;

static void check(int condition, const char* name) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

static trtmc_string_view text(const char* value) {
    trtmc_string_view view = {value, value ? (uint64_t)strlen(value) : 0};
    return view;
}

static int equals(trtmc_string_view value, const char* expected) {
    const size_t length = strlen(expected);
    return value.size == length && (length == 0 || memcmp(value.data, expected, length) == 0);
}

static void clear_error(trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}

static void expect_error(trtmc_status actual, trtmc_status expected, trtmc_error** error,
                         const char* name) {
    check(actual == expected, name);
    check(*error != NULL, "failure returns owned error detail");
    if (*error) {
        check(core->error_code(*error) == actual, "error code agrees with status");
        check(core->error_message(*error).size != 0, "error message is nonempty");
    }
    clear_error(error);
}

static int write_bundle(const char* path, const char* mode) {
    static const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[1024];
    const int length =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"api_fixture\",\"task\":\"%s\",\"backend\":\"fake\","
                 "\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 mode);
    FILE* output;
    unsigned shift;
    if (length < 0 || (size_t)length >= sizeof(header))
        return 0;
    output = fopen(path, "wb");
    if (!output)
        return 0;
    if (fwrite(magic, 1, sizeof(magic), output) != sizeof(magic)) {
        fclose(output);
        return 0;
    }
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), output);
    if (fwrite(header, 1, (size_t)length, output) != (size_t)length ||
        fwrite("PLAN", 1, 4, output) != 4) {
        fclose(output);
        return 0;
    }
    return fclose(output) == 0;
}

static void test_versions(void) {
    const trtmc_core_api_v1* rejected = (const trtmc_core_api_v1*)(uintptr_t)1;
    check(trtmc_get_api(1, 0, NULL) == TRTMC_INVALID_ARGUMENT, "bootstrap rejects null output");
    check(trtmc_get_api(2, 0, &rejected) == TRTMC_VERSION_MISMATCH && rejected == NULL,
          "unsupported major clears table output");
    rejected = core;
    check(trtmc_get_api(1, 1, &rejected) == TRTMC_VERSION_MISMATCH && rejected == NULL,
          "unsupported minor clears table output");
    check(core->header.major == 1 && core->header.minor == 0 &&
              core->header.byte_size >= sizeof(*core),
          "bootstrap returns exact version and complete table");
    check(core->runtime_version().size != 0, "runtime version is available without a model");
    core->model_release(NULL);
    core->result_release(NULL);
    core->error_release(NULL);
}

static void test_metadata(trtmc_model* model, trtmc_model* disabled,
                          const trtmc_text_continuation_api_v1** task) {
    trtmc_error* error = NULL;
    trtmc_model_info_v1 info = {0};
    trtmc_task_info_v1 task_info = {0};
    trtmc_config_field_v1 field = {0};
    const trtmc_api_header* table = NULL;
    uint64_t count = 0;
    const trtmc_string_view task_id = text(TRTMC_TASK_TEXT_CONTINUATION);
    check(core->model_info(model, &info, &error) == TRTMC_OK && !error,
          "model metadata is available");
    check(equals(info.family, "api_fixture") && equals(info.backend, "fake") &&
              equals(info.bundle_task, "text_generation"),
          "real bundle metadata reaches the C caller");
    check(core->model_task_count(model, &count, &error) == TRTMC_OK && count == 1,
          "enabled bundle advertises one semantic task");
    check(core->model_task_info(model, 0, &task_info, &error) == TRTMC_OK &&
              equals(task_info.id, TRTMC_TASK_TEXT_CONTINUATION) && task_info.major == 1 &&
              task_info.minor == 0,
          "task discovery reports its semantic ID and version");
    check(core->model_task_count(disabled, &count, &error) == TRTMC_OK && count == 0,
          "same C++ class in a different bundle mode advertises no task");
    expect_error(core->model_task_info(model, 1, &task_info, &error), TRTMC_INVALID_ARGUMENT,
                 &error, "task metadata index is bounds checked");
    check(core->model_get_task_api(model, task_id, 1, 0, &table, &error) == TRTMC_OK && table,
          "loaded model returns its implemented task table");
    *task = (const trtmc_text_continuation_api_v1*)table;
    check(table && table->byte_size >= sizeof(**task), "task table is complete");
    expect_error(core->model_get_task_api(disabled, task_id, 1, 0, &table, &error),
                 TRTMC_UNSUPPORTED, &error, "disabled model refuses task lookup");
    check(table == NULL, "failed task lookup clears output");
    expect_error(core->model_get_task_api(model, task_id, 2, 0, &table, &error),
                 TRTMC_VERSION_MISMATCH, &error, "task major negotiates independently");
    expect_error(core->model_get_task_api(model, text("missing_task"), 1, 0, &table, &error),
                 TRTMC_UNSUPPORTED, &error, "unknown task has no fallback");

    check(core->config_field_count(model, task_id, 1, 0, &count, &error) == TRTMC_OK && count == 8,
          "family-owned configuration schema is discoverable");
    check(core->config_field_info(model, task_id, 1, 0, 0, &field, &error) == TRTMC_OK &&
              equals(field.name, "max_new_tokens") && field.kind == TRTMC_CONFIG_I64 &&
              field.has_fixed_default && field.default_value.as.i64 == 4,
          "integer config default belongs to the family");
    check(core->config_field_info(model, task_id, 1, 0, 2, &field, &error) == TRTMC_OK &&
              field.kind == TRTMC_CONFIG_BOOL && field.has_fixed_default &&
              field.default_value.as.boolean == 1,
          "boolean default retains its type");
    check(core->config_field_info(model, task_id, 1, 0, 7, &field, &error) == TRTMC_OK &&
              !field.has_fixed_default,
          "context-dependent default is distinct from zero");
    expect_error(core->config_field_info(model, task_id, 1, 0, 8, &field, &error),
                 TRTMC_INVALID_ARGUMENT, &error, "config schema index is bounds checked");
    expect_error(core->config_field_count(disabled, task_id, 1, 0, &count, &error),
                 TRTMC_UNSUPPORTED, &error, "unavailable task exposes no configuration");
}

static void test_calls(trtmc_model* model, trtmc_model* disabled,
                       const trtmc_text_continuation_api_v1* task, trtmc_result** retained) {
    trtmc_text_continuation_request_v1 request = {0};
    trtmc_text_result_view_v1 view = {0};
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_config_entry_v1 entries[7] = {0};
    trtmc_config_view_v1 config = {entries, 7};
    const int64_t biases[] = {INT64_C(9007199254740993)};
    const double schedule[] = {0.25};
    const trtmc_string_view labels[] = {{"x", 1}, {NULL, 0}};
    const int32_t tokens[] = {101, 102};

    request.prefix.kind = TRTMC_TEXT_UTF8;
    request.prefix.as.text = text("Hello");
    check(task->run(model, &request, NULL, &result, &error) == TRTMC_OK && result && !error,
          "real family DSO executes through the C task table");
    check(task->result_view(result, &view, &error) == TRTMC_OK && equals(view.text, "Hello!|eos") &&
              view.decode_ms == 4.0 && view.prefill_ms == 0.75 && view.setup_ms == 32.0,
          "omitted config uses family defaults and forwards load options");
    check(view.segment_count == 1 && view.segments[0].start_seconds == 0.125 &&
              view.segments[0].end_seconds == 0.375 && view.segments[0].text.size == 8 &&
              memcmp(view.segments[0].text.data, "seg\0tail", 8) == 0 &&
              view.segments[0].token_ids.size == 2 && view.segments[0].token_ids.data[1] == 42,
          "owned result exposes complete timestamped segments including embedded NUL");
    core->result_release(result);
    result = NULL;

    entries[0].name = text("max_new_tokens");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 0;
    entries[1].name = text("temperature");
    entries[1].value.kind = TRTMC_CONFIG_F64;
    entries[1].value.as.f64 = 0.5;
    entries[2].name = text("emit_eos");
    entries[2].value.kind = TRTMC_CONFIG_BOOL;
    entries[2].value.as.boolean = 0;
    entries[3].name = text("suffix");
    entries[3].value.kind = TRTMC_CONFIG_STRING;
    entries[3].value.as.string = text("");
    entries[4].name = text("token_biases");
    entries[4].value.kind = TRTMC_CONFIG_I64_LIST;
    entries[4].value.as.i64_list.data = biases;
    entries[4].value.as.i64_list.size = 1;
    entries[5].name = text("schedule");
    entries[5].value.kind = TRTMC_CONFIG_F64_LIST;
    entries[5].value.as.f64_list.data = schedule;
    entries[5].value.as.f64_list.size = 1;
    entries[6].name = text("labels");
    entries[6].value.kind = TRTMC_CONFIG_STRING_LIST;
    entries[6].value.as.string_list.data = labels;
    entries[6].value.as.string_list.size = 2;
    check(task->run(model, &request, &config, &result, &error) == TRTMC_OK && result,
          "all seven config value types traverse the C ABI");
    check(task->result_view(result, &view, &error) == TRTMC_OK &&
              equals(view.text, "Hello|x||9007199254740993") && view.decode_ms == 0 &&
              view.prefill_ms == 0.5 && view.setup_ms == 32.25 && view.token_ids.size == 2 &&
              view.token_ids.data[0] == 11,
          "explicit false zero empty string lists and exact int64 values reach the family");
    *retained = result;
    result = NULL;

    request.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
    request.prefix.as.token_ids.data = tokens;
    request.prefix.as.token_ids.size = 2;
    check(task->run(model, &request, NULL, &result, &error) == TRTMC_OK && result,
          "token prefix is a distinct typed input");
    check(task->result_view(result, &view, &error) == TRTMC_OK && view.token_ids.size == 2 &&
              view.token_ids.data[0] == 101 && view.token_ids.data[1] == 102,
          "token input reaches the selected family unchanged");
    core->result_release(result);
    result = (trtmc_result*)(uintptr_t)1;
    expect_error(task->run(disabled, &request, NULL, &result, &error), TRTMC_UNSUPPORTED, &error,
                 "model A table cannot invoke a task disabled on model B");
    check(result == NULL, "failed run initializes its result output");

    config.count = 1;
    entries[0].name = text("unknown_option");
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "unknown config is rejected by the family");
    entries[0].name = text("max_new_tokens");
    entries[0].value.as.i64 = -1;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "invalid range is rejected by the family");
    entries[0].value.kind = TRTMC_CONFIG_F64;
    entries[0].value.as.f64 = 4.0;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "wrong numeric type is not silently converted");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 4;
    entries[0].name = text("context_limit");
    entries[0].value.as.i64 = 1;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "family applies an explicitly overridden context limit");
    entries[0].name = text("max_new_tokens");
    entries[0].value.as.i64 = 4;
    entries[1] = entries[0];
    config.count = 2;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "duplicate names survive transport and are rejected by the family");
    config.count = 1;
    entries[0].value.kind = TRTMC_CONFIG_BOOL;
    entries[0].value.as.boolean = 2;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_ARGUMENT,
                 &error, "noncanonical C boolean is rejected");
    entries[0].name.data = NULL;
    entries[0].name.size = 1;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_ARGUMENT,
                 &error, "invalid borrowed string view is rejected");
    expect_error(task->run(model, NULL, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
                 "null request is rejected");
    expect_error(task->run(model, &request, NULL, NULL, &error), TRTMC_INVALID_ARGUMENT, &error,
                 "null result output is rejected");
    expect_error(task->result_view(NULL, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
                 "null result view owner is rejected");
    config.entries = NULL;
    expect_error(task->run(model, &request, &config, &result, &error), TRTMC_INVALID_ARGUMENT,
                 &error, "nonempty config list requires entries");
    request.prefix.kind = 99;
    expect_error(task->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
                 "unknown input discriminator is rejected");
    request.prefix.kind = TRTMC_TEXT_UTF8;
    request.prefix.as.text = text("12345678901234567890123456789012345");
    expect_error(task->run(model, &request, NULL, &result, &error), TRTMC_INVALID_CONFIG, &error,
                 "computed family context default actually constrains input");
    result = (trtmc_result*)(uintptr_t)1;
    check(task->run(model, &request, NULL, &result, NULL) == TRTMC_INVALID_ARGUMENT && !result,
          "missing error output fails without leaving a partial result");
}

int main(int argc, char** argv) {
    char enabled_path[4096];
    char disabled_path[4096];
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    trtmc_model* disabled = NULL;
    trtmc_error* error = NULL;
    trtmc_result* retained = NULL;
    trtmc_text_result_view_v1 view = {0};
    const trtmc_text_continuation_api_v1* task = NULL;
    if (argc != 2 || strlen(argv[1]) + 32 >= sizeof(enabled_path))
        return 2;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        return 1;
    test_versions();
    snprintf(enabled_path, sizeof(enabled_path), "%s/enabled.bundle", argv[1]);
    snprintf(disabled_path, sizeof(disabled_path), "%s/disabled.bundle", argv[1]);
    if (!write_bundle(enabled_path, "text_generation") || !write_bundle(disabled_path, "disabled"))
        return 1;
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[1]);
    options.kv_cache_size_bytes = 32;
    options.struct_size = 1;
    model = (trtmc_model*)(uintptr_t)1;
    expect_error(core->model_load(text(enabled_path), &options, &model, &error),
                 TRTMC_INVALID_ARGUMENT, &error, "truncated load options are rejected");
    check(model == NULL, "failed model load clears its output");
    options.struct_size = sizeof(options);
    check(core->model_load(text(enabled_path), &options, &model, &error) == TRTMC_OK && model,
          "C consumer loads a real bundle and family DSO");
    if (!model) {
        if (error) {
            const trtmc_string_view message = core->error_message(error);
            fprintf(stderr, "%.*s\n", (int)message.size, message.data);
        }
        clear_error(&error);
        return 1;
    }
    check(core->model_load(text(disabled_path), &options, &disabled, &error) == TRTMC_OK &&
              disabled,
          "same family DSO loads another mode independently");
    if (!disabled) {
        clear_error(&error);
        core->model_release(model);
        return 1;
    }
    test_metadata(model, disabled, &task);
    if (task)
        test_calls(model, disabled, task, &retained);
    core->model_release(model);
    core->model_release(disabled);
    if (retained && task) {
        check(task->result_view(retained, &view, &error) == TRTMC_OK &&
                  equals(view.text, "Hello|x||9007199254740993") && view.segment_count == 1 &&
                  view.segments[0].token_ids.size == 2 && view.segments[0].token_ids.data[1] == 42,
              "result and nested views remain valid after model release");
        core->result_release(retained);
    }
    clear_error(&error);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trtmc_core_api_v1* api;
static int failures;

static trtmc_string_view str(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}
static int same(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) &&
           (value.size == 0 || memcmp(value.data, expected, (size_t)value.size) == 0);
}
static void check(int condition, const char* label) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static void status(trtmc_status value, trtmc_status expected, trtmc_error** error,
                   const char* label) {
    check(value == expected, label);
    if (value == TRTMC_OK)
        check(*error == NULL, "success clears error output");
    else
        check(*error && api->error_code(*error) == value, "owned error agrees with failure status");
    api->error_release(*error);
    *error = NULL;
}
static void path(char* output, size_t size, const char* root, const char* name) {
    const int length = snprintf(output, size, "%s/%s", root, name);
    if (length < 0 || (size_t)length >= size)
        exit(2);
}
static void write_file(const char* filename, const char* value, size_t length) {
    FILE* file = fopen(filename, "wb");
    if (!file || fwrite(value, 1, length, file) != length || fclose(file) != 0)
        exit(2);
}
static void write_bundle(const char* filename, const char* family, const char* mode,
                         const char* backend) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[1024];
    const int length =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"%s\",\"task\":\"%s\",\"backend\":\"%s\","
                 "\"sections\":{\"payload\":{\"offset\":0,\"length\":3}}}",
                 family, mode, backend);
    FILE* output;
    unsigned shift;
    if (length < 0 || (size_t)length >= sizeof(header))
        exit(2);
    output = fopen(filename, "wb");
    if (!output)
        exit(2);
    fwrite(magic, 1, sizeof(magic), output);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), output);
    fwrite(header, 1, (size_t)length, output);
    fwrite("A\0B", 1, 3, output);
    if (fclose(output) != 0)
        exit(2);
}

static void offline_bundle(const char* root) {
    char filename[4096];
    trtmc_bundle* bundle = NULL;
    trtmc_bundle_info_v1 info = {0};
    trtmc_result* bytes = NULL;
    trtmc_result* failed = NULL;
    trtmc_bytes_view view = {0};
    trtmc_error* error = NULL;
    path(filename, sizeof(filename), root, "c-control-offline.bundle");
    write_bundle(filename, "not_installed", "inspect_only", "not_installed");
    status(api->bundle_open(str(filename), &bundle, &error), TRTMC_OK, &error,
           "offline bundle opens without family or backend installed");
    if (!bundle)
        return;
    status(api->bundle_info(bundle, &info, &error), TRTMC_OK, &error, "bundle info is readable");
    check(info.format == 1 && same(info.family, "not_installed") && info.section_count == 1 &&
              same(info.sections[0].name, "payload") && info.sections[0].offset == 0 &&
              info.sections[0].length == 3,
          "bundle metadata is preserved");
    status(api->bundle_read_section(bundle, str("payload"), &bytes, &error), TRTMC_OK, &error,
           "named bytes use the existing BundleReader");
    check(api->bundle_read_section(bundle, str("absent"), &failed, &error) != TRTMC_OK && !failed,
          "missing section fails without partial output");
    api->error_release(error);
    error = NULL;
    api->bundle_release(bundle);
    api->bundle_release(NULL);
    status(api->bytes_result_view(bytes, &view, &error), TRTMC_OK, &error,
           "section result outlives bundle handle");
    check(view.size == 3 && memcmp(view.data, "A\0B", 3) == 0, "binary bytes retain embedded NUL");
    api->result_release(bytes);
    status(api->bundle_info(NULL, &info, &error), TRTMC_INVALID_ARGUMENT, &error,
           "null bundle is rejected");
    check(info.section_count == 0 && info.sections == NULL, "failed metadata output is empty");
    status(api->bytes_result_view(NULL, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
           "null byte owner is rejected");
    write_file(filename, "bad", 3);
    bundle = (trtmc_bundle*)(uintptr_t)1;
    check(api->bundle_open(str(filename), &bundle, &error) != TRTMC_OK && !bundle,
          "malformed bundle is rejected by the existing parser");
    api->error_release(error);
}

static void preload_byok(const char* root) {
    char module[4096];
    char extension[4096];
    char bad_root[4096];
    trtmc_byok_options_v1 options = {0};
    trtmc_error* first = NULL;
    trtmc_error* error = NULL;
    void* handle;
    int* calls;
    path(module, sizeof(module), root, "c-control-module.bin");
    path(extension, sizeof(extension), root, "libtrtmc_byok_tvm_ffi.so");
    write_file(module, "fixture", 7);
    options.struct_size = sizeof(options);
    options.runtime_root = str(root);
    options.library = str(module);
    options.function = str("run");
    options.kernel_name = str("fixture.copy");
    status(api->byok_load(&options, &error), TRTMC_OK, &error,
           "BYOK preload is callable before any model is loaded");
    options.function = str("first_error");
    check(api->byok_load(&options, &first) == TRTMC_INTERNAL_ERROR && first,
          "extension errors cross the stable C boundary");
    options.function = str("second_error");
    check(api->byok_load(&options, &error) == TRTMC_INTERNAL_ERROR && error,
          "subsequent extension error is returned independently");
    check(first && same(api->error_message(first), "unknown fixture function: first_error"),
          "error copy survives extension thread-local message reuse");
    api->error_release(first);
    api->error_release(error);
    error = NULL;
    handle = dlopen(extension, RTLD_NOW | RTLD_LOCAL);
    check(handle != NULL, "preloaded extension remains available");
    if (handle) {
        calls = (int*)dlsym(handle, "trtmc_test_byok_calls");
        check(calls && *calls == 3, "extension was retained rather than unloaded between calls");
        dlclose(handle);
    }
    path(bad_root, sizeof(bad_root), root, "missing-extension");
    options.runtime_root = str(bad_root);
    options.function = str("run");
    status(api->byok_load(&options, &error), TRTMC_INTERNAL_ERROR, &error,
           "missing optional BYOK extension is a clear failure");
    path(bad_root, sizeof(bad_root), root, "missing-symbol");
    options.runtime_root = str(bad_root);
    status(api->byok_load(&options, &error), TRTMC_INTERNAL_ERROR, &error,
           "wrong extension without the required entrypoint fails");
    options.runtime_root = str(root);
    options.library.data = "bad\0path";
    options.library.size = 8;
    status(api->byok_load(&options, &error), TRTMC_INVALID_ARGUMENT, &error,
           "embedded NUL is rejected before passing C string arguments");
    status(api->byok_load(NULL, &error), TRTMC_INVALID_ARGUMENT, &error, "null BYOK options fail");
}

static void adapters(const char* root) {
    char enabled_path[4096], disabled_path[4096], weights_path[4096];
    trtmc_load_options_v1 options = {0};
    trtmc_model* enabled = NULL;
    trtmc_model* other = NULL;
    trtmc_model* disabled = NULL;
    const trtmc_lora_api_v1* lora = NULL;
    const trtmc_lora_api_v1* rejected = NULL;
    const trtmc_api_header* task_header = NULL;
    const trtmc_text_continuation_api_v1* text_api;
    trtmc_error* error = NULL;
    trtmc_result* snapshot = NULL;
    trtmc_result* output = NULL;
    trtmc_strings_view names = {0};
    trtmc_text_result_view_v1 result = {0};
    trtmc_bytes_view wrong_view = {0};
    trtmc_text_continuation_request_v1 request = {0};
    trtmc_config_entry_v1 adapter = {0};
    trtmc_config_view_v1 config = {&adapter, 1};
    path(enabled_path, sizeof(enabled_path), root, "c-control-enabled.bundle");
    path(disabled_path, sizeof(disabled_path), root, "c-control-disabled.bundle");
    path(weights_path, sizeof(weights_path), root, "c-control-weights.txt");
    write_bundle(enabled_path, "control_fixture", "enabled", "fake");
    write_bundle(disabled_path, "control_fixture", "disabled", "fake");
    write_file(weights_path, "weights-one", 11);
    options.struct_size = sizeof(options);
    options.runtime_root = str(root);
    status(api->model_load(str(enabled_path), &options, &enabled, &error), TRTMC_OK, &error,
           "load LoRA model");
    status(api->model_load(str(enabled_path), &options, &other, &error), TRTMC_OK, &error,
           "load independent model");
    status(api->model_load(str(disabled_path), &options, &disabled, &error), TRTMC_OK, &error,
           "load model without LoRA");
    if (!enabled || !other || !disabled)
        goto cleanup;
    status(api->model_get_lora_api(enabled, 1, 0, &lora, &error), TRTMC_OK, &error,
           "family explicitly exposes loaded LoRA support");
    status(api->model_get_lora_api(disabled, 1, 0, &rejected, &error), TRTMC_UNSUPPORTED, &error,
           "C++ inheritance alone does not expose LoRA");
    check(!rejected, "failed LoRA lookup clears output");
    status(api->model_get_lora_api(enabled, 2, 0, &rejected, &error), TRTMC_VERSION_MISMATCH,
           &error, "LoRA table negotiates its own version");
    if (!lora)
        goto cleanup;
    status(lora->load(enabled, str("adapter"), str(weights_path), &error), TRTMC_OK, &error,
           "load named adapter");
    status(lora->load(disabled, str("adapter"), str(weights_path), &error), TRTMC_UNSUPPORTED,
           &error, "model A table cannot invoke LoRA on model B");
    status(lora->list(enabled, &snapshot, &error), TRTMC_OK, &error, "list returns owned snapshot");
    status(lora->list_view(snapshot, &names, &error), TRTMC_OK, &error, "adapter IDs are readable");
    check(names.size == 1 && same(names.data[0], "adapter"), "adapter snapshot has loaded ID");
    status(api->bytes_result_view(snapshot, &wrong_view, &error), TRTMC_INVALID_ARGUMENT, &error,
           "result type check rejects adapter snapshot as bytes");
    status(lora->list(other, &output, &error), TRTMC_OK, &error,
           "second model has its own adapter inventory");
    status(lora->list_view(output, &names, &error), TRTMC_OK, &error, "read independent inventory");
    check(names.size == 0, "adapter IDs are model-local");
    api->result_release(output);
    output = NULL;
    status(api->model_get_task_api(enabled, str(TRTMC_TASK_TEXT_CONTINUATION), 1, 0, &task_header,
                                   &error),
           TRTMC_OK, &error, "text task remains separately discoverable");
    text_api = (const trtmc_text_continuation_api_v1*)task_header;
    if (text_api) {
        request.prefix.kind = TRTMC_TEXT_UTF8;
        request.prefix.as.text = str("Hello");
        adapter.name = str("lora_adapter_id");
        adapter.value.kind = TRTMC_CONFIG_STRING;
        adapter.value.as.string = str("adapter");
        status(text_api->run(enabled, &request, &config, &output, &error), TRTMC_OK, &error,
               "request selects adapter through family-owned Config");
        status(text_api->result_view(output, &result, &error), TRTMC_OK, &error,
               "read adapted result");
        check(same(result.text, "Hello|weights-one"), "selected weights affect the result");
        api->result_release(output);
        output = NULL;
        write_file(weights_path, "weights-two", 11);
        status(lora->load(enabled, str("adapter"), str(weights_path), &error), TRTMC_OK, &error,
               "family retains existing replace-by-ID behavior");
        status(text_api->run(enabled, &request, &config, &output, &error), TRTMC_OK, &error,
               "run replaced adapter");
        status(text_api->result_view(output, &result, &error), TRTMC_OK, &error,
               "read replaced result");
        check(same(result.text, "Hello|weights-two"), "replacement takes effect");
        api->result_release(output);
        output = NULL;
        status(text_api->run(other, &request, &config, &output, &error), TRTMC_INVALID_CONFIG,
               &error, "selection cannot use another model's adapter");
        status(text_api->run(enabled, &request, NULL, &output, &error), TRTMC_OK, &error,
               "omitting adapter does not leave prior request selection active");
        status(text_api->result_view(output, &result, &error), TRTMC_OK, &error,
               "read base result");
        check(same(result.text, "Hello"), "base generation is restored per request");
        api->result_release(output);
        output = NULL;
    }
    status(lora->unload(enabled, str("adapter"), &error), TRTMC_OK, &error, "unload adapter");
    status(lora->unload(enabled, str("adapter"), &error), TRTMC_INVALID_ARGUMENT, &error,
           "unloading unknown adapter fails clearly");
cleanup:
    api->model_release(enabled);
    api->model_release(other);
    api->model_release(disabled);
    if (snapshot && lora) {
        status(lora->list_view(snapshot, &names, &error), TRTMC_OK, &error,
               "adapter list snapshot outlives unload and model destruction");
        check(names.size == 1 && same(names.data[0], "adapter"), "owned snapshot is unchanged");
    }
    api->result_release(snapshot);
    api->result_release(output);
}

int main(int argc, char** argv) {
    if (argc != 2 || trtmc_get_api(1, 0, &api) != TRTMC_OK || !api)
        return 2;
    offline_bundle(argv[1]);
    preload_byok(argv[1]);
    adapters(argv[1]);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

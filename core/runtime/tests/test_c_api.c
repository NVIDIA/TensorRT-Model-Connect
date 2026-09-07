/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/c_api.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char* name) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

static int write_bundle(const char* path, const char* task) {
    static const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[256];
    const int header_size = snprintf(header, sizeof(header),
                                     "{\"format\":1,\"family\":\"fake_c\",\"task\":\"%s\","
                                     "\"backend\":\"fake\",\"sections\":{}}",
                                     task);
    if (header_size <= 0 || (size_t)header_size >= sizeof(header))
        return 0;

    FILE* output = fopen(path, "wb");
    if (output == NULL)
        return 0;
    if (fwrite(magic, sizeof(magic), 1, output) != 1) {
        fclose(output);
        return 0;
    }
    const uint64_t length = (uint64_t)header_size;
    for (int shift = 0; shift < 64; shift += 8)
        fputc((int)((length >> shift) & 0xffU), output);
    const int ok = fwrite(header, (size_t)header_size, 1, output) == 1;
    return fclose(output) == 0 && ok;
}

struct thread_error {
    trtmc_status_t status;
    char message[128];
};

static void* set_thread_error(void* opaque) {
    struct thread_error* result = (struct thread_error*)opaque;
    const char* name = NULL;
    result->status = trtmc_task_name(NULL, &name);
    snprintf(result->message, sizeof(result->message), "%s", trtmc_last_error());
    return NULL;
}

static void test_error_contract(const char* bundle, const char* runtime_root) {
    trtmc_task_t* task = (trtmc_task_t*)(uintptr_t)1;
    check(trtmc_task_load(NULL, runtime_root, &task) == TRTMC_STATUS_INVALID_ARGUMENT,
          "null bundle path is rejected");
    check(task == NULL, "failed load clears output handle");
    check(strstr(trtmc_last_error(), "bundle_path") != NULL, "load failure sets last error");

    char missing_root[1024];
    snprintf(missing_root, sizeof(missing_root), "%s/missing", runtime_root);
    task = (trtmc_task_t*)(uintptr_t)1;
    check(trtmc_task_load(bundle, missing_root, &task) == TRTMC_STATUS_RUNTIME_ERROR,
          "explicit missing runtime root fails closed");
    check(task == NULL, "runtime load failure clears output handle");

    char main_error[1024];
    snprintf(main_error, sizeof(main_error), "%s", trtmc_last_error());
    struct thread_error thread_result = {TRTMC_STATUS_OK, {0}};
    pthread_t thread;
    check(pthread_create(&thread, NULL, set_thread_error, &thread_result) == 0,
          "error test thread starts");
    check(pthread_join(thread, NULL) == 0, "error test thread joins");
    check(thread_result.status == TRTMC_STATUS_INVALID_ARGUMENT,
          "thread receives its own error status");
    check(strstr(thread_result.message, "task") != NULL, "thread receives its own error text");
    check(strcmp(trtmc_last_error(), main_error) == 0, "last error is thread local");
}

static void test_text(const char* bundle, const char* runtime_root) {
    trtmc_task_t* task = NULL;
    check(trtmc_task_load(bundle, runtime_root, &task) == TRTMC_STATUS_OK,
          "text task loads through the runtime");
    check(task != NULL, "text task handle is returned");
    check(trtmc_last_error()[0] == '\0', "successful load clears last error");

    const char* task_name = NULL;
    check(trtmc_task_name(task, &task_name) == TRTMC_STATUS_OK, "text task name is available");
    check(task_name != NULL && strcmp(task_name, "text_generation") == 0,
          "text task name matches the bundle");

    trtmc_text_result_t result = {0};
    check(trtmc_text_generate(task, "hello", 5, &result) == TRTMC_STATUS_OK,
          "text generation succeeds");
    check(result.text != NULL && strcmp(result.text, "hello:5") == 0,
          "text result is copied to C ownership");
    check(result.text_length == 7, "text length excludes the null terminator");
    check(result.token_count == 3 && result.token_ids != NULL && result.token_ids[2] == 5,
          "token IDs are copied to C ownership");
    trtmc_text_result_free(&result);
    check(result.text == NULL && result.token_ids == NULL && result.text_length == 0 &&
              result.token_count == 0,
          "text result free is idempotent and clears ownership");
    trtmc_text_result_free(&result);

    const char* prompts[] = {"wrong task"};
    const uint32_t seeds[] = {1};
    trtmc_image_result_t image = {0};
    check(trtmc_image_generate_batch(task, prompts, seeds, 1, 4, 7.5F, &image) ==
              TRTMC_STATUS_WRONG_TASK,
          "typed image call rejects a text task");
    check(strstr(trtmc_last_error(), "image_generation_batch") != NULL,
          "wrong-task failure is actionable");

    trtmc_task_destroy(task);
}

static void test_image_batch(const char* bundle, const char* runtime_root) {
    trtmc_task_t* task = NULL;
    check(trtmc_task_load(bundle, runtime_root, &task) == TRTMC_STATUS_OK,
          "image batch task loads through the runtime");

    const char* prompts[] = {"cat", "puppy"};
    const uint32_t seeds[] = {42, 7};
    trtmc_image_result_t results[2] = {{0}};
    check(trtmc_image_generate_batch(task, prompts, seeds, 2, 4, 7.5F, results) == TRTMC_STATUS_OK,
          "image batch generation succeeds");
    check(results[0].height == 1, "first image height is copied");
    check(results[0].width == 4, "first image width is copied");
    check(results[0].channels == 1, "first image channel count is copied");
    check(results[0].num_frames == 1, "first image frame count is copied");
    check(results[0].pixel_count == 4, "first image pixel count is copied");
    check(results[1].pixel_count == 4, "second image pixel count is copied");
    check(results[0].pixels[0] == 3.0F, "first prompt length is preserved");
    check(results[0].pixels[1] == 42.0F, "first seed is preserved");
    check(results[0].pixels[2] == 4.0F, "step count is preserved");
    check(results[0].pixels[3] == 7.5F, "guidance scale is preserved");
    check(results[1].pixels[0] == 5.0F, "second prompt length is preserved");
    check(results[1].pixels[1] == 7.0F, "second seed is preserved");

    trtmc_image_result_free(&results[0]);
    trtmc_image_result_free(&results[1]);
    check(results[0].pixels == NULL, "first image pixels are cleared");
    check(results[0].pixel_count == 0, "first image size is cleared");
    check(results[1].pixels == NULL, "second image pixels are cleared");
    check(results[1].pixel_count == 0, "second image size is cleared");
    trtmc_task_destroy(task);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: test_c_api RUNTIME_ROOT\n");
        return 2;
    }
    const char* runtime_root = argv[1];
    char text_bundle[1024];
    char image_bundle[1024];
    snprintf(text_bundle, sizeof(text_bundle), "%s/fake-c-text.bundle", runtime_root);
    snprintf(image_bundle, sizeof(image_bundle), "%s/fake-c-image.bundle", runtime_root);
    check(write_bundle(text_bundle, "text_generation"), "text bundle is written");
    check(write_bundle(image_bundle, "image_generation_batch"), "image bundle is written");

    check(trtmc_version() != NULL && trtmc_version()[0] != '\0', "version is non-empty");
    test_error_contract(text_bundle, runtime_root);
    test_text(text_bundle, runtime_root);
    test_image_batch(image_bundle, runtime_root);
    trtmc_task_destroy(NULL);
    trtmc_text_result_free(NULL);
    trtmc_image_result_free(NULL);

    remove(text_bundle);
    remove(image_bundle);
    fprintf(stderr, "%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures;
}

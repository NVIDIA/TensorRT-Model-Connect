/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned flux3 end-to-end test.
 * Runs text_to_image, releases the model, then writes the image as raw float32
 * HWC RGB values in [0, 1] and prints its shape as JSON.
 *
 * Usage: test_flux3_sdk_c BUNDLE RUNTIME_ROOT PROMPT OUTPUT_F32
 *            [--initial-latents-raw FILE] [--seed N] [--num-steps N]
 * FILE holds raw float32 latents laid out [128, height / 16, width / 16]. */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/image.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "latents and pixels use float32");

#define FLUX3_MAX_CONFIG 2

typedef struct {
    const char* bundle;
    const char* runtime_root;
    const char* prompt;
    const char* output;
    const char* latents_path;
    trtmc_config_entry_v1 entries[FLUX3_MAX_CONFIG];
    uint64_t entry_count;
} flux3_arguments;

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int same(trtmc_string_view view, const char* expected) {
    const size_t size = strlen(expected);
    return view.size == size && view.data != NULL && memcmp(view.data, expected, size) == 0;
}

static int add_integer(flux3_arguments* args, const char* name, const char* value) {
    trtmc_config_entry_v1* entry;
    char* end = NULL;
    long long parsed;
    if (args->entry_count == FLUX3_MAX_CONFIG)
        return 0;
    errno = 0;
    parsed = strtoll(value, &end, 10);
    if (errno || end == value || *end)
        return 0;
    entry = &args->entries[args->entry_count++];
    entry->name = text(name);
    entry->value.kind = TRTMC_CONFIG_I64;
    entry->value.as.i64 = (int64_t)parsed;
    return 1;
}

static int parse_arguments(int argc, char** argv, flux3_arguments* args) {
    int index;
    if (argc < 5 || (argc - 5) % 2 != 0)
        return 0;
    args->bundle = argv[1];
    args->runtime_root = argv[2];
    args->prompt = argv[3];
    args->output = argv[4];
    for (index = 5; index < argc; index += 2) {
        const char* option = argv[index];
        const char* value = argv[index + 1];
        int accepted = 0;
        if (strcmp(option, "--initial-latents-raw") == 0) {
            args->latents_path = value;
            accepted = 1;
        } else if (strcmp(option, "--seed") == 0) {
            accepted = add_integer(args, "seed", value);
        } else if (strcmp(option, "--num-steps") == 0) {
            accepted = add_integer(args, "num_steps", value);
        }
        if (!accepted)
            return 0;
    }
    return 1;
}

/* Returns a malloc'ed copy of a nonempty raw float32 file, or NULL. */
static float* read_float32_file(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    float* values = NULL;
    long bytes = -1;
    *count = 0;
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) == 0)
        bytes = ftell(file);
    if (bytes > 0 && bytes % (long)sizeof(float) == 0 && fseek(file, 0, SEEK_SET) == 0)
        values = (float*)malloc((size_t)bytes);
    if (values && fread(values, 1, (size_t)bytes, file) == (size_t)bytes) {
        *count = (uint64_t)bytes / sizeof(float);
    } else {
        free(values);
        values = NULL;
    }
    fclose(file);
    return values;
}

static int check_config_fields(const trtmc_core_api_v1* core, const trtmc_model* model,
                               trtmc_error** error) {
    uint64_t count = 0, index;
    if (core->config_field_count(model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &count, error) !=
        TRTMC_OK)
        return 0;
    if (count != 2) {
        fprintf(stderr, "flux3 must expose exactly the seed and num_steps Config fields\n");
        return 0;
    }
    for (index = 0; index < count; ++index) {
        trtmc_config_field_v1 field;
        if (core->config_field_info(model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, index, &field,
                                    error) != TRTMC_OK)
            return 0;
        if (!(same(field.name, "seed") || same(field.name, "num_steps")) ||
            field.kind != TRTMC_CONFIG_I64 || !field.has_fixed_default) {
            fprintf(stderr, "unexpected flux3 Config field %.*s\n", (int)field.name.size,
                    field.name.data ? field.name.data : "");
            return 0;
        }
    }
    return 1;
}

static int check_image(const trtmc_image_result_view_v1* view) {
    uint64_t index;
    if (view->channels != 3 || !view->height || !view->width || !view->pixels ||
        view->pixel_count != (uint64_t)view->height * view->width * 3) {
        fprintf(stderr, "text_to_image returned an invalid image shape\n");
        return 0;
    }
    for (index = 0; index < view->pixel_count; ++index) {
        const float value = view->pixels[index];
        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            fprintf(stderr, "text_to_image returned a pixel outside [0, 1]\n");
            return 0;
        }
    }
    return 1;
}

static int write_float32_file(const char* path, const float* values, uint64_t count) {
    FILE* file = fopen(path, "wb");
    int written;
    if (!file)
        return 0;
    written = fwrite(values, sizeof(float), (size_t)count, file) == (size_t)count;
    return fclose(file) == 0 && written;
}

/* Handles owned by one run; close_session releases whatever is still held. */
typedef struct {
    const trtmc_core_api_v1* core;
    trtmc_model* model;
    trtmc_result* result;
    trtmc_error* error;
    float* latents;
    uint64_t latent_count;
} flux3_session;

/* Loads the model and returns its text_to_image table once its Config fields check out. */
static const trtmc_text_to_image_api_v1* open_task(flux3_session* session,
                                                   const flux3_arguments* args) {
    const trtmc_core_api_v1* core = session->core;
    const trtmc_api_header* header = NULL;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = text(args->runtime_root);
    if (core->model_load(text(args->bundle), &options, &session->model, &session->error) !=
            TRTMC_OK ||
        core->model_get_task_api(session->model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &header,
                                 &session->error) != TRTMC_OK)
        return NULL;
    if (!header || header->byte_size < sizeof(trtmc_text_to_image_api_v1)) {
        fprintf(stderr, "incomplete text_to_image C table\n");
        return NULL;
    }
    if (!check_config_fields(core, session->model, &session->error))
        return NULL;
    return (const trtmc_text_to_image_api_v1*)header;
}

/* Runs the request, releases the model and inputs, then emits the owned result. */
static int generate(flux3_session* session, const trtmc_text_to_image_api_v1* task,
                    const flux3_arguments* args) {
    trtmc_text_to_image_request_v1 request = {0};
    trtmc_config_view_v1 config = {0};
    trtmc_image_result_view_v1 view = {0};
    request.prompt = text(args->prompt);
    request.initial_latents.data = session->latents;
    request.initial_latents.size = session->latent_count;
    config.entries = args->entries;
    config.count = args->entry_count;
    if (task->run(session->model, &request, &config, &session->result, &session->error) != TRTMC_OK)
        return 0;
    /* The result must stay readable after its inputs and model are gone. */
    free(session->latents);
    session->latents = NULL;
    session->core->model_release(session->model);
    session->model = NULL;
    if (task->result_view(session->result, &view, &session->error) != TRTMC_OK ||
        !check_image(&view))
        return 0;
    if (!write_float32_file(args->output, view.pixels, view.pixel_count)) {
        fprintf(stderr, "failed to write %s\n", args->output);
        return 0;
    }
    printf("{\"task\":\"text_to_image\",\"height\":%" PRIu32 ",\"width\":%" PRIu32
           ",\"channels\":%" PRIu32 "}\n",
           view.height, view.width, view.channels);
    return fflush(stdout) == 0 && !ferror(stdout);
}

static void close_session(flux3_session* session) {
    const trtmc_core_api_v1* core = session->core;
    if (core && session->error) {
        const trtmc_string_view message = core->error_message(session->error);
        fprintf(stderr, "C API error %d: ", (int)core->error_code(session->error));
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(session->error);
    }
    if (core) {
        core->result_release(session->result);
        core->model_release(session->model);
    }
    free(session->latents);
}

int main(int argc, char** argv) {
    flux3_arguments args = {0};
    flux3_session session = {0};
    const trtmc_text_to_image_api_v1* task;
    int status = 1;

    if (!parse_arguments(argc, argv, &args)) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT PROMPT OUTPUT_F32 [--initial-latents-raw FILE]"
                " [--seed N] [--num-steps N]\n",
                argv[0]);
        return 2;
    }
    if (args.latents_path) {
        session.latents = read_float32_file(args.latents_path, &session.latent_count);
        if (!session.latents) {
            fprintf(stderr, "%s must hold a nonzero whole number of float32 values\n",
                    args.latents_path);
            return 1;
        }
    }
    if (trtmc_get_api(1, 0, &session.core) == TRTMC_OK && session.core) {
        task = open_task(&session, &args);
        status = task && generate(&session, task, &args) ? 0 : 1;
    } else {
        fprintf(stderr, "unable to obtain the v1 C API\n");
    }
    close_session(&session);
    return status;
}

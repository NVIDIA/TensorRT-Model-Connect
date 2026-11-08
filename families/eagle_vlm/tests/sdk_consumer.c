/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT text_to_embedding TEXT
 *   BUNDLE RUNTIME_ROOT text_query_documents_to_relevance QUERY DOCUMENT...
 */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/features.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}
static void json_string(trtmc_string_view value) {
    uint64_t i;
    putchar('"');
    for (i = 0; i < value.size; ++i) {
        const unsigned char byte = (unsigned char)value.data[i];
        if (byte == '"' || byte == '\\') {
            putchar('\\');
            putchar(byte);
        } else if (byte < 32) {
            printf("\\u%04x", (unsigned)byte);
        } else {
            putchar(byte);
        }
    }
    putchar('"');
}
static int finite_values(const float* values, uint64_t count) {
    uint64_t i;
    for (i = 0; i < count; ++i)
        if (!isfinite(values[i]))
            return 0;
    return 1;
}
static void print_values(const float* values, uint64_t count) {
    uint64_t i;
    putchar('[');
    for (i = 0; i < count; ++i) {
        if (i)
            putchar(',');
        printf("%.*g", FLT_DECIMAL_DIG, (double)values[i]);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_string_view* documents = NULL;
    uint64_t field_count = 0;
    int status = 1;
    if (argc < 5) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT TASK INPUT...\n", argv[0]);
        return 2;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->config_field_count(model, text(argv[3]), 1, 0, &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "Eagle-VLM must expose no runtime Config fields\n");
        goto cleanup;
    }

    if (strcmp(argv[3], TRTMC_TASK_TEXT_TO_EMBEDDING) == 0) {
        const trtmc_text_to_embedding_api_v1* task = NULL;
        trtmc_text_to_embedding_request_v1 request = {0};
        trtmc_semantic_embedding_view_v1 view = {0};
        if (argc != 5) {
            fprintf(stderr, "text_to_embedding takes exactly one TEXT\n");
            goto cleanup;
        }
        if (core->model_get_task_api(model, text(argv[3]), 1, 0, &header, &error) != TRTMC_OK)
            goto cleanup;
        if (!header || header->byte_size < sizeof(*task)) {
            fprintf(stderr, "incomplete embedding C table\n");
            goto cleanup;
        }
        task = (const trtmc_text_to_embedding_api_v1*)header;
        request.text = text(argv[4]);
        request.role = TRTMC_EMBEDDING_DEFAULT;
        if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
            goto cleanup;
        core->model_release(model);
        model = NULL;
        if (task->result_view(result, &view, &error) != TRTMC_OK)
            goto cleanup;
        if (!view.count || !finite_values(view.values, view.count)) {
            fprintf(stderr, "Eagle-VLM must provide a nonempty, finite embedding\n");
            goto cleanup;
        }
        printf("{\"task\":\"text_to_embedding\",\"pooling\":");
        json_string(view.pooling);
        printf(",\"normalization\":");
        json_string(view.normalization);
        printf(",\"dim\":%" PRIu64 ",\"values\":", view.count);
        print_values(view.values, view.count);
        puts("}");
    } else if (strcmp(argv[3], TRTMC_TASK_TEXT_QUERY_DOCUMENTS_TO_RELEVANCE) == 0) {
        const trtmc_text_query_documents_to_relevance_api_v1* task = NULL;
        trtmc_text_query_documents_to_relevance_request_v1 request = {0};
        trtmc_document_relevance_view_v1 view = {0};
        const uint64_t count = (uint64_t)(argc - 5);
        uint64_t i;
        if (count == 0) {
            fprintf(stderr, "text_query_documents_to_relevance needs QUERY and a DOCUMENT\n");
            goto cleanup;
        }
        documents = (trtmc_string_view*)malloc((size_t)count * sizeof(*documents));
        if (!documents)
            goto cleanup;
        for (i = 0; i < count; ++i)
            documents[i] = text(argv[5 + i]);
        if (core->model_get_task_api(model, text(argv[3]), 1, 0, &header, &error) != TRTMC_OK)
            goto cleanup;
        if (!header || header->byte_size < sizeof(*task)) {
            fprintf(stderr, "incomplete list-relevance C table\n");
            goto cleanup;
        }
        task = (const trtmc_text_query_documents_to_relevance_api_v1*)header;
        request.query = text(argv[4]);
        request.documents.data = documents;
        request.documents.size = count;
        if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
            goto cleanup;
        core->model_release(model);
        model = NULL;
        if (task->result_view(result, &view, &error) != TRTMC_OK)
            goto cleanup;
        if (view.count != count || !finite_values(view.scores, view.count)) {
            fprintf(stderr, "Eagle-VLM must score every document with a finite value\n");
            goto cleanup;
        }
        if (view.kind != TRTMC_SCORE_LOGIT) {
            fprintf(stderr, "Eagle-VLM relevance scores must be logits\n");
            goto cleanup;
        }
        printf("{\"task\":\"text_query_documents_to_relevance\",\"score_kind\":\"logit\","
               "\"scores\":");
        print_values(view.scores, view.count);
        puts("}");
    } else {
        fprintf(stderr, "unsupported TASK: %s\n", argv[3]);
        goto cleanup;
    }
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "C API error %d: ", (int)core->error_code(error));
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    free(documents);
    if (core) {
        core->result_release(result);
        core->model_release(model);
    }
    return status;
}

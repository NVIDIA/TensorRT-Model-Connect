/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test: both Tasks of one bundle. */
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
static int print_floats(const float* values, uint64_t count) {
    uint64_t i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            fprintf(stderr, "MobileBERT returned a nonfinite score\n");
            return 0;
        }
        if (i)
            putchar(',');
        printf("%.*g", FLT_DECIMAL_DIG, (double)values[i]);
    }
    return 1;
}
static int no_config(const trtmc_core_api_v1* core, const trtmc_model* model, const char* task,
                     trtmc_error** error) {
    uint64_t count = 0;
    if (core->config_field_count(model, text(task), 1, 0, &count, error) != TRTMC_OK)
        return 0;
    if (count != 0) {
        fprintf(stderr, "MobileBERT must expose no runtime Config fields\n");
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_masked_text_to_token_scores_api_v1* fill = NULL;
    const trtmc_text_pair_to_pretraining_relation_scores_api_v1* pair = NULL;
    trtmc_model* model = NULL;
    trtmc_result* masked = NULL;
    trtmc_result* relation = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_masked_text_to_token_scores_request_v1 fill_request = {0};
    trtmc_text_pair_to_pretraining_relation_scores_request_v1 pair_request = {0};
    trtmc_vocabulary_scores_view_v1 scores = {0};
    trtmc_label_scores_view_v1 labels = {0};
    uint64_t i;
    int status = 1;
    if (argc != 6) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT MASKED_TEXT FIRST SECOND\n", argv[0]);
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

    if (core->model_get_task_api(model, text(TRTMC_TASK_MASKED_TEXT_TO_TOKEN_SCORES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*fill)) {
        fprintf(stderr, "incomplete masked-token-scores C table\n");
        goto cleanup;
    }
    fill = (const trtmc_masked_text_to_token_scores_api_v1*)header;
    if (core->model_get_task_api(model, text(TRTMC_TASK_TEXT_PAIR_TO_PRETRAINING_RELATION_SCORES),
                                 1, 0, &header, &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*pair)) {
        fprintf(stderr, "incomplete pretraining-relation C table\n");
        goto cleanup;
    }
    pair = (const trtmc_text_pair_to_pretraining_relation_scores_api_v1*)header;
    if (!no_config(core, model, TRTMC_TASK_MASKED_TEXT_TO_TOKEN_SCORES, &error) ||
        !no_config(core, model, TRTMC_TASK_TEXT_PAIR_TO_PRETRAINING_RELATION_SCORES, &error))
        goto cleanup;

    fill_request.text.kind = TRTMC_TEXT_UTF8;
    fill_request.text.as.text = text(argv[3]);
    if (fill->run(model, &fill_request, NULL, &masked, &error) != TRTMC_OK)
        goto cleanup;
    pair_request.first = text(argv[4]);
    pair_request.second = text(argv[5]);
    if (pair->run(model, &pair_request, NULL, &relation, &error) != TRTMC_OK)
        goto cleanup;
    /* Both results outlive the model they came from. */
    core->model_release(model);
    model = NULL;
    if (fill->result_view(masked, &scores, &error) != TRTMC_OK ||
        pair->result_view(relation, &labels, &error) != TRTMC_OK)
        goto cleanup;
    if (!scores.logits.rows || scores.logits.rows != scores.position_count ||
        scores.logits.count != scores.logits.rows * scores.logits.columns) {
        fprintf(stderr, "MobileBERT masked scores have an inconsistent shape\n");
        goto cleanup;
    }
    if (labels.count != 2 || labels.labels.size != 2 || labels.kind != TRTMC_SCORE_LOGIT) {
        fprintf(stderr, "MobileBERT next-sentence scores must be two labelled logits\n");
        goto cleanup;
    }

    printf("{\"masked\":{\"shape\":[%" PRIu64 ",%" PRIu64 "],\"vocabulary_id\":",
           scores.logits.rows, scores.logits.columns);
    json_string(scores.vocabulary_id);
    printf(",\"positions\":[");
    for (i = 0; i < scores.position_count; ++i) {
        printf("%s{\"token_id\":%" PRId64 ",\"token_index\":%" PRIu64 ",\"input_index\":%d}",
               i ? "," : "", scores.positions[i].token_id, scores.positions[i].token_index,
               (int)scores.positions[i].input_index);
    }
    printf("],\"logits\":[");
    if (!print_floats(scores.logits.data, scores.logits.count))
        goto cleanup;
    printf("]},\"relation\":{\"scores\":[");
    if (!print_floats(labels.scores, labels.count))
        goto cleanup;
    printf("],\"labels\":[");
    for (i = 0; i < labels.labels.size; ++i) {
        if (i)
            putchar(',');
        json_string(labels.labels.data[i]);
    }
    printf("],\"vocabulary_id\":");
    json_string(labels.vocabulary_id);
    puts("}}");
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "C API error %d: ", (int)core->error_code(error));
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (core) {
        core->result_release(masked);
        core->result_release(relation);
        core->model_release(model);
    }
    return status;
}

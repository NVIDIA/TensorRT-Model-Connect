/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/features.h"
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
    trtmc_string_view result = {value, strlen(value)};
    return result;
}
static int has_prefix(trtmc_string_view value, const char* prefix) {
    const size_t count = strlen(prefix);
    return value.size >= count && memcmp(value.data, prefix, count) == 0;
}
static int write_bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    const int n = snprintf(
        header, sizeof(header),
        "{\"format\":1,\"family\":\"features_fixture\",\"task\":\"%s\",\"backend\":\"fake\","
        "\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
        mode);
    if (n < 0 || (size_t)n >= sizeof(header))
        return 0;
    FILE* output = fopen(path, "wb");
    if (!output)
        return 0;
    int ok = fwrite(magic, 1, sizeof(magic), output) == sizeof(magic);
    for (unsigned shift = 0; shift < 64; shift += 8)
        ok = (fputc((int)(((uint64_t)n >> shift) & 255U), output) != EOF) && ok;
    ok = (fwrite(header, 1, (size_t)n, output) == (size_t)n) && ok;
    ok = (fwrite("PLAN", 1, 4, output) == 4) && ok;
    return fclose(output) == 0 && ok;
}
static void consume_error(const trtmc_core_api_v1* core, trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}
static void padding_contracts(const trtmc_core_api_v1* core, trtmc_model* padded,
                              trtmc_model* valid_only, const char* root,
                              const trtmc_load_options_v1* options) {
    const trtmc_api_header* header = NULL;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_token_features_view_v1 view = {0};
    if (core->model_get_task_api(padded, str(TRTMC_TASK_TEXT_PAIR_TO_TOKEN_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK) {
        check(0, "C pair table for padding");
        consume_error(core, &error);
        return;
    }
    const trtmc_text_pair_to_token_features_api_v1* pair =
        (const trtmc_text_pair_to_token_features_api_v1*)header;
    trtmc_text_pair_to_token_features_request_v1 request = {0};
    request.kind = TRTMC_FEATURE_TEXT_PAIR;
    request.as.text = (trtmc_feature_text_pair_v1){str("a"), str("bc")};
    check(pair->run(padded, &request, NULL, &result, &error) == TRTMC_OK &&
              pair->result_view(result, &view, &error) == TRTMC_OK && view.features.rows == 4 &&
              view.tokens[0].input_index == 0 && view.tokens[1].input_index == 1 &&
              view.tokens[1].byte_end == 2 &&
              view.tokens[2].input_index == TRTMC_FEATURE_INPUT_SPECIAL &&
              view.tokens[2].token_id == 101 && view.features.data[4] == 8.25F &&
              view.tokens[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              !view.tokens[3].has_byte_offsets && view.features.data[6] == 37.5F &&
              view.features.data[7] == -91.25F,
          "C pair retains text, special and padding roles with raw values");
    core->result_release(result);
    result = NULL;
    const int32_t ids[] = {101, 17, 29, 99}, segments[] = {0, 0, 1, 0};
    const uint8_t mask[] = {1, 1, 1, 0};
    request.kind = TRTMC_FEATURE_TOKENIZED_PAIR;
    request.as.tokens = (trtmc_feature_tokenized_pair_v1){{ids, 4}, {segments, 4}, {mask, 4}};
    check(pair->run(padded, &request, NULL, &result, &error) == TRTMC_OK &&
              pair->result_view(result, &view, &error) == TRTMC_OK && view.features.rows == 4 &&
              view.tokens[3].token_id == 99 && view.tokens[3].token_index == 3 &&
              view.tokens[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              view.features.data[6] == 37.5F && view.features.data[7] == 99,
          "C full-profile tokenized pair preserves original masked padding");
    core->result_release(result);
    result = NULL;
    check(pair->run(valid_only, &request, NULL, &result, &error) == TRTMC_OK &&
              pair->result_view(result, &view, &error) == TRTMC_OK && view.features.rows == 3 &&
              view.tokens[2].token_id == 29,
          "C old valid-only pair path still excludes padding");
    core->result_release(result);
    result = NULL;
    if (core->model_get_task_api(padded, str(TRTMC_TASK_BATCH_TEXT_TO_TOKEN_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK) {
        check(0, "C batch table for padding");
        consume_error(core, &error);
        return;
    }
    const trtmc_batch_text_to_token_features_api_v1* batch =
        (const trtmc_batch_text_to_token_features_api_v1*)header;
    trtmc_batch_text_to_token_features_item_v1 items[2] = {0};
    items[0].input.text.kind = TRTMC_TEXT_UTF8;
    items[0].input.text.as.text = str("ab");
    items[1].input.text.kind = TRTMC_TEXT_TOKEN_IDS;
    items[1].input.text.as.token_ids = (trtmc_i32_view){ids + 1, 3};
    const trtmc_batch_text_to_token_features_request_v1 batch_input = {items, 2};
    uint64_t count = 0;
    check(batch->run(padded, &batch_input, &result, &error) == TRTMC_OK &&
              batch->result_count(result, &count, &error) == TRTMC_OK && count == 2 &&
              batch->result_item_view(result, 1, &view, &error) == TRTMC_OK &&
              view.features.rows == 4 && view.tokens[2].token_id == 99 &&
              view.tokens[2].input_index == 0 && view.tokens[3].token_id == 99 &&
              view.tokens[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              view.tokens[3].token_index == 3 && view.features.data[9] == 37.5F &&
              view.features.data[10] == -91.25F,
          "C batch keeps all nonzero padding rows without guessing from token ID");
    core->result_release(result);
    result = NULL;
    for (int which = 0; which < 2; ++which) {
        char path[4096];
        const char* mode = which ? "padding_bad_index" : "padding_offsets";
        trtmc_model* invalid = NULL;
        if (snprintf(path, sizeof(path), "%s/features_c_%s.bundle", root, mode) >=
                (int)sizeof(path) ||
            !write_bundle(path, mode) ||
            core->model_load(str(path), options, &invalid, &error) != TRTMC_OK) {
            check(0, "C invalid padding fixture load");
            consume_error(core, &error);
            continue;
        }
        check(batch->run(invalid, &batch_input, &result, &error) == TRTMC_INTERNAL_ERROR && !result,
              "C illegal padding byte offsets or input role reject the whole result");
        consume_error(core, &error);
        core->model_release(invalid);
    }
}

static void class_identity_contracts(const trtmc_core_api_v1* core, const char* root,
                                     const trtmc_load_options_v1* options) {
    const char* modes[] = {"unnamed_classes", "blank_classes", "empty_classes", "named_classes",
                           "ordinal_classes"};
    const float pixels[] = {0.5F, 0.25F, 0.125F};
    const trtmc_image_to_class_scores_request_v1 input = {
        {pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_FLOAT32}};
    const trtmc_batch_image_to_class_scores_item_v1 items[] = {{input, {NULL, 0}},
                                                               {input, {NULL, 0}}};
    const trtmc_batch_image_to_class_scores_request_v1 batch_input = {items, 2};
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        char path[4096];
        trtmc_model* model = NULL;
        trtmc_error* error = NULL;
        const trtmc_api_header *single_header = NULL, *batch_header = NULL;
        if (snprintf(path, sizeof(path), "%s/features_c_%s.bundle", root, modes[i]) >=
                (int)sizeof(path) ||
            !write_bundle(path, modes[i])) {
            check(0, "create C classifier identity fixture");
            continue;
        }
        if (core->model_load(str(path), options, &model, &error) != TRTMC_OK ||
            core->model_get_task_api(model, str(TRTMC_TASK_IMAGE_TO_CLASS_SCORES), 1, 0,
                                     &single_header, &error) != TRTMC_OK ||
            core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_CLASS_SCORES), 1, 0,
                                     &batch_header, &error) != TRTMC_OK) {
            check(0, "load C classifier identity contracts");
            consume_error(core, &error);
            core->model_release(model);
            continue;
        }
        const trtmc_image_to_class_scores_api_v1* single =
            (const trtmc_image_to_class_scores_api_v1*)single_header;
        const trtmc_batch_image_to_class_scores_api_v1* batch =
            (const trtmc_batch_image_to_class_scores_api_v1*)batch_header;
        trtmc_result* result = (trtmc_result*)(uintptr_t)1;
        trtmc_status status = single->run(model, &input, NULL, &result, &error);
        if (i < 3) {
            check(status == TRTMC_INTERNAL_ERROR && result == NULL && error,
                  "C rejects anonymous or empty class scores and clears output");
        } else {
            trtmc_label_scores_view_v1 view = {0};
            check(status == TRTMC_OK && result &&
                      single->result_view(result, &view, &error) == TRTMC_OK && view.count == 2 &&
                      view.scores[0] == 18 && view.kind == TRTMC_SCORE_LOGIT &&
                      ((i == 3 && view.labels.size == 2 && view.vocabulary_id.size == 0) ||
                       (i == 4 && view.labels.size == 0 && view.vocabulary_id.size != 0)),
                  "C preserves inline labels or explicit ordinal vocabulary without normalization");
            core->result_release(result);
        }
        consume_error(core, &error);
        result = (trtmc_result*)(uintptr_t)1;
        status = batch->run(model, &batch_input, &result, &error);
        if (i < 3) {
            check(status == TRTMC_INTERNAL_ERROR && result == NULL && error &&
                      has_prefix(core->error_message(error), "batch item[1]"),
                  "C rejects incomplete second classifier result without a partial batch");
        } else {
            trtmc_label_scores_view_v1 view = {0};
            check(status == TRTMC_OK && result &&
                      batch->result_item_view(result, 1, &view, &error) == TRTMC_OK &&
                      view.count == 4 && view.scores[0] == 21,
                  "C valid class identity survives native batch result packing");
            core->result_release(result);
        }
        consume_error(core, &error);
        core->model_release(model);
    }
}

static void unknown_embedding_space_contract(const trtmc_core_api_v1* core, const char* root,
                                             const trtmc_load_options_v1* options) {
    char path[4096];
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    if (snprintf(path, sizeof(path), "%s/features_c_unknown_space.bundle", root) >=
            (int)sizeof(path) ||
        !write_bundle(path, "unknown_embedding_space") ||
        core->model_load(str(path), options, &model, &error) != TRTMC_OK ||
        core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_EMBEDDING), 1, 0, &header, &error) !=
            TRTMC_OK) {
        check(0, "load C unknown-space embedding fixture");
        consume_error(core, &error);
        core->model_release(model);
        return;
    }
    const trtmc_text_to_embedding_api_v1* task = (const trtmc_text_to_embedding_api_v1*)header;
    const trtmc_text_to_embedding_request_v1 input = {str("local"), TRTMC_EMBEDDING_DEFAULT};
    check(task->run(model, &input, NULL, &result, &error) == TRTMC_OK && result,
          "C computes an embedding without a known checkpoint-space identifier");
    core->model_release(model);
    trtmc_semantic_embedding_view_v1 view = {0};
    check(task->result_view(result, &view, &error) == TRTMC_OK && view.count == 2 &&
              view.values[0] == 4 && view.embedding_space.size == 0 && view.pooling.size == 4 &&
              view.normalization.size == 4,
          "C preserves values and explicit unknown identity after model release");
    core->result_release(result);
    consume_error(core, &error);
}

static void head_score_contracts(const trtmc_core_api_v1* core, const char* root,
                                 const trtmc_load_options_v1* options) {
    char path[4096];
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    if (snprintf(path, sizeof(path), "%s/features_c_head.bundle", root) >= (int)sizeof(path) ||
        !write_bundle(path, "all") ||
        core->model_load(str(path), options, &model, &error) != TRTMC_OK ||
        core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_HEAD_SCORES), 1, 0, &header,
                                 &error) != TRTMC_OK) {
        check(0, "load C head score table");
        consume_error(core, &error);
        core->model_release(model);
        return;
    }
    const trtmc_text_to_head_scores_api_v1* task = (const trtmc_text_to_head_scores_api_v1*)header;
    check(header->byte_size == sizeof(*task), "C head-score v1 table layout");
    trtmc_text_to_head_scores_request_v1 input = {0};
    input.text.kind = TRTMC_TEXT_TOKEN_IDS;
    const int32_t ids[] = {7, 8};
    input.text.as.token_ids = (trtmc_i32_view){ids, 2};
    trtmc_result* result = NULL;
    check(task->run(model, &input, NULL, &result, &error) == TRTMC_OK && result,
          "C head scores accept the existing explicit token input");
    core->model_release(model);
    trtmc_head_scores_view_v1 view = {0};
    check(task->result_view(result, &view, &error) == TRTMC_OK && view.count == 4 &&
              view.values[0] == -2 && view.values[1] == 2 && view.rank == 3 && view.shape[0] == 1 &&
              view.shape[1] == 2 && view.shape[2] == 2 && view.kind == TRTMC_SCORE_LOGIT &&
              has_prefix(view.pooling, "none") && has_prefix(view.normalization, "none"),
          "C head values, shape and metadata survive model release without reinterpretation");
    core->result_release(result);
    consume_error(core, &error);
    const char* invalid_modes[] = {
        "head_bad_shape", "head_zero_dim",        "head_empty_shape",          "head_overflow",
        "head_bad_kind",  "head_missing_pooling", "head_missing_normalization"};
    for (size_t i = 0; i < sizeof(invalid_modes) / sizeof(invalid_modes[0]); ++i) {
        model = NULL;
        result = (trtmc_result*)1;
        if (snprintf(path, sizeof(path), "%s/features_c_%s.bundle", root, invalid_modes[i]) >=
                (int)sizeof(path) ||
            !write_bundle(path, invalid_modes[i]) ||
            core->model_load(str(path), options, &model, &error) != TRTMC_OK) {
            check(0, "load malformed C head-score fixture");
            consume_error(core, &error);
            continue;
        }
        check(task->run(model, &input, NULL, &result, &error) == TRTMC_INTERNAL_ERROR && !result,
              "C malformed head-score results fail without partial results");
        consume_error(core, &error);
        core->model_release(model);
    }
    memset(&view, 0x7f, sizeof(view));
    check(task->result_view(NULL, &view, &error) == TRTMC_INVALID_ARGUMENT && view.values == NULL &&
              view.shape == NULL && view.rank == 0,
          "C head result-view failure clears every borrowed field");
    consume_error(core, &error);
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    char path[4096], disabled_path[4096];
    if (snprintf(path, sizeof(path), "%s/features_c_all.bundle", argv[1]) >= (int)sizeof(path) ||
        snprintf(disabled_path, sizeof(disabled_path), "%s/features_c_none.bundle", argv[1]) >=
            (int)sizeof(disabled_path))
        return 2;
    if (!write_bundle(path, "all") || !write_bundle(disabled_path, "none"))
        return 2;
    const trtmc_core_api_v1* core = NULL;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    class_identity_contracts(core, argv[1], &options);
    unknown_embedding_space_contract(core, argv[1], &options);
    head_score_contracts(core, argv[1], &options);
    trtmc_model *model = NULL, *disabled = NULL;
    if (core->model_load(str(path), &options, &model, &error) != TRTMC_OK)
        return 2;
    if (core->model_load(str(disabled_path), &options, &disabled, &error) != TRTMC_OK)
        return 2;
    const trtmc_api_header* header = NULL;
    if (core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_TOKEN_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_text_to_token_features_api_v1* tokens =
        (const trtmc_text_to_token_features_api_v1*)header;
    trtmc_text_to_token_features_request_v1 request = {{0}};
    request.text.kind = TRTMC_TEXT_UTF8;
    request.text.as.text = str("abc");
    trtmc_config_entry_v1 scale = {0};
    scale.name = str("scale");
    scale.value.kind = TRTMC_CONFIG_F64;
    scale.value.as.f64 = 2.0;
    trtmc_config_view_v1 config = {&scale, 1};
    trtmc_result* result = NULL;
    check(tokens->run(model, &request, &config, &result, &error) == TRTMC_OK,
          "C typed token call succeeds");
    trtmc_token_features_view_v1 view = {0};
    check(tokens->result_view(result, &view, &error) == TRTMC_OK && view.features.rows == 1 &&
              view.features.columns == 2 && view.features.data[0] == 2 &&
              view.tokens[0].token_id == 17,
          "C reads checked feature axes and token mapping");
    {
        char padded_path[4096];
        trtmc_model* padded = NULL;
        trtmc_result* padded_result = NULL;
        trtmc_token_features_view_v1 padded_view = {0};
        if (snprintf(padded_path, sizeof(padded_path), "%s/features_c_padding.bundle", argv[1]) >=
                (int)sizeof(padded_path) ||
            !write_bundle(padded_path, "padding") ||
            core->model_load(str(padded_path), &options, &padded, &error) != TRTMC_OK)
            return 2;
        check(tokens->run(padded, &request, NULL, &padded_result, &error) == TRTMC_OK,
              "C raw padded feature result succeeds");
        padding_contracts(core, padded, model, argv[1], &options);
        core->model_release(padded);
        check(tokens->result_view(padded_result, &padded_view, &error) == TRTMC_OK &&
                  padded_view.features.rows == 2 && padded_view.features.data[2] == 37.5F &&
                  padded_view.features.data[3] == -91.25F &&
                  padded_view.tokens[1].input_index == TRTMC_FEATURE_INPUT_PADDING &&
                  padded_view.tokens[1].token_id == 99 && padded_view.tokens[1].token_index == 7 &&
                  !padded_view.tokens[1].has_byte_offsets,
              "C padding sentinel values and mapping survive model release");
        core->result_release(padded_result);
    }
    trtmc_result* rejected = NULL;
    check(tokens->run(disabled, &request, NULL, &rejected, &error) == TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "table from model A cannot bypass model B support");
    consume_error(core, &error);
    request.text.kind = TRTMC_TEXT_TOKEN_IDS;
    request.text.as.token_ids.data = NULL;
    request.text.as.token_ids.size = 2;
    check(tokens->run(model, &request, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "nonempty null token input fails before family code");
    consume_error(core, &error);
    header = NULL;
    check(core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0, &header,
                                   &error) == TRTMC_OK,
          "pooled features have an independent typed table");
    const trtmc_text_to_pooled_features_api_v1* pooled =
        (const trtmc_text_to_pooled_features_api_v1*)header;
    trtmc_pooled_features_view_v1 pooled_view = {0};
    check(pooled->result_view(result, &pooled_view, &error) == TRTMC_INVALID_ARGUMENT &&
              pooled_view.count == 0,
          "wrong-kind result is rejected by its typed view function");
    consume_error(core, &error);
    if (core->model_get_task_api(model, str(TRTMC_TASK_TEXT_PAIR_TO_TOKEN_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_text_pair_to_token_features_api_v1* pair =
        (const trtmc_text_pair_to_token_features_api_v1*)header;
    trtmc_text_pair_to_token_features_request_v1 pair_input = {0};
    pair_input.kind = TRTMC_FEATURE_TEXT_PAIR;
    pair_input.as.text.first = str("a");
    pair_input.as.text.second = str("bc");
    trtmc_result* pair_result = NULL;
    trtmc_token_features_view_v1 pair_view = {0};
    check(pair->run(model, &pair_input, NULL, &pair_result, &error) == TRTMC_OK,
          "C UTF-8 pair call uses both inputs");
    check(pair->result_view(pair_result, &pair_view, &error) == TRTMC_OK &&
              pair_view.features.rows == 2 && pair_view.tokens[1].input_index == 1 &&
              pair_view.tokens[1].byte_end == 2,
          "C UTF-8 pair preserves roles and offsets");
    core->result_release(pair_result);
    pair_result = NULL;
    const int32_t pair_ids[] = {21, 22, 23};
    const int32_t pair_segments[] = {0, 1, 1};
    const uint8_t pair_mask[] = {1, 0, 1};
    pair_input.kind = TRTMC_FEATURE_TOKENIZED_PAIR;
    pair_input.as.tokens.token_ids = (trtmc_i32_view){pair_ids, 3};
    pair_input.as.tokens.segment_ids = (trtmc_i32_view){pair_segments, 3};
    pair_input.as.tokens.attention_mask = (trtmc_feature_u8_view){pair_mask, 3};
    check(pair->run(model, &pair_input, NULL, &pair_result, &error) == TRTMC_OK,
          "C tokenized-pair request reaches the same typed task");
    check(pair->result_view(pair_result, &pair_view, &error) == TRTMC_OK &&
              pair_view.features.rows == 2 && pair_view.tokens[1].token_id == 23 &&
              pair_view.tokens[1].token_index == 2 && pair_view.tokens[1].input_index == 1,
          "C tokenized-pair mask and segment IDs are not discarded");
    core->result_release(pair_result);
    pair_result = NULL;
    pair_input.as.tokens.attention_mask.size = 1;
    check(pair->run(model, &pair_input, NULL, &pair_result, &error) == TRTMC_INVALID_ARGUMENT &&
              pair_result == NULL,
          "C pair mask length mismatch is rejected");
    consume_error(core, &error);

    if (core->model_get_task_api(model, str(TRTMC_TASK_IMAGE_TO_SPATIAL_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_image_to_spatial_features_api_v1* spatial =
        (const trtmc_image_to_spatial_features_api_v1*)header;
    const float pixel[] = {0.5F, 0.25F, 0.125F};
    trtmc_image_to_spatial_features_request_v1 spatial_input = {
        {pixel, sizeof(pixel), 1, 1, 3, TRTMC_IMAGE_FLOAT32}};
    trtmc_result* spatial_result = NULL;
    trtmc_spatial_features_view_v1 spatial_view = {0};
    check(spatial->run(model, &spatial_input, NULL, &spatial_result, &error) == TRTMC_OK,
          "C spatial feature call succeeds");
    check(spatial->result_view(spatial_result, &spatial_view, &error) == TRTMC_OK &&
              spatial_view.source_to_processed.scale_x == 2 &&
              spatial_view.source_to_processed.scale_y == 4 &&
              spatial_view.source_to_processed.offset_x == -3 &&
              spatial_view.source_to_processed.offset_y == 5,
          "C spatial result retains a nonidentity resize/crop transform");
    core->result_release(spatial_result);
    if (core->model_get_task_api(model, str(TRTMC_TASK_TEXT_QUERY_DOCUMENTS_TO_RELEVANCE), 1, 0,
                                 &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_text_query_documents_to_relevance_api_v1* ranking =
        (const trtmc_text_query_documents_to_relevance_api_v1*)header;
    const trtmc_string_view documents[] = {
        {"aaaaaaaaaaaaaaaaaaaa", 20}, {"x", 1}, {"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", 30}};
    trtmc_text_query_documents_to_relevance_request_v1 ranking_input = {str("qq"), {documents, 3}};
    trtmc_result* ranking_result = NULL;
    trtmc_document_relevance_view_v1 ranking_view = {0};
    check(ranking->run(model, &ranking_input, NULL, &ranking_result, &error) == TRTMC_OK,
          "C query/documents Task succeeds");
    check(ranking->result_view(ranking_result, &ranking_view, &error) == TRTMC_OK &&
              ranking_view.count == 3 && ranking_view.scores[0] == 1220 &&
              ranking_view.scores[1] == 1211 && ranking_view.scores[2] == 1250,
          "C relevance marker proves one shared-query family call and unsorted order");
    core->result_release(ranking_result);
    ranking_result = NULL;
    check(ranking->run(disabled, &ranking_input, NULL, &ranking_result, &error) ==
                  TRTMC_UNSUPPORTED &&
              ranking_result == NULL,
          "document table cannot bypass another model's support");
    consume_error(core, &error);
    ranking_input.documents.data = NULL;
    check(ranking->run(model, &ranking_input, NULL, &ranking_result, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              ranking_result == NULL,
          "nonempty document list requires backing storage");
    consume_error(core, &error);
    if (core->model_get_task_api(model, str(TRTMC_TASK_IMAGE_TO_TOKEN_AND_POOLED_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_image_to_token_and_pooled_features_api_v1* joint =
        (const trtmc_image_to_token_and_pooled_features_api_v1*)header;
    const trtmc_image_to_token_and_pooled_features_request_v1 joint_input = {spatial_input.image};
    trtmc_result* joint_result = NULL;
    trtmc_image_token_and_pooled_features_view_v1 joint_view = {0};
    check(joint->run(model, &joint_input, NULL, &joint_result, &error) == TRTMC_OK,
          "C joint extraction performs one family call");
    check(joint->result_view(joint_result, &joint_view, &error) == TRTMC_OK &&
              joint_view.tokens.features.data[0] == 20 && joint_view.tokens.features.data[1] == 1 &&
              joint_view.pooled.values[0] == 20 && joint_view.pooled.values[1] == 1,
          "token and pooler outputs have the same single-invocation marker");
    check(joint->result_view(result, &joint_view, &error) == TRTMC_INVALID_ARGUMENT &&
              joint_view.pooled.count == 0,
          "token-only result cannot masquerade as a joint extraction");
    consume_error(core, &error);
    check(joint->run(disabled, &joint_input, NULL, &rejected, &error) == TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "joint extraction table cannot bypass another model's declaration");
    consume_error(core, &error);
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_CLASS_SCORES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_image_to_class_scores_api_v1* batch_classes =
        (const trtmc_batch_image_to_class_scores_api_v1*)header;
    const trtmc_batch_image_to_class_scores_item_v1 batch_items[] = {
        {{spatial_input.image}, {NULL, 0}}, {{spatial_input.image}, config}};
    const trtmc_batch_image_to_class_scores_request_v1 batch_request = {batch_items, 2};
    trtmc_result* batch_result = NULL;
    check(batch_classes->run(model, &batch_request, &batch_result, &error) == TRTMC_OK,
          "C two-item class score batch succeeds");
    uint64_t batch_count = 0;
    trtmc_label_scores_view_v1 batch_view = {0};
    check(batch_classes->result_count(batch_result, &batch_count, &error) == TRTMC_OK &&
              batch_count == 2,
          "C batch retains input item count");
    check(batch_classes->result_item_view(batch_result, 1, &batch_view, &error) == TRTMC_OK &&
              batch_view.scores[0] == 42 && batch_view.scores[2] == 1,
          "C batch second item keeps its config and shared family invocation marker");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_image_to_token_features_api_v1* b0 =
        (const trtmc_batch_image_to_token_features_api_v1*)header;
    const trtmc_batch_image_to_token_features_item_v1 b0_items[] = {
        {{spatial_input.image}, {NULL, 0}}, {{spatial_input.image}, config}};
    const trtmc_batch_image_to_token_features_request_v1 b0_request = {b0_items, 2};
    trtmc_result* b0_result = NULL;
    trtmc_image_token_features_view_v1 b0_view = {0};
    check(b0->run(model, &b0_request, &b0_result, &error) == TRTMC_OK,
          "C BatchImageToTokenFeatures executes its direct batch interface");
    check(b0->result_count(b0_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b0->result_item_view(b0_result, 1, &b0_view, &error) == TRTMC_OK &&
              b0_view.features.data[0] == 44 && b0_view.token_count == 2 &&
              b0_view.tokens[1].role == TRTMC_IMAGE_TOKEN_PATCH,
          "C BatchImageToTokenFeatures retains its exact typed axes and metadata");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_SPATIAL_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_image_to_spatial_features_api_v1* b1 =
        (const trtmc_batch_image_to_spatial_features_api_v1*)header;
    const trtmc_batch_image_to_spatial_features_item_v1 b1_items[] = {
        {{spatial_input.image}, {NULL, 0}}, {{spatial_input.image}, config}};
    const trtmc_batch_image_to_spatial_features_request_v1 b1_request = {b1_items, 2};
    trtmc_result* b1_result = NULL;
    trtmc_spatial_features_view_v1 b1_view = {0};
    check(b1->run(model, &b1_request, &b1_result, &error) == TRTMC_OK,
          "C BatchImageToSpatialFeatures executes its direct batch interface");
    check(b1->result_count(b1_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b1->result_item_view(b1_result, 1, &b1_view, &error) == TRTMC_OK &&
              b1_view.map_count == 2 && b1_view.maps[0].width == 2 &&
              b1_view.maps[0].values[0] == 46 && b1_view.source_to_processed.scale_y == 4,
          "C BatchImageToSpatialFeatures retains its exact typed axes and metadata");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_POOLED_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_image_to_pooled_features_api_v1* b2 =
        (const trtmc_batch_image_to_pooled_features_api_v1*)header;
    const trtmc_batch_image_to_pooled_features_item_v1 b2_items[] = {
        {{spatial_input.image}, {NULL, 0}}, {{spatial_input.image}, config}};
    const trtmc_batch_image_to_pooled_features_request_v1 b2_request = {b2_items, 2};
    trtmc_result* b2_result = NULL;
    trtmc_pooled_features_view_v1 b2_view = {0};
    check(b2->run(model, &b2_request, &b2_result, &error) == TRTMC_OK,
          "C BatchImageToPooledFeatures executes its direct batch interface");
    check(b2->result_count(b2_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b2->result_item_view(b2_result, 1, &b2_view, &error) == TRTMC_OK && b2_view.count == 3 &&
              b2_view.values[0] == 48 && b2_view.pooling.size == 3,
          "C BatchImageToPooledFeatures retains its exact typed axes and metadata");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_TEXT_TO_EMBEDDING), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_text_to_embedding_api_v1* b3 =
        (const trtmc_batch_text_to_embedding_api_v1*)header;
    const trtmc_batch_text_to_embedding_item_v1 b3_items[] = {
        {{str("ab"), TRTMC_EMBEDDING_QUERY}, {NULL, 0}},
        {{str("abcd"), TRTMC_EMBEDDING_DOCUMENT}, config}};
    const trtmc_batch_text_to_embedding_request_v1 b3_request = {b3_items, 2};
    trtmc_result* b3_result = NULL;
    trtmc_semantic_embedding_view_v1 b3_view = {0};
    check(b3->run(model, &b3_request, &b3_result, &error) == TRTMC_OK,
          "C BatchTextToEmbedding executes its direct batch interface");
    check(b3->result_count(b3_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b3->result_item_view(b3_result, 1, &b3_view, &error) == TRTMC_OK && b3_view.count == 4 &&
              b3_view.values[0] == 50 && b3_view.values[2] == 2 &&
              b3_view.embedding_space.size != 0,
          "C BatchTextToEmbedding retains its exact typed axes and metadata");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_TEXT_TO_TOKEN_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_text_to_token_features_api_v1* b4 =
        (const trtmc_batch_text_to_token_features_api_v1*)header;
    const int32_t b4_ids[] = {77, 88, 99};
    trtmc_batch_text_to_token_features_item_v1 b4_items[2] = {0};
    b4_items[0].input.text.kind = TRTMC_TEXT_UTF8;
    b4_items[0].input.text.as.text = str("ab");
    b4_items[1].input.text.kind = TRTMC_TEXT_TOKEN_IDS;
    b4_items[1].input.text.as.token_ids = (trtmc_i32_view){b4_ids, 3};
    b4_items[1].config = config;
    const trtmc_batch_text_to_token_features_request_v1 b4_request = {b4_items, 2};
    trtmc_result* b4_result = NULL;
    trtmc_token_features_view_v1 b4_view = {0};
    check(b4->run(model, &b4_request, &b4_result, &error) == TRTMC_OK,
          "C BatchTextToTokenFeatures executes its direct batch interface");
    check(b4->result_count(b4_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b4->result_item_view(b4_result, 1, &b4_view, &error) == TRTMC_OK &&
              b4_view.features.rows == 3 && b4_view.features.data[0] == 52 &&
              b4_view.tokens[0].token_id == 77 && !b4_view.tokens[0].has_byte_offsets,
          "C BatchTextToTokenFeatures retains its exact typed axes and metadata");
    if (core->model_get_task_api(model, str(TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_AND_POOLED_FEATURES), 1,
                                 0, &header, &error) != TRTMC_OK)
        return 2;
    const trtmc_batch_image_to_token_and_pooled_features_api_v1* b5 =
        (const trtmc_batch_image_to_token_and_pooled_features_api_v1*)header;
    const trtmc_batch_image_to_token_and_pooled_features_item_v1 b5_items[] = {
        {{spatial_input.image}, {NULL, 0}}, {{spatial_input.image}, config}};
    const trtmc_batch_image_to_token_and_pooled_features_request_v1 b5_request = {b5_items, 2};
    trtmc_result* b5_result = NULL;
    trtmc_image_token_and_pooled_features_view_v1 b5_view = {0};
    check(b5->run(model, &b5_request, &b5_result, &error) == TRTMC_OK,
          "C BatchImageToTokenAndPooledFeatures executes its direct batch interface");
    check(b5->result_count(b5_result, &batch_count, &error) == TRTMC_OK && batch_count == 2,
          "C batch retains two ordered items");
    check(b5->result_item_view(b5_result, 1, &b5_view, &error) == TRTMC_OK &&
              b5_view.tokens.features.data[0] == 54 && b5_view.pooled.values[0] == 54 &&
              b5_view.tokens.features.data[2] == b5_view.pooled.values[1],
          "C BatchImageToTokenAndPooledFeatures retains its exact typed axes and metadata");

    check(b0->result_item_view(b2_result, 0, &b0_view, &error) == TRTMC_INVALID_ARGUMENT &&
              b0_view.features.count == 0,
          "pooled batch cannot masquerade as image token batch");
    consume_error(core, &error);
    check(b5->result_item_view(b5_result, 2, &b5_view, &error) == TRTMC_INVALID_ARGUMENT &&
              b5_view.pooled.count == 0,
          "batch item view rejects index past the end and clears output");
    consume_error(core, &error);
    check(batch_classes->run(disabled, &batch_request, &rejected, &error) == TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "batch table from model A cannot bypass model B support");
    consume_error(core, &error);
    trtmc_batch_image_to_class_scores_item_v1 invalid_items[2] = {batch_items[0], batch_items[1]};
    invalid_items[1].input.image.byte_size = 1;
    const trtmc_batch_image_to_class_scores_request_v1 invalid_request = {invalid_items, 2};
    check(batch_classes->run(model, &invalid_request, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "malformed second batch item returns no partial result");
    check(error && has_prefix(core->error_message(error), "batch item[1]"),
          "transport error identifies its batch item");
    consume_error(core, &error);
    invalid_items[1] = batch_items[1];
    trtmc_config_entry_v1 bad_boolean = {0};
    bad_boolean.name = str("annotate");
    bad_boolean.value.kind = TRTMC_CONFIG_BOOL;
    bad_boolean.value.as.boolean = 2;
    invalid_items[1].config = (trtmc_config_view_v1){&bad_boolean, 1};
    check(batch_classes->run(model, &invalid_request, &rejected, &error) ==
                  TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "malformed second-item C config bool is rejected");
    check(error && has_prefix(core->error_message(error), "batch item[1]"),
          "config transport preserves item error index");
    consume_error(core, &error);
    bad_boolean.value.as.boolean = 0;
    trtmc_config_entry_v1 explicit_fields[3] = {0};
    explicit_fields[0] = scale;
    explicit_fields[1] = bad_boolean;
    explicit_fields[0].value.as.f64 = 0;
    explicit_fields[2].name = str("tag");
    explicit_fields[2].value.kind = TRTMC_CONFIG_STRING;
    explicit_fields[2].value.as.string = (trtmc_string_view){NULL, 0};
    invalid_items[1].config = (trtmc_config_view_v1){explicit_fields, 3};
    trtmc_result* explicit_result = NULL;
    check(batch_classes->run(model, &invalid_request, &explicit_result, &error) == TRTMC_OK,
          "per-item explicit zero/false/empty cross the C transport");
    check(batch_classes->result_item_view(explicit_result, 1, &batch_view, &error) == TRTMC_OK &&
              batch_view.scores[0] == 0 && batch_view.scores[1] == 0.5F &&
              batch_view.scores[3] == 0 && batch_view.vocabulary_id.size == 16 &&
              memcmp(batch_view.vocabulary_id.data, "fixture..classes", 16) == 0,
          "C batch preserves false/zero/empty and reports no scalar class invocation");
    core->result_release(explicit_result);

    core->model_release(model);
    core->model_release(disabled);
    check(tokens->result_view(result, &view, &error) == TRTMC_OK && view.features.data[0] == 2,
          "C feature result owns data after every model handle is released");
    core->result_release(result);
    check(joint->result_view(joint_result, &joint_view, &error) == TRTMC_OK &&
              joint_view.pooled.values[1] == 1,
          "both outputs of a joint extraction outlive the model");
    core->result_release(joint_result);
    check(batch_classes->result_item_view(batch_result, 0, &batch_view, &error) == TRTMC_OK &&
              batch_view.scores[0] == 21,
          "C batch results remain owned after model release");
    core->result_release(batch_result);
    check(b0->result_item_view(b0_result, 1, &b0_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b0_result);
    check(b1->result_item_view(b1_result, 1, &b1_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b1_result);
    check(b2->result_item_view(b2_result, 1, &b2_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b2_result);
    check(b3->result_item_view(b3_result, 1, &b3_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b3_result);
    check(b4->result_item_view(b4_result, 1, &b4_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b4_result);
    check(b5->result_item_view(b5_result, 1, &b5_view, &error) == TRTMC_OK,
          "C typed batch snapshot survives model release");
    core->result_release(b5_result);

    consume_error(core, &error);
    return failures == 0 ? 0 : 1;
}

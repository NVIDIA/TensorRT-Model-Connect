/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/recurrent.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <string.h>
static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view string(const char* value) {
    trtmc_string_view out = {value, strlen(value)};
    return out;
}
static void check(int ok, const char* message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static void status(trtmc_status value, trtmc_status expected, trtmc_error** error,
                   const char* label) {
    check(value == expected, label);
    core->error_release(*error);
    *error = NULL;
}
static trtmc_model* load_mode(const char* root, const char* mode) {
    char path[4096], header[256];
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    FILE* file;
    unsigned shift;
    int length = snprintf(header, sizeof(header),
                          "{\"format\":1,\"family\":\"recurrent_fixture\",\"task\":\"%s\","
                          "\"backend\":\"fake\",\"sections\":{}}",
                          mode);
    if (length < 0 || length >= (int)sizeof(header) ||
        snprintf(path, sizeof(path), "%s/recurrent-c-%s.bundle", root, mode) >= (int)sizeof(path))
        return NULL;
    file = fopen(path, "wb");
    if (!file)
        return NULL;
    fwrite(magic, 1, 8, file);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), file);
    fwrite(header, 1, (size_t)length, file);
    if (fclose(file))
        return NULL;
    options.struct_size = sizeof(options);
    options.runtime_root = string(root);
    status(core->model_load(string(path), &options, &model, &error), TRTMC_OK, &error,
           "C extended mode model loads");
    return model;
}
static const trtmc_api_header* get_task(trtmc_model* model, const char* task) {
    const trtmc_api_header* table = NULL;
    trtmc_error* error = NULL;
    status(core->model_get_task_api(model, string(task), 1, 0, &table, &error), TRTMC_OK, &error,
           "C distinct recurrent Task discovered");
    return table;
}
static void extended(const char* root) {
    trtmc_model* model = load_mode(root, "enabled");
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *batch_result = NULL;
    trtmc_recurrent_state *a = NULL, *b = NULL;
    trtmc_recurrent_state_info_v1 info = {0};
    trtmc_recurrent_hidden_view_v1 hidden = {0};
    trtmc_recurrent_logits_view_v1 logits = {0};
    int32_t ids[] = {1, 2}, bad_id = 99;
    float embeds[] = {1, 0, 2, 0};
    trtmc_recurrent_tokens_logits_request_v1 tl = {0};
    trtmc_recurrent_tokens_hidden_request_v1 th = {0};
    trtmc_recurrent_embeddings_logits_request_v1 el = {0};
    trtmc_recurrent_embeddings_hidden_request_v1 eh = {0};
    trtmc_config_entry_v1 option = {0};
    trtmc_config_view_v1 config = {&option, 1};
    uint64_t count = 0;
    const trtmc_recurrent_tokens_to_logits_api_v1* task;
    const trtmc_recurrent_embeddings_to_logits_api_v1* et;
    const trtmc_recurrent_tokens_to_hidden_states_api_v1* ht;
    const trtmc_recurrent_embeddings_to_hidden_states_api_v1* eht;
    const trtmc_batch_recurrent_tokens_to_logits_api_v1* bt;
    const trtmc_batch_recurrent_embeddings_to_logits_api_v1* be;
    const trtmc_batch_recurrent_tokens_to_hidden_states_api_v1* bh;
    const trtmc_batch_recurrent_embeddings_to_hidden_states_api_v1* beh;
    const trtmc_recurrent_state_api_v1* state;
    trtmc_batch_recurrent_tokens_to_logits_item_v1 ti[2] = {0};
    trtmc_batch_recurrent_embeddings_to_logits_item_v1 ei[2] = {0};
    trtmc_batch_recurrent_tokens_to_hidden_states_item_v1 hi[2] = {0};
    trtmc_batch_recurrent_embeddings_to_hidden_states_item_v1 ehi[2] = {0};
    if (!model)
        return;
#define GET(Type, Id) ((const Type*)get_task(model, Id))
    task = GET(trtmc_recurrent_tokens_to_logits_api_v1, TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS);
    et =
        GET(trtmc_recurrent_embeddings_to_logits_api_v1, TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_LOGITS);
    ht = GET(trtmc_recurrent_tokens_to_hidden_states_api_v1,
             TRTMC_TASK_RECURRENT_TOKENS_TO_HIDDEN_STATES);
    eht = GET(trtmc_recurrent_embeddings_to_hidden_states_api_v1,
              TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES);
    bt = GET(trtmc_batch_recurrent_tokens_to_logits_api_v1,
             TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_LOGITS);
    be = GET(trtmc_batch_recurrent_embeddings_to_logits_api_v1,
             TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_LOGITS);
    bh = GET(trtmc_batch_recurrent_tokens_to_hidden_states_api_v1,
             TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_HIDDEN_STATES);
    beh = GET(trtmc_batch_recurrent_embeddings_to_hidden_states_api_v1,
              TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES);
#undef GET
    if (!task || !et || !ht || !eht || !bt || !be || !bh || !beh) {
        core->model_release(model);
        return;
    }
    state = task->state_api;
    status(task->create_state(model, &a, &error), TRTMC_OK, &error, "C first state");
    status(bt->create_state(model, &b, &error), TRTMC_OK, &error, "C batch factory state");
    if (!a || !b)
        goto finish;
    tl.input.token_ids = (trtmc_i32_view){ids, 2};
    th.input = tl.input;
    el.input.embeddings = (trtmc_recurrent_array_v1){
        {embeds, sizeof(embeds), TRTMC_RECURRENT_FLOAT32, TRTMC_RECURRENT_HOST, -1}, 2, 2};
    eh.input = el.input;
    status(et->forward(a, &el, NULL, &result, &error), TRTMC_OK, &error, "C embeddings to logits");
    status(et->result_view(result, &logits, &error), TRTMC_OK, &error, "C embedding logits view");
    check(((const float*)logits.logits.buffer.data)[16] == 3, "C embeddings preserved");
    status(ht->result_view(result, &hidden, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C wrong result type rejected");
    core->result_release(result);
    result = NULL;
    status(state->reset(a, &error), TRTMC_OK, &error, "C reset between distinct tasks");
    status(ht->forward(a, &th, NULL, &result, &error), TRTMC_OK, &error, "C tokens to hidden");
    status(ht->result_view(result, &hidden, &error), TRTMC_OK, &error, "C token hidden view");
    check(hidden.hidden.rows == 2 && hidden.hidden.columns == 2 &&
              ((const float*)hidden.hidden.buffer.data)[3] == -3,
          "C hidden has T H axes");
    core->result_release(result);
    result = NULL;
    status(state->reset(a, &error), TRTMC_OK, &error, "C reset embeddings hidden state");
    status(eht->forward(a, &eh, NULL, &result, &error), TRTMC_OK, &error, "C embeddings to hidden");
    core->result_release(result);
    result = NULL;
    tl.input.token_ids = (trtmc_i32_view){&bad_id, 1};
    status(task->forward(a, &tl, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C family invalid token preflight");
    status(state->info(a, &info, &error), TRTMC_OK, &error, "C state after invalid token");
    check(!info.poisoned && info.tokens_seen == 2, "C family invalid input preserves prefix");
    tl.input.token_ids = (trtmc_i32_view){ids, 2};
    option.name = string("unknown");
    option.value.kind = TRTMC_CONFIG_BOOL;
    status(task->forward(a, &tl, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
           "C family ConfigError preflight");
    tl.output_memory = TRTMC_RECURRENT_CUDA;
    status(task->forward(a, &tl, NULL, &result, &error), TRTMC_UNSUPPORTED, &error,
           "C unsupported target preflight");
    status(state->info(a, &info, &error), TRTMC_OK, &error,
           "C state after configuration rejection");
    check(!info.poisoned && info.tokens_seen == 2, "C rejected config or target keeps prefix");
    tl.output_memory = TRTMC_RECURRENT_HOST;
    status(state->reset(a, &error), TRTMC_OK, &error, "C reset before batches");
    ti[0].state = a;
    ti[1].state = b;
    ti[0].request = ti[1].request = tl;
    status(bt->forward(ti, 2, &batch_result, &error), TRTMC_OK, &error,
           "C native tokens logits batch");
    status(bt->result_count(batch_result, &count, &error), TRTMC_OK, &error, "C batch count");
    status(bt->result_item(batch_result, 1, &logits, &error), TRTMC_OK, &error, "C batch item");
    check(count == 2 && ((const float*)logits.logits.buffer.data)[16] == 3,
          "C batch order and values");
    status(bt->result_item(batch_result, 2, &logits, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C batch item bounds");
    ti[1].state = a;
    status(bt->forward(ti, 2, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C batch state alias rejected");
    ti[1].state = b;
    ti[1].config = config;
    status(bt->forward(ti, 2, &result, &error), TRTMC_INVALID_CONFIG, &error,
           "C batch config preflight before all mutations");
    ti[1].config = (trtmc_config_view_v1){0};
    ti[1].request.input.token_ids = (trtmc_i32_view){&bad_id, 1};
    status(bt->forward(ti, 2, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C batch input preflight");
    ti[1].request = tl;
    ti[1].request.output_memory = TRTMC_RECURRENT_CUDA;
    status(bt->forward(ti, 2, &result, &error), TRTMC_UNSUPPORTED, &error,
           "C batch unsupported target preflight");
    status(state->info(a, &info, &error), TRTMC_OK, &error, "C batch first state after preflights");
    check(!info.poisoned && info.tokens_seen == 2, "C first prefix preserved by batch preflight");
    status(state->info(b, &info, &error), TRTMC_OK, &error, "C batch last state after preflights");
    check(!info.poisoned && info.tokens_seen == 2, "C last prefix preserved by batch preflight");
    ti[1].request = tl;
    option.name = string("failure");
    option.value.kind = TRTMC_CONFIG_STRING;
    option.value.as.string = string("after_mutation");
    ti[0].config = config;
    status(bt->forward(ti, 2, &result, &error), TRTMC_INTERNAL_ERROR, &error,
           "C native batch mutation fault");
    status(state->info(a, &info, &error), TRTMC_OK, &error, "C first batch poison");
    check(info.poisoned, "C first poisoned");
    status(state->info(b, &info, &error), TRTMC_OK, &error, "C second batch poison");
    check(info.poisoned, "C second poisoned");
    status(state->reset(a, &error), TRTMC_OK, &error, "C first reset");
    status(state->reset(b, &error), TRTMC_OK, &error, "C second reset");
    ei[0].state = a;
    ei[1].state = b;
    ei[0].request = ei[1].request = el;
    status(be->forward(ei, 2, &result, &error), TRTMC_OK, &error,
           "C native embeddings logits batch");
    core->result_release(result);
    result = NULL;
    hi[0].state = a;
    hi[1].state = b;
    hi[0].request = hi[1].request = th;
    status(bh->forward(hi, 2, &result, &error), TRTMC_OK, &error, "C native tokens hidden batch");
    status(bh->result_item(result, 0, &hidden, &error), TRTMC_OK, &error, "C hidden batch item");
    check(((const float*)hidden.hidden.buffer.data)[2] == 6, "C token hidden batch continuation");
    core->result_release(result);
    result = NULL;
    ehi[0].state = a;
    ehi[1].state = b;
    ehi[0].request = ehi[1].request = eh;
    status(beh->forward(ehi, 2, &result, &error), TRTMC_OK, &error,
           "C native embeddings hidden batch");
    status(beh->result_item(result, 1, &hidden, &error), TRTMC_OK, &error,
           "C embeddings hidden item");
    check(((const float*)hidden.hidden.buffer.data)[2] == 9,
          "C all four batch methods are real dispatch");
finish:
    state->release(a);
    state->release(b);
    core->model_release(model);
    core->result_release(result);
    if (batch_result) {
        status(bt->result_item(batch_result, 0, &logits, &error), TRTMC_OK, &error,
               "C batch result retains storage after states and model release");
        check(((const float*)logits.logits.buffer.data)[16] == 3,
              "C batch snapshot remains immutable");
    }
    core->result_release(batch_result);
}
static void interchange(const char* root, int mamba) {
    trtmc_model* model = load_mode(root, mamba ? "mamba" : "rwkv");
    trtmc_model* other_model = load_mode(root, "enabled");
    const trtmc_recurrent_tokens_to_logits_api_v1 *task, *other_task;
    const trtmc_recurrent_state_api_v1* states;
    const trtmc_mamba1_state_exchange_api_v1* ma = NULL;
    const trtmc_rwkv4_state_exchange_api_v1* rw = NULL;
    trtmc_recurrent_state *state = NULL, *other = NULL;
    trtmc_result *saved = NULL, *result = NULL;
    trtmc_error* error = NULL;
    int32_t token = 3;
    trtmc_recurrent_tokens_logits_request_v1 request = {0};
    trtmc_recurrent_state_info_v1 info = {0};
    trtmc_mamba1_state_view_v1 mv = {0};
    trtmc_rwkv4_state_view_v1 rv = {0};
    trtmc_config_entry_v1 fault = {0};
    trtmc_config_view_v1 config = {&fault, 1};
    if (!model || !other_model)
        goto models;
    task = (const trtmc_recurrent_tokens_to_logits_api_v1*)get_task(
        model, TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS);
    other_task = (const trtmc_recurrent_tokens_to_logits_api_v1*)get_task(
        other_model, TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS);
    if (!task || !other_task)
        goto models;
    states = task->state_api;
    status(task->create_state(model, &state, &error), TRTMC_OK, &error,
           "C typed interchange state");
    status(other_task->create_state(other_model, &other, &error), TRTMC_OK, &error,
           "C unsupported interchange state");
    if (!state || !other)
        goto release;
    request.input.token_ids = (trtmc_i32_view){&token, 1};
    status(task->forward(state, &request, NULL, &result, &error), TRTMC_OK, &error,
           "C typed interchange prefix");
    core->result_release(result);
    result = NULL;
    if (mamba) {
        status(states->get_mamba1_exchange(state, 2, 0, &ma, &error), TRTMC_VERSION_MISMATCH,
               &error, "C Mamba interchange version checked");
        check(!ma, "C incompatible exchange returns null");
        status(states->get_mamba1_exchange(state, 1, 0, &ma, &error), TRTMC_OK, &error,
               "C Mamba getter exposes only declared interface");
        status(ma->snapshot(state, TRTMC_RECURRENT_HOST, &saved, &error), TRTMC_OK, &error,
               "C Mamba snapshot owns arrays");
        status(ma->result_view(saved, &mv, &error), TRTMC_OK, &error, "C Mamba snapshot view");
        check(mv.layer_count == 1 && mv.layers[0].convolution.rows == 2 &&
                  mv.layers[0].convolution.columns == 3 && mv.layers[0].ssm.columns == 4 &&
                  mv.layers[0].has_previous_state && mv.has_tokens_seen && mv.tokens_seen == 1,
              "C Mamba named per-layer axes and position flags");
        status(ma->assign(other, &mv, &error), TRTMC_UNSUPPORTED, &error,
               "C saved exchange table cannot grant another model missing capability");
        status(ma->snapshot(state, TRTMC_RECURRENT_CUDA, &result, &error), TRTMC_UNSUPPORTED,
               &error, "C no implicit host-to-device snapshot fallback");
        status(states->reset(state, &error), TRTMC_OK, &error, "C reset before Mamba assign");
        status(ma->assign(state, &mv, &error), TRTMC_OK, &error, "C Mamba typed snapshot replay");
    } else {
        status(states->get_rwkv4_exchange(state, 1, 1, &rw, &error), TRTMC_VERSION_MISMATCH, &error,
               "C RWKV interchange minor checked");
        status(states->get_rwkv4_exchange(state, 1, 0, &rw, &error), TRTMC_OK, &error,
               "C RWKV getter exposes typed interface");
        status(rw->snapshot(state, TRTMC_RECURRENT_HOST, &saved, &error), TRTMC_OK, &error,
               "C RWKV snapshot owns arrays");
        status(rw->result_view(saved, &rv, &error), TRTMC_OK, &error, "C RWKV snapshot view");
        check(rv.ffn_previous.rows == 2 && rv.ffn_previous.columns == 2 &&
                  ((const float*)rv.wkv_running_max.buffer.data)[0] == -1.0e30F,
              "C RWKV channel-layer layout and family max initialization");
        status(rw->assign(other, &rv, &error), TRTMC_UNSUPPORTED, &error,
               "C RWKV table does not confer capability on unsupported model");
        status(rw->snapshot(state, TRTMC_RECURRENT_CUDA, &result, &error), TRTMC_UNSUPPORTED,
               &error, "C RWKV target cannot silently change");
        status(states->reset(state, &error), TRTMC_OK, &error, "C reset before RWKV assign");
        status(rw->assign(state, &rv, &error), TRTMC_OK, &error, "C RWKV typed snapshot replay");
    }
    fault.name = string("failure");
    fault.value.kind = TRTMC_CONFIG_STRING;
    fault.value.as.string = string("after_mutation");
    status(task->forward(state, &request, &config, &result, &error), TRTMC_INTERNAL_ERROR, &error,
           "C interchange target becomes poisoned after mutation fault");
    if (mamba) {
        trtmc_mamba1_layer_state_v1 layer = mv.layers[0];
        trtmc_mamba1_state_view_v1 bad = mv;
        layer.convolution.buffer.scalar = TRTMC_RECURRENT_FLOAT16;
        layer.convolution.buffer.byte_size /= 2;
        bad.layers = &layer;
        status(ma->assign(state, &bad, &error), TRTMC_UNSUPPORTED, &error,
               "C unsupported Mamba dtype import rejected");
    } else {
        trtmc_rwkv4_state_view_v1 bad = rv;
        bad.ffn_previous.buffer.scalar = TRTMC_RECURRENT_BFLOAT16;
        bad.ffn_previous.buffer.byte_size /= 2;
        status(rw->assign(state, &bad, &error), TRTMC_UNSUPPORTED, &error,
               "C unsupported RWKV dtype import rejected");
    }
    status(states->info(state, &info, &error), TRTMC_OK, &error,
           "C after rejected recovery import");
    check(info.poisoned, "C rejected import preserves previous poison");
    status(mamba ? ma->assign(state, &mv, &error) : rw->assign(state, &rv, &error), TRTMC_OK,
           &error, "C successful typed import recovers state");
    status(states->info(state, &info, &error), TRTMC_OK, &error, "C imported state info");
    check(!info.poisoned && info.context_valid && info.tokens_seen == 1,
          "C typed assign restores exact prefix position");
release:
    states->release(state);
    states->release(other);
    core->result_release(result);
    if (saved) {
        if (mamba) {
            status(ma->result_view(saved, &mv, &error), TRTMC_OK, &error,
                   "C Mamba snapshot outlives state");
            check(((const float*)mv.layers[0].convolution.buffer.data)[0] == 3,
                  "C Mamba snapshot remains immutable");
        } else {
            status(rw->result_view(saved, &rv, &error), TRTMC_OK, &error,
                   "C RWKV snapshot outlives state");
            check(((const float*)rv.ffn_previous.buffer.data)[0] == 3,
                  "C RWKV snapshot remains immutable");
        }
    }
    core->result_release(saved);
models:
    core->model_release(model);
    core->model_release(other_model);
}
int main(int argc, char** argv) {
    char path[4096];
    FILE* file;
    unsigned shift;
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const char header[] = "{\"format\":1,\"family\":\"recurrent_fixture\",\"task\":\"enabled\","
                          "\"backend\":\"fake\",\"sections\":{}}";
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* api_header = NULL;
    const trtmc_recurrent_tokens_to_logits_api_v1* task = NULL;
    const trtmc_recurrent_state_api_v1* states = NULL;
    trtmc_recurrent_state *state = NULL, *branch = NULL, *idle = NULL;
    trtmc_result *first = NULL, *result = NULL;
    trtmc_recurrent_logits_view_v1 view = {0};
    trtmc_recurrent_state_info_v1 info = {0};
    int32_t ids[2] = {1, 2}, next = 4;
    trtmc_recurrent_tokens_logits_request_v1 request = {0};
    trtmc_config_entry_v1 fault = {0};
    trtmc_config_view_v1 config = {&fault, 1};
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    extended(argv[1]);
    interchange(argv[1], 1);
    interchange(argv[1], 0);
    if (snprintf(path, sizeof(path), "%s/recurrent-c.bundle", argv[1]) >= (int)sizeof(path))
        return 2;
    file = fopen(path, "wb");
    if (!file)
        return 2;
    fwrite(magic, 1, 8, file);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)(sizeof(header) - 1) >> shift) & 255), file);
    fwrite(header, 1, sizeof(header) - 1, file);
    if (fclose(file))
        return 2;
    options.struct_size = sizeof(options);
    options.runtime_root = string(argv[1]);
    status(core->model_load(string(path), &options, &model, &error), TRTMC_OK, &error,
           "C model loads");
    if (!model)
        goto cleanup;
    status(core->model_get_task_api(model, string(TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS), 1, 0,
                                    &api_header, &error),
           TRTMC_OK, &error, "C recurrent task table");
    task = (const trtmc_recurrent_tokens_to_logits_api_v1*)api_header;
    if (!task)
        goto cleanup;
    states = task->state_api;
    status(task->create_state(model, &state, &error), TRTMC_OK, &error,
           "C creates state without Config");
    status(task->create_state(model, &idle, &error), TRTMC_OK, &error, "C idle states coexist");
    if (!state || !idle)
        goto cleanup;
    request.input.token_ids = (trtmc_i32_view){ids, 2};
    status(task->forward(state, &request, NULL, &first, &error), TRTMC_OK, &error,
           "C zero-default output target is Host");
    status(task->result_view(first, &view, &error), TRTMC_OK, &error, "C logits typed view");
    check(view.logits.rows == 2 && view.logits.columns == 16 &&
              view.logits.buffer.memory == TRTMC_RECURRENT_HOST &&
              ((const float*)view.logits.buffer.data)[16] == 3,
          "C token rows retained; no sampling");
    status(states->clone(state, &branch, &error), TRTMC_OK, &error, "C explicit independent clone");
    core->model_release(model);
    model = NULL;
    request.input.token_ids = (trtmc_i32_view){&next, 1};
    status(task->forward(state, &request, NULL, &result, &error), TRTMC_OK, &error,
           "C state retains model after owner handle release");
    status(task->result_view(result, &view, &error), TRTMC_OK, &error, "C continued result");
    check(((const float*)view.logits.buffer.data)[0] == 7, "C state advances in place");
    core->result_release(result);
    result = NULL;
    next = 7;
    status(task->forward(branch, &request, NULL, &result, &error), TRTMC_OK, &error,
           "C cloned branch forward");
    status(task->result_view(result, &view, &error), TRTMC_OK, &error, "C clone result");
    check(((const float*)view.logits.buffer.data)[0] == 10, "C clone has independent prefix state");
    core->result_release(result);
    result = NULL;
    request.rows.kind = TRTMC_RECURRENT_ROWS_LAST;
    request.rows.last_count = 0;
    status(task->forward(state, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C malformed selector rejected before mutation");
    status(states->info(state, &info, &error), TRTMC_OK, &error, "C state info");
    check(!info.poisoned && info.tokens_seen == 3, "C pre-call error leaves state usable");
    request.rows.kind = TRTMC_RECURRENT_ROWS_ALL;
    fault.name = string("failure");
    fault.value.kind = TRTMC_CONFIG_STRING;
    fault.value.as.string = string("after_mutation");
    status(task->forward(state, &request, &config, &result, &error), TRTMC_INTERNAL_ERROR, &error,
           "C delegated mutation failure");
    status(states->info(state, &info, &error), TRTMC_OK, &error, "C poisoned state info");
    check(info.poisoned, "C failure poisons state");
    status(task->forward(state, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "C poisoned state is not silently retried");
    status(states->reset(state, &error), TRTMC_OK, &error, "C reset recovers");
    status(task->forward(state, &request, NULL, &result, &error), TRTMC_OK, &error,
           "C recovered forward");
    status(task->result_view(result, &view, &error), TRTMC_OK, &error, "C reset result");
    check(((const float*)view.logits.buffer.data)[0] == 7, "C reset is family initialization");
cleanup:
    if (states) {
        states->release(state);
        states->release(branch);
        states->release(idle);
        states->release(NULL);
    }
    core->model_release(model);
    core->result_release(result);
    if (task && first) {
        status(task->result_view(first, &view, &error), TRTMC_OK, &error,
               "C result outlives all states");
        check(((const float*)view.logits.buffer.data)[16] == 3, "C old result remains immutable");
    }
    core->result_release(first);
    fprintf(stderr, failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}

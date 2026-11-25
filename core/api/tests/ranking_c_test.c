/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(offsetof(trtmc_structured_ranking_api_v1, header) == 0,
               "ranking ABI header is first");
static trtmc_string_view text(const char* s) {
    trtmc_string_view v = {s, 0};
    v.size = strlen(s);
    return v;
}
int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* table = NULL;
    const trtmc_structured_ranking_api_v1* task;
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_candidate_scores_view_v1 view = {0};
    trtmc_ranking_cache_stats_v1 stats = {0};
    const float x[] = {.25F, .5F, .75F, 1.F};
    const int32_t n[] = {3, 4};
    const int64_t t[] = {INT64_C(1) << 40, INT64_C(2) << 40};
    const int64_t matrix[] = {2, 2}, vector[] = {2}, negative[] = {-1},
                  overflow[] = {INT64_MAX, INT64_MAX};
    trtmc_ranking_feature_v1 features[3];
    trtmc_structured_ranking_request_v1 request;
    char path[4096];
    FILE* file;
    unsigned shift;
    uint64_t i;
    int failed = 0;
    const char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const char json[] = "{\"format\":1,\"family\":\"ranking_fixture\",\"task\":\"structured_"
                        "ranking\",\"backend\":\"fake\",\"sections\":{}}";
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    if (snprintf(path, sizeof(path), "%s/ranking-c.bundle", argv[1]) >= (int)sizeof(path))
        return 2;
    file = fopen(path, "wb");
    if (!file)
        return 2;
    if (fwrite(magic, 1, 8, file) != 8)
        return 2;
    for (shift = 0; shift < 64; shift += 8)
        if (fputc((int)(((uint64_t)(sizeof(json) - 1) >> shift) & 255), file) == EOF)
            return 2;
    if (fwrite(json, 1, sizeof(json) - 1, file) != sizeof(json) - 1 || fclose(file))
        return 2;
#define OK(call)                                                                                   \
    do {                                                                                           \
        if ((call) != TRTMC_OK) {                                                                  \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
#define INVALID(call)                                                                              \
    do {                                                                                           \
        if ((call) != TRTMC_INVALID_ARGUMENT || !error) {                                          \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
        core->error_release(error);                                                                \
        error = NULL;                                                                              \
    } while (0)
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[1]);
    OK(core->model_load(text(path), &options, &model, &error));
    OK(core->model_get_task_api(model, text(TRTMC_TASK_STRUCTURED_RANKING), 1, 0, &table, &error));
    if (!table || table->byte_size != sizeof(*task)) {
        failed = 1;
        goto done;
    }
    task = (const trtmc_structured_ranking_api_v1*)table;
    features[0] = (trtmc_ranking_feature_v1){
        text("continuous"), TRTMC_RANKING_FLOAT32, matrix, 2, x, sizeof(x)};
    features[1] =
        (trtmc_ranking_feature_v1){text("counts"), TRTMC_RANKING_INT32, vector, 1, n, sizeof(n)};
    features[2] =
        (trtmc_ranking_feature_v1){text("times"), TRTMC_RANKING_INT64, vector, 1, t, sizeof(t)};
    request.features = features;
    request.feature_count = 3;
    OK(task->run(model, &request, NULL, &result, &error));
    OK(task->result_view(result, &view, &error));
    if (view.scores.count != 4 || view.scores.data[0] != 4.25F || view.scores.data[3] != 7.F) {
        failed = 1;
        goto done;
    }
    core->result_release(result);
    result = NULL;
    INVALID(task->run(model, NULL, NULL, &result, &error));
    if (result) {
        failed = 1;
        goto done;
    }
    features[1].shape = negative;
    INVALID(task->run(model, &request, NULL, &result, &error));
    features[1].shape = overflow;
    features[1].rank = 2;
    INVALID(task->run(model, &request, NULL, &result, &error));
    features[1].shape = NULL;
    INVALID(task->run(model, &request, NULL, &result, &error));
    features[1].shape = vector;
    features[1].rank = 1;
    features[1].data = NULL;
    INVALID(task->run(model, &request, NULL, &result, &error));
    features[1].data = n;
    OK(task->configure_cache(model, UINT64_MAX, 0, &error));
    OK(task->cache_stats(model, &stats, &error));
    if (stats.budget_bytes != UINT64_MAX || !stats.supported) {
        failed = 1;
        goto done;
    }
    OK(task->run(model, &request, NULL, &result, &error));
    core->model_release(model);
    model = NULL;
    OK(task->result_view(result, &view, &error));
    for (i = 0; i < view.scores.count; ++i)
        if (view.scores.data[i] < 4 || view.scores.data[i] > 7)
            failed = 1;
done:
    if (failed && error) {
        trtmc_string_view m = core->error_message(error);
        fprintf(stderr, "%.*s\n", (int)m.size, m.data);
    }
    core->error_release(error);
    core->result_release(result);
    core->model_release(model);
    if (failed)
        return 1;
    puts("ranking C ABI, shapes, INT64, budget and lifetime: PASS");
    return 0;
}

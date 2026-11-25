/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sdk_fixture.h"
#include "trtmc/trtmc.h"

#include <math.h>

static trtmc_string_view text(const char* s) {
    trtmc_string_view v;
    v.data = s;
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
    trtmc_load_options_v1 options;
    trtmc_candidate_scores_view_v1 view;
    trtmc_structured_ranking_request_v1 request;
    trtmc_ranking_cache_stats_v1 stats;
    HstuSdkFixture f;
    uint64_t i;
    float atol, rtol;
    int failed = 0;
    if (argc != 6 || !hstu_fixture_load(argv[3], &f) || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    atol = strtof(argv[4], NULL);
    rtol = strtof(argv[5], NULL);
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
#define REQUIRE_OK(call)                                                                           \
    do {                                                                                           \
        if ((call) != TRTMC_OK) {                                                                  \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
    REQUIRE_OK(core->model_load(text(argv[1]), &options, &model, &error));
    REQUIRE_OK(
        core->model_get_task_api(model, text(TRTMC_TASK_STRUCTURED_RANKING), 1, 0, &table, &error));
    if (!table || table->byte_size < sizeof(*task)) {
        failed = 1;
        goto done;
    }
    task = (const trtmc_structured_ranking_api_v1*)table;
    REQUIRE_OK(task->configure_cache(model, 134217728, 0, &error));
    REQUIRE_OK(task->cache_stats(model, &stats, &error));
    if (!stats.supported || stats.budget_bytes != 134217728 || !stats.reserved_bytes) {
        failed = 1;
        goto done;
    }
    request.features = f.features;
    request.feature_count = 10;
    REQUIRE_OK(task->run(model, &request, NULL, &result, &error));
    REQUIRE_OK(task->result_view(result, &view, &error));
    if (view.kind != TRTMC_SCORE_PROBABILITY || view.scores.rows != f.batch ||
        view.scores.columns != f.candidates || view.scores.count != f.batch * f.candidates) {
        failed = 1;
        goto done;
    }
    for (i = 0; i < view.scores.count; ++i)
        if (!isfinite(view.scores.data[i]) ||
            fabsf(view.scores.data[i] - f.expected[i]) > atol + rtol * fabsf(f.expected[i])) {
            failed = 1;
            goto done;
        }
    REQUIRE_OK(task->cache_stats(model, &stats, &error));
    if (stats.used_bytes || stats.active_sequences || !stats.peak_bytes) {
        failed = 1;
        goto done;
    }
    core->model_release(model);
    model = NULL;
    REQUIRE_OK(task->result_view(result, &view, &error));
    if (view.scores.count != f.batch * f.candidates)
        failed = 1;
done:
    if (failed && error) {
        trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "%.*s\n", (int)message.size, message.data);
    }
    core->error_release(error);
    core->result_release(result);
    core->model_release(model);
    hstu_fixture_free(&f);
    if (failed)
        return 1;
    puts("HSTU C SDK scores, cache accounting and result lifetime: PASS");
    return 0;
}

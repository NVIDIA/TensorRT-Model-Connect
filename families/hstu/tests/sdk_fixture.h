/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef HSTU_TEST_SDK_FIXTURE_H
#define HSTU_TEST_SDK_FIXTURE_H
#include "trtmc/ranking.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Family-test data only: three native-endian uint64 capacities, the ten
 * documented input arrays in schema order, then the oracle score matrix. */
typedef struct {
    uint64_t batch, history, candidates;
    trtmc_ranking_feature_v1 features[10];
    int64_t shapes[10][3];
    void* storage[10];
    float* expected;
} HstuSdkFixture;
static void hstu_fixture_free(HstuSdkFixture* f) {
    unsigned i;
    for (i = 0; i < 10; ++i)
        free(f->storage[i]);
    free(f->expected);
}
static int hstu_fixture_load(const char* path, HstuSdkFixture* f) {
    static const char* names[] = {"history_item_embeddings",   "history_category_embeddings",
                                  "history_lengths",           "history_timestamps",
                                  "history_actions",           "user_embeddings",
                                  "query_timestamp",           "num_candidates",
                                  "candidate_item_embeddings", "candidate_category_embeddings"};
    static const uint32_t types[] = {1, 1, 2, 3, 2, 1, 3, 2, 1, 1};
    uint64_t header[3];
    unsigned i;
    FILE* file;
    memset(f, 0, sizeof(*f));
    file = fopen(path, "rb");
    if (!file)
        return 0;
    if (fread(header, sizeof(uint64_t), 3, file) != 3 || !header[0] || header[0] > 64 ||
        !header[1] || header[1] > 16384 || !header[2] || header[2] > 2048) {
        fclose(file);
        return 0;
    }
    f->batch = header[0];
    f->history = header[1];
    f->candidates = header[2];
    for (i = 0; i < 10; ++i) {
        uint64_t count;
        trtmc_ranking_feature_v1* t = &f->features[i];
        t->name.data = names[i];
        t->name.size = strlen(names[i]);
        t->type = types[i];
        f->shapes[i][0] = (int64_t)f->batch;
        f->shapes[i][1] = (int64_t)(i < 2 || i == 3 || i == 4 ? f->history
                                    : i >= 8                  ? f->candidates
                                                              : 1);
        t->rank = (i < 2 || i == 5 || i >= 8) ? 3 : 2;
        f->shapes[i][2] = 512;
        t->shape = f->shapes[i];
        count = f->batch * (uint64_t)f->shapes[i][1] * (t->rank == 3 ? 512 : 1);
        t->byte_size = count * (types[i] == TRTMC_RANKING_INT64 ? 8 : 4);
        f->storage[i] = malloc((size_t)t->byte_size);
        t->data = f->storage[i];
        if (!t->data || fread(f->storage[i], 1, (size_t)t->byte_size, file) != t->byte_size) {
            fclose(file);
            hstu_fixture_free(f);
            return 0;
        }
    }
    f->expected = (float*)malloc((size_t)(f->batch * f->candidates * sizeof(float)));
    if (!f->expected || fread(f->expected, sizeof(float), (size_t)(f->batch * f->candidates),
                              file) != f->batch * f->candidates) {
        fclose(file);
        hstu_fixture_free(f);
        return 0;
    }
    if (fgetc(file) != EOF) {
        fclose(file);
        hstu_fixture_free(f);
        return 0;
    }
    fclose(file);
    return 1;
}
#endif

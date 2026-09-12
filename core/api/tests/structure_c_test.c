/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/structure.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(offsetof(trtmc_molecular_document_to_structure_api_v1, header) == 0,
               "molecular structure table begins with the ABI header");

static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view text(const char* value) {
    const trtmc_string_view out = {value, (uint64_t)strlen(value)};
    return out;
}
static int same(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) &&
           (!value.size || !memcmp(value.data, expected, (size_t)value.size));
}
static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static void checked(trtmc_status actual, trtmc_status expected, trtmc_error** error,
                    const char* message) {
    check(actual == expected, message);
    check(actual == TRTMC_OK ? *error == NULL : *error != NULL, "C error ownership is consistent");
    core->error_release(*error);
    *error = NULL;
}
static trtmc_model* load(const char* root, const char* mode) {
    const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char path[4096], header[512];
    int size;
    unsigned shift;
    FILE* file;
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    if (snprintf(path, sizeof(path), "%s/structure-c-%s.bundle", root, mode) >= (int)sizeof(path))
        exit(2);
    size = snprintf(header, sizeof(header),
                    "{\"format\":1,\"family\":\"structure_fixture\",\"task\":\"%s\","
                    "\"backend\":\"fake\",\"sections\":{}}",
                    mode);
    if (size < 0 || (size_t)size >= sizeof(header) || !(file = fopen(path, "wb")))
        exit(2);
    if (fwrite(magic, 1, sizeof(magic), file) != sizeof(magic))
        exit(2);
    for (shift = 0; shift < 64; shift += 8)
        if (fputc((int)(((uint64_t)size >> shift) & 255), file) == EOF)
            exit(2);
    if (fwrite(header, 1, (size_t)size, file) != (size_t)size || fclose(file))
        exit(2);
    options.struct_size = sizeof(options);
    options.runtime_root = text(root);
    checked(core->model_load(text(path), &options, &model, &error), TRTMC_OK, &error,
            "C structure fixture loads");
    return model;
}

int main(int argc, char** argv) {
    const uint8_t prepared[] = {'B', '2', 'R', 'Q', 0, 255, 1};
    const uint8_t yaml[] = {'a', ':', ' ', 'A', '\n'};
    const uint8_t json[] = {'{', '"', 'a', '"', ':', '1', '}'};
    trtmc_model *model, *disabled, *guarded, *without_confidence;
    const trtmc_api_header* header = NULL;
    const trtmc_molecular_document_to_structure_api_v1* task;
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *other = NULL;
    trtmc_molecular_structure_view_v1 view = {0};
    trtmc_molecular_document_to_structure_request_v1 request = {0};
    trtmc_config_entry_v1 entries[4] = {0};
    trtmc_config_view_v1 config = {entries, 1};
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    model = load(argv[1], "molecular_document_to_structure");
    disabled = load(argv[1], "disabled");
    guarded = load(argv[1], "must_not_run");
    without_confidence = load(argv[1], "without_confidence");
    if (!model || !disabled || !guarded || !without_confidence)
        return 2;
    checked(core->model_get_task_api(model, text(TRTMC_TASK_MOLECULAR_DOCUMENT_TO_STRUCTURE), 1, 0,
                                     &header, &error),
            TRTMC_OK, &error, "C structure table lookup");
    if (!header)
        return 2;
    task = (const trtmc_molecular_document_to_structure_api_v1*)header;
    check(header->major == 1 && header->minor == 0 && header->byte_size == sizeof(*task),
          "C complete versioned structure table");
    request.document = prepared;
    request.document_size = sizeof(prepared);
    request.encoding = text("b2rq");
    request.source_path = text("not-opened/relative/request.yaml");
    checked(task->run(model, &request, NULL, &result, &error), TRTMC_OK, &error,
            "C prepared binary structure request");
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error, "C structure result view");
    check(same(view.structure, "data_fixture\n# b2rq:4232525100ff01\n") &&
              view.format == TRTMC_STRUCTURE_MMCIF &&
              same(view.metadata_json,
                   "{\"encoding\":\"b2rq\",\"source_path\":\"not-opened/"
                   "relative/request.yaml\",\"seed\":42,\"sampling_steps\":200}"),
          "C embedded NUL/FF bytes, encoding and complete source path survive");
    check(view.has_confidence && view.confidence.confidence_score == 0.1F &&
              view.confidence.ptm == 0.2F && view.confidence.iptm == 0.3F &&
              view.confidence.ligand_iptm == 0.4F && view.confidence.protein_iptm == 0.5F &&
              view.confidence.complex_plddt == 66 && view.confidence.complex_iplddt == 77 &&
              view.confidence.plddt.size == 3 && view.confidence.plddt.data[0] == 88 &&
              view.confidence.plddt.data[1] == 0 && view.confidence.plddt.data[2] == 99,
          "C preserves every confidence field and PLDDT order/units");
    {
        const uint8_t* documents[] = {yaml, json};
        const uint64_t sizes[] = {sizeof(yaml), sizeof(json)};
        const char* encodings[] = {"yaml", "json"};
        const char* expected[] = {"data_fixture\n# yaml:613a20410a\n",
                                  "data_fixture\n# json:7b2261223a317d\n"};
        size_t i;
        for (i = 0; i < 2; ++i) {
            trtmc_molecular_document_to_structure_request_v1 next = request;
            next.document = documents[i];
            next.document_size = sizes[i];
            next.encoding = text(encodings[i]);
            checked(task->run(model, &next, NULL, &other, &error), TRTMC_OK, &error,
                    "C explicitly encoded text document");
            checked(task->result_view(other, &view, &error), TRTMC_OK, &error,
                    "C text document result view");
            check(same(view.structure, expected[i]), "C text bytes and encoding remain unchanged");
            core->result_release(other);
            other = NULL;
        }
    }
    checked(task->run(without_confidence, &request, NULL, &other, &error), TRTMC_OK, &error,
            "C confidence-free family call");
    checked(task->result_view(other, &view, &error), TRTMC_OK, &error, "C confidence-free view");
    check(view.has_confidence == 0, "C absent confidence is explicitly absent, not zero scores");
    core->result_release(other);
    other = NULL;
    entries[0].name = text("unknown");
    entries[0].value.kind = TRTMC_CONFIG_BOOL;
    entries[0].value.as.boolean = 1;
    checked(task->run(guarded, &request, &config, &other, &error), TRTMC_INVALID_CONFIG, &error,
            "C unknown Config is rejected before family execution");
    entries[0].name = text("seed");
    checked(task->run(guarded, &request, &config, &other, &error), TRTMC_INVALID_CONFIG, &error,
            "C wrong Config type is rejected before family execution");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 0;
    entries[1] = entries[0];
    config.count = 2;
    checked(task->run(guarded, &request, &config, &other, &error), TRTMC_INVALID_CONFIG, &error,
            "C duplicate Config is rejected before family execution");
    entries[1].name = text("include_confidence");
    entries[1].value.kind = TRTMC_CONFIG_BOOL;
    entries[1].value.as.boolean = 0;
    entries[2].name = text("output_format");
    entries[2].value.kind = TRTMC_CONFIG_STRING;
    entries[2].value.as.string = text("pdb");
    entries[3].name = text("sampling_steps");
    entries[3].value.kind = TRTMC_CONFIG_I64;
    entries[3].value.as.i64 = 12;
    config.count = 4;
    checked(task->run(model, &request, &config, &other, &error), TRTMC_OK, &error,
            "C explicit zero/false/string Config controls remain family-owned");
    checked(task->result_view(other, &view, &error), TRTMC_OK, &error, "C PDB result view");
    check(view.format == TRTMC_STRUCTURE_PDB && !view.has_confidence &&
              same(view.metadata_json, "{\"encoding\":\"b2rq\",\"source_path\":\"not-opened/"
                                       "relative/request.yaml\",\"seed\":0,\"sampling_steps\":12}"),
          "C seed zero, absent confidence and explicit PDB format are not replaced by defaults");
    core->result_release(other);
    other = NULL;
    checked(task->run(disabled, &request, NULL, &other, &error), TRTMC_UNSUPPORTED, &error,
            "C foreign table does not grant an unsupported Task");
    check(other == NULL, "C unsupported call clears result");
    {
        const char* modes[] = {"empty_structure", "bad_format", "bad_confidence", "bad_plddt"};
        size_t i;
        for (i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
            trtmc_model* invalid = load(argv[1], modes[i]);
            checked(task->run(invalid, &request, NULL, &other, &error), TRTMC_INTERNAL_ERROR,
                    &error, "C malformed family structure output is rejected");
            check(other == NULL, "C malformed output never returns a partial result");
            core->model_release(invalid);
        }
    }
    {
        trtmc_molecular_document_to_structure_request_v1 invalid = request;
        invalid.document = NULL;
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
                "C nonempty NULL document rejects before dereference");
        invalid = request;
        invalid.document_size = UINT64_MAX;
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
                "C impossible document length rejects before dereference");
        invalid = request;
        invalid.encoding = (trtmc_string_view){NULL, 1};
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
                "C malformed encoding storage rejects");
        invalid = request;
        invalid.source_path = (trtmc_string_view){NULL, 1};
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
                "C malformed source-path storage rejects");
        invalid = request;
        invalid.encoding = text("csv");
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_UNSUPPORTED, &error,
                "C family rejects unsupported encoding without conversion");
        invalid = request;
        invalid.document = json;
        invalid.document_size = sizeof(json);
        checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
                "C malformed prepared content is not retried as another encoding");
    }
    checked(task->run(model, NULL, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C NULL request rejects");
    checked(task->result_view(result, NULL, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C NULL result-view output rejects");
    core->model_release(model);
    core->model_release(disabled);
    core->model_release(guarded);
    core->model_release(without_confidence);
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
            "C structure result survives model release");
    check(same(view.structure, "data_fixture\n# b2rq:4232525100ff01\n") &&
              view.confidence.plddt.data[2] == 99,
          "C retained strings and confidence array remain owned");
    core->result_release(result);
    puts("C molecular structure: YAML, JSON and binary prepared requests preserved");
    return failures ? 1 : 0;
}

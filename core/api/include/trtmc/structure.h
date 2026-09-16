/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_STRUCTURE_H
#define TRTMC_STRUCTURE_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All fields borrow through synchronous run. document is a nonempty byte
 * sequence, NOT a NUL-terminated string. encoding is explicit, for example
 * yaml, json or b2rq; the family validates accepted representations. Core
 * never parses, guesses, converts, or prepares a document. source_path is
 * the original document location for family-owned reference resolution;
 * empty means unspecified. Core does not open that path. */
typedef struct {
    const uint8_t* document;
    uint64_t document_size;
    trtmc_string_view encoding;
    trtmc_string_view source_path;
} trtmc_molecular_document_to_structure_request_v1;

enum { TRTMC_STRUCTURE_MMCIF = 1, TRTMC_STRUCTURE_PDB = 2 };

/* Scores retain the family's documented units; Core performs no normalization.
 * plddt is an owned-result view in the exact family residue/token order. */
typedef struct {
    float confidence_score;
    float ptm, iptm;
    float ligand_iptm, protein_iptm;
    float complex_plddt, complex_iplddt;
    trtmc_f32_view plddt;
} trtmc_structure_confidence_view_v1;

/* All nested views borrow the result lifetime. has_confidence=0 means absent,
 * not zero confidence; ignore confidence in that case. metadata_json retains
 * family output verbatim and is not interpreted by Core. */
typedef struct {
    trtmc_string_view structure;
    uint32_t format;
    trtmc_string_view metadata_json;
    uint32_t has_confidence;
    trtmc_structure_confidence_view_v1 confidence;
} trtmc_molecular_structure_view_v1;

#define TRTMC_TASK_MOLECULAR_DOCUMENT_TO_STRUCTURE "molecular_document_to_structure"

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_molecular_document_to_structure_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_molecular_structure_view_v1*,
                                          trtmc_error**);
} trtmc_molecular_document_to_structure_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_STRUCTURE_H */

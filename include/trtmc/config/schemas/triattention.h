/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Schema for the "triattention" namespace. Mirrors
// python/tensorrt_model_connect/runtime_config/schemas/triattention.py one-for-one.
//
// The cross-language field-set match test (tests/builder/test_config_schemas_crosslang.py)
// will fail fast if the two sides drift. Until the codegen pipeline lands,
// keeping these in sync is manual.
//
// This header exists so tests can reach into the schema directly (e.g. for
// typed getters by name). The registration itself lives in the matching
// .cpp file next to it; force-link anchor ensures the static-init isn't
// stripped when trtmc_core is linked as a static archive.

#include "trtmc/config/schema_registry.h"

namespace trtmc::config::schemas {

// Construct the TriAttention schema. Called from the anchor .cpp; also
// available to tests that want to inspect field metadata directly.
Schema make_triattention_schema();

} // namespace trtmc::config::schemas

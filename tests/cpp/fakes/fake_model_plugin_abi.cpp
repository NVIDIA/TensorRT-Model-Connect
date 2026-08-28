/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <cstdint>
#include <cstdlib>

namespace trtmc {
class PipelineRegistry;
}

#ifndef TRTMC_FAKE_MODEL_PLUGIN_OMIT_ABI
#ifndef TRTMC_FAKE_MODEL_PLUGIN_ABI_VERSION
#define TRTMC_FAKE_MODEL_PLUGIN_ABI_VERSION ::trtmc::kModelPluginAbiVersion
#endif
extern "C" std::uint32_t trtmc_model_plugin_abi_version() {
    return TRTMC_FAKE_MODEL_PLUGIN_ABI_VERSION;
}
#endif

// The loader must reject these fixtures before consulting either legacy
// entrypoint. Calling one is a hard test failure.
extern "C" const char* trtmc_model_plugin_id() {
    std::abort();
}

extern "C" void trtmc_register_model_plugin(::trtmc::PipelineRegistry*) {
    std::abort();
}

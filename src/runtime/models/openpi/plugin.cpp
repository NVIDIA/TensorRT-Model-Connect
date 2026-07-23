/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/openpi/config.h"
#include "runtime/models/openpi/paligemma_bpe.h"
#include "runtime/models/openpi/pipeline.h"
#include "runtime/models/openpi/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

#ifndef TRTMC_OPENPI_RMS_NORM_PLUGIN
#define TRTMC_OPENPI_RMS_NORM_PLUGIN 0
#endif

#if TRTMC_OPENPI_RMS_NORM_PLUGIN
extern "C" void trtmc_openpi_rms_norm_plugin_force_link();
#endif

#include <memory>
#include <stdexcept>
#include <utility>

namespace trtmc {

class OpenPIPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
#if TRTMC_OPENPI_RMS_NORM_PLUGIN
        // Keep the plugin translation unit linked and its static creator
        // registration observable before either plan is deserialized.
        trtmc_openpi_rms_norm_plugin_force_link();
#endif
        auto verified = openpi::verify_openpi_bundle_integrity(ctx.bundle);
        if (verified.config_json != ctx.config_json) {
            throw std::runtime_error(
                "OpenPI verified config.json differs from the factory configuration");
        }
        auto tokenizer_asset = openpi::parse_paligemma_bpe_asset(verified.tokenizer_bytes);

        ModuleCreateOptions prefill_options;
        prefill_options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        // Per-engine graph capture cannot represent alternating action buffers
        // or per-step timestep addresses. The complete ten-step loop remains
        // asynchronous and device-resident on one stream.
        prefill_options.cuda_graphs = false;
        auto prefill = openpi::load_openpi_module(ctx.backend, verified.prefill_plan, "engine_plan",
                                                  "openpi_prefill", prefill_options);

        ModuleCreateOptions action_options = prefill_options;
        action_options.stream = prefill->stream();
        auto action = openpi::load_openpi_module(ctx.backend, verified.action_plan,
                                                 "openpi_action_step_engine_plan",
                                                 "openpi_action_step", action_options);

        return std::make_unique<openpi::OpenPIPipeline>(
            std::move(prefill), std::move(action), std::move(verified.config),
            std::move(verified.normalization),
            openpi::PaligemmaBpeTokenizer(std::move(tokenizer_asset)), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_openpi_plugin, OpenPIPlugin, "openpi_vla");

} // namespace trtmc

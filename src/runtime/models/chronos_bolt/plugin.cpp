// ChronosBoltPlugin: handles "chronos_bolt_torchtrt" strategy.
//
// This is a numeric forecasting pipeline for Chronos-Bolt-style bundles.
// The C++ runtime keeps the implementation intentionally narrow: load the
// TRT engine, feed dense context tensors, and return the forecast tensor.

#include "runtime/models/chronos_bolt/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

class ChronosBoltPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "chronos_bolt forecast");

        const auto& json = ctx.config_json;
        int32_t context_length =
            extract_json_int(json, "context_length", ctx.config.max_cache_length);
        int32_t prediction_length = extract_json_int(json, "prediction_length",
                                                     extract_json_int(json, "forecast_length", 24));
        int32_t num_quantiles = extract_json_int(
            json, "num_quantiles",
            static_cast<int32_t>(extract_json_float_array(json, "quantiles").size()));
        if (num_quantiles <= 0)
            num_quantiles = 3;

        return std::make_unique<ChronosBoltPipeline>(std::move(loaded.module), context_length,
                                                     prediction_length, num_quantiles,
                                                     ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_chronos_bolt_plugin, ChronosBoltPlugin,
                                       "chronos_bolt_torchtrt");

} // namespace trtmc

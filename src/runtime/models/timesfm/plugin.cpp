// TimesFmPlugin: handles "timesfm_torchtrt" bundles.
// Loads a single TRT engine and routes solve() through TimesFmPipeline.

#include "runtime/models/timesfm/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

class TimesFmPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "timesfm engine");

        int32_t default_freq = extract_json_int(ctx.config_json, "timesfm_default_freq", 0);
        if (default_freq == 0)
            default_freq = extract_json_int(ctx.config_json, "freq", 0);

        int32_t prediction_length =
            extract_json_int(ctx.config_json, "timesfm_prediction_length", 0);
        if (prediction_length <= 0)
            prediction_length = extract_json_int(ctx.config_json, "prediction_length", 0);
        if (prediction_length <= 0)
            prediction_length = extract_json_int(ctx.config_json, "forecast_horizon", 0);

        return std::make_unique<TimesFmPipeline>(std::move(loaded.module), default_freq,
                                                 prediction_length, ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_timesfm_plugin, TimesFmPlugin, "timesfm_torchtrt");

} // namespace trtmc

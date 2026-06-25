// SamPlugin: SAM-owned prompted segmentation strategy.

#include "plugin_helpers.h"
#include "runtime/models/sam/sam_pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

SamConfig make_sam_config(const std::string& json) {
    SamConfig cfg;
    cfg.image_size = extract_json_int(json, "sam_image_size",
                                      extract_json_int(json, "input_image_h", cfg.image_size));
    cfg.image_embedding_size =
        extract_json_int(json, "sam_image_embedding_size", cfg.image_embedding_size);
    cfg.decoder_hidden_size =
        extract_json_int(json, "sam_decoder_hidden_size", cfg.decoder_hidden_size);
    cfg.num_mask_outputs = extract_json_int(json, "sam_num_mask_outputs", cfg.num_mask_outputs);
    cfg.num_multimask_outputs =
        extract_json_int(json, "sam_num_multimask_outputs", cfg.num_multimask_outputs);

    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);

    cfg.point_embed_bg =
        extract_json_float_array(json, "sam_point_embed_0", cfg.decoder_hidden_size);
    cfg.point_embed_fg =
        extract_json_float_array(json, "sam_point_embed_1", cfg.decoder_hidden_size);
    cfg.not_a_point_embed =
        extract_json_float_array(json, "sam_not_a_point_embed", cfg.decoder_hidden_size);
    cfg.shared_image_pe =
        extract_json_float_array(json, "sam_shared_image_pe", cfg.decoder_hidden_size);
    return cfg;
}

} // namespace

class SamPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        ModuleCreateOptions encoder_opts = opts;
        std::string encoder_section = "engine_plan";
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            encoder_opts.distributed_communicator = tp_group.communicator;
            encoder_opts.distributed_owner = tp_group.owner;
            encoder_section = tp_engine_section_name(tp_group.rank);
        }

        auto encoder =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, encoder_section),
                                      "sam image_encoder", encoder_opts);
        auto decoder = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "sam mask_decoder", opts);
        return std::make_unique<SamPipeline>(std::move(encoder.module), std::move(decoder.module),
                                             make_sam_config(ctx.config_json),
                                             ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam_plugin, SamPlugin, "sam_prompted_segmentation");

} // namespace trtmc

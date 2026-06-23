// SegmentationPlugin: handles "segmentation" and "prompted_segmentation"
// strategies. SegFormer (single encoder) and SAM (encoder + mask decoder).

#include "plugin_helpers.h"
#include "runtime/models/segmentation/sam3_pipeline.h"
#include "runtime/models/segmentation/sam_pipeline.h"
#include "runtime/models/segmentation/segment_pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
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

Sam3Config make_sam3_config(const std::string& json) {
    Sam3Config cfg;
    cfg.text_max_position_embeddings = extract_json_int(json, "sam3_text_max_position_embeddings",
                                                        cfg.text_max_position_embeddings);
    cfg.text_pad_token_id = extract_json_int(json, "sam3_text_pad_token_id", cfg.text_pad_token_id);
    cfg.text_projection_dim =
        extract_json_int(json, "sam3_text_projection_dim", cfg.text_projection_dim);
    cfg.image_size = extract_json_int(json, "sam3_image_size",
                                      extract_json_int(json, "input_image_h", cfg.image_size));
    cfg.low_res_mask_size = extract_json_int(json, "sam3_low_res_mask_size", cfg.low_res_mask_size);
    cfg.num_queries = extract_json_int(json, "sam3_num_queries", cfg.num_queries);
    cfg.score_threshold = extract_json_float(json, "sam3_score_threshold", cfg.score_threshold);
    cfg.mask_threshold = extract_json_float(json, "sam3_mask_threshold", cfg.mask_threshold);

    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);
    return cfg;
}

} // namespace

class SegmentationPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        const auto prompted_variant =
            extract_json_string(ctx.config_json, "prompted_segmentation_variant", "");
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        if (prompted_variant == "sam3_text_prompt_pcs") {
            auto text_encoder = load_trt_module_from_plan(
                ctx.backend, find_section(ctx.bundle, "engine_plan"), "sam3 text_encoder", opts);
            auto vision_encoder = load_trt_module_from_plan(
                ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "sam3 vision_encoder",
                opts);
            auto core_engine = load_trt_module_from_plan(
                ctx.backend, find_section(ctx.bundle, "sam3_core_engine_plan"), "sam3 core_engine",
                opts);
            auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
            return std::make_unique<Sam3Pipeline>(
                std::move(text_encoder.module), std::move(vision_encoder.module),
                std::move(core_engine.module), std::move(tokenizer),
                make_sam3_config(ctx.config_json), ctx.bundle.info.model_id);
        }

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

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, encoder_section), "engine_plan", encoder_opts);

        if (ctx.config.runtime_strategy == "prompted_segmentation") {
            auto decoder = try_load_trt_module_from_plan(
                ctx.backend, find_section(ctx.bundle, "vision_engine_plan"),
                "vision_plan (SAM mask_decoder)", opts);
            if (decoder.module && decoder.module->ok())
                return std::make_unique<SamPipeline>(
                    std::move(loaded.module), std::move(decoder.module),
                    make_sam_config(ctx.config_json), ctx.bundle.info.model_id);
            // Fallback to single-encoder segmentation if decoder failed
            return std::make_unique<SegmentPipeline>(std::move(loaded.module),
                                                     ctx.bundle.info.model_id);
        }

        // strategy == "segmentation"
        return std::make_unique<SegmentPipeline>(std::move(loaded.module),
                                                 ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_segmentation_plugin, SegmentationPlugin,
                                       "segmentation", "prompted_segmentation");

} // namespace trtmc

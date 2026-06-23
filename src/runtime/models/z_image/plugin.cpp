// ZImagePlugin: handles "diffusion_zimage" strategy.
// Z-Image diffusion pipeline with Qwen3 text encoder, denoiser, VAE,
// and Z-Image-specific preprocessor weights (timestep embedder, caption
// embedder, patch embedder).

#include "diffusion_helpers.h"
#include "plugin_helpers.h"
#include "runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h"
#include "runtime/models/z_image/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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

std::string tp_denoiser_section_name(int32_t rank) {
    return "denoiser_plan_tp_rank" + std::to_string(rank);
}

// Parse Z-Image-specific preprocessor weights from the preprocessor_weights
// bundle section. These weights are separate from the standard
// PreprocessorWeights and include timestep embedder MLP, caption embedder
// projection + norm, cap_pad_token, and patch (x) embedder.
ZImagePreprocessorWeights parse_zimage_preprocessor_weights(const std::vector<char>& data) {
    ZImagePreprocessorWeights w;
    std::string index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    if (!diffusion::extract_preprocessor_index(data, index_json, blob, blob_size))
        return w;

    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "t_embedder.mlp.0.weight",
                                        w.t_embedder_mlp_0_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "t_embedder.mlp.0.bias",
                                        w.t_embedder_mlp_0_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "t_embedder.mlp.2.weight",
                                        w.t_embedder_mlp_2_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "t_embedder.mlp.2.bias",
                                        w.t_embedder_mlp_2_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "cap_embedder.proj.weight",
                                        w.cap_proj_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "cap_embedder.proj.bias",
                                        w.cap_proj_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "cap_embedder.norm.weight",
                                        w.cap_norm_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "cap_pad_token",
                                        w.cap_pad_token);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "x_embedder.weight",
                                        w.x_embed_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "x_embedder.bias",
                                        w.x_embed_bias);

    // Derive dimensions
    if (!w.cap_proj_bias.empty())
        w.dit_dim = static_cast<int32_t>(w.cap_proj_bias.size());
    if (!w.cap_norm_weight.empty())
        w.cap_dim = static_cast<int32_t>(w.cap_norm_weight.size());
    if (!w.t_embedder_mlp_0_bias.empty())
        w.freq_dim = static_cast<int32_t>(w.t_embedder_mlp_0_bias.size());

    w.valid = !w.x_embed_weight.empty() && !w.t_embedder_mlp_0_weight.empty() &&
              !w.cap_proj_weight.empty() && !w.cap_norm_weight.empty();

    std::cerr << "[z-image] Preprocessor weights: " << (w.valid ? "OK" : "INCOMPLETE")
              << " (dit_dim=" << w.dit_dim << ", cap_dim=" << w.cap_dim << ")\n";
    return w;
}

} // namespace

class ZImagePlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        std::string denoiser_section_name = "denoiser_plan";
        ModuleCreateOptions denoiser_opts = opts;
        const ModuleCreateOptions* denoiser_options = nullptr;
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            denoiser_section_name = tp_denoiser_section_name(tp_group.rank);
            denoiser_opts.distributed_communicator = tp_group.communicator;
            denoiser_opts.distributed_owner = tp_group.owner;
            denoiser_options = &denoiser_opts;
        }

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts,
                                          denoiser_section_name, denoiser_options);

        // Extract first text encoder
        std::unique_ptr<TrtModule> te_module;
        if (!parts.text_encoders.empty())
            te_module = std::move(parts.text_encoders[0].module);

        // Parse Z-Image specific preprocessor weights
        ZImagePreprocessorWeights z_pw;
        const auto* pw_sec = find_section(ctx.bundle, "preprocessor_weights");
        if (pw_sec && !pw_sec->empty())
            z_pw = parse_zimage_preprocessor_weights(*pw_sec);

        return std::make_unique<ZImagePipeline>(
            std::move(te_module), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(z_pw),
            std::move(parts.tokenizer), ctx.bundle.info.model_id, ctx.bundle_path, tp_group.owner,
            tp_group.rank, tp_group.tp_size);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_zimage_plugin, ZImagePlugin, "diffusion_zimage");

} // namespace trtmc

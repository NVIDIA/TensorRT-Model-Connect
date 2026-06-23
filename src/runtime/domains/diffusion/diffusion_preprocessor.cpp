// diffusion_preprocessor.cpp — Preprocessor weight parsing for diffusion pipelines.
// Extracted from diffusion_backend_base.cpp during TrtModule migration.

#include "runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h"
#include "runtime/domains/diffusion/diffusion_types.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace trtmc {

namespace {

void load_preprocessor_weights(const std::string& index_json, const char* blob,
                               std::size_t blob_size, PreprocessorWeights& w) {
    diffusion::load_with_fallback(index_json, blob, blob_size, "patch_embedding.weight",
                                  "x_embedder.weight", w.patch_embed_weight);
    diffusion::load_with_fallback(index_json, blob, blob_size, "patch_embedding.bias",
                                  "x_embedder.bias", w.patch_embed_bias);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.time_embedding.0.weight",
        "time_text_embed.timestep_embedder.linear_1.weight", w.time_emb_0_weight);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.time_embedding.0.bias",
        "time_text_embed.timestep_embedder.linear_1.bias", w.time_emb_0_bias);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.time_embedding.2.weight",
        "time_text_embed.timestep_embedder.linear_2.weight", w.time_emb_2_weight);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.time_embedding.2.bias",
        "time_text_embed.timestep_embedder.linear_2.bias", w.time_emb_2_bias);

    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.time_proj.weight", w.time_proj_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.time_proj.bias", w.time_proj_bias);

    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.text_embedding.weight",
        "time_text_embed.text_embedder.linear_1.weight", w.text_proj_weight);
    diffusion::load_with_fallback(index_json, blob, blob_size,
                                  "condition_embedder.text_embedding.bias",
                                  "time_text_embed.text_embedder.linear_1.bias", w.text_proj_bias);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.text_embedding_2.weight",
        "time_text_embed.text_embedder.linear_2.weight", w.text_proj_2_weight);
    diffusion::load_with_fallback(
        index_json, blob, blob_size, "condition_embedder.text_embedding_2.bias",
        "time_text_embed.text_embedder.linear_2.bias", w.text_proj_2_bias);

    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "context_embedder.weight",
                                        w.context_embed_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "context_embedder.bias",
                                        w.context_embed_bias);

    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.guidance_embedding.0.weight",
                                        w.guidance_emb_0_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.guidance_embedding.0.bias",
                                        w.guidance_emb_0_bias);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.guidance_embedding.2.weight",
                                        w.guidance_emb_2_weight);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size,
                                        "condition_embedder.guidance_embedding.2.bias",
                                        w.guidance_emb_2_bias);

    // VAE BN denormalization stats
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "vae_bn.running_mean",
                                        w.vae_bn_mean);
    diffusion::load_preprocessor_floats(index_json, blob, blob_size, "vae_bn.running_var",
                                        w.vae_bn_var);
}

void finalize_preprocessor_weights(PreprocessorWeights& w) {
    if (!w.patch_embed_weight.empty() && !w.patch_embed_bias.empty()) {
        const auto dit_dim = static_cast<int32_t>(w.patch_embed_bias.size());
        w.patch_dim = static_cast<int32_t>(w.patch_embed_weight.size()) / dit_dim;
    }
    w.valid = !w.patch_embed_weight.empty() && !w.time_emb_0_weight.empty();
}

} // namespace

PreprocessorWeights parse_preprocessor_weights(const std::vector<char>& data) {
    PreprocessorWeights w;
    std::string index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    if (!diffusion::extract_preprocessor_index(data, index_json, blob, blob_size)) {
        return w;
    }

    load_preprocessor_weights(index_json, blob, blob_size, w);
    finalize_preprocessor_weights(w);

    std::cerr << "[diffusion] Preprocessor weights loaded: " << (w.valid ? "OK" : "INCOMPLETE")
              << " (patch_dim=" << w.patch_dim << ")\n";
    return w;
}

} // namespace trtmc

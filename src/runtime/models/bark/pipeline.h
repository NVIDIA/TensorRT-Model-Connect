#pragma once

// BarkPipeline: text-to-audio pipeline with semantic, coarse, fine, and codec stages.
// Uses TrtModule(semantic) + TrtModule(coarse) + TrtModule(codec) + TrtModule(fine) +
// KvCaches + embeddings.

#include "runtime/models/bark/bark_config.h"
#include "runtime/models/bark/inference_state.h"
#include "runtime/models/bark/kv_cache.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trtmc {

class BarkPipeline final : public IPipeline {
  public:
    BarkPipeline(std::unique_ptr<TrtModule> semantic, std::unique_ptr<TrtModule> coarse,
                 std::unique_ptr<BarkInferenceState> semantic_state,
                 std::unique_ptr<BarkInferenceState> coarse_state,
                 std::vector<float> semantic_embed, std::vector<float> coarse_embed,
                 BarkConfig config, cudaStream_t stream,
                 std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~BarkPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "BarkPipeline"; }

    void set_codec_module(std::unique_ptr<TrtModule> codec);
    void set_fine_module(std::unique_ptr<TrtModule> fine);
    void set_fine_embeddings(std::vector<float> embed, std::vector<float> pos_embed);

  private:
    std::vector<int32_t> run_semantic(const std::vector<int32_t>& text_ids, int32_t max_tokens);
    std::vector<int32_t> run_coarse(const std::vector<int32_t>& semantic_tokens);
    std::vector<int32_t> run_fine(const std::vector<int32_t>& coarse_tokens);
    std::vector<float> run_codec(const std::vector<int32_t>& coarse_tokens);
    std::vector<float> run_codec(const std::vector<int32_t>& codes_flat, int32_t n_frames);

    void run_step_with_embed(TrtModule& module, BarkInferenceState& state, const float* embed,
                             int32_t embed_dim, std::vector<float>& logits);
    void run_step_with_token(TrtModule& module, BarkInferenceState& state, int32_t token_id,
                             std::vector<float>& logits);
    int32_t sample_top_k(const float* logits, int32_t vocab_size, float temperature, int32_t top_k);

    std::unique_ptr<TrtModule> semantic_;
    std::unique_ptr<TrtModule> coarse_;
    std::unique_ptr<TrtModule> codec_;
    std::unique_ptr<TrtModule> fine_;
    std::unique_ptr<BarkInferenceState> semantic_state_;
    std::unique_ptr<BarkInferenceState> coarse_state_;
    std::vector<float> semantic_embed_;
    std::vector<float> coarse_embed_;
    std::vector<float> fine_embed_;
    std::vector<float> fine_position_embed_;
    BarkConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace trtmc

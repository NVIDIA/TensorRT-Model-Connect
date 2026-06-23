#pragma once

// OmniPipeline: omni multimodal pipeline with thinker + talker + code2wav.
// Uses TrtModule(thinker) + KvCache + TrtModule(talker) + KvCache + TrtModule(code2wav).

#include "runtime/models/omni/omni_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class OmniPipeline final : public IPipeline {
  public:
    OmniPipeline(std::unique_ptr<TrtModule> thinker, std::unique_ptr<IInferenceState> thinker_state,
                 std::unique_ptr<TrtModule> talker, std::unique_ptr<IInferenceState> talker_state,
                 std::unique_ptr<TrtModule> code2wav, OmniConfig config, cudaStream_t stream,
                 std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~OmniPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OmniPipeline"; }

  private:
    void run_thinker_step(int32_t token_id, std::vector<float>& logits,
                          std::vector<float>* hidden_state = nullptr);
    void run_talker_embed_step(const float* embed_ptr, int32_t embed_size,
                               std::vector<float>& logits);
    std::vector<int32_t> run_thinker(const std::vector<int32_t>& input_ids, int32_t max_tokens,
                                     std::vector<float>& hidden_states_out);
    std::vector<int32_t> run_talker(const std::vector<float>& hidden_states, int32_t num_tokens);
    std::vector<float> run_code2wav(const std::vector<int32_t>& codec_tokens, int32_t n_codebooks,
                                    int32_t n_frames);

    std::unique_ptr<TrtModule> thinker_;
    std::unique_ptr<IInferenceState> thinker_state_;
    std::unique_ptr<TrtModule> talker_;
    std::unique_ptr<IInferenceState> talker_state_;
    std::unique_ptr<TrtModule> code2wav_;
    std::unique_ptr<OmniConfig> config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc

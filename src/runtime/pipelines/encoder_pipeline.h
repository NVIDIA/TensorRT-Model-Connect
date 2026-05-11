#pragma once

// EncoderPipeline: single-pass encoder models (BERT, embedding, reranking).

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class EncoderPipeline final : public IPipeline {
  public:
    EncoderPipeline(std::unique_ptr<TrtModule> encoder, std::string mode,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "",
                    int32_t pad_token_id = 0);

    EmbeddingResult embed(const std::string& text) override;
    EmbeddingResult encode(const std::string& text) override;
    float rerank(const std::string& query, const std::string& document) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "EncoderPipeline"; }

    // Token-ID-based encoding (for unit tests and internal callers).
    EmbeddingResult encode_ids(const std::vector<int32_t>& input_ids);

  private:
    std::unique_ptr<TrtModule> encoder_;
    std::string mode_; // "encoder_only", "embedding", "reranking"
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    int32_t pad_token_id_;
};

} // namespace trtmc

#include "runtime/pipelines/encoder_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace trtmc {

namespace {

// Infer the hidden dimension from the last axis of the first output tensor.
int32_t infer_output_hidden_dim(const TrtModule& module) {
    for (const auto& info : module.output_info())
        if (!info.shape.empty())
            return static_cast<int32_t>(info.shape.back());
    return 0;
}

// Mean-pool [seq_len, hidden] over the first actual_len positions,
// then L2-normalize. Returns the pooled vector of size hidden.
std::vector<float> mean_pool_and_normalize(const float* data, int32_t actual_len, int32_t hidden) {
    std::vector<float> pooled(static_cast<std::size_t>(hidden), 0.0f);
    const float inv_len = 1.0f / static_cast<float>(actual_len);
    for (int32_t s = 0; s < actual_len; ++s)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] += data[static_cast<std::size_t>(s) * hidden + h];
    for (int32_t h = 0; h < hidden; ++h)
        pooled[h] *= inv_len;

    float norm = 0.0f;
    for (int32_t h = 0; h < hidden; ++h)
        norm += pooled[h] * pooled[h];
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] /= norm;

    return pooled;
}

// Check whether the engine's attention_mask input expects int32.
bool engine_mask_is_int32(const TrtModule& module) {
    for (const auto& info : module.input_info())
        if (info.name == "attention_mask")
            return info.dtype == DType::kInt32;
    return false;
}

int32_t infer_input_length(const TrtModule& module, const std::string& input_name) {
    for (const auto& info : module.input_info()) {
        if (info.name != input_name || info.shape.empty())
            continue;
        const int64_t len = info.shape.back();
        if (len > 0)
            return static_cast<int32_t>(len);
    }
    return 0;
}

bool model_needs_fnet_static_padding(const std::string& model_id) {
    return model_id.find("fnet") != std::string::npos || model_id.find("FNet") != std::string::npos;
}

bool is_encoder_output_name(const std::string& name) {
    return name.find("logits") != std::string::npos || name.find("embed") != std::string::npos ||
           name.find("output") != std::string::npos || name.find("hidden") != std::string::npos ||
           name.find("score") != std::string::npos;
}

} // namespace

// ─── EncoderPipeline ───

EncoderPipeline::EncoderPipeline(std::unique_ptr<TrtModule> encoder, std::string mode,
                                 std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                                 int32_t pad_token_id)
    : encoder_(std::move(encoder)), mode_(std::move(mode)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), pad_token_id_(pad_token_id) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EncoderPipeline: invalid encoder module");
}

EmbeddingResult EncoderPipeline::embed(const std::string& text) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // For embedding models: the TRT engine returns [max_seq, hidden] hidden
    // states. Mean-pool over actual input positions and L2-normalize.
    if (mode_ != "embedding" || raw.data.empty())
        return raw;

    const auto actual_len = static_cast<int32_t>(ids.size());
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden <= 0 || actual_len <= 0 || raw.dim < actual_len * hidden)
        return raw;

    auto pooled = mean_pool_and_normalize(raw.data.data(), actual_len, hidden);
    raw.data = std::move(pooled);
    raw.dim = hidden;
    return raw;
}

EmbeddingResult EncoderPipeline::encode(const std::string& text) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // Extract CLS token (first hidden_dim values) from the full hidden state
    // matrix [max_seq, hidden]. This matches HF model.encode() behavior for
    // encoder-only models (BERT, RoBERTa, etc.).
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden > 0 && raw.dim > hidden) {
        raw.data.resize(static_cast<std::size_t>(hidden));
        raw.dim = hidden;
    }
    return raw;
}

float EncoderPipeline::rerank(const std::string& query, const std::string& document) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    // Match the text-only reranking template documented by the supported
    // Nemotron rerank cross-encoder model card.
    std::string combined = "question:" + query + "   passage:" + document;
    auto ids = tokenizer_->encode(combined);
    auto result = encode_ids(ids);
    return result.data.empty() ? 0.0f : result.data[0];
}

EmbeddingResult EncoderPipeline::encode_ids(const std::vector<int32_t>& input_ids) {
    const int32_t engine_len =
        model_needs_fnet_static_padding(model_id_) ? infer_input_length(*encoder_, "input_ids") : 0;
    const auto target_len =
        engine_len > 0 ? static_cast<std::size_t>(engine_len) : input_ids.size();
    const auto valid_len = std::min(input_ids.size(), target_len);

    std::vector<int32_t> mask_i32(target_len, 0);
    std::vector<float> mask_f32(target_len, 0.0f);
    for (std::size_t i = 0; i < valid_len; ++i) {
        mask_i32[i] = 1;
        mask_f32[i] = 1.0f;
    }

    std::vector<int32_t> ids_copy(target_len, pad_token_id_);
    std::copy_n(input_ids.begin(), valid_len, ids_copy.begin());
    Tensor ids_t;
    ids_t.data = ids_copy.data();
    ids_t.shape = {static_cast<int64_t>(target_len)};
    ids_t.dtype = DType::kInt32;

    // Match the engine's expected dtype for the attention mask.
    Tensor mask_t;
    if (engine_mask_is_int32(*encoder_)) {
        mask_t.data = mask_i32.data();
        mask_t.shape = {static_cast<int64_t>(target_len)};
        mask_t.dtype = DType::kInt32;
    } else {
        mask_t.data = mask_f32.data();
        mask_t.shape = {static_cast<int64_t>(target_len)};
        mask_t.dtype = DType::kFloat32;
    }

    TensorMap inputs;
    inputs["input_ids"] = ids_t;
    inputs["attention_mask"] = mask_t;

    auto outputs = encoder_->forward(inputs);

    EmbeddingResult result;
    for (auto& [name, tensor] : outputs) {
        if (is_encoder_output_name(name)) {
            auto n = tensor.numel();
            result.data.resize(static_cast<std::size_t>(n));
            std::memcpy(result.data.data(), tensor.data, n * sizeof(float));
            result.dim = static_cast<int32_t>(n);
            break;
        }
    }

    return result;
}

} // namespace trtmc

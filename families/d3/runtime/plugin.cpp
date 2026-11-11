/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/d3/runtime/batch.h"
#include "families/d3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
namespace trtmc::d3 {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("D3 CUDA memory/stream operation failed: ") +
                                 cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value{nullptr};
    Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
std::vector<uint16_t> copy_hidden(const Tensor& tensor, std::size_t count) {
    if (!tensor.data || tensor.dtype != DType::kBFloat16 || tensor.numel() != count)
        throw std::runtime_error("D3 engine returned incompatible BF16 hidden state");
    const auto* data = static_cast<const uint16_t*>(tensor.data);
    return {data, data + count};
}
DeviceTensor device(std::vector<int64_t> shape, DType dtype, cudaStream_t stream) {
    DeviceTensor result(std::move(shape), dtype, stream);
    if (!result.ok())
        throw std::runtime_error("D3 device allocation failed");
    return result;
}
void upload(DeviceTensor& target, const void* source) {
    if (!target.copy_from_host(source))
        throw std::runtime_error("D3 input upload failed");
}
class Pipeline final : public IStructuredDecision, public IBatchStructuredDecision {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto bytes = context.reader.read_section("runtime.json");
        config_ = Record::parse(bytes.begin(), bytes.end());
        const auto version = config_.at("version").get<int32_t>();
        if ((version != 2 && version != 3) || config_.at("precision") != "bf16")
            throw std::invalid_argument("D3 runtime requires a version-2/3 BF16 bundle");
        const auto& text = config_.at("config").at("text_config");
        hidden_size_ = text.at("hidden_size").get<int32_t>();
        value_heads_ = text.at("linear_num_value_heads").get<int32_t>();
        key_dim_ = text.at("linear_key_head_dim").get<int32_t>();
        value_dim_ = text.at("linear_value_head_dim").get<int32_t>();
        conv_size_ = text.at("linear_conv_kernel_dim").get<int32_t>();
        conv_width_ = 2 * text.at("linear_num_key_heads").get<int32_t>() * key_dim_ +
                      value_heads_ * value_dim_;
        if (hidden_size_ <= 0 || value_heads_ <= 0 || key_dim_ <= 0 || value_dim_ <= 0 ||
            conv_size_ <= 0)
            throw std::invalid_argument("invalid D3 text dimensions");
        if (version == 2 &&
            (hidden_size_ != 1024 || text.at("num_hidden_layers") != 12 || value_heads_ != 16 ||
             key_dim_ != 128 || value_dim_ != 128 || conv_width_ != 6144 || conv_size_ != 4))
            throw std::invalid_argument("version-2 D3 bundles require the edge geometry");
        model_name_ = version == 3 ? config_.at("model_name").get<std::string>() : "d3-edge";
        maximum_ = config_.at("max_sequence_length").get<int32_t>();
        max_batch_ = config_.at("max_batch_size").get<int32_t>();
        height_ = config_.at("image_height").get<int32_t>();
        width_ = config_.at("image_width").get<int32_t>();
        if (maximum_ < 1 || max_batch_ < 1 || max_batch_ > 8 || height_ < 32 || width_ < 32 ||
            height_ % 32 || width_ % 32)
            throw std::invalid_argument("invalid D3 bundle capacities");
        temperature_ = config_.at("decision").at("temperature").get<float>();
        if (!std::isfinite(temperature_) || temperature_ <= 0)
            throw std::invalid_argument("invalid D3 bundle temperature");
        codes_ = config_.at("decision").at("codes").get<std::vector<std::string>>();
        if (codes_.size() != 255)
            throw std::invalid_argument("D3 bundle must contain 255 answer codes");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        const auto module = [&](const std::string& section) {
            const auto plan = context.reader.read_section(section);
            ModuleCreateOptions options;
            options.stream = stream_.value;
            auto result = context.backend.create_module(plan.data(), plan.size(), options);
            if (!result || !result->ok())
                throw std::runtime_error("cannot load D3 plan: " + section);
            return result;
        };
        embedding_ = module("embedding.plan");
        readout_ = module("readout.plan");
        vision_ = module("vision.plan");
        for (int32_t i = 0; i < text.at("num_hidden_layers").get<int32_t>(); ++i)
            layers_.push_back(module("layer_" + std::to_string(i) + ".plan"));
    }
    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        return decide_batch({request}).front();
    }
    std::vector<StructuredDecisionResult>
    decide_batch(const std::vector<StructuredDecisionRequest>& requests) override {
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<StructuredDecisionResult> results(requests.size());
        if (requests.empty())
            return results;
        std::vector<Record> answers(requests.size(), Record::object());
        std::vector<BatchRow> rows;
        std::vector<VisualInputs> images;
        std::vector<std::size_t> image_owners;
        const auto& visual = config_.at("config").at("vision_config");
        const int32_t visual_tokens = height_ / 32 * (width_ / 32);
        const int32_t marker = config_.at("config").at("image_token_id").get<int32_t>();
        // Validate and tokenize the whole call before any engine is executed.
        for (std::size_t owner = 0; owner < requests.size(); ++owner) {
            const auto& request = requests[owner];
            if (!request.videos.empty() || request.max_state_tokens != -1 ||
                request.images.size() > 1)
                throw std::invalid_argument(
                    "D3 supports text/one image without video or state truncation");
            const auto record = Record::parse(request.document);
            const auto questions_for_request = questions(record);
            std::string media;
            if (!request.images.empty()) {
                images.push_back(prepare_image(request.images.front(), height_, width_,
                                               visual.at("num_position_embeddings").get<int32_t>(),
                                               visual.at("hidden_size").get<int32_t>() /
                                                   visual.at("num_heads").get<int32_t>()));
                image_owners.push_back(owner);
                media = "<|vision_start|>";
                for (int32_t i = 0; i < visual_tokens; ++i)
                    media += "<|image_pad|>";
                media += "<|vision_end|>";
            }
            for (const auto& question : questions_for_request) {
                auto ids = tokenizer_->encode(
                    prompt(record.value("state", Record()), question, codes_, media));
                if (ids.empty() || ids.size() > static_cast<std::size_t>(maximum_))
                    throw std::invalid_argument(
                        "D3 prompt exceeds bundle capacity; nothing was truncated");
                auto angles =
                    rope_angles(ids, marker, height_ / 32, width_ / 32, request.images.size());
                rows.push_back({owner, question, std::move(ids), std::move(angles)});
            }
        }
        std::vector<std::vector<uint16_t>> features(requests.size());
        for (std::size_t begin = 0; begin < images.size(); begin += max_batch_) {
            const auto end = std::min(images.size(), begin + max_batch_);
            auto input = pack_visual(images, begin, end);
            const auto out = vision_->forward(input.tensors());
            const auto values = copy_hidden(out.at("visual_features"),
                                            (end - begin) * visual_tokens * hidden_size_);
            const std::size_t stride = static_cast<std::size_t>(visual_tokens) * hidden_size_;
            for (auto i = begin; i < end; ++i)
                features[image_owners[i]] = {values.begin() + (i - begin) * stride,
                                             values.begin() + (i - begin + 1) * stride};
        }
        for (std::size_t begin = 0; begin < rows.size(); begin += max_batch_) {
            const auto end = std::min(rows.size(), begin + max_batch_);
            auto packed = pack_rows(rows, begin, end);
            const auto logits = execute(packed, rows, begin, features, marker);
            for (auto i = begin; i < end; ++i) {
                const auto& row = rows[i];
                DecisionScores scores;
                scores.question_id = row.question.id;
                scores.option_ids = row.question.keys;
                for (std::size_t option = 0; option < row.question.keys.size(); ++option) {
                    const float value = logits[(i - begin) * 255 + option];
                    if (!std::isfinite(value))
                        throw std::runtime_error("D3 returned non-finite logits");
                    scores.logits.push_back(value / temperature_);
                }
                const auto max = *std::max_element(scores.logits.begin(), scores.logits.end());
                float total = 0;
                for (const float value : scores.logits) {
                    scores.probabilities.push_back(std::exp(value - max));
                    total += scores.probabilities.back();
                }
                for (auto& value : scores.probabilities)
                    value /= total;
                answers[row.owner][row.question.id] = answer(row.question, scores.probabilities);
                results[row.owner].scores.push_back(std::move(scores));
                results[row.owner].input_tokens += static_cast<int32_t>(row.ids.size());
            }
        }
        for (std::size_t i = 0; i < results.size(); ++i)
            results[i].document =
                Record{{"model", model_name_},
                       {"answers", answers[i]},
                       {"usage", {{"input_tokens", results[i].input_tokens}, {"output_tokens", 0}}}}
                    .dump();
        return results;
    }

  private:
    std::vector<float> execute(PackedRows& input, const std::vector<BatchRow>& rows,
                               std::size_t begin,
                               const std::vector<std::vector<uint16_t>>& features, int32_t marker) {
        const auto batch = input.batch, sequence = input.sequence;
        const std::size_t row_bytes = hidden_size_ * sizeof(uint16_t);
        const auto embedded = embedding_->forward(
            {{"input_ids", {input.ids.data(), {batch, sequence}, DType::kInt32}}});
        auto host_hidden = copy_hidden(embedded.at("hidden"), batch * sequence * hidden_size_);
        for (int64_t b = 0; b < batch; ++b) {
            const auto& feature = features[rows[begin + b].owner];
            std::size_t source = 0;
            for (int64_t t = 0; t < input.lengths[b]; ++t)
                if (input.ids[b * sequence + t] == marker) {
                    if ((source + 1) * hidden_size_ > feature.size())
                        throw std::runtime_error("D3 visual feature count does not match tokens");
                    std::copy_n(feature.data() + source++ * hidden_size_, hidden_size_,
                                host_hidden.data() + (b * sequence + t) * hidden_size_);
                }
            if (source * hidden_size_ != feature.size())
                throw std::runtime_error("D3 visual features have no matching tokens");
        }
        auto hidden = device({batch, sequence, hidden_size_}, DType::kBFloat16, stream_.value);
        auto angles = device({batch, sequence, 32}, DType::kFloat32, stream_.value);
        auto mask = device({batch, sequence}, DType::kFloat32, stream_.value);
        upload(hidden, host_hidden.data());
        upload(angles, input.angles.data());
        upload(mask, input.mask.data());
        const auto& types = config_.at("config").at("text_config").at("layer_types");
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            auto next = device({batch, sequence, hidden_size_}, DType::kBFloat16, stream_.value);
            auto& engine = *layers_[layer];
            if (types.at(layer) == "linear_attention") {
                auto step = device({batch, hidden_size_}, DType::kBFloat16, stream_.value);
                auto conv =
                    device({batch, conv_width_, conv_size_}, DType::kBFloat16, stream_.value);
                auto state = device({batch, value_heads_, key_dim_, value_dim_}, DType::kFloat32,
                                    stream_.value);
                checked(cudaMemsetAsync(conv.data(), 0, conv.nbytes(), stream_.value));
                checked(cudaMemsetAsync(state.data(), 0, state.nbytes(), stream_.value));
                for (int64_t token = 0; token < sequence; ++token) {
                    checked(cudaMemcpy2DAsync(step.data(), row_bytes,
                                              static_cast<const char*>(hidden.data()) +
                                                  token * row_bytes,
                                              sequence * row_bytes, row_bytes, batch,
                                              cudaMemcpyDeviceToDevice, stream_.value));
                    engine.forward_device_async(
                        {{"hidden", &step}, {"conv", &conv}, {"state", &state}});
                    checked(cudaMemcpy2DAsync(static_cast<char*>(next.data()) + token * row_bytes,
                                              sequence * row_bytes, engine.device_ptr("hidden_out"),
                                              row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice,
                                              stream_.value));
                    checked(cudaMemcpyAsync(conv.data(), engine.device_ptr("conv_out"),
                                            conv.nbytes(), cudaMemcpyDeviceToDevice,
                                            stream_.value));
                    checked(cudaMemcpyAsync(state.data(), engine.device_ptr("state_out"),
                                            state.nbytes(), cudaMemcpyDeviceToDevice,
                                            stream_.value));
                }
                checked(cudaStreamSynchronize(stream_.value));
            } else {
                engine.forward_device_async(
                    {{"hidden", &hidden}, {"angles", &angles}, {"attention_mask", &mask}});
                checked(cudaMemcpyAsync(next.data(), engine.device_ptr("hidden_out"), next.nbytes(),
                                        cudaMemcpyDeviceToDevice, stream_.value));
                checked(cudaStreamSynchronize(stream_.value));
            }
            hidden = std::move(next);
        }
        auto last = device({batch, hidden_size_}, DType::kBFloat16, stream_.value);
        for (int64_t b = 0; b < batch; ++b)
            checked(cudaMemcpyAsync(static_cast<char*>(last.data()) + b * row_bytes,
                                    static_cast<const char*>(hidden.data()) +
                                        (b * sequence + input.lengths[b] - 1) * row_bytes,
                                    row_bytes, cudaMemcpyDeviceToDevice, stream_.value));
        readout_->forward_device_async({{"hidden", &last}});
        std::vector<float> logits(batch * 255);
        checked(cudaMemcpyAsync(logits.data(), readout_->device_ptr("logits"),
                                logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                stream_.value));
        checked(cudaStreamSynchronize(stream_.value));
        return logits;
    }
    int32_t hidden_size_, value_heads_, key_dim_, value_dim_, conv_size_, conv_width_;
    std::string model_name_;
    Record config_;
    int32_t maximum_, max_batch_, height_, width_;
    float temperature_;
    Stream stream_;
    std::vector<std::string> codes_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITrtModule> embedding_, readout_, vision_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::mutex mutex_;
};
} // namespace
} // namespace trtmc::d3
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("D3 does not accept a KV cache budget");
    return new trtmc::d3::Pipeline(context);
}

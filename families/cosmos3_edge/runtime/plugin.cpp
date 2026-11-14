/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "media.h"
#include "request.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "edge_llm/adapter.h"
#endif
#include "sampler.h"
#include "tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace trtmc::cosmos3_edge {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace internal;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
std::string read_text(const BundleReader& reader, const char* name) {
    auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}
std::unique_ptr<ITrtModule> load(const FamilyContext& context, const char* name) {
    auto plan = context.reader.read_section(name);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Cosmos3-Edge failed to load ") + name);
    module->set_timing_label(name);
    return module;
}
std::vector<uint8_t> pack(const float* values, size_t count, DType dtype) {
    std::vector<uint8_t> result(count * dtype_size(dtype));
    if (dtype == DType::kFloat32)
        std::memcpy(result.data(), values, result.size());
    else if (dtype == DType::kFloat16) {
        auto* output = reinterpret_cast<__half*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2half_rn(values[i]);
    } else if (dtype == DType::kBFloat16) {
        auto* output = reinterpret_cast<__nv_bfloat16*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2bfloat16_rn(values[i]);
    } else
        throw std::runtime_error("Invalid Cosmos3-Edge engine precision");
    return result;
}
std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \n\r\t\f\v");
    return begin == std::string::npos
               ? ""
               : value.substr(begin, value.find_last_not_of(" \n\r\t\f\v") - begin + 1);
}
struct VisualSpan {
    Frame frame;
    int32_t token;
    size_t offset{};
};
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<std::array<int, 3>> positions;
    std::vector<VisualSpan> visual;
    int next_position{};
};

class Pipeline final : public ITextGeneration,
                       public IVisionLanguageGeneration,
                       public IModel,
                       public ITextContinuation,
                       public IImageVideoTextToText,
                       public IImagesTextToText,
                       public IVideoTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : task_(context.reader.info().task),
          metadata_(Json::parse(read_text(context.reader, "runtime.json"))),
          config_(metadata_.at("config")), tokenizer_(read_text(context.reader, "tokenizer.json")),
          embeddings_(context.reader.read_section("embeddings.f32")),
          image_config_(Json::parse(read_text(context.reader, "preprocessor_config.json"))),
          video_config_(Json::parse(read_text(context.reader, "video_preprocessor_config.json"))) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("cosmos3_edge does not support --kv-cache-size");
        if (metadata_.at("format") != 2 || config_.at("model_type") != "cosmos3_edge")
            throw std::invalid_argument("Invalid Cosmos3-Edge reasoning bundle");
        const auto& text = config_.at("text_config");
        hidden_ = text.at("hidden_size");
        vocab_ = text.at("vocab_size");
        layers_ = text.at("num_hidden_layers");
        heads_ = text.at("num_key_value_heads");
        dim_ = text.at("head_dim");
        capacity_ = metadata_.at("max_cache_length");
        max_input_ = metadata_.at("max_input_length");
        max_patches_ = metadata_.at("max_patches");
        theta_ = text.at("rope_parameters").at("rope_theta");
        sections_ = text.at("rope_parameters").at("mrope_section").get<std::array<int, 3>>();
        patch_ = config_.at("vision_config").at("patch_size");
        merge_ = config_.at("vision_config").at("spatial_merge_size");
        table_side_ = int(std::sqrt(config_.at("vision_config").at("num_patches").get<int>()));
        eos_ = text.value("eos_token_id", 11);
        if (hidden_ <= 0 || vocab_ <= 0 || capacity_ <= 0 || dim_ <= 0 || dim_ % 2 ||
            embeddings_.size() != size_t(hidden_) * vocab_ * sizeof(float))
            throw std::invalid_argument("Invalid Cosmos3-Edge embedding table or dimensions");
        decoder_ = load(context, "decoder.plan");
        visual_ = load(context, "visual.plan");
        dtype_ = decoder_->tensor_dtype("inputs_embeds");
        if (dtype_ != visual_->tensor_dtype("pixel_values"))
            throw std::invalid_argument("Cosmos3-Edge component precisions differ");
        for (int i = 0; i < layers_; ++i)
            for (const auto* kind : {"k", "v"}) {
                std::string name = std::string("cache_") + kind + "_" + std::to_string(i);
                cache_.emplace_back(
                    DeviceTensor::zeros({heads_, capacity_, dim_}, dtype_, decoder_->stream()));
                if (!cache_.back().ok())
                    throw std::runtime_error("Cosmos3-Edge cache allocation failed");
                decoder_->bind_external(name, cache_.back().data());
            }
    }
    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override { return 128; }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextContinuation>(*this, field_view(false)),
                bind<IImagesTextToText>(*this, field_view()),
                bind<IVideoTextToText>(*this, field_view()),
                bind<IImageVideoTextToText>(*this, field_view())};
    }
    TextResult generate(const std::string& text, const TextGenerationConfig& cfg) override {
        std::vector<ImageVideoTextPart> parts{TextPartView{text}};
        ImageVideoTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, cfg);
    }
    TextResult generate(const std::string& text, const float* pixels, int32_t h, int32_t w,
                        const TextGenerationConfig& cfg) override {
        if (h <= 0 || w <= 0)
            throw std::invalid_argument("Cosmos3-Edge image dimensions must be positive");
        std::vector<ImageVideoTextPart> parts{ImageView{pixels, size_t(h) * w * 3 * sizeof(float),
                                                        uint32_t(h), uint32_t(w), 3,
                                                        ImageFormat::Float32},
                                              TextPartView{text}};
        ImageVideoTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, cfg);
    }
    TextResult run(const TextContinuationRequest& input, ConfigView supplied) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument(
                "Cosmos3-Edge continuation accepts text, not raw token IDs");
        return generate(std::string(*text), read_config(supplied, false));
    }
    TextResult run(const ImageVideoTextToTextRequest& input, ConfigView supplied) override {
        return execute(input.messages, read_config(supplied));
    }

    TextResult run(const ImagesTextToTextRequest& input, ConfigView supplied) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Cosmos3-Edge reasoning does not expose tools");
        return media_messages(input.messages, supplied);
    }
    TextResult run(const VideoTextToTextRequest& input, ConfigView supplied) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Cosmos3-Edge reasoning does not expose tools");
        return media_messages(input.messages, supplied);
    }
    template <class Part>
    TextResult media_messages(Span<const MediaMessage<Part>> input, ConfigView supplied) {
        std::vector<std::vector<ImageVideoTextPart>> storage;
        std::vector<ImageVideoTextMessage> messages;
        storage.reserve(input.size());
        messages.reserve(input.size());
        for (const auto& source : input) {
            storage.emplace_back();
            for (const auto& part : source.parts) {
                std::visit(
                    [&](const auto& value) {
                        using Value = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<Value, TextPartView> ||
                                      std::is_same_v<Value, ImageView> ||
                                      std::is_same_v<Value, VideoView>)
                            storage.back().emplace_back(value);
                        else
                            throw std::invalid_argument(
                                "Cosmos3-Edge reasoning task requires text or visual media parts");
                    },
                    part);
            }
            messages.push_back({source.role, {storage.back().data(), storage.back().size()}});
        }
        return execute({messages.data(), messages.size()}, read_config(supplied));
    }

  private:
    std::string task_;
    Json metadata_, config_;
    Tokenizer tokenizer_;
    std::vector<char> embeddings_;
    Json image_config_, video_config_;
    int hidden_{}, vocab_{}, layers_{}, heads_{}, dim_{}, capacity_{}, max_input_{}, max_patches_{};
    int patch_{}, merge_{}, table_side_{}, eos_{};
    double theta_{};
    std::array<int, 3> sections_{};
    DType dtype_{};
    // Destroy contexts before their external cache storage.
    std::vector<DeviceTensor> cache_;
    std::unique_ptr<ITrtModule> decoder_, visual_;
    std::mutex mutex_;

    std::string frame_text(Prompt& prompt, ImageView image, bool video, int frame_count) {
        const auto& processor = video ? video_config_ : image_config_;
        auto frame =
            plan_frame(image, patch_, merge_, frame_count, processor.at("size").at("shortest_edge"),
                       processor.at("size").at("longest_edge"), max_patches_);
        const int token = config_.at(video ? "video_token_id" : "image_token_id");
        prompt.visual.push_back({frame, token, 0});
        std::string result = "<|vision_start|>";
        for (int i = 0; i < frame.tokens(); ++i)
            result += video ? "<|video_pad|>" : "<|image_pad|>";
        return result + "<|vision_end|>";
    }
    std::string render_parts(Prompt& prompt, const ImageVideoTextMessage& message) {
        std::string result;
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part))
                result += text->text;
            else if (const auto* image = std::get_if<ImageView>(&part))
                result += frame_text(prompt, *image, false, 1);
            else {
                const auto& video = std::get<VideoView>(part);
                if (video.frames.empty() || video.frames.size() > size_t(capacity_) ||
                    (!video.timestamps_seconds.empty() &&
                     video.timestamps_seconds.size() != video.frames.size()))
                    throw std::invalid_argument(
                        "Cosmos3-Edge video requires frames and one timestamp per frame");
                double previous = -1;
                for (size_t i = 0; i < video.frames.size(); ++i) {
                    const double timestamp = video.timestamps_seconds.empty()
                                                 ? double(i) / 24
                                                 : video.timestamps_seconds[i];
                    if (!std::isfinite(timestamp) || timestamp < 0 || timestamp < previous ||
                        video.frames[i].height != video.frames[0].height ||
                        video.frames[i].width != video.frames[0].width)
                        throw std::invalid_argument("Cosmos3-Edge video requires ordered "
                                                    "timestamps and equal frame dimensions");
                    previous = timestamp;
                    std::ostringstream label;
                    label.imbue(std::locale::classic());
                    label << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
                    result += label.str() +
                              frame_text(prompt, video.frames[i], true, int(video.frames.size()));
                }
            }
        }
        return result;
    }
    Prompt prepare(Span<const ImageVideoTextMessage> messages, const TextGenerationConfig& cfg) {
        if (messages.empty())
            throw std::invalid_argument("Cosmos3-Edge requires a nonempty prompt");
        Prompt prompt;
        std::string text;
        size_t first = 0, last_user = 0;
        if (cfg.use_chat_template) {
            std::string system = cfg.system_prompt;
            if (messages[0].role == MessageRole::System) {
                if (!system.empty())
                    throw std::invalid_argument("Supply one Cosmos3-Edge system message, not two");
                for (const auto& part : messages[0].parts) {
                    const auto* p = std::get_if<TextPartView>(&part);
                    if (!p)
                        throw std::invalid_argument(
                            "Cosmos3-Edge initial system message must be text");
                    system += p->text;
                }
                first = 1;
            }
            text = "\n<|im_start|>system\n" + system + "<|im_end|>\n";
        } else if (messages.size() != 1 || !cfg.system_prompt.empty())
            throw std::invalid_argument(
                "Cosmos3-Edge multi-message prompts require the chat template");
        for (size_t i = first; i < messages.size(); ++i)
            if (messages[i].role == MessageRole::User)
                last_user = i;
        for (size_t i = first; i < messages.size(); ++i) {
            const auto& message = messages[i];
            const auto visual_before = prompt.visual.size();
            auto body = render_parts(prompt, message);
            if (message.role == MessageRole::Assistant && cfg.use_chat_template) {
                if (prompt.visual.size() != visual_before)
                    throw std::invalid_argument("Cosmos3-Edge assistant history must contain text");
                if (body.find("<think>") == std::string::npos &&
                    body.find("</think>") == std::string::npos)
                    body = "<think></think>" + body;
                if (i < last_user && body.find("<think>") != std::string::npos &&
                    body.rfind("</think>") != std::string::npos)
                    body = "<think></think>" + body.substr(body.rfind("</think>") + 8);
                body = trim(body);
            }
            text += cfg.use_chat_template
                        ? "<|im_start|>" + role_name(message.role) + "\n" + body + "<|im_end|>\n"
                        : body;
        }
        if (cfg.use_chat_template)
            text += cfg.enable_thinking ? "<|im_start|>assistant\n<think>\n"
                                        : "<|im_start|>assistant\n<think></think>";
        prompt.ids = tokenizer_.encode(text);
        if (prompt.ids.empty() || prompt.ids.size() > size_t(max_input_) ||
            prompt.ids.size() > size_t(capacity_) ||
            size_t(cfg.max_new_tokens - 1) > size_t(capacity_) - prompt.ids.size())
            throw std::invalid_argument(
                "Cosmos3-Edge prompt and output budget exceed the built decoder capacity");
        size_t span = 0;
        int position = 0;
        for (size_t i = 0; i < prompt.ids.size();) {
            const int32_t id = prompt.ids[i];
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("Cosmos3-Edge prompt token is outside vocabulary");
            if (id != config_.at("image_token_id").get<int>() &&
                id != config_.at("video_token_id").get<int>()) {
                prompt.positions.push_back({position, position, position});
                ++position;
                ++i;
                continue;
            }
            if (span == prompt.visual.size() || prompt.visual[span].token != id)
                throw std::invalid_argument(
                    "Cosmos3-Edge media placeholders do not match the supplied media");
            auto& entry = prompt.visual[span++];
            entry.offset = i;
            const int h = entry.frame.rows() / merge_, w = entry.frame.cols() / merge_;
            if (i + size_t(h) * w > prompt.ids.size())
                throw std::invalid_argument("Truncated Cosmos3-Edge media span");
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x, ++i) {
                    if (prompt.ids[i] != id)
                        throw std::invalid_argument("Invalid Cosmos3-Edge media span");
                    prompt.positions.push_back({position, position + y, position + x});
                }
            position += std::max(h, w);
        }
        if (span != prompt.visual.size())
            throw std::invalid_argument("Cosmos3-Edge unused media input");
        prompt.next_position = position;
        return prompt;
    }
    std::vector<uint8_t> embed(const std::vector<int32_t>& ids) {
        std::vector<float> values(size_t(ids.size()) * hidden_);
        for (size_t i = 0; i < ids.size(); ++i)
            std::memcpy(values.data() + i * hidden_,
                        embeddings_.data() + size_t(ids[i]) * hidden_ * sizeof(float),
                        hidden_ * sizeof(float));
        return pack(values.data(), values.size(), dtype_);
    }
    void insert_visual(const VisualSpan& span, std::vector<uint8_t>& embeddings) {
        auto pixels = patchify(span.frame);
        auto packed = pack(pixels.data(), pixels.size(), dtype_);
        std::vector<float> weights;
        interpolate_positions(span.frame, table_side_, weights);
        const int patches = span.frame.rows() * span.frame.cols();
        TensorMap input{
            {"pixel_values", {packed.data(), {patches, 3 * patch_ * patch_}, dtype_}},
            {"position_weights",
             {weights.data(), {patches, table_side_ * table_side_}, DType::kFloat32}},
        };
        const auto output = visual_->forward(input).at("image_features");
        const size_t bytes = size_t(span.frame.tokens()) * hidden_ * dtype_size(dtype_);
        if (output.dtype != dtype_ || output.nbytes() != bytes)
            throw std::runtime_error("Cosmos3-Edge visual output shape mismatch");
        std::memcpy(embeddings.data() + span.offset * hidden_ * dtype_size(dtype_), output.data,
                    bytes);
    }
    std::vector<float> step(std::vector<uint8_t>& embeddings,
                            const std::vector<std::array<int, 3>>& positions, int offset) {
        const int count = int(positions.size());
        std::vector<float> cosine(size_t(count) * dim_), sine(cosine.size());
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < dim_ / 2; ++j) {
                int axis = 0;
                if (j % 3 == 1 && j < sections_[1] * 3)
                    axis = 1;
                if (j % 3 == 2 && j < sections_[2] * 3)
                    axis = 2;
                const float phase =
                    float(positions[i][axis]) * float(std::pow(theta_, -2.0 * j / dim_));
                for (int half = 0; half < 2; ++half) {
                    cosine[size_t(i) * dim_ + j + half * (dim_ / 2)] = std::cos(phase);
                    sine[size_t(i) * dim_ + j + half * (dim_ / 2)] = std::sin(phase);
                }
            }
        const int key_length = capacity_ + count;
        std::vector<float> mask(size_t(count) * key_length, std::numeric_limits<float>::lowest());
        for (int i = 0; i < count; ++i) {
            std::fill_n(mask.data() + size_t(i) * key_length, offset, 0.0F);
            std::fill_n(mask.data() + size_t(i) * key_length + capacity_, i + 1, 0.0F);
        }
        TensorMap input{
            {"inputs_embeds", {embeddings.data(), {count, hidden_}, dtype_}},
            {"rope_cos", {cosine.data(), {count, dim_}, DType::kFloat32}},
            {"rope_sin", {sine.data(), {count, dim_}, DType::kFloat32}},
            {"attention_mask", {mask.data(), {1, count, key_length}, DType::kFloat32}},
        };
        decoder_->forward_async(input);
        std::vector<float> logits(vocab_);
        cuda_check(cudaMemcpyAsync(
            logits.data(),
            static_cast<float*>(decoder_->device_ptr("logits")) + size_t(count - 1) * vocab_,
            logits.size() * sizeof(float), cudaMemcpyDeviceToHost, decoder_->stream()));
        const size_t row = size_t(dim_) * dtype_size(dtype_);
        for (int i = 0; i < layers_; ++i)
            for (int kind = 0; kind < 2; ++kind) {
                const auto name =
                    std::string("present_") + (kind ? "v_" : "k_") + std::to_string(i);
                cuda_check(cudaMemcpy2DAsync(
                    static_cast<uint8_t*>(cache_[2 * i + kind].data()) + offset * row,
                    capacity_ * row, decoder_->device_ptr(name), count * row, count * row, heads_,
                    cudaMemcpyDeviceToDevice, decoder_->stream()));
            }
        decoder_->sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Cosmos3-Edge decoder produced nonfinite logits");
        return logits;
    }
    TextResult execute(Span<const ImageVideoTextMessage> messages,
                       const TextGenerationConfig& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto setup = Clock::now();
        validate_request(cfg);
        if (cfg.eos_token_id >= vocab_)
            throw std::invalid_argument("Cosmos3-Edge EOS override is outside vocabulary");
        auto prompt = prepare(messages, cfg);
        auto embeddings = embed(prompt.ids);
        for (const auto& entry : prompt.visual)
            insert_visual(entry, embeddings);
        for (auto& tensor : cache_)
            cuda_check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), decoder_->stream()));
        auto params = cosmos3_edge_sampling_params_from_config(cfg, eos_);
        auto sampler = create_cosmos3_edge_sampler(params);
        TextResult result;
        result.setup_ms = elapsed(setup);
        const auto prefill = Clock::now();
        auto logits = step(embeddings, prompt.positions, 0);
        result.prefill_ms = elapsed(prefill);
        const auto decode = Clock::now();
        std::vector<bool> seen(vocab_);
        for (const int id : prompt.ids)
            seen[id] = true;
        for (int i = 0; i < cfg.max_new_tokens; ++i) {
            if (cfg.repetition_penalty != 1.0F)
                for (int id = 0; id < vocab_; ++id)
                    if (seen[id])
                        logits[id] = logits[id] < 0 ? logits[id] * cfg.repetition_penalty
                                                    : logits[id] / cfg.repetition_penalty;
            const auto sample = sampler->sample(logits.data(), vocab_, params);
            result.token_ids.push_back(sample.token_id);
            seen[sample.token_id] = true;
            if (sample.is_eos || i + 1 == cfg.max_new_tokens)
                break;
            embeddings = embed({sample.token_id});
            const int position = prompt.next_position + i;
            logits = step(embeddings, {{position, position, position}}, int(prompt.ids.size()) + i);
        }
        result.decode_ms = elapsed(decode);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }
};
} // namespace
} // namespace trtmc::cosmos3_edge

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_HAS_EDGE_LLM
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("cosmos3_edge does not support --kv-cache-size");
        return trtmc::cosmos3_edge::edge_llm::create(context.reader);
#else
        throw std::runtime_error(
            "This Cosmos3-Edge runtime was built without the optional Edge SDK");
#endif
    }
    return new trtmc::cosmos3_edge::Pipeline(context);
}

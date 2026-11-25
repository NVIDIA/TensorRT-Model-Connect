/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "contract.h"
#include "json.h"
#include "media.h"
#include "request.h"
#include "tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>

namespace trtmc::hunyuan_vl {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
std::string read_text(const BundleReader& reader, const char* name) {
    const auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}
std::unique_ptr<ITrtModule> load(const FamilyContext& context, const char* name) {
    const auto plan = context.reader.read_section(name);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("HunYuanVL failed to load ") + name);
    module->set_timing_label(name);
    return module;
}
// The loaded engine's tensors in the form check_engine reads (contract.h).
std::vector<EngineTensor> engine_tensors(const ITrtModule& module) {
    if (module.optimization_profile_count() < 1)
        throw std::invalid_argument("Invalid HunYuanVL bundle: an engine has no profile");
    std::vector<EngineTensor> result;
    for (const auto& info : module.input_info()) {
        EngineTensor tensor{info.name,  info.dtype, true, module.input_is_dynamic(info.name),
                            info.shape, {},         {}};
        if (tensor.dynamic) {
            tensor.profile_min =
                module.input_profile_shape(info.name, 0, ProfileShapeSelector::kMin);
            tensor.profile_max =
                module.input_profile_shape(info.name, 0, ProfileShapeSelector::kMax);
        }
        result.push_back(std::move(tensor));
    }
    for (const auto& info : module.output_info())
        result.push_back({info.name, info.dtype, false, false, info.shape, {}, {}});
    return result;
}
std::vector<uint8_t> pack(const float* values, size_t count, DType dtype) {
    std::vector<uint8_t> result(count * dtype_size(dtype));
    if (dtype == DType::kFloat32)
        std::memcpy(result.data(), values, result.size());
    else if (dtype == DType::kBFloat16) {
        auto* output = reinterpret_cast<__nv_bfloat16*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2bfloat16_rn(values[i]);
    } else
        throw std::runtime_error("Invalid HunYuanVL engine precision");
    return result;
}
// The reference casts RoPE cosines and sines to the activation dtype before applying them.
float round_to(float value, DType dtype) {
    return dtype == DType::kBFloat16 ? __bfloat162float(__float2bfloat16_rn(value)) : value;
}

struct ImageSpan {
    ImagePlan plan;
    size_t offset{}; // first placeholder token
};
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<ImageSpan> images;
};

class Pipeline final : public IModel, public IImagesTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : task_(context.reader.info().task),
          tokenizer_(read_text(context.reader, "tokenizer.json")),
          embeddings_(context.reader.read_section("embeddings.f32")) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("hunyuan_vl does not support --kv-cache-size");
        if (task_ != IImagesTextToText::kTask)
            throw std::invalid_argument("Invalid HunYuanVL bundle");
        const auto metadata =
            parse_metadata(parse_json(read_text(context.reader, "runtime.json"), "runtime.json"),
                           embeddings_.size());
        hidden_ = metadata.hidden;
        layers_ = metadata.layers;
        kv_heads_ = metadata.kv_heads;
        head_dim_ = metadata.head_dim;
        vocab_ = metadata.vocab;
        capacity_ = metadata.capacity;
        max_input_ = metadata.max_input;
        max_patches_ = metadata.max_patches;
        edge_ = metadata.edge;
        inverse_frequencies_ = metadata.inverse_frequencies;
        sections_ = metadata.sections;
        settings_ = metadata.image;
        bos_ = metadata.bos;
        system_end_ = metadata.system_end;
        user_ = metadata.user;
        assistant_ = metadata.assistant;
        image_ = metadata.image_token;
        image_start_ = metadata.image_start;
        image_end_ = metadata.image_end;
        eos_.insert(metadata.eos.begin(), metadata.eos.end());
        decoder_ = load(context, "decoder.plan");
        visual_ = load(context, "visual.plan");
        check_engine("decoder.plan", engine_tensors(*decoder_), decoder_contract(metadata));
        check_engine("visual.plan", engine_tensors(*visual_), visual_contract(metadata));
        dtype_ = metadata.dtype;
        for (int i = 0; i < layers_; ++i)
            for (const auto* kind : {"k", "v"}) {
                cache_.emplace_back(DeviceTensor::zeros({kv_heads_, capacity_, head_dim_}, dtype_,
                                                        decoder_->stream()));
                if (!cache_.back().ok())
                    throw std::runtime_error("HunYuanVL cache allocation failed");
                decoder_->bind_external(std::string("cache_") + kind + "_" + std::to_string(i),
                                        cache_.back().data());
            }
    }
    const char* task() const noexcept override { return task_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<IImagesTextToText>(*this, generation_fields())};
    }
    TextResult run(const ImagesTextToTextRequest& input, ConfigView supplied) override {
        if (!input.tools.empty())
            throw std::invalid_argument("HunYuanVL does not expose tools");
        const auto generation = read_generation(supplied);
        std::lock_guard<std::mutex> lock(mutex_);
        return execute(input.messages, generation);
    }

  private:
    std::string task_;
    Tokenizer tokenizer_;
    std::vector<char> embeddings_;
    ImageSettings settings_;
    std::vector<float> inverse_frequencies_;
    std::vector<int> sections_;
    std::unordered_set<int32_t> eos_;
    int hidden_{}, layers_{}, kv_heads_{}, head_dim_{}, vocab_{}, capacity_{}, max_input_{};
    int max_patches_{}, edge_{};
    int32_t bos_{}, system_end_{}, user_{}, assistant_{}, image_{}, image_start_{}, image_end_{};
    DType dtype_{};
    // Destroy contexts before their external cache storage.
    std::vector<DeviceTensor> cache_;
    std::unique_ptr<ITrtModule> decoder_, visual_;
    std::mutex mutex_;

    // Tokenizes within the capacity left, so an oversized text fails before or while it is
    // tokenized rather than after.
    void append_text(Prompt& prompt, const std::string& text) const {
        append_tokens(prompt.ids, tokenizer_.encode(text, size_t(capacity_) - prompt.ids.size()),
                      capacity_);
    }
    void append(Prompt& prompt, int32_t id, size_t count = 1) const {
        append_tokens(prompt.ids, count, id, capacity_);
    }
    // The parts of a text-only message, gathered within the capacity left.
    std::string text_only(const Prompt& prompt, const ImagesTextMessage& message,
                          const char* role) const {
        TextRun run(tokenizer_, size_t(capacity_) - prompt.ids.size());
        for (const auto& part : message.parts) {
            const auto* value = std::get_if<TextPartView>(&part);
            if (!value)
                throw std::invalid_argument(std::string("HunYuanVL ") + role +
                                            " messages accept only text");
            run.append(value->text);
        }
        return run.text();
    }
    // The released chat template, rendered directly as token ids around encoded text runs. The
    // prompt never grows past the decoder's capacity.
    Prompt prepare(Span<const ImagesTextMessage> messages) const {
        if (messages.empty() || messages[messages.size() - 1].role != MessageRole::User)
            throw std::invalid_argument("HunYuanVL needs a conversation ending with a user turn");
        Prompt prompt;
        append(prompt, bos_);
        size_t first = 0;
        if (messages[0].role == MessageRole::System) {
            append_text(prompt, text_only(prompt, messages[0], "system"));
            append(prompt, system_end_);
            first = 1;
        }
        for (size_t i = first; i < messages.size(); ++i) {
            const auto& message = messages[i];
            if (message.role == MessageRole::Assistant) {
                append_text(prompt, text_only(prompt, message, "assistant"));
                append(prompt, assistant_);
                continue;
            }
            if (message.role != MessageRole::User)
                throw std::invalid_argument(
                    "HunYuanVL accepts one leading system message, then user and assistant turns");
            TextRun pending(tokenizer_, size_t(capacity_) - prompt.ids.size());
            for (const auto& part : message.parts) {
                if (const auto* text = std::get_if<TextPartView>(&part)) {
                    pending.append(text->text);
                    continue;
                }
                const auto* view = std::get_if<ImageView>(&part);
                if (!view)
                    throw std::invalid_argument("HunYuanVL user turns accept text and images");
                append_text(prompt, pending.text());
                pending.restart(size_t(capacity_) - prompt.ids.size());
                const auto plan = plan_image(*view, settings_, max_patches_);
                append(prompt, image_start_);
                prompt.images.push_back({plan, prompt.ids.size()});
                append(prompt, image_, size_t(plan.tokens()));
                append(prompt, image_end_);
            }
            append_text(prompt, pending.text());
            append(prompt, user_);
        }
        for (const int32_t id : prompt.ids)
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("HunYuanVL prompt token is outside the vocabulary");
        return prompt;
    }
    // Every axis counts tokens; inside an image grid the last three axes carry the merged
    // column (including the row separator), the merged row and the image ordinal.
    std::vector<int32_t> positions(const Prompt& prompt) const {
        const size_t axes = sections_.size(), count = prompt.ids.size();
        std::vector<int32_t> result(count * axes);
        for (size_t t = 0; t < count; ++t)
            std::fill_n(result.data() + t * axes, axes, int32_t(t));
        int32_t ordinal = 0;
        for (const auto& span : prompt.images) {
            const int rows = span.plan.grid_rows(), cols = span.plan.grid_cols();
            for (int y = 0; y < rows; ++y)
                for (int x = 0; x <= cols; ++x) {
                    int32_t* at =
                        result.data() + (span.offset + 1 + size_t(y) * (cols + 1) + x) * axes;
                    at[axes - 3] = x;
                    at[axes - 2] = y;
                    at[axes - 1] = ordinal;
                }
            ++ordinal;
        }
        return result;
    }
    std::vector<uint8_t> embed(const int32_t* ids, size_t count) const {
        std::vector<float> values(count * size_t(hidden_));
        for (size_t i = 0; i < count; ++i)
            std::memcpy(values.data() + i * hidden_,
                        embeddings_.data() + size_t(ids[i]) * hidden_ * sizeof(float),
                        size_t(hidden_) * sizeof(float));
        return pack(values.data(), values.size(), dtype_);
    }
    void insert_image(const ImageSpan& span, std::vector<uint8_t>& embeddings) {
        const auto& plan = span.plan;
        const auto pixels = patchify(plan, settings_);
        const auto packed = pack(pixels.data(), pixels.size(), dtype_);
        std::vector<int32_t> ids;
        std::vector<float> lambdas;
        position_corners(plan, edge_, ids, lambdas);
        const int64_t patches = plan.patches();
        TensorMap input{
            {"pixel_values",
             {const_cast<uint8_t*>(packed.data()),
              {patches, 3 * int64_t(plan.patch) * plan.patch},
              dtype_}},
            {"position_ids", {ids.data(), {patches, 4}, DType::kInt32}},
            {"position_lambdas", {lambdas.data(), {patches, 4}, DType::kFloat32}},
        };
        const auto output = visual_->forward(input).at("image_features");
        const int cells = plan.grid_rows() * plan.grid_cols();
        const size_t row = size_t(hidden_) * dtype_size(dtype_);
        if (output.dtype != dtype_ || output.nbytes() != size_t(cells + 3) * row)
            throw std::runtime_error("HunYuanVL vision output shape mismatch");
        const auto* source = static_cast<const uint8_t*>(output.data);
        uint8_t* target = embeddings.data() + span.offset * row;
        const auto take = [&](int index) {
            std::memcpy(target, source + size_t(index) * row, row);
            target += row;
        };
        take(cells + 1); // image begin
        for (int y = 0; y < plan.grid_rows(); ++y) {
            for (int x = 0; x < plan.grid_cols(); ++x)
                take(y * plan.grid_cols() + x);
            take(cells); // row separator
        }
        take(cells + 2); // image end
    }
    std::vector<float> step(const uint8_t* embeddings, const int32_t* axis_positions, int count,
                            int offset) {
        const size_t axes = sections_.size();
        std::vector<int> axis_of(static_cast<size_t>(head_dim_));
        for (size_t section = 0, d = 0; section < axes; ++section)
            for (int i = 0; i < 2 * sections_[section]; ++i)
                axis_of[d++] = int(section);
        std::vector<float> cosine(size_t(count) * head_dim_), sine(cosine.size());
        const int half = head_dim_ / 2;
        for (int t = 0; t < count; ++t)
            for (int d = 0; d < head_dim_; ++d) {
                const float angle = float(axis_positions[size_t(t) * axes + size_t(axis_of[d])]) *
                                    inverse_frequencies_[size_t(d % half)];
                cosine[size_t(t) * head_dim_ + d] = round_to(std::cos(angle), dtype_);
                sine[size_t(t) * head_dim_ + d] = round_to(std::sin(angle), dtype_);
            }
        const int keys = capacity_ + count;
        std::vector<float> mask(size_t(count) * keys, std::numeric_limits<float>::lowest());
        for (int i = 0; i < count; ++i) {
            std::fill_n(mask.data() + size_t(i) * keys, offset, 0.0F);
            std::fill_n(mask.data() + size_t(i) * keys + capacity_, i + 1, 0.0F);
        }
        TensorMap input{
            {"inputs_embeds", {const_cast<uint8_t*>(embeddings), {count, hidden_}, dtype_}},
            {"rope_cos", {cosine.data(), {count, head_dim_}, DType::kFloat32}},
            {"rope_sin", {sine.data(), {count, head_dim_}, DType::kFloat32}},
            {"attention_mask", {mask.data(), {1, count, keys}, DType::kFloat32}},
        };
        decoder_->forward_async(input);
        std::vector<float> logits(static_cast<size_t>(vocab_));
        cuda_check(cudaMemcpyAsync(logits.data(), decoder_->device_ptr("logits"),
                                   logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   decoder_->stream()));
        const size_t row = size_t(head_dim_) * dtype_size(dtype_);
        for (int i = 0; i < layers_; ++i)
            for (int kind = 0; kind < 2; ++kind) {
                const auto name =
                    std::string("present_") + (kind ? "v_" : "k_") + std::to_string(i);
                cuda_check(
                    cudaMemcpy2DAsync(static_cast<uint8_t*>(cache_[size_t(2 * i + kind)].data()) +
                                          size_t(offset) * row,
                                      size_t(capacity_) * row, decoder_->device_ptr(name),
                                      size_t(count) * row, size_t(count) * row, size_t(kv_heads_),
                                      cudaMemcpyDeviceToDevice, decoder_->stream()));
            }
        decoder_->sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("HunYuanVL decoder produced nonfinite logits");
        return logits;
    }
    TextResult execute(Span<const ImagesTextMessage> messages, const Generation& generation) {
        const auto setup = Clock::now();
        const auto prompt = prepare(messages);
        const size_t length = prompt.ids.size();
        if (length > size_t(capacity_) ||
            size_t(generation.max_new_tokens - 1) > size_t(capacity_) - length)
            throw std::invalid_argument(
                "HunYuanVL prompt and output budget exceed the built decoder capacity");
        const auto axis_positions = positions(prompt);
        auto embeddings = embed(prompt.ids.data(), length);
        for (const auto& span : prompt.images)
            insert_image(span, embeddings);
        for (auto& tensor : cache_)
            cuda_check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), decoder_->stream()));
        TextResult result;
        result.setup_ms = elapsed(setup);
        const auto prefill = Clock::now();
        const size_t axes = sections_.size(), row = size_t(hidden_) * dtype_size(dtype_);
        std::vector<float> logits;
        for (size_t start = 0; start < length; start += size_t(max_input_)) {
            const int count = int(std::min(length - start, size_t(max_input_)));
            logits = step(embeddings.data() + start * row, axis_positions.data() + start * axes,
                          count, int(start));
        }
        result.prefill_ms = elapsed(prefill);
        const auto decode = Clock::now();
        std::vector<bool> seen(static_cast<size_t>(vocab_));
        for (const int32_t id : prompt.ids)
            seen[size_t(id)] = true;
        Sampler sampler(generation);
        const auto penalty = float(generation.repetition_penalty);
        for (int i = 0; i < generation.max_new_tokens; ++i) {
            apply_repetition_penalty(logits, seen, penalty);
            const int32_t token = sampler(logits);
            result.token_ids.push_back(token);
            seen[size_t(token)] = true;
            if (eos_.count(token) || i + 1 == generation.max_new_tokens)
                break;
            const auto next = embed(&token, 1);
            const std::vector<int32_t> position(axes, int32_t(length) + i);
            logits = step(next.data(), position.data(), 1, int(length) + i);
        }
        result.decode_ms = elapsed(decode);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }
};
} // namespace
} // namespace trtmc::hunyuan_vl

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new trtmc::hunyuan_vl::Pipeline(context);
}

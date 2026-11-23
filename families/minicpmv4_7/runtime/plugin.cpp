/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_7/runtime/image_preprocess.h"
#include "families/minicpmv4_7/runtime/prompt.h"
#include "families/minicpmv4_7/runtime/request.h"
#include "families/minicpmv4_7/runtime/sampler.h"
#include "families/minicpmv4_7/runtime/state.h"
#include "families/minicpmv4_7/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_7 {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr const char* kTask = "images_text_to_text";

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* found = bundle.find_section(name);
    if (found == nullptr || found->length == 0)
        throw std::runtime_error("minicpmv4_7 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::string text_section(const BundleReader& bundle, std::string_view name) {
    const auto data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct RuntimeConfig {
    DType work{DType::kBFloat16};
    int32_t capacity{0};
    int32_t hidden{0};
    int32_t vocab{0};
    StateGeometry state;
    std::vector<int32_t> eos;
    TokenIds tokens;
    ProcessorConfig processor;
    Json prompts;
};

template <typename T>
T get(const Json& object, const char* name) {
    if (!object.contains(name))
        throw std::runtime_error(std::string("minicpmv4_7 runtime.json is missing ") + name);
    return object.at(name).get<T>();
}

RuntimeConfig parse_runtime(const BundleReader& bundle) {
    const auto json = Json::parse(text_section(bundle, "runtime.json"));
    if (get<int>(json, "schema_version") != 1)
        throw std::runtime_error("minicpmv4_7 runtime.json has an unsupported schema");
    RuntimeConfig config;
    const auto precision = get<std::string>(json, "precision");
    if (precision != "bf16" && precision != "fp32")
        throw std::runtime_error("minicpmv4_7 runtime.json has an invalid precision");
    config.work = precision == "bf16" ? DType::kBFloat16 : DType::kFloat32;
    config.capacity = get<int32_t>(json, "capacity");
    config.hidden = get<int32_t>(json, "hidden_size");
    config.vocab = get<int32_t>(json, "vocab_size");
    auto& s = config.state;
    s.capacity = config.capacity;
    s.linear_layers = get<int32_t>(json, "linear_layers");
    s.full_layers = get<int32_t>(json, "full_layers");
    s.conv_dim = get<int32_t>(json, "conv_dim");
    s.conv_kernel = get<int32_t>(json, "conv_kernel");
    s.linear_heads = get<int32_t>(json, "linear_value_heads");
    s.linear_head_dim = get<int32_t>(json, "linear_head_dim");
    s.kv_width = get<int32_t>(json, "kv_width");
    s.cache_dtype = config.work;
    config.eos = get<std::vector<int32_t>>(json, "eos_token_ids");
    const auto& ids = json.at("token_ids");
    auto& t = config.tokens;
    t.image_pad = get<int32_t>(ids, "image_pad");
    t.image_start = get<int32_t>(ids, "image_start");
    t.image_end = get<int32_t>(ids, "image_end");
    t.slice_start = get<int32_t>(ids, "slice_start");
    t.slice_end = get<int32_t>(ids, "slice_end");
    t.image_id_start = get<int32_t>(ids, "image_id_start");
    t.image_id_end = get<int32_t>(ids, "image_id_end");
    t.newline = get<int32_t>(ids, "newline");
    const auto& vision = json.at("vision");
    auto& p = config.processor;
    p.patch_size = get<int32_t>(vision, "patch_size");
    p.patches_per_side = get<int32_t>(vision, "patches_per_side");
    p.scale_resolution = get<int32_t>(vision, "scale_resolution");
    p.max_slice_nums = get<int32_t>(vision, "max_slice_nums");
    p.max_patches = get<int32_t>(vision, "max_patches");
    p.mean = get<std::array<float, 3>>(vision, "image_mean");
    p.stddev = get<std::array<float, 3>>(vision, "image_std");
    config.prompts = json.at("prompts");
    if (config.capacity < 16 || config.hidden <= 0 || config.vocab <= 0 || config.eos.empty() ||
        p.patch_size <= 0 || p.patches_per_side <= 0 || p.scale_resolution <= 0 ||
        p.max_slice_nums <= 0 || p.max_patches < 16 || !config.prompts.contains("false") ||
        !config.prompts.contains("true"))
        throw std::runtime_error("minicpmv4_7 runtime.json contains invalid values");
    return config;
}

class Model final : public internal::IModel, public internal::IImagesTextToText {
  public:
    explicit Model(const FamilyContext& context)
        : config_(parse_runtime(context.reader)), task_(context.reader.info().task) {
        if (task_ != kTask)
            throw std::runtime_error("minicpmv4_7 bundle task must be images_text_to_text");
        const auto vision_plan = require_section(context.reader, "vision.plan");
        vision_ = context.backend.create_module(vision_plan.data(), vision_plan.size(), {});
        if (!vision_ || !vision_->ok() || !vision_->has_output("image_features") ||
            vision_->tensor_dtype("image_features") != config_.work)
            throw std::runtime_error("minicpmv4_7 vision engine failed to load");
        vision_->set_timing_label("minicpmv4_7 vision");
        {
            const auto plan = require_section(context.reader, "decoder.plan");
            decoder_ = context.backend.create_module(plan.data(), plan.size(), {});
        }
        if (!decoder_ || !decoder_->ok() || !decoder_->has_output("logits"))
            throw std::runtime_error("minicpmv4_7 decoder engine failed to load");
        decoder_->set_timing_label("minicpmv4_7 decoder");
        if (decoder_->tensor_dtype("visual_embedding") != config_.work ||
            decoder_->tensor_shape("visual_embedding") != std::vector<int64_t>{1, config_.hidden} ||
            decoder_->tensor_shape("logits") != std::vector<int64_t>{1, config_.vocab})
            throw std::runtime_error("minicpmv4_7 decoder disagrees with runtime.json");
        state_ = std::make_unique<DecoderState>(config_.state, decoder_->stream());
        state_->bind(*decoder_);
        const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size());
        if (!tokenizer_)
            throw std::runtime_error("minicpmv4_7 tokenizer construction failed");
        const auto& t = config_.tokens;
        for (const auto& [id, symbol] :
             std::vector<std::pair<int32_t, const char*>>{{t.image_pad, "<|image_pad|>"},
                                                          {t.image_start, "<image>"},
                                                          {t.image_end, "</image>"},
                                                          {t.slice_start, "<slice>"},
                                                          {t.slice_end, "</slice>"},
                                                          {t.image_id_start, "<image_id>"},
                                                          {t.image_id_end, "</image_id>"}})
            if (tokenizer_->id_for_token(symbol) != id)
                throw std::runtime_error("minicpmv4_7 tokenizer disagrees with runtime.json");
        zero_row_.assign(static_cast<std::size_t>(config_.hidden) * dtype_size(config_.work), 0);
    }

    const char* task() const noexcept override { return task_.c_str(); }

    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImagesTextToText>(*this, request_config_fields())};
    }

    TextResult run(const internal::ImagesTextToTextRequest& request,
                   internal::ConfigView supplied) override {
        const auto setup_begin = Clock::now();
        // Host preparation reads only immutable members; engines and state need the lock.
        auto prepared = prepare_request(
            request, supplied, {config_.prompts, config_.processor, *tokenizer_, config_.capacity});
        const auto& ids = prepared.ids;
        const int64_t max_new = prepared.max_new_tokens;

        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::array<int32_t, 2>> grids;
        std::vector<uint8_t> features;
        for (auto& image : prepared.images)
            for (auto& crop : image.crops) {
                grids.push_back({crop.grid_h, crop.grid_w});
                encode_crop(crop, features);
            }
        const auto canvas =
            canvas_positions(ids, config_.tokens, grids, config_.processor.merge_factor);
        if (canvas.decode_position + max_new > config_.capacity + 1)
            throw std::invalid_argument(
                "MiniCPM-V 4.7 rotary positions exceed the decoder capacity");
        const std::size_t row_bytes = zero_row_.size();
        std::size_t visual_rows = 0;
        for (const auto id : ids)
            visual_rows += id == config_.tokens.image_pad;
        if (visual_rows * row_bytes != features.size())
            throw std::logic_error("MiniCPM-V 4.7 visual tokens disagree with encoded features");
        const auto setup_end = Clock::now();

        state_->reset();
        std::vector<float> logits;
        std::size_t next_row = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const uint8_t* row = zero_row_.data();
            const bool visual = ids[i] == config_.tokens.image_pad;
            if (visual)
                row = features.data() + row_bytes * next_row++;
            step(ids[i], canvas.positions[i], row, visual, logits);
        }
        const auto prefill_end = Clock::now();

        std::vector<int32_t> generated;
        const int32_t delta = canvas.decode_position - static_cast<int32_t>(ids.size());
        while (true) {
            const int32_t token = prepared.sampler.sample(logits);
            generated.push_back(token);
            const bool stop =
                std::find(config_.eos.begin(), config_.eos.end(), token) != config_.eos.end();
            if (stop || static_cast<int64_t>(generated.size()) >= max_new)
                break;
            const int32_t position = state_->position() + delta;
            step(token, {position, position, position}, zero_row_.data(), false, logits);
        }
        const auto decode_end = Clock::now();
        // token_ids keep a terminating EOS as generation emitted it; the text omits it.
        std::vector<int32_t> visible = generated;
        if (std::find(config_.eos.begin(), config_.eos.end(), visible.back()) != config_.eos.end())
            visible.pop_back();
        TextResult result{tokenizer_->decode(visible), generated,
                          elapsed_ms(setup_end, prefill_end), elapsed_ms(prefill_end, decode_end)};
        result.setup_ms = elapsed_ms(setup_begin, setup_end);
        return result;
    }

  private:
    void encode_crop(Crop& p, std::vector<uint8_t>& features) {
        const auto count = static_cast<int64_t>(p.grid_h) * p.grid_w;
        TensorMap inputs{
            {"pixel_patches",
             {p.patches.data(),
              {count, static_cast<int64_t>(3) * config_.processor.patch_size *
                          config_.processor.patch_size},
              DType::kFloat32}},
            {"position_ids", {p.position_ids.data(), {count}, DType::kInt32}},
            {"window_index", {p.window_index.data(), {count}, DType::kInt32}},
            {"merge_index", {p.merge_index.data(), {count / 4}, DType::kInt32}},
        };
        const auto output = vision_->forward(inputs);
        const auto& image = output.at("image_features");
        const int64_t tokens = p.tokens(config_.processor.merge_factor);
        if (image.dtype != config_.work ||
            image.shape != std::vector<int64_t>{tokens, config_.hidden})
            throw std::runtime_error("minicpmv4_7 vision output has unexpected geometry");
        const auto* bytes = static_cast<const uint8_t*>(image.data);
        features.insert(features.end(), bytes, bytes + image.nbytes());
    }

    void step(int32_t token, const std::array<int32_t, 3>& rotary, const uint8_t* visual_row,
              bool visual, std::vector<float>& logits) {
        token_input_ = token;
        use_visual_ = visual ? 1 : 0;
        for (int axis = 0; axis < 3; ++axis) {
            if (rotary[static_cast<std::size_t>(axis)] < 0 ||
                rotary[static_cast<std::size_t>(axis)] > config_.capacity)
                throw std::invalid_argument("MiniCPM-V 4.7 rotary position exceeds the capacity");
            offsets_[static_cast<std::size_t>(axis)] =
                rotary[static_cast<std::size_t>(axis)] - state_->position();
        }
        TensorMap inputs;
        inputs["token_id"] = Tensor{&token_input_, {1}, DType::kInt32};
        inputs["visual_embedding"] =
            Tensor{const_cast<uint8_t*>(visual_row), {1, config_.hidden}, config_.work};
        inputs["use_visual_embedding"] = Tensor{&use_visual_, {1, 1}, DType::kInt32};
        inputs["rotary_position_offsets"] = Tensor{offsets_.data(), {3}, DType::kInt32};
        state_->prepare(inputs);
        decoder_->forward_async(inputs);
        decoder_->sync();
        if (logits_ == nullptr)
            logits_ = decoder_->device_ptr("logits");
        if (logits_ == nullptr)
            throw std::runtime_error("minicpmv4_7 decoder exposes no logits buffer");
        logits.resize(static_cast<std::size_t>(config_.vocab));
        const auto status = cudaMemcpy(logits.data(), logits_, logits.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("minicpmv4_7 logits copy failed: ") +
                                     cudaGetErrorString(status));
        state_->advance();
    }

    RuntimeConfig config_;
    std::string task_;
    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<DecoderState> state_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::vector<uint8_t> zero_row_;
    void* logits_{nullptr};
    int32_t token_input_{0};
    int32_t use_visual_{0};
    std::array<int32_t, 3> offsets_{};
    std::mutex mutex_;
};

} // namespace
} // namespace trtmc::minicpmv4_7

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minicpmv4_7 does not support --kv-cache-size");
    return new trtmc::minicpmv4_7::Model(context);
}

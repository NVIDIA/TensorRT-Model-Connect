/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma3n/runtime/image.h"
#include "families/gemma3n/runtime/prompt.h"
#include "families/gemma3n/runtime/request.h"
#include "families/gemma3n/runtime/sampler.h"
#include "families/gemma3n/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace trtmc::gemma3n {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace internal;

constexpr std::string_view kImageToken = "<image_soft_token>";

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Gemma 3n CUDA error: ") + cudaGetErrorString(error));
}

std::string section_text(const BundleReader& reader, const char* name) {
    const auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}

std::size_t element_size(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return 4;
    case DType::kBFloat16:
        return 2;
    default:
        throw std::runtime_error("Gemma 3n engines use FP32 or BF16 activations");
    }
}

class Pipeline final : public IModel, public IImagesTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : runtime_(Json::parse(section_text(context.reader, "runtime.json"))),
          tokenizer_(section_text(context.reader, "tokenizer.json")) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("gemma3n does not support --kv-cache-size");
        if (context.reader.info().task != IImagesTextToText::kTask)
            throw std::invalid_argument("gemma3n bundles serve images_text_to_text");
        if (runtime_.at("format") != 1)
            throw std::invalid_argument("Unsupported gemma3n bundle format");
        capacity_ = runtime_.at("max_cache_length");
        hidden_ = runtime_.at("hidden_size");
        vocab_ = runtime_.at("vocab_size");
        kv_width_ = runtime_.at("kv_width");
        image_size_ = runtime_.at("image_size");
        soft_tokens_ = runtime_.at("soft_tokens_per_image");
        image_token_ = runtime_.at("image_token_id");
        rescale_ = runtime_.at("rescale_factor").get<float>();
        for (const auto& id : runtime_.at("eos_token_ids"))
            eos_.push_back(id.get<int32_t>());
        cache_layers_ = runtime_.at("cache_layers").get<std::vector<int32_t>>();
        if (capacity_ <= 0 || hidden_ <= 0 || vocab_ != tokenizer_.vocab_size() || kv_width_ <= 0 ||
            image_size_ <= 0 || soft_tokens_ <= 0 || eos_.empty() || cache_layers_.empty() ||
            tokenizer_.token_id(kImageToken) != image_token_)
            throw std::invalid_argument("Invalid gemma3n runtime configuration");
        image_sequence_ = image_sequence(soft_tokens_);

        const auto vision = context.reader.read_section("vision.plan");
        vision_ = context.backend.create_module(vision.data(), vision.size(), {});
        if (!vision_ || !vision_->ok())
            throw std::runtime_error("gemma3n failed to load the vision engine");
        vision_->set_timing_label("gemma3n vision");
        const auto decoder = context.reader.read_section("decoder.plan");
        auto modules =
            context.backend.create_dual_profile_modules(decoder.data(), decoder.size(), {});
        if (!modules.prefill || !modules.prefill->ok() || !modules.decode || !modules.decode->ok())
            throw std::runtime_error("gemma3n failed to load the decoder profiles");
        prefill_ = std::move(modules.prefill);
        decode_ = std::move(modules.decode);
        prefill_->set_timing_label("gemma3n prefill");
        decode_->set_timing_label("gemma3n decode");
        dtype_ = prefill_->tensor_dtype("media_features");
        if (dtype_ != vision_->tensor_dtype("image_features"))
            throw std::invalid_argument("gemma3n vision and decoder precisions differ");
        for (const auto layer : cache_layers_)
            for (const char* kind : {"k", "v"}) {
                const auto name = std::string("cache_") + kind + "_" + std::to_string(layer);
                caches_.push_back(
                    DeviceTensor::zeros({capacity_, kv_width_}, dtype_, prefill_->stream()));
                if (!caches_.back().ok())
                    throw std::runtime_error("gemma3n KV cache allocation failed");
                prefill_->bind_external(name, caches_.back().data());
                decode_->bind_external(name, caches_.back().data());
            }
    }

    const char* task() const noexcept override { return IImagesTextToText::kTask.data(); }

    std::vector<TaskInstance> task_bindings() override {
        return {bind<IImagesTextToText>(*this, config_view())};
    }

    TextResult run(const ImagesTextToTextRequest& request, ConfigView supplied) override {
        const auto setup = Clock::now();
        const auto generation = read_generation(supplied);
        if (!request.tools.empty())
            throw std::invalid_argument("gemma3n does not accept tool definitions");
        const auto prompt = render_prompt(request.messages, image_sequence_);
        auto ids = tokenizer_.encode(prompt.text);
        const auto length = static_cast<int32_t>(ids.size());
        if (length > capacity_ ||
            int64_t{length} + generation.max_new_tokens - 1 > int64_t{capacity_})
            throw std::invalid_argument("gemma3n prompt and max_new_tokens exceed the bundle's "
                                        "max_sequence_length of " +
                                        std::to_string(capacity_));
        std::lock_guard<std::mutex> lock(mutex_);
        auto media = image_rows(ids, prompt.images);
        for (auto& cache : caches_)
            cuda_check(cudaMemsetAsync(cache.data(), 0, cache.nbytes(), prefill_->stream()));
        Sampler sampler(generation);
        TextResult result;
        result.setup_ms = elapsed_ms(setup);
        const auto prefill = Clock::now();
        // Only placeholder rows filled from an image take encoder soft tokens; a
        // generated placeholder id later uses its hard embedding, as in the reference.
        std::vector<int32_t> filled(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i)
            filled[i] = ids[i] == image_token_ ? 1 : 0;
        auto logits = step(*prefill_, ids, 0, media, filled);
        result.prefill_ms = elapsed_ms(prefill);
        const auto decode = Clock::now();
        for (int32_t produced = 0;;) {
            const auto token = sampler.next(logits);
            result.token_ids.push_back(token);
            ++produced;
            if (std::find(eos_.begin(), eos_.end(), token) != eos_.end() ||
                produced == generation.max_new_tokens)
                break;
            const std::vector<uint8_t> none(std::size_t(hidden_) * element_size(dtype_));
            logits = step(*decode_, {token}, length + produced - 1, none, {0});
        }
        result.decode_ms = elapsed_ms(decode);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }

  private:
    Json runtime_;
    Tokenizer tokenizer_;
    int32_t capacity_{}, hidden_{}, vocab_{}, kv_width_{}, image_size_{}, soft_tokens_{};
    int32_t image_token_{};
    float rescale_{};
    std::vector<int32_t> eos_, cache_layers_;
    std::string image_sequence_;
    DType dtype_{};
    // Contexts are destroyed before the cache storage they are bound to.
    std::vector<DeviceTensor> caches_;
    std::unique_ptr<ITrtModule> vision_, prefill_, decode_;
    std::mutex mutex_;

    // Encoder soft tokens for every image placeholder row, zero elsewhere.
    std::vector<uint8_t> image_rows(const std::vector<int32_t>& ids,
                                    const std::vector<ImageView>& images) {
        const auto row_bytes = std::size_t(hidden_) * element_size(dtype_);
        std::vector<uint8_t> rows(ids.size() * row_bytes);
        std::size_t image = 0;
        for (std::size_t i = 0; i < ids.size();) {
            if (ids[i] != image_token_) {
                ++i;
                continue;
            }
            std::size_t run = i;
            while (run < ids.size() && ids[run] == image_token_)
                ++run;
            if (run - i != std::size_t(soft_tokens_) || image == images.size())
                throw std::invalid_argument("gemma3n image placeholders do not match the images");
            const auto pixels = pixel_values(to_rgb8(images[image++]), image_size_, rescale_);
            TensorMap input{{"pixel_values",
                             {const_cast<float*>(pixels.data()),
                              {1, 3, image_size_, image_size_},
                              DType::kFloat32}}};
            const auto features = vision_->forward(input).at("image_features");
            if (features.dtype != dtype_ ||
                features.nbytes() != std::size_t(soft_tokens_) * row_bytes)
                throw std::runtime_error("gemma3n vision output has an unexpected shape");
            std::memcpy(rows.data() + i * row_bytes, features.data, features.nbytes());
            i = run;
        }
        if (image != images.size())
            throw std::invalid_argument("gemma3n image placeholders do not match the images");
        return rows;
    }

    std::vector<float> step(ITrtModule& module, const std::vector<int32_t>& ids, int32_t offset,
                            const std::vector<uint8_t>& media, std::vector<int32_t> media_mask) {
        const auto count = static_cast<int32_t>(ids.size());
        const auto keys = capacity_ + count;
        if (media_mask.size() != ids.size())
            throw std::logic_error("gemma3n media mask does not cover the step's rows");
        std::vector<int32_t> positions(ids.size());
        for (int32_t i = 0; i < count; ++i)
            positions[std::size_t(i)] = offset + i;
        // Cache rows before `offset` and the causal prefix of the new rows.
        std::vector<float> mask(std::size_t(count) * std::size_t(keys),
                                -std::numeric_limits<float>::infinity());
        for (int32_t i = 0; i < count; ++i) {
            std::fill_n(mask.data() + std::size_t(i) * std::size_t(keys), offset, 0.0F);
            std::fill_n(mask.data() + std::size_t(i) * std::size_t(keys) + capacity_, i + 1, 0.0F);
        }
        TensorMap input{
            {"token_id", {const_cast<int32_t*>(ids.data()), {count}, DType::kInt32}},
            {"position_id", {positions.data(), {count}, DType::kInt32}},
            {"attention_mask", {mask.data(), {count, keys}, DType::kFloat32}},
            {"media_features", {const_cast<uint8_t*>(media.data()), {count, hidden_}, dtype_}},
            {"media_mask", {media_mask.data(), {count}, DType::kInt32}},
        };
        module.forward_async(input);
        std::vector<float> logits(static_cast<std::size_t>(vocab_));
        cuda_check(cudaMemcpyAsync(logits.data(), module.device_ptr("logits"),
                                   logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   module.stream()));
        const auto row = std::size_t(kv_width_) * element_size(dtype_);
        for (std::size_t c = 0; c < cache_layers_.size(); ++c)
            for (int kind = 0; kind < 2; ++kind) {
                const auto name = std::string(kind == 0 ? "present_k_" : "present_v_") +
                                  std::to_string(cache_layers_[c]);
                cuda_check(cudaMemcpyAsync(
                    static_cast<uint8_t*>(caches_[2 * c + std::size_t(kind)].data()) +
                        std::size_t(offset) * row,
                    module.device_ptr(name), std::size_t(count) * row, cudaMemcpyDeviceToDevice,
                    module.stream()));
            }
        module.sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("gemma3n decoder produced non-finite logits");
        return logits;
    }
};

} // namespace
} // namespace trtmc::gemma3n

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new trtmc::gemma3n::Pipeline(context);
}

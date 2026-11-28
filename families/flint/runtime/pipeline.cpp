/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/flint/runtime/diagnostics.h"
#include "families/flint/runtime/sampler.h"
#include "families/flint/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace trtmc::flint {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// TensorRT asks for 256-byte aligned tensor addresses. Each position value
// gets its own aligned slot so a layer binds it without a per-step upload.
constexpr std::size_t kPositionStride = 256 / sizeof(std::int32_t);

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Flint ") + what + ": " + cudaGetErrorString(status));
}

std::string clean_text(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\0'), text.end());
    for (const std::string marker : {"<think>", "</think>"}) {
        for (auto pos = text.find(marker); pos != std::string::npos; pos = text.find(marker))
            text.erase(pos, marker.size());
    }
    for (const std::string role : {"begin_of_text", "end_of_text", "system", "developer", "user",
                                   "assistant", "observation"}) {
        const auto marker = "<|" + role + "|>";
        const auto replacement = "&lt;|" + role + "|&gt;";
        for (auto pos = text.find(marker); pos != std::string::npos;
             pos = text.find(marker, pos + replacement.size()))
            text.replace(pos, marker.size(), replacement);
    }
    return text;
}

class Stream final {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream"); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
};

class PinnedBuffer final {
  public:
    explicit PinnedBuffer(std::size_t bytes) { check_cuda(cudaMallocHost(&data_, bytes), "host"); }
    ~PinnedBuffer() { cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    template <typename T>
    T* as() const {
        return static_cast<T*>(data_);
    }

  private:
    void* data_{nullptr};
};

// Engine plans are up to 1.6 GB each. Reading every plan into one reused
// page-locked buffer avoids a fresh allocation per plan and lets TensorRT
// upload the weights from pinned memory.
class PlanBuffer final : public std::streambuf {
  public:
    explicit PlanBuffer(std::size_t capacity) : capacity_(capacity) {
        if (cudaMallocHost(reinterpret_cast<void**>(&pinned_), capacity) != cudaSuccess) {
            (void)cudaGetLastError();
            pinned_ = nullptr;
            pageable_.resize(capacity);
        }
    }
    ~PlanBuffer() override {
        if (pinned_ != nullptr)
            cudaFreeHost(pinned_);
    }
    PlanBuffer(const PlanBuffer&) = delete;
    PlanBuffer& operator=(const PlanBuffer&) = delete;

    std::size_t read(const BundleReader& reader, const std::string& name) {
        size_ = 0;
        std::ostream output(this);
        reader.copy_section(name, output);
        if (size_ != reader.find_section(name)->length)
            throw std::runtime_error("cannot read Flint engine " + name);
        return size_;
    }
    const char* data() const { return pinned_ != nullptr ? pinned_ : pageable_.data(); }

  protected:
    std::streamsize xsputn(const char* source, std::streamsize count) override {
        const auto bytes = static_cast<std::size_t>(count);
        if (bytes > capacity_ - size_)
            return 0;
        std::memcpy((pinned_ != nullptr ? pinned_ : pageable_.data()) + size_, source, bytes);
        size_ += bytes;
        return count;
    }
    int_type overflow(int_type value) override {
        if (traits_type::eq_int_type(value, traits_type::eof()))
            return traits_type::not_eof(value);
        const char byte = traits_type::to_char_type(value);
        return xsputn(&byte, 1) == 1 ? value : traits_type::eof();
    }

  private:
    std::size_t capacity_{0}, size_{0};
    char* pinned_{nullptr};
    std::vector<char> pageable_;
};

DeviceTensor device_tensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream) {
    DeviceTensor tensor(std::move(shape), dtype, stream);
    if (!tensor.ok())
        throw std::runtime_error("Flint cannot allocate device state");
    return tensor;
}

// One layer's K/V history on the device. The engine reads past_k[heads, n, dim]
// and writes present_k[heads, n + 1, dim], so two buffers alternate roles.
struct Cache {
    std::array<DeviceTensor, 2> k, v;
    int past{0};
    std::int64_t length{1};
    bool dummy{true};
};

class Pipeline final : public ITextGeneration, public Diagnostics {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto raw = context.reader.read_section("runtime.json");
        config_ = nlohmann::json::parse(raw.begin(), raw.end());
        hidden_ = config_.at("hidden_size");
        heads_ = config_.at("num_key_value_heads");
        dim_ = config_.at("head_dim");
        vocab_ = config_.at("vocab_size");
        limit_ = config_.at("max_sequence_length");
        eos_ = config_.at("eos_token_id").get<std::vector<std::int32_t>>();
        types_ = config_.at("layer_types").get<std::vector<std::string>>();
        if (hidden_ <= 0 || heads_ <= 0 || dim_ <= 0 || vocab_ <= 0 || limit_ <= 0 ||
            types_.empty() || eos_.empty())
            throw std::runtime_error("invalid Flint runtime dimensions");
        if (std::find(types_.begin(), types_.end(), "sliding_attention") != types_.end())
            window_ = config_.at("sliding_window");
        embedding_ = context.reader.read_section("embedding.bf16");
        if (embedding_.size() != static_cast<std::size_t>(hidden_) * vocab_ * 2)
            throw std::runtime_error("invalid Flint embedding section");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        // All engines share one stream, so a step is a single in-order chain.
        stream_ = std::make_shared<Stream>();
        ModuleCreateOptions options{};
        options.stream = stream_->get();
        std::vector<std::string> plans;
        for (std::size_t i = 0; i < types_.size(); ++i)
            plans.push_back("layers/" + std::to_string(i) + ".plan");
        plans.push_back("head.plan");
        std::size_t largest = 0;
        for (const auto& name : plans) {
            const auto* section = context.reader.find_section(name);
            if (section == nullptr)
                throw std::runtime_error("missing Flint engine " + name);
            largest = std::max<std::size_t>(largest, section->length);
        }
        PlanBuffer buffer(largest);
        auto load = [&](const std::string& name) {
            const auto size = buffer.read(context.reader, name);
            auto module = context.backend.create_module(buffer.data(), size, options);
            if (!module || !module->ok())
                throw std::runtime_error("cannot load Flint engine " + name);
            module->keep_alive(stream_);
            return module;
        };
        for (std::size_t i = 0; i < types_.size(); ++i)
            layers_.push_back(load(plans[i]));
        head_ = load(plans.back());
        allocate_device_state();
    }

    ~Pipeline() override { cudaStreamSynchronize(stream_->get()); }

    std::int32_t default_max_new_tokens() const override { return 128; }

    LogitTrace trace_logits(const std::string& prompt, std::int32_t steps) override {
        LogitTrace trace;
        trace.prompt_ids = tokenizer_->encode(prompt);
        if (steps < 1 || trace.prompt_ids.empty() ||
            trace.prompt_ids.size() + static_cast<std::size_t>(steps) >
                static_cast<std::size_t>(limit_))
            throw std::invalid_argument("invalid Flint logit trace request");
        reset();
        auto logits = prefill(trace.prompt_ids);
        for (int i = 0; i < steps; ++i) {
            trace.rows.push_back(logits);
            auto token = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            trace.generated_ids.push_back(token);
            if (std::find(eos_.begin(), eos_.end(), token) != eos_.end())
                break;
            if (i + 1 < steps)
                logits = step(token, true);
        }
        return trace;
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        if (cfg.max_new_tokens < 0)
            throw std::invalid_argument("negative Flint output budget");
        if (!cfg.lora_adapter_id.empty() || cfg.stop_on_boxed_answer)
            throw std::invalid_argument("unsupported Flint adapter/stop condition");
        std::string text = prompt;
        if (cfg.use_chat_template) {
            text = cfg.system_prompt.empty() ? "" : "<|system|>" + clean_text(cfg.system_prompt);
            text += "<|user|>" + clean_text(prompt) + "<|assistant|>";
        } else if (!cfg.system_prompt.empty()) {
            throw std::invalid_argument("Flint system_prompt requires the chat template");
        }
        auto ids = tokenizer_->encode(text);
        if (ids.empty())
            throw std::invalid_argument("Flint requires a non-empty prompt");
        if (ids.size() + static_cast<std::size_t>(cfg.max_new_tokens) >
            static_cast<std::size_t>(limit_))
            throw internal::ConfigError("Flint request exceeds the built context capacity");
        reset();
        TextResult result;
        if (cfg.max_new_tokens == 0)
            return result;
        auto params = flint_sampling_params_from_config(cfg, eos_.front());
        auto sampler = create_flint_sampler(params);
        const auto start = Clock::now();
        auto logits = prefill(ids);
        result.prefill_ms = elapsed(start);
        const auto decode_start = Clock::now();
        for (int i = 0; i < cfg.max_new_tokens; ++i) {
            const auto selected = sampler->sample(logits.data(), vocab_, params).token_id;
            result.token_ids.push_back(selected);
            const bool stop = cfg.eos_token_id >= 0
                                  ? selected == cfg.eos_token_id
                                  : std::find(eos_.begin(), eos_.end(), selected) != eos_.end();
            if (stop || i + 1 == cfg.max_new_tokens)
                break;
            logits = step(selected, true);
        }
        result.decode_ms = elapsed(decode_start);
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }

  private:
    static void require(const ITrtModule& module, const std::string& name, DType dtype,
                        bool input) {
        if ((input ? !module.has_input(name) : !module.has_output(name)) ||
            module.tensor_dtype(name) != dtype)
            throw std::runtime_error("Flint engine has an invalid tensor '" + name + "'");
    }

    static void bind(ITrtModule& module, const std::string& name, void* data,
                     const std::vector<int64_t>& shape = {}) {
        module.bind_external(name, data, shape);
        if (module.device_ptr(name) != data)
            throw std::runtime_error("Flint engine did not bind tensor '" + name + "'");
    }

    // Device buffers are sized once from the engines' maximum cache length and
    // stay bound across requests. Only the KV ping-pong addresses change.
    void allocate_device_state() {
        const auto stream = stream_->get();
        const auto layers = layers_.size();
        for (std::size_t i = 0; i < 2; ++i)
            hiddens_[i] = device_tensor({1, hidden_}, DType::kBFloat16, stream);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(limit_) * kPositionStride);
        for (std::int32_t p = 0; p < limit_; ++p)
            positions[static_cast<std::size_t>(p) * kPositionStride] = p;
        positions_ = device_tensor({static_cast<int64_t>(positions.size())}, DType::kInt32, stream);
        std::vector<float> mask(static_cast<std::size_t>(limit_) + 2, 0.0F);
        zero_mask_ = device_tensor({static_cast<int64_t>(mask.size())}, DType::kFloat32, stream);
        check_cuda(cudaMemcpyAsync(zero_mask_.data(), mask.data(), mask.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "mask upload");
        mask.front() = -1e30F;
        dummy_mask_ = device_tensor({static_cast<int64_t>(mask.size())}, DType::kFloat32, stream);
        check_cuda(cudaMemcpyAsync(dummy_mask_.data(), mask.data(), mask.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "mask upload");
        check_cuda(cudaMemcpyAsync(positions_.data(), positions.data(),
                                   positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream),
                   "position upload");
        caches_.resize(layers);
        capacities_.resize(layers);
        for (std::size_t i = 0; i < layers; ++i) {
            auto& module = *layers_[i];
            require(module, "hidden", DType::kBFloat16, true);
            require(module, "hidden_out", DType::kBFloat16, false);
            require(module, "mask", DType::kFloat32, true);
            for (const auto* name : {"past_k", "past_v"})
                require(module, name, DType::kBFloat16, true);
            for (const auto* name : {"present_k", "present_v"})
                require(module, name, DType::kBFloat16, false);
            const auto max = module.input_profile_shape("past_k", 0, ProfileShapeSelector::kMax);
            if (max.size() != 3 || max[0] != heads_ || max[2] != dim_ || max[1] < 1 ||
                max[1] > limit_)
                throw std::runtime_error("Flint engine has an invalid KV cache profile");
            capacities_[i] = max[1];
            for (auto* pair : {&caches_[i].k, &caches_[i].v})
                for (auto& buffer : *pair)
                    buffer = device_tensor({heads_, max[1] + 1, dim_}, DType::kBFloat16, stream);
            bind(module, "hidden", hiddens_[i % 2].data());
            bind(module, "hidden_out", hiddens_[(i + 1) % 2].data());
        }
        require(*head_, "hidden", DType::kBFloat16, true);
        require(*head_, "logits", DType::kFloat32, false);
        if (head_->tensor_shape("logits") != std::vector<int64_t>{1, vocab_})
            throw std::runtime_error("Flint head engine has an invalid logits shape");
        logits_ = device_tensor({1, vocab_}, DType::kFloat32, stream);
        bind(*head_, "hidden", hiddens_[layers % 2].data());
        bind(*head_, "logits", logits_.data());
        host_logits_ = std::make_unique<PinnedBuffer>(static_cast<std::size_t>(vocab_) * 4);
        // Embedding rows are staged per position, so queued uploads never share
        // a host slot that a later step overwrites.
        host_tokens_ =
            std::make_unique<PinnedBuffer>(static_cast<std::size_t>(limit_) * hidden_ * 2);
        check_cuda(cudaStreamSynchronize(stream), "device state setup");
    }

    void reset() {
        const auto stream = stream_->get();
        // A failed request may leave uploads queued from the staging rows.
        check_cuda(cudaStreamSynchronize(stream), "request reset");
        position_ = 0;
        const auto bytes = static_cast<std::size_t>(heads_) * dim_ * 2;
        for (auto& cache : caches_) {
            cache.past = 0;
            cache.length = 1;
            cache.dummy = true;
            // The masked dummy row must be finite: NaN * 0 would still poison
            // the attention output.
            check_cuda(cudaMemsetAsync(cache.k[0].data(), 0, bytes, stream), "cache reset");
            check_cuda(cudaMemsetAsync(cache.v[0].data(), 0, bytes, stream), "cache reset");
        }
    }

    std::vector<float> prefill(const std::vector<std::int32_t>& ids) {
        for (std::size_t i = 0; i + 1 < ids.size(); ++i)
            step(ids[i], false);
        return step(ids.back(), true);
    }

    void run_layer(std::size_t i) {
        auto& module = *layers_[i];
        auto& cache = caches_[i];
        const auto past = cache.past;
        const auto next = 1 - past;
        if (cache.length > capacities_[i])
            throw std::out_of_range("Flint cache exceeds the engine profile");
        const std::vector<int64_t> shape{heads_, cache.length, dim_};
        bind(module, "past_k", cache.k[past].data(), shape);
        bind(module, "past_v", cache.v[past].data(), shape);
        bind(module, "present_k", cache.k[next].data());
        bind(module, "present_v", cache.v[next].data());
        bind(module, "mask", (cache.dummy ? dummy_mask_ : zero_mask_).data(),
             {1, 1, cache.length + 1});
        if (module.has_input("position"))
            bind(module, "position",
                 static_cast<std::int32_t*>(positions_.data()) + position_ * kPositionStride);
        module.forward_device_async({});
        const auto length = cache.length + 1;
        // Keep the last window-1 real tokens for the next sliding step.
        const std::int64_t keep =
            types_[i] == "sliding_attention" ? std::min<std::int64_t>(length, window_ - 1) : length;
        if (keep == length) {
            cache.past = next;
        } else {
            // Move the newest rows of each head to the front of the past buffer.
            const auto row = static_cast<std::size_t>(dim_) * 2;
            const auto kept = static_cast<std::size_t>(keep) * row;
            const auto total = static_cast<std::size_t>(length) * row;
            for (auto* pair : {&cache.k, &cache.v}) {
                const auto* newest = static_cast<const char*>((*pair)[next].data()) + total - kept;
                check_cuda(cudaMemcpy2DAsync((*pair)[past].data(), kept, newest, total, kept,
                                             static_cast<std::size_t>(heads_),
                                             cudaMemcpyDeviceToDevice, stream_->get()),
                           "sliding cache copy");
            }
        }
        cache.dummy = cache.dummy && keep == length;
        cache.length = keep;
    }

    // One token through every layer on the device. Only the head output leaves
    // the GPU, and only when the caller needs logits.
    std::vector<float> step(std::int32_t token, bool logits) {
        if (token < 0 || token >= vocab_ || position_ >= limit_)
            throw std::out_of_range("Flint token or position outside model capacity");
        const auto stream = stream_->get();
        const auto bytes = static_cast<std::size_t>(hidden_) * 2;
        auto* row = host_tokens_->as<char>() + static_cast<std::size_t>(position_) * bytes;
        std::memcpy(row, embedding_.data() + static_cast<std::size_t>(token) * bytes, bytes);
        check_cuda(cudaMemcpyAsync(hiddens_[0].data(), row, bytes, cudaMemcpyHostToDevice, stream),
                   "embedding upload");
        for (std::size_t i = 0; i < layers_.size(); ++i)
            run_layer(i);
        ++position_;
        if (!logits)
            return {};
        head_->forward_device_async({});
        auto* host = host_logits_->as<float>();
        check_cuda(cudaMemcpyAsync(host, logits_.data(), static_cast<std::size_t>(vocab_) * 4,
                                   cudaMemcpyDeviceToHost, stream),
                   "logits download");
        check_cuda(cudaStreamSynchronize(stream), "step");
        return {host, host + vocab_};
    }

    nlohmann::json config_;
    std::int32_t hidden_, heads_, dim_, vocab_, limit_, window_{0}, position_{0};
    std::vector<std::int32_t> eos_;
    std::vector<std::string> types_;
    std::vector<char> embedding_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::shared_ptr<Stream> stream_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::unique_ptr<ITrtModule> head_;
    std::array<DeviceTensor, 2> hiddens_;
    DeviceTensor positions_, zero_mask_, dummy_mask_, logits_;
    std::vector<Cache> caches_;
    std::vector<std::int64_t> capacities_;
    std::unique_ptr<PinnedBuffer> host_logits_, host_tokens_;
};
} // namespace

ITask* create_pipeline(const FamilyContext& context) {
    return new Pipeline(context);
}
} // namespace trtmc::flint

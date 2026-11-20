/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// WhisperPipeline and WhisperKvCache themselves, without a GPU: a CPU CUDA
// runtime whose streams run their work in the order it was issued and no
// earlier than they must, and stand-in encoder and decoder modules that follow
// the TensorRT module's forward(), forward_async() and sync(). A pipeline
// whose decode steps queue their copies behind the engine must return the
// tokens, and leave the cache, of the pipeline whose steps wait for the engine
// first and of a plain host recomputation. Also checked: what a queued step
// issues, where its logits go, when the pipeline refuses to queue, what the
// factory does with a pipeline, and that a failed download leaves nothing
// queued and the pipeline usable.

#include "families/whisper/runtime/kv_cache.h"
#include "families/whisper/runtime/pipeline.h"
#include "families/whisper/runtime/plugin_helpers.h"
#include "families/whisper/runtime/whisper_decode_step.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// --- A CPU CUDA runtime ------------------------------------------------------
//
// Work issued on a stream runs in issue order. `g_eager` runs it when it is
// issued; otherwise it runs only when the host waits for that stream, or makes
// a blocking call, which waits for every stream. Results must not depend on
// which. An upload reads pageable memory before it returns. A download into
// page-locked memory is queued, and until it runs that memory holds NaN bytes:
// a host that reads it too early selects from them.

struct Issued {
    std::string what;
    cudaStream_t stream{nullptr};
    std::size_t bytes{0};
    const void* destination{nullptr};
};

struct Queued {
    cudaStream_t stream{nullptr};
    std::function<void()> work;
};

bool g_eager = false;
std::vector<Queued> g_queued;
std::vector<Issued> g_issued;
std::map<void*, std::size_t> g_device;
std::map<void*, std::size_t> g_page_locked;
int g_fail_host_allocations = 0;
int g_fail_download_at = 0; // the n-th queued download of the run fails; 0: none
int g_queued_downloads = 0;

void issue(const char* what, cudaStream_t stream, std::size_t bytes, const void* destination,
           std::function<void()> work) {
    g_issued.push_back({what, stream, bytes, destination});
    if (g_eager)
        work();
    else
        g_queued.push_back({stream, std::move(work)});
}

void run_queued(bool every_stream, cudaStream_t stream) {
    std::vector<Queued> later;
    std::vector<Queued> now;
    for (auto& queued : g_queued)
        (every_stream || queued.stream == stream ? now : later).push_back(std::move(queued));
    g_queued = std::move(later);
    for (auto& queued : now)
        queued.work();
}

bool page_locked(const void* ptr) {
    return g_page_locked.count(const_cast<void*>(ptr)) != 0;
}

} // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // Zeroed, so that rows a request never wrote compare equal between runs.
    *devPtr = std::calloc(1, size);
    g_device[*devPtr] = size;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (devPtr != nullptr) {
        check(g_device.erase(devPtr) == 1, "only held device memory is freed");
        std::free(devPtr);
    }
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (g_fail_host_allocations > 0) {
        --g_fail_host_allocations;
        return cudaErrorMemoryAllocation;
    }
    *ptr = std::malloc(size);
    g_page_locked[*ptr] = size;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    check(g_page_locked.erase(ptr) == 1, "only held page-locked memory is freed");
    std::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    run_queued(true, nullptr);
    g_issued.push_back({kind == cudaMemcpyDeviceToHost ? "blocking download" : "blocking copy",
                        nullptr, count, dst});
    std::memmove(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
    if (kind == cudaMemcpyHostToDevice) {
        const auto* bytes = static_cast<const char*>(src);
        std::vector<char> staged(bytes, bytes + count);
        issue("upload", stream, count, dst,
              [dst, staged] { std::memcpy(dst, staged.data(), staged.size()); });
    } else if (kind == cudaMemcpyDeviceToHost && page_locked(dst)) {
        if (++g_queued_downloads == g_fail_download_at)
            return cudaErrorInvalidValue;
        std::memset(dst, 0xFF, count);
        issue("download", stream, count, dst, [dst, src, count] { std::memcpy(dst, src, count); });
    } else if (kind == cudaMemcpyDeviceToHost) {
        run_queued(false, stream);
        g_issued.push_back({"blocking download", stream, count, dst});
        std::memcpy(dst, src, count);
    } else {
        issue("device copy", stream, count, dst,
              [dst, src, count] { std::memmove(dst, src, count); });
    }
    return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    run_queued(true, nullptr);
    std::memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    issue("memset", stream, count, devPtr,
          [devPtr, value, count] { std::memset(devPtr, value, count); });
    return cudaSuccess;
}

} // extern "C"

namespace {

using trtmc::DType;

// --- A module as TrtModuleImpl drives one, with static shapes ----------------

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    bool is_input;
};

class Module final : public trtmc::ITrtModule {
  public:
    using Engine = std::function<void(Module&)>;

    Module(const std::vector<TensorSpec>& tensors, cudaStream_t stream, Engine engine)
        : stream_(stream), engine_(std::move(engine)) {
        for (const auto& spec : tensors) {
            Buffer buffer;
            buffer.shape = spec.shape;
            buffer.dtype = spec.dtype;
            buffer.is_input = spec.is_input;
            buffer.bytes = trtmc::dtype_size(spec.dtype);
            for (const int64_t dim : spec.shape)
                buffer.bytes *= static_cast<std::size_t>(dim);
            cudaMalloc(&buffer.device, buffer.bytes);
            if (!spec.is_input)
                staging_[spec.name].resize(buffer.bytes);
            buffers_[spec.name] = buffer;
        }
    }

    ~Module() override {
        for (auto& [name, buffer] : buffers_) {
            if (!buffer.is_external)
                cudaFree(buffer.device);
        }
    }

    // The uploads, then the enqueue: the engine is one piece of work on the
    // stream. The module's stage timer records its end event here, when the
    // enqueue returns.
    void forward_async(const trtmc::TensorMap& inputs) override {
        for (const auto& [name, tensor] : inputs) {
            const auto found = buffers_.find(name);
            if (found == buffers_.end() || !found->second.is_input)
                continue;
            cudaMemcpyAsync(found->second.device, tensor.data,
                            std::min(tensor.nbytes(), found->second.bytes), cudaMemcpyHostToDevice,
                            stream_);
        }
        issue("engine", stream_, 0, nullptr, [this] { engine_(*this); });
    }

    void sync() override {
        g_issued.push_back({"wait", stream_, 0, nullptr});
        run_queued(false, stream_);
    }

    // The enqueue, the wait, then a blocking download of every output the
    // module owns into its pageable staging.
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        forward_async(inputs);
        sync();
        trtmc::TensorMap outputs;
        for (auto& [name, buffer] : buffers_) {
            if (buffer.is_input || buffer.is_external)
                continue;
            cudaMemcpy(staging_[name].data(), buffer.device, buffer.bytes, cudaMemcpyDeviceToHost);
            trtmc::Tensor tensor;
            tensor.data = staging_[name].data();
            tensor.shape = buffer.shape;
            tensor.dtype = buffer.dtype;
            outputs[name] = tensor;
        }
        return outputs;
    }

    cudaStream_t stream() const override { return stream_; }
    bool has_input(const std::string& name) const override {
        const auto found = buffers_.find(name);
        return found != buffers_.end() && found->second.is_input;
    }
    bool has_output(const std::string& name) const override {
        const auto found = buffers_.find(name);
        return found != buffers_.end() && !found->second.is_input;
    }
    DType tensor_dtype(const std::string& name) const override {
        const auto found = buffers_.find(name);
        return found == buffers_.end() ? DType::kFloat32 : found->second.dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto found = buffers_.find(name);
        return found == buffers_.end() ? std::vector<int64_t>{} : found->second.shape;
    }
    void* device_ptr(const std::string& name) const override {
        const auto found = buffers_.find(name);
        if (found == buffers_.end() || (hide_logits && name == "logits"))
            return nullptr;
        return found->second.device;
    }
    void bind_external(const std::string& name, void* ptr) override {
        Buffer& buffer = buffers_.at(name);
        if (ptr == buffer.device)
            return;
        if (!buffer.is_external)
            cudaFree(buffer.device);
        buffer.device = ptr;
        buffer.is_external = true;
    }
    void bind_external(const std::string& name, void* ptr, const std::vector<int64_t>&) override {
        bind_external(name, ptr);
    }
    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(buffers_.at(name).shape.size()) : 0;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(false); }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 0; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    template <typename T>
    T* at(const std::string& name) {
        return static_cast<T*>(buffers_.at(name).device);
    }

    bool hide_logits{false}; // device_ptr("logits") answers null

  private:
    struct Buffer {
        void* device{nullptr};
        std::vector<int64_t> shape;
        DType dtype{DType::kFloat32};
        std::size_t bytes{0};
        bool is_input{true};
        bool is_external{false};
    };

    std::vector<trtmc::TensorInfo> info(bool inputs) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, buffer] : buffers_) {
            if (buffer.is_input == inputs)
                result.push_back({name, buffer.shape, buffer.dtype, inputs});
        }
        return result;
    }

    cudaStream_t stream_;
    Engine engine_;
    std::unordered_map<std::string, Buffer> buffers_;
    std::unordered_map<std::string, std::vector<char>> staging_;
};

// --- The stand-in engines' arithmetic -----------------------------------------

constexpr int32_t kLayers = 4;
constexpr int32_t kKvDim = 6; // half-precision values per cache row
constexpr int32_t kVocab = 61;
constexpr int32_t kMelBins = 4;
constexpr int32_t kPositions = 8; // encoder positions; the mel has twice as many frames
constexpr std::size_t kRowBytes = kKvDim * sizeof(uint16_t);
constexpr std::size_t kCrossValues = 2 * kPositions * 4; // [1, heads, positions, head_dim]
unsigned g_request_seed = 0;
const std::vector<int32_t> kPrompt = {1, 2, 3, 4};

std::string layer_name(const char* stem, int32_t layer) {
    return stem + std::to_string(layer);
}

uint64_t mix(uint64_t hash, const void* data, std::size_t bytes) {
    const auto* byte = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i)
        hash = (hash ^ byte[i]) * 0x100000001B3ULL;
    return hash;
}

// One decoder step from its inputs: the token, the position, the rows of each
// layer's cache that the mask leaves visible, and the request's cross K/V. A
// row the cache holds wrongly or misses moves every later step. The logits
// take few values, so ties for the largest are common.
struct StepOutput {
    std::vector<float> logits = std::vector<float>(kVocab);
    std::vector<std::vector<uint16_t>> k = std::vector<std::vector<uint16_t>>(kLayers);
    std::vector<std::vector<uint16_t>> v = std::vector<std::vector<uint16_t>>(kLayers);
};

StepOutput decoder_step(int32_t token, int32_t position, const float* mask, int32_t max_length,
                        const std::vector<const uint16_t*>& cache_k,
                        const std::vector<const uint16_t*>& cache_v,
                        const std::vector<const float*>& cross_k,
                        const std::vector<const float*>& cross_v) {
    StepOutput out;
    uint64_t hash = 0xCBF29CE484222325ULL + static_cast<uint64_t>(token) * 1000003ULL +
                    static_cast<uint64_t>(position) * 7919ULL;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (int32_t row = 0; row < max_length; ++row) {
            if (mask[row] != 0.0F)
                continue;
            hash = mix(hash, cache_k[layer] + row * kKvDim, kRowBytes);
            hash = mix(hash, cache_v[layer] + row * kKvDim, kRowBytes);
        }
        hash = mix(hash, cross_k[layer], kCrossValues * sizeof(float));
        hash = mix(hash, cross_v[layer], kCrossValues * sizeof(float));
        for (int32_t i = 0; i < kKvDim; ++i) {
            out.k[layer].push_back(static_cast<uint16_t>(mix(hash, &i, sizeof(i)) >> 7));
            out.v[layer].push_back(static_cast<uint16_t>(mix(hash + 1, &i, sizeof(i)) >> 9));
        }
    }
    for (int32_t i = 0; i < kVocab; ++i)
        out.logits[i] = static_cast<float>(mix(hash, &i, sizeof(i)) % 23) * 0.5F - 4.0F;
    // Some steps have their one largest logit at an end of the vocabulary.
    const uint64_t pick = mix(hash, &position, sizeof(position)) >> 24;
    if (pick % 3 == 0)
        out.logits[kVocab - 1] = 50.0F;
    else if (pick % 3 == 1)
        out.logits[0] = 50.0F;
    return out;
}

// --- A pipeline over the stand-in modules --------------------------------------

struct Rig {
    int stream_tag{0};
    int other_tag{0};
    cudaStream_t stream{reinterpret_cast<cudaStream_t>(&stream_tag)};
    cudaStream_t other{reinterpret_cast<cudaStream_t>(&other_tag)};
    int32_t max_length;
    Module* encoder{nullptr};
    Module* decoder{nullptr};
    trtmc::WhisperKvCache* cache{nullptr};
    std::unique_ptr<trtmc::WhisperPipeline> pipeline;

    Rig(int32_t cache_rows, int32_t eot, bool cache_on_other_stream = false,
        DType logits_dtype = DType::kFloat32)
        : max_length(cache_rows) {
        const std::vector<int64_t> cross = {1, 2, kPositions, 4};
        std::vector<TensorSpec> encoder_io = {
            {"mel_features", {kMelBins, 2 * kPositions}, DType::kFloat32, true},
            {"encoder_padded_rows", {kPositions, 1}, DType::kFloat32, true},
        };
        std::vector<TensorSpec> decoder_io = {
            {"token_id", {1}, DType::kInt32, true},
            {"position_id", {1}, DType::kInt32, true},
            {"attention_mask", {1, max_length + 1}, DType::kFloat32, true},
            {"logits", {1, kVocab}, logits_dtype, false},
        };
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            for (const char* stem : {"cross_k_", "cross_v_"}) {
                encoder_io.push_back({layer_name(stem, layer), cross, DType::kFloat32, false});
                decoder_io.push_back({layer_name(stem, layer), cross, DType::kFloat32, true});
            }
            for (const char* stem : {"cache_k_", "cache_v_"})
                decoder_io.push_back(
                    {layer_name(stem, layer), {max_length, kKvDim}, DType::kFloat16, true});
            for (const char* stem : {"present_k_", "present_v_"})
                decoder_io.push_back(
                    {layer_name(stem, layer), {1, kKvDim}, DType::kFloat16, false});
        }
        // The encoder's output follows the request's seed, an integer, and not
        // the mel's floats: the token stream is the same under every compiler.
        auto encoder_module = std::make_unique<Module>(encoder_io, stream, [](Module& module) {
            const uint64_t hash = mix(1, &g_request_seed, sizeof(g_request_seed));
            for (int32_t layer = 0; layer < kLayers; ++layer) {
                for (std::size_t i = 0; i < kCrossValues; ++i) {
                    const uint64_t value = mix(hash + static_cast<uint64_t>(layer), &i, sizeof(i));
                    module.at<float>(layer_name("cross_k_", layer))[i] = (value % 97) * 0.125F;
                    module.at<float>(layer_name("cross_v_", layer))[i] = (value % 89) * 0.25F;
                }
            }
        });
        const int32_t rows = max_length;
        auto decoder_module = std::make_unique<Module>(decoder_io, stream, [rows](Module& module) {
            std::vector<const uint16_t*> cache_k;
            std::vector<const uint16_t*> cache_v;
            std::vector<const float*> cross_k;
            std::vector<const float*> cross_v;
            for (int32_t layer = 0; layer < kLayers; ++layer) {
                cache_k.push_back(module.at<uint16_t>(layer_name("cache_k_", layer)));
                cache_v.push_back(module.at<uint16_t>(layer_name("cache_v_", layer)));
                cross_k.push_back(module.at<float>(layer_name("cross_k_", layer)));
                cross_v.push_back(module.at<float>(layer_name("cross_v_", layer)));
            }
            const StepOutput out = decoder_step(
                *module.at<int32_t>("token_id"), *module.at<int32_t>("position_id"),
                module.at<float>("attention_mask"), rows, cache_k, cache_v, cross_k, cross_v);
            if (module.tensor_dtype("logits") == DType::kFloat32)
                std::memcpy(module.at<float>("logits"), out.logits.data(), kVocab * sizeof(float));
            for (int32_t layer = 0; layer < kLayers; ++layer) {
                std::memcpy(module.at<uint16_t>(layer_name("present_k_", layer)),
                            out.k[layer].data(), kRowBytes);
                std::memcpy(module.at<uint16_t>(layer_name("present_v_", layer)),
                            out.v[layer].data(), kRowBytes);
            }
        });
        encoder = encoder_module.get();
        decoder = decoder_module.get();
        auto state = std::make_unique<trtmc::WhisperKvCache>(
            kLayers, max_length, kKvDim, cache_on_other_stream ? other : stream, DType::kFloat16);
        cache = state.get();

        trtmc::WhisperConfig config;
        config.num_mel_bins = kMelBins;
        config.max_source_positions = kPositions;
        config.decoder_layers = kLayers;
        config.eot_token_id = eot;
        config.decoder_start_token_ids = kPrompt;
        trtmc::MelFilterbank mel;
        mel.n_freq_bins = 9;
        mel.n_mel_bins = kMelBins;
        for (int32_t i = 0; i < 9 * kMelBins; ++i)
            mel.data.push_back(0.01F * static_cast<float>(1 + (i * 7) % 13));
        // A 16-point transform every 4 samples of one second at 64 Hz.
        pipeline = std::make_unique<trtmc::WhisperPipeline>(
            std::move(encoder_module), std::move(decoder_module), std::move(state),
            std::move(config), kKvDim, kLayers, std::move(mel), 16, 4, 1, 64, 16, 0.0F, false,
            std::string("whisper"), stream);
    }

    // A step that waits first leaves its last row copies queued: they run
    // before the memory they name is released, as the next blocking call
    // would make them.
    ~Rig() { run_queued(true, nullptr); }

    std::vector<int32_t> transcribe(int32_t max_tokens, unsigned seed = 3) {
        g_request_seed = seed;
        std::vector<float> audio(64);
        for (std::size_t i = 0; i < audio.size(); ++i)
            audio[i] = static_cast<float>(mix(seed, &i, sizeof(i)) % 2001) / 1000.0F - 1.0F;
        trtmc::TranscriptionConfig request;
        request.max_output_tokens = max_tokens;
        request.input_sample_rate = 64;
        return pipeline->transcribe(audio.data(), 64, request).token_ids;
    }

    // Every layer's cache, after whatever is still queued has run.
    std::vector<char> cache_bytes() {
        run_queued(true, nullptr);
        std::vector<char> bytes;
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            for (trtmc::DeviceTensor* rows : {&cache->cache_k(layer), &cache->cache_v(layer)}) {
                const auto* begin = static_cast<const char*>(rows->data());
                bytes.insert(bytes.end(), begin, begin + max_length * kRowBytes);
            }
        }
        return bytes;
    }
};

struct Outcome {
    std::vector<int32_t> tokens;
    std::vector<char> cache;

    bool operator==(const Outcome& rhs) const { return tokens == rhs.tokens && cache == rhs.cache; }
};

Outcome run(bool queue, bool eager, int32_t cache_rows, int32_t eot, int32_t max_tokens,
            int requests = 1) {
    g_eager = eager;
    Rig rig(cache_rows, eot);
    if (queue)
        rig.pipeline->queue_decode_step_copies();
    Outcome outcome;
    for (int request = 0; request < requests; ++request) {
        outcome.tokens = rig.transcribe(max_tokens, 3 + static_cast<unsigned>(request));
        if (queue)
            check(g_queued.empty(), "a request whose steps queue their copies leaves none queued");
    }
    outcome.cache = rig.cache_bytes();
    g_eager = false;
    return outcome;
}

// The last request of run() with no stream, module, pipeline or cache object:
// each step recomputed from its own inputs, the cache kept as plain rows that
// a full cache shifts down by one, as WhisperKvCache documents.
Outcome reference(int32_t cache_rows, int32_t eot, int32_t max_tokens, int requests = 1) {
    // The request's cross K/V are the encoder's: read them from a pipeline
    // that ran the same requests and decoded nothing.
    Rig rig(cache_rows, eot);
    for (int request = 0; request < requests; ++request)
        rig.transcribe(0, 3 + static_cast<unsigned>(request));
    std::vector<const float*> cross_k;
    std::vector<const float*> cross_v;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        cross_k.push_back(rig.encoder->at<float>(layer_name("cross_k_", layer)));
        cross_v.push_back(rig.encoder->at<float>(layer_name("cross_v_", layer)));
    }
    using Rows = std::vector<uint16_t>;
    std::vector<Rows> k(kLayers, Rows(static_cast<std::size_t>(cache_rows) * kKvDim));
    std::vector<Rows> v = k;
    if (requests > 1) {
        // An earlier request's rows stay in the cache, hidden by the mask.
        const Outcome earlier = reference(cache_rows, eot, max_tokens, requests - 1);
        std::size_t offset = 0;
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            for (Rows* rows : {&k[layer], &v[layer]}) {
                std::memcpy(rows->data(), earlier.cache.data() + offset, cache_rows * kRowBytes);
                offset += cache_rows * kRowBytes;
            }
        }
    }
    int32_t position = 0;
    StepOutput last;
    const auto step = [&](int32_t token) {
        std::vector<float> mask(static_cast<std::size_t>(cache_rows) + 1, -1.0e4F);
        std::fill(mask.begin(), mask.begin() + std::min(position, cache_rows), 0.0F);
        mask.back() = 0.0F;
        std::vector<const uint16_t*> cache_k;
        std::vector<const uint16_t*> cache_v;
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            cache_k.push_back(k[layer].data());
            cache_v.push_back(v[layer].data());
        }
        last = decoder_step(token, position, mask.data(), cache_rows, cache_k, cache_v, cross_k,
                            cross_v);
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            for (auto [rows, row] : {std::pair<Rows*, const Rows*>{&k[layer], &last.k[layer]},
                                     std::pair<Rows*, const Rows*>{&v[layer], &last.v[layer]}}) {
                if (position == cache_rows) // full: every row moves down by one
                    std::copy(rows->begin() + kKvDim, rows->end(), rows->begin());
                const int32_t slot = std::min(position, cache_rows - 1);
                std::copy(row->begin(), row->end(), rows->begin() + slot * kKvDim);
            }
        }
        position = std::min(position + 1, cache_rows);
    };
    Outcome outcome;
    for (const int32_t token : kPrompt)
        step(token);
    for (int32_t i = 0; i < max_tokens; ++i) {
        int32_t best = 0;
        for (int32_t j = 1; j < kVocab; ++j) {
            if (last.logits[j] > last.logits[best])
                best = j;
        }
        outcome.tokens.push_back(best);
        if (best == eot)
            break;
        step(best);
    }
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const Rows* rows : {&k[layer], &v[layer]}) {
            const auto* begin = reinterpret_cast<const char*>(rows->data());
            outcome.cache.insert(outcome.cache.end(), begin, begin + cache_rows * kRowBytes);
        }
    }
    return outcome;
}

// --- Checks ----------------------------------------------------------------------

constexpr int32_t kNoEot = -1;

void check_same_results() {
    // An end-of-text token the free-running request selects part way through.
    const std::vector<int32_t> free_run = reference(32, kNoEot, 12).tokens;
    int32_t eot = kNoEot;
    for (std::size_t i = 4; i < free_run.size() && eot == kNoEot; ++i) {
        if (std::find(free_run.begin(), free_run.begin() + i, free_run[i]) == free_run.begin() + i)
            eot = free_run[i];
    }
    check(eot != kNoEot, "the stand-in request selects a new token after its fourth");

    struct Case {
        const char* name;
        int32_t cache_rows;
        int32_t eot;
        int32_t max_tokens;
        int requests;
    };
    const std::vector<Case> cases = {
        {"a request that ends at its token limit", 32, kNoEot, 12, 1},
        {"a request that ends at the end-of-text token", 32, eot, 12, 1},
        {"a request with no new token", 32, kNoEot, 0, 1},
        {"a request that fills the cache and shifts it", 8, kNoEot, 12, 1},
        {"a second request on the same pipeline", 32, kNoEot, 9, 2},
    };
    for (const Case& c : cases) {
        const Outcome expected = reference(c.cache_rows, c.eot, c.max_tokens, c.requests);
        check(expected.tokens.size() <= static_cast<std::size_t>(c.max_tokens) &&
                  (c.max_tokens == 0 || !expected.tokens.empty()),
              c.name);
        for (const bool eager : {false, true}) {
            check(run(false, eager, c.cache_rows, c.eot, c.max_tokens, c.requests) == expected,
                  "the step that waits first matches the host recomputation");
            check(run(true, eager, c.cache_rows, c.eot, c.max_tokens, c.requests) == expected,
                  "the step that queues its copies matches the host recomputation");
        }
    }
    check(reference(32, eot, 12).tokens.back() == eot && reference(32, eot, 12).tokens.size() < 12,
          "the end-of-text case stops early");
    check(std::count(free_run.begin(), free_run.end(), kVocab - 1) > 0 &&
              std::count(free_run.begin(), free_run.end(), 0) > 0,
          "the stand-in request selects the first and the last token of the vocabulary");
}

// What the decoder's steps of one request issued, split at each engine.
std::vector<std::vector<Issued>> decoder_steps(const Rig& rig) {
    std::vector<std::vector<Issued>> steps;
    int engines = 0;
    for (const Issued& issued : g_issued) {
        if (issued.what == "engine" && ++engines > 1) // the first engine is the encoder's
            steps.emplace_back();
        if (!steps.empty() && issued.stream == rig.stream && issued.what != "upload")
            steps.back().push_back(issued);
        else if (!steps.empty() && issued.what == "blocking download")
            steps.back().push_back(issued);
    }
    return steps;
}

void check_what_a_step_issues() {
    const std::size_t logits_bytes = kVocab * sizeof(float);
    {
        Rig rig(32, kNoEot);
        check(!rig.pipeline->decode_steps_queue_copies(), "a new pipeline waits first");
        rig.pipeline->queue_decode_step_copies();
        check(rig.pipeline->decode_steps_queue_copies(), "a pipeline asked to queue says it does");
        check(g_page_locked.size() == 1 && g_page_locked.begin()->second == logits_bytes,
              "the pipeline holds one page-locked buffer of the logits' size");
        const void* buffer = g_page_locked.begin()->first;
        g_issued.clear();
        rig.transcribe(5);
        const auto steps = decoder_steps(rig);
        check(steps.size() == kPrompt.size() + 5, "a request runs one step per token");
        for (const auto& step : steps) {
            std::vector<std::string> order;
            for (const Issued& issued : step)
                order.push_back(issued.what);
            std::vector<std::string> expected = {"engine", "download"};
            expected.insert(expected.end(), 2 * kLayers, "device copy");
            expected.emplace_back("wait");
            check(order == expected,
                  "a queued step issues the engine, the download, the row copies, one wait");
            check(step[1].bytes == logits_bytes && step[1].destination == buffer,
                  "the download fills the page-locked buffer with the logits");
        }
        check(g_page_locked.size() == 1, "a request allocates no page-locked memory");
    }
    check(g_page_locked.empty(), "the pipeline frees its page-locked buffer");
    {
        Rig rig(32, kNoEot);
        g_issued.clear();
        rig.transcribe(5);
        for (const auto& step : decoder_steps(rig)) {
            std::vector<std::string> order;
            for (const Issued& issued : step)
                order.push_back(issued.what);
            std::vector<std::string> expected = {"engine", "wait", "blocking download"};
            expected.insert(expected.end(), 2 * kLayers, "device copy");
            check(order == expected,
                  "a step that waits first issues the engine, the wait, a blocking download, "
                  "the row copies");
        }
        check(g_page_locked.empty(), "a pipeline that waits first holds no page-locked memory");
    }
}

std::string refusal_of(Rig& rig) {
    try {
        rig.pipeline->queue_decode_step_copies();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

void check_refusals() {
    {
        Rig rig(32, kNoEot, true);
        check(refusal_of(rig) == "WhisperPipeline: decode steps cannot queue their copies: the "
                                 "inference state does not advance on the decoder's stream",
              "a cache on another stream is refused");
    }
    {
        Rig rig(32, kNoEot, false, DType::kFloat16);
        check(refusal_of(rig) == "WhisperPipeline: decode steps cannot queue their copies: the "
                                 "decoder's logits are not float32: rebuild the bundle",
              "logits that are not float32 are refused");
        check(!rig.pipeline->decode_steps_queue_copies(), "a refused pipeline waits first");
    }
    {
        Rig rig(32, kNoEot);
        g_fail_host_allocations = 1;
        check(refusal_of(rig) == "WhisperPipeline: unable to allocate the logits buffer",
              "a failed logits allocation is reported");
        // The pipeline did not start to queue: it still decodes, waiting first.
        check(!rig.pipeline->decode_steps_queue_copies() &&
                  rig.transcribe(6) == reference(32, kNoEot, 6).tokens,
              "a pipeline whose allocation failed still waits first");
    }
    {
        // A pipeline that queues, asked again when the allocation fails, goes
        // back to waiting first: it holds no buffer to download into.
        Rig rig(32, kNoEot);
        rig.pipeline->queue_decode_step_copies();
        check(rig.pipeline->decode_steps_queue_copies(), "a pipeline that queues says so");
        g_fail_host_allocations = 1;
        check(refusal_of(rig) == "WhisperPipeline: unable to allocate the logits buffer" &&
                  !rig.pipeline->decode_steps_queue_copies() && g_page_locked.empty(),
              "a second call whose allocation fails leaves the pipeline waiting first");
        check(rig.transcribe(6) == reference(32, kNoEot, 6).tokens,
              "and the pipeline still decodes");
    }
    check(g_page_locked.empty(), "a refusal holds no page-locked memory");

    // A state that does not say its advance() queues on a stream is not asked
    // to: the interface's own answer is no.
    struct PlainState final : trtmc::WhisperInferenceState {
        void reset() override {}
        void bind_to(trtmc::ITrtModule&) override {}
        void prepare_step(trtmc::TensorMap&, int32_t) override {}
        void advance(int32_t) override {}
        int32_t position() const override { return 0; }
        int32_t max_length() const override { return 0; }
        int32_t num_layers() const override { return 0; }
        bool needs_attention_mask() const override { return false; }
        std::size_t device_memory_bytes() const override { return 0; }
        const char* state_type() const override { return "plain"; }
        bool ok() const override { return true; }
    };
    int stream_tag = 0;
    check(!PlainState().advance_queues_on(reinterpret_cast<cudaStream_t>(&stream_tag)) &&
              !PlainState().advance_queues_on(nullptr),
          "a state that does not override the question answers no");
}

// What the factory does with a pipeline, over pipelines themselves.
void check_factory_step() {
    {
        Rig rig(32, kNoEot);
        check(trtmc::whisper_choose_decode_step(*rig.pipeline, 1, "trt") ==
                      "[whisper] decode step: copies queued behind the engine" &&
                  rig.pipeline->decode_steps_queue_copies(),
              "the factory asks a single-GPU pipeline on the standard backend to queue");
        check(rig.transcribe(6) == reference(32, kNoEot, 6).tokens && g_queued.empty(),
              "and that pipeline decodes with queued steps");
    }
    for (const auto& [tensor_parallel_size, backend] :
         {std::pair<int32_t, const char*>{2, "trt"}, std::pair<int32_t, const char*>{1, "trt_rtx"},
          std::pair<int32_t, const char*>{1, nullptr}}) {
        Rig rig(32, kNoEot);
        check(trtmc::whisper_choose_decode_step(*rig.pipeline, tensor_parallel_size, backend) ==
                      "[whisper] decode step: waits for the engine before its copies" &&
                  !rig.pipeline->decode_steps_queue_copies() && g_page_locked.empty(),
              "the factory leaves a tensor-parallel pipeline, or one on another backend, as it is");
    }
    {
        // The line is the pipeline's own answer, not the choice: a pipeline
        // that already queues says so whatever it is handed.
        Rig rig(32, kNoEot);
        rig.pipeline->queue_decode_step_copies();
        check(trtmc::whisper_choose_decode_step(*rig.pipeline, 2, "trt") ==
                  "[whisper] decode step: copies queued behind the engine",
              "the line is read from the pipeline");
    }
    {
        Rig rig(32, kNoEot, true);
        std::string error;
        try {
            trtmc::whisper_choose_decode_step(*rig.pipeline, 1, "trt");
        } catch (const std::runtime_error& caught) {
            error = caught.what();
        }
        check(error == "WhisperPipeline: decode steps cannot queue their copies: the inference "
                       "state does not advance on the decoder's stream",
              "the factory passes a refusal on");
    }
}

void check_failed_download() {
    Rig rig(32, kNoEot);
    rig.pipeline->queue_decode_step_copies();
    g_queued_downloads = 0;
    g_fail_download_at = 6;
    std::string error;
    try {
        rig.transcribe(8);
    } catch (const std::runtime_error& caught) {
        error = caught.what();
    }
    g_fail_download_at = 0;
    check(error == "WhisperPipeline: unable to queue the logits download",
          "a download that cannot be queued fails the request");
    check(g_queued.empty(), "a failed request leaves nothing queued");
    check(rig.transcribe(8) == reference(32, kNoEot, 8).tokens,
          "the pipeline decodes the next request");

    // A decoder that hands out no device logits: the same failure, before
    // anything is copied.
    rig.decoder->hide_logits = true;
    error.clear();
    try {
        rig.transcribe(8);
    } catch (const std::runtime_error& caught) {
        error = caught.what();
    }
    rig.decoder->hide_logits = false;
    check(error == "WhisperPipeline: unable to queue the logits download" && g_queued.empty(),
          "a decoder without device logits fails the request and leaves nothing queued");
}

} // namespace

int main() {
    check_same_results();
    check_what_a_step_issues();
    check_refusals();
    check_factory_step();
    check_failed_download();
    check(g_device.empty() && g_page_locked.empty() && g_queued.empty(),
          "every pipeline released its memory and left nothing queued");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("whisper decode-pipeline checks passed\n");
    return 0;
}

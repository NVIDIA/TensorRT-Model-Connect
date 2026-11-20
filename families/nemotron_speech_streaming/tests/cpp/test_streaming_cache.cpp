/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Drives the real streaming transcription stream through CPU stand-ins for its
// engines, twice: on a backend where the stream gives the encoders their
// caches as host tensors, and on the standard TensorRT backend's name, where
// it binds device buffers to them. Both must decode the same frames into the
// same tokens, and every encoder step must read the caches its stream's last
// completed step wrote.
//
// The CUDA calls the stream makes are CPU stand-ins defined in this file, so a
// "device" buffer is host memory the stand-in encoders can read, each call can
// be failed in turn, and no GPU is needed.

#include "families/nemotron_speech_streaming/runtime/pipeline.h"
#include "families/nemotron_speech_streaming/runtime/plugin_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace trtmc;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

// --- CPU stand-ins for the CUDA runtime ---

// The CUDA stream of every stand-in engine and of the pipeline: an address
// that is never read, only compared.
int g_stream_tag = 0;
cudaStream_t test_stream() {
    return reinterpret_cast<cudaStream_t>(&g_stream_tag);
}

struct CudaStandIn {
    int allocations{0};     // cudaMalloc calls so far
    int fail_allocation{0}; // the call (counted from 1) that fails; 0 fails none
    int fills{0};           // cudaMemsetAsync calls so far
    int fail_fill{0};       // the call (counted from 1) that fails; 0 fails none
    int fills_on_other_stream{0};
    std::map<void*, std::size_t> live;
};

CudaStandIn g_cuda;

bool is_live_buffer(const void* pointer, std::size_t bytes) {
    const auto found = g_cuda.live.find(const_cast<void*>(pointer));
    return found != g_cuda.live.end() && found->second >= bytes;
}

} // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    ++g_cuda.allocations;
    *devPtr = nullptr;
    if (g_cuda.allocations == g_cuda.fail_allocation)
        return cudaErrorMemoryAllocation;
    void* buffer = std::malloc(size);
    if (buffer == nullptr)
        return cudaErrorMemoryAllocation;
    // A new buffer holds no zeros: a stream that skipped its fill would read these.
    std::memset(buffer, 0x5A, size);
    g_cuda.live[buffer] = size;
    *devPtr = buffer;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (devPtr == nullptr)
        return cudaSuccess;
    const auto found = g_cuda.live.find(devPtr);
    check(found != g_cuda.live.end(), "cudaFree is given a buffer this test allocated");
    if (found != g_cuda.live.end()) {
        g_cuda.live.erase(found);
        std::free(devPtr);
    }
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    ++g_cuda.fills;
    if (g_cuda.fills == g_cuda.fail_fill)
        return cudaErrorInvalidValue;
    if (stream != test_stream())
        ++g_cuda.fills_on_other_stream;
    check(is_live_buffer(devPtr, count), "cudaMemsetAsync stays inside one allocation");
    if (is_live_buffer(devPtr, count))
        std::memset(devPtr, value, count);
    return cudaSuccess;
}

} // extern "C"

namespace {

// --- Stand-in engines ---

const std::string kChannel = "cache_last_channel";
const std::string kTime = "cache_last_time";
const std::string kChannelNext = "cache_last_channel_next";
const std::string kTimeNext = "cache_last_time_next";

constexpr int32_t kLayers = 2;
constexpr int32_t kCacheLeft = 3;
constexpr int32_t kHidden = 2;
constexpr int32_t kTimeCache = 2;
constexpr int32_t kRightContext = 1;
constexpr std::size_t kChannelCount = static_cast<std::size_t>(kLayers) * kCacheLeft * kHidden;
constexpr std::size_t kTimeCount = static_cast<std::size_t>(kLayers) * kHidden * kTimeCache;

struct TensorSpec {
    std::vector<int64_t> shape;
    bool is_input{true};
    DType dtype{DType::kFloat32};
    bool dynamic{false};
};

// What one stand-in streaming encoder reports about itself. The default is
// what this family's engines report.
struct EncoderSpec {
    std::map<std::string, TensorSpec> tensors{
        {kChannel, {{kLayers, kCacheLeft, kHidden}, true}},
        {kTime, {{kLayers, kHidden, kTimeCache}, true}},
        {kChannelNext, {{kLayers, kCacheLeft, kHidden}, false}},
        {kTimeNext, {{kLayers, kHidden, kTimeCache}, false}},
    };
    cudaStream_t stream{test_stream()};
    bool cuda_graph{false};
};

// A module that reports no tensor.
EncoderSpec no_tensors() {
    EncoderSpec spec;
    spec.tensors.clear();
    return spec;
}

// The backend names of the two paths: the standard TensorRT backend, where a
// stream's caches are on the device, and another one, where they are host
// tensors.
const std::string kDeviceBackend = "trt";
const std::string kHostBackend = "trt_rtx";

// Binds counted over every encoder of a rig, and the one that is rejected.
struct BindFault {
    int calls{0};
    int fail_call{0}; // the bind (counted from 1) that is rejected; 0 rejects none
};

class Module final : public ITrtModule {
  public:
    using Invoke = std::function<TensorMap(Module&, const TensorMap&)>;

    explicit Module(Invoke invoke, EncoderSpec spec = no_tensors(), BindFault* fault = nullptr)
        : invoke_(std::move(invoke)), spec_(std::move(spec)), fault_(fault) {}

    TensorMap forward(const TensorMap& inputs) override { return invoke_(*this, inputs); }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return spec_.stream; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return spec_.cuda_graph; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        const auto found = spec_.tensors.find(name);
        return found != spec_.tensors.end() && found->second.is_input;
    }
    bool has_output(const std::string& name) const override {
        const auto found = spec_.tensors.find(name);
        return found != spec_.tensors.end() && !found->second.is_input;
    }
    DType tensor_dtype(const std::string& name) const override {
        const auto found = spec_.tensors.find(name);
        return found == spec_.tensors.end() ? DType::kFloat32 : found->second.dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto found = spec_.tensors.find(name);
        return found == spec_.tensors.end() ? std::vector<int64_t>{} : found->second.shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto found = bound_.find(name);
        return found == bound_.end() ? nullptr : found->second;
    }
    void bind_external(const std::string& name, void* pointer) override {
        ++bind_calls;
        if (fault_ != nullptr && ++fault_->calls == fault_->fail_call)
            throw std::runtime_error("stand-in module rejected external buffer for '" + name + "'");
        bound_[name] = pointer;
    }
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        bind_external(name, pointer);
    }
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensor_shape(name).size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        const auto found = spec_.tensors.find(name);
        return found != spec_.tensors.end() && found->second.dynamic;
    }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int bind_calls{0};

  private:
    Invoke invoke_;
    EncoderSpec spec_;
    BindFault* fault_;
    std::map<std::string, void*> bound_;
};

class Backend final : public IBackend {
  public:
    Backend(std::string name, std::function<std::unique_ptr<ITrtModule>()> create)
        : name_(std::move(name)), create_(std::move(create)) {}
    std::unique_ptr<ITrtModule> create_module(const void*, size_t,
                                              const ModuleCreateOptions&) override {
        return create_();
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::logic_error("unused");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::logic_error("unused");
    }
    const char* name() const override { return name_.c_str(); }

  private:
    std::string name_;
    std::function<std::unique_ptr<ITrtModule>()> create_;
};

Tensor tensor_of(std::vector<float>& values) {
    Tensor result;
    result.data = values.data();
    result.shape = {1, static_cast<int64_t>(values.size())};
    result.dtype = DType::kFloat32;
    return result;
}

// What a stand-in encoder computes in one step. Every output depends on both
// caches and on the chunk, so a step that read stale, unfilled or foreign
// caches decodes other frames.
void encoder_step(const float* channel, const float* time, const Tensor& mel, const Tensor& mask,
                  float* next_channel, float* next_time, std::vector<float>& encoder_output) {
    float mel_sum = 0.0F;
    const auto* mel_values = static_cast<const float*>(mel.data);
    for (std::size_t i = 0; i < mel.numel(); ++i)
        mel_sum += mel_values[i];
    float open = 0.0F;
    const auto* mask_values = static_cast<const float*>(mask.data);
    for (std::size_t i = 0; i < mask.numel(); ++i)
        open += mask_values[i] == 0.0F ? 1.0F : 0.0F;
    float channel_sum = 0.0F;
    for (std::size_t i = 0; i < kChannelCount; ++i)
        channel_sum += channel[i];
    float time_sum = 0.0F;
    for (std::size_t i = 0; i < kTimeCount; ++i)
        time_sum += time[i];

    for (std::size_t i = 0; i < kChannelCount; ++i)
        next_channel[i] =
            0.5F * channel[(i + 1) % kChannelCount] + 0.125F * mel_sum + static_cast<float>(i % 5);
    for (std::size_t i = 0; i < kTimeCount; ++i)
        next_time[i] = 0.25F * time[i] + channel[i % kChannelCount] + 1.0F;
    const auto query_frames = static_cast<std::size_t>(mask.shape.at(1));
    encoder_output.resize(query_frames * kHidden);
    for (std::size_t i = 0; i < encoder_output.size(); ++i)
        encoder_output[i] = channel_sum + 2.0F * time_sum + mel_sum + open + static_cast<float>(i);
}

// One encoder step as the stand-in engines saw it, in the order the steps ran.
struct StepLog {
    int stream{0}; // the stream the test was feeding, see StepHistory::feeding
    std::vector<float> mel;
    std::vector<float> mask;
    std::vector<int64_t> mask_shape;
    std::vector<float> encoder_output;
    bool completed{true}; // false when the decode of the step's frames failed
};

struct StepHistory {
    int feeding{0};
    int fail_step{0}; // the encoder step (counted from 1) whose enqueue fails; 0 fails none
    int attempts{0};  // encoder steps started, failed ones included
    std::vector<StepLog> steps;
};

// What one stand-in streaming encoder saw; the test reads it after a run.
struct EncoderRecord {
    int steps{0};
    bool given_cache_tensors{false}; // a cache tensor was among forward()'s host inputs
    bool unbound_step{false};        // a step ran without four live bound buffers
    bool overlapping_step{false};    // a step's input and output were one buffer
    bool bad_host_tensor{false};     // a host cache tensor was missing or of another size
    Module* module{nullptr};
    std::vector<float> encoder_output;
    std::vector<float> next_channel;
    std::vector<float> next_time;
};

bool is_host_cache(const TensorMap& inputs, const std::string& name, std::size_t count) {
    const auto found = inputs.find(name);
    return found != inputs.end() && found->second.data != nullptr &&
           found->second.dtype == DType::kFloat32 && found->second.numel() == count;
}

std::unique_ptr<ITrtModule> make_streaming_encoder(std::shared_ptr<EncoderRecord> record,
                                                   StepHistory* history, BindFault* fault,
                                                   const EncoderSpec& spec) {
    auto module = std::make_unique<Module>(
        [record, history](Module& self, const TensorMap& inputs) {
            const Tensor& mel = inputs.at("mel_features");
            const Tensor& mask = inputs.at("encoder_mask");
            const bool fails = ++history->attempts == history->fail_step;
            // Like a real module: it reads a cache where one was bound, and
            // from the host tensors it is given otherwise.
            const bool bound =
                self.device_ptr(kChannel) != nullptr || self.device_ptr(kTime) != nullptr ||
                self.device_ptr(kChannelNext) != nullptr || self.device_ptr(kTimeNext) != nullptr;
            TensorMap outputs;
            if (!bound) {
                if (!is_host_cache(inputs, kChannel, kChannelCount) ||
                    !is_host_cache(inputs, kTime, kTimeCount)) {
                    record->bad_host_tensor = true;
                    throw std::runtime_error("stand-in encoder: a host cache tensor is unusable");
                }
                if (fails)
                    throw std::runtime_error("stand-in encoder: enqueue failed");
                record->next_channel.resize(kChannelCount);
                record->next_time.resize(kTimeCount);
                encoder_step(static_cast<const float*>(inputs.at(kChannel).data),
                             static_cast<const float*>(inputs.at(kTime).data), mel, mask,
                             record->next_channel.data(), record->next_time.data(),
                             record->encoder_output);
                outputs[kChannelNext] = tensor_of(record->next_channel);
                outputs[kTimeNext] = tensor_of(record->next_time);
            } else {
                if (inputs.count(kChannel) != 0 || inputs.count(kTime) != 0)
                    record->given_cache_tensors = true;
                auto* channel = static_cast<float*>(self.device_ptr(kChannel));
                auto* time = static_cast<float*>(self.device_ptr(kTime));
                auto* next_channel = static_cast<float*>(self.device_ptr(kChannelNext));
                auto* next_time = static_cast<float*>(self.device_ptr(kTimeNext));
                const bool live = is_live_buffer(channel, kChannelCount * sizeof(float)) &&
                                  is_live_buffer(time, kTimeCount * sizeof(float)) &&
                                  is_live_buffer(next_channel, kChannelCount * sizeof(float)) &&
                                  is_live_buffer(next_time, kTimeCount * sizeof(float));
                if (!live) {
                    record->unbound_step = true;
                    throw std::runtime_error("stand-in encoder: a cache tensor is not bound");
                }
                if (channel == next_channel || time == next_time)
                    record->overlapping_step = true;
                if (fails) {
                    // A failed enqueue may leave anything in the buffers it writes.
                    std::memset(next_channel, 0x77, kChannelCount * sizeof(float));
                    std::memset(next_time, 0x77, kTimeCount * sizeof(float));
                    throw std::runtime_error("stand-in encoder: enqueue failed");
                }
                encoder_step(channel, time, mel, mask, next_channel, next_time,
                             record->encoder_output);
            }
            ++record->steps;
            StepLog step;
            step.stream = history->feeding;
            const auto* mel_values = static_cast<const float*>(mel.data);
            step.mel.assign(mel_values, mel_values + mel.numel());
            const auto* mask_values = static_cast<const float*>(mask.data);
            step.mask.assign(mask_values, mask_values + mask.numel());
            step.mask_shape = mask.shape;
            step.encoder_output = record->encoder_output;
            history->steps.push_back(std::move(step));
            outputs["encoder_output"] = tensor_of(record->encoder_output);
            return outputs;
        },
        spec, fault);
    record->module = module.get();
    return module;
}

// A pipeline over stand-in engines and what they recorded.
struct Rig {
    // `specs` describes the streaming encoders in the order the pipeline loads
    // them (the one a stream starts with, then the steady one); the last entry
    // repeats.
    explicit Rig(std::vector<EncoderSpec> specs, const std::string& backend_name = kDeviceBackend)
        : backend(backend_name, [this, specs]() {
              const std::size_t index = std::min(encoders.size(), specs.size() - 1);
              encoders.push_back(std::make_shared<EncoderRecord>());
              return make_streaming_encoder(encoders.back(), &history, &bind_fault, specs[index]);
          }) {
        auto whole_clip_encoder = std::make_unique<Module>(
            [](Module&, const TensorMap&) -> TensorMap { throw std::logic_error("unused"); });
        auto predictor = std::make_unique<Module>([this](Module&, const TensorMap& inputs) {
            ++predictor_runs;
            fills_at_predictor = g_cuda.fills;
            const auto token = *static_cast<const int32_t*>(inputs.at("token_id").data);
            const auto* state = static_cast<const float*>(inputs.at("state_h_0").data);
            pred_output = {static_cast<float>(token), 0.5F};
            next_h = {state[0] + 1.0F, state[1]};
            next_c = {static_cast<float>(token), 1.0F};
            return TensorMap{{"pred_output", tensor_of(pred_output)},
                             {"next_h_0", tensor_of(next_h)},
                             {"next_c_0", tensor_of(next_c)}};
        });
        auto joint = std::make_unique<Module>([this](Module&, const TensorMap& inputs) {
            const auto* frame = static_cast<const float*>(inputs.at("encoder_frame").data);
            const auto* pred = static_cast<const float*>(inputs.at("pred_output").data);
            frames.emplace_back(frame, frame + kHidden);
            if (++joint_calls == fail_joint_call) {
                // The step whose frames are being decoded does not complete:
                // its caches must not reach the next step.
                history.steps.back().completed = false;
                throw std::runtime_error("stand-in joint failure");
            }
            // One token per frame, chosen by the frame: emit it, then blank.
            const float folded =
                std::isfinite(frame[0]) ? std::fmod(std::fabs(frame[0]), 3.0F) : 0.0F;
            const int target = 1 + static_cast<int>(folded);
            logits.assign(4, 0.0F);
            logits[pred[0] == static_cast<float>(target) ? 0 : static_cast<std::size_t>(target)] =
                1.0F;
            return TensorMap{{"logits", tensor_of(logits)}};
        });

        RnntConfig config;
        config.encoder_hidden_size = kHidden;
        config.pred_hidden_size = 2;
        config.pred_num_layers = 1;
        config.vocab_size = 3;
        config.blank_id = 0;
        config.num_mel_bins = 2;
        config.mel_n_fft = config.mel_win_length = 16;
        config.mel_hop_length = 8;
        config.mel_chunk_length = 1;
        config.mel_length = 32;
        config.encoder_layers = kLayers;
        config.att_context_left = kCacheLeft;
        config.att_context_right = kRightContext;
        config.streaming_cache_left = kCacheLeft;
        config.streaming_time_cache = kTimeCache;
        config.supported_right_contexts = {kRightContext};
        MelFilterbank bank;
        bank.n_freq_bins = 9;
        bank.n_mel_bins = 2;
        bank.data.assign(18, 1.0F);
        pipeline = std::make_unique<RnntPipeline>(
            std::move(whole_clip_encoder), std::move(predictor), std::move(joint), nullptr,
            std::map<int32_t, std::vector<char>>{{kRightContext, {'p'}}}, &backend,
            ModuleCreateOptions{}, std::map<int32_t, std::vector<char>>{{kRightContext, {'p'}}},
            config, bank, test_stream());
    }

    // `pad_and_drop` makes a stream that runs every step on the steady
    // encoder; otherwise its first step runs on the first-step encoder.
    std::unique_ptr<ITranscriptionStream> make_stream(bool pad_and_drop = false) {
        TranscriptionStreamConfig config;
        config.att_context_right = kRightContext;
        config.pad_and_drop_preencoded = pad_and_drop;
        return pipeline->create_transcription_stream(config);
    }

    int encoder_steps() const {
        int steps = 0;
        for (const auto& encoder : encoders)
            steps += encoder->steps;
        return steps;
    }

    int bind_calls() const {
        int calls = 0;
        for (const auto& encoder : encoders)
            calls += encoder->module->bind_calls;
        return calls;
    }

    std::vector<std::shared_ptr<EncoderRecord>> encoders;
    StepHistory history; // history.feeding names the stream being fed
    BindFault bind_fault;
    std::vector<std::vector<float>> frames; // every encoder frame the joint was given
    int joint_calls{0};
    int fail_joint_call{0};
    int predictor_runs{0};
    int fills_at_predictor{0}; // cudaMemsetAsync calls made before the predictor last ran
    std::vector<float> pred_output, next_h, next_c, logits;
    Backend backend;
    std::unique_ptr<RnntPipeline> pipeline;
};

// Fixed pseudo-random audio; `voice` selects the sequence.
std::vector<float> make_audio(std::size_t samples, uint32_t voice) {
    std::vector<float> audio(samples);
    uint32_t state = 12345U + 977U * voice;
    for (auto& sample : audio) {
        state = state * 1664525U + 1013904223U;
        sample = static_cast<float>(state >> 8) / 16777216.0F - 0.5F;
    }
    return audio;
}

constexpr std::size_t kChunk = 300;

// What a scenario produced: the tokens after every call, and errors by position.
struct Trace {
    std::vector<std::vector<int32_t>> tokens;
    std::vector<std::string> errors;
};

// Runs one call on the stream named `name` and records its tokens or its error.
void call_stream(Rig& rig, int name, Trace& trace,
                 const std::function<TranscriptionStreamResult()>& call) {
    rig.history.feeding = name;
    try {
        trace.tokens.push_back(call().token_ids);
    } catch (const std::runtime_error& error) {
        trace.errors.push_back(std::to_string(trace.tokens.size()) + ": " + error.what());
    }
}

void feed(Rig& rig, int name, ITranscriptionStream& stream, const std::vector<float>& audio,
          std::size_t chunks, Trace& trace) {
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        call_stream(rig, name, trace, [&] {
            return stream.accept_audio(audio.data() + chunk * kChunk, static_cast<int32_t>(kChunk),
                                       false);
        });
    }
}

void finish(Rig& rig, int name, ITranscriptionStream& stream, Trace& trace) {
    call_stream(rig, name, trace, [&] { return stream.finish(); });
}

bool has_error(const Trace& trace, const std::string& text) {
    return std::any_of(trace.errors.begin(), trace.errors.end(), [&](const std::string& error) {
        return error.find(text) != std::string::npos;
    });
}

// One stream to its end, then the same stream again after reset(), then a
// second stream made after the first is gone. Each start has its own name:
// the caches start at zero three times.
Trace run_requests(Rig& rig) {
    Trace trace;
    const auto first = make_audio(5 * kChunk, 1);
    const auto second = make_audio(4 * kChunk, 2);
    {
        auto stream = rig.make_stream();
        feed(rig, 1, *stream, first, 5, trace);
        finish(rig, 1, *stream, trace);
        stream->reset();
        feed(rig, 2, *stream, second, 4, trace);
        finish(rig, 2, *stream, trace);
    }
    auto stream = rig.make_stream();
    feed(rig, 3, *stream, first, 5, trace);
    finish(rig, 3, *stream, trace);
    return trace;
}

// reset() in the middle of a stream, after an odd and after an even number of
// steps: either generation of the caches may be the one the next step reads.
Trace run_resets(Rig& rig) {
    Trace trace;
    const auto audio = make_audio(5 * kChunk, 6);
    auto stream = rig.make_stream();
    int name = 0;
    bool odd = false;
    bool even = false;
    for (const std::size_t chunks : {3U, 1U, 2U, 4U, 5U}) {
        const std::size_t before = rig.history.steps.size();
        feed(rig, ++name, *stream, audio, chunks, trace);
        const std::size_t steps = rig.history.steps.size() - before;
        odd = odd || steps % 2 == 1;
        even = even || (steps != 0 && steps % 2 == 0);
        stream->reset();
    }
    feed(rig, ++name, *stream, audio, 5, trace);
    finish(rig, name, *stream, trace);
    check(odd && even, "resets: the scenario resets after an odd and after an even step count");
    return trace;
}

// Two streams of one pipeline, a chunk at a time in turn.
Trace run_interleaved(Rig& rig) {
    Trace trace;
    const auto first = make_audio(5 * kChunk, 3);
    const auto second = make_audio(5 * kChunk, 4);
    auto one = rig.make_stream();
    auto two = rig.make_stream();
    for (std::size_t chunk = 0; chunk < 5; ++chunk) {
        const std::vector<float> a(first.begin() + chunk * kChunk,
                                   first.begin() + (chunk + 1) * kChunk);
        const std::vector<float> b(second.begin() + chunk * kChunk,
                                   second.begin() + (chunk + 1) * kChunk);
        feed(rig, 1, *one, a, 1, trace);
        feed(rig, 2, *two, b, 1, trace);
    }
    finish(rig, 2, *two, trace);
    finish(rig, 1, *one, trace);
    return trace;
}

// The joint fails in the middle of a step; the stream is then fed to its end.
Trace run_failed_decode(Rig& rig) {
    Trace trace;
    const auto audio = make_audio(5 * kChunk, 5);
    auto stream = rig.make_stream();
    rig.fail_joint_call = rig.joint_calls + 9;
    feed(rig, 1, *stream, audio, 5, trace);
    finish(rig, 1, *stream, trace);
    return trace;
}

// A stream that runs every step on the steady encoder.
Trace run_pad_and_drop(Rig& rig) {
    Trace trace;
    const auto audio = make_audio(5 * kChunk, 13);
    auto stream = rig.make_stream(true);
    feed(rig, 1, *stream, audio, 5, trace);
    finish(rig, 1, *stream, trace);
    check(rig.encoders.size() == 1, "pad and drop: the first-step encoder is never loaded");
    return trace;
}

// An encoder step's enqueue fails; the stream is then fed to its end.
Trace run_failed_enqueue(Rig& rig) {
    Trace trace;
    const auto audio = make_audio(5 * kChunk, 7);
    auto stream = rig.make_stream();
    rig.history.fail_step = rig.history.attempts + 4;
    feed(rig, 1, *stream, audio, 5, trace);
    finish(rig, 1, *stream, trace);
    return trace;
}

bool same_values(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool same_frames(const std::vector<std::vector<float>>& a,
                 const std::vector<std::vector<float>>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), same_values);
}

// Checks every step's encoder output against the cache chain recomputed from
// the steps' own inputs: a stream's caches start at zero, and a step reads
// what the last completed step of its stream wrote. This is independent of
// how the stream carries its caches, on the host or on the device. A stream's
// first step must also mask every cache position. Returns the number of
// incomplete steps.
std::size_t check_cache_chain(const Rig& rig, const std::string& scenario) {
    struct Caches {
        std::vector<float> channel = std::vector<float>(kChannelCount, 0.0F);
        std::vector<float> time = std::vector<float>(kTimeCount, 0.0F);
        bool started{false};
    };
    std::map<int, Caches> streams;
    std::size_t wrong = 0;
    std::size_t open_at_start = 0;
    std::size_t incomplete = 0;
    for (const auto& step : rig.history.steps) {
        Caches& caches = streams[step.stream];
        Caches next;
        next.started = true;
        std::vector<float> expected;
        std::vector<float> mel = step.mel;
        std::vector<float> mask = step.mask;
        Tensor mel_tensor = tensor_of(mel);
        Tensor mask_tensor = tensor_of(mask);
        mask_tensor.shape = step.mask_shape;
        encoder_step(caches.channel.data(), caches.time.data(), mel_tensor, mask_tensor,
                     next.channel.data(), next.time.data(), expected);
        wrong += same_values(expected, step.encoder_output) ? 0 : 1;
        if (!caches.started) {
            const auto keys = static_cast<std::size_t>(step.mask_shape.at(2));
            for (std::size_t i = 0; i < mask.size(); ++i)
                open_at_start += (i % keys < kCacheLeft && mask[i] == 0.0F) ? 1 : 0;
        }
        if (step.completed)
            caches = std::move(next);
        else
            ++incomplete;
    }
    check(wrong == 0, scenario +
                          ": every step reads the caches its stream's last completed step "
                          "wrote, zero at the start (" +
                          std::to_string(wrong) + " of " +
                          std::to_string(rig.history.steps.size()) + " steps differ)");
    check(open_at_start == 0, scenario + ": a stream's first step masks every cache position");
    return incomplete;
}

void check_device_engines(const Rig& rig, const std::string& scenario) {
    for (const auto& encoder : rig.encoders) {
        check(!encoder->given_cache_tensors,
              scenario + ": no cache tensor is handed to forward() as a host tensor");
        check(!encoder->unbound_step, scenario + ": every step runs on four bound buffers");
        check(!encoder->overlapping_step, scenario + ": a step never reads and writes one buffer");
    }
}

// Runs `scenario` with host-tensor caches and with device caches and requires
// the same decoding from both. `incomplete` is the number of steps
// the scenario's failing calls leave incomplete.
void compare_scenario(const std::string& name, const std::function<Trace(Rig&)>& scenario,
                      int min_steps, std::size_t expected_errors, std::size_t incomplete) {
    const int allocations_before = g_cuda.allocations;
    const int fills_before = g_cuda.fills;
    Trace host_trace;
    int host_steps = 0;
    std::vector<std::vector<float>> host_frames;
    {
        Rig host({EncoderSpec{}}, kHostBackend);
        host_trace = scenario(host);
        check(g_cuda.allocations == allocations_before && g_cuda.fills == fills_before &&
                  host.bind_calls() == 0,
              name + ": on another backend than the standard one a stream makes no CUDA call "
                     "and binds nothing");
        for (const auto& encoder : host.encoders)
            check(!encoder->bad_host_tensor, name + ": host cache tensors are usable");
        check(host.encoder_steps() >= min_steps, name + ": the scenario runs enough encoder steps");
        check(check_cache_chain(host, name + ", host caches") == incomplete,
              name + ": the failing calls leave the expected steps incomplete");
        check(host_trace.errors.size() == expected_errors,
              name + ": the host-tensor run fails where the scenario makes it fail");
        host_steps = host.encoder_steps();
        host_frames = host.frames;
        std::cout << name << ", host caches: " << host_steps << " encoder steps, "
                  << host_frames.size() << " frames to the joint, " << host.bind_calls()
                  << " binds\n";
    }

    Rig device({EncoderSpec{}});
    const Trace device_trace = scenario(device);
    check(device.bind_calls() == 4 * device.history.attempts,
          name + ": the four cache tensors are bound in every step");
    check_device_engines(device, name);
    check(g_cuda.fills > fills_before && g_cuda.fills_on_other_stream == 0,
          name + ": the caches are cleared on the encoders' CUDA stream");
    check(check_cache_chain(device, name + ", device caches") == incomplete,
          name + ": the failing calls leave the expected steps incomplete on the device too");
    check(device.encoder_steps() == host_steps,
          name + ": both runs take the same number of encoder steps");
    check(device_trace.tokens == host_trace.tokens,
          name + ": device caches decode the same tokens as host caches");
    check(device_trace.errors == host_trace.errors,
          name + ": device caches fail exactly where host caches fail");
    check(same_frames(device.frames, host_frames),
          name + ": the joint is given the same encoder frames, bit for bit");
    check(!host_trace.tokens.empty() && !host_trace.tokens.back().empty(),
          name + ": the scenario decodes tokens");
    std::cout << name << ", device caches: " << device.encoder_steps() << " encoder steps, "
              << device.frames.size() << " frames to the joint, " << device.bind_calls()
              << " binds\n";
}

void test_same_decoding() {
    compare_scenario("requests", run_requests, 12, 0, 0);
    compare_scenario("resets", run_resets, 12, 0, 0);
    compare_scenario("interleaved streams", run_interleaved, 12, 0, 0);
    compare_scenario("failed decode", run_failed_decode, 6, 1, 1);
    compare_scenario("failed enqueue", run_failed_enqueue, 6, 1, 0);
    compare_scenario("pad and drop", run_pad_and_drop, 6, 0, 0);
}

// The tokens of one stream fed five chunks of `voice` and finished.
std::vector<int32_t> decode(Rig& rig, int name, uint32_t voice, Trace& trace) {
    auto stream = rig.make_stream();
    feed(rig, name, *stream, make_audio(5 * kChunk, voice), 5, trace);
    finish(rig, name, *stream, trace);
    return trace.tokens.empty() ? std::vector<int32_t>{} : trace.tokens.back();
}

// A bind the module rejects, at each of a step's four binds: the step does not
// run, and the stream decodes the same tokens once it is fed on.
void test_rejected_bind() {
    Trace reference_trace;
    Rig reference({EncoderSpec{}}, kHostBackend);
    const auto expected = decode(reference, 1, 8, reference_trace);
    for (int position = 1; position <= 4; ++position) {
        Rig rig({EncoderSpec{}});
        Trace trace;
        rig.bind_fault.fail_call = 8 + position; // in the third encoder step
        const auto tokens = decode(rig, 1, 8, trace);
        const std::string name = "rejected bind " + std::to_string(position);
        check(trace.errors.size() == 1 && has_error(trace, "rejected external buffer"),
              name + ": the call whose bind is rejected fails");
        check(tokens == expected && !expected.empty(),
              name + ": the stream decodes the same tokens afterwards");
        check(rig.encoder_steps() == reference.encoder_steps(),
              name + ": no encoder step is lost or repeated");
        check(check_cache_chain(rig, name) == 0, name + ": the rejected step did not run");
        check_device_engines(rig, name);
    }
}

void test_buffers_are_lent_and_reused() {
    {
        Rig rig({EncoderSpec{}});
        check(g_cuda.allocations == 0, "a pipeline without a stream allocates nothing");
        {
            auto stream = rig.make_stream();
            check(rig.encoders.size() == 1,
                  "making a stream loads only the encoder it starts with");
            check(g_cuda.allocations == 4 && g_cuda.live.size() == 4,
                  "a stream holds two generations of two buffers");
            std::map<std::size_t, int> sizes;
            for (const auto& buffer : g_cuda.live)
                ++sizes[buffer.second];
            check(sizes[kChannelCount * sizeof(float)] == 2 &&
                      sizes[kTimeCount * sizeof(float)] == 2,
                  "the buffers have the sizes of the two cache tensors");
            check(g_cuda.fills == 2 && rig.fills_at_predictor == 2,
                  "a new stream clears the generation its first step reads, before its "
                  "predictor runs");
            stream->reset();
            check(g_cuda.fills == 4 && rig.fills_at_predictor == 4,
                  "reset() clears that generation again, before the predictor runs");
        }
        check(g_cuda.live.size() == 4, "a finished stream's buffers stay with the pipeline");
        {
            auto second = rig.make_stream();
            check(g_cuda.allocations == 4, "the next stream reuses them");
            check(g_cuda.fills == 6, "and clears them, since they hold the last stream's caches");
            auto third = rig.make_stream();
            check(g_cuda.allocations == 8 && g_cuda.live.size() == 8,
                  "a second live stream gets buffers of its own");
        }
        auto fourth = rig.make_stream();
        check(g_cuda.allocations == 8, "two returned sets serve later streams");

        // The buffers a live stream holds do not go with the pipeline.
        rig.pipeline.reset();
        check(g_cuda.live.size() == 8, "a live stream keeps the pool's buffers alive");
    }
    check(g_cuda.live.empty(), "every buffer is freed once the pipeline and its streams are gone");
}

bool stream_is_refused(Rig& rig) {
    try {
        rig.make_stream();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_failed_allocation_and_fill() {
    Rig rig({EncoderSpec{}});
    for (int failing = 1; failing <= 4; ++failing) {
        g_cuda.fail_allocation = g_cuda.allocations + failing;
        check(stream_is_refused(rig), "a failed cache allocation refuses the stream");
        check(g_cuda.live.empty(), "a refused stream leaves no buffer behind");
    }
    g_cuda.fail_allocation = 0;
    for (int failing = 1; failing <= 2; ++failing) {
        g_cuda.fail_fill = g_cuda.fills + failing;
        check(stream_is_refused(rig), "a failed fill refuses the stream");
        check(g_cuda.live.empty(), "a stream refused at its fill leaves no buffer behind");
    }
    g_cuda.fail_fill = 0;

    Trace trace;
    check(!decode(rig, 1, 9, trace).empty() && trace.errors.empty(),
          "the pipeline serves a stream after refusals");
}

// A fill that fails in reset(): the stream runs no step on caches that were
// not cleared, and works again once a reset succeeds.
void test_failed_fill_in_reset() {
    Trace reference_trace;
    Rig reference({EncoderSpec{}}, kHostBackend);
    const auto expected = decode(reference, 1, 10, reference_trace);
    for (int failing = 1; failing <= 2; ++failing) {
        const std::string name = "failed fill " + std::to_string(failing) + " in reset()";
        Rig rig({EncoderSpec{}});
        Trace trace;
        auto stream = rig.make_stream();
        const auto audio = make_audio(5 * kChunk, 10);
        feed(rig, 1, *stream, make_audio(3 * kChunk, 11), 3, trace);
        g_cuda.fail_fill = g_cuda.fills + failing;
        bool refused = false;
        try {
            stream->reset();
        } catch (const std::runtime_error&) {
            refused = true;
        }
        g_cuda.fail_fill = 0;
        check(refused, name + ": reset() reports the failure");
        const int attempts = rig.history.attempts;
        feed(rig, 2, *stream, audio, 1, trace);
        check(has_error(trace, "were not cleared") && rig.history.attempts == attempts,
              name + ": the stream runs no step on caches that were not cleared");
        stream->reset();
        Trace after;
        feed(rig, 3, *stream, audio, 5, after);
        finish(rig, 3, *stream, after);
        check(after.errors.empty() && !expected.empty() && after.tokens.back() == expected,
              name + ": after a reset that succeeds the stream decodes like a new one");
        check(check_cache_chain(rig, name) == 0, name + ": every step read the right caches");
    }
}

// Whether a stream over the encoders `first` and `second`, in the order the
// pipeline loads them, is refused when it is made (0 steps) or at the second
// encoder's first step (1 step), with `message`.
void check_refusal(const std::string& name, const EncoderSpec& first, const EncoderSpec& second,
                   bool at_creation, const std::string& message, bool pad_and_drop = false) {
    Rig rig({first, second});
    Trace trace;
    std::unique_ptr<ITranscriptionStream> stream;
    const int allocations = g_cuda.allocations;
    const int fills = g_cuda.fills;
    try {
        stream = rig.make_stream(pad_and_drop);
    } catch (const std::runtime_error& error) {
        trace.errors.push_back(error.what());
    }
    check((stream == nullptr) == at_creation,
          name + (at_creation ? ": refused when the stream is made" : ": the stream is made"));
    if (stream == nullptr) {
        check(g_cuda.allocations == allocations && g_cuda.fills == fills && g_cuda.live.empty() &&
                  rig.predictor_runs == 0,
              name + ": refused before a buffer is taken or the predictor runs");
    } else {
        feed(rig, 1, *stream, make_audio(3 * kChunk, 12), 3, trace);
    }
    check(has_error(trace, message), name + ": refused with '" + message + "'");
    check(rig.encoder_steps() == (at_creation ? 0 : 1) &&
              rig.bind_calls() == 4 * rig.encoder_steps(),
          name + ": nothing is bound or run on the refused encoder");
}

void test_mismatched_encoders_are_refused() {
    const std::string mismatch = "does not match the runtime configuration";
    for (const auto& tensor : {kChannel, kTime, kChannelNext, kTimeNext}) {
        EncoderSpec larger;
        ++larger.tensors[tensor].shape[1];
        check_refusal(tensor + " of another shape, first encoder", larger, EncoderSpec{}, true,
                      mismatch);
        check_refusal(tensor + " of another shape, steady encoder", EncoderSpec{}, larger, false,
                      mismatch);
        EncoderSpec half;
        half.tensors[tensor].dtype = DType::kFloat16;
        check_refusal(tensor + " in FP16", half, EncoderSpec{}, true, mismatch);
        EncoderSpec without;
        without.tensors.erase(tensor);
        check_refusal(tensor + " not reported, first encoder", without, EncoderSpec{}, true,
                      mismatch);
        check_refusal(tensor + " not reported, steady encoder", EncoderSpec{}, without, false,
                      mismatch);
        EncoderSpec reversed;
        reversed.tensors[tensor].is_input = !reversed.tensors[tensor].is_input;
        check_refusal(tensor + " reported in the other direction", reversed, EncoderSpec{}, true,
                      mismatch);
    }
    for (const auto& tensor : {kChannel, kTime}) {
        EncoderSpec dynamic;
        dynamic.tensors[tensor].dynamic = true;
        check_refusal(tensor + " dynamic", dynamic, EncoderSpec{}, true, mismatch);
    }
    EncoderSpec elsewhere;
    elsewhere.stream = nullptr;
    check_refusal("first encoder on another CUDA stream", elsewhere, EncoderSpec{}, true,
                  "another CUDA stream");
    check_refusal("steady encoder on another CUDA stream", EncoderSpec{}, elsewhere, false,
                  "another CUDA stream");
    check_refusal("no cache tensor reported, first encoder", no_tensors(), EncoderSpec{}, true,
                  mismatch);
    check_refusal("no cache tensor reported, steady encoder", EncoderSpec{}, no_tensors(), false,
                  mismatch);
    EncoderSpec captured;
    captured.cuda_graph = true;
    check_refusal("first encoder under a CUDA graph", captured, EncoderSpec{}, true, "CUDA graph");
    check_refusal("steady encoder under a CUDA graph", EncoderSpec{}, captured, false,
                  "CUDA graph");
    // A stream that runs every step on the steady encoder is checked against
    // that one: it is the first the pipeline loads for it.
    EncoderSpec larger;
    ++larger.tensors[kChannel].shape[1];
    check_refusal("pad and drop, steady encoder of another shape", larger, EncoderSpec{}, true,
                  mismatch, true);
    check(g_cuda.live.empty(), "refused streams leave no buffer behind");
}

// On another backend than the standard TensorRT one a stream gives any encoder
// host tensors, as before: also one that the standard backend would refuse.
void test_other_backends_take_any_encoder() {
    Trace reference_trace;
    Rig reference({EncoderSpec{}}, kHostBackend);
    const auto expected = decode(reference, 1, 14, reference_trace);
    EncoderSpec odd;
    odd.stream = nullptr;
    odd.cuda_graph = true;
    ++odd.tensors[kChannel].shape[1];
    odd.tensors[kTime].dynamic = true;
    odd.tensors[kChannelNext].dtype = DType::kFloat16;
    odd.tensors.erase(kTimeNext);
    const std::vector<std::string> backends{kHostBackend, "TRT", "trt2", "other", ""};
    for (const std::string& backend : backends) {
        const std::string name = "backend '" + backend + "'";
        Rig rig({odd}, backend);
        Trace trace;
        const auto tokens = decode(rig, 1, 14, trace);
        check(trace.errors.empty() && !expected.empty() && tokens == expected,
              name + ": a stream decodes with any encoder, as it did before");
        check(g_cuda.allocations == 0 && g_cuda.fills == 0 && rig.bind_calls() == 0,
              name + ": no CUDA call and no bind");
        check(check_cache_chain(rig, name) == 0, name + ": every step read the right caches");
    }
}

void reset_cuda_counts() {
    check(g_cuda.live.empty(), "no buffer outlives its test");
    check(g_cuda.fills_on_other_stream == 0, "no fill is issued on another CUDA stream");
    g_cuda.allocations = 0;
    g_cuda.fills = 0;
}

} // namespace

int main() {
    // The family library must resolve its CUDA calls to this file's stand-ins,
    // which takes an executable that exports its symbols. Otherwise the stream
    // would allocate from a CUDA runtime, and the stand-in encoders could not
    // read what it binds.
    if (dlsym(RTLD_DEFAULT, "cudaMalloc") != reinterpret_cast<void*>(&cudaMalloc)) {
        std::cerr << "FAIL: the family library would not call this test's CUDA stand-ins; "
                     "link the test with its symbols exported\n";
        return 1;
    }
    try {
        {
            Rig rig({EncoderSpec{}});
            auto stream = rig.make_stream();
            check(g_cuda.allocations == 4,
                  "a stream over cache-binding encoders allocates through the stand-ins");
        }
        const std::vector<std::function<void()>> tests{
            test_same_decoding,
            test_rejected_bind,
            test_buffers_are_lent_and_reused,
            test_failed_allocation_and_fill,
            test_failed_fill_in_reset,
            test_mismatched_encoders_are_refused,
            test_other_backends_take_any_encoder,
        };
        for (const auto& test : tests) {
            reset_cuda_counts();
            test();
        }
        reset_cuda_counts();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " streaming cache check(s) failed\n";
        return 1;
    }
    std::cout << "Streaming encoder cache tests passed\n";
    return 0;
}

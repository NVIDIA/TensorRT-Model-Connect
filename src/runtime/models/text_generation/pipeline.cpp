#include "runtime/models/text_generation/pipeline.h"

#include "runtime/core/trt_common.h"
#include "runtime/core/trt_engine_lifecycle.h"
#include "trtmc/runtime/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {

struct StepTraceConfig {
    bool enabled{false};
    std::string path;
    int32_t start_position{0};
    int32_t end_position{std::numeric_limits<int32_t>::max()};
    int32_t top_k{8};
};

// Process-wide step-trace state. Populated once from the resolved
// ConfigBundle by `apply_text_trace_config_from_registry` (below), called
// from the decoder plugin before pipeline construction. Replaces the
// TRTMC_TEXT_STEP_TRACE_* environment variables, which are now deleted.
StepTraceConfig& mutable_step_trace_config() {
    static StepTraceConfig cfg;
    return cfg;
}

const StepTraceConfig& step_trace_config() {
    return mutable_step_trace_config();
}

} // namespace

// Called from decoder_plugin::create() with values resolved from
// ctx.runtime_config for the "text_trace" namespace. An empty path keeps
// tracing disabled. When a non-empty path is supplied, this truncates the
// target file so repeated runs don't concatenate. Not re-entrant; the
// caller serializes creation.
void apply_text_trace_config_from_registry(const std::string& path, int32_t start_position,
                                           int32_t end_position, int32_t top_k) {
    StepTraceConfig& cfg = mutable_step_trace_config();
    cfg.path = path;
    cfg.enabled = !path.empty();
    cfg.start_position = start_position;
    cfg.end_position = end_position;
    cfg.top_k = std::max(int32_t{1}, top_k);
    if (cfg.enabled) {
        std::ofstream clear(cfg.path, std::ios::trunc);
    }
}

namespace {

std::vector<int32_t> top_logit_indices(const std::vector<float>& logits, int32_t top_n) {
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + top_n, order.end(), [&logits](int32_t lhs, int32_t rhs) {
            if (logits[static_cast<std::size_t>(lhs)] != logits[static_cast<std::size_t>(rhs)]) {
                return logits[static_cast<std::size_t>(lhs)] >
                       logits[static_cast<std::size_t>(rhs)];
            }
            return lhs < rhs;
        });
    return order;
}

void write_step_trace_line(std::ostream& out, int32_t position_before, int32_t token_id,
                           int32_t decoder_idx, int32_t rows_before, int32_t rows_after,
                           const std::vector<float>& logits, const std::vector<int32_t>& order,
                           int32_t top_n) {
    out << "{\"position_before\":" << position_before << ",\"token_id\":" << token_id
        << ",\"decoder_idx\":" << decoder_idx << ",\"rows_before\":" << rows_before
        << ",\"rows_after\":" << rows_after << ",\"argmax_token\":" << order.front()
        << ",\"argmax_logit\":" << logits[static_cast<std::size_t>(order.front())]
        << ",\"top_ids\":[";
    for (int32_t i = 0; i < top_n; ++i) {
        if (i > 0)
            out << ',';
        out << order[static_cast<std::size_t>(i)];
    }
    out << "],\"top_logits\":[";
    for (int32_t i = 0; i < top_n; ++i) {
        if (i > 0)
            out << ',';
        out << logits[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
    out << "]}\n";
}

void maybe_append_step_trace(int32_t position_before, int32_t token_id, int32_t decoder_idx,
                             int32_t rows_before, int32_t rows_after,
                             const std::vector<float>& logits) {
    const auto& cfg = step_trace_config();
    if (!cfg.enabled || position_before < cfg.start_position || position_before > cfg.end_position)
        return;
    if (logits.empty())
        return;
    const int32_t top_n = std::min<int32_t>(cfg.top_k, static_cast<int32_t>(logits.size()));
    const auto order = top_logit_indices(logits, top_n);
    std::ofstream out(cfg.path, std::ios::app);
    if (!out)
        return;
    write_step_trace_line(out, position_before, token_id, decoder_idx, rows_before, rows_after,
                          logits, order, top_n);
}

bool contains_boxed_answer(const std::string& text) {
    const std::string marker = "\\boxed{";
    const auto start = text.find(marker);
    if (start == std::string::npos)
        return false;
    return text.find('}', start + marker.size()) != std::string::npos;
}

bool contains_final_answer(const std::string& text) {
    const std::string marker = "Final answer:";
    const auto start = text.find(marker);
    if (start == std::string::npos)
        return false;
    for (std::size_t i = start + marker.size(); i < text.size(); ++i) {
        if (!std::isspace(static_cast<unsigned char>(text[i])))
            return true;
    }
    return false;
}

std::vector<TextGenerationPipeline::DecoderContext>
single_decoder_context(std::unique_ptr<TrtModule> decoder) {
    std::vector<TextGenerationPipeline::DecoderContext> decoders;
    decoders.push_back(TextGenerationPipeline::DecoderContext{0, std::move(decoder)});
    return decoders;
}
} // namespace

TextGenerationPipeline::TextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                                               std::unique_ptr<IInferenceState> state,
                                               TextGenConfig config, cudaStream_t stream,
                                               std::shared_ptr<ITokenizer> tokenizer,
                                               std::string model_id_str,
                                               std::unique_ptr<ISampler> sampler)
    : TextGenerationPipeline(single_decoder_context(std::move(decoder)), std::move(state),
                             std::move(config), stream, std::move(tokenizer),
                             std::move(model_id_str), std::move(sampler)) {}

TextGenerationPipeline::TextGenerationPipeline(
    std::vector<DecoderContext> decoders, std::unique_ptr<IInferenceState> state,
    TextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<ISampler> sampler, std::unique_ptr<TrtModule> prefill)
    : decoders_(std::move(decoders)), prefill_(std::move(prefill)), state_(std::move(state)),
      config_(std::move(config)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), sampler_(std::move(sampler)),
      logits_output_name_(config_.logits_output_name) {
    if (decoders_.empty()) {
        throw std::runtime_error("TextGenerationPipeline: no decoder modules");
    }
    for (const auto& decoder_ctx : decoders_) {
        if (!decoder_ctx.module || !decoder_ctx.module->ok()) {
            throw std::runtime_error("TextGenerationPipeline: invalid decoder module");
        }
    }
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("TextGenerationPipeline: invalid inference state");
    }

    // CUDA Graphs: capture TRT kernels on first step, replay on subsequent
    // steps. Disabled via --set runtime.disable_cuda_graph=true (replaces
    // the deleted TRTMC_DISABLE_CUDA_GRAPH env var).
    if (!config_.disable_cuda_graph) {
        for (auto& decoder_ctx : decoders_)
            decoder_ctx.module->enable_cuda_graph();
    }

    // GPU-side argmax is only valid for truly greedy decoding. Populated
    // from runtime.prefer_gpu_greedy (replaces the deleted TRTMC_GPU_ARGMAX
    // env var). We record the preference here and instantiate per-call
    // when the requested sampling parameters are actually greedy.
    prefer_gpu_greedy_ = config_.prefer_gpu_greedy;
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer, const TextGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && config.chat_template_format != ChatTemplateFormat::kNone) {
        effective = apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult TextGenerationPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_) {
        throw std::runtime_error("TextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = sampling_params_from_config(cfg, eos);
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg);

    // Decode only the NEW tokens (skip input)
    std::vector<int32_t> new_tokens(timed.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    timed.token_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens), timed.prefill_ms, timed.decode_ms};
}

TextGenerationPipeline::GenerationResult
TextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                     const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

std::unique_ptr<ISampler> TextGenerationPipeline::make_step_sampler(const SamplingParams& params) {
    const bool greedy_params =
        (params.temperature < 1e-6F) ||
        (params.top_k <= 1 && params.top_p >= 1.0F && params.min_p <= 0.0F && params.seed < 0);
    if (prefer_gpu_greedy_ && greedy_params) {
        if (auto gpu = create_gpu_greedy_sampler(stream_))
            return gpu;
    }
    return create_sampler(params);
}

// Helper: gather per-layer present_k/present_v device pointers from the
// prefill TrtModule. Returns false if any layer's tensor is missing — in
// that case the caller falls back to the per-token decode loop.
namespace {
bool gather_prefill_kv_pointers(TrtModule& prefill, const TextGenConfig& cfg,
                                std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t i = 0; i < cfg.num_layers; ++i) {
        const auto li = static_cast<std::size_t>(i);
        pk[li] = prefill.device_ptr(expand_layer_name(cfg.present_k_pattern, i));
        pv[li] = prefill.device_ptr(expand_layer_name(cfg.present_v_pattern, i));
        if (pk[li] == nullptr || pv[li] == nullptr)
            return false;
    }
    return true;
}

bool batched_prefill_supported(const TrtModule* prefill, const TextGenConfig& cfg, int32_t sq,
                               IInferenceState* state) {
    if (prefill == nullptr || sq <= 0)
        return false;
    if (cfg.prefill_max_length > 0 && sq > cfg.prefill_max_length)
        return false;
    if (cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        return false;
    return dynamic_cast<KvCache*>(state) != nullptr;
}
} // namespace

bool TextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                 std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    if (!batched_prefill_supported(prefill_.get(), config_, sq, state_.get()))
        return false;
    auto* kv = static_cast<KvCache*>(state_.get());

    // The prefill module shares the same external KV cache buffers as the
    // decode module(s), so we rebind the cache_k/cache_v inputs onto the
    // prefill execution context before running.
    kv->bind_cache_inputs(*prefill_);

    TensorMap inputs;
    Tensor tok_t;
    tok_t.data = const_cast<int32_t*>(input_ids.data());
    tok_t.shape = {static_cast<int64_t>(sq)};
    tok_t.dtype = DType::kInt32;
    inputs[config_.token_id_name] = tok_t;
    state_->prepare_step(inputs, sq);

    TensorMap outputs = prefill_->forward(inputs);
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        return false;

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& lt = logits_it->second;
    if (static_cast<std::size_t>(lt.numel()) < vocab)
        return false;
    logits.resize(vocab);
    std::memcpy(logits.data(), lt.data, vocab * sizeof(float));

    std::vector<const void*> pk, pv;
    if (!gather_prefill_kv_pointers(*prefill_, config_, pk, pv))
        return false;
    kv->write_prefill_kv(pk, pv, sq);
    if (trt_log_to_stderr_enabled())
        std::cerr << "[trtmc] Batched prefill (profile 0): " << sq << " tokens in one call\n";
    return true;
}

void TextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                         std::vector<float>& logits, bool gpu_sampling) {
    // Fast path: batched prefill engine writes K/V for the whole prompt in
    // one forward and returns last-token logits on host.
    if (!gpu_sampling && run_prefill_batched(input_ids, logits)) {
        state_->mark_prefill_complete();
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i) {
        if (gpu_sampling)
            run_step_device(input_ids[i]);
        else
            run_step(input_ids[i], logits);
    }
    const int32_t last_token = input_ids.back();
    if (gpu_sampling)
        run_step_device(last_token);
    else
        run_step(last_token, logits);
    state_->mark_prefill_complete();
}

TextGenerationPipeline::TimedGenResult
TextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                          int32_t max_new_tokens, const SamplingParams& params,
                                          const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    ISampler* active_sampler = sampler_.get();
    std::unique_ptr<ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = make_step_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_bound_ = false;
    for (auto& decoder_ctx : decoders_)
        decoder_ctx.module->reset_execution_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == LogitsLocation::DEVICE);
    const auto t0 = Clock::now();
    run_prefill(input_ids, logits, gpu_sampling);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler, params, output, logits, max_new_tokens, gpu_sampling, cfg,
                    static_cast<int32_t>(input_ids.size()));
    const auto t2 = Clock::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return TimedGenResult{std::move(output), prefill_ms, decode_ms};
}

bool TextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
                                                   int32_t prompt_token_count,
                                                   const GenerateConfig& cfg, int32_t steps,
                                                   int32_t stop_interval, bool is_eos) const {
    if (!cfg.stop_on_boxed_answer || !tokenizer_)
        return false;
    if ((steps % stop_interval) != 0 && !is_eos)
        return false;
    std::vector<int32_t> new_tokens(output.begin() + prompt_token_count, output.end());
    const std::string decoded = tokenizer_->decode(new_tokens);
    return contains_boxed_answer(decoded) || contains_final_answer(decoded);
}

void TextGenerationPipeline::log_decode_summary(int32_t steps, double ms) const {
    if (steps <= 0 || !trt_log_to_stderr_enabled())
        return;
    const double tps = steps * 1000.0 / ms;
    const bool cuda_graph_on =
        active_decoder_index_ >= 0 &&
        decoders_[static_cast<std::size_t>(active_decoder_index_)].module->cuda_graph_active();
    std::cerr << "[trtmc] Decode: " << steps << " tokens, " << ms << " ms, " << tps << " tok/s"
              << (cuda_graph_on ? " [CUDA Graph ON]" : "") << '\n';
}

int32_t TextGenerationPipeline::run_decode_loop(ISampler* sampler, const SamplingParams& params,
                                                std::vector<int32_t>& output,
                                                std::vector<float>& logits, int32_t max_new_tokens,
                                                bool gpu_sampling, const GenerateConfig& cfg,
                                                int32_t prompt_token_count) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    const auto decode_start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_ptr = gpu_sampling ? d_logits_ptr_ : logits.data();
        const SampleResult result = sampler->sample(sample_ptr, vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval,
                                  result.is_eos))
            break;
        if (result.is_eos)
            break;
        if (gpu_sampling)
            run_step_device(result.token_id);
        else
            run_step(result.token_id, logits);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    log_decode_summary(steps, ms);
    return steps;
}

int32_t TextGenerationPipeline::select_decoder_index(int32_t desired_rows) const {
    if (decoders_.size() == 1)
        return 0;

    int32_t fallback_idx = 0;
    int32_t fallback_rows = std::numeric_limits<int32_t>::max();
    for (std::size_t i = 0; i < decoders_.size(); ++i) {
        const int32_t kv_rows = decoders_[i].kv_rows;
        if (kv_rows == desired_rows)
            return static_cast<int32_t>(i);
        if (kv_rows > 0 && kv_rows >= desired_rows && kv_rows < fallback_rows) {
            fallback_rows = kv_rows;
            fallback_idx = static_cast<int32_t>(i);
        }
    }
    return fallback_idx;
}

TrtModule& TextGenerationPipeline::bind_decoder_for_step() {
    const int32_t desired_rows = std::max(state_->preferred_cache_rows(), 1);
    const int32_t next_idx = select_decoder_index(desired_rows);
    if (!state_bound_ || next_idx != active_decoder_index_) {
        active_decoder_index_ = next_idx;
        state_->bind_to(*decoders_[static_cast<std::size_t>(active_decoder_index_)].module);
        state_bound_ = true;
    }
    return *decoders_[static_cast<std::size_t>(active_decoder_index_)].module;
}

void TextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    const int32_t position_before = state_->position();
    const int32_t rows_before = std::max(state_->preferred_cache_rows(), 1);

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    TrtModule& decoder = bind_decoder_for_step();
    state_->prepare_step(inputs);

    TensorMap outputs = decoder.forward(inputs);

    auto it = outputs.find(logits_output_name_);
    if (it == outputs.end()) {
        throw std::runtime_error("TextGenerationPipeline: no '" + logits_output_name_ + "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
    maybe_append_step_trace(position_before, token_id, active_decoder_index_, rows_before,
                            std::max(state_->preferred_cache_rows(), 1), logits);
}

void TextGenerationPipeline::run_step_device(int32_t token_id) {
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    TrtModule& decoder = bind_decoder_for_step();
    state_->prepare_step(inputs);

    // Use forward_async + sync instead of forward() to skip the D2H output copy.
    // The GPU argmax kernel reads logits directly from the device buffer.
    decoder.forward_async(inputs);
    decoder.sync();

    // Get device pointer to logits output buffer (still on GPU).
    d_logits_ptr_ = static_cast<const float*>(decoder.device_ptr(logits_output_name_));

    state_->advance();
}

int32_t TextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc

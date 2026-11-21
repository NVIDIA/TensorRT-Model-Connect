/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/mtp_scheduler.h"

#include "families/qwen3_8/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {
constexpr float kMaskedScore = -1.0e4F;

int32_t argmax(const float* data, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] > data[best])
            best = i;
    }
    return static_cast<int32_t>(best);
}
} // namespace

Qwen38MtpScheduler::Qwen38MtpScheduler(
    std::unique_ptr<ITrtModule> mtp_module, std::unique_ptr<ITrtModule> draft_chain_module,
    std::unique_ptr<ITrtModule> multi_token_module, Qwen38HybridState& state, int32_t hidden_size,
    int32_t vocab_size, int32_t num_draft_tokens, cudaStream_t stream,
    std::unique_ptr<ITrtModule> prefill_module, int32_t prefill_chunk_len)
    : mtp_module_(std::move(mtp_module)), draft_chain_module_(std::move(draft_chain_module)),
      multi_token_module_(std::move(multi_token_module)),
      prefill_module_(std::move(prefill_module)), prefill_chunk_len_(prefill_chunk_len),
      state_(state), stream_(stream), hidden_size_(hidden_size), vocab_size_(vocab_size),
      num_draft_tokens_(num_draft_tokens) {
    if (!mtp_module_ || !mtp_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid MTP module");
    if (draft_chain_module_ && !draft_chain_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid MTP draft-chain module");
    if (!multi_token_module_ || !multi_token_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid multi-token module");
    if (num_draft_tokens_ < 1)
        throw std::runtime_error("Qwen38MtpScheduler: num_draft_tokens must be >= 1");

    int media_bindings = 0;
    for (const auto& input : mtp_module_->input_info()) {
        if (input.name == "visual_embedding") {
            if (input.shape != std::vector<int64_t>{1, hidden_size_} ||
                (input.dtype != DType::kFloat16 && input.dtype != DType::kBFloat16 &&
                 input.dtype != DType::kFloat32))
                throw std::runtime_error("Qwen MTP media embedding has invalid geometry/dtype");
            media_dtype_ = input.dtype;
            ++media_bindings;
        } else if (input.name == "use_visual_embedding" ||
                   input.name == "rotary_position_offsets") {
            const auto shape = input.name == "use_visual_embedding" ? std::vector<int64_t>{1, 1}
                                                                    : std::vector<int64_t>{3};
            if (input.shape != shape || input.dtype != DType::kInt32)
                throw std::runtime_error(
                    "Qwen MTP media selector/coordinates have invalid metadata");
            ++media_bindings;
        }
    }
    if (media_bindings != 0 && media_bindings != 3)
        throw std::runtime_error("Qwen MTP has an incomplete media input contract");
    media_inputs_ = media_bindings == 3;
    bool verifier_media = false;
    for (const auto& input : multi_token_module_->input_info()) {
        if (input.name == "rotary_position_offsets") {
            if (input.shape != std::vector<int64_t>{3} || input.dtype != DType::kInt32)
                throw std::runtime_error(
                    "Qwen MTP verifier has invalid rotary coordinate metadata");
            verifier_media = true;
        }
    }
    if (verifier_media != media_inputs_ || (media_inputs_ && draft_chain_module_))
        throw std::runtime_error("Qwen MTP draft/verifier media contracts disagree");
    if (media_inputs_)
        zero_visual_.resize(static_cast<std::size_t>(hidden_size_) * dtype_size(media_dtype_));

    Qwen38KvCache& main_kv = *state_.kv_cache();

    // All engines sharing cache storage must agree on its element type.
    // In particular, never bind one-byte FP8 buffers to a stale FP16 plan.
    auto check_dtype = [&](ITrtModule& module, const std::string& name) {
        if (module.tensor_dtype(name) != main_kv.dtype())
            throw std::runtime_error("Qwen38MtpScheduler: inconsistent cache dtype for " + name);
    };
    for (const char* name : {"mtp_cache_k", "mtp_cache_v", "mtp_present_k", "mtp_present_v"}) {
        check_dtype(*mtp_module_, name);
        if (draft_chain_module_)
            check_dtype(*draft_chain_module_, name);
    }
    for (int32_t i = 0; i < main_kv.num_layers(); ++i)
        for (const char* name : {"cache_k_", "cache_v_", "present_k_", "present_v_"})
            check_dtype(*multi_token_module_, std::string(name) + std::to_string(i));

    Qwen38KvCacheNames mtp_names;
    mtp_names.cache_k = {"mtp_cache_k"};
    mtp_names.cache_v = {"mtp_cache_v"};
    mtp_names.present_k = {"mtp_present_k"};
    mtp_names.present_v = {"mtp_present_v"};
    const int32_t draft_capacity =
        main_kv.max_length() + (draft_chain_module_ ? 0 : num_draft_tokens_ - 1);
    mtp_cache_ = std::make_unique<Qwen38KvCache>(1, draft_capacity, main_kv.kv_dim(), stream_,
                                                 main_kv.dtype(), mtp_names);
    if (!mtp_cache_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: failed to allocate MTP's own KV cache");
    if (!draft_chain_module_) {
        draft_scratch_cache_ = std::make_unique<Qwen38KvCache>(1, draft_capacity, main_kv.kv_dim(),
                                                               stream_, main_kv.dtype(), mtp_names);
        if (!draft_scratch_cache_->ok())
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate speculative draft cache");
    }

    const int32_t seq_len_val = num_draft_tokens_ + 1;
    multi_present_k_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    multi_present_v_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        multi_present_k_.emplace_back(std::vector<int64_t>{seq_len_val, main_kv.kv_dim()},
                                      main_kv.dtype(), stream_);
        multi_present_v_.emplace_back(std::vector<int64_t>{seq_len_val, main_kv.kv_dim()},
                                      main_kv.dtype(), stream_);
        if (!multi_present_k_.back().ok() || !multi_present_v_.back().ok())
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate multi-token present K/V scratch");
    }

    if (draft_chain_module_) {
        // Single entry each -- draft_chain_module_'s present_k/v are not
        // per-main-model-layer, they're MTP's own one layer (matching
        // build_mtp_draft_chain_engine's singular mtp_present_k/v names).
        draft_chain_present_k_.emplace_back(
            std::vector<int64_t>{num_draft_tokens_, main_kv.kv_dim()}, main_kv.dtype(), stream_);
        draft_chain_present_v_.emplace_back(
            std::vector<int64_t>{num_draft_tokens_, main_kv.kv_dim()}, main_kv.dtype(), stream_);
        if (!draft_chain_present_k_.back().ok() || !draft_chain_present_v_.back().ok()) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate draft-chain present K/V scratch");
        }
    }

    if (prefill_module_) {
        if (!prefill_module_->ok() || prefill_chunk_len_ < 1)
            throw std::runtime_error("Qwen38MtpScheduler: invalid MTP prefill module");
        prefill_present_k_.emplace_back(std::vector<int64_t>{prefill_chunk_len_, main_kv.kv_dim()},
                                        main_kv.dtype(), stream_);
        prefill_present_v_.emplace_back(std::vector<int64_t>{prefill_chunk_len_, main_kv.kv_dim()},
                                        main_kv.dtype(), stream_);
        if (!prefill_present_k_.back().ok() || !prefill_present_v_.back().ok())
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate MTP prefill present K/V scratch");
    }
}

void Qwen38MtpScheduler::bind_state() {
    mtp_cache_->bind_to(*mtp_module_);

    // draft_chain_module_ reads MTP's persistent cache (read-only, same
    // convention as the multi-token engine reading the main model's
    // cache) but its own present_k/v outputs are never committed back --
    // bind them to throwaway scratch (TensorRT requires every marked
    // output bound before execute, even unused ones).
    if (draft_chain_module_) {
        mtp_cache_->bind_cache_inputs(*draft_chain_module_);
        draft_chain_module_->bind_external("mtp_present_k", draft_chain_present_k_[0].data());
        draft_chain_module_->bind_external("mtp_present_v", draft_chain_present_v_[0].data());
    }

    if (prefill_module_) {
        prefill_module_->bind_external("mtp_present_k", prefill_present_k_[0].data());
        prefill_module_->bind_external("mtp_present_v", prefill_present_v_[0].data());
    }

    Qwen38KvCache& main_kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    main_kv.bind_cache_inputs(*multi_token_module_);

    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        multi_token_module_->bind_external("present_k" + suffix,
                                           multi_present_k_[static_cast<std::size_t>(i)].data());
        multi_token_module_->bind_external("present_v" + suffix,
                                           multi_present_v_[static_cast<std::size_t>(i)].data());
    }

    for (int32_t i = 0; i < ssm.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        multi_token_module_->bind_external("conv_state" + suffix, ssm.state_ptr(0, i));
        multi_token_module_->bind_external("ssm_state" + suffix, ssm.state_ptr(1, i));
        multi_token_module_->bind_external("present_conv" + suffix, ssm.present_ptr(0, i));
        multi_token_module_->bind_external("present_ssm" + suffix, ssm.present_ptr(1, i));
    }

    bound_ = true;
}

void Qwen38MtpScheduler::reset() {
    mtp_cache_->reset();
}

void Qwen38MtpScheduler::set_media_prompt(const Qwen38MediaPrompt* prompt) {
    if (prompt &&
        (!media_inputs_ || prompt->dtype != media_dtype_ || prompt->hidden_size != hidden_size_))
        throw std::invalid_argument("Qwen MTP media request disagrees with draft metadata");
    media_prompt_ = prompt;
}

void Qwen38MtpScheduler::sync() {
    // Drain every module even when one reports an error, before request-owned
    // visual host storage can be released by the pipeline's exception path.
    std::exception_ptr error;
    for (auto* module : {mtp_module_.get(), multi_token_module_.get(), draft_chain_module_.get()}) {
        if (!module)
            continue;
        try {
            module->sync();
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
    }
    if (error)
        std::rethrow_exception(error);
}

int32_t Qwen38MtpScheduler::draft(int32_t token, int32_t position, const float* hidden_state) {
    return draft_step(*mtp_cache_, token, position, hidden_state);
}

int32_t Qwen38MtpScheduler::draft_step(Qwen38KvCache& cache, int32_t token, int32_t position,
                                       const float* hidden_state) {
    if (!bound_)
        throw std::runtime_error("Qwen38MtpScheduler: bind_state() must be called before draft()");
    // cache.position() tracks CALL COUNT (how many rows have been
    // written to MTP's own cache so far -- used for mask validity and
    // cache row indexing), not the RoPE position of the token being
    // embedded. MTP's first-ever call always embeds the token at absolute
    // sequence position 1 (the main model embeds position 0 by itself,
    // with no MTP involvement), so the invariant is call_count ==
    // position - 1, not call_count == position.
    if (cache.position() != position - 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::draft: MTP cache position out of sync with caller's position "
            "(every main-model token must get exactly one draft()/warm-up call, in order)");
    }

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["next_token_id"] = token_t;

    Tensor hidden_t;
    hidden_t.data = const_cast<float*>(hidden_state);
    hidden_t.shape = {1, hidden_size_};
    hidden_t.dtype = DType::kFloat32;
    inputs["hidden_state"] = hidden_t;

    // prepare_step() writes both attention_mask (correct: call-count-based
    // validity) and position_id (WRONG here: it writes mtp_cache_'s own
    // call-count, not the caller's absolute `position`) -- overwrite
    // position_id right after with the correct value.
    cache.prepare_step(inputs, 1);
    Tensor position_t;
    position_t.data = &position;
    position_t.shape = {1};
    position_t.dtype = DType::kInt32;
    inputs["position_id"] = position_t;

    int32_t use_visual = 0;
    std::array<int32_t, 3> offsets{0, 0, 0};
    if (media_inputs_) {
        const uint8_t* features = zero_visual_.data();
        if (media_prompt_) {
            const auto& prompt = *media_prompt_;
            if (static_cast<std::size_t>(position) < prompt.positions.size()) {
                for (int axis = 0; axis < 3; ++axis)
                    offsets[axis] = prompt.positions[position][axis] - position;
                const auto row = prompt.feature_rows[position];
                if (row >= 0) {
                    use_visual = 1;
                    features = prompt.features.data() +
                               static_cast<std::size_t>(row) * zero_visual_.size();
                }
            } else {
                offsets.fill(prompt.decode_position -
                             static_cast<int32_t>(prompt.positions.size()));
            }
        }
        inputs["visual_embedding"] =
            Tensor{const_cast<uint8_t*>(features), {1, hidden_size_}, media_dtype_};
        inputs["use_visual_embedding"] = Tensor{&use_visual, {1, 1}, DType::kInt32};
        inputs["rotary_position_offsets"] = Tensor{offsets.data(), {3}, DType::kInt32};
    }

    // The cache is owned by the target stream; a loaded draft module may
    // execute on a separate stream. Publish the prefix copy/previous append
    // before the next forward reads it.
    if (mtp_module_->stream() != stream_) {
        const auto status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess)
            throw std::runtime_error(
                std::string("Qwen38MtpScheduler: draft cache synchronization failed: ") +
                cudaGetErrorString(status));
    }
    mtp_module_->forward_async(inputs);
    mtp_module_->sync();

    if (mtp_logits_device_ptr_ == nullptr) {
        mtp_logits_device_ptr_ = mtp_module_->device_ptr("mtp_logits");
        if (mtp_logits_device_ptr_ == nullptr)
            throw std::runtime_error("Qwen38MtpScheduler: MTP module has no 'mtp_logits' output");
    }
    mtp_logits_host_.resize(static_cast<std::size_t>(vocab_size_));
    const cudaError_t copy_status =
        cudaMemcpy(mtp_logits_host_.data(), mtp_logits_device_ptr_,
                   mtp_logits_host_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy mtp_logits: ") +
                                 cudaGetErrorString(copy_status));

    cache.advance(1);

    return argmax(mtp_logits_host_.data(), mtp_logits_host_.size());
}

void Qwen38MtpScheduler::prefill_warmup(const int32_t* tokens, int32_t first_position,
                                        const float* hidden_rows, int32_t m) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before prefill_warmup()");
    }
    if (!prefill_module_)
        throw std::runtime_error("Qwen38MtpScheduler::prefill_warmup: no MTP prefill module");
    if (m < 1 || m > prefill_chunk_len_)
        throw std::runtime_error("Qwen38MtpScheduler::prefill_warmup: m out of range");
    if (mtp_cache_->position() != first_position - 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::prefill_warmup: MTP cache position out of sync with caller's "
            "position (every main-model token must get exactly one draft()/warm-up call, in "
            "order)");
    }

    // Rows are independent, so padding rows just repeat the last real one.
    const auto chunk = static_cast<std::size_t>(prefill_chunk_len_);
    const auto hs = static_cast<std::size_t>(hidden_size_);
    prefill_tokens_.resize(chunk);
    prefill_positions_.resize(chunk);
    prefill_hidden_.resize(chunk * hs);
    for (std::size_t i = 0; i < chunk; ++i) {
        const auto r = std::min(i, static_cast<std::size_t>(m) - 1);
        prefill_tokens_[i] = tokens[r];
        prefill_positions_[i] = first_position + static_cast<int32_t>(r);
        std::copy(hidden_rows + r * hs, hidden_rows + (r + 1) * hs,
                  prefill_hidden_.begin() + i * hs);
    }

    TensorMap inputs;
    Tensor t;
    t.data = prefill_tokens_.data();
    t.shape = {prefill_chunk_len_};
    t.dtype = DType::kInt32;
    inputs["next_token_ids"] = t;
    t.data = prefill_positions_.data();
    inputs["position_id"] = t;
    t.data = prefill_hidden_.data();
    t.shape = {prefill_chunk_len_, hidden_size_};
    t.dtype = DType::kFloat32;
    inputs["hidden_state"] = t;

    prefill_module_->forward_async(inputs);
    prefill_module_->sync();

    mtp_cache_->append_prefill_kv({prefill_present_k_[0].data()}, {prefill_present_v_[0].data()},
                                  m);
}

Qwen38MtpScheduler::DraftChainResult
Qwen38MtpScheduler::draft_chain_sequential(int32_t token, int32_t position,
                                           const float* hidden_state) {
    const int32_t prefix = mtp_cache_->position();
    if (prefix != position - 1 || num_draft_tokens_ > mtp_cache_->max_length() - prefix)
        throw std::runtime_error("Qwen38MtpScheduler: speculative draft prefix/capacity mismatch");

    // Copy only the confirmed prefix. Stale suffix rows stay masked, and no
    // speculative write can change the persistent draft cache or target state.
    draft_scratch_cache_->set_position(0);
    draft_scratch_cache_->write_prefill_kv({mtp_cache_->cache_k(0).data()},
                                           {mtp_cache_->cache_v(0).data()}, prefix);
    draft_scratch_cache_->bind_to(*mtp_module_);
    DraftChainResult result;
    result.hidden_states.resize(static_cast<std::size_t>(num_draft_tokens_) * hidden_size_);
    result.token_ids.reserve(static_cast<std::size_t>(num_draft_tokens_));
    const float* current_hidden = hidden_state;
    try {
        for (int32_t i = 0; i < num_draft_tokens_; ++i) {
            token = draft_step(*draft_scratch_cache_, token, position + i, current_hidden);
            result.token_ids.push_back(token);
            void* output = mtp_module_->device_ptr("mtp_hidden_state");
            if (!output)
                throw std::runtime_error(
                    "Qwen38MtpScheduler: missing single-step MTP hidden state");
            auto* row = result.hidden_states.data() + static_cast<std::size_t>(i) * hidden_size_;
            const auto status =
                cudaMemcpy(row, output, static_cast<std::size_t>(hidden_size_) * sizeof(float),
                           cudaMemcpyDeviceToHost);
            if (status != cudaSuccess)
                throw std::runtime_error(
                    std::string("Qwen38MtpScheduler: failed to copy draft hidden state: ") +
                    cudaGetErrorString(status));
            current_hidden = row;
        }
    } catch (...) {
        mtp_cache_->bind_to(*mtp_module_);
        throw;
    }
    mtp_cache_->bind_to(*mtp_module_);
    return result;
}

Qwen38MtpScheduler::DraftChainResult
Qwen38MtpScheduler::draft_chain(int32_t token, int32_t position, const float* hidden_state) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before draft_chain()");
    }
    if (!draft_chain_module_)
        return draft_chain_sequential(token, position, hidden_state);

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["next_token_id"] = token_t;

    Tensor pos_t;
    pos_t.data = &position;
    pos_t.shape = {1};
    pos_t.dtype = DType::kInt32;
    inputs["position_id"] = pos_t;

    Tensor hidden_t;
    hidden_t.data = const_cast<float*>(hidden_state);
    hidden_t.shape = {1, hidden_size_};
    hidden_t.dtype = DType::kFloat32;
    inputs["hidden_state"] = hidden_t;

    // Same persistent-prefix mask convention as draft()'s
    // mtp_cache_->prepare_step() builds (shape (1, max_length+1), valid
    // prefix + the final "self" column always valid) -- built directly
    // here since this module takes position_id as a plain input, with no
    // auto-write to override afterward like draft() needs.
    const int32_t max_len = mtp_cache_->max_length();
    mask_scratch_.assign(static_cast<std::size_t>(max_len) + 1, kMaskedScore);
    const int32_t valid = std::min(mtp_cache_->position(), max_len);
    std::fill(mask_scratch_.begin(), mask_scratch_.begin() + valid, 0.0F);
    mask_scratch_[static_cast<std::size_t>(max_len)] = 0.0F;
    Tensor mask_t;
    mask_t.data = mask_scratch_.data();
    mask_t.shape = {1, max_len + 1};
    mask_t.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask_t;

    draft_chain_module_->forward_async(inputs);
    draft_chain_module_->sync();

    if (draft_chain_ids_device_ptr_ == nullptr) {
        draft_chain_ids_device_ptr_ = draft_chain_module_->device_ptr("mtp_draft_token_ids");
        if (draft_chain_ids_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: draft-chain module has no 'mtp_draft_token_ids' output");
        }
    }
    if (draft_chain_hidden_device_ptr_ == nullptr) {
        draft_chain_hidden_device_ptr_ = draft_chain_module_->device_ptr("mtp_draft_hidden_states");
        if (draft_chain_hidden_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: draft-chain module has no 'mtp_draft_hidden_states' output");
        }
    }

    DraftChainResult result;
    result.token_ids.resize(static_cast<std::size_t>(num_draft_tokens_));
    cudaError_t copy_status =
        cudaMemcpy(result.token_ids.data(), draft_chain_ids_device_ptr_,
                   result.token_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Qwen38MtpScheduler: failed to copy draft token ids: ") +
            cudaGetErrorString(copy_status));
    }

    result.hidden_states.resize(static_cast<std::size_t>(num_draft_tokens_) *
                                static_cast<std::size_t>(hidden_size_));
    copy_status = cudaMemcpy(result.hidden_states.data(), draft_chain_hidden_device_ptr_,
                             result.hidden_states.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Qwen38MtpScheduler: failed to copy draft hidden states: ") +
            cudaGetErrorString(copy_status));
    }

    return result;
}

Qwen38MtpScheduler::VerifyResult Qwen38MtpScheduler::verify_and_maybe_commit(
    int32_t real_next, const std::vector<int32_t>& draft_tokens, int32_t base_step,
    const std::function<Qwen38SampleResult(const float*)>& sample_next, int32_t max_tokens) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before verify_and_maybe_commit()");
    }
    if (static_cast<int32_t>(draft_tokens.size()) != num_draft_tokens_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::verify_and_maybe_commit: draft_tokens.size() must equal "
            "num_draft_tokens()");
    }

    Qwen38KvCache& kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    if (kv.position() != base_step + 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::verify_and_maybe_commit: shared state position out of sync "
            "with base_step");
    }

    if (!sample_next || max_tokens < 1 || max_tokens > kv.max_length() - kv.position())
        throw std::runtime_error("Qwen38MtpScheduler: invalid verification sampling budget");

    const int32_t n = seq_len();
    std::vector<int32_t> tokens(static_cast<std::size_t>(n));
    std::vector<int32_t> positions(static_cast<std::size_t>(n));
    tokens[0] = real_next;
    positions[0] = base_step + 1;
    for (int32_t i = 0; i < num_draft_tokens_; ++i) {
        tokens[static_cast<std::size_t>(i) + 1] = draft_tokens[static_cast<std::size_t>(i)];
        positions[static_cast<std::size_t>(i) + 1] = base_step + 2 + i;
    }

    TensorMap inputs;

    Tensor tok_t;
    tok_t.data = tokens.data();
    tok_t.shape = {n};
    tok_t.dtype = DType::kInt32;
    inputs["token_ids"] = tok_t;

    Tensor pos_t;
    pos_t.data = positions.data();
    pos_t.shape = {n};
    pos_t.dtype = DType::kInt32;
    inputs["position_ids"] = pos_t;

    std::array<int32_t, 3> offsets{0, 0, 0};
    if (media_inputs_) {
        if (media_prompt_) {
            if (static_cast<std::size_t>(positions.front()) < media_prompt_->positions.size())
                throw std::runtime_error("Qwen MTP verifier cannot run inside media prefill");
            offsets.fill(media_prompt_->decode_position -
                         static_cast<int32_t>(media_prompt_->positions.size()));
        }
        inputs["rotary_position_offsets"] = Tensor{offsets.data(), {3}, DType::kInt32};
    }

    // mask_persistent_only(valid = kv.position()): only the already-
    // committed prefix is visible; the n new tokens' causal visibility is
    // handled internally by the engine's own per-substep mask extension.
    const int32_t max_len = kv.max_length();
    mask_scratch_.assign(static_cast<std::size_t>(max_len), kMaskedScore);
    const int32_t valid = std::min(kv.position(), max_len);
    std::fill(mask_scratch_.begin(), mask_scratch_.begin() + valid, 0.0F);
    Tensor mask_t;
    mask_t.data = mask_scratch_.data();
    mask_t.shape = {1, max_len};
    mask_t.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask_t;

    if (multi_token_module_->stream() != stream_) {
        const cudaError_t status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen38MtpScheduler: target cache sync failed: ") +
                                     cudaGetErrorString(status));
    }
    multi_token_module_->forward_async(inputs);
    multi_token_module_->sync();

    if (multi_logits_device_ptr_ == nullptr) {
        multi_logits_device_ptr_ = multi_token_module_->device_ptr("logits");
        if (multi_logits_device_ptr_ == nullptr)
            throw std::runtime_error(
                "Qwen38MtpScheduler: multi-token module has no 'logits' output");
    }
    multi_logits_host_.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(vocab_size_));
    cudaError_t copy_status =
        cudaMemcpy(multi_logits_host_.data(), multi_logits_device_ptr_,
                   multi_logits_host_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy logits: ") +
                                 cudaGetErrorString(copy_status));

    // A deterministic proposal has probability one at its selected token.
    // Draw once from the target row: equality accepts; otherwise that draw
    // is the replacement. No argmax override or second rejection draw.
    VerifyResult result;
    result.accepted_tokens.push_back(real_next);
    for (int32_t row = 0; row < n; ++row) {
        if (static_cast<int32_t>(result.accepted_tokens.size()) == max_tokens) {
            result.stopped = true;
            break;
        }
        const auto sampled =
            sample_next(multi_logits_host_.data() + static_cast<std::size_t>(row) * vocab_size_);
        if (sampled.token_id < 0 || sampled.token_id >= vocab_size_)
            throw std::runtime_error("Qwen38MtpScheduler: sampler returned an invalid token");
        if (row == num_draft_tokens_ || sampled.token_id != draft_tokens[row]) {
            result.next_real_candidate = sampled.token_id;
            break;
        }
        result.accepted_tokens.push_back(sampled.token_id);
        if (sampled.is_eos) {
            result.stopped = true;
            break;
        }
    }
    result.accepted_length = static_cast<int32_t>(result.accepted_tokens.size());
    result.full_accept = !result.stopped && result.accepted_length == n;
    if (!result.full_accept)
        return result; // no speculative state committed; recover only if continuing

    // --- full accept: commit directly. append_prefill_kv()/advance() are
    // already-correct, existing multi-token-aware methods -- no new
    // cache-indexing math here. ---
    std::vector<const void*> pk;
    std::vector<const void*> pv;
    pk.reserve(static_cast<std::size_t>(kv.num_layers()));
    pv.reserve(static_cast<std::size_t>(kv.num_layers()));
    for (int32_t i = 0; i < kv.num_layers(); ++i) {
        pk.push_back(multi_present_k_[static_cast<std::size_t>(i)].data());
        pv.push_back(multi_present_v_[static_cast<std::size_t>(i)].data());
    }
    kv.append_prefill_kv(pk, pv, n);
    ssm.advance(n);

    if (multi_hidden_device_ptr_ == nullptr) {
        multi_hidden_device_ptr_ = multi_token_module_->device_ptr("hidden_states");
        if (multi_hidden_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: multi-token module has no 'hidden_states' output");
        }
    }
    result.hidden_states.resize(static_cast<std::size_t>(n) *
                                static_cast<std::size_t>(hidden_size_));
    copy_status = cudaMemcpy(result.hidden_states.data(), multi_hidden_device_ptr_,
                             result.hidden_states.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy hidden_states: ") +
                                 cudaGetErrorString(copy_status));
    }

    return result;
}

} // namespace trtmc

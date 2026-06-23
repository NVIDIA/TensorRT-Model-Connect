#include "runtime/core/device_kv_cache.h"

#include "runtime/core/device_kv_cache_update_plan.h"
#include "runtime/core/trt_decode_runtime.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string_view>

namespace trtmc::detail {

CacheRowUpdatePlan plan_cache_row_update(int32_t cache_length, int32_t max_cache_length,
                                         std::size_t row_bytes) {
    CacheRowUpdatePlan plan{};
    plan.shift_existing_rows = cache_length >= max_cache_length;
    if (!plan.shift_existing_rows) {
        plan.append_offset_bytes = static_cast<std::size_t>(cache_length) * row_bytes;
    } else {
        plan.shift_source_offset_bytes = row_bytes;
        plan.shift_copy_bytes = static_cast<std::size_t>(max_cache_length - 1) * row_bytes;
        plan.tail_offset_bytes = static_cast<std::size_t>(max_cache_length - 1) * row_bytes;
    }
    plan.next_cache_length = std::min(cache_length + 1, max_cache_length);
    return plan;
}

} // namespace trtmc::detail

namespace trtmc {

namespace {

struct StepInputs {
    int32_t position_id{0};
    std::vector<float> mask;
};

bool buffers_ok(const std::vector<CudaBuffer>& buffers) {
    for (const auto& buf : buffers) {
        if (!buf.ok()) {
            return false;
        }
    }
    return true;
}

bool optional_buffer_ok(const CudaBuffer& buffer) {
    return buffer.size() == 0 || buffer.ok();
}

bool required_buffers_ok(std::initializer_list<const CudaBuffer*> buffers) {
    for (const CudaBuffer* buffer : buffers) {
        if (buffer == nullptr || !buffer->ok()) {
            return false;
        }
    }
    return true;
}

template <typename FailFn>
bool copy_async_or_fail(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
                        cudaStream_t stream, std::string_view error_message, const FailFn& fail) {
    if (cudaMemcpyAsync(dst, src, bytes, kind, stream) != cudaSuccess) {
        return fail(error_message);
    }
    return true;
}

template <typename FailFn>
bool bind_tensor_or_fail(TrtModule* module, const std::string& tensor_name, void* device_ptr,
                         std::string_view error_message, const FailFn& fail) {
    if (module == nullptr)
        return fail(error_message);
    module->bind_external(tensor_name, device_ptr);
    return true;
}

template <typename FailFn>
bool bind_tensor_or_fail(TrtModule* module, const char* tensor_name, void* device_ptr,
                         std::string_view error_message, const FailFn& fail) {
    return bind_tensor_or_fail(module, std::string(tensor_name), device_ptr, error_message, fail);
}

template <typename FailFn>
bool transfer_small_inputs(const DecoderStepEngine& engine, DeviceResources& resources,
                           int32_t token_id, const StepInputs& step_inputs, cudaStream_t stream,
                           const FailFn& fail) {
    if (!copy_async_or_fail(resources.d_token_id.data(), &token_id, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream, "H2D token_id failed", fail)) {
        return false;
    }

    if (engine.requires_position_input) {
        if (!copy_async_or_fail(resources.d_position_id.data(), &step_inputs.position_id,
                                sizeof(int32_t), cudaMemcpyHostToDevice, stream,
                                "H2D position_id failed", fail)) {
            return false;
        }
    }

    const std::size_t mask_bytes = step_inputs.mask.size() * sizeof(float);
    return copy_async_or_fail(resources.d_mask.data(), step_inputs.mask.data(), mask_bytes,
                              cudaMemcpyHostToDevice, stream, "H2D mask failed", fail);
}

template <typename FailFn>
bool transfer_input_embed_inputs(const DecoderStepEngine& engine, DeviceResources& resources,
                                 const float* input_embed_host, int32_t embed_dim,
                                 float& use_input_embed, bool input_embed_device_ready,
                                 cudaStream_t stream, const FailFn& fail) {
    if (engine.module == nullptr || !has_io_tensor(*engine.module, "input_embed")) {
        return true;
    }

    if (input_embed_device_ready && use_input_embed > 0.5F) {
        // Caller already wrote to resources.d_input_embed on device -- skip H2D.
    } else if (input_embed_host != nullptr && embed_dim > 0 && use_input_embed > 0.5F) {
        const std::size_t embed_bytes = static_cast<std::size_t>(embed_dim) * sizeof(float);
        if (!copy_async_or_fail(resources.d_input_embed.data(), input_embed_host, embed_bytes,
                                cudaMemcpyHostToDevice, stream, "H2D input_embed failed", fail)) {
            return false;
        }
    } else {
        cudaMemsetAsync(resources.d_input_embed.data(), 0, resources.d_input_embed.size(), stream);
        use_input_embed = 0.0F;
    }

    if (resources.d_use_input_embed.size() == 0) {
        return true;
    }

    return copy_async_or_fail(resources.d_use_input_embed.data(), &use_input_embed, sizeof(float),
                              cudaMemcpyHostToDevice, stream, "H2D use_input_embed failed", fail);
}

template <typename FailFn>
bool transfer_deepstack_inputs(const DecoderStepEngine& engine, DeviceResources& resources,
                               const std::vector<const float*>& deepstack_embeds_host,
                               float deepstack_active, cudaStream_t stream, const FailFn& fail) {
    if (resources.d_deepstack_embeds.empty()) {
        return true;
    }
    if (engine.module == nullptr || !has_io_tensor(*engine.module, "deepstack_active")) {
        return true;
    }

    const std::size_t ds_embed_bytes =
        static_cast<std::size_t>(std::max(engine.hidden_size, 1)) * sizeof(float);
    for (std::size_t di = 0; di < resources.d_deepstack_embeds.size(); ++di) {
        const bool should_copy = di < deepstack_embeds_host.size() &&
                                 deepstack_embeds_host[di] != nullptr && deepstack_active > 0.5F;
        if (should_copy) {
            if (!copy_async_or_fail(resources.d_deepstack_embeds[di].data(),
                                    deepstack_embeds_host[di], ds_embed_bytes,
                                    cudaMemcpyHostToDevice, stream, "H2D deepstack_embed failed",
                                    fail)) {
                return false;
            }
        } else {
            cudaMemsetAsync(resources.d_deepstack_embeds[di].data(), 0,
                            resources.d_deepstack_embeds[di].size(), stream);
        }
    }

    float ds_active_val = deepstack_active;
    return copy_async_or_fail(resources.d_deepstack_active.data(), &ds_active_val, sizeof(float),
                              cudaMemcpyHostToDevice, stream, "H2D deepstack_active failed", fail);
}

template <typename FailFn>
bool transfer_decoder_inputs(const DecoderStepEngine& engine, DeviceResources& resources,
                             int32_t token_id, const StepInputs& step_inputs,
                             const float* input_embed_host, int32_t embed_dim,
                             float& use_input_embed,
                             const std::vector<const float*>& deepstack_embeds_host,
                             float deepstack_active, bool input_embed_device_ready,
                             cudaStream_t stream, const FailFn& fail) {
    if (!transfer_small_inputs(engine, resources, token_id, step_inputs, stream, fail)) {
        return false;
    }
    if (!transfer_input_embed_inputs(engine, resources, input_embed_host, embed_dim,
                                     use_input_embed, input_embed_device_ready, stream, fail)) {
        return false;
    }
    return transfer_deepstack_inputs(engine, resources, deepstack_embeds_host, deepstack_active,
                                     stream, fail);
}

template <typename FailFn>
bool bind_core_tensors(const DecoderStepEngine& engine, DeviceResources& resources,
                       const FailFn& fail) {
    // token_id may be absent in embed-only decoder engines that only use input_embed.
    if (engine.module != nullptr && has_io_tensor(*engine.module, engine.token_input_name)) {
        if (!bind_tensor_or_fail(engine.module, engine.token_input_name,
                                 resources.d_token_id.data(), "bind token_id failed", fail)) {
            return false;
        }
    }

    if (engine.requires_position_input) {
        if (!bind_tensor_or_fail(engine.module, engine.position_input_name,
                                 resources.d_position_id.data(), "bind position_id failed", fail)) {
            return false;
        }
    }

    if (!bind_tensor_or_fail(engine.module, engine.mask_input_name, resources.d_mask.data(),
                             "bind attention_mask failed", fail)) {
        return false;
    }

    return bind_tensor_or_fail(engine.module, engine.logits_output_name, resources.d_logits.data(),
                               "bind logits failed", fail);
}

template <typename FailFn>
bool bind_input_embed_tensors(const DecoderStepEngine& engine, DeviceResources& resources,
                              const FailFn& fail) {
    if (engine.module == nullptr || !has_io_tensor(*engine.module, "input_embed")) {
        return true;
    }

    if (!bind_tensor_or_fail(engine.module, "input_embed", resources.d_input_embed.data(),
                             "bind input_embed failed", fail)) {
        return false;
    }

    if (!has_io_tensor(*engine.module, "use_input_embed")) {
        return true;
    }

    return bind_tensor_or_fail(engine.module, "use_input_embed", resources.d_use_input_embed.data(),
                               "bind use_input_embed failed", fail);
}

template <typename FailFn>
bool bind_deepstack_tensors(const DecoderStepEngine& engine, DeviceResources& resources,
                            const FailFn& fail) {
    for (std::size_t di = 0; di < resources.d_deepstack_embeds.size(); ++di) {
        std::string ds_name = "deepstack_embed_" + std::to_string(di);
        if (engine.module == nullptr || !has_io_tensor(*engine.module, ds_name)) {
            continue;
        }
        if (!bind_tensor_or_fail(engine.module, ds_name, resources.d_deepstack_embeds[di].data(),
                                 "bind deepstack_embed failed", fail)) {
            return false;
        }
    }

    if (resources.d_deepstack_active.size() == 0) {
        return true;
    }
    if (engine.module == nullptr || !has_io_tensor(*engine.module, "deepstack_active")) {
        return true;
    }
    return bind_tensor_or_fail(engine.module, "deepstack_active",
                               resources.d_deepstack_active.data(), "bind deepstack_active failed",
                               fail);
}

template <typename FailFn>
bool bind_cache_tensors(const DecoderStepEngine& engine, DeviceKvCache& cache,
                        DeviceResources& resources, const FailFn& fail) {
    for (int32_t layer = 0; layer < engine.num_layers; ++layer) {
        const auto idx = static_cast<std::size_t>(layer);
        if (!bind_tensor_or_fail(engine.module, engine.cache_k_input_names[idx],
                                 cache.cache_k_device_ptr(layer), "bind cache_k failed", fail)) {
            return false;
        }
        if (!bind_tensor_or_fail(engine.module, engine.cache_v_input_names[idx],
                                 cache.cache_v_device_ptr(layer), "bind cache_v failed", fail)) {
            return false;
        }
        if (!bind_tensor_or_fail(engine.module, engine.present_k_output_names[idx],
                                 resources.d_present_k[idx].data(), "bind present_k failed",
                                 fail)) {
            return false;
        }
        if (!bind_tensor_or_fail(engine.module, engine.present_v_output_names[idx],
                                 resources.d_present_v[idx].data(), "bind present_v failed",
                                 fail)) {
            return false;
        }
    }
    return true;
}

template <typename FailFn>
bool bind_decoder_tensors(const DecoderStepEngine& engine, DeviceKvCache& cache,
                          DeviceResources& resources, const FailFn& fail) {
    if (!bind_core_tensors(engine, resources, fail)) {
        return false;
    }
    if (!bind_input_embed_tensors(engine, resources, fail)) {
        return false;
    }
    if (!bind_deepstack_tensors(engine, resources, fail)) {
        return false;
    }
    return bind_cache_tensors(engine, cache, resources, fail);
}

template <typename FailFn>
bool execute_and_collect_logits(const DecoderStepEngine& engine, DeviceKvCache& cache,
                                DeviceResources& resources, std::vector<float>& logits,
                                cudaStream_t stream, bool skip_logits_d2h, bool skip_sync,
                                const FailFn& fail) {
    if (engine.module == nullptr) {
        return fail("enqueueV3 failed");
    }
    engine.module->forward_async({});

    cache.update_after_step(resources.d_present_k, resources.d_present_v, stream);

    // D2H logits (skip when caller will consume logits on device, e.g. GPU argmax)
    if (!skip_logits_d2h) {
        logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
        const std::size_t logits_bytes = logits.size() * sizeof(float);
        if (!copy_async_or_fail(logits.data(), resources.d_logits.data(), logits_bytes,
                                cudaMemcpyDeviceToHost, stream, "D2H logits failed", fail)) {
            return false;
        }
    }

    // Sync (skip when caller will sync later, e.g. batched prefill)
    if (!skip_sync) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return fail("cudaStreamSynchronize failed");
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// DeviceKvCache
// ---------------------------------------------------------------------------

DeviceKvCache::DeviceKvCache(const DecoderStepEngine& engine)
    : mCacheStateSize(engine.cache_state_size), mMaxCacheLength(engine.max_cache_length),
      mNumLayers(engine.num_layers), mIncludeCurrentSlot(engine.requires_position_input),
      mPositionLimit(mIncludeCurrentSlot ? mMaxCacheLength : std::max(mMaxCacheLength - 1, 0)) {
    // Detect cache element size from the engine's cache_k_0 tensor dtype.
    // FP16 engines will have kHALF cache tensors; FP32 engines will have kFLOAT.
    if (engine.module != nullptr && !engine.cache_k_input_names.empty() &&
        has_io_tensor(*engine.module, engine.cache_k_input_names[0])) {
        switch (engine.module->tensor_dtype(engine.cache_k_input_names[0])) {
        case DType::kFloat16:
            mCacheElementSize = 2;
            break;
        case DType::kBFloat16:
            mCacheElementSize = 2;
            break;
        default:
            mCacheElementSize = sizeof(float);
            break;
        }
    }

    const std::size_t cache_bytes = static_cast<std::size_t>(mMaxCacheLength) *
                                    static_cast<std::size_t>(mCacheStateSize) * mCacheElementSize;

    mCacheK.reserve(static_cast<std::size_t>(mNumLayers));
    mCacheV.reserve(static_cast<std::size_t>(mNumLayers));
    for (int32_t i = 0; i < mNumLayers; ++i) {
        mCacheK.emplace_back(cache_bytes);
        mCacheV.emplace_back(cache_bytes);
    }

    // Zero-initialize (synchronous — only done once at construction)
    for (int32_t i = 0; i < mNumLayers; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        if (mCacheK[idx].data() != nullptr) {
            cudaMemset(mCacheK[idx].data(), 0, cache_bytes);
        }
        if (mCacheV[idx].data() != nullptr) {
            cudaMemset(mCacheV[idx].data(), 0, cache_bytes);
        }
    }
}

void DeviceKvCache::prepare_step(int32_t& out_position_id, std::vector<float>& out_mask) {
    out_position_id = std::min(mCacheLength, mPositionLimit);
    out_mask = build_attention_mask(mCacheLength, mMaxCacheLength, mIncludeCurrentSlot);
}

void DeviceKvCache::update_after_step(const std::vector<CudaBuffer>& present_k,
                                      const std::vector<CudaBuffer>& present_v,
                                      cudaStream_t stream) {
    const std::size_t row_bytes = static_cast<std::size_t>(mCacheStateSize) * mCacheElementSize;
    const detail::CacheRowUpdatePlan plan =
        detail::plan_cache_row_update(mCacheLength, mMaxCacheLength, row_bytes);

    auto copy_one = [&](CudaBuffer& cache_buf, const CudaBuffer& present_buf) {
        auto* cache_ptr = static_cast<char*>(cache_buf.data());
        const auto* present_ptr = present_buf.data();

        if (!plan.shift_existing_rows) {
            cudaMemcpyAsync(cache_ptr + plan.append_offset_bytes, present_ptr, row_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(cache_ptr, cache_ptr + plan.shift_source_offset_bytes,
                            plan.shift_copy_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(cache_ptr + plan.tail_offset_bytes, present_ptr, row_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        }
    };

    for (int32_t layer = 0; layer < mNumLayers; ++layer) {
        const auto idx = static_cast<std::size_t>(layer);
        copy_one(mCacheK[idx], present_k[idx]);
        copy_one(mCacheV[idx], present_v[idx]);
    }

    mCacheLength = plan.next_cache_length;
}

void DeviceKvCache::reset(cudaStream_t stream) {
    const std::size_t cache_bytes = static_cast<std::size_t>(mMaxCacheLength) *
                                    static_cast<std::size_t>(mCacheStateSize) * mCacheElementSize;

    for (int32_t i = 0; i < mNumLayers; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        if (mCacheK[idx].data() != nullptr) {
            cudaMemsetAsync(mCacheK[idx].data(), 0, cache_bytes, stream);
        }
        if (mCacheV[idx].data() != nullptr) {
            cudaMemsetAsync(mCacheV[idx].data(), 0, cache_bytes, stream);
        }
    }
    mCacheLength = 0;
}

void* DeviceKvCache::cache_k_device_ptr(int32_t layer) const {
    return mCacheK[static_cast<std::size_t>(layer)].data();
}

void* DeviceKvCache::cache_v_device_ptr(int32_t layer) const {
    return mCacheV[static_cast<std::size_t>(layer)].data();
}

bool DeviceKvCache::ok() const {
    if (!buffers_ok(mCacheK)) {
        return false;
    }
    return buffers_ok(mCacheV);
}

// ---------------------------------------------------------------------------
// DeviceResources
// ---------------------------------------------------------------------------

DeviceResources::DeviceResources(const DecoderStepEngine& engine)
    : d_token_id(sizeof(int32_t)), d_position_id(sizeof(int32_t)),
      d_mask(static_cast<std::size_t>(engine.attention_mask_size) * sizeof(float)),
      d_logits(static_cast<std::size_t>(engine.vocab_size) * sizeof(float)),
      d_input_embed(engine.module != nullptr && has_io_tensor(*engine.module, "input_embed")
                        ? static_cast<std::size_t>(std::max(engine.hidden_size, 1)) * sizeof(float)
                        : 0),
      d_use_input_embed(engine.module != nullptr && has_io_tensor(*engine.module, "input_embed")
                            ? sizeof(float)
                            : 0),
      d_deepstack_active(engine.module != nullptr &&
                                 has_io_tensor(*engine.module, "deepstack_active")
                             ? sizeof(float)
                             : 0) {
    // Detect cache element size from the engine's present_k_0 tensor dtype.
    std::size_t cache_elem_size = sizeof(float);
    if (engine.module != nullptr && !engine.present_k_output_names.empty() &&
        has_io_tensor(*engine.module, engine.present_k_output_names[0])) {
        switch (engine.module->tensor_dtype(engine.present_k_output_names[0])) {
        case DType::kFloat16:
        case DType::kBFloat16:
            cache_elem_size = 2;
            break;
        default:
            cache_elem_size = sizeof(float);
            break;
        }
    }

    const std::size_t state_bytes =
        static_cast<std::size_t>(engine.cache_state_size) * cache_elem_size;
    d_present_k.reserve(static_cast<std::size_t>(engine.num_layers));
    d_present_v.reserve(static_cast<std::size_t>(engine.num_layers));
    for (int32_t i = 0; i < engine.num_layers; ++i) {
        d_present_k.emplace_back(state_bytes);
        d_present_v.emplace_back(state_bytes);
    }

    // DeepStack embed buffers: auto-detect from engine bindings
    const std::size_t embed_bytes =
        static_cast<std::size_t>(std::max(engine.hidden_size, 1)) * sizeof(float);
    for (int32_t i = 0;; ++i) {
        std::string name = "deepstack_embed_" + std::to_string(i);
        if (engine.module == nullptr || !has_io_tensor(*engine.module, name))
            break;
        d_deepstack_embeds.emplace_back(embed_bytes);
    }
}

bool DeviceResources::ok() const {
    if (!stream.ok()) {
        return false;
    }
    if (!required_buffers_ok({&d_token_id, &d_position_id, &d_mask, &d_logits})) {
        return false;
    }
    if (!optional_buffer_ok(d_input_embed)) {
        return false;
    }
    if (!optional_buffer_ok(d_use_input_embed)) {
        return false;
    }
    if (!optional_buffer_ok(d_deepstack_active)) {
        return false;
    }
    if (!buffers_ok(d_deepstack_embeds)) {
        return false;
    }
    if (!buffers_ok(d_present_k)) {
        return false;
    }
    return buffers_ok(d_present_v);
}

// ---------------------------------------------------------------------------
// run_decoder_step_device
// ---------------------------------------------------------------------------

bool run_decoder_step_device(const DecoderStepEngine& engine, DeviceKvCache& cache,
                             DeviceResources& resources, int32_t token_id,
                             std::vector<float>& logits, std::string& error,
                             const float* input_embed_host, int32_t embed_dim,
                             float use_input_embed,
                             const std::vector<const float*>& deepstack_embeds_host,
                             float deepstack_active, bool input_embed_device_ready,
                             bool skip_logits_d2h, bool skip_sync, bool skip_bind) {
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    cudaStream_t stream =
        engine.module != nullptr ? engine.module->stream() : resources.stream.get();

    // 1. Prepare step: compute position_id and mask on CPU
    StepInputs step_inputs;
    cache.prepare_step(step_inputs.position_id, step_inputs.mask);

    if (!transfer_decoder_inputs(engine, resources, token_id, step_inputs, input_embed_host,
                                 embed_dim, use_input_embed, deepstack_embeds_host,
                                 deepstack_active, input_embed_device_ready, stream, fail)) {
        return false;
    }

    // 3. Bind tensor addresses (skip when addresses haven't changed since last call)
    if (!skip_bind) {
        if (!bind_decoder_tensors(engine, cache, resources, fail)) {
            return false;
        }
    }

    // 4-7. Execute, cache update, logits readback, and stream sync
    return execute_and_collect_logits(engine, cache, resources, logits, stream, skip_logits_d2h,
                                      skip_sync, fail);
}

} // namespace trtmc

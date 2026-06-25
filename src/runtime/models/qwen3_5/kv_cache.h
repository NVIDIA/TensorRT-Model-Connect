#pragma once

// Qwen35KvCache: autoregressive KV cache state manager.
// HF equivalent: DynamicCache / past_key_values.
//
// Manages per-layer K/V device tensors, position tracking, and attention mask
// construction. Binds directly to a TrtModule via bind_to().

#include "runtime/models/qwen3_5/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

// Explicit tensor names for KV cache I/O binding.
// Per-layer vectors hold expanded names; scalar names are for single inputs.
struct Qwen35KvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string position_id{"position_id"};
    std::string attention_mask{"attention_mask"};
};

class Qwen35KvCache : public Qwen35InferenceState {
  public:
    // Allocate cache buffers for the given configuration.
    // kv_dim = num_kv_heads * head_dim (size of one K or V row per layer).
    // cache_dtype controls the element type for K/V cache buffers (default FP32).
    // names provides explicit tensor names for engine I/O binding.
    Qwen35KvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                  DType cache_dtype = DType::kFloat32, Qwen35KvCacheNames names = {});

    // --- Qwen35InferenceState overrides ---
    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t preferred_cache_rows() const override;
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "dense_kv_cache"; }
    bool ok() const override;

    // --- Qwen35KvCache-specific methods (not on the interface) ---

    // DEPRECATED: Use prepare_step() instead.
    // Kept for backward compatibility with tests that call this directly.
    void build_attention_mask(std::vector<float>& mask) const;

    // Direct access for advanced use (cross-attention, VL embedding).
    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }

    // Write per-layer KV produced by a batched prefill engine into the cache
    // at positions [0, seq_len). Device-to-device copy on this cache's stream;
    // advances position_ to seq_len. Requires seq_len <= max_length.
    void write_prefill_kv(const std::vector<const void*>& prefill_k,
                          const std::vector<const void*>& prefill_v, int32_t seq_len);

    // Prepare a multi-token block whose new tokens may attend bidirectionally
    // to one another while still seeing the valid prefix cache.
    void prepare_bidirectional_step(TensorMap& inputs, int32_t seq_len);

    // Append batched present K/V at the current position. Used after causal
    // block verification in diffusion-style text decoders.
    void append_prefill_kv(const std::vector<const void*>& prefill_k,
                           const std::vector<const void*>& prefill_v, int32_t seq_len);

    // Move the logical cache length without touching device memory. Stale rows
    // remain masked out by subsequent prepare_step calls.
    void set_position(int32_t position);

    // Bind only the cache_k/v INPUT pointers to `module`. Used for the
    // prefill TrtModule whose present_k/v outputs have shape (Sq, kv_dim)
    // — too big for Qwen35KvCache's single-row present buffer. The caller reads
    // the prefill outputs directly from the module's own allocations and
    // copies them via write_prefill_kv().
    void bind_cache_inputs(TrtModule& module);

  private:
    void rebind_cache_rows(int32_t cache_rows);
    std::vector<int64_t> mask_shape_for_engine(int32_t mask_width, std::size_t mask_buf_size) const;
    void write_position_input(TensorMap& inputs, int32_t seq_len);
    void write_batched_mask(TensorMap& inputs, int32_t seq_len);
    void write_bidirectional_mask(TensorMap& inputs, int32_t seq_len);
    void write_decode_mask(TensorMap& inputs);

    std::vector<DeviceTensor> cache_k_;   // [num_layers], shape [max_length, kv_dim]
    std::vector<DeviceTensor> cache_v_;   // [num_layers]
    std::vector<DeviceTensor> present_k_; // [num_layers], shape [1, kv_dim] (single step output)
    std::vector<DeviceTensor> present_v_; // [num_layers]
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
    // Buffers owned by this object — Tensor.data in prepare_step() points here.
    std::vector<float> mask_buf_;
    std::vector<int32_t> pos_buf_vec_;
    bool has_position_input_{false};
    bool dynamic_binding_enabled_{false};
    int32_t bound_cache_rows_{0};
    DType cache_dtype_{DType::kFloat32};
    std::size_t cache_element_size_{sizeof(float)};
    Qwen35KvCacheNames names_;
    TrtModule* bound_module_{nullptr};
};

} // namespace trtmc

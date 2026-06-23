#pragma once

// HybridState: composes KvCache + RecurrentState behind IInferenceState.
// Used by hybrid models that interleave attention and recurrent layers.

#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/recurrent_state.h"

#include <memory>

namespace trtmc {

class HybridState : public IInferenceState {
  public:
    HybridState(std::unique_ptr<KvCache> kv, std::unique_ptr<RecurrentState> ssm);

    // --- IInferenceState overrides ---
    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override;
    int32_t max_length() const override;
    int32_t num_layers() const override;
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "hybrid_kv_recurrent"; }
    bool ok() const override;

    // --- HybridState-specific access ---
    KvCache* kv_cache() { return kv_.get(); }
    const KvCache* kv_cache() const { return kv_.get(); }
    RecurrentState* recurrent_state() { return ssm_.get(); }
    const RecurrentState* recurrent_state() const { return ssm_.get(); }

  private:
    std::unique_ptr<KvCache> kv_;
    std::unique_ptr<RecurrentState> ssm_;
};

} // namespace trtmc

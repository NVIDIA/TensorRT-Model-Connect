#pragma once

#include "runtime/models/nemotron_h/inference_state.h"
#include "runtime/models/nemotron_h/kv_cache.h"
#include "runtime/models/nemotron_h/recurrent_state.h"

#include <memory>

namespace trtmc {

class NemotronHHybridState final : public NemotronHInferenceState {
  public:
    NemotronHHybridState(std::unique_ptr<NemotronHKvCache> kv,
                         std::unique_ptr<NemotronHRecurrentState> ssm);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override;
    int32_t max_length() const override;
    int32_t num_layers() const override;
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "nemotron_h_hybrid_kv_recurrent"; }
    bool ok() const override;

    NemotronHKvCache* kv_cache() { return kv_.get(); }
    const NemotronHKvCache* kv_cache() const { return kv_.get(); }
    NemotronHRecurrentState* recurrent_state() { return ssm_.get(); }
    const NemotronHRecurrentState* recurrent_state() const { return ssm_.get(); }

  private:
    std::unique_ptr<NemotronHKvCache> kv_;
    std::unique_ptr<NemotronHRecurrentState> ssm_;
};

} // namespace trtmc

#pragma once

#include "runtime/models/nemotron_h/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class NemotronHRecurrentState final : public NemotronHInferenceState {
  public:
    struct TensorSpec {
        std::string name;
        std::vector<int64_t> shape;
        std::string output_prefix;
    };

    NemotronHRecurrentState(int32_t num_layers, std::vector<TensorSpec> specs, cudaStream_t stream);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return -1; }
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "nemotron_h_recurrent"; }
    bool ok() const override;

    const std::vector<TensorSpec>& specs() const { return specs_; }

  private:
    std::vector<TensorSpec> specs_;
    std::vector<std::vector<DeviceTensor>> state_;
    std::vector<std::vector<DeviceTensor>> present_;
    int32_t num_layers_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace trtmc

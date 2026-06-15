#pragma once

// ChronosBoltPipeline: numeric forecasting pipeline for Chronos-Bolt bundles.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class ChronosBoltPipeline final : public IPipeline {
  public:
    ChronosBoltPipeline(std::unique_ptr<TrtModule> forecast, int32_t context_length,
                        int32_t prediction_length, int32_t num_quantiles, std::string model_id_str);

    EmbeddingResult solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
                          int32_t trunk_len) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "ChronosBoltPipeline"; }

  private:
    std::unique_ptr<TrtModule> forecast_;
    int32_t context_length_{0};
    int32_t prediction_length_{0};
    int32_t num_quantiles_{0};
    std::string model_id_;
};

} // namespace trtmc

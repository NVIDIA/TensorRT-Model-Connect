#pragma once

// SegmentPipeline: single-pass segmentation (SegFormer).
// Uses a single TrtModule for pixel_values -> logits/mask output.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class SegmentPipeline final : public IPipeline {
  public:
    explicit SegmentPipeline(std::unique_ptr<TrtModule> model, std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SegmentPipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    std::string model_id_;
};

} // namespace trtmc

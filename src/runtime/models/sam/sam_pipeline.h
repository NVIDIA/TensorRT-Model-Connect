#pragma once

// SamPipeline: two-stage segmentation (SAM -- encoder + decoder).
// Uses TrtModule(image_encoder) + TrtModule(mask_decoder).

#include "runtime/models/sam/sam_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class SamPipeline final : public IPipeline {
  public:
    SamPipeline(std::unique_ptr<TrtModule> image_encoder, std::unique_ptr<TrtModule> mask_decoder,
                SamConfig config, std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;
    PromptedSegmentationResult segment_prompted(const float* image_pixels, int32_t image_height,
                                                int32_t image_width, float point_x = 0.5F,
                                                float point_y = 0.5F,
                                                bool is_foreground = true) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SamPipeline"; }

  private:
    std::unique_ptr<TrtModule> image_encoder_;
    std::unique_ptr<TrtModule> mask_decoder_;
    SamConfig config_;
    std::string model_id_;
};

} // namespace trtmc

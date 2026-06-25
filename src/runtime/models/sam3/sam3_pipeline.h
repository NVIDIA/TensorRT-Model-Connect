#pragma once

// Sam3Pipeline: SAM3 image PCS runtime surface.
// Runs native tokenization plus SAM3 text, vision, and DETR/mask/scoring TRT
// plans, then postprocesses into model-card masks, boxes, and scores.

#include "runtime/models/sam3/sam3_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Sam3TextFeatures {
    std::vector<float> features;
    std::vector<int64_t> features_shape;
    std::vector<float> hidden_states;
    std::vector<int64_t> hidden_states_shape;
    std::vector<int32_t> attention_mask;
};

class Sam3Pipeline final : public IPipeline {
  public:
    Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder, std::shared_ptr<ITokenizer> tokenizer,
                 Sam3Config config, std::string model_id_str = "");
    Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> vision_encoder,
                 std::shared_ptr<ITokenizer> tokenizer, Sam3Config config,
                 std::string model_id_str = "");
    Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> vision_encoder,
                 std::unique_ptr<TrtModule> core_engine, std::shared_ptr<ITokenizer> tokenizer,
                 Sam3Config config, std::string model_id_str = "");

    PromptedSegmentationResult segment_prompted(const float* image_pixels, int32_t image_height,
                                                int32_t image_width, float point_x = 0.5F,
                                                float point_y = 0.5F,
                                                bool is_foreground = true) override;

    PromptedSegmentationResult segment_prompted_text(const float* image_pixels,
                                                     int32_t image_height, int32_t image_width,
                                                     const std::string& text_prompt) override;

    Sam3TextFeatures encode_text_prompt_for_test(const std::string& text_prompt) const;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Sam3Pipeline"; }

  private:
    Sam3TextFeatures encode_text_prompt(const std::string& text_prompt) const;

    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> vision_encoder_;
    std::unique_ptr<TrtModule> core_engine_;
    std::shared_ptr<ITokenizer> tokenizer_;
    Sam3Config config_;
    std::string model_id_;
};

} // namespace trtmc

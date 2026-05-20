#include "runtime/domains/diffusion/qwen_image_types.h"
#include "runtime/models/qwen_image/pipeline.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

int failures = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << "\n";
        ++failures;
    }
}

template <typename Fn>
void check_throws_contains(const char* name, const std::string& needle, Fn&& fn) {
    try {
        fn();
        std::cerr << "FAIL: " << name << " did not throw\n";
        ++failures;
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find(needle) == std::string::npos) {
            std::cerr << "FAIL: " << name << " threw unexpected message: " << e.what() << "\n";
            ++failures;
        }
    }
}

const char* kEditConfig = R"JSON({
  "engine_backend": "trt",
  "runtime_strategy": "diffusion_qwen_image",
  "model_family": "qwen_image",
  "model_variant": "qwen-image-edit-2511",
  "task_mode": "edit",
  "diffusion": {
    "use_dynamic_shifting": true,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential"
  },
  "text_encoder": {
    "type": "qwen2_5_vl_multimodal",
    "tokenizer_template_kind": "qwen_image_edit_hardcoded"
  },
  "vae": {
    "has_encoder": true,
    "has_decoder": true
  },
  "tokenizer": {
    "prompt_template_kind": "qwen_image_edit_hardcoded",
    "prompt_template_drop_idx": 64
  },
  "vision_encoder": {
    "type": "qwen2_5_vl_vision",
    "image_size": 384,
    "patch_size": 14,
    "merge_size": 2,
    "hidden_size": 1280,
    "num_layers": 32,
    "out_hidden_size": 3584
  },
  "image_conditioning": {
    "vl_image_size": 384,
    "vae_image_size": 1024,
    "vae_concat_axis": "sequence",
    "max_input_images": 1
  }
})JSON";

} // namespace

int main() {
    auto cfg = trtmc::QwenImageConfig::parse(kEditConfig);

    check(cfg.task_mode == trtmc::QwenImageTaskMode::Edit, "task_mode edit");
    check(cfg.model_variant == "qwen-image-edit-2511", "model_variant");
    check(cfg.diffusion.use_dynamic_shifting, "dynamic shift");
    check(std::abs(cfg.diffusion.base_shift - 0.5F) < 1e-6F, "base_shift");
    check(std::abs(cfg.diffusion.max_shift - 0.9F) < 1e-6F, "max_shift");
    check(std::abs(cfg.diffusion.shift_terminal - 0.02F) < 1e-6F, "shift_terminal");
    check(cfg.diffusion.time_shift_type == "exponential", "time_shift_type");
    check(cfg.text_encoder.type == "qwen2_5_vl_multimodal", "text_encoder type");
    check(cfg.vae.has_encoder, "vae has_encoder");
    check(cfg.tokenizer.prompt_template_drop_idx == 64, "edit drop idx");
    check(cfg.vision_encoder.image_size == 384, "vision image size");
    check(cfg.vision_encoder.out_hidden_size == 3584, "vision out hidden");
    check(cfg.image_conditioning.vae_concat_axis == "sequence", "vae concat axis");
    check(cfg.image_conditioning.max_input_images == 1, "max input images");

    trtmc::QwenImagePipeline::Construction construction;
    construction.config = cfg;
    construction.model_id = "qwen-image-edit-2511";
    auto pipeline = trtmc::QwenImagePipeline(std::move(construction));

    check_throws_contains("edit requires image overload", "require an input image", [&]() {
        (void)pipeline.generate_image("make it watercolor");
    });
    float one_pixel[3] = {0.0F, 0.0F, 0.0F};
    check_throws_contains("edit missing engines guard", "runtime is not complete", [&]() {
        (void)pipeline.generate_image("make it watercolor", one_pixel, 1, 1);
    });

    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    return 0;
}

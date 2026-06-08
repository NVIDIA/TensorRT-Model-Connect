#include "runtime/models/segmentation/sam3_pipeline.h"

#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

bool close(float actual, float expected) {
    const float diff = std::fabs(actual - expected);
    return diff < 1.0e-5F;
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        return ids;
    }

    std::string decode(const std::vector<int32_t>& /*ids*/) const override { return {}; }
    int32_t id_for_token(std::string_view /*token*/) const override { return -1; }
    std::string token_for_id(int32_t /*id*/) const override { return {}; }

    std::vector<int32_t> ids{7, 8};
    mutable std::string last_text;
};

class FakeSam3TextModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto ids_it = inputs.find("input_ids");
        const auto mask_it = inputs.find("attention_mask");
        if (ids_it == inputs.end() || mask_it == inputs.end() || !ids_it->second.data ||
            !mask_it->second.data) {
            throw std::runtime_error("missing SAM3 text inputs");
        }
        const auto* ids = static_cast<const int32_t*>(ids_it->second.data);
        const auto* mask = static_cast<const int32_t*>(mask_it->second.data);
        saw_expected_ids = ids[0] == 7 && ids[1] == 8 && ids[2] == 0 && ids[3] == 0;
        saw_expected_mask = mask[0] == 1 && mask[1] == 1 && mask[2] == 0 && mask[3] == 0;
        saw_shape = ids_it->second.shape == std::vector<int64_t>{4} &&
                    mask_it->second.shape == std::vector<int64_t>{4};

        features_ = {1.0F, 2.0F, 3.0F, 4.0F};
        hidden_ = {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};

        trtmc::Tensor features;
        features.data = features_.data();
        features.shape = {4, 1};
        features.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor hidden;
        hidden.data = hidden_.data();
        hidden.shape = {4, 2};
        hidden.dtype = trtmc::DType::kFloat32;

        return {{"sam3_text_features", features}, {"sam3_text_hidden_states", hidden}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& /*name*/) const override { return false; }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& /*name*/) const override { return {}; }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_expected_ids{false};
    bool saw_expected_mask{false};
    bool saw_shape{false};

  private:
    std::vector<float> features_;
    std::vector<float> hidden_;
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3VisionModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto image_it = inputs.find("pixel_values");
        if (image_it == inputs.end() || !image_it->second.data)
            throw std::runtime_error("missing SAM3 image input");
        const auto* pixels = static_cast<const float*>(image_it->second.data);
        saw_shape = image_it->second.shape == std::vector<int64_t>({1, 3, 4, 4});
        saw_normalized_pixels = close(pixels[0], -1.0262009F) && close(pixels[16], 0.1964286F) &&
                                close(pixels[32], 1.5288889F);

        trtmc::TensorMap out;
        for (int32_t level = 0; level < 3; ++level) {
            const auto level_index = static_cast<std::size_t>(level);
            fpn_hidden_[level_index] = {10.0F + static_cast<float>(level),
                                        11.0F + static_cast<float>(level)};
            fpn_position_[level_index] = {20.0F + static_cast<float>(level),
                                          21.0F + static_cast<float>(level)};

            trtmc::Tensor hidden;
            hidden.data = fpn_hidden_[level_index].data();
            hidden.shape = {1, 2, 1, 1};
            hidden.dtype = trtmc::DType::kFloat32;
            out["sam3_fpn_hidden_" + std::to_string(level)] = hidden;

            trtmc::Tensor pos;
            pos.data = fpn_position_[level_index].data();
            pos.shape = {1, 2, 1, 1};
            pos.dtype = trtmc::DType::kFloat32;
            out["sam3_fpn_position_" + std::to_string(level)] = pos;
        }
        return out;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& /*name*/) const override { return false; }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& /*name*/) const override { return {}; }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_shape{false};
    bool saw_normalized_pixels{false};

  private:
    std::vector<float> fpn_hidden_[3];
    std::vector<float> fpn_position_[3];
    std::shared_ptr<void> keep_alive_;
};

class FakeSam3CoreModule final : public trtmc::TrtModule {
  public:
    bool ok() const override { return true; }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto text_it = inputs.find("sam3_text_features");
        const auto mask_it = inputs.find("sam3_text_attention_mask");
        if (text_it == inputs.end() || mask_it == inputs.end() || !text_it->second.data ||
            !mask_it->second.data) {
            throw std::runtime_error("missing SAM3 core text inputs");
        }
        saw_text_shape = text_it->second.shape == std::vector<int64_t>({1, 4, 1});
        const auto* mask = static_cast<const int32_t*>(mask_it->second.data);
        saw_mask = mask[0] == 1 && mask[1] == 1 && mask[2] == 0 && mask[3] == 0;
        saw_vision_inputs = true;
        for (int32_t level = 0; level < 3; ++level) {
            saw_vision_inputs = saw_vision_inputs &&
                                inputs.count("sam3_fpn_hidden_" + std::to_string(level)) == 1 &&
                                inputs.count("sam3_fpn_position_" + std::to_string(level)) == 1;
        }

        pred_masks_ = {
            -1.0F, -1.0F, -1.0F, -1.0F, -2.0F, 2.0F, 2.0F, -2.0F,
        };
        pred_boxes_ = {
            0.0F, 0.0F, 1.0F, 1.0F, 0.25F, 0.5F, 0.75F, 1.0F,
        };
        if (omit_second_box) {
            pred_boxes_.resize(4);
        }
        pred_logits_ = {0.0F, 2.0F};
        presence_logits_ = {2.0F};

        trtmc::Tensor masks;
        masks.data = pred_masks_.data();
        masks.shape = {1, 2, 2, 2};
        masks.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor boxes;
        boxes.data = pred_boxes_.data();
        boxes.shape = {1, omit_second_box ? 1 : 2, 4};
        boxes.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor logits;
        logits.data = pred_logits_.data();
        logits.shape = {1, 2};
        logits.dtype = trtmc::DType::kFloat32;

        trtmc::Tensor presence;
        presence.data = presence_logits_.data();
        presence.shape = {1, 1};
        presence.dtype = trtmc::DType::kFloat32;

        return {{"pred_masks", masks},
                {"pred_boxes", boxes},
                {"pred_logits", logits},
                {"presence_logits", presence}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& /*name*/) const override { return false; }
    bool has_output(const std::string& /*name*/) const override { return false; }
    trtmc::DType tensor_dtype(const std::string& /*name*/) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& /*name*/) const override { return {}; }
    std::vector<int64_t>
    input_profile_shape(const std::string& /*name*/, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }

    bool saw_text_shape{false};
    bool saw_mask{false};
    bool saw_vision_inputs{false};
    bool omit_second_box{false};

  private:
    std::vector<float> pred_masks_;
    std::vector<float> pred_boxes_;
    std::vector<float> pred_logits_;
    std::vector<float> presence_logits_;
    std::shared_ptr<void> keep_alive_;
};

trtmc::Sam3Config make_config() {
    trtmc::Sam3Config cfg;
    cfg.text_max_position_embeddings = 4;
    cfg.text_pad_token_id = 0;
    cfg.text_projection_dim = 1;
    cfg.image_size = 4;
    return cfg;
}

void test_text_prompt_runs_tokenizer_and_text_encoder() {
    auto* module_ptr = new FakeSam3TextModule();
    auto module = std::unique_ptr<trtmc::TrtModule>(module_ptr);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(module), tokenizer, make_config());

    const auto features = pipeline.encode_text_prompt_for_test("ear");

    check(tokenizer->last_text == "ear", "sam3 tokenizer receives prompt");
    check(module_ptr->saw_expected_ids, "sam3 text ids padded");
    check(module_ptr->saw_expected_mask, "sam3 attention mask padded");
    check(module_ptr->saw_shape, "sam3 text input shapes");
    check(features.features_shape == std::vector<int64_t>({4, 1}), "sam3 features shape");
    check(features.hidden_states_shape == std::vector<int64_t>({4, 2}), "sam3 hidden shape");
    check(features.features.size() == 4, "sam3 features copied");
    check(features.hidden_states.size() == 8, "sam3 hidden copied");
    check(features.attention_mask == std::vector<int32_t>({1, 1, 0, 0}),
          "sam3 attention mask copied");
}

void test_point_prompt_rejected() {
    auto module = std::make_unique<FakeSam3TextModule>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(module), tokenizer, make_config());

    bool threw = false;
    try {
        float pixel = 0.0F;
        (void)pipeline.segment_prompted(&pixel, 1, 1);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("requires a text prompt") != std::string::npos;
    }
    check(threw, "sam3 rejects point prompt path");
}

void test_text_prompt_reports_remaining_runtime_gap() {
    auto* module_ptr = new FakeSam3TextModule();
    auto module = std::unique_ptr<trtmc::TrtModule>(module_ptr);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(module), tokenizer, make_config());

    bool threw = false;
    try {
        float pixels[3] = {0.25F, 0.5F, 0.75F};
        (void)pipeline.segment_prompted_text(pixels, 1, 1, "ear");
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("missing vision_engine_plan") != std::string::npos;
    }
    check(threw, "sam3 text path reports missing vision engine");
    check(module_ptr->saw_expected_ids, "sam3 text path ran text encoder before gap");
}

void test_text_prompt_runs_image_preprocess_and_vision_encoder() {
    auto* text_ptr = new FakeSam3TextModule();
    auto text = std::unique_ptr<trtmc::TrtModule>(text_ptr);
    auto* vision_ptr = new FakeSam3VisionModule();
    auto vision = std::unique_ptr<trtmc::TrtModule>(vision_ptr);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(text), std::move(vision), tokenizer, make_config());

    bool threw = false;
    try {
        float pixels[3] = {0.25F, 0.5F, 0.75F};
        (void)pipeline.segment_prompted_text(pixels, 1, 1, "ear");
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("sam3_core_engine_plan") != std::string::npos;
    }
    check(threw, "sam3 text+vision path reports missing core engine");
    check(text_ptr->saw_expected_ids, "sam3 text+vision path ran text encoder");
    check(vision_ptr->saw_shape, "sam3 vision input shape");
    check(vision_ptr->saw_normalized_pixels, "sam3 vision input normalized");
}

void test_text_prompt_runs_core_engine_and_postprocesses_outputs() {
    auto* text_ptr = new FakeSam3TextModule();
    auto text = std::unique_ptr<trtmc::TrtModule>(text_ptr);
    auto* vision_ptr = new FakeSam3VisionModule();
    auto vision = std::unique_ptr<trtmc::TrtModule>(vision_ptr);
    auto* core_ptr = new FakeSam3CoreModule();
    auto core = std::unique_ptr<trtmc::TrtModule>(core_ptr);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(text), std::move(vision), std::move(core), tokenizer,
                                 make_config());

    std::vector<float> pixels(static_cast<std::size_t>(2 * 4 * 3), 0.5F);
    const auto result = pipeline.segment_prompted_text(pixels.data(), 2, 4, "ear");

    check(text_ptr->saw_expected_ids, "sam3 core path ran text encoder");
    check(vision_ptr->saw_shape, "sam3 core path ran vision encoder");
    check(core_ptr->saw_text_shape, "sam3 core text features are batched");
    check(core_ptr->saw_mask, "sam3 core receives attention mask");
    check(core_ptr->saw_vision_inputs, "sam3 core receives named FPN inputs");
    check(result.num_masks == 1, "sam3 postprocess score filter keeps one mask");
    check(result.height == 2 && result.width == 4, "sam3 postprocess resizes to original image");
    check(result.masks == std::vector<float>({0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F}),
          "sam3 postprocess binarizes resized mask");
    check(result.iou_scores.size() == 1 && close(result.iou_scores[0], 0.775803F),
          "sam3 postprocess combines text and presence scores");
    check(result.boxes.size() == 4 && close(result.boxes[0], 1.0F) &&
              close(result.boxes[1], 1.0F) && close(result.boxes[2], 3.0F) &&
              close(result.boxes[3], 2.0F),
          "sam3 postprocess scales boxes to absolute xyxy");
}

void test_text_prompt_requires_box_for_each_output_instance() {
    auto text = std::make_unique<FakeSam3TextModule>();
    auto vision = std::make_unique<FakeSam3VisionModule>();
    auto* core_ptr = new FakeSam3CoreModule();
    core_ptr->omit_second_box = true;
    auto core = std::unique_ptr<trtmc::TrtModule>(core_ptr);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    trtmc::Sam3Pipeline pipeline(std::move(text), std::move(vision), std::move(core), tokenizer,
                                 make_config());

    std::vector<float> pixels(static_cast<std::size_t>(2 * 4 * 3), 0.5F);
    const auto result = pipeline.segment_prompted_text(pixels.data(), 2, 4, "ear");

    check(result.num_masks == 0, "sam3 drops scored query without complete box");
    check(result.masks.empty(), "sam3 does not emit mask without box");
    check(result.iou_scores.empty(), "sam3 does not emit score without box");
    check(result.boxes.empty(), "sam3 does not emit partial boxes");
}

void test_constructor_requires_tokenizer() {
    bool threw = false;
    try {
        auto module = std::make_unique<FakeSam3TextModule>();
        trtmc::Sam3Pipeline pipeline(std::move(module), nullptr, make_config());
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("tokenizer") != std::string::npos;
    }
    check(threw, "sam3 constructor requires tokenizer");
}

} // namespace

int main() {
    test_text_prompt_runs_tokenizer_and_text_encoder();
    test_point_prompt_rejected();
    test_text_prompt_reports_remaining_runtime_gap();
    test_text_prompt_runs_image_preprocess_and_vision_encoder();
    test_text_prompt_runs_core_engine_and_postprocesses_outputs();
    test_text_prompt_requires_box_for_each_output_instance();
    test_constructor_requires_tokenizer();
    std::cout << "PASS\n";
    return 0;
}

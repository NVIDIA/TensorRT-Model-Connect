/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>

namespace {

class FakeForecast final : public trtmc::ITimeSeriesForecast {
  public:
    FakeForecast(trtmc::IBackend& backend, std::uint64_t kv_cache_size_bytes)
        : backend_(backend), kv_cache_size_bytes_(kv_cache_size_bytes) {}

    trtmc::ForecastResult forecast(const trtmc::ForecastRequest& request) override {
        const char* backend_name = backend_.name();
        if (backend_name == nullptr ||
            (std::string(backend_name) != "fake" && std::string(backend_name) != "trt_rtx")) {
            throw std::runtime_error("backend lifetime did not extend to task execution");
        }
        if (std::string(backend_name) == "trt_rtx")
            (void)backend_.create_module(nullptr, 0, {});
        if (!request.observed_mask.empty() &&
            request.observed_mask.size() != request.past_values.size()) {
            throw std::invalid_argument("observed_mask length must match past_values");
        }

        trtmc::ForecastResult result;
        result.values.assign(request.past_values.begin(), request.past_values.end());
        result.shape = {
            static_cast<std::int64_t>(kv_cache_size_bytes_ == 0 ? 1 : kv_cache_size_bytes_),
            static_cast<std::int64_t>(result.values.size()),
        };
        return result;
    }

  private:
    trtmc::IBackend& backend_;
    std::uint64_t kv_cache_size_bytes_;
};

class FakeEncoding final : public trtmc::IEncoding {
  public:
    trtmc::EmbeddingResult encode(const std::string& text) override {
        return {{static_cast<float>(text.size()), 1.0F, 2.0F, 3.0F}, 2};
    }
};

class FakeEmbedding final : public trtmc::IEmbedding {
  public:
    trtmc::EmbeddingResult embed(const std::string& text) override {
        return {{static_cast<float>(text.size()), 4.0F}, 2};
    }
};

class FakeImageFeatures final : public trtmc::IImageFeatureExtractor {
  public:
    trtmc::ImageFeaturesResult extract_image_features(const float*, std::int32_t,
                                                      std::int32_t) override {
        return {{1.0F, 2.0F, 3.0F, 4.0F}, {1, 2, 2}, {0.25F, 0.75F}, {1, 2}};
    }
};

class FakeImageGeneration final : public trtmc::IImageGeneration,
                                  public trtmc::IImageEditing,
                                  public trtmc::IImageBatchGeneration {
  public:
    const char* task() const noexcept override { return trtmc::IImageGeneration::kTask; }

    trtmc::ImageResult generate_image(const std::string&,
                                      const trtmc::ImageGenerationConfig&) override {
        return {{0.25F, 0.5F, 0.75F}, 1, 1, 3, 1};
    }

    trtmc::ImageResult generate_image(const std::string&, const float*, std::int32_t, std::int32_t,
                                      const trtmc::ImageGenerationConfig&) override {
        return {{0.75F, 0.5F, 0.25F}, 1, 1, 3, 1};
    }

    std::vector<trtmc::ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts, const std::vector<std::uint32_t>&,
                         const trtmc::ImageGenerationConfig& config) override {
        std::vector<trtmc::ImageResult> results;
        for (const auto& prompt : prompts)
            results.push_back(generate_image(prompt, config));
        return results;
    }
};

class FakeAudioGeneration final : public trtmc::IAudioGeneration {
  public:
    trtmc::AudioResult generate_audio(const std::string&,
                                      const trtmc::AudioGenerationConfig&) override {
        return {{0.25F, -0.5F, 0.75F, -1.0F}, 4, 8000};
    }
};

class FakeSegmentation final : public trtmc::ISegmentation {
  public:
    trtmc::SegmentResult segment(const float*, std::int32_t height, std::int32_t width) override {
        return {std::vector<std::int32_t>(static_cast<std::size_t>(height) * width, 7), height,
                width};
    }
};

class FakeObjectDetection final : public trtmc::IObjectDetection {
  public:
    trtmc::ObjectDetectionResult detect(const float*, std::int32_t height,
                                        std::int32_t width) override {
        trtmc::ObjectDetectionResult result;
        result.boxes = {{-2.0F, 1.0F, 5.0F, 1.0F, 0.75F, 42}, {0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 7}};
        result.image_height = height;
        result.image_width = width;
        return result;
    }
};

class FakeMonocularGeometry final : public trtmc::IMonocularGeometry {
  public:
    trtmc::GeometryResult estimate_geometry(const float*, std::int32_t height,
                                            std::int32_t width) override {
        trtmc::GeometryResult result;
        result.points = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        result.depth = {3.0F, 6.0F};
        result.mask = {1, 0};
        result.intrinsics = {1.0F, 0.0F, 0.5F, 0.0F, 1.0F, 0.5F, 0.0F, 0.0F, 1.0F};
        result.height = height;
        result.width = width;
        return result;
    }
};

class FakePointPromptedSegmentation final : public trtmc::IPointPromptedSegmentation {
  public:
    trtmc::PromptedSegmentationResult segment_prompted(const float*, std::int32_t height,
                                                       std::int32_t width, float, float,
                                                       bool) override {
        trtmc::PromptedSegmentationResult result;
        result.masks = {1.0F, -1.0F, -2.0F, 2.0F};
        result.iou_scores = {0.5F, 0.75F};
        result.boxes = {0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 2.0F, 1.0F};
        result.num_masks = 2;
        result.height = height;
        result.width = width;
        return result;
    }
};

trtmc::ITask* create_fake_perception_task(const trtmc::FamilyContext& context) {
    if (context.reader.info().task == trtmc::ISegmentation::kTask)
        return new FakeSegmentation();
    if (context.reader.info().task == trtmc::IObjectDetection::kTask)
        return new FakeObjectDetection();
    if (context.reader.info().task == trtmc::IMonocularGeometry::kTask)
        return new FakeMonocularGeometry();
    if (context.reader.info().task == trtmc::IPointPromptedSegmentation::kTask)
        return new FakePointPromptedSegmentation();
    return nullptr;
}

trtmc::ITask* create_fake_task(const trtmc::FamilyContext& context) {
    if (auto* task = create_fake_perception_task(context))
        return task;
    if (context.reader.info().task == trtmc::IEncoding::kTask)
        return new FakeEncoding();
    if (context.reader.info().task == trtmc::IEmbedding::kTask)
        return new FakeEmbedding();
    if (context.reader.info().task == trtmc::IImageFeatureExtractor::kTask)
        return new FakeImageFeatures();
    if (context.reader.info().task == trtmc::IImageGeneration::kTask)
        return new FakeImageGeneration();
    if (context.reader.info().task == trtmc::IAudioGeneration::kTask)
        return new FakeAudioGeneration();
    if (context.reader.info().task == trtmc::ITimeSeriesForecast::kTask)
        return new FakeForecast(context.backend, context.kv_cache_size_bytes);
    if (context.reader.info().task == trtmc::ITextGeneration::kTask) {
        // Deliberately violate the factory contract for the loader's mismatch test.
        return new FakeForecast(context.backend, context.kv_cache_size_bytes);
    }
    throw std::runtime_error("unsupported fake task");
}

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "fake")
        throw std::runtime_error("unexpected family");
    const char* backend_name = context.backend.name();
    if (backend_name == nullptr ||
        (std::string(backend_name) != "fake" && std::string(backend_name) != "trt_rtx")) {
        throw std::runtime_error("unexpected backend");
    }
    const auto plan = context.reader.read_section("engine.plan");
    if (std::string(plan.begin(), plan.end()) != "PLAN")
        throw std::runtime_error("unexpected engine plan");
    return create_fake_task(context);
}

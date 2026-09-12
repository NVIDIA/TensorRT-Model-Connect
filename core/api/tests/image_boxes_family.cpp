/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/family_factory.h"

#include <cmath>
#include <limits>

namespace {
using namespace trtmc::internal;

class ImageBoxesModel final : public IModel, public IImageToBoxes {
  public:
    explicit ImageBoxesModel(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "disabled")
            return {};
        return {bind<IImageToBoxes>(*this, {fields_, 2})};
    }
    DetectedBoxesResult run(const ImageToBoxesRequest& request, ConfigView config) override {
        if (mode_ == "must_not_run")
            throw std::runtime_error("invalid Config reached the detector");
        const trtmc::Span<const ConfigField> fields{fields_, 2};
        const auto score = config_get<double>(config, fields, "score").value();
        const auto empty = config_get<bool>(config, fields, "empty").value();
        if (!std::isfinite(score) || score < 0 || score > 1)
            throw ConfigError("score must be finite and between zero and one");
        DetectedBoxesResult result;
        result.image_height = static_cast<std::int32_t>(request.image.height);
        result.image_width = static_cast<std::int32_t>(request.image.width);
        if (!empty) {
            const float first_pixel =
                request.image.format == ImageFormat::Float32
                    ? static_cast<const float*>(request.image.data)[0]
                    : static_cast<const std::uint8_t*>(request.image.data)[0] / 255.0F;
            // Deliberately outside the image: shared code must not clamp boxes
            // or substitute a detector-specific postprocessing policy.
            result.boxes.push_back(
                {-2.0F, first_pixel, static_cast<float>(request.image.width) + 3.0F,
                 static_cast<float>(request.image.height), static_cast<float>(score), 42});
            result.boxes.push_back({0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 7});
        }
        if (mode_ == "bad_dimensions")
            result.image_width = 0;
        if (mode_ == "wrong_image_dimensions")
            result.image_width += 1;
        if (mode_ == "bad_box")
            result.boxes[0].x_max = -3.0F;
        if (mode_ == "bad_score")
            result.boxes[0].score = std::numeric_limits<float>::quiet_NaN();
        return result;
    }

  private:
    std::string mode_;
    ConfigField fields_[2] = {
        {"score", ConfigKind::F64, ConfigValue{0.75}, "Synthetic detector score"},
        {"empty", ConfigKind::Bool, ConfigValue{false}, "Return no detections"},
    };
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new ImageBoxesModel(context.reader.info().task);
}

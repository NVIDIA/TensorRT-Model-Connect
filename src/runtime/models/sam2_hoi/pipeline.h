/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::sam2_hoi {

class IPafpnComposite;

// Model-owned decoded-frame types. They intentionally stay behind the
// SAM2-HOI runtime DSO instead of extending the shared pipeline surface.
struct Sam2HoiVideoFrameView {
    const float* pixels{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct Sam2HoiVideoFrame {
    std::vector<float> pixels;
    int32_t height{0};
    int32_t width{0};

    bool empty() const { return pixels.empty(); }
    Sam2HoiVideoFrameView view() const { return {pixels.data(), height, width}; }
};

// Validate the complete model-owned output-path contract before JPEG decode or
// inference begins. The C ABI uses this seam to keep argument failures from
// poisoning a reusable session.
void validateVideoOutputPaths(const std::string& output_json, const std::string& output_masks_dir,
                              std::size_t input_frame_count);

class Sam2HoiPipeline final : public IPipeline {
  public:
    // Legacy six-plan bundles keep the monolithic image feature engine and
    // its original explicit image-engine synchronization boundary.
    Sam2HoiPipeline(std::unique_ptr<ITrtModule> image_features,
                    std::unique_ptr<ITrtModule> detector, std::unique_ptr<ITrtModule> interaction,
                    std::unique_ptr<ITrtModule> prompt_tracker,
                    std::unique_ptr<ITrtModule> recurrent_tracker,
                    std::unique_ptr<ITrtModule> memory_encoder, std::string model_id);

    // Phase-A bundles split the front from the 137-node PAFPN composite and
    // share one caller-owned stream through the detector.
    Sam2HoiPipeline(std::shared_ptr<void> image_stream_owner,
                    std::unique_ptr<ITrtModule> image_front, std::unique_ptr<IPafpnComposite> pafpn,
                    std::unique_ptr<ITrtModule> detector, std::unique_ptr<ITrtModule> interaction,
                    std::unique_ptr<ITrtModule> prompt_tracker,
                    std::unique_ptr<ITrtModule> recurrent_tracker,
                    std::unique_ptr<ITrtModule> memory_encoder, std::string model_id);
    ~Sam2HoiPipeline() override;

    Sam2HoiVideoFrame load_video_frame(const std::string& path);
    std::vector<Sam2HoiVideoFrame> load_video_frames(const std::vector<std::string>& paths);
    std::size_t max_video_frame_load_concurrency() const noexcept;

    int32_t track_video(const std::vector<Sam2HoiVideoFrameView>& frames,
                        const std::string& output_json, const std::string& output_masks_dir);

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Sam2HoiPipeline"; }

  private:
    // Destruction is reverse declaration order: all users of the raw stream
    // disappear before the owning CudaStream resource. The owner and PAFPN
    // are null only for the legacy six-plan path.
    std::shared_ptr<void> image_stream_owner_;
    std::unique_ptr<ITrtModule> image_front_;
    std::unique_ptr<IPafpnComposite> pafpn_;
    std::unique_ptr<ITrtModule> detector_;
    std::unique_ptr<ITrtModule> interaction_;
    std::unique_ptr<ITrtModule> prompt_tracker_;
    std::unique_ptr<ITrtModule> recurrent_tracker_;
    std::unique_ptr<ITrtModule> memory_encoder_;
    std::string model_id_;
};

} // namespace trtmc::sam2_hoi

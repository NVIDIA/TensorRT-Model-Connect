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

class Sam2HoiPipeline final : public IPipeline,
                              public IVideoTrackingPipeline,
                              public IVideoFrameBatchLoader {
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

    VideoFrame load_video_frame(const std::string& path) override;
    std::vector<VideoFrame> load_video_frames(const std::vector<std::string>& paths) override;
    std::size_t max_video_frame_load_concurrency() const noexcept override;

    int32_t track_video(const std::vector<VideoFrameView>& frames, const std::string& output_json,
                        const std::string& output_masks_dir) override;

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

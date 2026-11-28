# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for VideoMAE pre-training checkpoints."""

from tensorrt_model_connect.model_support import family_support


# Only the pre-training head carries the pixel decoder this family runs. Video
# classification fine-tunes share model_type "videomae" and stay unclaimed.
describe = family_support(
    architectures=("VideoMAEForPreTraining",),
    tasks=("masked_video_text_to_video",),
    default_task="masked_video_text_to_video",
)

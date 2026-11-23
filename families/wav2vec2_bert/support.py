# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("wav2vec2-bert",),
    architectures=("Wav2Vec2BertModel",),
    tasks=("spectrogram_to_frame_features",),
    default_task="spectrogram_to_frame_features",
    default_precision="fp32",
)

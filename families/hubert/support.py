# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import family_support

# The CTC head is the identity: HubertModel and HubertForSequenceClassification
# share the model type but have no transcription head, so they are not claimed.
describe = family_support(
    architectures=("HubertForCTC",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="fp32",
)

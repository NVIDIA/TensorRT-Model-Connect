# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# PointNet S3DIS test fixtures

`pointnet-s3dis-parity-input-4096.f32` is a **deterministic synthetic tensor**
that matches the PointNet S3DIS input contract (9 channels: centered XYZ, RGB in
[0,1], and room-normalized XYZ). It is used **only** for numerical/runtime
parity between the upstream PyTorch reference and the TensorRT engine. It is
**not** an S3DIS sample and is not used for model-quality (accuracy/mIoU)
evaluation.

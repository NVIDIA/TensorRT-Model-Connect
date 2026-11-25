# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt TensorRT's streaming I/O callback to a bundle section file."""

import tensorrt as trt


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)

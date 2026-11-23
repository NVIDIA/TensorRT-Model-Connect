# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT stream I/O binding for large serialized family engine plans."""

import tensorrt as trt


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        trt.IStreamWriter.__init__(self)
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)

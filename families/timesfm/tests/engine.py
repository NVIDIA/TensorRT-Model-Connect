# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the exact built plan to validate every exported forecast channel."""

import numpy as np


class ForecastEngine:
    def __init__(self, plan: bytes, context_length: int):
        import tensorrt as trt

        self.context_length = context_length
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        assert self.context is not None

    def run(self, inputs: dict) -> dict:
        import tensorrt as trt
        import torch

        raw = np.asarray(inputs["past_values"], dtype=np.float32)[-self.context_length :]
        values = np.zeros((1, self.context_length), dtype=np.float32)
        values[0, -raw.size :] = raw
        padding = np.ones((1, self.context_length), dtype=np.int32)
        observed = np.asarray(
            inputs.get("observed_mask", [1] * len(inputs["past_values"])), dtype=np.int32
        )[-self.context_length :]
        assert observed.size == raw.size and np.isin(observed, (0, 1)).all()
        padding[0, -raw.size :] = 1 - observed
        tensors = {
            name: torch.tensor(array, device="cuda")
            for name, array in {
                "past_values": values,
                "past_values_padding": padding,
                "freq": np.asarray([inputs["frequency"]], dtype=np.int32),
            }.items()
        }
        outputs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                assert self.engine.get_tensor_dtype(name) == trt.float32
                tensors[name] = torch.empty(
                    tuple(self.engine.get_tensor_shape(name)), device="cuda", dtype=torch.float32
                )
                outputs.append(name)
            assert self.context.set_tensor_address(name, tensors[name].data_ptr())
        assert self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return {name: tensors[name].cpu().numpy() for name in outputs}

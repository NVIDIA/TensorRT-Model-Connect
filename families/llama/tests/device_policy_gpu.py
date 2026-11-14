# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU policy contract probes: python -m families.llama.tests.device_policy_gpu."""

import json

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

from ..speculative.device_policy import (
    LOGGER, build_accept, build_gather_features, build_gather_kv, build_mapping, build_prepare, parents_for,
)
from .selection_gpu import checked


class Runner:
    def __init__(self, plan):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def __call__(self, **inputs):
        stream = checked(cuda.cudaStreamCreate())
        buffers = []
        outputs = {}
        try:
            for name, value in inputs.items():
                assert value.flags.c_contiguous
                if -1 in self.engine.get_tensor_shape(name):
                    assert self.context.set_input_shape(name, value.shape)
                pointer = checked(cuda.cudaMalloc(value.nbytes))
                buffers.append(pointer)
                assert self.context.set_tensor_address(name, pointer)
                checked(cuda.cudaMemcpyAsync(pointer, value.ctypes.data, value.nbytes,
                                             cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    value = np.empty(tuple(self.context.get_tensor_shape(name)),
                                     dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
                    pointer = checked(cuda.cudaMalloc(value.nbytes))
                    buffers.append(pointer)
                    assert self.context.set_tensor_address(name, pointer)
                    outputs[name] = (value, pointer)
            assert self.context.execute_async_v3(stream)
            for value, pointer in outputs.values():
                checked(cuda.cudaMemcpyAsync(value.ctypes.data, pointer, value.nbytes,
                                             cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            checked(cuda.cudaStreamSynchronize(stream))
            return {name: value for name, (value, _) in outputs.items()}
        finally:
            checked(cuda.cudaStreamSynchronize(stream))
            for pointer in buffers:
                checked(cuda.cudaFree(pointer))
            checked(cuda.cudaStreamDestroy(stream))


def check_policies():
    rng = np.random.default_rng(192)
    checks = 0
    for depth in (0, 1, 4):
        for width in (1, 2):
            parents = parents_for(depth, width)
            rows = len(parents)
            prepare = Runner(build_prepare(32, parents))
            for start in (0, 7, 32 - rows):
                actual = prepare(start=np.array([start], np.int32))
                mask = np.zeros((rows, 32), np.int32)
                positions = []
                for row, parent in enumerate(parents):
                    positions.append(start if parent < 0 else positions[parent] + 1)
                    mask[row, :start] = 1
                    ancestor = row
                    while ancestor >= 0:
                        mask[row, start + ancestor] = 1
                        ancestor = parents[ancestor]
                np.testing.assert_array_equal(actual['attention_mask'], mask)
                np.testing.assert_array_equal(actual['position_id'], positions)
                np.testing.assert_array_equal(actual['key_value_lengths'], [start + rows])
                checks += 1
            accept = Runner(build_accept(depth, width, [42, 43]))
            for case in range(80):
                tokens = np.arange(rows, dtype=np.int32) + 40
                selection = np.column_stack([rng.integers(40, 40 + rows + 2, rows),
                                             np.ones(rows)]).astype(np.int32)
                # Include full best-branch acceptance and a second-choice leaf.
                if case < 4:
                    for row in range(rows):
                        children = [i for i, parent in enumerate(parents) if parent == row]
                        if children:
                            selection[row, 0] = tokens[children[-1 if case % 2 else 0]]
                if case % 9 == 0:
                    selection[case % rows, 1] = 0
                flags = np.ones(depth, np.int32)
                if depth and case % 13 == 0:
                    flags[-1] = 0
                remaining, ignore_eos = case % (depth + 2) + 1, case % 2
                inputs = dict(tokens=tokens, selection=selection,
                              limits=np.array([remaining, ignore_eos], np.int32))
                if depth:
                    inputs['draft_valid'] = flags
                actual = accept(**inputs)
                path = [0]
                while True:
                    child = next((i for i, parent in enumerate(parents)
                                  if parent == path[-1] and tokens[i] == selection[path[-1], 0]), None)
                    if child is None:
                        break
                    path.append(child)
                output = [int(tokens[row]) for row in path[1:]] + [int(selection[path[-1], 0])]
                emitted = output[:remaining]
                if not ignore_eos:
                    first_eos = next((i for i, token in enumerate(emitted) if token in (42, 43)), None)
                    if first_eos is not None:
                        emitted = emitted[:first_eos + 1]
                stopped = len(emitted) == remaining or (not ignore_eos and emitted[-1] in (42, 43))
                valid = bool(flags.all() and selection[path, 1].all())
                np.testing.assert_array_equal(actual['path'][:len(path)], path)
                np.testing.assert_array_equal(actual['tokens_out'][:len(output)], output)
                np.testing.assert_array_equal(actual['status'], [len(path), len(emitted), stopped, valid])
                np.testing.assert_array_equal(actual['bonus'], [output[-1]])
                checks += 1
    mapping = np.array([7, 2, 9, 6], np.int32)
    mapped = Runner(build_mapping(mapping))
    for selection in ([[0, 3, 1]], [[2, 1, 1]], [[4, -1, 0]]):
        actual = mapped(selection=np.array(selection, np.int32))
        np.testing.assert_array_equal(actual['tokens'], mapping[np.clip(selection[0][:2], 0, 3)])
        np.testing.assert_array_equal(actual['valid'], [selection[0][2]])
        checks += 1
    features = rng.normal(size=(9, 12)).astype(np.float16)
    gather_features = Runner(build_gather_features(12, 9, 5))
    gather_kv = Runner(build_gather_kv(2, 32, 4, 5))
    keys = rng.normal(size=(1, 2, 32, 4)).astype(np.float16)
    values = rng.normal(size=keys.shape).astype(np.float16)
    for rows in ([0, 1, 4], [0], [0, 1, 3, 5, 8], [0, 2]):
        path = np.array(rows, np.int32)
        actual = gather_features(features=features, path=path)
        np.testing.assert_array_equal(actual['features_out'], features[path])
        actual = gather_kv(k=keys, v=values, start=np.array([7], np.int32), path=path)
        np.testing.assert_array_equal(actual['k_out'], keys[:, :, 7 + path, :])
        np.testing.assert_array_equal(actual['v_out'], values[:, :, 7 + path, :])
        checks += 1
    return {'cases': checks, 'passed': True}


if __name__ == '__main__':
    print(json.dumps(check_policies()))

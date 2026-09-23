# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless TensorRT policy graphs. Model plans and the attention ABI are unchanged.

Topology is compiled for the bounded EAGLE3 chain/two-choice policy. Acceptance
returns a padded path plus an explicit length; padding is never committed.
"""

import numpy as np
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)


class PolicyGraph:
    def __init__(self):
        self.builder = trt.Builder(LOGGER)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 1
        self.profile = self.builder.create_optimization_profile()
        self.dynamic = False

    def input(self, name, dtype, shape, maximum=None):
        result = self.net.add_input(name, dtype, shape)
        if maximum is not None:
            minimum = tuple(1 if d == -1 else d for d in shape)
            self.profile.set_shape(name, minimum, maximum, maximum)
            self.dynamic = True
        return result

    def const(self, value, dtype=np.int32):
        array = np.ascontiguousarray(value, dtype=dtype)
        return self.net.add_constant(array.shape, array).get_output(0)

    def op(self, name, a, b):
        return self.net.add_elementwise(a, b, getattr(trt.ElementWiseOperation, name)).get_output(0)

    def gather(self, values, indices, axis=0):
        return self.net.add_gather(values, indices, axis).get_output(0)

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def at(self, value, index):
        return self.gather(value, self.const([index]))

    def choose(self, condition, yes, no):
        return self.net.add_select(condition, yes, no).get_output(0)

    def integer(self, value):
        return self.net.add_cast(value, trt.int32).get_output(0)

    def reduce(self, value, operation, axis=0):
        return self.net.add_reduce(value, getattr(trt.ReduceOperation, operation),
                                   1 << axis, True).get_output(0)

    def concat(self, values):
        layer = self.net.add_concatenation(values)
        layer.axis = 0
        return layer.get_output(0)

    def output(self, name, value):
        value.name = name
        self.net.mark_output(value)

    def finish(self):
        if self.dynamic:
            self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.net, self.config)
        if plan is None:
            raise RuntimeError("failed to build device policy graph")
        return bytes(plan)


def parents_for(depth, width):
    return [-1] + [0 if level == 0 else 1 + (level - 1) * width
                   for level in range(depth) for _ in range(width)]


def build_prepare(capacity, parents):
    g = PolicyGraph()
    rows = len(parents)
    start = g.input("start", trt.int32, (1,))
    depths = np.zeros(rows, np.int32)
    ancestors = np.zeros((rows, rows), np.int32)
    for row, parent in enumerate(parents):
        depths[row] = 0 if parent < 0 else depths[parent] + 1
        current = row
        while current >= 0:
            ancestors[row, current] = 1
            current = parents[current]
    columns = g.const(np.arange(capacity, dtype=np.int32))
    relative = g.op("SUB", columns, start)
    clipped = g.op("MIN", g.op("MAX", relative, g.const([0])), g.const([rows - 1]))
    visible = g.gather(g.const(ancestors), clipped, 1)
    in_span = g.op("AND", g.op("GREATER", relative, g.const([-1])),
                   g.op("LESS", relative, g.const([rows])))
    visible = g.op("AND", g.op("EQUAL", visible, g.const([[1]])),
                   g.reshape(in_span, (1, capacity)))
    prefix = g.reshape(g.op("LESS", columns, start), (1, capacity))
    g.output("attention_mask", g.integer(g.op("OR", prefix, visible)))
    g.output("position_id", g.op("SUM", start, g.const(depths)))
    g.output("key_value_lengths", g.op("SUM", start, g.const([rows])))
    g.output("logits_all", g.const(np.arange(rows, dtype=np.int32)))
    g.output("logits_last", g.const([rows - 1]))
    return g.finish()


def build_mapping(mapping):
    g = PolicyGraph()
    selection = g.reshape(g.input("selection", trt.int32, (1, 3)), (3,))
    ids = g.gather(selection, g.const([0, 1]))
    # Invalid rows are reported at the round boundary; keep indexing safe while
    # queued work finishes, even if selection returned an invalid sentinel.
    ids = g.op("MIN", g.op("MAX", ids, g.const([0])), g.const([len(mapping) - 1]))
    g.output("tokens", g.gather(g.const(mapping), ids))
    g.output("valid", g.at(selection, 2))
    return g.finish()


def build_accept(depth, width, eos):
    g = PolicyGraph()
    parents = parents_for(depth, width)
    rows = len(parents)
    tokens = g.input("tokens", trt.int32, (rows,))
    selection = g.input("selection", trt.int32, (rows, 2))
    limits = g.input("limits", trt.int32, (2,))  # remaining output IDs, ignore EOS
    zero, one = g.const([0]), g.const([1])
    valid = g.op("EQUAL", one, one)
    if depth:
        draft_valid = g.input("draft_valid", trt.int32, (depth,))
        valid = g.op("EQUAL", g.reduce(draft_valid, "MIN"), one)
    row, length = zero, one
    active = g.op("EQUAL", one, one)
    path = [zero]
    bonus = zero
    for level in range(depth + 1):
        prediction = g.reshape(g.gather(selection, row), (2,))
        next_token = g.at(prediction, 0)
        finite = g.op("EQUAL", g.at(prediction, 1), one)
        valid = g.op("AND", valid, g.choose(active, finite, g.op("EQUAL", one, one)))
        bonus = g.choose(active, next_token, bonus)
        if level == depth:
            break
        match = g.op("AND", g.op("EQUAL", g.const(parents), row),
                     g.op("EQUAL", tokens, next_token))
        child = g.reduce(g.choose(match, g.const(np.arange(rows)), g.const([rows])), "MIN")
        active = g.op("AND", active, g.op("LESS", child, g.const([rows])))
        row = g.choose(active, child, zero)
        path.append(row)
        length = g.op("SUM", length, g.integer(active))
    path = g.concat(path) if len(path) > 1 else path[0]
    slots = g.const(np.arange(depth + 1))
    next_rows = g.gather(path, g.const(np.minimum(np.arange(depth + 1) + 1, depth)))
    output = g.choose(g.op("LESS", slots, g.op("SUB", length, one)),
                      g.gather(tokens, next_rows), bonus)
    is_eos = g.op("EQUAL", output, g.const([-1]))
    for token in eos:
        is_eos = g.op("OR", is_eos, g.op("EQUAL", output, g.const([token])))
    is_eos = g.op("AND", is_eos, g.op("EQUAL", g.at(limits, 1), zero))
    is_eos = g.op("AND", is_eos, g.op("LESS", slots, length))
    stop_at = g.reduce(g.choose(is_eos, g.op("SUM", slots, one), g.const([depth + 2])), "MIN")
    count = g.op("MIN", g.op("MIN", length, g.at(limits, 0)), stop_at)
    stopped = g.op("OR", g.op("EQUAL", count, g.at(limits, 0)),
                    g.op("EQUAL", count, stop_at))
    g.output("path", path)
    g.output("tokens_out", output)  # first L rows also serve as shifted feedback tokens
    g.output("bonus", bonus)
    g.output("status", g.concat([length, count, g.integer(stopped), g.integer(valid)]))
    return g.finish()


def build_gather_features(width, max_rows, max_path):
    g = PolicyGraph()
    features = g.input("features", trt.float16, (-1, width), (max_rows, width))
    path = g.input("path", trt.int32, (-1,), (max_path,))
    g.output("features_out", g.gather(features, path))
    return g.finish()


def build_gather_kv(heads, capacity, dim, max_path):
    g = PolicyGraph()
    path = g.input("path", trt.int32, (-1,), (max_path,))
    start = g.input("start", trt.int32, (1,))
    slots = g.op("SUM", path, start)
    for name in ("k", "v"):
        cache = g.input(name, trt.float16, (1, heads, capacity, dim))
        g.output(f"{name}_out", g.gather(cache, slots, 2))
    return g.finish()


def add_device_policy_plans(writer, target, draft, depth, mapping, eos):
    """Add optional policy plans without rebuilding or modifying model plans."""
    if target.greedy_selection != "device_v1" or draft.greedy_selection != "device_v1":
        raise ValueError("device policy requires device_v1 selection for both models")
    writer.add_bytes("policy_mapping.plan", build_mapping(mapping))
    for width in (1, 2):
        for level in range(depth + 1):
            parents = parents_for(level, width)
            writer.add_bytes(f"policy_prepare_{width}_{level}.plan",
                             build_prepare(target.capacity, parents))
            writer.add_bytes(f"policy_accept_{width}_{level}.plan", build_accept(level, width, eos))
    writer.add_bytes("policy_features.plan", build_gather_features(
        target.feature_width, 2 * depth + 1, depth + 1))
    writer.add_bytes("policy_kv.plan", build_gather_kv(
        target.kv_heads, target.capacity, target.head_dim, depth + 1))

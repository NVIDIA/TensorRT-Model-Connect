# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph-contract checks for memory-bounded Turbo AdaLN partitions."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.minimax_h3.config import SOL_ENGINE_1344X768_124F, TURBO_124_TO_362F


@pytest.fixture
def builder_functions():
    # Execute the actual builder functions with recording native-API doubles.
    # No TensorRT import, device creation, checkpoint allocation, or GPU work.
    source = Path(__file__).resolve().parents[1] / "adaln_builder.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_block_range", "checkpoint_keys", "build_adaln_precompute_engine"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in functions:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    outputs = []
    linears = []

    class Tensor:
        name = ""

    class Network:
        def add_input(self, *_args):
            return Tensor()

        def add_shuffle(self, _value):
            result = Tensor()
            return SimpleNamespace(get_output=lambda _index: result)

        def mark_output(self, value):
            outputs.append(value.name)

    network = Network()
    configuration = SimpleNamespace(clear_flag=lambda _flag: None)
    builder = SimpleNamespace(
        create_network=lambda _flags: network,
        create_builder_config=lambda: configuration,
        build_serialized_network=lambda _network, _config: b"cpu-graph-contract",
    )

    class Logger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, _level):
            pass

    def linear(_network, _tensor, weight, bias=None, **_kwargs):
        linears.append((weight, bias))
        return Tensor()

    globals_ = {
        "SOL_ENGINE_1344X768_124F": SOL_ENGINE_1344X768_124F,
        "TurboLoraWeight": type("TurboLoraWeight", (), {}),
        "trt": SimpleNamespace(
            Logger=Logger,
            Builder=lambda _logger: builder,
            float32="fp32",
            bfloat16="bf16",
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
            BuilderFlag=SimpleNamespace(TF32="tf32"),
        ),
        "op": SimpleNamespace(
            configure_builder=lambda *_args, **_kwargs: None,
            linear=linear,
            silu=lambda _network, value: value,
            cast=lambda _network, value, _dtype: value,
            validate_native_network=lambda *_args, **_kwargs: None,
            release_weight_buffers=lambda _network: None,
        ),
        "sys": SimpleNamespace(stderr=None),
        "gc": SimpleNamespace(collect=lambda: None),
    }
    exec(compile(module, str(source), "exec"), globals_)
    return SimpleNamespace(**globals_, outputs=outputs, linears=linears)


def test_checkpoint_partitions_cover_original_keys_without_extra_payload(builder_functions):
    keys = builder_functions.checkpoint_keys
    profile = TURBO_124_TO_362F
    full = keys(profile)
    first = keys(profile, block_end=25)
    second = keys(profile, block_start=25, include_final=False)
    shared = set(full[:4])
    assert set(first) & set(second) == shared
    assert set(first) | set(second) == set(full)
    assert len(first) == 56
    assert len(second) == 54
    assert "norm_out.linear.weight" in first
    assert "norm_out.linear.weight" not in second
    assert keys(SOL_ENGINE_1344X768_124F) == full


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_start": -1},
        {"block_end": 51},
        {"block_start": 25, "block_end": 25},
        {"block_start": 50},
        {"block_start": True},
        {"block_end": 25.0},
        {"include_final": 0},
    ],
)
def test_invalid_ranges_fail_before_graph_creation(builder_functions, kwargs):
    with pytest.raises(ValueError):
        builder_functions.checkpoint_keys(TURBO_124_TO_362F, **kwargs)
    with pytest.raises(ValueError):
        builder_functions.build_adaln_precompute_engine({}, TURBO_124_TO_362F, **kwargs)
    assert not builder_functions.outputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_end": 25},
        {"block_start": 25},
        {"include_final": False},
    ],
)
def test_original_profile_cannot_silently_be_partitioned(builder_functions, kwargs):
    with pytest.raises(ValueError, match="require the Turbo profile"):
        builder_functions.checkpoint_keys(SOL_ENGINE_1344X768_124F, **kwargs)
    with pytest.raises(ValueError, match="require the Turbo profile"):
        builder_functions.build_adaln_precompute_engine({}, SOL_ENGINE_1344X768_124F, **kwargs)


@pytest.mark.parametrize(
    "profile,start,end,final",
    [
        (TURBO_124_TO_362F, 0, 25, True),
        (TURBO_124_TO_362F, 25, 50, False),
        (SOL_ENGINE_1344X768_124F, 0, 50, True),
    ],
)
def test_build_emits_selected_global_block_names_only(
    builder_functions, profile, start, end, final
):
    kwargs = dict(block_start=start, block_end=end, include_final=final)
    keys = builder_functions.checkpoint_keys(profile, **kwargs)
    weights = {key: key for key in keys}
    result = builder_functions.build_adaln_precompute_engine(
        weights,
        profile,
        consume_weights=True,
        **kwargs,
    )
    assert result == b"cpu-graph-contract"
    assert weights == {}
    assert builder_functions.outputs == [
        *[f"block_modulation_{index}" for index in range(start, end)],
        *(["final_modulation"] if final else []),
    ]
    assert {name for pair in builder_functions.linears for name in pair} == set(keys)

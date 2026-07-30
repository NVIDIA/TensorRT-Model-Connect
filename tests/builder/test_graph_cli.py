# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import graph_cli
from tensorrt_model_connect.graph_patch import (
    GraphSnapshot,
    Node,
    Tensor,
    load_selection,
)


def _parse(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    graph_cli.configure_parser(parser)
    return parser.parse_args(arguments)


def test_list_prints_explicit_node_ids(monkeypatch, capsys):
    snapshot = SimpleNamespace(
        fingerprint="sha256:graph",
        nodes=(
            SimpleNamespace(
                id="node:7",
                op="MATRIX_MULTIPLY",
                name="attention/qk",
                inputs=("tensor:q", "tensor:k"),
                outputs=("tensor:scores",),
            ),
            SimpleNamespace(
                id="node:8",
                op="SOFTMAX",
                name="attention/softmax",
                inputs=("tensor:scores",),
                outputs=("tensor:probabilities",),
            ),
        ),
    )
    monkeypatch.setattr(graph_cli, "load_snapshot", lambda path: snapshot)

    result = graph_cli.run(_parse("list", "graph.json", "--match", "SOFTMAX"))

    assert result == 0
    assert capsys.readouterr().out.splitlines() == [
        "graph: sha256:graph",
        "ID\tOP\tNAME\tINPUTS\tOUTPUTS",
        "node:8\tSOFTMAX\tattention/softmax\ttensor:scores\ttensor:probabilities",
    ]


def test_inspect_forwards_a_small_native_build_command(monkeypatch):
    recorded = SimpleNamespace(command=None)

    def run(command, *, check):
        recorded.command = command
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(graph_cli.subprocess, "run", run)
    result = graph_cli.run(
        _parse(
            "inspect",
            "--snapshot",
            "graph.json",
            "--engine-role",
            "decode",
            "Qwen/Qwen3-8B",
            "--model-revision",
            "revision",
            "--precision",
            "bf16",
            "--family-option",
            "value",
        )
    )

    assert result == 0
    assert recorded.command[-13:] == [
        "Qwen/Qwen3-8B",
        "--model-revision",
        "revision",
        "--precision",
        "bf16",
        "--family-option",
        "value",
        "-o",
        graph_cli.os.devnull,
        "--graph-snapshot",
        "graph.json",
        "--graph-role",
        "decode",
    ]


def test_select_forwards_only_the_requested_replacement_contract(monkeypatch, capsys, tmp_path):
    snapshot = GraphSnapshot(
        nodes=(
            Node(
                "node:7",
                "attention/qk",
                "MATRIX_MULTIPLY",
                ("tensor:q", "tensor:k"),
                ("tensor:scores",),
            ),
            Node("node:8", "attention/softmax", "SOFTMAX", ("tensor:scores",), ("tensor:context",)),
            Node("node:9", "attention/output", "IDENTITY", ("tensor:context",), ("tensor:final",)),
        ),
        tensors=(
            Tensor("tensor:q", "q", "FLOAT", (-1, 64), None, ("node:7",), False, "DEVICE"),
            Tensor("tensor:k", "k", "FLOAT", (-1, 64), None, ("node:7",), False, "DEVICE"),
            Tensor(
                "tensor:scores", "scores", "FLOAT", (-1, 64), "node:7", ("node:8",), False, "DEVICE"
            ),
            Tensor(
                "tensor:context",
                "context",
                "FLOAT",
                (-1, 64),
                "node:8",
                ("node:9",),
                False,
                "DEVICE",
            ),
            Tensor("tensor:final", "final", "FLOAT", (-1, 64), "node:9", (), False, "DEVICE"),
        ),
        inputs=("tensor:q", "tensor:k"),
        outputs=("tensor:final",),
        metadata={"engine_role": "decoder"},
        fingerprint="sha256:graph",
    )
    monkeypatch.setattr(graph_cli, "load_snapshot", lambda path: snapshot)
    output = tmp_path / "attention.json"
    arguments = _parse(
        "select",
        "graph.json",
        "--nodes",
        "node:7",
        "node:8",
        "--binding-id",
        "flashinfer.decode@1",
        "--workspace-bytes",
        "4096",
        "--output-shape-like-input",
        "0",
        "--extra-arg",
        '{"type":"int","value":32}',
        "--extra-arg",
        '{"type":"float","value":0.5}',
        "-o",
        str(output),
    )

    result = graph_cli.run(arguments)
    selection = load_selection(output)

    assert result == 0
    assert selection.node_ids == ("node:7", "node:8")
    assert selection.input_tensor_ids == ("tensor:q", "tensor:k")
    assert selection.output_tensor_ids == ("tensor:context",)
    assert selection.engine_role == "decoder"
    assert selection.binding_id == "flashinfer.decode@1"
    assert selection.kernel_name == "trtmc.slot.flashinfer.decode@1"
    assert len(selection.abi_sha256) == 64
    assert set(selection.abi_sha256) <= set("0123456789abcdef")
    assert selection.workspace_bytes == 4096
    assert selection.output_shape_input == 0
    assert selection.extra_args == (
        {"type": "int", "value": 32},
        {"type": "float", "value": 0.5},
    )
    assert capsys.readouterr().out.splitlines() == [
        f"Wrote selection to {output}",
        "input[0]: tensor:q name='q' dtype=FLOAT shape=[-1, 64]",
        "input[1]: tensor:k name='k' dtype=FLOAT shape=[-1, 64]",
        "output[0]: tensor:context name='context' dtype=FLOAT shape=[-1, 64]",
        f"abi_sha256: {selection.abi_sha256}",
    ]


def test_recipe_is_sugar_for_the_same_selection(monkeypatch, capsys, tmp_path):
    snapshot = GraphSnapshot(
        nodes=(
            Node("node:0", "known_region", "IDENTITY", ("tensor:input",), ("tensor:middle",)),
            Node("node:1", "consumer", "IDENTITY", ("tensor:middle",), ("tensor:output",)),
        ),
        tensors=(
            Tensor(
                "tensor:input",
                "input",
                "HALF",
                (1, 8),
                None,
                ("node:0",),
                False,
                "DEVICE",
            ),
            Tensor(
                "tensor:middle",
                "middle",
                "HALF",
                (1, 8),
                "node:0",
                ("node:1",),
                False,
                "DEVICE",
            ),
            Tensor(
                "tensor:output",
                "output",
                "HALF",
                (1, 8),
                "node:1",
                (),
                False,
                "DEVICE",
            ),
        ),
        inputs=("tensor:input",),
        outputs=("tensor:output",),
        metadata={
            "engine_role": "decode",
            "graph_recipes": [
                {
                    "id": "family.known_region@1",
                    "instance": "decoder.layer.0",
                    "node_ids": ["node:0"],
                    "workspace_bytes": 0,
                    "extra_args": [],
                    "output_shape_input": None,
                }
            ],
        },
        fingerprint="sha256:graph",
    )
    monkeypatch.setattr(graph_cli, "load_snapshot", lambda path: snapshot)

    assert graph_cli.run(_parse("recipe", "list", "graph.json")) == 0
    assert "family.known_region@1\tdecoder.layer.0\tnode:0" in (
        capsys.readouterr().out
    )

    output = tmp_path / "recipe.json"
    assert graph_cli.run(
        _parse(
            "recipe",
            "apply",
            "graph.json",
            "family.known_region@1",
            "--instance",
            "decoder.layer.0",
            "-o",
            str(output),
        )
    ) == 0
    recipe_selection = load_selection(output)
    assert recipe_selection.binding_id == "family.known_region@1"
    assert recipe_selection.node_ids == ("node:0",)
    assert recipe_selection.input_tensor_ids == ("tensor:input",)
    assert recipe_selection.output_tensor_ids == ("tensor:middle",)


def test_select_rejects_negative_workspace():
    with pytest.raises(SystemExit):
        _parse(
            "select",
            "graph.json",
            "--nodes",
            "node:7",
            "--binding-id",
            "kernel@1",
            "--workspace-bytes",
            "-1",
            "-o",
            "selection.json",
        )

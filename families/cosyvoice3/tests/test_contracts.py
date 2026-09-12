# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light tests. These do not assert end-to-end TTS support."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from families.cosyvoice3.checkpoint_mapper import expected_shapes, validate_weights
from families.cosyvoice3.config import FlowConfig, ShapeProfile, read_config
from families.cosyvoice3.flow_runtime import validate_input_shapes
from families.cosyvoice3.constants import time_frequencies


YAML = """
sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
token_frame_rate: 25
token_mel_ratio: 2
llm: !new:cosyvoice.llm.llm.CosyVoice3LM
  speech_token_size: 6561
flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
  vocab_size: 6561
  decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
    estimator: !new:cosyvoice.flow.DiT.dit.DiT
      dim: 1024
      depth: 22
      heads: 16
      dim_head: 64
      ff_mult: 2
      mel_dim: 80
      mu_dim: 80
      spk_dim: 80
      out_channels: 80
"""


def test_read_published_config_without_executing_tags(tmp_path):
    target = tmp_path / "executed"
    payload = YAML + f'\ndanger: !!python/object/apply:pathlib.Path.touch ["{target.as_posix()}"]\n'
    (tmp_path / "cosyvoice3.yaml").write_text(payload, encoding="utf-8")
    (tmp_path / "config.json").write_text("{}")
    assert read_config(tmp_path) == FlowConfig()
    assert not target.exists()


@pytest.mark.parametrize("old,new", [("24000", "22050"), ("depth: 22", "depth: 21"),
                                       ("mu_dim: 80", "mu_dim: 64"), ("speech_token_size: 6561", "speech_token_size: 4096")])
def test_reject_other_architecture(tmp_path, old, new):
    (tmp_path / "cosyvoice3.yaml").write_text(YAML.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(tmp_path)


@pytest.mark.parametrize("value", ["[]", "null", "flow: {}"])
def test_reject_malformed_yaml(tmp_path, value):
    (tmp_path / "cosyvoice3.yaml").write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(tmp_path)


@pytest.mark.parametrize("values", [(0, 1, 2), (5, 4, 8), (1, 4, 3), (1, 4, 15001), (True, 4, 8)])
def test_bad_profile(values):
    with pytest.raises(ValueError):
        ShapeProfile(*values)


@pytest.fixture
def tiny_config():
    return FlowConfig(dim=16, depth=2, heads=2, head_dim=8, ff_mult=2,
                      mel_dim=4, spk_dim=4, time_dim=8, conv_kernel=3, conv_groups=2)


@pytest.fixture
def tiny_weights(tiny_config):
    rng = np.random.default_rng(0)
    return {name: rng.normal(0, 0.1, shape).astype(np.float32) for name, shape in expected_shapes(tiny_config).items()}


def test_checkpoint_shapes(tiny_config, tiny_weights):
    assert len(validate_weights(tiny_weights, tiny_config)) == 42


def test_checkpoint_preserves_rotary_buffer(tiny_config, tiny_weights):
    inv = (1 / (10000 ** (np.arange(0, tiny_config.head_dim, 2) / tiny_config.head_dim))).astype(np.float32)
    tiny_weights["rotary_embed.inv_freq"] = inv
    np.testing.assert_array_equal(validate_weights(tiny_weights, tiny_config)["rotary_embed.inv_freq"], inv)


def test_published_time_coefficients():
    frequencies = time_frequencies()
    assert frequencies.shape == (128,) and frequencies.dtype == np.float32
    assert np.isfinite(frequencies).all() and (np.diff(frequencies) < 0).all()
    expected = np.exp(-np.arange(128) * np.log(10000) / 127)
    np.testing.assert_allclose(frequencies, expected, rtol=1e-6)
    # Coefficients must not silently be recalculated with float64 exp.
    assert frequencies[1].view(np.uint32) == 1064179564


@pytest.mark.parametrize("defect", ["missing", "extra", "shape", "nan", "integer", "rope"])
def test_checkpoint_rejects_defects(tiny_config, tiny_weights, defect):
    key = "proj_out.weight"
    if defect == "missing":
        del tiny_weights[key]
    elif defect == "extra":
        tiny_weights["wrong"] = np.zeros(1)
    elif defect == "shape":
        tiny_weights[key] = tiny_weights[key].T
    elif defect == "nan":
        tiny_weights[key][0, 0] = np.nan
    elif defect == "integer":
        tiny_weights[key] = tiny_weights[key].astype(np.int32)
    else:
        tiny_weights["rotary_embed.inv_freq"] = np.zeros(tiny_config.head_dim // 2)
    with pytest.raises(ValueError):
        validate_weights(tiny_weights, tiny_config)


def input_shapes(frames=8):
    return {"x": (2, 80, frames), "mask": (2, 1, frames), "mu": (2, 80, frames),
            "t": (2,), "spks": (2, 80), "cond": (2, 80, frames)}


def test_input_contract():
    assert validate_input_shapes(input_shapes(), FlowConfig(), ShapeProfile()) == 8


@pytest.mark.parametrize("name,shape", [("mu", (2, 80, 7)), ("x", (1, 80, 8)), ("spks", (2, 192)),
                                        ("t", ()), ("mask", (2, 80, 8)), ("x", (2, 80, 257))])
def test_bad_input_contract(name, shape):
    shapes = input_shapes()
    shapes[name] = shape
    with pytest.raises(ValueError):
        validate_input_shapes(shapes, FlowConfig(), ShapeProfile())


def test_package_import_and_help_are_dependency_light():
    code = ("import sys; import families.cosyvoice3; "
            "assert 'torch' not in sys.modules; assert 'tensorrt' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run([sys.executable, "-m", "families.cosyvoice3", "--help"],
                            check=True, capture_output=True, text=True)
    assert "not full TTS" in result.stdout


def test_inspect_cli(tmp_path):
    (tmp_path / "cosyvoice3.yaml").write_text(YAML, encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "families.cosyvoice3",
                             "inspect", "--model-dir", str(tmp_path)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["status"] == "component_only"


def test_no_sibling_family_or_onnx_builder_dependency():
    source = (Path(__file__).parents[1] / "flow_builder.py").read_text(encoding="utf-8")
    assert "OnnxParser" not in source
    assert "torch.onnx" not in source


def test_build_records_workspace(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from families.cosyvoice3 import trt_compat
    from families.cosyvoice3 import __main__ as cli
    from families.cosyvoice3 import checkpoint_mapper, flow_builder

    model = tmp_path / "model"
    model.mkdir()
    for name in ("flow.pt", "cosyvoice3.yaml"):
        (model / name).write_bytes(b"fixture")
    monkeypatch.setattr(cli, "read_config", lambda path: FlowConfig())
    monkeypatch.setattr(checkpoint_mapper, "load_flow_weights", lambda *args: {})
    monkeypatch.setattr(flow_builder, "build_flow_engine", lambda *args, **kwargs: b"test plan")
    monkeypatch.setattr(trt_compat, "module_version", lambda: "test")
    args = SimpleNamespace(output=tmp_path / "output", model_dir=model,
                           min_frames=4, opt_frames=32, max_frames=128, workspace_mib=256)
    cli.build(args)
    metadata = json.loads((args.output / "manifest.json").read_text())
    assert metadata["workspace_mib"] == 256
    assert metadata["component"] == "cosyvoice3_flow_estimator"


def test_component_publication_preserves_existing_output(tmp_path):
    from families.cosyvoice3.artifacts import write_component

    output = tmp_path / "component"
    manifest = {"component": "fixture"}
    write_component(output, "flow.plan", b"plan", manifest)
    assert (output / "flow.plan").read_bytes() == b"plan"
    assert json.loads((output / "manifest.json").read_text()) == manifest
    original = (output / "flow.plan").read_bytes()
    with pytest.raises(FileExistsError):
        write_component(output, "flow.plan", b"replacement", {})
    assert (output / "flow.plan").read_bytes() == original


def test_component_publication_failure_is_not_visible(tmp_path, monkeypatch):
    from families.cosyvoice3 import artifacts

    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(artifacts.os, "rename", fail)
    with pytest.raises(OSError, match="simulated publication"):
        artifacts.write_component(tmp_path / "component", "flow.plan", b"plan", {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["../flow.plan", "manifest.json"])
def test_component_publication_rejects_invalid_plan_name(tmp_path, filename):
    from families.cosyvoice3.artifacts import write_component

    with pytest.raises(ValueError, match="plan_name"):
        write_component(tmp_path / "component", filename, b"plan", {})
    assert list(tmp_path.iterdir()) == []

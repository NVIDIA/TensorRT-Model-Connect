# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete checkpoint, profile and input-boundary contracts."""

import numpy as np
import pytest
from ..checkpoint import expected_shapes, validate_weights
from ..config import Config, ServingProfile
from ..reference import example_inputs


def test_official_architecture_and_public_inputs():
    shapes = expected_shapes()
    assert len(shapes) == 86
    assert sum(np.prod(v) for v in shapes.values()) == 16693594
    c = Config()
    schema = c.runtime(4, "bf16")
    assert schema["stages"][0]["outputs"][0]["shape"] == [-1, 5, 4, 513, 128]
    inputs = example_inputs(c, 4)
    assert set(inputs) == {x["name"] for x in schema["inputs"]}
    for spec in schema["inputs"]:
        assert inputs[spec["name"]].shape == (4, *spec["shape"][1:])
        assert str(inputs[spec["name"]].dtype) == spec["dtype"]
    assert schema["outputs"] == [dict(name="scores", dtype="float32", shape=[-1, 128])]


@pytest.mark.parametrize(
    "profile",
    [
        dict(history_length=0),
        dict(candidates=2049),
        dict(batching="other"),
        dict(min_batch_size=True),
    ],
)
def test_invalid_serving_profile_is_rejected(profile):
    with pytest.raises(ValueError):
        ServingProfile(**profile)


def test_larger_profile_preserves_official_architecture():
    c = Config(serving=ServingProfile(history_length=2048, candidates=256))
    assert c.runtime(4, "fp32")["stages"][0]["outputs"][0]["shape"] == [-1, 5, 4, 2049, 128]
    with pytest.raises(ValueError):
        c.runtime(0, "bf16")
    with pytest.raises(ValueError):
        c.runtime(4, "fp16")


def test_converter_rejects_partial_models():
    with pytest.raises(ValueError, match="86"):
        validate_weights({})


def test_unknown_bf16_policy_is_rejected():
    with pytest.raises(ValueError, match="BF16 policy"):
        Config(bf16_policy="unknown")


def test_bf16_policy_keeps_reduced_operations_explicit():
    config = Config(fp32_block_indices=tuple(range(5)), bf16_policy="conservative")
    with pytest.raises(ValueError, match="reduced-precision"):
        config.runtime(4, "bf16")
    config = Config(fp32_block_indices=tuple(range(5)), bf16_policy="expanded")
    assert config.runtime(4, "bf16")["bf16_extra_ops"] == [
        "candidate_av",
        "item_mlp",
        "time_projection",
        "prediction_hidden",
    ]
    assert config.runtime(4, "fp32")["bf16_extra_ops"] == []


def test_conversion_failure_is_retryable_and_provenance_names_input(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from .. import checkpoint
    from ..checkpoint import POST, PRE

    weights = {
        name: np.zeros(shape, np.int64 if name.endswith("_combined_action_weights") else np.float32)
        for name, shape in expected_shapes().items()
    }
    weights[PRE + "_action_encoder._combined_action_weights"][:] = [1, 2, 4, 8, 16]
    weights[POST + "_period_units"][:] = [[3600, 86400]]
    weights[POST + "_units_per_period"][:] = [[24, 7]]

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            load=lambda *_args, **_kwargs: {
                "dense_dict": {k: Tensor(v) for k, v in weights.items()}
            }
        ),
    )
    original = Config.save

    def fail_save(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(Config, "save", fail_save)
    output = tmp_path / "converted"
    with pytest.raises(OSError, match="disk full"):
        checkpoint.convert(tmp_path / "supplied.ckpt", output)
    assert not output.exists() and not list(tmp_path.glob(".converted-*"))
    monkeypatch.setattr(Config, "save", original)
    checkpoint.convert(tmp_path / "supplied.ckpt", output)
    provenance = json.loads((output / "checkpoint_provenance.json").read_text())
    assert provenance["source"] == str(tmp_path / "supplied.ckpt")
    assert provenance["upstream_reference"] == checkpoint.DEFAULT_URL


@pytest.mark.gpu
def test_fixed_build_uses_validated_batch_schema(dlrm_root, dlrm_bridge_root, tmp_path):
    from dataclasses import replace
    import os
    import shutil
    from tensorrt_model_connect import BuildRequest, build
    from ..checkpoint import load_weights
    from ..reference import Reference
    from ..triton.native_client import NativeClient

    source = dlrm_root / "dlrmv3-short-fp32"
    config = Config.load(source)
    config = replace(config, serving=replace(config.serving, min_batch_size=65, opt_batch_size=65))
    for name in ("dlrmv3_weights.npz", "checkpoint_provenance.json"):
        shutil.copyfile(source / name, tmp_path / name)
    config.save(tmp_path)
    bundle = tmp_path / "model.bundle"
    build(
        BuildRequest(
            model_dir=tmp_path,
            output_path=bundle,
            family="hstu",
            task="structured_ranking",
            precision="fp32",
            max_batch_size=4,
        )
    )
    inputs = example_inputs(config, 4)
    inputs["query_timestamp"].fill(1700009984)
    with NativeClient(
        bundle, os.environ["TRTMC_RUNTIME_ROOT"], bridge_root=dlrm_bridge_root
    ) as task:
        actual = task.score(inputs)["scores"]
    expected = Reference(config, load_weights(tmp_path)).score(inputs)["scores"]
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.001)

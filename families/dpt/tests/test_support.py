# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity, default task, build-configuration policy and E2E selection for DPT, without a GPU."""

import copy
import re

import pytest

from families.dpt import config as config_module
from families.dpt.support import describe
from tensorrt_model_connect.model_support import ModelMetadata

# The fields of Intel/dpt-hybrid-midas that decide the graph and the image pipeline.
_HYBRID = {
    "model_type": "dpt",
    "architectures": ["DPTForDepthEstimation"],
    "is_hybrid": True,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "image_size": 384,
    "patch_size": 16,
    "qkv_bias": True,
    "backbone_out_indices": [2, 5, 8, 11],
    "readout_type": "project",
    "reassemble_factors": [1, 1, 1, 0.5],
    "neck_hidden_sizes": [256, 512, 768, 768],
    "fusion_hidden_size": 256,
    "head_in_index": -1,
    "use_batch_norm_in_fusion_residual": False,
    "backbone_featmap_shape": [1, 1024, 24, 24],
    "neck_ignore_stages": [0, 1],
    "backbone_config": {
        "model_type": "bit",
        "depths": [3, 4, 9],
        "embedding_dynamic_padding": True,
        "embedding_size": 64,
        "global_padding": "SAME",
        "hidden_act": "relu",
        "hidden_sizes": [256, 512, 1024, 2048],
        "layer_type": "bottleneck",
        "num_groups": 32,
        "out_features": ["stage1", "stage2", "stage3"],
        "output_stride": 32,
    },
}
_PROCESSOR = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "ensure_multiple_of": 1,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "keep_aspect_ratio": False,
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 384, "width": 384},
}


def _config(**overrides) -> dict:
    raw = copy.deepcopy(_HYBRID)
    raw.update(overrides)
    return raw


def _backbone(**overrides) -> dict:
    raw = _config()
    raw["backbone_config"].update(overrides)
    return raw


def test_model_type_resolves_to_monocular_depth():
    support = describe(ModelMetadata(config={"model_type": "dpt"}, model_index={}))
    assert support is not None
    assert support.tasks == ("monocular_depth",)
    assert support.default_task == "monocular_depth"
    assert support.default_precision == "fp32"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["DPTForDepthEstimation"]},
                                     model_index={}))
    assert support is not None
    assert support.default_task == "monocular_depth"


def test_depth_anything_is_not_claimed():
    # Depth Anything carries a DPT neck too, but it is its own model type.
    for config in ({"model_type": "depth_anything"},
                   {"architectures": ["DepthAnythingForDepthEstimation"]}):
        assert describe(ModelMetadata(config=config, model_index={})) is None


def test_hybrid_midas_geometry():
    resolved = config_module.resolve(_config(), _PROCESSOR)
    backbone = resolved["backbone"]
    # SAME padding: 7x7 stride 2 on 384 pads 2 before and 3 after; the pool 0 and 1.
    assert backbone["stem_padding"] == (2, 3)
    assert backbone["pool_padding"] == (0, 1)
    stages = backbone["stages"]
    assert [stage["out_channels"] for stage in stages] == [256, 512, 1024]
    assert [stage["mid_channels"] for stage in stages] == [64, 128, 256]
    assert [stage["stride"] for stage in stages] == [1, 2, 2]
    assert [stage["out_side"] for stage in stages] == [96, 48, 24]
    # Stride-2 3x3 convolutions pad only after; stride-1 ones pad symmetrically.
    assert [stage["conv2_padding"] for stage in stages] == [(1, 1), (0, 1), (0, 1)]
    assert [stage["downsample_padding"] for stage in stages] == [(0, 0), (0, 0), (0, 0)]
    assert resolved["patch_grid"] == 24
    assert resolved["tap_layers"] == [8, 11]
    assert resolved["head_dim"] == 64
    assert resolved["fusion_bias"] is True
    assert resolved["processor"] == {
        "rescale_factor": 0.00392156862745098,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }


def test_vit_taps_follow_layer_order_like_the_reference():
    # The reference keeps encoder layers in order and filters by membership, so a
    # reversed list still feeds layer 8 before layer 11 into the neck.
    resolved = config_module.resolve(_config(backbone_out_indices=[2, 5, 11, 8]), _PROCESSOR)
    assert resolved["tap_layers"] == [8, 11]


def test_a_vit_layer_named_twice_is_refused():
    with pytest.raises(ValueError, match="twice"):
        config_module.resolve(_config(backbone_out_indices=[2, 5, 8, 8]), _PROCESSOR)


def test_a_hybrid_config_without_backbone_config_takes_the_hybrid_defaults():
    raw = _config()
    del raw["backbone_config"]
    resolved = config_module.resolve(raw, _PROCESSOR)
    assert [stage["depth"] for stage in resolved["backbone"]["stages"]] == [3, 4, 9]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_config(is_hybrid=False), "DPT-Hybrid"),
        (_config(architectures=["DPTForSemanticSegmentation"]), "DPTForDepthEstimation"),
        (_config(model_type="depth_anything"), "model_type"),
        (_config(add_projection=True), "add_projection"),
        (_config(use_batch_norm_in_fusion_residual=True), "batch norm"),
        (_config(neck_ignore_stages=[]), "neck_ignore_stages"),
        (_config(reassemble_factors=[1, 1, 2, 0.5]), "reassemble factors"),
        (_config(patch_size=32), "patch grid"),
        (_backbone(layer_type="preactivation"), "bottleneck"),
        (_backbone(global_padding=None), "SAME"),
        (_backbone(model_type="dinov2"), "BiT backbone"),
        (_backbone(out_features=["stage3"]), "three-stage"),
    ],
)
def test_unsupported_configs_are_refused(raw, message):
    with pytest.raises((NotImplementedError, ValueError), match=message):
        config_module.resolve(raw, _PROCESSOR)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"keep_aspect_ratio": True}, "keep_aspect_ratio"),
        ({"resample": 3}, "bilinear"),
        ({"size": {"height": 512, "width": 512}}, "384x384"),
        ({"do_pad": True}, "padded"),
        ({"do_normalize": False}, "do_normalize"),
        ({"image_std": [0.5, 0.0, 0.5]}, "non-zero"),
    ],
)
def test_unsupported_image_processing_is_refused(override, message):
    processor = dict(_PROCESSOR, **override)
    with pytest.raises((NotImplementedError, ValueError), match=message):
        config_module.resolve(_config(), processor)


def test_processor_defaults_are_the_library_defaults():
    # DPTImageProcessor resamples bicubically unless told otherwise, which this
    # family does not reproduce.
    processor = {key: value for key, value in _PROCESSOR.items() if key != "resample"}
    with pytest.raises(NotImplementedError, match="bilinear"):
        config_module.resolve(_config(), processor)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_config(hidden_size=768.0), "hidden_size must be an integer"),
        (_config(num_channels=True), "num_channels must be an integer"),
        (_config(hidden_size=None), "hidden_size must be an integer"),
        (_config(image_size=2**40), "image_size is out of range"),
        (_config(qkv_bias="false"), "qkv_bias must be true or false"),
        (_config(qkv_bias=None), "qkv_bias must be true or false"),
        (_config(is_hybrid=1), "is_hybrid must be true or false"),
        (_config(layer_norm_eps=float("nan")), "layer_norm_eps must be a finite number"),
        (_config(layer_norm_eps="1e-12"), "layer_norm_eps must be a finite number"),
        (_config(backbone_out_indices=[2, 5, 8.0, 11]), "backbone_out_indices must be an integer"),
        (_config(reassemble_factors=[1, 1, 1, "0.5"]), "reassemble_factors must be a finite number"),
        (_config(use_bias_in_fusion_residual="true"), "use_bias_in_fusion_residual"),
        (_config(architectures=[]), "architectures"),
        (_config(backbone="google/bit-50"), "named backbone"),
        (_backbone(depths=[3, 4, 9.0]), "BiT depths must be an integer"),
        (_backbone(num_groups=None), "BiT num_groups must be an integer"),
        (_backbone(width_factor="1"), "BiT width_factor must be a finite number"),
        (_backbone(embedding_dynamic_padding="true"), "embedding_dynamic_padding"),
    ],
)
def test_config_values_keep_their_json_types(raw, message):
    # Each of these would pass a coercing reader (int(768.0), bool("false"), an
    # explicit null taking the default) and build something the checkpoint is not.
    with pytest.raises((NotImplementedError, ValueError), match=message):
        config_module.resolve(raw, _PROCESSOR)


def test_null_is_accepted_where_transformers_accepts_it():
    raw = _config(architectures=None, use_bias_in_fusion_residual=None, backbone=None,
                  backbone_config=None)
    resolved = config_module.resolve(raw, _PROCESSOR)
    assert resolved["fusion_bias"] is True
    assert [stage["depth"] for stage in resolved["backbone"]["stages"]] == [3, 4, 9]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rescale_factor": "0.00392156862745098"}, "rescale_factor must be a finite number"),
        ({"rescale_factor": None}, "rescale_factor must be a finite number"),
        ({"resample": 2.0}, "resample must be an integer"),
        ({"do_resize": "true"}, "do_resize must be true or false"),
        ({"size": {"height": 384.0, "width": 384}}, "size.height must be an integer"),
        ({"size": {"height": 384, "width": 384, "shortest_edge": 384}}, "384x384"),
        ({"image_mean": 0.5}, "image_mean must be a list"),
        # Accepted by a finite-and-non-zero check, but the runtime computes in float32:
        ({"rescale_factor": 1e308}, "keep 255 within float32"),
        ({"image_mean": [0.5, 0.5, 1e39]}, "fit float32"),
        ({"image_std": [0.5, 1e-50, 0.5]}, "non-zero in float32"),
        ({"image_std": [0.5, 1e-40, 0.5]}, "overflow float32"),
    ],
)
def test_image_processor_values_are_checked_as_the_runtime_computes(override, message):
    processor = dict(_PROCESSOR, **override)
    with pytest.raises((NotImplementedError, ValueError), match=message):
        config_module.resolve(_config(), processor)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"model_type": "dpt", "hidden_size": 768, "hidden_size": 1024}', "repeats the key"),
        ('{"model_type": "dpt", "backbone_config": {"depths": [3], "depths": [3, 4, 9]}}',
         "repeats the key"),
        ('{"model_type": "dpt", "layer_norm_eps": NaN}', "non-finite"),
        ('{"model_type": "dpt", "layer_norm_eps": 1e400}', "non-finite"),
        ('["dpt"]', "JSON object"),
    ],
)
def test_json_files_are_read_strictly(tmp_path, text, message):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        config_module._read_json(path)


class _Options:
    """The three E2E selectors, and the paths given to pytest, as its config holds them."""

    def __init__(self, models=(), testcases=(), models_file=None, args=()):
        from pathlib import Path
        from types import SimpleNamespace

        self._values = {"--e2e-model": list(models), "--e2e-testcase": list(testcases),
                        "--e2e-models-file": models_file}
        self.args = list(args)
        self.invocation_params = SimpleNamespace(dir=Path(__file__).resolve().parents[3])

    def getoption(self, name, default=None):
        return self._values.get(name, default)


_FP32 = ["dpt-hybrid-midas-landscape", "dpt-hybrid-midas-portrait"]
_FP16 = ["dpt-hybrid-midas-fp16-landscape", "dpt-hybrid-midas-fp16-portrait"]


@pytest.mark.parametrize(
    ("models", "testcases", "expected"),
    [
        ([], [], (sorted(_FP16 + _FP32), False)),
        (["dpt"], [], (sorted(_FP16 + _FP32), True)),
        (["dpt-hybrid-midas-fp16"], [], (_FP16, True)),
        ([], ["dpt-hybrid-midas-landscape"], (["dpt-hybrid-midas-landscape"], True)),
        (["dpt"], ["dpt-hybrid-midas-portrait, dpt-hybrid-midas-fp16-portrait"],
         (["dpt-hybrid-midas-fp16-portrait", "dpt-hybrid-midas-portrait"], True)),
        # Another family's selection leaves this file empty, so pytest skips it.
        (["wavlm"], [], ([], True)),
        ([], ["wavlm-base-plus-ctc"], ([], True)),
        # A selection of several families cannot tell another family's case from a typo.
        (["dpt", "wavlm"], ["dpt-hybrid-midas-landscape", "wavlm-base-plus-ctc"],
         (["dpt-hybrid-midas-landscape"], True)),
        ([], ["dpt-hybrid-midas-landscape,wavlm-base-plus-ctc"], (["dpt-hybrid-midas-landscape"], True)),
    ],
)
def test_e2e_selection(models, testcases, expected):
    from families.dpt.tests.test_e2e import _selected_cases

    assert _selected_cases(_Options(models, testcases)) == expected


@pytest.mark.parametrize(
    ("models", "testcases"),
    [
        (["dpt"], ["dpt-hybrid-midas-landscap"]),
        (["dpt-hybrid-midas-fp16"], ["dpt-hybrid-midas-landscape"]),
        (["wavlm"], ["dpt-hybrid-midas-landscape"]),
    ],
)
def test_e2e_selection_that_names_dpt_but_matches_nothing_fails(models, testcases):
    from families.dpt.tests.test_e2e import _selected_cases

    with pytest.raises(pytest.UsageError, match="the E2E selection of dpt"):
        _selected_cases(_Options(models, testcases))


@pytest.mark.parametrize(
    ("models", "testcases", "args"),
    [
        # The family alone is selected, so the misspelled case would silently not run.
        (["dpt"], ["dpt-hybrid-midas-landscape,dpt-hybrid-midas-landscap"], []),
        (["dpt-hybrid-midas-fp16"], ["dpt-hybrid-midas-fp16-landscape", "dpt-hybrid-midas-portrait"], []),
        # Only this family's files are collected, the way CI runs them.
        ([], ["dpt-hybrid-midas-landscape,dpt-hybrid-midas-landscap"],
         ["families/dpt/tests/test_e2e.py"]),
        (["dpt", "wavlm"], [], ["families/dpt/tests/test_e2e.py::test_e2e"]),
    ],
)
def test_e2e_selection_of_dpt_alone_must_run_every_requested_case(models, testcases, args):
    from families.dpt.tests.test_e2e import _selected_cases

    with pytest.raises(pytest.UsageError, match="asks for what it cannot run"):
        _selected_cases(_Options(models, testcases, args=args))
    assert _selected_cases(_Options(["dpt"], ["dpt-hybrid-midas-landscape"],
                                    args=["families/dpt/tests/test_e2e.py"])) \
        == (["dpt-hybrid-midas-landscape"], True)
    # Another family's selection still leaves this file empty, whatever is collected.
    assert _selected_cases(_Options(["wavlm"], args=["families/dpt/tests/test_e2e.py"])) == ([], True)


@pytest.mark.parametrize(
    ("models", "testcases"),
    [
        # Several families collected: a selector extending a dpt manifest name is still dpt's.
        ([], ["dpt-hybrid-midas-landscap"]),
        ([], ["dpt-hybrid-midas-landscape,dpt-hybrid-midas-landscap,wavlm-base-plus-ctc"]),
        (["dpt-hybrid-midas-fp1"], []),
        (["dpt-hybrid-midas-fp1", "wavlm"], ["wavlm-base-plus-ctc"]),
    ],
)
def test_e2e_selection_of_a_misspelled_dpt_name_fails_in_any_session(models, testcases):
    from families.dpt.tests.test_e2e import _selected_cases

    with pytest.raises(pytest.UsageError, match="asks for what it cannot run"):
        _selected_cases(_Options(models, testcases))


def test_e2e_models_file_selects_and_fails_the_same_way(tmp_path):
    from families.dpt.tests.test_e2e import _selected_cases

    listing = tmp_path / "models.txt"
    listing.write_text("# comment\ndpt-hybrid-midas-fp16\n\n", encoding="utf-8")
    assert _selected_cases(_Options(models_file=str(listing))) == (_FP16, True)
    with pytest.raises(pytest.UsageError, match="the E2E selection of dpt"):
        _selected_cases(_Options(testcases=["dpt-hybrid-midas-landscape"],
                                 models_file=str(listing)))
    with pytest.raises(pytest.UsageError, match="asks for what it cannot run"):
        _selected_cases(_Options(testcases=["dpt-hybrid-midas-fp16-landscape,dpt-hybrid-midas-fp16-landscap"],
                                 models_file=str(listing)))


def test_e2e_model_dir_override_stands_for_one_checkpoint(monkeypatch):
    from families.dpt.tests import test_e2e as e2e

    monkeypatch.setenv(e2e.MODEL_DIR_OVERRIDE, "/checkpoint")
    e2e._require_one_checkpoint(sorted(e2e.CASES))  # both manifests use one checkpoint
    cases = copy.deepcopy(e2e.CASES)
    path, manifest, case = cases["dpt-hybrid-midas-fp16-landscape"]
    cases["dpt-hybrid-midas-fp16-landscape"] = (path, dict(manifest, hf_revision="0" * 40), case)
    monkeypatch.setattr(e2e, "CASES", cases)
    with pytest.raises(pytest.UsageError, match="stands for one checkpoint"):
        e2e._require_one_checkpoint(sorted(cases))
    monkeypatch.delenv(e2e.MODEL_DIR_OVERRIDE)
    e2e._require_one_checkpoint(sorted(cases))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda case: case.update(min_depth_correlation="0.999999"), "min_depth_correlation"),
        (lambda case: case.update(max_depth_relative_error=True), "max_depth_relative_error"),
        (lambda case: case.update(max_mean_depth_relative_error=float("nan")), "max_mean"),
        (lambda case: case.update(max_depth_relative_error=0), "max_depth_relative_error"),
        (lambda case: case.update(transpose="false"), "transpose must be a boolean"),
        (lambda case: case.pop("premerge"), "premerge must be a boolean"),
        (lambda case: case.update(test_image="../manifests/dpt-hybrid-midas.json"), "test_image"),
    ],
)
def test_e2e_manifests_are_checked(change, message):
    import json
    from pathlib import Path

    from families.dpt.tests import test_e2e as e2e

    path = Path(e2e.MANIFEST_ROOT) / "dpt-hybrid-midas.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    e2e._check_manifest(path, manifest)
    change(manifest["testcases"][0])
    with pytest.raises(ValueError, match=message):
        e2e._check_manifest(path, manifest)


def test_depth_shapes_are_checked_before_any_error_is_computed():
    import numpy as np

    from families.dpt.tests.test_e2e import _depth_metrics

    row = np.linspace(1.0, 2.0, 6)
    # A (1, 6) reference broadcasts against a (4, 6) map without complaint.
    with pytest.raises(AssertionError, match="differs from the reference"):
        _depth_metrics(np.tile(row, (4, 1)), row[None, :])
    values = _depth_metrics(np.tile(row, (4, 1)), np.tile(row, (4, 1)) * 1.01)
    assert values["max_error"] == pytest.approx(0.01 / 1.01)
    with pytest.raises(AssertionError, match="not finite"):
        _depth_metrics(np.full((2, 3), np.nan), np.ones((2, 3)))


def test_bit_weights_must_have_the_shapes_the_configuration_implies():
    from types import SimpleNamespace

    resolved = config_module.resolve(_config(), _PROCESSOR)
    shapes = dict(config_module.bit_weight_shapes(resolved))
    bit = config_module.BIT_PREFIX
    # The stem, and 3 + 4 + 9 bottleneck layers of three convolutions plus one
    # downsample per stage.
    assert len(shapes) == 1 + 3 * (3 + 4 + 9) + 3
    assert shapes[f"{bit}.embedder.convolution.weight"] == (64, 3, 7, 7)
    assert shapes[f"{bit}.encoder.stages.0.layers.0.downsample.conv.weight"] == (256, 64, 1, 1)
    assert shapes[f"{bit}.encoder.stages.1.layers.0.conv1.weight"] == (128, 256, 1, 1)
    assert shapes[f"{bit}.encoder.stages.2.layers.8.conv3.weight"] == (1024, 256, 1, 1)
    checkpoint = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
    config_module.check_weight_shapes(checkpoint, shapes.items())

    # A configuration that names a wider stem than the checkpoint holds is refused,
    # rather than built from the 64-channel weights.
    wider = config_module.resolve(_backbone(embedding_size=128), _PROCESSOR)
    with pytest.raises(ValueError, match=r"embedder\.convolution\.weight has shape \(64, 3, 7, 7\)"):
        config_module.check_weight_shapes(checkpoint, config_module.bit_weight_shapes(wider))
    deeper = config_module.resolve(_backbone(depths=[3, 4, 10]), _PROCESSOR)
    with pytest.raises(KeyError, match=r"stages\.2\.layers\.9"):
        config_module.check_weight_shapes(checkpoint, config_module.bit_weight_shapes(deeper))


def test_an_extreme_bit_depth_is_refused_at_the_first_missing_weight():
    import time
    from types import SimpleNamespace

    shapes = dict(config_module.bit_weight_shapes(config_module.resolve(_config(), _PROCESSOR)))
    checkpoint = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
    deepest = config_module.resolve(_backbone(depths=[3, 4, 2**31 - 1]), _PROCESSOR)
    started = time.monotonic()
    # Checked lazily: the expected names stop at the checkpoint's last layer,
    # instead of first listing two billion layers.
    with pytest.raises(KeyError, match=r"stages\.2\.layers\.9\.conv1"):
        config_module.check_weight_shapes(checkpoint, config_module.bit_weight_shapes(deepest))
    assert time.monotonic() - started < 5.0


@pytest.mark.parametrize("value", [384.5, 384.0, "384", True])
def test_a_requested_image_size_must_be_an_integer(value):
    with pytest.raises(ValueError, match="must be an integer"):
        config_module.check_requested_image_size("image_height", value, 384)


def test_a_requested_image_size_must_be_the_checkpoints():
    config_module.check_requested_image_size("image_width", None, 384)
    config_module.check_requested_image_size("image_width", 384, 384)
    with pytest.raises(NotImplementedError, match="native image_width"):
        config_module.check_requested_image_size("image_width", 512, 384)


def test_the_fp16_engine_input_must_fit_fp16():
    processor = config_module.resolve(_config(), _PROCESSOR)["processor"]
    config_module.check_engine_input(processor, "fp16")  # [-1, 1]
    # Finite in float32, so the configuration resolves; (1 - 0.5) / 1e-6 is past fp16.
    narrow = config_module.resolve(_config(), dict(_PROCESSOR, image_std=[1e-6] * 3))["processor"]
    config_module.check_engine_input(narrow, "fp32")
    with pytest.raises(ValueError, match="past fp16"):
        config_module.check_engine_input(narrow, "fp16")


@pytest.mark.parametrize("eps", [1e100, 1e-100])
def test_layer_norm_eps_must_be_a_positive_float32_value(eps):
    # 1e100 is infinite and 1e-100 zero once TensorRT stores the epsilon as float32.
    with pytest.raises(ValueError, match="positive float32"):
        config_module.resolve(_config(layer_norm_eps=eps), _PROCESSOR)


@pytest.mark.parametrize("factor", [2**-100, 1 / 48, 0.4])
def test_a_tap_reassemble_factor_must_give_a_stride_within_the_grid(factor):
    # 2**-100 would be a stride of 2**100; 1/48 strides past the 24x24 grid; 0.4 is not 1/n.
    with pytest.raises(NotImplementedError, match="n from 2 to 24"):
        config_module.resolve(_config(reassemble_factors=[1, 1, 1, factor]), _PROCESSOR)
    # The first tap's map is resized onto the fused one, so any of its strides works.
    assert config_module.resolve(_config(reassemble_factors=[1, 1, 0.25, 0.5]), _PROCESSOR)


@pytest.mark.parametrize(("factor", "side"), [(1.0, 24), (0.25, 6), (1 / 3, 8)])
def test_the_last_tap_factor_must_give_a_depth_map_of_the_input_size(factor, side):
    # The depth map is 32 times the deepest map's side: 12 for a 384-pixel input. The
    # graph would build 768, 192 or 256 pixels instead.
    with pytest.raises(NotImplementedError, match=re.escape(f"factor {factor} gives {side}")):
        config_module.resolve(_config(reassemble_factors=[1, 1, 1, factor]), _PROCESSOR)


@pytest.mark.parametrize("key", ["size", "image_mean", "image_std"])
def test_a_null_processor_field_takes_the_image_processors_default(key):
    # DPTImageProcessor replaces these three nulls with its defaults; the checkpoint's
    # values are those defaults, so the resolved pipeline is the same.
    expected = config_module.resolve(_config(), _PROCESSOR)["processor"]
    assert config_module.resolve(_config(), dict(_PROCESSOR, **{key: None}))["processor"] == expected


@pytest.mark.parametrize(
    "selectors",
    [
        {"out_features": None, "out_indices": [1, 2, 3]},
        {"out_features": None, "out_indices": [-3, -2, -1]},
        {"out_features": ["stage1", "stage2", "stage3"], "out_indices": [1, 2, 3]},
    ],
)
def test_bit_outputs_may_be_named_by_index(selectors):
    resolved = config_module.resolve(_backbone(**selectors), _PROCESSOR)
    assert [stage["out_channels"] for stage in resolved["backbone"]["stages"]] == [256, 512, 1024]


@pytest.mark.parametrize(
    ("selectors", "error", "message"),
    [
        ({"out_indices": [0, 1, 2]}, ValueError, "name different stages"),
        ({"out_features": None, "out_indices": [1, 2, 2]}, ValueError, "distinct and in order"),
        ({"out_features": None, "out_indices": [1, 2, 3.0]}, ValueError, "must be an integer"),
        ({"out_features": None, "out_indices": [1, 2, 4]}, ValueError, "must index"),
        ({"out_features": ["stage1", "stage3", "stage2"]}, ValueError, "in order"),
        ({"out_features": None, "out_indices": None}, NotImplementedError, "three-stage"),
        ({"out_features": None, "out_indices": [2, 3]}, NotImplementedError, "three-stage"),
    ],
)
def test_bit_output_selectors_are_aligned_like_transformers(selectors, error, message):
    with pytest.raises(error, match=message):
        config_module.resolve(_backbone(**selectors), _PROCESSOR)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda torch: torch.zeros(2, dtype=torch.int64), "not floating point"),
        (lambda torch: torch.zeros(2, dtype=torch.complex64), "not floating point"),
        (lambda torch: torch.tensor([0.0, float("nan")]), "not finite"),
        (lambda torch: torch.tensor([1e300], dtype=torch.float64), "not finite"),
    ],
)
def test_checkpoint_tensors_must_be_finite_floating_point(make, message):
    import torch

    from families.dpt.checkpoint import tensor_arrays

    with pytest.raises(ValueError, match=message):
        tensor_arrays({"weight": make(torch)}, "checkpoint")


def test_a_checkpoint_is_read_as_float32_with_weights_only(tmp_path):
    import numpy as np
    import torch

    from families.dpt import checkpoint

    torch.save({"weight": torch.ones(2, dtype=torch.bfloat16)}, tmp_path / checkpoint.CHECKPOINT)
    calls = []
    load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "load", spy)
        arrays = checkpoint.load_tensors(tmp_path)
        # The loader is restricted to tensors and plain containers, never full pickle.
        assert [call.get("weights_only") for call in calls] == [True]
        assert arrays["weight"].dtype == np.float32 and arrays["weight"].tolist() == [1.0, 1.0]
        torch.save([torch.ones(2)], tmp_path / checkpoint.CHECKPOINT)
        with pytest.raises(ValueError, match="does not hold a state dict"):
            checkpoint.load_tensors(tmp_path)


@pytest.mark.parametrize("version", ["2.9.1", "2.6.0+cu124", "2.10.0.dev20251201"])
def test_a_torch_that_can_run_code_from_a_checkpoint_is_refused_before_loading(tmp_path, version):
    import torch

    from families.dpt import checkpoint

    torch.save({"weight": torch.ones(2)}, tmp_path / checkpoint.CHECKPOINT)
    calls = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "__version__", version)
        patch.setattr(torch, "load", lambda *args, **kwargs: calls.append(kwargs))
        with pytest.raises(RuntimeError, match="torch 2.10.0 or later"):
            checkpoint.load_tensors(tmp_path)
    assert calls == []


def test_vit_embeddings_are_checked_before_the_graph():
    from types import SimpleNamespace

    resolved = config_module.resolve(_config(), _PROCESSOR)
    shapes = dict(config_module.vit_embedding_shapes(resolved))
    assert shapes["dpt.embeddings.position_embeddings"] == (1, 577, 768)
    assert shapes["dpt.embeddings.projection.weight"] == (768, 1024, 1, 1)
    checkpoint = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
    config_module.check_weight_shapes(checkpoint, shapes.items())
    # A rank-one tensor is reported with its shape, not by indexing a missing axis.
    for wrong in ((577,), (1, 1025, 768)):
        checkpoint["dpt.embeddings.position_embeddings"] = SimpleNamespace(shape=wrong)
        with pytest.raises(ValueError, match=re.escape(f"has shape {wrong}")):
            config_module.check_weight_shapes(checkpoint, config_module.vit_embedding_shapes(resolved))


def test_engine_weights_must_fit_their_storage_dtype():
    import numpy as np

    from families.dpt.checkpoint import as_engine_weights

    bias = np.array([0.5, 1e5], dtype=np.float32)
    assert as_engine_weights(bias, np.float32).dtype == np.float32
    # Finite in float32, past fp16's 65504: the fp16 engine would hold an infinity.
    with pytest.raises(ValueError, match="not finite in float16"):
        as_engine_weights(bias, np.float16)
    with pytest.raises(ValueError, match="not finite in float32"):
        as_engine_weights(np.array([np.nan]), np.float32)


def test_the_image_size_stays_within_the_runtimes_input_limit():
    # The runtime refuses more than INT32_MAX / 4 pixels; 32768 x 32768 is past it.
    with pytest.raises(ValueError, match="runtime's input limit"):
        config_module.resolve(_config(image_size=2**15), _PROCESSOR)


def _e2e_item(case_name, name="test_e2e"):
    from types import SimpleNamespace

    return SimpleNamespace(originalname=name, callspec=SimpleNamespace(params={"case_name": case_name}))


def test_a_testcase_only_selection_of_other_families_checks_the_session():
    from types import SimpleNamespace

    from families.dpt.tests import test_e2e as e2e

    calls = []
    metafunc = SimpleNamespace(fixturenames=["case_name"], config=_Options([], ["wavlm-base-plus-ctc"]),
                               parametrize=lambda *args, **kwargs: calls.append(args[1]))
    e2e.pytest_generate_tests(metafunc)
    assert calls == [[e2e.ELSEWHERE]]

    owned = SimpleNamespace(config=metafunc.config, session=SimpleNamespace(
        items=[_e2e_item(e2e.ELSEWHERE), _e2e_item("wavlm-base-plus-ctc")]))
    e2e._check_session_selection(owned)
    # No collected E2E test runs the requested case: the session must not pass.
    unowned = SimpleNamespace(config=metafunc.config, session=SimpleNamespace(
        items=[_e2e_item(e2e.ELSEWHERE), _e2e_item("wavlm-base-plus-ctc", name="test_tokenizer")]))
    with pytest.raises(pytest.fail.Exception, match="wavlm-base-plus-ctc"):
        e2e._check_session_selection(unowned)


def test_a_misspelled_selector_fails_even_when_another_matched():
    from types import SimpleNamespace

    from families.dpt.tests import test_e2e as e2e

    request = SimpleNamespace(
        config=_Options([], ["dpt-hybrid-midas-landscape,landscap"]),
        session=SimpleNamespace(items=[_e2e_item("dpt-hybrid-midas-landscape"),
                                       _e2e_item("wavlm-base-plus-ctc")]))
    with pytest.raises(pytest.fail.Exception, match="landscap"):
        e2e._check_session_selection(request)
    request.config = _Options(["dpt", "wavlm"], ["dpt-hybrid-midas-landscape,wavlm-base-plus-ctc"])
    e2e._check_session_selection(request)


def test_another_familys_model_selection_still_leaves_the_file_empty():
    from types import SimpleNamespace

    from families.dpt.tests import test_e2e as e2e

    calls = []
    metafunc = SimpleNamespace(fixturenames=["case_name"], config=_Options(["wavlm"]),
                               parametrize=lambda *args, **kwargs: calls.append(args[1]))
    e2e.pytest_generate_tests(metafunc)
    assert calls == [[]]


def test_the_placeholder_is_not_a_case_of_the_session():
    from types import SimpleNamespace

    from families.dpt.tests import test_e2e as e2e

    request = SimpleNamespace(config=_Options([], [e2e.ELSEWHERE]),
                              session=SimpleNamespace(items=[_e2e_item(e2e.ELSEWHERE)]))
    with pytest.raises(pytest.fail.Exception, match=e2e.ELSEWHERE):
        e2e._check_session_selection(request)


def test_the_sessions_e2e_selection_holds_on_the_host_too(request):
    """The E2E selectors given to this session, checked as test_e2e.py checks them.

    Without selectors this passes. A host-only run with ``--e2e-testcase`` (of this
    family or of several) fails here, since no collected E2E test runs the case,
    instead of passing without noticing the selection.
    """
    from families.dpt.tests.test_e2e import check_selection

    check_selection(request)


@pytest.mark.parametrize(
    ("models", "testcases", "args"),
    [
        ([], ["wavlm-base-plus-ctc"], []),  # several families' host tests, another family's case
        ([], ["dpt-hybrid-midas-landscape"], ["families/dpt/tests/test_support.py"]),
        (["dpt"], ["dpt-hybrid-midas-landscape"], []),
    ],
)
def test_a_host_only_session_with_testcases_fails(models, testcases, args):
    from types import SimpleNamespace

    from families.dpt.tests.test_e2e import check_selection

    host_items = [_e2e_item("unused", name="test_bit_weights_must_have_the_shapes_the_configuration_implies")]
    request = SimpleNamespace(config=_Options(models, testcases, args=args),
                              session=SimpleNamespace(items=host_items))
    with pytest.raises(pytest.fail.Exception, match="names no E2E case of this session"):
        check_selection(request)
    request.session.items = host_items + [_e2e_item(name) for name in testcases]
    check_selection(request)

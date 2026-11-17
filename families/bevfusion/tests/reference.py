# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute upstream BEV classes without importing unrelated OpenMMLab CUDA ops."""

import ast
import argparse
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SOURCE_REVISION = "326653dc06e0938edf1aae7d01efcd158ba83de5"
CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"]


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg


def _conv(config, *args, **kwargs):
    options = dict(config)
    assert options.pop("type") == "Conv2d"
    return nn.Conv2d(*args, **options, **kwargs)


def _norm(config, channels):
    options = dict(config)
    assert options.pop("type") in {"BN", "BN2d"}
    return "bn", nn.BatchNorm2d(channels, **options)


def _upsample(config, *args, **kwargs):
    options = dict(config)
    assert options.pop("type") == "deconv"
    return nn.ConvTranspose2d(*args, **options, **kwargs)


def _classes(source, relative, names):
    """Only registry/dependency plumbing is replaced; class bodies are upstream."""
    path = Path(source) / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope = dict(
        torch=torch,
        nn=nn,
        F=F,
        np=np,
        warnings=warnings,
        BaseModule=BaseModule,
        build_conv_layer=_conv,
        build_norm_layer=_norm,
        build_upsample_layer=_upsample,
        auto_fp16=lambda: lambda function: function,
    )
    exec(compile(tree, str(path), "exec"), scope)
    return [scope[name] for name in names]


class Reference(nn.Module):
    def __init__(self, source, checkpoint):
        super().__init__()
        (fuser,) = _classes(source, "mmdet3d/models/fusers/conv.py", ["ConvFuser"])
        (backbone,) = _classes(source, "mmdet3d/models/backbones/second.py", ["SECOND"])
        (neck,) = _classes(source, "mmdet3d/models/necks/second.py", ["SECONDFPN"])
        _, head = _classes(
            source,
            "mmdet3d/models/heads/segm/vanilla.py",
            ["BEVGridTransform", "BEVSegmentationHead"],
        )
        self.fuser = fuser(in_channels=[80, 256], out_channels=256)
        self.decoder = nn.ModuleDict(
            dict(
                backbone=backbone(
                    in_channels=256,
                    out_channels=[128, 256],
                    layer_nums=[5, 5],
                    layer_strides=[1, 2],
                ),
                neck=neck(
                    in_channels=[128, 256],
                    out_channels=[256, 256],
                    upsample_strides=[1, 2],
                    use_conv_for_no_stride=True,
                ),
            )
        )
        self.heads = nn.ModuleDict(
            dict(
                map=head(
                    in_channels=512,
                    grid_transform=dict(
                        input_scope=[[-51.2, 51.2, 0.8]] * 2, output_scope=[[-50.0, 50.0, 0.5]] * 2
                    ),
                    classes=CLASSES,
                    loss="focal",
                )
            )
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        state = {
            name: value
            for name, value in state.items()
            if name.startswith(("fuser.", "decoder.", "heads.map."))
        }
        self.load_state_dict(state, strict=True)
        self.eval()

    def forward(self, camera_bev, lidar_bev):
        features = self.fuser([camera_bev, lidar_bev])
        return self.heads["map"](self.decoder["neck"](self.decoder["backbone"](features)))


def prepare(directory):
    """Download the same explicit inputs prepared by CI, before offline proof."""
    directory = Path(directory)
    manifest = json.loads((Path(__file__).parent / "manifests/bevfusion-bev-fp32.json").read_text())
    for item in manifest["external_files"]:
        destination = directory / item["path"]
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        try:
            with urlopen(item["url"], timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare declared BEVFusion checkpoint/reference inputs"
    )
    parser.add_argument("--prepare", type=Path, required=True)
    prepare(parser.parse_args().prepare)

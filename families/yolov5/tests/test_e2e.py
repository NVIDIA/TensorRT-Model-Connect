# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and ultralytics reference proof for YOLOv5."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "yolov5"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "object_detection"
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selection(config) -> set[str]:
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option, default=[]) or []:
            selected.update(value.strip() for value in str(raw).split(",") if value.strip())
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selection(metafunc.config)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, name, manifest["name"]}
    ]
    if not selected:
        names = [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="real YOLOv5 E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected YOLOv5 E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected YOLOv5 E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_YOLOv5_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_YOLOv5_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected YOLOv5 E2E image is missing: {path}"
    return path


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )
    image = _asset(case)
    completed = subprocess.run(
        [
            str(binary),
            "detect",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    actual = json.loads(completed.stdout)

    import copy

    import torch
    from PIL import Image
    from ultralytics.nn.tasks import parse_model
    from families.yolov5.checkpoint import _collect, _install_placeholders

    # The reference must run the archive under test, and must see the same
    # letterboxed pixels the runtime seam produced. It is rebuilt rather than
    # called: a YOLOv5 archive pickles classes from the standalone yolov5
    # repository, so the restored object carries weights but cannot run.
    #
    # The topology comes from the yaml the archive itself stores, expanded by
    # ultralytics' own parser, so this reference does not share the family's
    # stage table. Only the anchor head is written out here, because
    # ultralytics replaced it with an anchor-free one.
    _install_placeholders()
    blob = torch.load(str(model_dir / "yolov5n.pt"), map_location="cpu", weights_only=False)
    archive = blob["model"]
    spec = copy.deepcopy(archive.__dict__["yaml"])
    # The parser has no anchor head to build, so it is dropped and the three
    # 1x1 convolutions it holds are applied by hand below.
    spec["head"].pop()
    stages, _ = parse_model(spec, ch=3, verbose=False)

    weights: dict = {}
    _collect(archive, "", weights)
    body = {
        key[len("model.") :]: value.float()
        for key, value in weights.items()
        if key.startswith("model.")
    }
    # The head sits one past the last stage the parser built.
    head_index = str(len(stages))
    missing, unexpected = stages.load_state_dict(
        {k: v for k, v in body.items() if not k.startswith(f"{head_index}.")}, strict=False
    )
    assert not missing, missing
    assert all(key.endswith("num_batches_tracked") for key in unexpected), unexpected
    for module in stages.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            # YOLOv5 resets every norm to this after construction. The value is
            # pickled with the module but is not part of the state dict.
            module.eps = 1e-3
    stages.eval()

    classes = len(archive.__dict__["names"])
    per_anchor = classes + 5
    anchors = weights[f"model.{head_index}.anchors"].float()
    strides = archive.__dict__["stride"].float()
    head = [body[f"{head_index}.m.{level}.weight"] for level in range(len(strides))]
    biases = [body[f"{head_index}.m.{level}.bias"] for level in range(len(strides))]

    size = 640
    source = Image.open(image).convert("RGB")
    scale = min(size / source.height, size / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - resized.width) // 2
    pad_y = (size - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    with torch.no_grad():
        outputs: list = []
        tensor = torch.from_numpy(pixels.astype(np.float32))
        for stage in stages:
            if stage.f != -1:
                taken = [tensor if index == -1 else outputs[index] for index in stage.f]
                tensor = stage(taken)
            else:
                tensor = stage(tensor)
            outputs.append(tensor)

        levels = []
        for level, source_index in enumerate((17, 20, 23)):
            raw = torch.nn.functional.conv2d(outputs[source_index], head[level], biases[level])
            rows, columns = raw.shape[-2], raw.shape[-1]
            values = raw.view(1, len(anchors[level]), per_anchor, rows, columns)
            values = values.permute(0, 1, 3, 4, 2).sigmoid()
            grid_y, grid_x = torch.meshgrid(
                torch.arange(rows), torch.arange(columns), indexing="ij"
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).float()
            centre = (values[..., 0:2] * 2 - 0.5 + grid) * strides[level]
            extent = (
                (values[..., 2:4] * 2) ** 2 * anchors[level].view(1, -1, 1, 1, 2) * strides[level]
            )
            levels.append(
                torch.cat((centre, extent, values[..., 4:]), dim=-1).reshape(1, -1, per_anchor)
            )
        raw = torch.cat(levels, dim=1)
    # Suppression is done here rather than with ultralytics' helper: that one
    # reads the anchor-free layout of the later generations, where there is no
    # objectness and the channels come before the anchors. Handing it a YOLOv5
    # tensor is silently misread rather than rejected.
    import torchvision

    predictions = raw[0]
    confidence = predictions[:, 4:5] * predictions[:, 5:]
    best, labels = confidence.max(dim=1)
    survivors = best > 0.25
    centre, extent = predictions[survivors, 0:2], predictions[survivors, 2:4]
    corners = torch.cat((centre - extent / 2, centre + extent / 2), dim=1)
    order = torchvision.ops.batched_nms(corners, best[survivors], labels[survivors], 0.7)[:300]
    kept = torch.cat(
        (
            corners[order],
            best[survivors][order, None],
            labels[survivors][order, None].float(),
        ),
        dim=1,
    ).numpy()

    assert len(kept), "the reference found nothing above the threshold to compare against"
    assert len(actual["scores"]) == len(kept), (
        f"engine returned {len(actual['scores'])} detections, reference {len(kept)}"
    )
    for index, row in enumerate(kept):
        x0, y0, x1, y1, score, label = row
        assert int(actual["classes"][index]) == int(label)
        assert abs(float(actual["scores"][index]) - float(score)) < 5e-3
        # The engine reports source-image pixels, so the reference box has to
        # come back out of the letterbox before they can be compared.
        expected = (
            (float(x0) - pad_x) / scale,
            (float(y0) - pad_y) / scale,
            (float(x1) - pad_x) / scale,
            (float(y1) - pad_y) / scale,
        )
        for axis, want in enumerate(expected):
            # The CLI writes the boxes as one flat array, four per detection.
            assert abs(float(actual["boxes"][index * 4 + axis]) - want) < 1.0

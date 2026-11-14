# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for OneFormer semantic segmentation.

The oracle is the official Transformers pipeline at the pinned revision: the
fast ``OneFormerImageProcessor`` (the class ``OneFormerProcessor.from_pretrained``
loads in Transformers 5.2.0), ``OneFormerForUniversalSegmentation`` in FP32 with
TF32 disabled, and ``post_process_semantic_segmentation`` to the original image
size. The processor's class metadata (training annotations only) is rebuilt
from the checkpoint's own ``preprocessor_config.json`` instead of being fetched
from a second Hub repository.

Both sides read the same lossless PNG. The native label map must agree with
the reference on at least the precision's minimum share of the pixels
(``ORACLE``), and every pixel where it differs must be a near tie of the
reference: the reference's own score for the native label is within the
precision's margin of its winning score. The public C (float32 pixels) and C++
(uint8 pixels) SDK consumers must reproduce the CLI's label map exactly.

The same selection also checks the family graph stage by stage: an engine with
the backbone, pixel-decoder, mask-feature and logit tensors as extra outputs
must reproduce the reference's tensors on the processor's own pixel values.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "oneformer"
TASKS = frozenset({"image_to_semantic_segmentation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
# Oracle per engine precision: (minimum label agreement, largest reference
# score margin of a differing pixel). The FP32 engine's stride-4 score maps
# match the reference's to 1.6e-5 (scores span 0 to about 1.4), so a pixel may
# differ only where the reference is within 1e-3 of a tie.
ORACLE = {"fp32": (0.9999, 1e-3)}
# The files the family and the reference read; the repository also carries the
# original Detectron2 checkpoint (.pth), which neither side uses.
CHECKPOINT_FILES = [
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            allow_patterns=CHECKPOINT_FILES,
            local_files_only=True,
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    return env


def _native(binary: Path, runtime_root: Path, bundle: Path, image: Path) -> dict:
    completed = subprocess.run(
        [
            str(binary),
            "segment",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(runtime_root),
        timeout=1800,
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert len(payloads) == 1, (
        f"native segment must print one JSON result: {completed.stdout[-1000:]}"
    )
    return payloads[0]


@functools.lru_cache(maxsize=1)
def _official_pipeline(model_dir: str, info_path: str):
    import torch
    from transformers import AutoTokenizer, OneFormerForUniversalSegmentation, OneFormerProcessor
    from transformers.models.oneformer.image_processing_oneformer_fast import (
        OneFormerImageProcessorFast,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    metadata = json.loads(
        (Path(model_dir) / "preprocessor_config.json").read_text(encoding="utf-8")
    )["metadata"]
    class_info = {
        str(index): {"name": name, "isthing": index in metadata["thing_ids"]}
        for index, name in enumerate(metadata["class_names"])
    }
    Path(info_path).write_text(json.dumps(class_info), encoding="utf-8")
    image_processor = OneFormerImageProcessorFast.from_pretrained(
        model_dir, repo_path=None, class_info_file=info_path
    )
    processor = OneFormerProcessor(
        image_processor=image_processor, tokenizer=AutoTokenizer.from_pretrained(model_dir)
    )
    model = OneFormerForUniversalSegmentation.from_pretrained(model_dir).eval().cuda()
    assert model.dtype == torch.float32
    return processor, model


def _official(model_dir: Path, image_path: Path, info_path: Path):
    """Reference labels and the reference's resized score maps (classes, H, W)."""
    from types import SimpleNamespace

    import torch
    from PIL import Image
    from torchvision.transforms.v2 import functional as tvF

    processor, model = _official_pipeline(str(model_dir), str(info_path))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"].cuda(), task_inputs=inputs["task_inputs"].cuda()
        )
    cpu = SimpleNamespace(
        class_queries_logits=outputs.class_queries_logits.float().cpu(),
        masks_queries_logits=outputs.masks_queries_logits.float().cpu(),
    )
    labels = processor.post_process_semantic_segmentation(
        cpu, target_sizes=[(image.height, image.width)]
    )[0]
    segmentation = torch.einsum(
        "bqc, bqhw -> bchw",
        cpu.class_queries_logits.softmax(dim=-1)[..., :-1],
        cpu.masks_queries_logits.sigmoid(),
    )
    resized = tvF.resize(
        segmentation, size=[image.height, image.width], interpolation=tvF.InterpolationMode.BILINEAR
    )[0]
    return labels.numpy(), resized.numpy(), model.config.id2label


def _assert_parity(native: dict, reference, precision: str) -> dict:
    import numpy as np

    min_agreement, near_tie = ORACLE[precision]
    expected, scores, id2label = reference
    height, width = expected.shape
    assert (native["height"], native["width"]) == (height, width)
    assert native["class_ids"] == list(range(len(id2label)))
    assert native["class_names"] == [id2label[index] for index in range(len(id2label))]
    assert (
        native["class_scores"] == [] and native["score_height"] == 0 and native["score_width"] == 0
    )
    labels = np.asarray(native["mask"], dtype=np.int64).reshape(height, width)
    differ = labels != expected
    agreement = 1.0 - float(differ.mean())
    rows, cols = np.nonzero(differ)
    margins = scores[expected[rows, cols], rows, cols] - scores[labels[rows, cols], rows, cols]
    worst = float(margins.max()) if margins.size else 0.0
    assert agreement >= min_agreement, f"label agreement {agreement:.6f} < {min_agreement}"
    assert worst <= near_tie, (
        f"a differing pixel is not a reference near tie: margin {worst:.4g} > {near_tie}"
    )
    return {"agreement": agreement, "differing_pixels": int(differ.sum()), "worst_margin": worst}


def _digest(labels) -> str:
    import numpy as np

    value = 1469598103934665603
    for byte in np.asarray(labels, dtype="<i4").tobytes():
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _sdk(runtime_root: Path, bundle: Path, image_path: Path, tmp_path: Path) -> list[dict]:
    """The public C (float32 pixels) and C++ (uint8 pixels) consumers on the CLI's decoded pixels."""
    import numpy as np
    from PIL import Image

    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    with Image.open(image_path) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    floats, bytes_ = tmp_path / "sdk.rgb.f32", tmp_path / "sdk.rgb.u8"
    (pixels.astype(np.float32) / np.float32(255.0)).tofile(floats)
    pixels.tofile(bytes_)
    results = []
    for language, raw in (("c", floats), ("cpp", bytes_)):
        consumer = native_build / f"test_oneformer_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [
                str(consumer),
                str(bundle),
                str(runtime_root),
                str(raw),
                str(pixels.shape[0]),
                str(pixels.shape[1]),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_environment(runtime_root),
            timeout=1800,
        )
        payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(payloads) == 1, f"the {language} consumer must print one result"
        results.append(json.loads(payloads[0]))
    return results


def _variants(image_path: Path, tmp_path: Path) -> dict[str, Path]:
    """Lossless inputs covering each resize path of the processor and the post-processing.

    fixture 382 x 640 and portrait 640 x 382 are upscaled to a 1024 shortest edge;
    downscaled 1146 x 1920 is downscaled to 1024 x 1715; frame 1024 x 2048 (the
    Cityscapes frame size) is not resized at all; wide 220 x 640 hits the 2048
    longest-edge cap. The label maps are resized back to each input size.
    """
    from PIL import Image

    with Image.open(image_path) as opened:
        base = opened.convert("RGB")
    images = {
        "fixture": base,
        "portrait": base.transpose(Image.Transpose.ROTATE_90),
        "downscaled": base.resize((base.width * 3, base.height * 3), Image.Resampling.BICUBIC),
        "frame": base.resize((2048, 1024), Image.Resampling.BICUBIC),
        "wide": base.crop((0, 120, base.width, 340)),
    }
    paths = {}
    for name, image in images.items():
        paths[name] = tmp_path / f"{name}.png"
        image.save(paths[name])
    return paths


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    """Native labels versus the official pipeline on five geometries, plus the C and C++ SDK."""
    import numpy as np

    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    images = _variants(TEST_ROOT / case["test_image"], tmp_path)
    report = {}
    for name, image in images.items():
        native = _native(binary, runtime_root, bundle, image)
        report[name] = _assert_parity(
            native, _official(model_dir, image, tmp_path / "class_info.json"), manifest["precision"]
        )
        if name == "fixture":
            digest = _digest(native["mask"])
            counts = np.bincount(
                np.asarray(native["mask"]), minlength=len(native["class_ids"])
            ).tolist()
            for sdk in _sdk(runtime_root, bundle, image, tmp_path):
                assert sdk["task"] == "image_to_semantic_segmentation" and not sdk["has_scores"]
                assert (sdk["height"], sdk["width"]) == (native["height"], native["width"])
                assert sdk["class_names"] == native["class_names"]
                assert sdk["counts"] == counts and sdk["fnv1a64"] == digest, (
                    "SDK labels differ from the CLI"
                )
    print(json.dumps({"case": case_name, "parity": report}, sort_keys=True))


# Engine stages compared with the reference's tensors on the processor's own
# pixel values. Each must match within GRAPH_RELATIVE of the reference's
# largest magnitude (the B300 fp32 engine stays below 2.2e-4, at backbone stage
# 4) and with cosine similarity of at least 1 - GRAPH_COSINE_GAP (worst 5.5e-11).
GRAPH_STAGES = (
    "backbone_stage1",
    "backbone_stage2",
    "backbone_stage3",
    "backbone_stage4",
    "pixel_decoder_level0",
    "pixel_decoder_level1",
    "pixel_decoder_level2",
    "mask_features",
    "class_queries_logits",
    "masks_queries_logits",
)
GRAPH_RELATIVE = 2e-3
GRAPH_COSINE_GAP = 1e-8


def _engine_outputs(plan: bytes, pixels) -> dict:
    import tensorrt as trt
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    pixels = pixels.cuda().contiguous()
    assert context.set_input_shape("pixel_values", tuple(pixels.shape))
    context.set_tensor_address("pixel_values", pixels.data_ptr())
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def test_engine_stages_match_the_reference(case_name: str, tmp_path: Path) -> None:
    """The family graph, stage by stage, against OneFormerForUniversalSegmentation in fp32.

    Runs with the E2E selection but is not an E2E case of its own: CI's E2E
    evidence accounting covers ``test_official_checkpoint_e2e``; a failure here
    fails the selected run through pytest's exit status.
    """
    import torch
    from PIL import Image

    from families.oneformer.config import load_config
    from families.oneformer.model import OUTPUT, build_engine, task_token_ids
    from families.oneformer.weights import _open_torch_checkpoint

    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} graph check requires CUDA"
    processor, model = _official_pipeline(str(model_dir), str(tmp_path / "class_info.json"))
    with Image.open(TEST_ROOT / case["test_image"]) as opened:
        inputs = processor(
            images=opened.convert("RGB"), task_inputs=["semantic"], return_tensors="pt"
        )
    config = load_config(model_dir)
    task_ids = task_token_ids(model_dir, config.task_seq_len)
    assert inputs["task_inputs"][0].tolist() == task_ids, (
        "build-time task ids differ from the processor's"
    )

    captured = {}
    level_module = model.model.pixel_level_module
    hooks = [
        level_module.encoder.register_forward_hook(
            lambda module, args, output: captured.__setitem__("backbone", output.feature_maps)
        ),
        level_module.decoder.register_forward_hook(
            lambda module, args, output: captured.__setitem__("decoder", output)
        ),
    ]
    try:
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"].cuda(), task_inputs=inputs["task_inputs"].cuda()
            )
    finally:
        for hook in hooks:
            hook.remove()
    reference = {f"backbone_stage{i + 1}": t for i, t in enumerate(captured["backbone"])}
    reference.update(
        {
            f"pixel_decoder_level{i}": t
            for i, t in enumerate(captured["decoder"].multi_scale_features)
        }
    )
    reference["mask_features"] = captured["decoder"].mask_features
    reference["class_queries_logits"] = outputs.class_queries_logits
    reference["masks_queries_logits"] = outputs.masks_queries_logits
    reference[OUTPUT] = torch.einsum(
        "bqc, bqhw -> bchw",
        outputs.class_queries_logits.softmax(dim=-1)[..., :-1],
        outputs.masks_queries_logits.sigmoid(),
    )

    plan = build_engine(
        config,
        _open_torch_checkpoint(model_dir),
        task_ids,
        max_height=int(manifest["image_height"]),
        max_width=int(manifest["image_width"]),
        debug=GRAPH_STAGES,
    )
    native = _engine_outputs(plan, inputs["pixel_values"])
    report = {}
    for name, expected in reference.items():
        actual = native[name]
        assert tuple(actual.shape) == tuple(expected.shape), name
        assert bool(torch.isfinite(actual).all()), f"{name} is not finite"
        difference = (actual.double() - expected.double()).abs().max().item()
        scale = expected.double().abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), expected.double().flatten(), dim=0
        ).item()
        report[name] = {"max_abs": difference, "relative": difference / scale, "cosine": cosine}
        assert difference <= GRAPH_RELATIVE * scale, (
            f"{name}: max |diff| {difference:.3g} vs {scale:.3g}"
        )
        assert cosine >= 1.0 - GRAPH_COSINE_GAP, f"{name}: cosine {cosine:.12f}"
    print(json.dumps({"case": case_name, "stages": report}, sort_keys=True))

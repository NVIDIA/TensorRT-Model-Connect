# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native caption tokens and FP32 engine tensors against the upstream model.

Gates are fixed before the first GPU run: exact greedy continuation tokens,
vision atol=5e-4/rtol=1e-4, decoder logits atol=1e-3/rtol=1e-4. The decoder
is also checked with upstream image features to isolate it from the encoder.
"""

import json
import os
from pathlib import Path
import struct
import subprocess
import traceback

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/blip-image-captioning-base.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


def input_image():
    reference = os.environ.get("TRTMC_BLIP_REFERENCE_IMAGE")
    if reference:
        reference = Path(reference)
    else:
        asset_dir = os.environ.get("TRTMC_BLIP_MODEL_DIR")
        if not asset_dir:
            raise FileNotFoundError(
                "BLIP E2E requires the pinned image: set TRTMC_BLIP_REFERENCE_IMAGE "
                "or stage the manifest external_files in TRTMC_BLIP_MODEL_DIR"
            )
        reference = Path(asset_dir) / MANIFEST["external_files"][0]["path"]
    if not reference.is_file():
        raise FileNotFoundError(f"BLIP reference image was not staged: {reference}")
    image = Path(os.environ.get("TRTMC_BLIP_IMAGE", str(reference)))
    assert image.read_bytes() == reference.read_bytes(), "BLIP input must match the pinned image"
    return image


def checkpoint_dir():
    explicit = os.environ.get("TRTMC_BLIP_MODEL_DIR")
    if explicit:
        directory = Path(explicit)
        if (directory / "config.json").is_file():
            return directory
        # CI uses MODEL_DIR for external_files, while Hub snapshots stay in
        # the separately staged cache. An image-only directory is not a model.
        if not (directory / MANIFEST["external_files"][0]["path"]).is_file():
            raise FileNotFoundError(
                f"BLIP directory contains neither config.json nor the staged image: {directory}"
            )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    )


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    config = metafunc.config
    models = {v.strip() for raw in config.getoption("--e2e-model") for v in raw.split(",")}
    path = config.getoption("--e2e-models-file")
    if path:
        models.update(
            v.strip()
            for v in Path(path).read_text().splitlines()
            if v.strip() and not v.lstrip().startswith("#")
        )
    cases = {v.strip() for raw in config.getoption("--e2e-testcase") for v in raw.split(",")}
    enabled = bool(models or cases)
    names = [
        name
        for name in CASES
        if (not models or models.intersection({"blip", MANIFEST["name"], name}))
        and (not cases or name in cases)
    ]
    metafunc.parametrize(
        "case_name",
        [
            name
            if enabled
            else pytest.param(
                name, marks=pytest.mark.skip(reason="select the blip E2E family explicitly")
            )
            for name in names
        ],
        scope="module",
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, request):
    image_path = input_image()
    model_dir = checkpoint_dir()
    import torch
    from transformers import BlipConfig, BlipForConditionalGeneration, BlipProcessor

    assert torch.cuda.is_available(), "selected blip E2E requires CUDA"
    binary, runtime = Path(os.environ["TRTMC_BINARY"]), Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert binary.is_file() and (runtime / "libtrtmc_model_blip.so").is_file()
    work = tmp_path_factory.mktemp("blip-e2e")
    bundle = work / MANIFEST["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family="blip",
            task=MANIFEST["task"],
            precision=MANIFEST["precision"],
            max_sequence_length=MANIFEST["max_sequence_length"],
            tensor_parallel_size=MANIFEST["tensor_parallel_size"],
        )
    )
    # This legacy state dict stores both decoder parameter pairs explicitly.
    # Load every parameter strictly instead of inferring ties from serialization.
    reference_config = BlipConfig.from_pretrained(model_dir, local_files_only=True)
    reference_config.tie_word_embeddings = False
    reference_config.text_config.tie_word_embeddings = False
    model = BlipForConditionalGeneration(reference_config).float()
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # Modern Transformers regenerates this nonpersistent buffer. Verify the
    # legacy table before removing it from the learned-parameter state dict.
    position_ids = state.pop("text_decoder.bert.embeddings.position_ids")
    assert torch.equal(position_ids, model.text_decoder.bert.embeddings.position_ids), (
        "BLIP checkpoint position_ids differs from the reference position table"
    )
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().cuda()
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(previous_precision))
    request.addfinalizer(lambda: setattr(torch.backends.cudnn, "allow_tf32", previous_cudnn_tf32))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    processor = BlipProcessor.from_pretrained(model_dir, use_fast=False)
    (work / "tokenizer.json").write_bytes((model_dir / "tokenizer.json").read_bytes())
    explicit = os.environ.get("TRTMC_BLIP_MODEL_DIR")
    revision = (
        None if explicit and (Path(explicit) / "config.json").is_file() else MANIFEST["hf_revision"]
    )
    return bundle, binary, runtime, model, processor, image_path, model_dir, revision


def section(bundle, name):
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"BUNDLE\x01\x00"
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        info = header["sections"][name]
        stream.seek(16 + length + info["offset"])
        return stream.read(info["length"])


def engine_run(bundle, stage, values):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(section(bundle, stage + ".plan"))
    assert engine is not None
    context = engine.create_execution_context()
    inputs = {}
    for name, value in values.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        inputs[name] = tensor
        if -1 in engine.get_tensor_shape(name):
            assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    output = "image_embeds" if stage == "vision" else "logits"
    result = torch.empty(
        tuple(context.get_tensor_shape(output)), dtype=torch.float32, device="cuda"
    )
    assert context.set_tensor_address(output, result.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return result.cpu().numpy()


@pytest.fixture(scope="module")
def case_inputs(case_name, artifacts, tmp_path_factory):
    import torch
    from PIL import Image, ImageOps

    bundle, binary, runtime, model, processor, image_path = artifacts[:6]
    case = CASES[case_name]
    image = Image.open(image_path).convert("RGB")
    if case["mirror"]:
        image = ImageOps.mirror(image)
    native_image = tmp_path_factory.mktemp("blip-" + case_name) / "input.png"
    image.save(native_image)
    inputs = processor(images=image, text=case["prompt"] or None, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        expected = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1)[0]
        vision = model.vision_model(inputs["pixel_values"]).last_hidden_state
    prefix_length = inputs["input_ids"].shape[1] - 1 if "input_ids" in inputs else 1
    continuation = expected[prefix_length:].cpu().tolist()
    return native_image, inputs, expected, vision, continuation


def _check_caption(case_name, artifacts, case_inputs, tmp_path):
    bundle, binary, runtime, _model, processor, _image_path = artifacts[:6]
    native_image, inputs, _expected, _vision, continuation = case_inputs
    case = CASES[case_name]
    evidence_image = tmp_path / "input.png"
    evidence_image.write_bytes(native_image.read_bytes())
    record_evidence(
        "inputs",
        {
            "image": evidence_image,
            "prompt": case["prompt"],
            "mirror": case["mirror"],
            "task": MANIFEST["task"],
        },
    )
    record_evidence(
        "thresholds",
        {
            "caption_tokens": "exact",
            "caption_text": "exact",
            "pixels": {"atol": 1e-6, "rtol": 0},
            "vision": {"atol": 5e-4, "rtol": 1e-4},
            "logits": {"atol": 1e-3, "rtol": 1e-4},
        },
    )
    reference_evidence = {
        "token_ids": continuation,
        "text": processor.decode(continuation, skip_special_tokens=True),
        "oracle": "transformers.BlipForConditionalGeneration",
        "hf_revision": artifacts[7],
    }
    record_evidence("reference", reference_evidence)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime),
        "--image",
        str(native_image),
        "--prompt",
        case["prompt"],
        "--set",
        "max_new_tokens=32",
    ]
    native = subprocess.run(
        command, env=env, check=True, text=True, capture_output=True, timeout=180
    )
    native_payload = json.loads(native.stdout)
    record_evidence("commands", {"argv": command})
    record_evidence("native", native_payload)
    expected_tokens = continuation
    expected_text = reference_evidence["text"]
    pair = {}
    for role, value in (("native", native_payload), ("reference", reference_evidence)):
        path = tmp_path / f"caption-{role}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        pair[role] = path
    record_evidence(
        "reference_comparison",
        {
            "label": "Greedy caption continuation",
            "scope": "independent_reference",
            "enforced": True,
            **pair,
            "checks": [
                {
                    "name": "token_mismatch",
                    "scope": "independent_reference",
                    "actual": int(native_payload["token_ids"] != expected_tokens),
                    "operator": "==",
                    "expected": 0,
                    "passed": native_payload["token_ids"] == expected_tokens,
                },
                {
                    "name": "text_mismatch",
                    "scope": "independent_reference",
                    "actual": int(native_payload["text"] != expected_text),
                    "operator": "==",
                    "expected": 0,
                    "passed": native_payload["text"] == expected_text,
                },
            ],
        },
    )
    assert native_payload["token_ids"] == expected_tokens
    assert native_payload["text"] == expected_text

    helper = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"]) / "test_blip_native"
    assert helper.is_file(), "selected blip E2E requires test_blip_native"
    pixel_dump = tmp_path / "native-pixels.bin"
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(section(bundle, "tokenizer.json"))
    probe = subprocess.run(
        [str(helper), "preprocess", str(tokenizer_file), str(pixel_dump), str(native_image), ""],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    native_pixels = np.fromfile(pixel_dump, np.float32).reshape(1, 3, 384, 384)
    native_evidence = {**native_payload, "pixels": native_pixels}
    reference_evidence["pixels"] = inputs["pixel_values"].cpu().numpy()
    record_evidence("native", native_evidence)
    record_evidence("reference", reference_evidence)
    np.testing.assert_allclose(
        native_pixels,
        inputs["pixel_values"].cpu().numpy(),
        atol=1e-6,
        rtol=0,
    )
    for row in json.loads(probe.stdout):
        assert row["ids"] == processor.tokenizer.encode(row["text"])
        assert row["decoded"] == processor.tokenizer.decode(row["ids"], skip_special_tokens=True)
    repeated = subprocess.run(
        [str(helper), "inference", str(bundle), str(runtime), str(native_image), case["prompt"]],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    repeat_payload = json.loads(repeated.stdout)
    assert repeat_payload["token_ids"] == continuation
    assert repeat_payload["repeated_request"] == "passed"
    assert repeat_payload["invalid_requests"] == 4
    native_evidence.update(request_reuse=repeat_payload, tokenizer=json.loads(probe.stdout))
    print(json.dumps({"case": case_name, "tokens": continuation, "text": native_payload["text"]}))


def _check_engine_tensors(case_name, artifacts, case_inputs, tmp_path):
    import torch

    bundle, _binary, _runtime, model, _processor, _image_path = artifacts[:6]
    _native_image, inputs, expected, vision, _continuation = case_inputs
    actual_vision = engine_run(
        bundle, "vision", {"pixel_values": inputs["pixel_values"].cpu().numpy()}
    )
    expected_vision = vision[0].cpu().numpy()
    # Teacher forcing covers a longer prefix and cross-attention, not just BOS.
    teacher = expected[: min(len(expected) - 1, 12)].unsqueeze(0)
    with torch.inference_mode():
        expected_logits = (
            model.text_decoder(input_ids=teacher, encoder_hidden_states=vision, use_cache=False)
            .logits[:, -1]
            .cpu()
            .numpy()
        )
    length = teacher.shape[1]
    mask = np.triu(np.full((1, length, length), -np.finfo(np.float32).max, dtype=np.float32), 1)
    decoder_inputs = {
        "input_ids": teacher[0].cpu().numpy().astype(np.int32),
        "position_ids": np.arange(length, dtype=np.int32),
        "causal_mask": mask,
        "last_index": np.array([length - 1], np.int32),
    }
    native_evidence = {"vision": actual_vision}
    reference_evidence = {
        "vision": expected_vision,
        "logits": expected_logits,
        "oracle": "transformers.BlipForConditionalGeneration",
        "hf_revision": artifacts[7],
    }
    record_evidence("reference", reference_evidence)
    logit_errors = []
    failures = []
    for embeddings in (expected_vision, actual_vision):
        actual_logits = engine_run(
            bundle, "decoder", {**decoder_inputs, "image_embeds": embeddings}
        )
        native_evidence["logits"] = actual_logits
        native_evidence["decoder_feature_source"] = (
            "reference" if embeddings is expected_vision else "native"
        )
        record_evidence("native", native_evidence)
        source = native_evidence["decoder_feature_source"]
        _record_array_comparison(
            tmp_path, "decoder-" + source, actual_logits, expected_logits, 1e-3, 1e-4
        )
        logit_errors.append(float(np.max(np.abs(actual_logits - expected_logits))))
        try:
            with evidence_stage("compare"):
                np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-3, rtol=1e-4)
        except AssertionError as error:
            failures.append(f"decoder with {source} features: {error}")
    print(
        json.dumps(
            {
                "case": case_name,
                "vision_maxabs": float(np.max(np.abs(actual_vision - expected_vision))),
                "logit_maxabs": float(np.max(np.abs(actual_logits - expected_logits))),
            }
        )
    )
    # Check the encoder after the decoder so its failure does not conceal the
    # independent decoder comparison. Both original numerical gates apply.
    _record_array_comparison(tmp_path, "vision", actual_vision, expected_vision, 5e-4, 1e-4)
    try:
        with evidence_stage("compare"):
            np.testing.assert_allclose(actual_vision, expected_vision, atol=5e-4, rtol=1e-4)
    except AssertionError as error:
        failures.append(f"vision features: {error}")
    record_evidence(
        "metrics",
        {
            "vision_maxabs": float(np.max(np.abs(actual_vision - expected_vision))),
            "decoder_with_reference_features_maxabs": logit_errors[0],
            "decoder_with_native_features_maxabs": logit_errors[1],
        },
    )

    if failures:
        raise AssertionError("\n\n".join(failures))


def _record_array_comparison(tmp_path, label, native, reference, atol, rtol):
    pair = {}
    for role, values in (("native", native), ("reference", reference)):
        path = tmp_path / f"{label}-{role}.npy"
        np.save(path, values, allow_pickle=False)
        pair[role] = path
    mismatches = int(
        np.count_nonzero(~np.isclose(native, reference, atol=atol, rtol=rtol, equal_nan=True))
    )
    record_evidence(
        "reference_comparison",
        {
            "label": label,
            "scope": "independent_reference",
            "enforced": True,
            "atol": atol,
            "rtol": rtol,
            **pair,
            "checks": [
                {
                    "name": "values_outside_tolerance",
                    "scope": "independent_reference",
                    "actual": mismatches,
                    "operator": "==",
                    "expected": 0,
                    "passed": mismatches == 0,
                }
            ],
        },
    )


def test_official_checkpoint_e2e(case_name, artifacts, case_inputs, tmp_path):
    record_evidence(
        "checkpoint",
        {
            "hf_id": MANIFEST["hf_id"],
            "expected_hf_revision": MANIFEST["hf_revision"],
            "hf_revision": artifacts[7],
            "model_dir": str(artifacts[6]),
            "precision": "fp32",
            "revision_status": "unverified" if artifacts[7] is None else "pinned_cache_lookup",
        },
    )
    failures = []
    for label, check in (("caption", _check_caption), ("tensors", _check_engine_tensors)):
        try:
            with evidence_stage(label):
                check(case_name, artifacts, case_inputs, tmp_path)
        except Exception as error:
            # Defer reporting until both independent gates have run. Their
            # original assertions and numerical criteria remain authoritative.
            record_evidence(
                label + "_failure",
                {"type": type(error).__name__, "traceback": traceback.format_exc()},
            )
            failures.append((label, error))
    if failures:
        raise AssertionError(
            "\n\n".join(f"{label}: {error}" for label, error in failures)
        ) from failures[0][1]

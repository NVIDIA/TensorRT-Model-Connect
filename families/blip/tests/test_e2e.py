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

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/blip-image-captioning-base.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


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
def artifacts(tmp_path_factory):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    assert torch.cuda.is_available(), "selected blip E2E requires CUDA"
    model_dir = os.environ.get("TRTMC_BLIP_MODEL_DIR")
    if not model_dir:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    model_dir = Path(model_dir)
    binary, runtime = Path(os.environ["TRTMC_BINARY"]), Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert binary.is_file() and (runtime / "libtrtmc_model_blip.so").is_file()
    work = tmp_path_factory.mktemp("blip-e2e")
    reference_image = os.environ.get("TRTMC_BLIP_REFERENCE_IMAGE")
    if reference_image:
        reference_image = Path(reference_image)
    else:
        from urllib.request import urlretrieve

        reference_image = work / "reference-demo.jpg"
        urlretrieve(MANIFEST["test_image_url"], reference_image)
    image_path = Path(os.environ.get("TRTMC_BLIP_IMAGE", str(reference_image)))
    assert image_path.read_bytes() == reference_image.read_bytes()
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
    model = (
        BlipForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval().cuda()
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = BlipProcessor.from_pretrained(model_dir, use_fast=False)
    (work / "tokenizer.json").write_bytes((model_dir / "tokenizer.json").read_bytes())
    return bundle, binary, runtime, model, processor, image_path


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

    bundle, binary, runtime, model, processor, image_path = artifacts
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


def test_official_checkpoint_e2e(case_name, artifacts, case_inputs, tmp_path):
    bundle, binary, runtime, _model, processor, _image_path = artifacts
    native_image, inputs, _expected, _vision, continuation = case_inputs
    case = CASES[case_name]
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
    payload = json.loads(native.stdout)
    assert payload["token_ids"] == continuation
    assert payload["text"] == processor.decode(continuation, skip_special_tokens=True)

    helper = runtime / "test_blip_native"
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
    np.testing.assert_allclose(
        np.fromfile(pixel_dump, np.float32).reshape(1, 3, 384, 384),
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
    print(json.dumps({"case": case_name, "tokens": continuation, "text": payload["text"]}))


def test_official_checkpoint_tensors(case_name, artifacts, case_inputs):
    import torch

    bundle, _binary, _runtime, model, _processor, _image_path = artifacts
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
    for embeddings in (expected_vision, actual_vision):
        actual_logits = engine_run(
            bundle, "decoder", {**decoder_inputs, "image_embeds": embeddings}
        )
        np.testing.assert_allclose(actual_logits, expected_logits, atol=1e-3, rtol=1e-4)
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
    np.testing.assert_allclose(actual_vision, expected_vision, atol=5e-4, rtol=1e-4)

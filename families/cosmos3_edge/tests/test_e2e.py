# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real owning build/public-task/reference tests for explicitly scoped profiles.

FP16 video accuracy failures are reproducible and documented in the family
recipe, not represented as passing or hidden behind a looser quality threshold.
"""

from pathlib import Path
import gc
import json
import os
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
FAMILY = "cosmos3_edge"
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text())
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)


def _selection(config, name):
    return {part.strip() for value in (config.getoption(name, default=[]) or [])
            for part in value.split(",") if part.strip()}


def _selected(config, manifest, case):
    models = _selection(config, "--e2e-model")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(line.strip() for line in Path(path).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
    cases = _selection(config, "--e2e-testcase")
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real family E2E requires explicit selection")
    if models and not models.intersection({FAMILY, manifest["name"], case["name"]}):
        pytest.skip("model profile was not selected")
    if cases and case["name"] not in cases:
        pytest.skip("testcase was not selected")


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_COSMOS3_EDGE_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "model.safetensors.index.json").is_file()
        return root
    from huggingface_hub import hf_hub_download, snapshot_download
    index = hf_hub_download(manifest["hf_id"], "model.safetensors.index.json",
                            revision=manifest["hf_revision"])
    shards = set(json.loads(Path(index).read_text())["weight_map"].values())
    return Path(snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"],
                                  allow_patterns=["*.json", "*.jinja", *sorted(shards)]))


def _asset(manifest, name):
    explicit = os.environ.get("TRTMC_COSMOS3_EDGE_ASSET_DIR")
    if explicit:
        result = Path(explicit) / "assets" / name
        assert result.is_file()
        return result
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(manifest["hf_id"], "assets/" + name,
                               revision=manifest["hf_revision"]))


def _inputs(manifest, case, directory):
    from PIL import Image
    image = Image.open(_asset(manifest, "example_reasoning_input.png")).convert("RGB")
    image_path = directory / "image.rgb"
    np.asarray(image).tofile(image_path)
    image_spec = {"path": str(image_path), "height": image.height, "width": image.width}
    requests = [
        {"prompt": "What is the capital of France? Answer briefly."},
        {"prompt": "What is 7 times 8? Answer briefly."},
        {"prompt": "Describe the objects and their colors in this image briefly.", "image": image_spec},
    ]
    arrays, metadata = None, None
    if "video" in case["modalities"]:
        import av
        from transformers.video_utils import VideoMetadata
        with av.open(str(_asset(manifest, "example_action_id_av_0_input.mp4"))) as container:
            stream = container.streams.video[0]
            fps, total = float(stream.average_rate), int(stream.frames)
            indices = np.linspace(0, total - 1, 4).round().astype(int).tolist()
            selected = {}
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    picture = frame.to_image().convert("RGB")
                    scale = min(1.0, 384 / max(picture.size))
                    picture = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.Resampling.BICUBIC)
                    selected[i] = (np.asarray(picture), float(frame.time))
        arrays, specs, stamps = [], [], []
        for number, index in enumerate(indices):
            pixels, stamp = selected[index]
            assert abs(stamp - index / fps) < 1e-3
            path = directory / f"frame-{number}.rgb"
            pixels.tofile(path)
            arrays.append(pixels)
            stamps.append(stamp)
            specs.append({"path": str(path), "height": pixels.shape[0], "width": pixels.shape[1]})
        metadata = VideoMetadata(total_num_frames=total, fps=fps, frames_indices=indices)
        video = {"frames": specs, "timestamps": stamps}
        requests.extend([
            {"prompt": "Describe what the robot does in this video briefly.", "video": video, "max_new_tokens": 96},
            {"prompt": "Describe the image and the action in the video briefly.", "image": image_spec,
             "video": video, "max_new_tokens": 96},
        ])
    return requests, image, arrays, metadata


def _reference(checkpoint, requests, image, frames, metadata, trust_remote_code):
    import torch
    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=trust_remote_code)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    model = None
    try:
        model, info = Cosmos3EdgeForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, device_map="cuda",
            attn_implementation="eager", output_loading_info=True, trust_remote_code=trust_remote_code)
        assert not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")), info
        model.eval()
        results = []
        for request in requests:
            parts = ([{"type": "image"}] if "image" in request else [])
            parts += ([{"type": "video"}] if "video" in request else [])
            parts.append({"type": "text", "text": request["prompt"]})
            prompt = processor.apply_chat_template([{"role": "user", "content": parts}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            media = {"images": [image]} if "image" in request else {}
            if "video" in request:
                media.update(videos=[np.stack(frames)], video_metadata=[metadata], do_sample_frames=False)
            inputs = processor(text=[prompt], return_tensors="pt", **media).to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=request.get("max_new_tokens", 64))
            tokens = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
            results.append({"text": processor.decode(tokens, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False), "token_ids": tokens})
            del inputs, output
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _ned(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = current
    return row[-1] / max(len(a), len(b), 1)


def _runner(runtime):
    roots = [runtime]
    configured = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if configured:
        roots.insert(0, Path(configured))
    # Community CI supplies a small runtime tree of symlinks to its native build.
    for name in ("libtrtmc_model_cosmos3_edge.so", "trtmc_model_cosmos3_edge.dll"):
        library = runtime / name
        if library.is_file():
            roots.append(library.resolve().parent)
    suffix = ".exe" if os.name == "nt" else ""
    for root in roots:
        for configuration in ("", "Release", "Debug"):
            candidate = root / "families/cosmos3_edge" / configuration / ("cosmos3_edge_inference_test" + suffix)
            if candidate.is_file():
                return candidate
    raise AssertionError("build the owning cosmos3_edge_inference_test CMake target")


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path):
    manifest, case = CASES[case_name]
    _selected(request.config, manifest, case)
    assert case["reference_backend"] == "hf_transformers"
    assert case["reference_precision"] == "fp32"
    assert manifest["trust_remote_code"] is False
    import torch
    assert torch.cuda.is_available(), "selected Cosmos3-Edge E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    runner = _runner(runtime)
    assert any((runtime / name).is_file() for name in
               ("libtrtmc_model_cosmos3_edge.so", "trtmc_model_cosmos3_edge.dll"))
    checkpoint = _checkpoint(manifest)
    requests, image, frames, metadata = _inputs(manifest, case, tmp_path)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=FAMILY,
            task=manifest["task"], precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"]))
    source, destination = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(requests))
    command = [str(runner), str(bundle), str(runtime), str(source), str(destination)]
    env = dict(os.environ)
    library_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[library_path] = str(runtime) + os.pathsep + env.get(library_path, "")
    env["TMPDIR"] = str(tmp_path)
    with evidence_stage("native"):
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=1800)
        record_evidence("commands", {"argv": command, "returncode": completed.returncode})
        record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
        assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(destination.read_text())
    with evidence_stage("reference"):
        expected = _reference(checkpoint, requests, image, frames, metadata, manifest["trust_remote_code"])
        record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32", "results": expected})
    assert len(actual) == len(expected) == len(requests)
    threshold = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())["threshold_overrides"]["normalized_text_edit_distance"]
    results = []
    for a, b, item in zip(actual, expected, requests):
        assert a["text"] and 0 < len(a["token_ids"]) <= item.get("max_new_tokens", 64)
        distance = _ned(a["text"], b["text"])
        results.append({"prompt": item["prompt"], "actual": a, "reference": b,
                        "normalized_text_edit_distance": distance})
    record_evidence("metrics", {"cases": results, "normalized_text_edit_distance_max": threshold})
    assert all(row["normalized_text_edit_distance"] <= threshold for row in results), results

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and SDK, contract and official-reference E2E for flux3.

The reference is the Diffusers ``Flux3DistilledModularPipeline`` with the prompt
upsampler removed and the components loaded as the model card loads them
(VAE float32, everything else bfloat16), started from the same initial
latents as the bundle. The sampler amplifies BF16 rounding, so the comparison
bounds drift (PSNR, windowed SSIM, block-mean layout correlation) on prompt and
seed pairs on which the official pipeline's own BF16 and FP32 runs agree.
"""

from __future__ import annotations

import gc
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "flux3"
TASK = "text_to_image"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
DEFAULT_THRESHOLDS = {
    "psnr_db_min": 32.0,
    "ssim_min": 0.95,
    "layout_corr_min": 0.98,
    "text_rel_l2_max": 0.02,
    "velocity_rel_l2_max": 0.03,
}
LATENT_CHANNELS = 128


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        for case in manifest["testcases"]:
            assert case["name"] not in result
            result[case["name"]] = (manifest, case)
    return result


CASES = _cases()


def _selected(config) -> tuple[list[str], bool]:
    models: set[str] = set()
    for raw in config.getoption("--e2e-model") or []:
        models.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcases: set[str] = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcases.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not models and not testcases:
        return sorted(CASES), False
    selected = [
        name
        for name, (manifest, _) in CASES.items()
        if (not models or FAMILY in models or name in models or manifest["name"] in models)
        and (not testcases or name in testcases)
    ]
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected(metafunc.config)
        params = (
            names
            if enabled
            else [
                pytest.param(
                    name, marks=pytest.mark.skip(reason="direct E2E needs an explicit E2E selector")
                )
                for name in names
            ]
        )
        metafunc.parametrize("case_name", params, ids=names)


def _required(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_FLUX3_MODEL_DIR")
    if explicit:
        return _required(explicit, "TRTMC_FLUX3_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest["hf_revision"],
                local_files_only=True,
                allow_patterns=["*.json", "*.md", "text_encoder/*", "transformer/*", "vae/*"],
            )
        )
    except Exception as error:  # noqa: BLE001 - surfaced as a clear assertion
        raise AssertionError(
            f"selected {FAMILY} E2E requires the cached gated checkpoint {manifest['hf_id']}"
        ) from error


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    values = dict(DEFAULT_THRESHOLDS)
    if path.is_file():
        values.update(json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"])
    return values


def _env(runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        v for v in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if v
    )
    return env


def _cli(binary: Path, runtime_root: Path, bundle: Path, *arguments: str, check: bool = True):
    argv = [
        str(binary),
        "generate-image",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, env=_env(runtime_root), timeout=1800, check=False
    )
    record_evidence("commands", {"argv": argv, "returncode": completed.returncode})
    record_evidence(
        "native", {"stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    )
    if check:
        assert completed.returncode == 0, completed.stderr[-2000:]
    return completed


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _initial_latents(case: dict, height: int, width: int) -> np.ndarray:
    """Seeded float32 noise in the reference layout ``[128, H/16, W/16]``, as torch draws it on the CPU."""

    import torch

    generator = torch.Generator().manual_seed(int(case["seed"]))
    shape = (1, LATENT_CHANNELS, height // 16, width // 16)
    return torch.randn(shape, generator=generator, dtype=torch.float32).numpy()[0]


COMPONENT_PROMPT = "A red fox sitting in fresh snow at sunrise, soft golden light, photograph."
FRAME_PROMPTS = ("A red fox.", "  spaces  ", "汉字 and emoji \U0001f600", "")


def _official_pipeline(model_dir: Path):
    import torch
    from diffusers.modular_pipelines import Flux3DistilledAutoBlocks

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    blocks = Flux3DistilledAutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(str(model_dir))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    pipe.to("cuda")
    return pipe


def _reference(pipe, case: dict, latents: np.ndarray, height: int, width: int) -> np.ndarray:
    import torch

    start = torch.from_numpy(latents)[None].to("cuda", torch.bfloat16)
    with torch.no_grad():
        images = pipe(
            prompt=case["prompt"],
            latents=start,
            height=height,
            width=width,
            num_inference_steps=int(case["num_inference_steps"]),
            output="images",
            output_type="np",
        )
    return np.asarray(images[0], dtype=np.float32)


def _bundle_sections(bundle: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    """Selected sections of a bundle file (magic, little-endian header length, JSON header, data)."""

    with bundle.open("rb") as handle:
        assert handle.read(8) == b"BUNDLE\x01\x00"
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
        sections = {}
        for name in names:
            entry = header["sections"][name]
            handle.seek(16 + length + entry["offset"])
            sections[name] = handle.read(entry["length"])
    return sections


def _run_engine(plan: bytes, **inputs):
    """One synchronous execution of a serialized plan on torch CUDA tensors."""

    import tensorrt as trt
    import torch

    dtypes = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.int32: torch.int32}
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None, "TensorRT could not deserialize the plan"
    context = engine.create_execution_context()
    for name, tensor in inputs.items():
        assert tensor.dtype == dtypes[engine.get_tensor_dtype(name)], name
        assert context.set_input_shape(name, tuple(tensor.shape)), name
        context.set_tensor_address(name, tensor.data_ptr())
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            dtype = dtypes[engine.get_tensor_dtype(name)]
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # inputs were produced on the current stream
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def _relative_error(actual, expected) -> float:
    import torch

    a, e = actual.double(), expected.double()
    return float(torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e))


def _frame_checks(runtime: dict, pipe) -> None:
    """The bundle's prompt frame and pad id equal what the reference processor produces."""

    text = runtime["text"]
    for prompt in FRAME_PROMPTS:
        expected = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        assert text["prompt_prefix"] + prompt + text["prompt_suffix"] == expected, prompt
    assert text["pad_token_id"] == pipe.tokenizer.tokenizer.pad_token_id
    record_evidence(
        "contract", {"frame_prompts": list(FRAME_PROMPTS), "pad_token_id": text["pad_token_id"]}
    )


def _component_checks(
    bundle: Path, runtime: dict, pipe, latents: np.ndarray, thresholds: dict, tmp_path: Path
):
    """Short-prompt text embeddings (an 80-token window) and one denoiser call vs the official BF16 modules."""

    import torch
    from diffusers.models.embeddings import get_timestep_embedding
    from diffusers.modular_pipelines.flux3.before_denoise import _prepare_image_ids
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep

    text = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": COMPONENT_PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = pipe.tokenizer.tokenizer(text, return_tensors="pt")["input_ids"][0]
    window = -(-int(ids.shape[0]) // 80) * 80
    padded = torch.full((1, window), runtime["text"]["pad_token_id"], dtype=torch.int32)
    padded[0, : ids.shape[0]] = ids.to(torch.int32)
    with torch.no_grad():
        official_text = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
            pipe.text_encoder, pipe.tokenizer, COMPONENT_PROMPT, device="cuda"
        )
    native_text = _run_engine(
        _bundle_sections(bundle, ("text_encoder.plan",))["text_encoder.plan"],
        input_ids=padded.cuda(),
        valid_length=torch.tensor([ids.shape[0]], dtype=torch.int32, device="cuda"),
    )["text_embeds"]
    assert tuple(native_text.shape) == tuple(official_text.shape)

    height, width = runtime["latent_height"], runtime["latent_width"]
    tokens = torch.from_numpy(latents)[None].to("cuda", torch.bfloat16).flatten(2).transpose(1, 2)
    # The first step of the 16-step schedule: sigma 1, jumping to sigma_1 (shift 6.93 applied to 15 / 16).
    sigma, target = 1.0, float(np.float32(6.93 * 15 / 16) / np.float32(1 + 5.93 * 15 / 16))

    def projection(value: float):
        feed = (torch.tensor([value], device="cuda").to(torch.bfloat16) * 1000).float()
        return get_timestep_embedding(
            feed, 256, flip_sin_to_cos=True, downscale_freq_shift=0
        ).float()

    txt_ids = torch.zeros(window, 4, device="cuda")
    txt_ids[:, 3] = torch.arange(window, device="cuda")
    with torch.no_grad():
        official_velocity = pipe.transformer(
            hidden_states=tokens,
            encoder_hidden_states=official_text,
            timestep=torch.tensor([sigma], device="cuda", dtype=torch.bfloat16),
            target_timestep=torch.tensor([target], device="cuda", dtype=torch.bfloat16),
            img_ids=_prepare_image_ids(height, width, 0.5).cuda(),
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
    native_velocity = _run_engine(
        _bundle_sections(bundle, ("denoiser.plan",))["denoiser.plan"],
        latents=tokens.float().contiguous(),
        text_embeds=official_text.contiguous(),
        timestep_proj=projection(sigma),
        target_timestep_proj=projection(target),
    )["velocity"]

    paths = {}
    for name, value in (
        ("native_text", native_text),
        ("official_text", official_text),
        ("native_velocity", native_velocity),
        ("official_velocity", official_velocity),
    ):
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], value.float().cpu().numpy())
    text_error = _relative_error(native_text, official_text)
    velocity_error = _relative_error(native_velocity, official_velocity)
    for label, native, reference, check in (
        (
            "short-prompt text embeddings",
            paths["native_text"],
            paths["official_text"],
            _check(
                "text_rel_l2",
                "Text embeddings, relative L2 vs official BF16",
                text_error,
                "<=",
                thresholds["text_rel_l2_max"],
            ),
        ),
        (
            "one denoiser call (first step)",
            paths["native_velocity"],
            paths["official_velocity"],
            _check(
                "velocity_rel_l2",
                "Velocity, relative L2 vs official BF16",
                velocity_error,
                "<=",
                thresholds["velocity_rel_l2_max"],
            ),
        ),
    ):
        record_evidence(
            "reference_comparison",
            {
                "label": label,
                "scope": "independent_reference",
                "enforced": True,
                "native": native,
                "reference": reference,
                "checks": [check],
            },
        )
        assert check["passed"], check


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def _ssim(a: np.ndarray, b: np.ndarray, window: int = 8) -> float:
    """Mean SSIM over non-overlapping ``window x window`` blocks of the luma channel."""

    weights = np.array([0.299, 0.587, 0.114])
    x, y = (np.asarray(v, dtype=np.float64) @ weights for v in (a, b))
    h, w = (x.shape[0] // window) * window, (x.shape[1] // window) * window
    x = x[:h, :w].reshape(h // window, window, w // window, window)
    y = y[:h, :w].reshape(h // window, window, w // window, window)
    mx, my = x.mean(axis=(1, 3)), y.mean(axis=(1, 3))
    vx, vy = x.var(axis=(1, 3)), y.var(axis=(1, 3))
    cov = ((x - mx[:, None, :, None]) * (y - my[:, None, :, None])).mean(axis=(1, 3))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(ssim.mean())


def _layout_corr(a: np.ndarray, b: np.ndarray, block: int = 32) -> float:
    def means(v: np.ndarray) -> np.ndarray:
        h, w = (v.shape[0] // block) * block, (v.shape[1] // block) * block
        return v[:h, :w].reshape(h // block, block, w // block, block, -1).mean(axis=(1, 3)).ravel()

    return float(np.corrcoef(means(a), means(b))[0, 1])


def _check(name: str, label: str, actual: float, operator: str, expected: float) -> dict:
    passed = actual >= expected if operator == ">=" else actual <= expected
    return {
        "name": name,
        "label": label,
        "scope": "independent_reference",
        "operator": operator,
        "actual": float(actual),
        "expected": float(expected),
        "passed": bool(passed),
    }


def _sdk_consumers(
    build_dir: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    latents_path: Path,
    tmp_path: Path,
):
    outputs = {}
    for name in ("c", "cpp"):
        binary = build_dir / f"test_flux3_sdk_{name}"
        assert binary.is_file(), f"missing SDK consumer {binary}"
        output = tmp_path / f"sdk_{name}.raw"
        argv = [
            str(binary),
            str(bundle),
            str(runtime_root),
            case["prompt"],
            str(output),
            "--initial-latents-raw",
            str(latents_path),
            "--num-steps",
            str(int(case["num_inference_steps"])),
        ]
        completed = subprocess.run(
            argv, capture_output=True, text=True, env=_env(runtime_root), timeout=1800, check=False
        )
        record_evidence("commands", {"argv": argv, "returncode": completed.returncode})
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs[name] = np.fromfile(output, dtype=np.float32)
    return outputs


def test_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    binary = _required(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    build_dir = _required(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    height, width = int(manifest["image_height"]), int(manifest["image_width"])
    assert int(manifest["tensor_parallel_size"]) == 1, "flux3 runs on one GPU"
    assert case["reference_precision"] == "bf16", "the oracle is the official BF16 pipeline"
    thresholds = _thresholds(case_name)
    record_evidence(
        "checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]}
    )
    record_evidence("thresholds", thresholds)

    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=TASK,
                precision=manifest["precision"],
                max_sequence_length=manifest.get("max_sequence_length"),
                image_height=height,
                image_width=width,
            )
        )

    latents = _initial_latents(case, height, width)
    latents_path = tmp_path / "initial-latents.raw"
    np.ascontiguousarray(latents, dtype=np.float32).tofile(latents_path)
    record_evidence(
        "inputs", {"prompt": case["prompt"], "seed": case["seed"], "raw_file": latents_path}
    )
    steps = str(int(case["num_inference_steps"]))

    with evidence_stage("native"):
        native_png = tmp_path / "native.png"
        _cli(
            binary,
            runtime_root,
            bundle,
            "--prompt",
            case["prompt"],
            "--output",
            str(native_png),
            "--initial-latents-raw",
            str(latents_path),
            "--num-steps",
            steps,
        )
        native = _read_png(native_png)
        assert native.shape == (height, width, 3)
        sdk = _sdk_consumers(build_dir, runtime_root, bundle, case, latents_path, tmp_path)
        for name, pixels in sdk.items():
            assert pixels.size == height * width * 3, name
            # PNG quantisation is the only difference between the CLI and the SDK float output.
            assert (
                float(np.max(np.abs(pixels.reshape(height, width, 3) - native))) <= 0.5 / 255 + 1e-6
            ), name

    with evidence_stage("contract"):
        seeded = [tmp_path / f"seeded_{i}.png" for i in range(3)]
        for path, seed in zip(seeded, ("11", "11", "12")):
            _cli(
                binary,
                runtime_root,
                bundle,
                "--prompt",
                case["prompt"],
                "--output",
                str(path),
                "--seed",
                seed,
                "--num-steps",
                steps,
            )
        first, again, other = (_read_png(path) for path in seeded)
        assert np.array_equal(first, again), "seeded generation must be reproducible"
        assert not np.array_equal(first, other), "a different seed must change the image"
        assert float(first.std()) > 0.02, "seeded generation produced a flat image"
        rejected = {
            "unknown option": ("--prompt", case["prompt"], "--guidance-scale", "3.5"),
            "wrong latent size": (
                "--prompt",
                case["prompt"],
                "--initial-latents-raw",
                str(_short_latents(tmp_path)),
            ),
            "prompt over the window": (
                "--prompt",
                "word " * (int(manifest.get("max_sequence_length", 4000)) + 80),
            ),
        }
        for label, arguments in rejected.items():
            completed = _cli(
                binary,
                runtime_root,
                bundle,
                *arguments,
                "--output",
                str(tmp_path / "rejected.png"),
                check=False,
            )
            assert completed.returncode != 0, f"{label} must be rejected"

    import torch

    pipe = None
    try:
        with evidence_stage("reference"):
            pipe = _official_pipeline(model_dir)
            runtime = json.loads(_bundle_sections(bundle, ("runtime.json",))["runtime.json"])
        with evidence_stage("comparison"):
            _frame_checks(runtime, pipe)
        with evidence_stage("comparison"):
            _component_checks(bundle, runtime, pipe, latents, thresholds, tmp_path)
        with evidence_stage("reference"):
            reference = _reference(pipe, case, latents, height, width)
    finally:
        # Both cases run in one process: return the official pipeline's ~20 GB before the next build.
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    with evidence_stage("comparison"):
        reference_path = tmp_path / "reference.npy"
        native_path = tmp_path / "native.npy"
        np.save(reference_path, reference)
        np.save(native_path, native)
        checks = [
            _check(
                "psnr_db",
                "PSNR against the official BF16 pipeline (dB)",
                _psnr(native, reference),
                ">=",
                thresholds["psnr_db_min"],
            ),
            _check(
                "ssim",
                "Windowed SSIM (8 x 8 luma blocks)",
                _ssim(native, reference),
                ">=",
                thresholds["ssim_min"],
            ),
            _check(
                "layout_corr",
                "Block-mean layout correlation (32 x 32)",
                _layout_corr(native, reference),
                ">=",
                thresholds["layout_corr_min"],
            ),
        ]
        record_evidence(
            "reference_comparison",
            {
                "label": case_name,
                "scope": "independent_reference",
                "enforced": True,
                "native": native_path,
                "reference": reference_path,
                "checks": checks,
            },
        )
        failed = [check for check in checks if not check["passed"]]
        assert not failed, failed


def _short_latents(tmp_path: Path) -> Path:
    path = tmp_path / "short-latents.raw"
    np.zeros(LATENT_CHANNELS, dtype=np.float32).tofile(path)
    return path

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face eager/torch.compile MiniMax-H3 reference and timing receipt."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path
from types import MethodType
import wave

import numpy as np
import torch
from PIL import Image

try:
    from tests.e2e.models.minimax_h3.audio_metrics import (
        EXPECTED_AUDIO_SAMPLE_RATE,
        audio_summary,
        canonical_hf_audio,
        write_float32_wav,
    )
except ModuleNotFoundError:  # Direct execution exposes the sibling script directory.
    from audio_metrics import (  # type: ignore[no-redef]
        EXPECTED_AUDIO_SAMPLE_RATE,
        audio_summary,
        canonical_hf_audio,
        write_float32_wav,
    )
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    checkpoint_snapshot_record,
    file_identity,
    stable_file_record,
    validate_file_identity,
    validate_git_archive_source_unchanged,
    validate_source_revision,
    validated_git_archive_source_record,
    validated_git_source_record,
)

DIFFUSERS_REVISION = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
TRANSFORMERS_COMPAT_REVISION = "bed02e1faee69e866e382f835b4f7b0a3c7b8431"
BASE_TRANSFORMERS_VERSION = "5.2.0"
BASE_TRANSFORMERS_ENTRYPOINT = Path(
    "/opt/venv/lib/python3.12/site-packages/transformers/__init__.py"
)
BASE_TRANSFORMERS_ENTRYPOINT_RECORD = {
    "bytes": 38424,
    "sha256": "91b2c544c6848f4ce8213c770aaa705ce682ee656c995f4ce58352c4b7368ee7",
}
EXPECTED_NUM_FRAMES = 124
_REFERENCE_FLAGS = {
    "--reference-image": "image",
    "--reference-video": "video",
    "--reference-audio": "audio",
}


@dataclass(frozen=True)
class _ResolvedVideoReference:
    manifest_path: Path
    fps: float
    frames: tuple[tuple[str, Path], ...]
    soundtrack: tuple[str, Path] | None


_BoundFileIdentity = tuple[Path, dict[str, int], str]


class _OrderedReferenceAction(argparse.Action):
    """Append heterogeneous reference flags to one encounter-ordered list."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser
        references = list(getattr(namespace, self.dest, None) or [])
        references.append((_REFERENCE_FLAGS[str(option_string)], str(values)))
        setattr(namespace, self.dest, references)


def _write_report_frames(frames: np.ndarray, frames_dir: Path) -> list[Path]:
    """Materialize the full decoded video as canonical PNG report evidence."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("MiniMax-H3 HF report frames must have shape [frames, height, width, 3]")
    if array.shape[0] != EXPECTED_NUM_FRAMES:
        raise ValueError(
            f"MiniMax-H3 HF returned {array.shape[0]} frames instead of {EXPECTED_NUM_FRAMES}"
        )
    if not np.isfinite(array).all():
        raise ValueError("MiniMax-H3 HF report frames contain non-finite pixels")

    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    for index, frame in enumerate(array):
        pixels = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(pixels).save(frames_dir / f"frame_{index:04d}.png")

    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != EXPECTED_NUM_FRAMES:
        raise RuntimeError(
            f"MiniMax-H3 HF wrote {len(paths)} report frames instead of {EXPECTED_NUM_FRAMES}"
        )
    return paths


def _materialize_report_frames(
    frames: np.ndarray,
    frames_dir: Path,
    *,
    output_type: str,
) -> dict | None:
    """Write human-review media only when the pipeline returned decoded RGB."""
    if output_type == "latent":
        return None
    if output_type != "np":
        raise ValueError(f"Unsupported MiniMax-H3 HF output type: {output_type}")

    started = time.perf_counter()
    paths = _write_report_frames(frames, frames_dir)
    return {
        "count": len(paths),
        "directory": frames_dir.name,
        "write_s": time.perf_counter() - started,
        "included_in_median_request_s": False,
    }


def _has_git_checkout(entrypoint: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(entrypoint.resolve(strict=True).parent),
                "rev-parse",
                "--show-toplevel",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def qualified_diffusers_source(entrypoint: Path, evidence_path: Path | None) -> dict:
    if evidence_path is not None:
        return validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers source",
        )
    return {
        "qualification": "clean_git_checkout",
        **validated_git_source_record(
            entrypoint,
            expected_revision=DIFFUSERS_REVISION,
            label="Diffusers source",
        ),
    }


def qualified_transformers_source(entrypoint: Path, version: str) -> dict:
    entrypoint = entrypoint.resolve(strict=True)
    if _has_git_checkout(entrypoint):
        return {
            "qualification": "clean_git_checkout",
            **validated_git_source_record(
                entrypoint,
                expected_revision=TRANSFORMERS_COMPAT_REVISION,
                label="Transformers source",
            ),
        }
    if entrypoint != BASE_TRANSFORMERS_ENTRYPOINT.resolve():
        raise ValueError(
            "MiniMax-H3 Transformers source is neither the pinned Git checkout "
            "nor the qualified immutable CI base"
        )
    if version != BASE_TRANSFORMERS_VERSION:
        raise ValueError(
            "MiniMax-H3 immutable base Transformers version mismatch: "
            f"expected {BASE_TRANSFORMERS_VERSION}, got {version}"
        )
    entrypoint_record, _ = stable_file_record(entrypoint, "Transformers base entrypoint")
    if entrypoint_record != BASE_TRANSFORMERS_ENTRYPOINT_RECORD:
        raise ValueError("MiniMax-H3 immutable base Transformers entrypoint mismatch")
    return {
        "qualification": "immutable_base_5_2_plus_local_shim",
        "version": version,
        "entrypoint": str(entrypoint),
        "entrypoint_record": entrypoint_record,
    }


def create_mm_token_type_ids(processor, input_ids):
    def modality_ids(name):
        plural = getattr(processor, f"{name}_token_ids", None)
        return plural if plural is not None else [getattr(processor, f"{name}_token_id", None)]

    result = []
    for tokenizer_input in input_ids:
        if not isinstance(tokenizer_input, list):
            tokenizer_input = tokenizer_input.tolist()
        tokenizer_input = np.asarray(tokenizer_input)
        token_types = np.zeros_like(tokenizer_input)
        token_types[np.isin(tokenizer_input, modality_ids("image"))] = 1
        token_types[np.isin(tokenizer_input, modality_ids("video"))] = 2
        token_types[np.isin(tokenizer_input, modality_ids("audio"))] = 3
        result.append(token_types.tolist())
    return result


def _processor_method_identity(processor) -> tuple[object, object]:
    method = getattr(processor, "create_mm_token_type_ids", None)
    if not callable(method):
        raise ValueError("MiniMax-H3 processor has no callable create_mm_token_type_ids")
    return getattr(method, "__self__", None), getattr(method, "__func__", method)


def prepare_processor_compat(processor, transformers_source: dict) -> tuple[str | None, tuple]:
    qualification = transformers_source.get("qualification")
    if qualification == "immutable_base_5_2_plus_local_shim":
        if hasattr(processor, "create_mm_token_type_ids"):
            raise ValueError(
                "MiniMax-H3 immutable Transformers base unexpectedly provides "
                "create_mm_token_type_ids"
            )
        processor.create_mm_token_type_ids = MethodType(create_mm_token_type_ids, processor)
        identity = _processor_method_identity(processor)
        if identity != (processor, create_mm_token_type_ids):
            raise ValueError("MiniMax-H3 could not bind its local processor compatibility helper")
        return "local-create-mm-token-type-ids-for-transformers-5.2.0", identity
    if qualification != "clean_git_checkout":
        raise ValueError("MiniMax-H3 Transformers source has an unknown qualification")
    return None, _processor_method_identity(processor)


def validate_processor_method_unchanged(processor, expected: tuple[object, object]) -> None:
    current = _processor_method_identity(processor)
    if current[0] is not expected[0] or current[1] is not expected[1]:
        raise ValueError("MiniMax-H3 processor compatibility helper changed during the run")


def pipeline_arguments(
    *,
    prompt: str,
    generator,
    steps: int,
    output_type: str,
    image=None,
    last_image=None,
    references=None,
) -> dict:
    """Construct one official Diffusers T2VA/FL2VA/Ref2VA invocation."""

    arguments = {
        "prompt": prompt,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": output_type,
        "output": ["videos", "audio", "sampling_rate"],
    }
    if image is not None:
        arguments["image"] = image
    if last_image is not None:
        arguments["last_image"] = last_image
    if references is not None:
        arguments["references"] = references
    return arguments


def create_official_pipeline(
    *,
    workflow: str,
    model_path: Path,
    components_manager,
    modular_pipeline_type,
    ref2va_blocks_type,
):
    """Construct the pinned official pipeline for the requested H3 checkpoint partition."""

    if workflow == "ref2va":
        return ref2va_blocks_type().init_pipeline(
            model_path,
            components_manager=components_manager,
        )
    if workflow not in {"t2va", "fl2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 HF workflow: {workflow!r}")
    return modular_pipeline_type.from_pretrained(
        model_path,
        components_manager=components_manager,
    )


def validate_official_pipeline_partition(pipe, workflow: str) -> dict:
    """Fail closed if Diffusers selected the other H3 transformer partition."""

    pipeline_class = type(pipe).__name__
    blocks_class = type(pipe._blocks).__name__
    component_names = sorted(pipe.component_names)
    component_set = set(component_names)
    expected = "transformer_ref" if workflow == "ref2va" else "transformer"
    unexpected = "transformer" if workflow == "ref2va" else "transformer_ref"
    if expected not in component_set or unexpected in component_set:
        raise ValueError(
            f"MiniMax-H3 {workflow} HF pipeline selected the wrong checkpoint partition: "
            f"expected {expected} without {unexpected}, got {component_names}"
        )
    block_inputs = sorted(pipe._blocks.input_names)
    if workflow == "ref2va" and "references" not in block_inputs:
        raise ValueError("MiniMax-H3 Ref2VA HF pipeline does not consume references")
    if workflow == "ref2va" and (
        pipeline_class != "MiniMaxH3Ref2VAModularPipeline"
        or blocks_class != "MiniMaxH3Ref2VABlocks"
    ):
        raise ValueError(
            "MiniMax-H3 Ref2VA HF pipeline selected the wrong official pipeline types: "
            f"got {pipeline_class}/{blocks_class}"
        )
    return {
        "pipeline_class": pipeline_class,
        "blocks_class": blocks_class,
        "component_names": component_names,
        "block_inputs": block_inputs,
    }


def _workflow_transformer_component(workflow: str) -> str:
    if workflow == "ref2va":
        return "transformer_ref"
    if workflow in {"t2va", "fl2va"}:
        return "transformer"
    raise ValueError(f"Unsupported MiniMax-H3 HF workflow: {workflow!r}")


def compile_official_transformer(
    pipe,
    workflow: str,
    *,
    mode: str,
    compile_function,
) -> str:
    """Compile and replace exactly the transformer partition used by ``workflow``."""

    component = _workflow_transformer_component(workflow)
    unexpected = "transformer" if component == "transformer_ref" else "transformer_ref"
    component_names = set(pipe.component_names)
    if component not in component_names or unexpected in component_names:
        raise ValueError(
            f"MiniMax-H3 {workflow} HF compile selected the wrong checkpoint partition: "
            f"expected {component} without {unexpected}, got {sorted(component_names)}"
        )
    transformer = getattr(pipe, component, None)
    if transformer is None:
        raise ValueError(f"MiniMax-H3 {workflow} HF compile component {component} is unavailable")
    update_components = getattr(pipe, "update_components", None)
    if not callable(update_components):
        raise ValueError("MiniMax-H3 HF pipeline cannot update its compiled component")

    compiled_transformer = compile_function(transformer, mode=mode, dynamic=False)
    if compiled_transformer is None:
        raise ValueError("MiniMax-H3 HF compile did not return a transformer")
    update_components(**{component: compiled_transformer})
    if (
        component not in set(pipe.component_names)
        or getattr(pipe, component, None) is not compiled_transformer
    ):
        raise ValueError(f"MiniMax-H3 HF pipeline did not install compiled component {component}")
    return component


def _keyframe_mode(image, last_image) -> str:
    if image is not None and last_image is not None:
        return "first_and_last"
    if image is not None:
        return "first"
    if last_image is not None:
        return "last"
    return "zero"


def _decode_reference_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Decode the model-owned PCM fixture without Diffusers' optional PyAV path."""

    path = path.resolve(strict=True)
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frame_count = stream.getnframes()
        compression = stream.getcomptype()
        payload = stream.readframes(frame_count)

    if compression != "NONE" or sample_width != 2:
        raise ValueError("MiniMax-H3 HF reference audio must be uncompressed 16-bit PCM WAV")
    if channels not in (1, 2):
        raise ValueError("MiniMax-H3 HF reference audio must be mono or stereo")
    expected_bytes = frame_count * channels * sample_width
    if len(payload) != expected_bytes:
        raise ValueError(
            "MiniMax-H3 HF reference WAV payload is truncated: "
            f"expected {expected_bytes} bytes, got {len(payload)}"
        )

    interleaved = np.frombuffer(payload, dtype="<i2")
    frames = interleaved.reshape(frame_count, channels).astype(np.float32)
    frames *= np.float32(1.0 / 32768.0)
    waveform = torch.from_numpy(np.ascontiguousarray(frames.T))
    return waveform, sample_rate


def _resolve_video_reference(path: Path) -> _ResolvedVideoReference:
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) not in (
        {"fps", "frames"},
        {"fps", "frames", "audio"},
    ):
        raise ValueError("MiniMax-H3 reference video manifest has an invalid schema")
    fps = manifest["fps"]
    frames = manifest["frames"]
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("MiniMax-H3 reference video fps must be finite and positive")
    if not isinstance(frames, list) or not frames:
        raise ValueError("MiniMax-H3 reference video frames must be a non-empty list")
    frame_paths = []
    for value in frames:
        if not isinstance(value, str) or not value:
            raise ValueError("MiniMax-H3 reference video frame paths must be strings")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MiniMax-H3 reference video frame paths must be relative")
        frame_paths.append((relative.as_posix(), (path.parent / relative).resolve(strict=True)))
    audio = manifest.get("audio")
    soundtrack = None
    if audio is not None:
        if not isinstance(audio, str) or not audio:
            raise ValueError("MiniMax-H3 reference video audio must be a path string")
        relative = Path(audio)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MiniMax-H3 reference video audio must be relative")
        soundtrack = (relative.as_posix(), (path.parent / relative).resolve(strict=True))
    return _ResolvedVideoReference(
        manifest_path=path,
        fps=float(fps),
        frames=tuple(frame_paths),
        soundtrack=soundtrack,
    )


def _video_reference_arguments(
    path: Path,
    load_image,
    *,
    resolved: _ResolvedVideoReference | None = None,
) -> dict:
    path = path.resolve(strict=True)
    resolved = _resolve_video_reference(path) if resolved is None else resolved
    if resolved.manifest_path != path:
        raise ValueError("MiniMax-H3 resolved video does not match its reference manifest")
    arguments = {
        "video": [load_image(str(frame_path)) for _relative, frame_path in resolved.frames],
        "fps": resolved.fps,
    }
    if resolved.soundtrack is not None:
        _relative, soundtrack_path = resolved.soundtrack
        waveform, sample_rate = _decode_reference_wav(soundtrack_path)
        arguments["audio"] = waveform
        arguments["sample_rate"] = sample_rate
    return arguments


def bind_reference_input_records(
    reference_specs: list[tuple[str, Path]],
) -> tuple[
    list[dict],
    list[_BoundFileIdentity],
    dict[int, _ResolvedVideoReference],
]:
    """Bind ordered reference receipts to every file the HF oracle will read."""

    records = []
    identities = []
    resolved_videos = {}
    for index, (kind, raw_path) in enumerate(reference_specs):
        if kind not in {"image", "video", "audio"}:
            raise ValueError(f"Unsupported MiniMax-H3 reference kind: {kind!r}")
        path = Path(raw_path).resolve(strict=True)
        label = f"reference {index} {kind}"
        record, identity = stable_file_record(path, label)
        receipt_record = {"kind": kind, **record}
        identities.append((path, identity, label))
        if kind == "video":
            video = _resolve_video_reference(path)
            resolved_videos[index] = video
            frame_records = []
            for frame_index, (relative, frame_path) in enumerate(video.frames):
                frame_label = f"reference {index} video frame {frame_index}"
                frame_record, frame_identity = stable_file_record(frame_path, frame_label)
                frame_records.append({"path": relative, **frame_record})
                identities.append((frame_path, frame_identity, frame_label))
            receipt_record["frames"] = frame_records
            if video.soundtrack is not None:
                relative, soundtrack_path = video.soundtrack
                soundtrack_label = f"reference {index} video soundtrack"
                soundtrack_record, soundtrack_identity = stable_file_record(
                    soundtrack_path,
                    soundtrack_label,
                )
                receipt_record["soundtrack"] = {"path": relative, **soundtrack_record}
                identities.append((soundtrack_path, soundtrack_identity, soundtrack_label))
            validate_file_identity(path, identity, label)
        records.append(receipt_record)
    return records, identities, resolved_videos


def validate_bound_reference_identities(identities: list[_BoundFileIdentity]) -> None:
    """Revalidate all direct and manifest-nested reference files after generation."""

    for path, identity, label in identities:
        validate_file_identity(path, identity, label)


def build_official_references(
    reference_specs,
    reference_type,
    load_image,
    *,
    resolved_videos: dict[int, _ResolvedVideoReference] | None = None,
) -> list:
    """Create official Diffusers references without regrouping their modalities."""

    if resolved_videos is not None:
        expected_video_indices = {
            index for index, (kind, _raw_path) in enumerate(reference_specs) if kind == "video"
        }
        if set(resolved_videos) != expected_video_indices:
            raise ValueError("MiniMax-H3 resolved videos do not match ordered references")
    references = []
    for index, (kind, raw_path) in enumerate(reference_specs):
        path = Path(raw_path).resolve(strict=True)
        if kind == "video":
            resolved = None if resolved_videos is None else resolved_videos[index]
            arguments = _video_reference_arguments(path, load_image, resolved=resolved)
        elif kind == "image":
            arguments = {"image": load_image(str(path))}
        elif kind == "audio":
            waveform, sample_rate = _decode_reference_wav(path)
            arguments = {"audio": waveform, "sample_rate": sample_rate}
        else:
            raise ValueError(f"Unsupported MiniMax-H3 reference kind: {kind!r}")
        references.append(reference_type(**arguments))
    return references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workflow", choices=("t2va", "fl2va", "ref2va"), default="t2va")
    parser.add_argument("--first-image")
    parser.add_argument("--last-image")
    for flag in _REFERENCE_FLAGS:
        parser.add_argument(flag, dest="reference_specs", action=_OrderedReferenceAction)
    parser.add_argument("--diffusers-evidence")
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-type", default="np", choices=("np", "latent"))
    args = parser.parse_args()
    source_revision = validate_source_revision(args.source_revision)
    if args.warmup < 0 or args.measure < 1 or args.steps < 1:
        raise ValueError("warmup must be non-negative; measure and steps must be positive")

    model_path = Path(args.model_path)
    snapshot_record = checkpoint_snapshot_record(model_path, workflow=args.workflow)
    prompt_path = Path(args.prompt_file)
    prompt_identity = file_identity(prompt_path)
    prompt_spec = json.loads(prompt_path.read_text())
    prompt_record, prompt_hashed_identity = stable_file_record(prompt_path, "prompt file")
    if prompt_hashed_identity != prompt_identity:
        raise ValueError("MiniMax-H3 prompt file changed while it was being read")
    if not isinstance(prompt_spec.get("prompt"), str) or not prompt_spec["prompt"]:
        raise ValueError("MiniMax-H3 prompt file must contain a non-empty prompt")
    if not isinstance(prompt_spec.get("seed"), int) or isinstance(prompt_spec["seed"], bool):
        raise ValueError("MiniMax-H3 prompt file must contain an integer seed")
    keyframe_paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("first_image", args.first_image),
            ("last_image", args.last_image),
        )
        if value
    }
    reference_paths = [
        (kind, Path(value).resolve(strict=True)) for kind, value in (args.reference_specs or [])
    ]
    if args.workflow == "ref2va":
        if keyframe_paths:
            raise ValueError("MiniMax-H3 Ref2VA reference does not accept FL2VA keyframes")
        if not reference_paths:
            raise ValueError("MiniMax-H3 Ref2VA reference requires ordered references")
        kinds = [kind for kind, _path in reference_paths]
        if not ({"image", "video"} & set(kinds)):
            raise ValueError(
                "MiniMax-H3 audio references require at least one image or video reference"
            )
    elif reference_paths:
        raise ValueError(
            f"MiniMax-H3 {args.workflow.upper()} reference does not accept omni-references"
        )
    elif args.workflow == "t2va" and keyframe_paths:
        raise ValueError("MiniMax-H3 T2VA reference does not accept keyframe images")
    keyframe_records = {}
    keyframe_identities = {}
    for name, path in keyframe_paths.items():
        identity = file_identity(path)
        record, hashed_identity = stable_file_record(path, name.replace("_", " "))
        if hashed_identity != identity:
            raise ValueError(f"MiniMax-H3 {name} changed while it was being read")
        keyframe_records[name] = record
        keyframe_identities[name] = hashed_identity
    reference_records, reference_identities, resolved_videos = bind_reference_input_records(
        reference_paths
    )
    input_records = {"prompt_file": prompt_record, **keyframe_records}
    if reference_records:
        input_records["references"] = reference_records
    script_path = Path(__file__).resolve()
    script_record, script_identity = stable_file_record(script_path, "HF reference helper")
    request = {
        "prompt": prompt_spec["prompt"],
        "seed": int(prompt_spec["seed"]),
        "workflow": args.workflow,
        "keyframe_mode": _keyframe_mode(
            keyframe_paths.get("first_image"),
            keyframe_paths.get("last_image"),
        ),
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "num_inference_steps": args.steps,
        "output_type": args.output_type,
        "outputs": ["videos", "audio", "sampling_rate"],
        "warmup": args.warmup,
        "measure": args.measure,
    }
    if reference_paths:
        request["reference_kinds"] = [kind for kind, _path in reference_paths]

    import diffusers
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3Ref2VABlocks,
        MiniMaxH3Reference,
    )
    from diffusers.utils import load_image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    for stale in (
        output_dir / "hf_receipt.json",
        output_dir / "hf_frames.npy",
        output_dir / "hf_audio.npy",
        output_dir / "audio.wav",
    ):
        stale.unlink(missing_ok=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    # Keep the lexical evidence path intact. The provenance validator owns
    # existence, stability, and no-symlink enforcement.
    diffusers_evidence = (
        Path(args.diffusers_evidence).absolute() if args.diffusers_evidence else None
    )
    diffusers_source = qualified_diffusers_source(
        Path(diffusers.__file__),
        diffusers_evidence,
    )
    keyframe_images = {name: load_image(str(path)) for name, path in keyframe_paths.items()}
    references = build_official_references(
        reference_paths,
        MiniMaxH3Reference,
        load_image,
        resolved_videos=resolved_videos,
    )
    manager = ComponentsManager()
    started = time.perf_counter()
    pipe = create_official_pipeline(
        workflow=args.workflow,
        model_path=model_path,
        components_manager=manager,
        modular_pipeline_type=ModularPipeline,
        ref2va_blocks_type=MiniMaxH3Ref2VABlocks,
    )
    pipeline_record = validate_official_pipeline_partition(pipe, args.workflow)
    # The published modular index records the Hub repo for each component. Override
    # that field so a pinned local snapshot remains fully offline and revision exact.
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=args.model_path)
    import transformers

    transformers_source = qualified_transformers_source(
        Path(transformers.__file__),
        transformers.__version__,
    )
    processor_compat, processor_method_identity = prepare_processor_compat(
        pipe.processor, transformers_source
    )
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    phase = "compile"
    compile_component = _workflow_transformer_component(args.workflow) if args.use_compile else None
    try:
        if args.use_compile:
            compile_official_transformer(
                pipe,
                args.workflow,
                mode=args.compile_mode,
                compile_function=torch.compile,
            )

        def run():
            generator = torch.Generator().manual_seed(int(prompt_spec["seed"]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            state = pipe(
                **pipeline_arguments(
                    prompt=prompt_spec["prompt"],
                    generator=generator,
                    steps=args.steps,
                    output_type=args.output_type,
                    image=keyframe_images.get("first_image"),
                    last_image=keyframe_images.get("last_image"),
                    references=references if args.workflow == "ref2va" else None,
                )
            )
            torch.cuda.synchronize()
            return state, time.perf_counter() - begin

        phase = "warmup"
        for _ in range(args.warmup):
            run()
        phase = "measure"
        timings, state = [], None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.measure):
            state, elapsed = run()
            timings.append(elapsed)
    except Exception as error:
        failure_receipt = {
            "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
            "status": "failed",
            "failure_phase": phase,
            "error": f"{type(error).__name__}: {error}",
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
            "source_revision": source_revision,
            "builder_source": script_record,
            "checkpoint_snapshot": snapshot_record,
            "inputs": input_records,
            "request": request,
            "diffusers_revision": DIFFUSERS_REVISION,
            "diffusers_version": diffusers.__version__,
            "diffusers_source": diffusers_source,
            "transformers_source": transformers_source,
            "compile_mode": args.compile_mode if args.use_compile else None,
            "compile_component": compile_component,
            "load_s": load_s,
            "torch": torch.__version__,
            "processor_compat": processor_compat,
            "pipeline": pipeline_record,
            "gpu": torch.cuda.get_device_name(0),
            "host": platform.node(),
        }
        atomic_write_json(output_dir / "hf_receipt.json", failure_receipt)
        print(json.dumps(failure_receipt, indent=2))
        raise
    videos = state.get("videos")
    if isinstance(videos, torch.Tensor):
        frames = videos.detach().float().cpu().numpy()
    else:
        frames = np.asarray(videos[0])
    frames_path = output_dir / "hf_frames.npy"
    np.save(frames_path, frames)
    frames_record, _ = stable_file_record(frames_path, "HF decoded frames")
    audio_record = None
    audio_wav_record = None
    audio_evidence = None
    if args.output_type != "latent":
        raw_audio = state.get("audio")
        if isinstance(raw_audio, torch.Tensor):
            raw_audio = raw_audio.detach().float().cpu().numpy()
        sampling_rate = state.get("sampling_rate")
        if isinstance(sampling_rate, torch.Tensor):
            if sampling_rate.numel() != 1:
                raise ValueError("MiniMax-H3 HF sampling_rate must be a scalar")
            sampling_rate = sampling_rate.detach().cpu().item()
        audio = canonical_hf_audio(raw_audio, sampling_rate)
        audio_path = output_dir / "hf_audio.npy"
        np.save(audio_path, audio, allow_pickle=False)
        audio_record, _ = stable_file_record(audio_path, "HF decoded audio")
        audio_wav_path = output_dir / "audio.wav"
        write_float32_wav(audio_wav_path, audio, EXPECTED_AUDIO_SAMPLE_RATE)
        audio_wav_record, _ = stable_file_record(audio_wav_path, "HF decoded audio WAV")
        audio_evidence = {
            **audio_summary(audio, EXPECTED_AUDIO_SAMPLE_RATE),
            "raw_shape": [1, *[int(value) for value in audio.shape]],
        }
    report_frames = _materialize_report_frames(
        frames,
        frames_dir,
        output_type=args.output_type,
    )
    validate_file_identity(prompt_path, prompt_hashed_identity, "prompt file")
    for name, path in keyframe_paths.items():
        validate_file_identity(path, keyframe_identities[name], name.replace("_", " "))
    validate_bound_reference_identities(reference_identities)
    validate_file_identity(script_path, script_identity, "HF reference helper")
    if diffusers_evidence is not None:
        validate_git_archive_source_unchanged(
            Path(diffusers.__file__),
            evidence_path=diffusers_evidence,
            expected_record=diffusers_source,
            label="Diffusers source",
        )
    elif qualified_diffusers_source(Path(diffusers.__file__), None) != diffusers_source:
        raise ValueError("MiniMax-H3 Diffusers source changed during the HF reference run")
    if (
        qualified_transformers_source(Path(transformers.__file__), transformers.__version__)
        != transformers_source
    ):
        raise ValueError("MiniMax-H3 Transformers source changed during the HF reference run")
    validate_processor_method_unchanged(pipe.processor, processor_method_identity)
    if checkpoint_snapshot_record(model_path, workflow=args.workflow) != snapshot_record:
        raise ValueError("MiniMax-H3 checkpoint snapshot changed during the HF reference run")
    receipt = {
        "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
        "source_revision": source_revision,
        "builder_source": script_record,
        "checkpoint_snapshot": snapshot_record,
        "inputs": input_records,
        "request": request,
        "diffusers_revision": DIFFUSERS_REVISION,
        "diffusers_version": diffusers.__version__,
        "diffusers_source": diffusers_source,
        "transformers_source": transformers_source,
        "compile_mode": args.compile_mode if args.use_compile else None,
        "compile_component": compile_component,
        "status": "passed",
        "load_s": load_s,
        "request_s": timings,
        "median_request_s": float(np.median(timings)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "torch": torch.__version__,
        "processor_compat": processor_compat,
        "pipeline": pipeline_record,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "shape": list(frames.shape),
        "frames": frames_record,
    }
    if audio_evidence is not None:
        receipt.update(
            {
                "audio_shape": audio_evidence["shape"],
                "raw_audio_shape": audio_evidence["raw_shape"],
                "audio_sample_rate_hz": audio_evidence["sample_rate_hz"],
                "audio_num_samples_per_channel": audio_evidence["num_samples_per_channel"],
                "audio_duration_s": audio_evidence["duration_s"],
                "audio_all_finite": audio_evidence["all_finite"],
                "audio_rms": audio_evidence["rms"],
                "audio_peak_absolute": audio_evidence["peak_absolute"],
                "audio_layout": audio_evidence["layout"],
                "audio_encoding": audio_evidence["encoding"],
                "audio_wav_encoding": "ieee_float32le",
                "audio": audio_record,
                "audio_wav": audio_wav_record,
            }
        )
    if report_frames is not None:
        receipt["report_frames"] = report_frames
    atomic_write_json(output_dir / "hf_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the two changed continuation plans from an explicitly pinned bundle.

The caller selects compatible compiled artifacts and owns writer finish/abort.
This does not adapt plans to another platform or qualify trained video output.
"""

from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import re
import struct

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC

from . import reference_build as owner

REUSED_PLANS = tuple(n for n in owner.PLAN_NAMES if n not in ("denoiser.plan", "audio_vae.plan"))
_CHUNK = 8 << 20


def _checkpoint_identity(model_dir):
    """Recognize the snapshot identity used by this family's source-bundle gate."""
    if model_dir is None:
        return None
    try:
        path = Path(model_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    repository = path.parent.parent.name
    if (
        path.parent.name != "snapshots"
        or re.fullmatch("[0-9a-f]{40}", path.name) is None
        or not repository.startswith("models--")
    ):
        return None
    return {
        "hf_id": repository.removeprefix("models--").replace("--", "/"),
        "revision": path.name,
    }


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class _ImageBundle:
    """Validate every byte once; copy selected plans with bounded host buffers."""

    def __init__(self, path, expected_sha256, expected_runtime, checkpoint, tokenizer):
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", expected_sha256
        ):
            raise ValueError("MiniMax-H3 source bundle requires an explicit lowercase SHA-256")
        self.path = Path(path)
        self.sha256 = expected_sha256
        self.identity = _identity(self.path.stat())
        with self.path.open("rb") as source:
            prefix = source.read(16)
            if len(prefix) != 16 or prefix[:8] != BUNDLE_MAGIC:
                raise ValueError("MiniMax-H3 source is not a format-v1 bundle")
            size = struct.unpack("<Q", prefix[8:])[0]
            if not 0 < size <= 100 << 20 or size > self.identity[2] - 16:
                raise ValueError("MiniMax-H3 source bundle has an invalid header length")
            raw_header = source.read(size)
            header = json.loads(raw_header)
            expected_header = dict(format=1, family="minimax_h3", task=owner.TASK, backend="trt")
            if any(header.get(k) != v for k, v in expected_header.items()):
                raise ValueError("MiniMax-H3 source bundle has a different family/task/backend")
            self.base = 16 + size
            self.sections = header["sections"]
            required = {
                *owner.PLAN_NAMES,
                "runtime.json",
                "tokenizer.json",
                "trtmc.provenance.json",
            }
            if set(self.sections) != required:
                raise ValueError(
                    "MiniMax-H3 source requires the complete original eight-plan bundle"
                )
            position = 0
            for name, section in sorted(self.sections.items(), key=lambda x: x[1]["offset"]):
                offset, length = section["offset"], section["length"]
                if (
                    type(offset) is not int
                    or type(length) is not int
                    or offset != position
                    or length <= 0
                ):
                    raise ValueError(
                        "MiniMax-H3 source section ranges must be disjoint and complete"
                    )
                position += length
                if position > self.identity[2] - self.base:
                    raise ValueError("MiniMax-H3 source section exceeds its bundle")
            if position != self.identity[2] - self.base:
                raise ValueError("MiniMax-H3 source bundle has unaccounted trailing bytes")

            def metadata(name):
                section = self.sections[name]
                if section["length"] > 16 << 20:
                    raise ValueError("MiniMax-H3 source metadata is unexpectedly large")
                source.seek(self.base + section["offset"])
                return json.loads(source.read(section["length"]))

            if metadata("runtime.json") != expected_runtime:
                raise ValueError(
                    "MiniMax-H3 source profile/config/statistics differ from the requested base"
                )
            record = metadata("trtmc.provenance.json")
            if not checkpoint or record.get("checkpoint") != checkpoint:
                raise ValueError(
                    "MiniMax-H3 source and new build require the same pinned HF checkpoint"
                )
            build = record.get("build", {})
            if build.get("precision") != "bf16" or build.get("quantization") not in (None, "none"):
                raise ValueError("MiniMax-H3 source requires the original unquantized BF16 build")
            owner._require(
                build,
                dict(
                    tensor_parallel_size=1,
                    context_parallel_size=1,
                    max_batch_size=1,
                    dynamic_kv_cache=False,
                    fp32_layers=[],
                    max_sequence_length=None,
                    mtp_seq_len=None,
                    graph_transform=False,
                ),
                "source bundle build overrides",
            )
            source.seek(self.base)
            whole = hashlib.sha256(prefix + raw_header)
            self.digests = {}
            for name, section in sorted(self.sections.items(), key=lambda x: x[1]["offset"]):
                digest = hashlib.sha256()
                remaining = section["length"]
                while remaining:
                    data = source.read(min(_CHUNK, remaining))
                    if not data:
                        raise ValueError("MiniMax-H3 source bundle was truncated")
                    whole.update(data)
                    digest.update(data)
                    remaining -= len(data)
                self.digests[name] = digest.hexdigest()
            if whole.hexdigest() != expected_sha256:
                raise ValueError("MiniMax-H3 source bundle SHA-256 mismatch")
            if self.digests["tokenizer.json"] != hashlib.sha256(tokenizer).hexdigest():
                raise ValueError("MiniMax-H3 source tokenizer differs from the new checkpoint")
        self._unchanged()

    def _unchanged(self):
        if _identity(self.path.stat()) != self.identity:
            raise ValueError("MiniMax-H3 source bundle changed during reuse")

    def copy_to(self, name, writer):
        if name not in REUSED_PLANS:
            raise ValueError("MiniMax-H3 continuation must rebuild its DiT and audio decoder")
        self._unchanged()
        section = self.sections[name]
        digest = hashlib.sha256()
        with self.path.open("rb") as source, writer.open_section(name) as output:
            source.seek(self.base + section["offset"])
            remaining = section["length"]
            while remaining:
                data = source.read(min(_CHUNK, remaining))
                if not data:
                    raise ValueError("MiniMax-H3 source plan was truncated during copy")
                output.write(data)
                digest.update(data)
                remaining -= len(data)
            if digest.hexdigest() != self.digests[name]:
                raise ValueError("MiniMax-H3 source plan content changed during copy")
            self._unchanged()


def build(
    request,
    writer,
    *,
    source_bundle,
    source_sha256,
    memory_preset="minimal",
    reference_capacity="six_portrait",
):
    """Build an explicit 226-frame bundle, reusing six byte-identical plans.

    Validate the complete source digest, original profile, checkpoint identity,
    configuration, statistics and tokenizer before loading any new weights.
    The source's six compiled plans must be compatible with the selected backend;
    no numerical or hardware fallback is performed.
    """
    if Path(source_bundle).resolve() == Path(request.output_path).resolve():
        raise ValueError("MiniMax-H3 continuation must preserve its original source bundle")
    if request.graph_transform is not None:
        raise ValueError("MiniMax-H3 incremental continuation does not support graph transforms")
    from . import dit_builder as dit, audio_vae_builder as audio

    height, width, target, reference = owner._target(
        request, memory_preset, reference_capacity, True
    )
    root = Path(request.model_dir)
    configs, hashes = owner._configs(root)
    tokenizer = (root / "tokenizer/tokenizer.json").read_bytes()
    base_request = replace(request, video_num_frames=124)
    _, _, base_target, base_reference = owner._target(
        base_request, memory_preset, reference_capacity, False
    )
    common = dict(memory_preset=memory_preset, reference_capacity=reference_capacity)
    expected_base = owner.runtime_metadata(
        height, width, base_target, base_reference, configs, hashes, continuation=False, **common
    )
    source = _ImageBundle(
        source_bundle, source_sha256, expected_base, _checkpoint_identity(root), tokenizer
    )
    writer.enable_direct_staging()
    writer.set_header(family="minimax_h3", task=owner.TASK, backend="trt")
    writer.record_build(request)
    for name in owner.PLAN_NAMES:
        if name in REUSED_PLANS:
            source.copy_to(name, writer)
            continue
        if name == "denoiser.plan":
            component, keys = "transformer_ref", dit.checkpoint_keys(target)
        else:
            component, keys = "audio_vae", audio.checkpoint_keys()
        state = owner.load_selected_component_state_dict(root / component, keys)
        weights = owner.numpy_state(state)
        del state
        options = dict(
            verbose=request.verbose, consume_weights=True, workspace_bytes=owner.WORKSPACE_BYTES
        )
        if name == "denoiser.plan":
            payload = dit.build_dit_engine(weights, target, reference_profile=reference, **options)
        else:
            payload = audio.build_audio_vae_decoder_engine(
                weights,
                configs["audio_vae"]["latents_mean"],
                configs["audio_vae"]["latents_std"],
                latent_frames=377,
                **options,
            )
        del weights
        gc.collect()
        writer.add_bytes(name, payload)
        del payload
        gc.collect()
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        owner.runtime_metadata(
            height, width, target, reference, configs, hashes, continuation=True, **common
        ),
    )
    writer.add_json(
        "minimax_h3.reused_plans.json",
        dict(
            source_bundle_sha256=source.sha256,
            reused_sections={n: source.digests[n] for n in REUSED_PLANS},
            rebuilt_sections=["denoiser.plan", "audio_vae.plan"],
        ),
    )

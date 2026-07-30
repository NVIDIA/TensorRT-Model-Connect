# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream Lance image-understanding reference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


_REFERENCE_REPO_ENV = "TRTMC_LANCE_REFERENCE_REPO"
_REFERENCE_ENTRYPOINT = "inference_lance.py"
_MODEL_DIRECTORY = "Lance_3B"
_VIT_DIRECTORY = "Qwen2.5-VL-ViT"
_IMAGE_REFERENCE_COMPAT = Path(__file__).with_name("lance_image_compat")


def _image_reference_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Expose only the optional imports needed by upstream's image path."""
    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(_IMAGE_REFERENCE_COMPAT), existing)
        if value
    )
    return environment


def _cached_model_root(model_id: str) -> Path:
    path = Path(model_id)
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(model_id, local_files_only=True)
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Lance reference checkpoint {model_id!r} is not available locally"
        ) from exc


def _official_source() -> Path:
    value = os.environ.get(_REFERENCE_REPO_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"Lance reference requires {_REFERENCE_REPO_ENV}; "
            "trtmc-validate prepares this pinned source automatically"
        )
    source = Path(value).resolve()
    entrypoint = source / _REFERENCE_ENTRYPOINT
    if not entrypoint.is_file():
        raise RuntimeError(
            f"Lance reference checkout is missing {entrypoint}"
        )
    return source


def _result_text(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Lance official reference did not produce a valid {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(
            f"Lance official reference expected one result, got {payload!r}"
        )
    answer = str(payload[0].get("answer", "") or "").strip()
    if not answer:
        raise RuntimeError("Lance official reference produced an empty answer")
    return answer


class LanceOfficialReference:
    """Run the pinned upstream ``inference_lance.py`` x2t_image path."""

    @property
    def backend_name(self) -> str:
        return "lance_official"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if (
            case.task_strategy != "vision_language_generation"
            or stage.name != "full_generation"
        ):
            return StageOutput(
                stage_name=stage.name,
                data={
                    "error": "Lance official reference only supports "
                    "vision_language_generation/full_generation"
                },
            )

        image = Path(str(case.inputs.get("image", "") or "")).resolve()
        if not image.is_file():
            raise RuntimeError(f"Lance reference image does not exist: {image}")
        prompt = str(case.inputs.get("prompt", "") or "").strip()
        if not prompt:
            raise RuntimeError("Lance reference requires a non-empty prompt")

        source = _official_source()
        model_root = _cached_model_root(case.hf_id)
        model_path = model_root / _MODEL_DIRECTORY
        vit_path = model_root / _VIT_DIRECTORY
        if not model_path.is_dir() or not vit_path.is_dir():
            raise RuntimeError(
                "Lance reference checkpoint must contain "
                f"{_MODEL_DIRECTORY}/ and {_VIT_DIRECTORY}/ under {model_root}"
            )

        artifact_dir = Path(
            _case_artifact_dir(
                ctx.artifacts_dir or tempfile.gettempdir(),
                case.name,
            )
        )
        request_path = artifact_dir / "lance_x2t_request.json"
        result_dir = artifact_dir / "official_output"
        request_path.write_text(
            json.dumps(
                {
                    "000000": {
                        "interleave_array": [
                            str(image),
                            ["", prompt, ""],
                        ],
                        "element_dtype_array": ["image", "text"],
                        "istarget_in_interleave": [0, 1],
                    }
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        environment = _image_reference_environment()
        command = [
            "env",
            f"PYTHONPATH={environment['PYTHONPATH']}",
            ctx.reference_python_path() or sys.executable,
            str(source / _REFERENCE_ENTRYPOINT),
            "--model_path",
            str(model_path),
            "--llm_path",
            str(model_path),
            "--vit_path",
            str(vit_path),
            "--vit_type",
            "qwen_2_5_vl_original",
            "--llm_qk_norm",
            "true",
            "--llm_qk_norm_und",
            "true",
            "--llm_qk_norm_gen",
            "true",
            "--tie_word_embeddings",
            "false",
            "--copy_init_moe",
            "true",
            "--max_num_frames",
            "121",
            "--max_latent_size",
            "64",
            "--latent_patch_size",
            "1",
            "1",
            "1",
            "--visual_und",
            "true",
            "--visual_gen",
            "true",
            "--vae_model_type",
            "wan",
            "--apply_qwen_2_5_vl_pos_emb",
            "true",
            "--apply_chat_template",
            "false",
            "--cfg_type",
            "0",
            "--validation_data_seed",
            str(case.determinism.get("seed", 42)),
            "--task",
            "x2t_image",
            "--save_path_gen",
            str(result_dir),
            "--val_dataset_config_file",
            str(request_path),
            "--text_template",
            "false",
            "--use_KVcache",
            "true",
            "--enhance_prompt",
            "false",
        ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=source,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.monotonic() - started
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir or tempfile.gettempdir(),
            "lance_official_reference",
            case.name,
        )
        if result.returncode != 0:
            detail = stderr.strip() or (result.stdout or "").strip()
            raise RuntimeError(
                "Lance official reference failed "
                f"(rc={result.returncode}): {detail or 'no subprocess output'}"
            )
        text = _result_text(result_dir / "result.json")
        return StageOutput(
            stage_name=stage.name,
            data={"text": text, "result_path": str(result_dir / "result.json")},
            text=text,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": result.returncode,
                "stderr": stderr,
                "stderr_log": stderr_log,
                "source_revision": "4baeee086648996f6ab12e673cbe461b0b149997",
            },
        )


plugin = LanceOfficialReference()

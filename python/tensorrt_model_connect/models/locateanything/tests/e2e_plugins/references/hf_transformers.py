# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference backend — gold-standard L1 oracle.

Runs HF model inference in a subprocess for GPU memory isolation and returns
per-step logits + generated text for comparison against TRT outputs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_MODEL_TEST_DIR = Path(__file__).resolve().parents[2]
_MODEL_TEST_RELATIVE = (
    Path("python/tensorrt_model_connect/models")
    / _MODEL_TEST_DIR.parent.name
    / "tests"
)


_PRECISION_TO_TORCH_DTYPE = {
    "fp16": "torch.float16",
    "fp32": "torch.float32",
    "bf16": "torch.bfloat16",
}


def _torch_dtype_for_case(case: E2ECase) -> str:
    """Return the explicit reference dtype, falling back to DUT precision.

    FP16 acceptance manifests set reference_precision=fp32 so changing the
    engine precision does not also change the oracle.
    """
    precision = case.metadata.get(
        "reference_precision", case.metadata.get("precision", "fp32"))
    return _PRECISION_TO_TORCH_DTYPE.get(precision, "torch.float32")


def _vl_prompt_has_image_placeholder(text: str) -> bool:
    """Return true when a rendered VL prompt still carries an image placeholder."""
    return any(marker in text for marker in (
        "<|image_pad|>",
        "<|vision_start|>",
        "<image>",
        "<IMG_CONTEXT>",
    ))


def _normalize_vl_prompt_guard(text: str) -> str:
    """Normalize decoded VL text for prompt-only reference detection."""
    normalized = " ".join(str(text or "").split()).strip().lower()
    for marker in (
        "<img_context>",
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ):
        normalized = normalized.replace(marker, " ")
    return " ".join(normalized.split()).strip()


def _is_prompt_only_vl_text(text: str, prompt_texts: tuple[str, ...]) -> bool:
    """Return true when decoded VL text contains only the input prompt/template."""
    normalized_text = _normalize_vl_prompt_guard(text)
    if not normalized_text:
        return True

    for prompt_text in prompt_texts:
        normalized_prompt = _normalize_vl_prompt_guard(prompt_text)
        if not normalized_prompt:
            continue
        if normalized_text == normalized_prompt:
            return True
        if normalized_text.startswith(normalized_prompt):
            tail = normalized_text[len(normalized_prompt):].strip(" :")
            if tail in {"", "assistant", "answer"}:
                return True
        if normalized_text.endswith(normalized_prompt):
            return True
    return False


def _decode_vl_generated_text(
    processor,
    generated_ids,
    input_len: int,
    prompt_texts: tuple[str, ...] = (),
) -> str:
    """Decode VL generation whether generate() returns full ids or generated ids only."""
    token_count = len(generated_ids)

    def _decode_token_ids(token_ids) -> str:
        return processor.decode(token_ids, skip_special_tokens=True).strip()

    if input_len > 0 and token_count > input_len:
        text = _decode_token_ids(generated_ids[input_len:])
        if text and not _is_prompt_only_vl_text(text, prompt_texts):
            return text

    text = _decode_token_ids(generated_ids)
    if text and not _is_prompt_only_vl_text(text, prompt_texts):
        return text
    return ""


def _resolve_cached_model_ref(hf_id: str) -> str:
    """Prefer the selectively cached Model Connect snapshot."""
    if not hf_id:
        return hf_id
    p = Path(hf_id)
    if p.exists():
        return hf_id

    try:
        from huggingface_hub import snapshot_download
        from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

        return snapshot_download(
            hf_id,
            allow_patterns=hf_snapshot_allow_patterns(),
            local_files_only=True,
        )
    except Exception:
        return hf_id


ReferenceOutputReader = Callable[[], dict[str, Any]]


def _coerce_stream_text(stream: object) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return str(stream)


def _read_text_artifact(path: str, *, encoding: str = "utf-8") -> str:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return ""
    return artifact_path.read_text(encoding=encoding)


def _json_output_reader(path: str, *, encoding: str = "utf-8") -> ReferenceOutputReader:
    def _reader() -> dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            return {}
        return json.loads(artifact_path.read_text(encoding=encoding))

    return _reader


def _json_text_reader(
    path: str, key: str = "text", *, encoding: str = "utf-8"
) -> Callable[[], str]:
    def _reader() -> str:
        data = _json_output_reader(path, encoding=encoding)()
        value = data.get(key, "")
        return "" if value is None else str(value)

    return _reader


def _npy_output_reader(
    path: str,
    data_key: str,
    *,
    path_key: str = "",
    allow_pickle: bool = False,
) -> ReferenceOutputReader:
    def _reader() -> dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            return {}
        import numpy as np

        data: dict[str, Any] = {}
        if path_key:
            data[path_key] = path
        data[data_key] = np.load(artifact_path, allow_pickle=allow_pickle)
        return data

    return _reader


def _existing_path_reader(path: str, data_key: str) -> ReferenceOutputReader:
    def _reader() -> dict[str, Any]:
        return {data_key: path} if Path(path).is_file() else {}

    return _reader


def _reference_env(ctx: RunContext) -> dict[str, str]:
    env = dict(os.environ)
    if ctx.ld_library_path:
        env["LD_LIBRARY_PATH"] = ctx.ld_library_path
    return env


def run_reference_subprocess(
    *,
    command: Sequence[str],
    timeout_s: float,
    label: str,
    artifact_dir: str,
    case_name: str,
    stage_name: str,
    env: Mapping[str, str] | None = None,
    output_readers: Iterable[ReferenceOutputReader] = (),
    text_reader: Callable[[], str] | None = None,
    logits_reader: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_stdio_metadata: bool = False,
    failure_label: str | None = None,
) -> StageOutput:
    """Run a reference subprocess and build the matching StageOutput."""
    failure_prefix = failure_label or label.replace("_", " ")
    cmd = list(command)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        stderr = _coerce_stream_text(exc.stderr)
        truncated, log_path = save_full_stderr(
            stderr, artifact_dir, label, case_name
        )
        msg = f"{failure_prefix} timed out for {case_name} after {elapsed:.0f}s"
        if truncated:
            msg += f":\n{truncated}"
        if log_path:
            msg += f" (full stderr: {log_path})"
        raise RuntimeError(msg) from exc
    except Exception as exc:
        raise RuntimeError(f"{failure_prefix} failed for {case_name}: {exc}") from exc
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        truncated, log_path = save_full_stderr(
            result.stderr or "", artifact_dir, label, case_name
        )
        msg = (
            f"{failure_prefix} failed for {case_name} "
            f"(rc={result.returncode}):\n{truncated}"
        )
        if log_path:
            msg += f" (full stderr: {log_path})"
        raise RuntimeError(msg)

    data: dict[str, Any] = {}
    for reader in output_readers:
        data.update(reader() or {})

    output_metadata: dict[str, Any] = {"returncode": result.returncode}
    if include_stdio_metadata:
        output_metadata.update({"stdout": result.stdout, "stderr": result.stderr})
    if metadata:
        output_metadata.update(dict(metadata))

    return StageOutput(
        stage_name=stage_name,
        data=data,
        text=text_reader() if text_reader is not None else None,
        logits=logits_reader() if logits_reader is not None else None,
        timing_s=elapsed,
        metadata=output_metadata,
    )


class HfTransformersReference:
    """Run HuggingFace Transformers inference as the reference oracle."""

    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name == "full_generation":
            return self._run_full_generation(case, stage, ctx)
        if stage.name == "full_inference":
            return self._run_full_inference(case, stage, ctx)
        if stage.name == "vision_encode":
            # Vision encode is TRT-side only; reference skips this stage
            return StageOutput(
                stage_name=stage.name,
                data={"skipped": True},
                metadata={"reason": "vision_encode handled by TRT runner only"},
            )
        raise ValueError(f"Unknown stage for hf_transformers: {stage.name!r}")

    def _run_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model inference in a subprocess, collecting per-step logits.

        Dispatches to task-specific methods for non-standard tasks:
        - vision_language_generation -> _run_vl_full_generation()
        """
        task = case.task_strategy
        if task == "vision_language_generation":
            return self._run_vl_full_generation(case, stage, ctx)

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        logits_path = str(Path(model_dir) / "hf_logits.npy")
        text_path = str(Path(model_dir) / "hf_text.txt")

        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        torch_dtype_expr = _torch_dtype_for_case(case)

        contract_config = case.metadata.get("contract_config", {})
        use_chat_template = contract_config.get("use_chat_template", False)
        enable_thinking = contract_config.get("enable_thinking", True)

        script = textwrap.dedent(f"""\
            import sys, numpy as np, torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            logits_path = {logits_path!r}
            text_path = {text_path!r}
            use_chat_template = {use_chat_template!r}
            enable_thinking = {enable_thinking!r}

            def _np(t):
                return t.detach().float().cpu().numpy()

            tokenizer = AutoTokenizer.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code)
            if use_chat_template:
                messages = [{{"role": "user", "content": prompt}}]
                try:
                    chat_kwargs = {{"add_generation_prompt": True}}
                    if not enable_thinking:
                        chat_kwargs["enable_thinking"] = False
                    text_input = tokenizer.apply_chat_template(
                        messages, tokenize=False, **chat_kwargs)
                    input_ids = tokenizer.encode(text_input, add_special_tokens=False)
                except Exception:
                    # Fallback: model doesn't support chat template
                    input_ids = tokenizer.encode(prompt)
            else:
                input_ids = tokenizer.encode(prompt)

            load_kwargs = {{
                "trust_remote_code": trust_remote_code,
                "torch_dtype": {torch_dtype_expr},
            }}
            # Detect encoder-decoder models by checking config
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
            is_seq2seq = getattr(_cfg, "is_encoder_decoder", False)

            if is_seq2seq:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_ref, **load_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
            model.eval()

            ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            all_logits = []

            with torch.no_grad():
                if is_seq2seq:
                    # Encoder-decoder: use model.generate() for greedy decoding
                    output_ids = model.generate(
                        ids_tensor, max_new_tokens=max_new_tokens,
                        do_sample=False, num_beams=1)
                    generated_token_ids = output_ids[0].tolist()
                    # Re-run to get logits for each decoder step
                    decoder_ids = torch.tensor([generated_token_ids], dtype=torch.long)
                    outputs = model(ids_tensor, decoder_input_ids=decoder_ids)
                    for i in range(outputs.logits.shape[1]):
                        all_logits.append(_np(outputs.logits[0, i]))
                    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                else:
                    # Decoder-only: step-by-step autoregressive
                    outputs = model(ids_tensor)
                    prefill_logits = _np(outputs.logits[0])
                    for i in range(len(input_ids)):
                        all_logits.append(prefill_logits[i])

                    gen_ids = list(input_ids)
                    generated_token_ids = []
                    eos_id = getattr(tokenizer, "eos_token_id", None)
                    for _ in range(max_new_tokens):
                        next_token = int(np.argmax(all_logits[-1]))
                        generated_token_ids.append(next_token)
                        if eos_id is not None and next_token == eos_id:
                            break
                        gen_ids.append(next_token)
                        ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
                        outputs = model(ids_tensor)
                        all_logits.append(_np(outputs.logits[0, -1]))
                    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            with open(text_path, "w") as f:
                f.write(text)

            # Pad and save logits
            max_len = max(l.shape[0] for l in all_logits)
            padded = np.zeros((len(all_logits), max_len), dtype=np.float32)
            for i, l in enumerate(all_logits):
                padded[i, :l.shape[0]] = l
            np.save(logits_path, padded)

            print(f"OK steps={{len(all_logits)}} vocab={{max_len}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        logger.info("HF reference: running %s", case.name)
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=1800,
            label="hf_full_generation",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_existing_path_reader(logits_path, "logits_path"),),
            text_reader=lambda: _read_text_artifact(text_path),
            logits_reader=(
                lambda: logits_path if Path(logits_path).is_file() else None
            ),
            metadata={"trust_remote_code": trust_remote_code},
            include_stdio_metadata=True,
            failure_label="HF reference",
        )

    def _run_full_inference(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model forward pass for non-generative tasks.

        Dispatches based on task_strategy to the appropriate HF Auto class.
        """
        task = case.task_strategy
        if task == "encoder_only_nlp":
            return self._run_encoder_only(case, stage, ctx)
        if task == "segmentation":
            return self._run_segmentation_ref(case, stage, ctx)
        if task == "embedding":
            return self._run_embedding_ref(case, stage, ctx)
        if task == "reranking":
            return self._run_reranking_ref(case, stage, ctx)
        if task == "object_detection":
            return self._run_object_detection_ref(case, stage, ctx)
        raise ValueError(
            f"full_inference not implemented for task_strategy={task!r}")

    def _run_encoder_only(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF encoder-only model (e.g. BERT) and return CLS embedding."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_encoder.json")

        prompt = case.inputs.get("prompt", "Hello world")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModel, AutoTokenizer

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # CLS token embedding from last_hidden_state
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                cls_embedding = outputs.last_hidden_state[0, 0].float().cpu().numpy().tolist()
            else:
                first_out = outputs[0]
                if first_out.ndim == 3:
                    cls_embedding = first_out[0, 0].float().cpu().numpy().tolist()
                elif first_out.ndim == 2:
                    cls_embedding = first_out[0].float().cpu().numpy().tolist()
                else:
                    cls_embedding = first_out.float().cpu().numpy().tolist()
            result = {{"cls_embedding": cls_embedding}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=600,
            label="hf_encoder_only",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_json_output_reader(output_path),),
            failure_label="HF encoder-only",
        )

    @staticmethod
    def _resolve_image_path(image_path: str) -> str:
        """Resolve image path, handling relative paths from manifests."""
        if not image_path:
            return image_path
        path = Path(image_path)
        if path.is_absolute():
            resolved = path
        else:
            try:
                path = path.relative_to(_MODEL_TEST_RELATIVE)
            except ValueError:
                pass
            if ".." in path.parts:
                raise FileNotFoundError(
                    "Model-owned image asset not found: " + str(path)
                )
            resolved = _MODEL_TEST_DIR / path
        if not resolved.is_file():
            raise FileNotFoundError(
                "Model-owned image asset not found: " + str(resolved)
            )
        return str(resolved)

    def _run_embedding_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF embedding model as reference — mean pool + L2 normalize."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_embedding.json")

        prompt = case.inputs.get("prompt", "What is machine learning?")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModel, AutoTokenizer

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            # Generic forward pass: tokenize -> forward -> mean pool -> L2 norm
            # (We use the raw forward pass to match TRT, not encode_queries()
            # which adds an instruction prefix that TRT doesn't replicate.)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                               truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Try last_hidden_state first, then fall back to hidden_states[-1]
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
            else:
                raise RuntimeError("Model output has no hidden states")
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embedding = pooled[0].float().cpu().numpy().tolist()

            result = {{"embedding": embedding}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=600,
            label="hf_embedding",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_json_output_reader(output_path),),
            failure_label="HF embedding ref",
        )

    def _run_segmentation_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF segmentation model as reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_seg.npy")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import numpy as np, torch
            from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoImageProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForSemanticSegmentation.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0].float().cpu().numpy()
            class_map = np.argmax(logits, axis=0).astype(np.int32)
            np.save(output_path, class_map)
            print(f"OK classes={{class_map.max() + 1}}")
        """)

        def _segmentation_outputs() -> dict[str, Any]:
            data: dict[str, Any] = {}
            if Path(output_path).is_file():
                data["class_map_path"] = output_path
                import numpy as np

                data["class_map"] = np.load(output_path)

                try:
                    from PIL import Image

                    cmap = data["class_map"]
                    num_classes = int(cmap.max()) + 1
                    np.random.seed(42)
                    palette = np.random.randint(
                        0, 255, (num_classes, 3), dtype=np.uint8
                    )
                    palette[0] = [0, 0, 0]
                    colored = palette[cmap]
                    viz_path = output_path.replace(".npy", "_viz.png")
                    Image.fromarray(colored).save(viz_path)
                    data["viz_path"] = viz_path
                except Exception as e:
                    logger.warning("Failed to save segmentation viz: %s", e)
            return data

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=600,
            label="hf_segmentation",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_segmentation_outputs,),
            failure_label="HF segmentation",
        )

    def _run_reranking_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF cross-encoder reranking and return one score per document."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_rerank.json")

        prompt = case.inputs.get("prompt", "query: test")
        documents = case.inputs.get("documents")
        if documents is None:
            document = case.inputs.get("document", "")
            documents = [document] if document else []
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch
            from transformers import AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            documents = {documents!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}
            torch_dtype = {torch_dtype_expr}

            model = AutoModelForSequenceClassification.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            examples = [
                {{"question": prompt, "doc_text": doc, "doc_image": ""}}
                for doc in documents
            ]

            try:
                processor = AutoProcessor.from_pretrained(
                    model_ref, trust_remote_code=trust_remote_code,
                    max_input_tiles=6, use_thumbnail=True,
                    rerank_max_length=8192)
                if not hasattr(processor, "process_queries_documents_crossencoder"):
                    raise AttributeError("processor has no cross-encoder helper")
                inputs = processor.process_queries_documents_crossencoder(examples)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_ref, trust_remote_code=trust_remote_code)
                texts = [
                    f"question:{{prompt}}   passage:{{doc}}"
                    for doc in documents
                ]
                inputs = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True)

            inputs = {{
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.detach().float().cpu()
            if logits.ndim == 2 and logits.shape[-1] == 1:
                scores = logits[:, 0].tolist()
            elif logits.ndim == 2:
                scores = logits[:, -1].tolist()
            else:
                scores = logits.reshape(-1).tolist()
            result = {{"scores": scores}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=600,
            label="hf_reranking",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_json_output_reader(output_path),),
            failure_label="HF reranking",
        )

    def _run_object_detection_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF object detection model as reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_det.json")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoImageProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForObjectDetection.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Post-process to get boxes + scores
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5)[0]
            detections = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                detections.append({{
                    "score": score.item(),
                    "label": label.item(),
                    "box": box.tolist(),
                }})
            with open(output_path, "w") as f:
                json.dump({{"detections": detections}}, f)
            print(f"OK detections={{len(detections)}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=600,
            label="hf_object_detection",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(_json_output_reader(output_path),),
            failure_label="HF object detection",
        )

    def _run_vl_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run LocateAnything vision-language reference generation."""
        return self._run_locateanything_vl_full_generation(case, stage, ctx)
    def _run_locateanything_vl_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run LocateAnything HF reference without the optional remote processor."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        text_path = str(Path(model_dir) / "hf_vl_text.txt")

        prompt = case.inputs.get("prompt", "Describe this image.")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json
            import sys
            from pathlib import Path

            import torch
            from tensorrt_model_connect.models.locateanything.vl_debug_runner import (
                preprocess_image_inputs_for_trt,
            )
            from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig

            if torch.cuda.is_available():
                # Avoid Thor PyTorch/cuDNN sublibrary mismatches in HF reference conv2d.
                torch.backends.cudnn.enabled = False

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            image_path = {image_path!r}
            text_path = {text_path!r}

            def _install_tied_weight_compat():
                from transformers.cache_utils import DynamicCache
                from transformers.modeling_utils import PreTrainedModel

                if not hasattr(DynamicCache, "to_legacy_cache"):
                    def _to_legacy_cache(self):
                        return tuple(
                            (layer.keys, layer.values) for layer in self.layers
                        )

                    DynamicCache.to_legacy_cache = _to_legacy_cache

                if not hasattr(DynamicCache, "from_legacy_cache"):
                    @classmethod
                    def _from_legacy_cache(cls, past_key_values=None):
                        cache = cls()
                        if past_key_values is None:
                            return cache
                        for layer_idx, (key_states, value_states) in enumerate(
                            past_key_values
                        ):
                            cache.update(key_states, value_states, layer_idx)
                        return cache

                    DynamicCache.from_legacy_cache = _from_legacy_cache

                if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                    PreTrainedModel.all_tied_weights_keys = {{}}
                original = PreTrainedModel.get_expanded_tied_weights_keys

                def _compat_get_expanded_tied_weights_keys(self, all_submodels=False):
                    tied_mapping = getattr(self, "_tied_weights_keys", None)
                    if not isinstance(tied_mapping, list):
                        return original(self, all_submodels=all_submodels)
                    if all_submodels:
                        expanded_tied_weights = {{}}
                        for prefix, submodule in self.named_modules(
                            remove_duplicate=False
                        ):
                            if isinstance(submodule, PreTrainedModel):
                                submodel_tied_weights = (
                                    submodule.get_expanded_tied_weights_keys(
                                        all_submodels=False
                                    )
                                )
                                if prefix:
                                    submodel_tied_weights = {{
                                        f"{{prefix}}.{{key}}": f"{{prefix}}.{{value}}"
                                        for key, value in submodel_tied_weights.items()
                                    }}
                                expanded_tied_weights.update(submodel_tied_weights)
                        return expanded_tied_weights
                    if not getattr(self.config, "tie_word_embeddings", False):
                        return {{}}
                    if "lm_head.weight" in tied_mapping:
                        return {{"lm_head.weight": "model.embed_tokens.weight"}}
                    return {{}}

                PreTrainedModel.get_expanded_tied_weights_keys = (
                    _compat_get_expanded_tied_weights_keys
                )

            def _load_locateanything_raw_config(model_ref):
                raw_config_path = Path(model_ref) / "config.json"
                if raw_config_path.is_file():
                    return json.loads(raw_config_path.read_text(encoding="utf-8"))
                try:
                    raw_config, _ = PretrainedConfig.get_config_dict(
                        model_ref, trust_remote_code=trust_remote_code)
                except Exception:
                    return {{}}
                return raw_config if isinstance(raw_config, dict) else {{}}

            def _load_locateanything_config(model_ref):
                config = AutoConfig.from_pretrained(
                    model_ref, trust_remote_code=trust_remote_code)
                raw_config = _load_locateanything_raw_config(model_ref)
                if hasattr(config, "text_config") and not hasattr(
                    config.text_config, "rope_theta"
                ):
                    raw_text_config = raw_config.get("text_config", {{}})
                    if not isinstance(raw_text_config, dict):
                        raw_text_config = {{}}
                    rope_theta = raw_text_config.get("rope_theta")
                    if rope_theta is None:
                        rope_parameters = raw_text_config.get("rope_parameters", {{}})
                        if isinstance(rope_parameters, dict):
                            rope_theta = rope_parameters.get("rope_theta")
                    if rope_theta is None:
                        rope_scaling = raw_text_config.get("rope_scaling", {{}})
                        if isinstance(rope_scaling, dict):
                            rope_theta = rope_scaling.get("rope_theta")
                    if rope_theta is None:
                        rope_theta = raw_config.get("rope_theta", 10000.0)
                    config.text_config.rope_theta = float(rope_theta)
                return config

            def _load_locateanything_tokenizer(model_ref):
                try:
                    return AutoTokenizer.from_pretrained(
                        model_ref, trust_remote_code=trust_remote_code)
                except Exception as auto_exc:
                    from huggingface_hub import hf_hub_download
                    from tokenizers import Tokenizer

                    model_path = Path(model_ref)
                    if model_path.is_dir():
                        tokenizer_json = model_path / "tokenizer.json"
                        tokenizer_config_path = model_path / "tokenizer_config.json"
                    else:
                        tokenizer_json = Path(
                            hf_hub_download(model_ref, "tokenizer.json"))
                        try:
                            tokenizer_config_path = Path(
                                hf_hub_download(model_ref, "tokenizer_config.json"))
                        except Exception:
                            tokenizer_config_path = Path()
                    raw_tokenizer = Tokenizer.from_file(str(tokenizer_json))
                    tokenizer_config = {{}}
                    if tokenizer_config_path.is_file():
                        tokenizer_config = json.loads(
                            tokenizer_config_path.read_text(encoding="utf-8"))
                    model_max_length = int(
                        tokenizer_config.get("model_max_length", 16384))

                    class TokenizersWrapper:
                        def __init__(self, tok, max_length):
                            self._tok = tok
                            self.model_max_length = max_length

                        def encode(self, text, add_special_tokens=False):
                            return self._tok.encode(
                                text,
                                add_special_tokens=add_special_tokens).ids

                        def __call__(self, text, return_tensors=None):
                            ids = self.encode(text, add_special_tokens=False)
                            if return_tensors == "pt":
                                input_ids = torch.tensor([ids], dtype=torch.long)
                                attention_mask = torch.ones_like(input_ids)
                                return {{
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                }}
                            return {{"input_ids": ids}}

                        def decode(self, ids, skip_special_tokens=True):
                            if torch.is_tensor(ids):
                                ids = ids.detach().cpu().tolist()
                            if isinstance(ids, int):
                                ids = [ids]
                            return self._tok.decode(
                                [int(token) for token in ids],
                                skip_special_tokens=skip_special_tokens)

                        def batch_decode(self, batch_ids, skip_special_tokens=True):
                            return [
                                self.decode(ids, skip_special_tokens=skip_special_tokens)
                                for ids in batch_ids
                            ]

                    print(
                        f"WARN AutoTokenizer failed ({{auto_exc}}); "
                        "using tokenizer.json fallback",
                        file=sys.stderr,
                    )
                    return TokenizersWrapper(raw_tokenizer, model_max_length)

            def _repair_locateanything_rotary_buffers(model):
                # Restore non-persistent RoPE buffers zeroed by remote loading.
                repaired = 0
                model_device = next(model.parameters()).device
                for module in model.language_model.modules():
                    if not (
                        hasattr(module, "_set_cos_sin_cache")
                        and hasattr(module, "base")
                        and hasattr(module, "dim")
                    ):
                        continue
                    buffer_device = getattr(
                        getattr(module, "inv_freq", None), "device", model_device)
                    inv_freq = 1.0 / (
                        float(module.base) ** (
                            torch.arange(
                                0,
                                int(module.dim),
                                2,
                                device=buffer_device,
                                dtype=torch.float32,
                            ) / float(module.dim)
                        )
                    )
                    module.register_buffer("inv_freq", inv_freq, persistent=False)
                    seq_len = int(
                        getattr(
                            module,
                            "max_position_embeddings",
                            getattr(model.config.text_config, "max_position_embeddings", 32768),
                        )
                    )
                    module._set_cos_sin_cache(
                        seq_len=seq_len, device=inv_freq.device, dtype=torch.float32)
                    repaired += 1
                if repaired == 0:
                    raise RuntimeError("LocateAnything reference did not find RoPE buffers")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _install_tied_weight_compat()
            config = _load_locateanything_config(model_ref)
            tokenizer = _load_locateanything_tokenizer(model_ref)
            model = AutoModel.from_pretrained(
                model_ref, config=config, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.to(device)
            model.eval()
            _repair_locateanything_rotary_buffers(model)

            image_inputs = preprocess_image_inputs_for_trt(
                image_path,
                preprocessor_type="patchify_chw",
                fixed_image_size=448,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
                patch_size=14,
                interpolation="bicubic",
            )
            pixel_values = torch.from_numpy(image_inputs["pixel_values"]).to(device)
            image_grid_hws = torch.from_numpy(
                image_inputs["image_grid_hws"]).to(device=device, dtype=torch.int32)

            image_pads = "<IMG_CONTEXT>" * 256
            prompt_text = (
                "<|im_start|>system\\n"
                "You are a helpful assistant.<|im_end|>\\n"
                "<|im_start|>user\\n"
                f"<img>{{image_pads}}</img>{{prompt}}<|im_end|>\\n"
                "<|im_start|>assistant\\n"
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            generate_kwargs = {{
                "pixel_values": pixel_values,
                "image_grid_hws": image_grid_hws,
                "input_ids": input_ids,
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "generation_mode": "slow",
                "do_sample": False,
            }}
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                output = model.generate(**generate_kwargs)

            if isinstance(output, str):
                text = output
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
                text = output[0]
            else:
                input_len = input_ids.shape[-1]
                generated_ids = output[0][input_len:] if output.ndim > 1 else output[input_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if not text.strip():
                raise RuntimeError("HF LocateAnything reference produced empty text")

            with open(text_path, "w") as f:
                f.write(text)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=1800,
            label="hf_vl_generation",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(lambda: {"text": _read_text_artifact(text_path)},),
            text_reader=lambda: _read_text_artifact(text_path),
            metadata={
                "trust_remote_code": trust_remote_code,
                "reference_variant": "locateanything_manual_processor",
            },
            failure_label="HF VL generation",
        )


plugin = HfTransformersReference()

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
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


_PRECISION_TO_TORCH_DTYPE = {
    "fp16": "torch.float16",
    "fp32": "torch.float32",
    "bf16": "torch.bfloat16",
}


def _torch_dtype_for_case(case: E2ECase) -> str:
    """Return a torch dtype expression string matching the manifest precision.

    The reference runner injects this into subprocess scripts so the HF model
    loads at the same precision as the TRT engine, keeping comparisons fair.
    """
    precision = case.metadata.get("precision", "fp32")
    return _PRECISION_TO_TORCH_DTYPE.get(precision, "torch.float32")


def _reference_generation_mode(case: E2ECase) -> str:
    """Return the text-generation reference mode requested by a contract plugin."""
    contract_config = case.metadata.get("contract_config", {})
    if not isinstance(contract_config, dict):
        return ""
    return str(contract_config.get("reference_generation_mode", "") or "")


def _vl_fallback_prompt(hf_id: str, prompt: str) -> str:
    """Return a model-family prompt that preserves one image placeholder."""
    lower_id = hf_id.lower()
    if "qwen" in lower_id and "vl" in lower_id:
        return f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
    if "internvl" in lower_id:
        return f"<IMG_CONTEXT>\n{prompt}"
    return prompt


def _decode_vl_generated_text(processor, generated_ids, input_len: int) -> str:
    """Decode VL generation whether generate() returns full ids or generated ids only."""
    token_count = len(generated_ids)

    def _decode_token_ids(token_ids) -> str:
        return processor.decode(token_ids, skip_special_tokens=True).strip()

    if input_len > 0 and token_count > input_len:
        text = _decode_token_ids(generated_ids[input_len:])
        if text:
            return text
    return _decode_token_ids(generated_ids)


def _resolve_cached_model_ref(hf_id: str) -> str:
    """Prefer a locally cached HF snapshot to avoid Hub API rate limits."""
    if not hf_id:
        return hf_id
    p = Path(hf_id)
    if p.exists():
        return hf_id

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(hf_id, local_files_only=True)
    except Exception:
        return hf_id


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
        - text_to_audio -> _run_text_to_audio_ref()
        - vision_language_generation -> _run_vl_full_generation()
        - speech_to_text -> _run_speech_to_text_ref() (via full_inference)
        """
        task = case.task_strategy
        if task == "text_to_audio":
            return self._run_text_to_audio_ref(case, stage, ctx)
        if task == "vision_language_generation":
            return self._run_vl_full_generation(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)

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
        reference_generation_mode = _reference_generation_mode(case)

        script = textwrap.dedent(f"""\
            import sys, numpy as np, torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
            from tensorrt_model_connect.transformers_compat import (
                patch_legacy_dynamic_cache_api,
                remap_token_ids_to_model_vocab,
            )

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            logits_path = {logits_path!r}
            text_path = {text_path!r}
            use_chat_template = {use_chat_template!r}
            enable_thinking = {enable_thinking!r}
            reference_generation_mode = {reference_generation_mode!r}

            def _np(t):
                return t.detach().float().cpu().numpy()

            if trust_remote_code:
                patch_legacy_dynamic_cache_api()

            _cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
            is_seq2seq = getattr(_cfg, "is_encoder_decoder", False)

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
            input_ids = remap_token_ids_to_model_vocab(
                tokenizer, model_ref, input_ids, getattr(_cfg, "vocab_size", None))

            load_kwargs = {{
                "trust_remote_code": trust_remote_code,
                "torch_dtype": {torch_dtype_expr},
            }}

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
                    if reference_generation_mode == "hf_generate":
                        attention_mask = torch.ones_like(ids_tensor)
                        output_ids = model.generate(
                            ids_tensor,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=getattr(tokenizer, "eos_token_id", None),
                        )
                        full_ids = output_ids[0].tolist()
                        if len(full_ids) >= len(input_ids):
                            generated_token_ids = full_ids[len(input_ids):]
                        else:
                            generated_token_ids = full_ids
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

            if all_logits:
                # Pad and save logits
                max_len = max(l.shape[0] for l in all_logits)
                padded = np.zeros((len(all_logits), max_len), dtype=np.float32)
                for i, l in enumerate(all_logits):
                    padded[i, :l.shape[0]] = l
                np.save(logits_path, padded)
                print(f"OK steps={{len(all_logits)}} vocab={{max_len}}")
            else:
                vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
                if vocab_size is None:
                    vocab_size = len(tokenizer)
                print(f"OK generated_steps={{len(generated_token_ids)}} vocab={{vocab_size}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        logger.info("HF reference: running %s", case.name)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True, text=True, timeout=1800,
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            raise RuntimeError(
                f"HF reference timed out for {case.name} after {elapsed:.0f}s"
            )
        except Exception as e:
            raise RuntimeError(f"HF reference failed for {case.name}: {e}") from e
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_full_generation", case.name)
            msg = f"HF reference failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        # Read generated text
        text = ""
        if Path(text_path).is_file():
            text = Path(text_path).read_text(encoding="utf-8")

        data = {}
        if Path(logits_path).is_file():
            data["logits_path"] = logits_path

        meta = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "trust_remote_code": trust_remote_code,
        }

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=text,
            logits=logits_path if Path(logits_path).is_file() else None,
            timing_s=elapsed,
            metadata=meta,
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
        if task == "prompted_segmentation":
            return self._run_prompted_segmentation_ref(case, stage, ctx)
        if task == "embedding":
            return self._run_embedding_ref(case, stage, ctx)
        if task == "reranking":
            return self._run_reranking_ref(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)
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

            # Load model — try AutoModel first, fall back to base model
            # for specialized wrappers (DPR, etc.) that don't return
            # last_hidden_state in the expected format.
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code)
            model_type = getattr(config, 'model_type', '')

            if model_type == 'dpr':
                # DPR wraps BERT under ctx_encoder.bert_model or
                # question_encoder.bert_model prefix.  AutoModel loads
                # the wrong class (DPRQuestionEncoder) with missing weights.
                # Load as DPRContextEncoder and extract the inner BERT.
                from transformers import DPRContextEncoder
                _dpr = DPRContextEncoder.from_pretrained(
                    model_ref, trust_remote_code=trust_remote_code,
                    torch_dtype={torch_dtype_expr})
                model = _dpr.ctx_encoder.bert_model
            else:
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
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_encoder_only", case.name)
            msg = f"HF encoder-only failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    @staticmethod
    def _resolve_image_path(image_path: str) -> str:
        """Resolve image path, handling relative paths from manifests."""
        if not image_path:
            return image_path
        if os.path.isabs(image_path):
            return image_path
        # Resolve relative to tests/e2e/ directory
        e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
        resolved = e2e_dir / image_path
        if resolved.exists():
            return str(resolved)
        # Also try relative to project root
        project_dir = Path(__file__).resolve().parents[3]
        resolved2 = project_dir / image_path
        if resolved2.exists():
            return str(resolved2)
        return image_path

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
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_embedding", case.name)
            msg = (f"HF embedding ref failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

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

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_segmentation", case.name)
            msg = f"HF segmentation failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data["class_map_path"] = output_path
            import numpy as np
            data["class_map"] = np.load(output_path)

            # Save colorized PNG for human inspection
            try:
                from PIL import Image
                cmap = data["class_map"]
                num_classes = int(cmap.max()) + 1
                # Simple colormap: class index -> hue
                h, w = cmap.shape
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(num_classes):
                    mask = cmap == c
                    # Distribute hues evenly across classes
                    hue = int(255 * c / max(num_classes, 1))
                    rgb[mask] = [hue, 180, 200 if c > 0 else 40]
                # Convert HSV-like to simple distinguishable colors
                np.random.seed(42)
                palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
                palette[0] = [0, 0, 0]  # background black
                colored = palette[cmap]
                viz_path = output_path.replace(".npy", "_viz.png")
                Image.fromarray(colored).save(viz_path)
                data["viz_path"] = viz_path
            except Exception as e:
                logger.warning("Failed to save segmentation viz: %s", e)

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

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
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_reranking", case.name)
            msg = f"HF reranking failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_speech_to_text_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run speech-to-text reference (Whisper via HF, NeMo ASR via NeMo)."""
        family = case.metadata.get("family", case.family)
        if family in {"canary", "nemotron_speech_streaming"}:
            return self._run_canary_ref(case, stage, ctx)

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_stt.json")

        audio_path = self._resolve_image_path(case.inputs.get("audio", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import scipy.io.wavfile as wav

            hf_id = {hf_id!r}
            audio_path = {audio_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            sr, audio = wav.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to model's expected sample rate (e.g. 16kHz for Whisper)
            target_sr = getattr(processor.feature_extractor, "sampling_rate", sr)
            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(audio) * target_sr / sr)
                audio = resample(audio, num_samples).astype(np.float32)
                sr = target_sr

            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            # Cast floating-point inputs to match model dtype (e.g. fp16 mel features)
            model_dtype = next(model.parameters()).dtype
            inputs = {{k: v.to(model_dtype) if v.is_floating_point() else v
                       for k, v in inputs.items()}}
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            text = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]

            result = {{"text": text}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_speech_to_text", case.name)
            msg = f"HF speech-to-text failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        text = ""
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
            text = data.get("text", "")

        return StageOutput(
            stage_name=stage.name, data=data, text=text, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_canary_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run NeMo Canary model for speech-to-text reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_stt.json")

        audio_path = self._resolve_image_path(case.inputs.get("audio", ""))
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, numpy as np
            import scipy.io.wavfile as wav

            hf_id = {hf_id!r}
            audio_path = {audio_path!r}
            output_path = {output_path!r}

            # Try NeMo ASR model
            try:
                import nemo.collections.asr as nemo_asr
                import tempfile, struct
                # Convert to mono WAV if needed (Canary requires mono)
                sr_raw, audio_raw = wav.read(audio_path)
                if audio_raw.dtype == np.int16:
                    audio_f = audio_raw.astype(np.float32) / 32768.0
                elif audio_raw.dtype == np.int32:
                    audio_f = audio_raw.astype(np.float32) / 2147483648.0
                else:
                    audio_f = audio_raw.astype(np.float32)
                if len(audio_f.shape) > 1:
                    audio_f = audio_f.mean(axis=1)
                # Write mono 16kHz WAV
                target_sr = 16000
                if sr_raw != target_sr:
                    from scipy.signal import resample
                    audio_f = resample(audio_f, int(len(audio_f)*target_sr/sr_raw)).astype(np.float32)
                mono_path = audio_path + ".mono.wav"
                audio_i16 = np.clip(audio_f * 32768, -32768, 32767).astype(np.int16)
                wav.write(mono_path, target_sr, audio_i16)
                model = nemo_asr.models.ASRModel.from_pretrained(hf_id, map_location="cpu")
                model = model.cpu()
                model.eval()
                transcriptions = model.transcribe([mono_path], batch_size=1)
                if isinstance(transcriptions, list):
                    if hasattr(transcriptions[0], 'text'):
                        text = transcriptions[0].text
                    else:
                        text = str(transcriptions[0])
                else:
                    text = str(transcriptions)
            except ImportError:
                # Fallback: try HF pipeline
                import torch
                from transformers import pipeline
                sr, audio = wav.read(audio_path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=hf_id,
                    torch_dtype={torch_dtype_expr})
                result = pipe(audio)
                text = result.get("text", "")

            result = {{"text": text}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "nemo_canary_stt", case.name)
            msg = (f"NeMo Canary reference failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        text = ""
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
            text = data.get("text", "")

        return StageOutput(
            stage_name=stage.name, data=data, text=text, timing_s=elapsed,
            metadata={"returncode": result.returncode})

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
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_object_detection", case.name)
            msg = f"HF object detection failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})


    def _run_text_to_audio_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Bark model for text-to-audio reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_audio.json")
        wav_path = str(Path(model_dir) / "hf_audio.wav")

        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        seed = int(case.determinism.get("seed", 42))
        voice_preset = case.inputs.get("voice_preset", "")

        script = textwrap.dedent(f"""\
            import json, random, struct
            import numpy as np
            import torch
            from transformers import AutoProcessor, BarkModel, set_seed

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            seed = {seed!r}
            voice_preset = {voice_preset!r}
            output_path = {output_path!r}
            wav_path = {wav_path!r}

            # Make Bark reference generation deterministic across runs.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            set_seed(seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = BarkModel.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype={torch_dtype_expr})
            model.eval()

            if voice_preset:
                inputs = processor(
                    prompt, voice_preset=voice_preset, return_tensors="pt")
            else:
                inputs = processor(prompt, return_tensors="pt")
            with torch.no_grad():
                audio_values = model.generate(**inputs)

            audio = audio_values.cpu().numpy().squeeze()
            sample_rate = model.generation_config.sample_rate

            # Write WAV
            audio_f32 = audio.astype(np.float32)
            data_bytes = audio_f32.tobytes()
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + len(data_bytes)))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 3, 1, sample_rate,
                        sample_rate * 4, 4, 32))
                f.write(b"data")
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(data_bytes)

            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
            duration = len(audio_f32) / sample_rate
            result = {{"rms": rms, "duration_s": duration,
                      "sample_rate": sample_rate, "num_samples": len(audio_f32),
                      "seed": seed, "voice_preset": voice_preset}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK seed={{seed}} rms={{rms:.4f}} duration={{duration:.2f}}s")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_text_to_audio", case.name)
            msg = (f"HF text-to-audio failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
        if Path(wav_path).is_file():
            data["wav_path"] = wav_path

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_vl_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF vision-language model for reference generation."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        text_path = str(Path(model_dir) / "hf_vl_text.txt")

        prompt = case.inputs.get("prompt", "Describe this image.")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        hf_id = case.hf_id
        model_ref = _resolve_cached_model_ref(hf_id)
        fallback_text = _vl_fallback_prompt(hf_id, prompt)
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import sys, torch
            from transformers import AutoProcessor
            from PIL import Image
            from tests.e2e_harness.references.hf_transformers import (
                _decode_vl_generated_text,
            )

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            prompt = {prompt!r}
            fallback_text = {fallback_text!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            image_path = {image_path!r}
            text_path = {text_path!r}

            processor = AutoProcessor.from_pretrained(
                model_ref, trust_remote_code=trust_remote_code)

            # Try VL-specific auto classes in preference order
            import transformers
            model = None
            for cls_name in ["AutoModelForImageTextToText",
                             "AutoModelForVision2Seq"]:
                try:
                    cls = getattr(transformers, cls_name)
                    model = cls.from_pretrained(
                        model_ref, trust_remote_code=trust_remote_code,
                        torch_dtype={torch_dtype_expr})
                    break
                except (AttributeError, ImportError, ValueError, KeyError):
                    continue
            # Fallback for models registered as causal LM with multimodal
            # inputs (e.g. Phi-4-multimodal)
            if model is None:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_ref, trust_remote_code=True,
                    torch_dtype={torch_dtype_expr})
            model.eval()

            image = Image.open(image_path).convert("RGB")

            # Build conversation for chat-template models
            messages = [
                {{"role": "user", "content": [
                    {{"type": "image", "image": image_path}},
                    {{"type": "text", "text": prompt}},
                ]}}
            ]
            try:
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                if not isinstance(text_input, str):
                    raise TypeError("processor.apply_chat_template did not return text")
                if not any(marker in text_input for marker in (
                    "<|image_pad|>", "<|vision_start|>", "<image>"
                )):
                    raise ValueError("chat template produced no image placeholder")
                inputs = processor(
                    text=text_input, images=image, return_tensors="pt")
            except Exception:
                # Fallback for models without chat template
                inputs = processor(
                    text=fallback_text, images=image, return_tensors="pt")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens)

            # Decode only the generated portion (after input)
            input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
            text = _decode_vl_generated_text(processor, generated_ids[0], input_len)

            with open(text_path, "w") as f:
                f.write(text)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=1800, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_vl_generation", case.name)
            msg = (f"HF VL generation failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        text = ""
        if Path(text_path).is_file():
            text = Path(text_path).read_text(encoding="utf-8")

        return StageOutput(
            stage_name=stage.name,
            data={"text": text},
            text=text,
            timing_s=elapsed,
            metadata={"returncode": result.returncode,
                       "trust_remote_code": trust_remote_code},
        )

    def _run_prompted_segmentation_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF SAM model for prompted segmentation reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_sam.json")
        masks_path = str(Path(model_dir) / "hf_sam_masks.npy")
        segmented_image_path = str(Path(model_dir) / "hf_sam_segmented.png")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        point_x = case.inputs.get("point_x", 0.5)
        point_y = case.inputs.get("point_y", 0.5)
        hf_id = case.hf_id
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import SamModel, SamProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}
            masks_path = {masks_path!r}
            segmented_image_path = {segmented_image_path!r}
            point_x_frac = {point_x!r}
            point_y_frac = {point_y!r}

            processor = SamProcessor.from_pretrained(hf_id)
            model = SamModel.from_pretrained(
                hf_id, torch_dtype={torch_dtype_expr})
            model.eval()

            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # Convert fractional coords to pixel coords
            px = int(point_x_frac * w)
            py = int(point_y_frac * h)
            input_points = [[[px, py]]]

            inputs = processor(
                image, input_points=input_points, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]

            iou_scores = outputs.iou_scores[0, 0].cpu().numpy().tolist()
            mask_np = masks[0].cpu().numpy().astype(np.uint8)
            np.save(masks_path, mask_np)

            selected_mask = int(np.argmax(iou_scores)) if iou_scores else 0
            selected_mask = min(selected_mask, mask_np.shape[0] - 1)
            overlay_mask = mask_np[selected_mask].astype(bool)
            if overlay_mask.shape != (h, w):
                mask_img = Image.fromarray(overlay_mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                overlay_mask = np.asarray(mask_img, dtype=np.uint8) > 0

            image_arr = np.asarray(image, dtype=np.float32)
            overlay = np.zeros_like(image_arr)
            overlay[..., 0] = 255.0
            overlay[..., 1] = 96.0
            alpha = 0.55
            image_arr[overlay_mask] = (
                image_arr[overlay_mask] * (1.0 - alpha)
                + overlay[overlay_mask] * alpha
            )
            Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8)).save(
                segmented_image_path)

            result = {{
                "iou_scores": iou_scores,
                "num_masks": mask_np.shape[0],
                "mask_shape": list(mask_np.shape),
                "segmented_image_path": segmented_image_path,
            }}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK masks={{mask_np.shape[0]}} iou={{iou_scores}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_prompted_segmentation", case.name)
            msg = (f"HF prompted segmentation failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
        if Path(masks_path).is_file():
            data["masks_path"] = masks_path
        if Path(segmented_image_path).is_file():
            data["segmented_image_path"] = segmented_image_path

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})


plugin = HfTransformersReference()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face text references: generate, translate, encode, embed, rerank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ..base import BackendError, Invocation
from .common import ReferenceSpec, invocation, load_image, maybe_compile, required, tensor_observation, timed


def _generation_kwargs(request: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_new_tokens": int(request.get("max_new_tokens", 64))}
    temperature = float(request.get("temperature", 0.0))
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
        for name in ("top_k", "top_p", "min_p"):
            if name in request:
                kwargs[name] = request[name]
    else:
        kwargs["do_sample"] = False
    if int(request.get("seed", -1)) >= 0:
        torch.manual_seed(int(request["seed"]))
    return kwargs


class TextGeneration:
    """``generate`` for causal LMs, encoder-decoders, and image-text-to-text models.

    Timed boundary: ``model.generate`` after tokenization/image preprocessing.
    """

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        config = transformers.AutoConfig.from_pretrained(spec.model, **spec.pretrained_kwargs())
        kwargs = {"dtype": spec.dtype, **spec.pretrained_kwargs()}
        if hasattr(config, "vision_config"):
            self.processor = transformers.AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
            self.tokenizer = self.processor.tokenizer
            model_cls = transformers.AutoModelForImageTextToText
        else:
            self.processor = None
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model, **spec.pretrained_kwargs())
            model_cls = (transformers.AutoModelForSeq2SeqLM if config.is_encoder_decoder
                         else transformers.AutoModelForCausalLM)
        self.encoder_decoder = bool(config.is_encoder_decoder)
        self.model = maybe_compile(model_cls.from_pretrained(spec.model, **kwargs).to(spec.device).eval(), spec)

    def _inputs(self, request: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        prompt = str(required(request, "prompt"))
        image_path = request.get("image_path")
        # Vision-capable checkpoints (a vision_config) still take text-only requests as plain text:
        # the chat template applies only with an image or when the request asks for it.
        if self.processor is not None and image_path:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if image_path:
                content.insert(0, {"type": "image", "image": load_image(image_path)})
            messages = [{"role": "user", "content": content}]
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        if image_path:
            raise BackendError("image_path requires an image-text-to-text model")
        if request.get("use_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=bool(request.get("enable_thinking", False)))
        return self.tokenizer(prompt, return_tensors="pt")

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        inputs = self._inputs(request).to(self.spec.device)
        kwargs = _generation_kwargs(request)
        if self.encoder_decoder:
            # Transformers counts the decoder start token in max_new_tokens; TRTMC does not.
            kwargs["max_new_tokens"] += 1
        output, model_ms = timed(lambda: self.model.generate(**inputs, **kwargs))
        prompt_tokens = 0 if self.encoder_decoder else int(inputs["input_ids"].shape[1])
        token_ids = output[0, prompt_tokens:].tolist()
        if self.encoder_decoder:
            token_ids = self._strip_start_and_eos(token_ids)[: int(request.get("max_new_tokens", 64))]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return invocation({"output_tokens": len(token_ids), "token_ids": token_ids, "text": text},
                          model_ms, prompt_tokens=int(inputs["input_ids"].shape[1]))


    def _strip_start_and_eos(self, token_ids: list[int]) -> list[int]:
        """Encoder-decoder output as TRTMC reports it: no decoder start token, no trailing EOS/pad
        (the release baseline's ``strip-start-and-eos`` policy)."""
        config = self.model.generation_config
        if token_ids and token_ids[0] == getattr(config, "decoder_start_token_id", None):
            token_ids = token_ids[1:]
        eos = getattr(config, "eos_token_id", None)
        stops = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {getattr(config, "pad_token_id", None)}
        while token_ids and token_ids[-1] in stops:
            token_ids = token_ids[:-1]
        return token_ids


class Translation(TextGeneration):
    """``translate``: ``source_text`` into ``target_language`` (NLLB-style language tokens)."""

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        if "source_language" in request and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = request["source_language"]
        inputs = self.tokenizer(str(required(request, "source_text")), return_tensors="pt").to(self.spec.device)
        kwargs = _generation_kwargs(request)
        if "target_language" in request:
            kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(request["target_language"])
        output, model_ms = timed(lambda: self.model.generate(**inputs, **kwargs))
        token_ids = output[0].tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return invocation({"output_tokens": len(token_ids), "token_ids": token_ids, "text": text}, model_ms)


class TextEncoder:
    """``encode`` (first-token hidden state, TRTMC ``feature_kind: token``) and ``embed``
    (attention-mask mean pool, L2-normalized, ``feature_kind: pooled``); both report ``values``."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model, **spec.pretrained_kwargs())
        config = transformers.AutoConfig.from_pretrained(spec.model, **spec.pretrained_kwargs())
        model_cls = transformers.AutoModel
        if config.model_type == "dpr" and config.architectures:
            # AutoModel maps every DPR checkpoint to the question encoder; use the declared encoder.
            model_cls = getattr(transformers, config.architectures[0])
        model = model_cls.from_pretrained(spec.model, dtype=spec.dtype, **spec.pretrained_kwargs())
        self.model = maybe_compile(model.to(spec.device).eval(), spec)
        # Remote-code embedding models may return a causal-LM output; they need hidden states requested.
        self.forward_kwargs: dict[str, Any] = {}

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        batch = self.tokenizer(str(required(request, "prompt")), return_tensors="pt").to(self.spec.device)
        pooled = self.spec.operation == "embed"

        def run() -> torch.Tensor:
            outputs = self.model(**batch, **self.forward_kwargs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and getattr(outputs, "pooler_output", None) is not None:
                return outputs.pooler_output  # DPR encoders return only the pooled first-token representation
            if hidden is None:
                if getattr(outputs, "hidden_states", None) is None:
                    self.forward_kwargs = {"output_hidden_states": True}
                    outputs = self.model(**batch, **self.forward_kwargs)
                if getattr(outputs, "hidden_states", None) is None:
                    raise BackendError(f"{type(outputs).__name__} exposes no hidden states to pool")
                hidden = outputs.hidden_states[-1]
            if not pooled:
                return hidden[:, 0]
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return torch.nn.functional.normalize(((hidden * mask).sum(1) / mask.sum(1)).float(), dim=-1)

        output, model_ms = timed(run)
        observation = tensor_observation(output, artifact_base)
        # Same fields the TRTMC worker reports for TextToEmbedding / pooled or token features.
        observation.update(dim=int(output.shape[-1]), values=output[0].float().tolist(),
                           feature_kind="pooled" if pooled else "token")
        return invocation(observation, model_ms)


class Reranker:
    """``rerank``: cross-encoder logits for ``query`` against ``documents`` (input order)."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        kwargs = spec.pretrained_kwargs()
        self.processor = transformers.AutoProcessor.from_pretrained(
            spec.model, **kwargs, **dict(spec.options.get("processor_kwargs", {})))
        self.model = maybe_compile(transformers.AutoModelForSequenceClassification.from_pretrained(
            spec.model, dtype=spec.dtype, **kwargs).to(spec.device).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        query = str(required(request, "query"))
        documents = [str(document) for document in required(request, "documents")]
        if hasattr(self.processor, "process_queries_documents_crossencoder"):
            batch = self.processor.process_queries_documents_crossencoder(
                [{"question": query, "doc_text": document, "doc_image": ""} for document in documents])
        else:
            batch = self.processor([query] * len(documents), documents, padding=True, truncation=True,
                                   return_tensors="pt")
        batch = {key: value.to(self.spec.device) if hasattr(value, "to") else value for key, value in batch.items()}
        logits, model_ms = timed(lambda: self.model(**batch).logits.float().view(-1))
        scores = logits.cpu().tolist()
        return invocation({"documents": len(documents), "scores": scores, "order": "input_documents"}, model_ms)

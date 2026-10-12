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
    # One beam unless the request asks for more: a checkpoint's generation config may default to beam
    # search (Marian: four), while TRTMC decodes greedily.
    kwargs: dict[str, Any] = {"max_new_tokens": int(request.get("max_new_tokens", 64)),
                              "num_beams": int(request.get("num_beams", 1))}
    if float(request.get("repetition_penalty", 1.0)) != 1.0:
        kwargs["repetition_penalty"] = float(request["repetition_penalty"])
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


def _language_controls(tokenizer: Any, request: Mapping[str, Any]) -> tuple[dict[str, int], int | None]:
    """NLLB/M2M-style translation controls of a request, applied as the family reference applies them:
    the tokenizer's source language (else the source token replacing the final unknown token), and the
    target language as the forced first decoder token. Returns (generate kwargs, manual source id)."""
    def explicit(name: str) -> int | None:
        value = request.get(name)
        return None if value is None or int(value) < 0 else int(value)

    source, target = request.get("source_language"), request.get("target_language")
    source_id, target_id = explicit("source_language_token_id"), explicit("forced_bos_token_id")
    manual = None
    if source_id is not None and source is None:
        source = tokenizer.convert_ids_to_tokens(source_id)
    if source is not None:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = source
        elif source_id is not None and request.get("source_language_placement") == "replace-final-unk":
            manual = source_id
    if target is not None and hasattr(tokenizer, "src_lang"):
        resolved = tokenizer.convert_tokens_to_ids(target)
        if target_id is not None and target_id != resolved:
            raise BackendError("target_language disagrees with forced_bos_token_id")
        target_id = resolved
    return ({"forced_bos_token_id": target_id} if target_id is not None else {}), manual


def _without_appended_eos(inputs: Any, tokenizer: Any, prompt: str) -> Any:
    """A causal LM's plain prompt without the end-of-sequence token its tokenizer appends (OLMo, XGLM): the model
    continues the prompt instead of starting a new document. An encoder's input keeps it."""
    eos = tokenizer.eos_token_id
    ids = inputs["input_ids"]
    if eos is None or ids.shape[1] < 2 or int(ids[0, -1]) != eos or prompt.endswith(str(tokenizer.eos_token)):
        return inputs
    for key in list(inputs.keys()):
        inputs[key] = inputs[key][:, :-1]
    return inputs


def _place_source_language(inputs: Mapping[str, torch.Tensor], token_id: int | None, tokenizer: Any) -> None:
    if token_id is None:
        return
    ids, mask = inputs["input_ids"], inputs.get("attention_mask")
    for row in range(int(ids.shape[0])):
        index = int(ids.shape[1]) - 1 if mask is None else int(mask[row].nonzero()[-1].item())
        if int(ids[row, index]) not in (token_id, getattr(tokenizer, "unk_token_id", None)):
            raise BackendError("the tokenizer did not emit a source-language placeholder")
        ids[row, index] = token_id


class TextGeneration:
    """``generate`` for causal LMs, encoder-decoders, and image-text-to-text models.

    ``model_only_ms`` is ``model.generate`` after tokenization/image preprocessing; torch.compile
    references of causal LMs generate with a static KV cache when the architecture supports one.
    """

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        config = transformers.AutoConfig.from_pretrained(spec.model, **spec.pretrained_kwargs())
        kwargs = spec.model_kwargs()
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
        # The strong compiled baseline for causal LMs: a static KV cache (fixed shapes the compiled
        # forward reuses); models that cannot use one fall back to the dynamic cache.
        self.static_cache = spec.mode == "compile" and not self.encoder_decoder and self.processor is None

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
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs if self.encoder_decoder else _without_appended_eos(inputs, self.tokenizer, prompt)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        language, manual_source = (_language_controls(self.tokenizer, request) if self.encoder_decoder
                                   else ({}, None))
        inputs = self._inputs(request)
        _place_source_language(inputs, manual_source, self.tokenizer)
        inputs = inputs.to(self.spec.device)
        # max_new_tokens excludes the decoder start token, as TRTMC's budget does: the same work.
        kwargs = {**_generation_kwargs(request), **language}
        if self.static_cache:
            try:
                output, model_ms = timed(lambda: self.model.generate(**inputs, **kwargs, cache_implementation="static"))
            except Exception:  # noqa: BLE001 - this architecture has no static cache: keep the dynamic one
                self.static_cache = False
        if not self.static_cache:
            output, model_ms = timed(lambda: self.model.generate(**inputs, **kwargs))
        prompt_tokens = 0 if self.encoder_decoder else int(inputs["input_ids"].shape[1])
        token_ids = output[0, prompt_tokens:].tolist()
        if self.encoder_decoder:
            token_ids = self._strip_start_and_eos(token_ids)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return invocation({"output_tokens": len(token_ids), "token_ids": token_ids, "text": text},
                          model_ms, prompt_tokens=int(inputs["input_ids"].shape[1]),
                          kv_cache="static" if self.static_cache else "dynamic")


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


def encoder_input_limit(tokenizer: Any, config: Any) -> int | None:
    """Tokens an encoder accepts: the tokenizer's declared maximum, else the position embeddings."""
    declared = getattr(tokenizer, "model_max_length", None)
    if isinstance(declared, int) and 0 < declared < 1_000_000:
        return declared
    positions = getattr(config, "max_position_embeddings", None)
    return int(positions) if isinstance(positions, int) and positions > 0 else None


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
        model = model_cls.from_pretrained(spec.model, **spec.model_kwargs())
        self.model = maybe_compile(model.to(spec.device).eval(), spec)
        # Remote-code embedding models may return a causal-LM output; they need hidden states requested.
        self.forward_kwargs: dict[str, Any] = {}
        # Longer inputs are cut at the model's position limit (an overflow is a device-side assert that
        # poisons the CUDA context for every later request).
        self.max_length = encoder_input_limit(self.tokenizer, config)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        batch = self.tokenizer(str(required(request, "prompt")), return_tensors="pt", truncation=self.max_length is not None,
                               max_length=self.max_length).to(self.spec.device)
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
            spec.model, **spec.model_kwargs()).to(spec.device).eval(), spec)

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

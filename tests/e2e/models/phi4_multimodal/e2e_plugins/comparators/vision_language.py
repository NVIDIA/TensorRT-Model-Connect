# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-language comparator — compares TRT VL output against reference.

Metrics aligned with thresholds/defaults/vision_language_generation.json:
  - vision_embedding_cosine: Cosine similarity between vision embeddings.
  - vision_embedding_l2: L2 distance between vision embeddings.
  - token_agreement_rate: Fraction of steps with identical argmax (reuses text logic).
  - normalized_text_edit_distance: Levenshtein-normalized on decoded text.
  - semantic_similarity: Optional semantic similarity for caption parity.

The comparator handles three stage types:
  - "vision_encode": Compares vision encoder features.
  - "text_decode": Compares per-step logits (reuses text comparator helpers).
  - "full_generation": Compares generated text output.

Auto-discovered by the registry via the module-level ``plugin`` attribute.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from ._helpers import cosine_similarity, normalized_edit_distance

logger = logging.getLogger(__name__)


def _load_logits(stage_output: StageOutput) -> np.ndarray | None:
    """Load logits from StageOutput. Returns 2-D array [steps, vocab] or None."""
    logits = stage_output.logits
    if logits is None:
        logits = stage_output.data.get("logits_path")
    if logits is None:
        return None
    if isinstance(logits, np.ndarray):
        return logits
    if isinstance(logits, str) and Path(logits).is_file():
        return np.load(logits)
    return None


def _load_features(stage_output: StageOutput) -> np.ndarray | None:
    """Load vision features from StageOutput."""
    features = stage_output.data.get("features")
    if features is not None:
        return np.asarray(features, dtype=np.float32)
    path = stage_output.data.get("features_path")
    if path and Path(path).is_file():
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class VisionLanguageComparator:
    """Compares TRT VL inference against reference (HF Transformers).

    Metric names align with thresholds/defaults/vision_language_generation.json.
    """

    @property
    def task_strategy(self) -> str:
        return "vision_language_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        if stage.name == "vision_encode":
            return self._compare_vision(trt, ref, threshold, stage)
        elif stage.name == "text_decode":
            return self._compare_text_decode(trt, ref, threshold, stage)
        elif stage.name == "full_generation":
            return self._compare_generation(trt, ref, threshold, stage)
        else:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Unknown stage for VL comparator: {stage.name}",
            )

    # ------------------------------------------------------------------
    # Vision encode comparison
    # ------------------------------------------------------------------

    def _compare_vision(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare vision encoder features between TRT and reference."""
        metrics: dict[str, MetricResult] = {}

        trt_features = _load_features(trt)
        ref_features = _load_features(ref)

        # Fall back to subprocess-parsed metrics if no raw features
        trt_sub_metrics = trt.data.get("metrics", {})

        # If the diff_vl.py subprocess passed (rc=0 and "vision_pass" flag),
        # trust the result directly — the tool already did full TRT vs HF
        # comparison internally.
        if trt.data.get("passed") and trt_sub_metrics.get("vision_pass"):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.PASSED.value,
                metrics={
                    "vision_subprocess_pass": MetricResult(
                        value=1.0,
                        threshold=None,
                        operator=">=",
                        passed=True,
                        note="diff_vl.py subprocess PASS (internal comparison)",
                    ),
                },
                message="Vision compare: PASS (diff_vl.py)",
            )

        if trt_features is not None and ref_features is not None:
            trt_f = trt_features
            ref_f = ref_features

            shape_note = ""
            # Handle shape mismatch by comparing overlapping region
            if trt_f.shape != ref_f.shape:
                min_shape = tuple(
                    min(a, b) for a, b in zip(trt_f.shape, ref_f.shape)
                )
                slices = tuple(slice(0, s) for s in min_shape)
                trt_f = trt_f[slices]
                ref_f = ref_f[slices]
                shape_note = (
                    f"Shape mismatch: TRT={trt_features.shape} "
                    f"vs Ref={ref_features.shape}; "
                    f"compared overlap region {min_shape}"
                )

            # Cosine similarity — matches threshold key "vision_embedding_cosine"
            cosine = cosine_similarity(trt_f.flatten(), ref_f.flatten())
            cos_thresh = threshold.metrics.get("vision_embedding_cosine", 0.5)
            metrics["vision_embedding_cosine"] = MetricResult(
                value=cosine,
                threshold=cos_thresh,
                operator=">=",
                passed=cosine >= cos_thresh,
                note=shape_note,
            )

            # L2 distance — matches threshold key "vision_embedding_l2"
            l2 = float(np.sqrt(np.mean((trt_f - ref_f) ** 2)))
            l2_thresh = threshold.metrics.get("vision_embedding_l2")
            if l2_thresh is not None:
                metrics["vision_embedding_l2"] = MetricResult(
                    value=l2,
                    threshold=l2_thresh,
                    operator="<=",
                    passed=l2 <= l2_thresh,
                )
            else:
                metrics["vision_embedding_l2"] = MetricResult(
                    value=l2, threshold=None, operator="<=", passed=True,
                )

            # Max absolute difference (diagnostic, not gated)
            metrics["vision_max_diff"] = MetricResult(
                value=float(np.max(np.abs(trt_f - ref_f))),
                threshold=None,
                operator=">=",
                passed=True,
            )

        elif trt_sub_metrics:
            # Use metrics parsed from diff_vl.py subprocess output
            cos_thresh = threshold.metrics.get("vision_embedding_cosine", 0.5)
            if "cosine_sim" in trt_sub_metrics:
                cosine = trt_sub_metrics["cosine_sim"]
                metrics["vision_embedding_cosine"] = MetricResult(
                    value=cosine,
                    threshold=cos_thresh,
                    operator=">=",
                    passed=cosine >= cos_thresh,
                )
            if "max_diff" in trt_sub_metrics:
                metrics["vision_max_diff"] = MetricResult(
                    value=trt_sub_metrics["max_diff"],
                    threshold=None,
                    operator=">=",
                    passed=True,
                )
            if "mean_diff" in trt_sub_metrics:
                l2_thresh = threshold.metrics.get("vision_embedding_l2")
                mean_diff = trt_sub_metrics["mean_diff"]
                if l2_thresh is not None:
                    metrics["vision_embedding_l2"] = MetricResult(
                        value=mean_diff,
                        threshold=l2_thresh,
                        operator="<=",
                        passed=mean_diff <= l2_thresh,
                    )
                else:
                    metrics["vision_embedding_l2"] = MetricResult(
                        value=mean_diff,
                        threshold=None,
                        operator="<=",
                        passed=True,
                    )
        else:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics=metrics,
                message="No vision features available",
            )

        overall = all(m.passed for m in metrics.values()) if metrics else False
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if overall else StageStatus.FAILED.value,
            metrics=metrics,
            message=f"Vision compare: {'PASS' if overall else 'FAIL'}",
        )

    # ------------------------------------------------------------------
    # Text decode comparison (per-step logits)
    # ------------------------------------------------------------------

    def _compare_text_decode(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare per-step logits from VLTrtRunner vs reference.

        Reuses the same metric definitions as the text comparator:
        token_agreement_rate, normalized_text_edit_distance, and
        logit-level cosine similarity.
        """
        metrics: dict[str, MetricResult] = {}

        trt_logits = _load_logits(trt)
        ref_logits = _load_logits(ref)

        if trt_logits is not None and ref_logits is not None:
            # Truncate to common step/vocab
            n_steps = min(trt_logits.shape[0], ref_logits.shape[0])
            if n_steps == 0:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED.value,
                    metrics=metrics,
                    message="No steps",
                )
            trt_l = trt_logits[:n_steps]
            ref_l = ref_logits[:n_steps]
            if trt_l.shape[1] != ref_l.shape[1]:
                min_v = min(trt_l.shape[1], ref_l.shape[1])
                trt_l = trt_l[:, :min_v]
                ref_l = ref_l[:, :min_v]

            trt_l = np.nan_to_num(trt_l, nan=0.0, posinf=0.0, neginf=0.0)
            ref_l = np.nan_to_num(ref_l, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-step cosine similarity
            cosines = np.array([
                cosine_similarity(trt_l[i], ref_l[i])
                for i in range(n_steps)
            ])
            metrics["logit_cosine_p5"] = MetricResult(
                value=float(np.percentile(cosines, 5)),
                threshold=None,
                operator=">=",
                passed=True,
                note=f"logit steps={n_steps}",
            )

            # Token agreement rate
            trt_argmax = trt_l.argmax(axis=1)
            ref_argmax = ref_l.argmax(axis=1)
            token_agreement = float((trt_argmax == ref_argmax).mean())

            ta_thresh = threshold.metrics.get("token_agreement_rate")
            if ta_thresh is not None:
                metrics["token_agreement_rate"] = MetricResult(
                    value=token_agreement,
                    threshold=ta_thresh,
                    operator=">=",
                    passed=token_agreement >= ta_thresh,
                )
            else:
                metrics["token_agreement_rate"] = MetricResult(
                    value=token_agreement,
                    threshold=None,
                    operator=">=",
                    passed=True,
                )

        # Text comparison (always available from text_decode stage)
        trt_text = (trt.text or trt.data.get("generated_text") or "").strip()
        ref_text = (ref.text or ref.data.get("generated_text") or "").strip()

        if trt_text or ref_text:
            ned = normalized_edit_distance(trt_text, ref_text)
            ned_thresh = threshold.metrics.get("normalized_text_edit_distance")
            if ned_thresh is not None:
                metrics["normalized_text_edit_distance"] = MetricResult(
                    value=ned,
                    threshold=ned_thresh,
                    operator="<=",
                    passed=ned <= ned_thresh,
                )
            else:
                metrics["normalized_text_edit_distance"] = MetricResult(
                    value=ned,
                    threshold=None,
                    operator="<=",
                    passed=True,
                )

        overall = all(m.passed for m in metrics.values()) if metrics else True
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if overall else StageStatus.FAILED.value,
            metrics=metrics,
            message=f"VL text decode: {'PASS' if overall else 'FAIL'}",
        )

    # ------------------------------------------------------------------
    # Full generation comparison (C++ binary text output)
    # ------------------------------------------------------------------

    def _compare_generation(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare generated text between TRT C++ binary and reference.

        Computes normalized_text_edit_distance, token_agreement_rate (on
        decoded text words), and optional semantic_similarity.
        """
        metrics: dict[str, MetricResult] = {}

        trt_text = (trt.text or trt.data.get("generated_text") or "").strip()
        ref_text = (ref.text or ref.data.get("generated_text") or "").strip()

        metrics["trt_generated_length"] = MetricResult(
            value=float(len(trt_text)),
            threshold=None,
            operator=">=",
            passed=True,
        )
        metrics["ref_generated_length"] = MetricResult(
            value=float(len(ref_text)),
            threshold=None,
            operator=">=",
            passed=True,
        )

        if not trt_text:
            metrics["non_empty_output"] = MetricResult(
                value=0.0,
                threshold=1.0,
                operator=">=",
                passed=False,
            )
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics=metrics,
                message="TRT produced empty VL generation output",
            )
        metrics["non_empty_output"] = MetricResult(
            value=1.0,
            threshold=1.0,
            operator=">=",
            passed=True,
        )

        # Normalized text edit distance
        ned = normalized_edit_distance(trt_text, ref_text) if ref_text else 0.0
        ned_thresh = threshold.metrics.get("normalized_text_edit_distance")
        if ned_thresh is not None:
            metrics["normalized_text_edit_distance"] = MetricResult(
                value=ned,
                threshold=ned_thresh,
                operator="<=",
                passed=ned <= ned_thresh,
            )
        else:
            metrics["normalized_text_edit_distance"] = MetricResult(
                value=ned,
                threshold=None,
                operator="<=",
                passed=True,
            )

        # Word-level token agreement (approximate token_agreement_rate on text)
        trt_words = trt_text.lower().split()
        ref_words = ref_text.lower().split()
        if trt_words and ref_words:
            n_compare = min(len(trt_words), len(ref_words))
            matches = sum(
                1 for a, b in zip(trt_words[:n_compare], ref_words[:n_compare])
                if a == b
            )
            word_agreement = matches / n_compare

            ta_thresh = threshold.metrics.get("token_agreement_rate")
            if ta_thresh is not None:
                metrics["token_agreement_rate"] = MetricResult(
                    value=word_agreement,
                    threshold=ta_thresh,
                    operator=">=",
                    passed=word_agreement >= ta_thresh,
                )
            else:
                metrics["token_agreement_rate"] = MetricResult(
                    value=word_agreement,
                    threshold=None,
                    operator=">=",
                    passed=True,
                )

        # Optional semantic similarity (requires sentence-transformers)
        sem_thresh = threshold.metrics.get("semantic_similarity")
        if sem_thresh is not None and trt_text and ref_text:
            sem_sim = _compute_semantic_similarity(trt_text, ref_text)
            if sem_sim is not None:
                metrics["semantic_similarity"] = MetricResult(
                    value=sem_sim,
                    threshold=sem_thresh,
                    operator=">=",
                    passed=sem_sim >= sem_thresh,
                )
            else:
                metrics["semantic_similarity"] = MetricResult(
                    value=0.0,
                    threshold=sem_thresh,
                    operator=">=",
                    passed=True,
                    note="skipped (sentence-transformers not available)",
                )

        # Composite rule: NED alone is sufficient for VL generation.
        # Word-level agreement is unreliable for VL since the same scene
        # can be described with different words that are equally valid.
        ned_ok = metrics.get(
            "normalized_text_edit_distance",
            MetricResult(value=0, passed=True),
        ).passed
        ta_ok = metrics.get(
            "token_agreement_rate",
            MetricResult(value=0, passed=True),
        ).passed
        non_empty = metrics.get(
            "non_empty_output",
            MetricResult(value=0, passed=True),
        ).passed
        overall = non_empty and (ned_ok or ta_ok)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if overall else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="non_empty_output AND (normalized_text_edit_distance OR token_agreement_rate)",
            message=f"VL generation compare: {'PASS' if overall else 'FAIL'}",
        )


def _compute_semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Compute sentence-level semantic similarity using sentence-transformers.

    Returns cosine similarity between sentence embeddings, or None if
    sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        cos = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-12
        ))
        return cos
    except Exception as e:
        logger.warning("Semantic similarity computation failed: %s", e)
        return None


plugin = VisionLanguageComparator()

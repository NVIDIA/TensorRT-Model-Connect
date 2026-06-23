"""Contract test plugin for encoder-only models."""
from __future__ import annotations
import numpy as np

from ..contracts import MetricResult
from .base import contract_config, make_pass, make_fail, make_error


_MIN_CONTRACT_COSINE = 0.80


class EncoderFeaturesPlugin:
    reference_families = ["encoder_base_features"]
    user_contract = "representation_parity"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        # Extract embeddings from data
        trt_emb = trt_output.data.get("embedding") or trt_output.data.get("cls_embedding")
        ref_emb = ref_output.data.get("embedding") or ref_output.data.get("cls_embedding")

        if trt_emb is None or ref_emb is None:
            return make_error("full_inference", "Missing embedding in output data")

        trt_arr = np.asarray(trt_emb, dtype=np.float32).flatten()
        ref_arr = np.asarray(ref_emb, dtype=np.float32).flatten()

        # Cosine similarity
        norm_t = np.linalg.norm(trt_arr)
        norm_r = np.linalg.norm(ref_arr)
        if norm_t < 1e-12 or norm_r < 1e-12:
            cosine = 0.0
        else:
            cosine = float(np.dot(trt_arr, ref_arr) / (norm_t * norm_r))

        configured_cosine_threshold = threshold.metrics.get(
            "contract_cosine_threshold",
            threshold.metrics.get("cls_embedding_cosine", 0.80))
        cosine_threshold = max(configured_cosine_threshold, _MIN_CONTRACT_COSINE)
        note = ""
        if cosine_threshold != configured_cosine_threshold:
            note = (
                f"configured threshold {configured_cosine_threshold} raised to "
                f"{_MIN_CONTRACT_COSINE} floor")

        metrics = {
            "cosine_similarity": MetricResult(
                value=cosine, threshold=cosine_threshold, operator=">=",
                passed=cosine >= cosine_threshold, note=note),
        }

        if cosine >= cosine_threshold:
            return make_pass("full_inference", metrics, "cosine >= threshold")
        return make_fail("full_inference", metrics, "cosine >= threshold",
                        f"Representation diverged: cosine={cosine:.4f}")


plugin = EncoderFeaturesPlugin()

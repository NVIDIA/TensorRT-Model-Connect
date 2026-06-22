"""codegen model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class CodegenHfTransformersReference(HfTransformersReference):
    """codegen local reference for hf_transformers."""

reference = CodegenHfTransformersReference()

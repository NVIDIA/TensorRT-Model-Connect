"""Ownership contract tests for agentic quantization rollout.

These tests enforce the repo-level boundary between the shared quantization
core and family-local quantization policy.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOC = REPO_ROOT / "website/docs/wiki/Agentic-Quantization-Core-Minimal-Plan.md"

SHARED_CORE_FILES = [
    "python/tensorrt_model_connect/quantization/plan.py",
    "python/tensorrt_model_connect/quantization/context.py",
    "python/tensorrt_model_connect/quantization/formats.py",
    "python/tensorrt_model_connect/quantization/profile.py",
    "python/tensorrt_model_connect/quantization/scales.py",
    "python/tensorrt_model_connect/quantization/scale_providers.py",
    "python/tensorrt_model_connect/quantization/adapters.py",
    "python/tensorrt_model_connect/quantization/__init__.py",
    "python/tensorrt_model_connect/graph_blocks.py",
    "python/tensorrt_model_connect/graph_ops.py",
]

# Keep this list to unambiguous, multi-character family names so the regex
# stays stable and avoids false positives on common short words.
FORBIDDEN_FAMILY_NAME_TOKENS = [
    "bart",
    "bloom",
    "canary",
    "convbert",
    "deepseek",
    "falcon",
    "flux",
    "gemma",
    "internlm",
    "llama",
    "magpie",
    "mistral",
    "mixtral",
    "nemotron",
    "olmo",
    "pixart",
    "qwen",
    "roberta",
    "rwkv",
    "segformer",
    "stablelm",
    "starcoder",
    "whisper",
    "xglm",
    "yolox",
]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


class TestQuantizationOwnershipDoc:
    def test_policy_doc_contains_normative_ownership_language(self):
        text = POLICY_DOC.read_text(encoding="utf-8").lower()
        required_phrases = [
            "## ownership standard",
            "family agent default scope",
            "core agent scope",
            "shared quantization code must not import specific family plugins",
            "shared quantization code must not branch on concrete family names",
            "family-specific quantization policy belongs in plugin hooks",
            "## test enforcement",
        ]
        for phrase in required_phrases:
            assert phrase in text


class TestQuantizationSharedCoreBoundary:
    def test_shared_core_files_exist(self):
        for rel_path in SHARED_CORE_FILES:
            assert (REPO_ROOT / rel_path).is_file(), rel_path

    def test_shared_core_files_are_not_family_plugins(self):
        for rel_path in SHARED_CORE_FILES:
            assert "/families/" not in rel_path

    def test_shared_core_does_not_import_concrete_family_modules(self):
        import_re = re.compile(
            r"from\s+\.\.families\b"
            r"|from\s+\.families\b"
            r"|import\s+.*families\b"
        )
        for rel_path in SHARED_CORE_FILES:
            text = _read(rel_path)
            assert import_re.search(text) is None, rel_path

    def test_shared_core_does_not_contain_family_name_branches(self):
        token_re = re.compile(
            r"['\"](" + "|".join(FORBIDDEN_FAMILY_NAME_TOKENS) + r")['\"]")
        for rel_path in SHARED_CORE_FILES:
            text = _read(rel_path)
            assert token_re.search(text) is None, rel_path


class TestFamilyLocalQuantHooks:
    def test_qwen_quant_policy_lives_in_family_plugin(self):
        text = _read("python/tensorrt_model_connect/families/qwen/plugin.py")
        assert "def quant_exclude_patterns" in text
        assert "def quant_adapter" in text

    def test_shared_core_does_not_define_qwen_specific_quant_policy(self):
        for rel_path in SHARED_CORE_FILES:
            text = _read(rel_path)
            assert "layer.*.w_o" not in text, rel_path
            assert "layer.*.w_gate" not in text, rel_path
            assert "layer.*.w_up" not in text, rel_path
            assert "layer.*.w_down" not in text, rel_path

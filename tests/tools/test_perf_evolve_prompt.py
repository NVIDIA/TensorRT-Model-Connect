"""Unit tests for scripts/perf_evolve_prompt.py — prompt builder validation.

Verifies that the generated prompt contains all required sections, keywords,
and structure. No GPU or container needed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from perf_evolve_prompt import build_evolve_prompt, _infer_family

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASELINE = {
    "throughput_tps": 265.7,
    "decode_ms": 3.8,
    "prefill_ms": 5.0,
    "per_token_ms": 3.8,
}

SOL_DATA = {
    "sol_tps": 955,
    "utilization_pct": 27.8,
    "bottleneck": "bandwidth",
    "fp16_sol_tps": 1909,
}


@pytest.fixture
def full_prompt():
    """Generate a full prompt with all data."""
    return build_evolve_prompt(
        model="example-org/example-decoder",
        container="trtmc-test-evolve",
        baseline=BASELINE,
        max_iterations=5,
        max_cache_length=256,
        sol_data=SOL_DATA,
    )


@pytest.fixture
def prompt_no_sol():
    """Generate a prompt without SOL data."""
    return build_evolve_prompt(
        model="example-org/example-decoder",
        container="trtmc-test-evolve",
        baseline=BASELINE,
        max_iterations=5,
        max_cache_length=256,
    )


# ---------------------------------------------------------------------------
# Required sections
# ---------------------------------------------------------------------------

class TestRequiredSections:
    """Every prompt MUST contain these sections for the agent to work correctly."""

    REQUIRED = [
        "STEP 0: Profile Before Optimizing",
        "Classify Bottleneck",
        "Optimization Knowledge Base",
        "Measured Results",
        "Validation Protocol",
        "CRITICAL RULES",
    ]

    @pytest.mark.parametrize("section", REQUIRED)
    def test_section_present(self, full_prompt, section):
        assert section in full_prompt, f"Missing required section: {section}"

    @pytest.mark.parametrize("section", REQUIRED)
    def test_section_present_without_sol(self, prompt_no_sol, section):
        """Required sections must be present even without SOL data."""
        assert section in prompt_no_sol


# ---------------------------------------------------------------------------
# Optimization levels (the core fix from Phase 1)
# ---------------------------------------------------------------------------

class TestOptimizationLevels:
    """Agent must be guided to try ALL levels, in the right order."""

    def test_cuda_graphs_mentioned(self, full_prompt):
        assert "CUDA Graphs" in full_prompt

    def test_fp16_mentioned(self, full_prompt):
        assert "FP16" in full_prompt

    def test_graph_topology_warning(self, full_prompt):
        """Graph topology should be marked as low priority with 0% Phase 0 result."""
        assert "0%" in full_prompt
        assert "Graph Topology" in full_prompt or "graph_topology" in full_prompt

    def test_priority_order(self, full_prompt):
        """L1 Runtime should appear before L3 Graph in the prompt."""
        runtime_pos = full_prompt.find("Level 1: Runtime")
        graph_pos = full_prompt.find("Level 3: Graph")
        assert runtime_pos < graph_pos, "L1 Runtime must come before L3 Graph"

    def test_l1_before_l3_rule(self, full_prompt):
        assert "L1 before L3" in full_prompt


# ---------------------------------------------------------------------------
# SOL integration
# ---------------------------------------------------------------------------

class TestSolIntegration:
    def test_sol_section_with_data(self, full_prompt):
        assert "Speed-of-Light" in full_prompt
        assert "955" in full_prompt  # FP32 SOL
        assert "27.8%" in full_prompt  # utilization
        assert "bandwidth" in full_prompt

    def test_fp16_sol_shown(self, full_prompt):
        assert "1,909" in full_prompt  # FP16 SOL with comma formatting

    def test_stopping_rule(self, full_prompt):
        assert "80%" in full_prompt
        assert "stop optimizing" in full_prompt.lower() or "Stopping" in full_prompt

    def test_sol_fallback_without_data(self, prompt_no_sol):
        """Without SOL data, prompt should tell agent to run sol_estimate.py."""
        assert "sol_estimate.py" in prompt_no_sol


# ---------------------------------------------------------------------------
# Profiling commands
# ---------------------------------------------------------------------------

class TestProfilingCommands:
    def test_cpu_profile_command(self, full_prompt):
        assert "cpu_profile.py" in full_prompt

    def test_trtmc_profile_command(self, full_prompt):
        assert "trtmc profile" in full_prompt

    def test_perf_compare_command(self, full_prompt):
        assert "perf_compare.py" in full_prompt

    def test_diff_logits_command(self, full_prompt):
        assert "diff_logits.py" in full_prompt


# ---------------------------------------------------------------------------
# Validation protocol
# ---------------------------------------------------------------------------

class TestValidation:
    def test_build_step(self, full_prompt):
        assert "./build/trtmc build" in full_prompt

    def test_correctness_step(self, full_prompt):
        assert "diff_logits" in full_prompt
        assert "atol" in full_prompt

    def test_benchmark_step(self, full_prompt):
        assert "perf_compare" in full_prompt

    def test_record_step(self, full_prompt):
        assert "evolve_results.jsonl" in full_prompt

    def test_fp16_tolerance(self, full_prompt):
        """FP16 changes need relaxed tolerance."""
        assert "0.1" in full_prompt  # FP16 atol


# ---------------------------------------------------------------------------
# Container and model interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_container_name(self, full_prompt):
        assert "trtmc-test-evolve" in full_prompt

    def test_model_name(self, full_prompt):
        assert "example-org/example-decoder" in full_prompt

    def test_baseline_throughput(self, full_prompt):
        assert "265.7" in full_prompt

    def test_max_cache_length(self, full_prompt):
        assert "256" in full_prompt


# ---------------------------------------------------------------------------
# Focus area
# ---------------------------------------------------------------------------

class TestFocusArea:
    def test_runtime_focus(self):
        prompt = build_evolve_prompt(
            model="test", container="c", baseline=BASELINE,
            focus_area="runtime",
        )
        assert "CUDA Graphs" in prompt
        assert "FOCUS AREA: runtime" in prompt

    def test_precision_focus(self):
        prompt = build_evolve_prompt(
            model="test", container="c", baseline=BASELINE,
            focus_area="precision",
        )
        assert "FP16" in prompt
        assert "FOCUS AREA: precision" in prompt

    def test_graph_focus(self):
        prompt = build_evolve_prompt(
            model="test", container="c", baseline=BASELINE,
            focus_area="graph_topology",
        )
        assert "Graph Topology" in prompt


# ---------------------------------------------------------------------------
# Family inference
# ---------------------------------------------------------------------------

class TestInferFamily:
    def test_uses_family_registry(self, monkeypatch):
        from tensorrt_model_connect import families

        monkeypatch.setattr(
            families,
            "resolve_family_id",
            lambda model_type: "owned_family" if model_type == "owned-model" else None,
        )

        assert _infer_family("example-org/owned-model") == "owned_family"

    def test_falls_back_to_normalized_prefix(self):
        assert _infer_family("example-org/example-decoder") == "example"

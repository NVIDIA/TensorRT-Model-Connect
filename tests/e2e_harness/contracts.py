# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain types and protocol contracts for the unified E2E testing framework.

This module is the stable foundation that all other harness components depend on.
It contains only stdlib imports (dataclasses, enum, typing) and defines:

- Domain dataclasses: E2ECase, StageSpec, PreflightRequirement, ThresholdProfile,
  StageOutput, CompareResult, E2EResult, RunContext
- Enums: FailureType, OracleLevel, E2EStatus
- Protocols: TaskStrategyRunner, ReferenceBackendRunner, Comparator,
  ReproCommandProvider, ArtifactSink, DeterminismPolicy

The orchestrator imports ONLY these types and protocols. Concrete
implementations live in strategy runners, reference backends, and comparators.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailureType(enum.Enum):
    """Classification of how an E2E run failed.

    Used in E2EResult to indicate the phase where failure occurred,
    enabling fast triage without reading full logs.
    """

    PRECHECK_FAIL = "precheck_fail"
    BUILD_FAIL = "build_fail"
    TRT_RUN_FAIL = "trt_run_fail"
    REFERENCE_RUN_FAIL = "reference_run_fail"
    COMPARE_FAIL = "compare_fail"
    DETERMINISM_FAIL = "determinism_fail"
    ARTIFACT_WRITE_FAIL = "artifact_write_fail"


class OracleLevel(enum.Enum):
    """Strength of the reference oracle used for comparison.

    Higher levels imply stronger parity guarantees. The level is recorded in
    result.json so that consumers know what a "pass" actually means.

    L1: External reference (HF Transformers / Diffusers / official lib).
    L2: Internal reference (custom torch / Python implementation).
    L3: Snapshot regression (golden outputs from a trusted prior run).
    L4: Invariants only (metamorphic / property-based checks, no reference).
    """

    L1_EXTERNAL_REFERENCE = "L1_external_reference"
    L2_INTERNAL_REFERENCE = "L2_internal_reference"
    L3_SNAPSHOT_REGRESSION = "L3_snapshot_regression"
    L4_INVARIANTS = "L4_invariants"


class E2EStatus(enum.Enum):
    """Terminal status of an E2E run."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class ReferenceFamily(enum.Enum):
    """Classification of how a model's ground truth reference is defined.

    Each model is assigned to exactly one reference family, which determines
    what the CI should verify and how.  Strong-behavior families have clear
    end-user contracts; weak-behavior families only guarantee parity.

    See the E2E contract documentation in this module for the current definitions.
    """

    # --- Text / causal ---
    CAUSAL_BASE_CONTINUATION = "causal_base_continuation"
    CODE_BASE_COMPLETION = "code_base_completion"
    CHAT_INSTRUCT_TEMPLATE = "chat_instruct_template"
    TRANSLATION_CHAT_TEMPLATE = "translation_chat_template"

    # --- Seq2seq ---
    SEQ2SEQ_TEXT2TEXT = "seq2seq_text2text"
    SEQ2SEQ_TRANSLATION = "seq2seq_translation"
    SEQ2SEQ_BASE_WEAK = "seq2seq_base_weak"

    # --- Encoder / embedding / rerank ---
    ENCODER_BASE_FEATURES = "encoder_base_features"
    SENTENCE_TRANSFORMER_EMBED = "sentence_transformer_embed"
    BGE_RETRIEVAL_EMBED = "bge_retrieval_embed"
    VL_EMBED_RETRIEVAL = "vl_embed_retrieval"
    VL_RERANK = "vl_rerank"

    # --- Vision-language ---
    VL_INSTRUCT_QA = "vl_instruct_qa"
    OCR_MARKDOWN = "ocr_markdown"

    # --- Speech / audio ---

    # --- Segmentation ---
    SEMANTIC_SEGMENTATION = "semantic_segmentation"

    # --- Image classification ---
    IMAGE_CLASSIFICATION = "image_classification"

    # --- Diffusion ---
    DIFFUSERS_IMAGE_GEN = "diffusers_image_gen"
    DIFFUSERS_VIDEO_GEN = "diffusers_video_gen"
    ELF_UNCONDITIONAL_TEXT = "elf_unconditional_text"
    ELF_CONDITIONAL_TEXT = "elf_conditional_text"

    # --- Time series ---
    TIME_SERIES_POINT_FORECAST = "time_series_point_forecast"
    TIME_SERIES_QUANTILE_FORECAST = "time_series_quantile_forecast"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_REGRESSION = "time_series_regression"


class ArtifactType(enum.Enum):
    """Type of artifact produced by a pipeline stage.

    Used by StageSpec to declare what a stage outputs, enabling typed
    comparison dispatch and artifact persistence with schema validation.

    See the E2E contract documentation in this module.
    """

    TOKEN_IDS = "token_ids"
    NORMALIZED_TEXT = "normalized_text"
    EMBEDDING = "embedding"
    CLASS_MAP = "class_map"
    SEMANTIC_TOKENS = "semantic_tokens"
    WAVEFORM = "waveform"
    IMAGE = "image"
    VIDEO_FRAMES = "video_frames"
    TEXT_SAMPLES = "text_samples"
    LOGITS = "logits"
    SCORE = "score"
    RANKING = "ranking"


class ComparisonMode(enum.Enum):
    """How to compare TRT output against reference output.

    Each mode implies specific metrics and gating logic.  A stage's
    artifact_type + comparison_mode together determine the comparator
    behavior.

    See the E2E contract documentation in this module.
    """

    EXACT_TEXT = "exact_text"
    PREFIX_TEXT = "prefix_text"
    TOKEN_EXACT = "token_exact"
    NUMERIC_TENSOR = "numeric_tensor"
    STRUCTURED_RANKING = "structured_ranking"
    MASK_OVERLAP = "mask_overlap"
    DETECTION_MATCH = "detection_match"
    MEDIA_SIMILARITY = "media_similarity"
    SEMANTIC_JUDGMENT = "semantic_judgment"
    TEXT_QUALITY_METRICS = "text_quality_metrics"
    INVARIANT_CHECK = "invariant_check"


class CILane(enum.Enum):
    """CI execution lane — determines when and how a test runs.

    See the E2E contract documentation in this module.
    """

    ACCEPTANCE = "acceptance"
    NIGHTLY_PARITY = "nightly_parity"
    CROSSOVER_DEBUG = "crossover_debug"


class UserContract(enum.Enum):
    """The user-facing behavior that CI should verify for a model.

    Strong-behavior models have clear product contracts (chat response,
    transcript, etc.).  Weak-behavior models can only verify parity
    (continuation, representation).

    See the E2E contract documentation in this module.
    """

    # --- Strong behavior contracts ---
    CHAT_RESPONSE = "chat_response"
    CODE_COMPLETION = "code_completion"
    TRANSLATION = "translation"
    SEQ2SEQ_OUTPUT = "seq2seq_output"
    EXACT_TRANSCRIPT = "exact_transcript"
    TTS_AUDIO = "tts_audio"
    SPEECH_RESPONSE = "speech_response"
    SEGMENTATION_MASK = "segmentation_mask"
    PROMPTED_MASK = "prompted_mask"
    IMAGE_CLASSIFICATION = "image_classification"
    EMBEDDING_VECTOR = "embedding_vector"
    RANKING_ORDER = "ranking_order"
    VL_ANSWER = "vl_answer"
    OCR_TEXT = "ocr_text"
    TEXT_GENERATION = "text-generation"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    ANY_TO_ANY = "any-to-any"
    DIFFUSION_IMAGE = "diffusion_image"
    DIFFUSION_VIDEO = "diffusion_video"
    DIFFUSION_TEXT_GENERATION = "diffusion_text_generation"
    MODEL_CARD_GENERATION_PARITY = "model_card_generation_parity"

    # --- Time series ---
    TIME_SERIES_POINT_FORECAST = "time_series_point_forecast"
    TIME_SERIES_QUANTILE_FORECAST = "time_series_quantile_forecast"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_REGRESSION = "time_series_regression"

    # --- Weak / parity contracts ---
    CONTINUATION_PARITY = "continuation_parity"
    REPRESENTATION_PARITY = "representation_parity"


class StageStatus(enum.Enum):
    """Status of a single stage comparison.

    PASSED: All metrics met their thresholds.
    FAILED: One or more gated metrics failed.
    SKIPPED: Stage was not compared (no reference/comparator available).
    ERROR: Stage comparison raised an exception.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreflightRequirement:
    """A single prerequisite that must be satisfied before a case can run.

    Attributes:
        kind: Requirement type identifier. Supported kinds:
            - "binary_exists": args must contain "path".
            - "gpu_memory_min_gb": args must contain "min_gb".
            - "hf_auth_token_present": no args needed.
            - "asset_exists": args must contain "path".
            - "python_module_available": args must contain "module" and may
              optionally contain "phase" ("build", "runtime", "reference")
              and "timeout_s". The module must import successfully in that
              profile.
        args: Parameters specific to the requirement kind.
        gating: If True (default), an unmet requirement causes PRECHECK_FAIL.
            If False, the requirement is advisory and logged but does not block.
    """

    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    gating: bool = True


@dataclass
class StageSpec:
    """Describes one stage in the E2E lifecycle for a model case.

    For simple models (text generation), there is typically one stage.
    For composite pipelines (diffusion, omni), there can be many.

    Attributes:
        name: Stage identifier (e.g. "generate", "preprocess", "vae_decode").
        required: If True, stage failure causes overall case failure.
        runner_override: Optional strategy runner name to use instead of the
            default for this case's task_strategy.
        comparator_override: Optional comparator name to use instead of the
            default for this case's task_strategy.
        artifact_type: What type of artifact this stage produces (see
            ArtifactType). Enables typed comparison dispatch.
        comparison_mode: How to compare TRT vs reference for this stage
            (see ComparisonMode). Overrides default mode for the artifact type.
        ci_lanes: Which CI lanes should test this stage. Defaults to
            ["acceptance"]. Stages marked "nightly_parity" are skipped in
            acceptance runs.
    """

    name: str
    required: bool = True
    runner_override: Optional[str] = None
    comparator_override: Optional[str] = None
    artifact_type: str = ""
    comparison_mode: str = ""
    ci_lanes: List[str] = field(default_factory=lambda: [CILane.ACCEPTANCE.value])


@dataclass
class ThresholdProfile:
    """Tolerance profile for comparing TRT vs reference outputs.

    Attributes:
        task_strategy: The task strategy this profile applies to.
        profile_name: Human-readable profile name (e.g. "fp16_default").
        metrics: Metric name -> threshold value mapping. Metric names are
            strategy-specific (e.g. "logit_cosine_p5", "token_agreement_rate",
            "mIoU", "mel_distance", "psnr").
        percentile_gates: Optional percentile-based gates (e.g.
            {"logit_rel_l2_p95": 0.01, "logit_rel_l2_p99": 0.05}).
        composite_rules: Optional list of composite gating rules that
            combine multiple metrics (e.g. "pass if token_agreement >= 0.9
            OR (cosine >= 0.999 AND topk_hit >= 0.95)").
    """

    task_strategy: str
    profile_name: str = "default"
    metrics: Dict[str, float] = field(default_factory=dict)
    percentile_gates: Dict[str, float] = field(default_factory=dict)
    composite_rules: List[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Self-describing result for a single comparison metric.

    Combines the metric value, its threshold, the comparison operator,
    and the pass/fail outcome into one structure.

    Attributes:
        value: Computed metric value.
        threshold: Threshold the value was compared against (None if informational).
        operator: Comparison operator as a string (e.g. ">=", "<=", "==").
        passed: Whether this metric met its threshold.
        note: Optional human-readable annotation.
    """

    value: float
    threshold: Optional[float] = None
    operator: str = ">="
    passed: bool = True
    note: str = ""


@dataclass
class E2ECase:
    """Complete definition of one testcase within an E2E model manifest.

    Contains everything needed to run, compare, and report one input contract
    against the bundle shared by its owning model.

    Attributes:
        name: Unique case identifier (e.g. "example-model").
        hf_id: HuggingFace model ID or local path.
        family: Family plugin name (e.g. "example_family").
        runtime_strategy: C++ runtime backend selector (e.g. "example_family_decoder_kv_cache").
        task_strategy: Logical task type declared by the model manifest
            (e.g. "text_generation_causal").
        reference_backend: Which reference implementation to compare against
            (e.g. "hf_transformers", "torch_reference", "golden_snapshot").
        oracle_level: Strength of the reference oracle. See OracleLevel.
        bundle: Bundle filename (e.g. "example-model.trtfb").
        inputs: Task-specific inputs (prompt, image path, audio path, etc.).
        preflight: List of prerequisites to check before running.
        stages: Ordered list of stages to execute.
        comparison_profile: Name of the ThresholdProfile to use.
        threshold_overrides: Per-metric overrides that take precedence over
            the profile defaults.
        determinism: Settings for determinism/reproducibility checks
            (e.g. {"reruns": 2, "seed": 42}).
        execution_profiles: Named Python environment profiles for the build,
            runtime, and reference phases.
        metadata: Arbitrary extra fields (notes, trust_remote_code, etc.).
    """

    name: str
    hf_id: str
    family: str
    runtime_strategy: str
    task_strategy: str = ""
    reference_backend: str = "hf_transformers"
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value
    reference_family: str = ""
    user_contract: str = ""
    ci_lane: str = CILane.ACCEPTANCE.value
    bundle: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    preflight: List[PreflightRequirement] = field(default_factory=list)
    stages: List[StageSpec] = field(default_factory=list)
    comparison_profile: str = "default"
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
    determinism: Dict[str, Any] = field(default_factory=dict)
    execution_profiles: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Use the runtime selector as a generic fallback if not set."""
        if not self.task_strategy:
            self.task_strategy = self.runtime_strategy


@dataclass
class E2EModel:
    """One buildable model bundle with one or more E2E testcases."""

    name: str
    hf_id: str
    family: str
    bundle: str
    testcases: List[E2ECase] = field(default_factory=list)
    manifest_path: str = ""

    @property
    def build_case(self) -> E2ECase:
        """Return the canonical case carrying this model's build settings."""
        if not self.testcases:
            raise ValueError(f"Model {self.name!r} has no testcases")
        for case in self.testcases:
            if case.name == self.name:
                return case
        return self.testcases[0]


@dataclass
class StageOutput:
    """Output produced by running one stage (TRT or reference).

    Intentionally flexible: different task strategies populate different
    fields. The ``data`` dict is the primary carrier for modality-specific
    outputs (logits array, image path, audio samples, embeddings, etc.).

    Attributes:
        stage_name: Which stage produced this output.
        data: Modality-specific output data. Keys depend on task_strategy.
            Examples:
            - text_generation: {"token_ids": [...], "logits_path": "/tmp/..."}
            - segmentation: {"mask": <np.ndarray>, "class_ids": [...]}
            - diffusion: {"frames_dir": "/tmp/...", "latents": <np.ndarray>}
        text: Generated text (for text-producing strategies).
        logits: Path to saved logits or in-memory array (optional).
        timing_s: Wall-clock time for this stage in seconds.
        metadata: Extra info (command used, return code, warnings, etc.).
    """

    stage_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    logits: Any = None  # numpy array or path string
    timing_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResult:
    """Outcome of comparing TRT output to reference output for one stage.

    Each metric is self-describing via :class:`MetricResult`, carrying its
    value, threshold, operator, and pass/fail in a single object.

    Attributes:
        stage_name: Which stage was compared.
        status: Stage-level status string (see :class:`StageStatus`).
        metrics: Per-metric results keyed by metric name.
        composite_rule: Human-readable description of the composite gating
            logic (e.g. "(cosine >= T OR rel_l2 <= T) AND agreement >= T").
        message: Summary message for display.
    """

    stage_name: str
    status: str = StageStatus.FAILED.value
    metrics: Dict[str, "MetricResult"] = field(default_factory=dict)
    composite_rule: str = ""
    message: str = ""

    @property
    def passed(self) -> bool:
        """Backward-compatible boolean: True when status is 'passed'."""
        return self.status == StageStatus.PASSED.value


@dataclass
class E2EResult:
    """Final structured result of an E2E test run for one model case.

    Serialized to a single consolidated ``result.json`` in the artifacts
    directory.  Contains everything previously spread across case.json,
    env_fingerprint.json, commands.json, and per-stage files.

    Attributes:
        case_name: The E2ECase.name this result belongs to.
        status: Terminal status (pass/fail/skip/error).
        failure_type: If status is fail/error, which phase failed.
        oracle_level: Oracle strength for this run (from the case).
        stages: Per-stage comparison results.
        determinism: Results of determinism reruns (if any).
        timing: Phase-level timing (bundle_build_s, trt_run_s, ref_run_s, etc.).
        detailed_timing: Normalized timing categories for report tables
            (weights_loading_s, trt_compile_s, inference_s, comparison_s, etc.).
        env_fingerprint: Environment info (GPU, driver, TRT version, etc.).
        timestamp: ISO 8601 timestamp of when this result was produced.
        repro_commands: Shell commands to reproduce each phase of the test.
        case_config: Snapshot of the E2ECase configuration (was case.json).
        commands: Subprocess command log (was commands.json).
        stage_outputs: Per-stage TRT/ref output summaries (was stages/ dir).
        artifacts: Map of artifact type to relative path(s) within model dir.
        log_file: Relative path to the merged log file (e2e_run.log).
    """

    case_name: str
    status: str = E2EStatus.PASS.value
    failure_type: Optional[str] = None
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value
    stages: Dict[str, CompareResult] = field(default_factory=dict)
    determinism: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    detailed_timing: Dict[str, float] = field(default_factory=dict)
    env_fingerprint: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    repro_commands: Dict[str, str] = field(default_factory=dict)
    case_config: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    log_file: str = ""


@dataclass
class RunContext:
    """Runtime context passed to strategy runners and reference backends.

    Carries resolved paths, flags, and the case definition needed to
    execute a stage.

    Attributes:
        case: The E2ECase being executed.
        artifacts_dir: Directory for writing stage outputs and logs.
        binary_path: Path to the C++ trtmc binary.
        hf_python: Base Python interpreter used for the default "base" profile.
        build_python: Resolved Python interpreter for bundle build.
        runtime_python: Resolved Python interpreter for TRT-side Python helpers.
        reference_python: Resolved Python interpreter for reference backends.
        build_profile: Symbolic profile name selected for build.
        runtime_profile: Symbolic profile name selected for runtime helpers.
        reference_profile: Symbolic profile name selected for references.
        ld_library_path: LD_LIBRARY_PATH with TRT/CUDA libs.
        engine_dir: Directory containing .trtfb bundles.
        model_plugin_dir: Directory containing isolated model plugin DSOs.
        rebuild: If True, force rebuild bundles from HF.
        verbose: If True, emit extra debug output.
    """

    case: E2ECase
    artifacts_dir: str = ""
    binary_path: str = ""
    hf_python: str = ""
    build_python: str = ""
    runtime_python: str = ""
    reference_python: str = ""
    build_profile: str = "base"
    runtime_profile: str = "base"
    reference_profile: str = "base"
    ld_library_path: str = ""
    engine_dir: str = ""
    model_plugin_dir: str = ""
    rebuild: bool = False
    verbose: bool = False

    def build_python_path(self) -> str:
        """Interpreter for bundle build subprocesses."""
        return self.build_python or self.hf_python

    def runtime_python_path(self) -> str:
        """Interpreter for TRT-side Python helper subprocesses."""
        return self.runtime_python or self.hf_python

    def reference_python_path(self) -> str:
        """Interpreter for reference backend subprocesses."""
        return self.reference_python or self.hf_python

    def runtime_cli_hf_python(self) -> str:
        """Optional --hf-python value for the C++ CLI."""
        if not bool(self.case.metadata.get("runtime_cli_requires_hf_python")):
            return ""
        return self.runtime_python_path()


@dataclass(frozen=True)
class PluginRuntimeContext:
    """Resolved runtime paths made available to contract plugins.

    This context is intentionally narrower than :class:`RunContext`: it contains
    only paths that contract plugins may need for validation helpers. Empty
    strings mean the corresponding path was not configured for this run.

    Attributes:
        engine_dir: Directory containing resolved TensorRT engine bundles.
        binary_path: Path to the TensorRT-Model-Connect CLI binary.
        hf_python: Optional Python interpreter value passed as ``--hf-python``
            to compatible runtime commands.
        runtime_python: Python interpreter for TRT-side helper subprocesses.
        reference_python: Python interpreter for reference/validation tools.
        artifacts_dir: Directory where this case writes E2E artifacts.
        model_plugin_dir: Directory containing isolated model plugin DSOs.
    """

    engine_dir: str = ""
    binary_path: str = ""
    hf_python: str = ""
    runtime_python: str = ""
    reference_python: str = ""
    artifacts_dir: str = ""
    model_plugin_dir: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskStrategyRunner(Protocol):
    """Executes the TRT inference path for a specific task strategy.

    Implementations handle the details of invoking the C++ binary or Python
    debug runner for their task type (text generation, VL, diffusion, etc.).
    """

    @property
    def strategy_name(self) -> str:
        """Unique identifier matching a task_strategy value."""
        ...

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Execute one stage and return its output."""
        ...


@runtime_checkable
class ReferenceBackendRunner(Protocol):
    """Executes the reference inference path for comparison.

    Implementations may use HF Transformers, Diffusers, custom torch code,
    or load golden snapshots.
    """

    @property
    def backend_name(self) -> str:
        """Unique identifier matching a reference_backend value."""
        ...

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Execute one reference stage and return its output."""
        ...


@runtime_checkable
class Comparator(Protocol):
    """Compares TRT output against reference output for a task strategy.

    Implementations compute task-appropriate metrics (cosine similarity for
    text, mIoU for segmentation, PSNR for diffusion, etc.) and gate on
    the provided threshold profile.
    """

    @property
    def task_strategy(self) -> str:
        """The task strategy this comparator handles."""
        ...

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Compare TRT vs reference outputs and return gated result."""
        ...


@runtime_checkable
class ReproCommandProvider(Protocol):
    """Builds model-owned repro commands for E2E result artifacts.

    Implementations should return argv-style tokens for the TRT inference
    command, or ``None`` when the provider does not handle the case. The
    orchestrator handles generic wrapping and string rendering.
    """

    @property
    def family_name(self) -> str:
        """Model family this provider owns."""
        ...

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> Optional[List[str]]:
        """Build the TRT inference repro command for this case."""
        ...


@runtime_checkable
class ArtifactSink(Protocol):
    """Persists commands, stage outputs, comparison results, and final report.

    Implementations write to disk (the common case) or could aggregate
    results in memory for testing.
    """

    def log_command(self, command: List[str], rc: int, stdout: str, stderr: str) -> None:
        """Record a subprocess invocation and its output."""
        ...

    def write_stage_output(self, name: str, output: StageOutput) -> None:
        """Persist one stage's output to artifacts."""
        ...

    def write_compare(self, name: str, result: CompareResult) -> None:
        """Persist one stage's comparison result."""
        ...

    def register_artifact(self, key: str, rel_path: str) -> None:
        """Register a modality artifact (PNG, WAV, NPY) by key and path."""
        ...

    def finalize(self, result: E2EResult) -> str:
        """Write the final result.json and return its path."""
        ...


@runtime_checkable
class DeterminismPolicy(Protocol):
    """Checks TRT output reproducibility across multiple runs.

    Called with outputs from N reruns of the same stage. Returns a
    CompareResult indicating whether intra-run variance is acceptable.
    """

    def check(self, case: E2ECase, outputs: List[StageOutput]) -> CompareResult:
        """Evaluate determinism across multiple stage outputs."""
        ...

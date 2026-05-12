"""Domain types and protocol contracts for the unified E2E testing framework.

This module is the stable foundation that all other harness components depend on.
It contains only stdlib imports (dataclasses, enum, typing) and defines:

- Domain dataclasses: E2ECase, StageSpec, PreflightRequirement, ThresholdProfile,
  StageOutput, CompareResult, E2EResult, RunContext
- Enums: FailureType, OracleLevel, E2EStatus
- Protocols: TaskStrategyRunner, ReferenceBackendRunner, Comparator,
  ArtifactSink, DeterminismPolicy
- Constants: RUNTIME_TO_TASK_STRATEGY mapping

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

    See docs/context/plans/e2e_ci_single_source_of_truth.md Section 5 for full definitions.
    """

    # --- Text / causal ---
    CAUSAL_BASE_CONTINUATION = "causal_base_continuation"
    CODE_BASE_COMPLETION = "code_base_completion"
    CHAT_INSTRUCT_TEMPLATE = "chat_instruct_template"
    CHAT_QWEN3_POSTTRAINED = "chat_qwen3_posttrained"
    MULTIMODAL_CHAT_QWEN35 = "multimodal_chat_qwen35"
    TRANSLATION_CHAT_TEMPLATE = "translation_chat_template"

    # --- Seq2seq ---
    SEQ2SEQ_TEXT2TEXT = "seq2seq_text2text"
    SEQ2SEQ_TRANSLATION = "seq2seq_translation"
    SEQ2SEQ_BASE_WEAK = "seq2seq_base_weak"

    # --- Encoder / embedding / rerank ---
    ENCODER_BASE_FEATURES = "encoder_base_features"
    DPR_CONTEXT_EMBED = "dpr_context_embed"
    SENTENCE_TRANSFORMER_EMBED = "sentence_transformer_embed"
    BGE_RETRIEVAL_EMBED = "bge_retrieval_embed"
    VL_EMBED_RETRIEVAL = "vl_embed_retrieval"
    VL_RERANK = "vl_rerank"

    # --- Vision-language ---
    VL_INSTRUCT_QA = "vl_instruct_qa"
    OCR_MARKDOWN = "ocr_markdown"

    # --- Speech / audio ---
    ASR_WHISPER = "asr_whisper"
    ASR_CANARY = "asr_canary"
    TTS_BARK = "tts_bark"
    TTS_MAGPIE = "tts_magpie"
    S2S_PERSONAPLEX = "s2s_personaplex"

    # --- Segmentation ---
    SEGMENTATION_SEGFORMER = "segmentation_segformer"
    PROMPTED_SEGMENTATION_SAM = "prompted_segmentation_sam"

    # --- Diffusion ---
    DIFFUSERS_IMAGE_GEN = "diffusers_image_gen"
    DIFFUSERS_VIDEO_GEN = "diffusers_video_gen"

    # --- Time series ---
    TIME_SERIES_POINT_FORECAST = "time_series_point_forecast"
    TIME_SERIES_QUANTILE_FORECAST = "time_series_quantile_forecast"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_REGRESSION = "time_series_regression"


class ArtifactType(enum.Enum):
    """Type of artifact produced by a pipeline stage.

    Used by StageSpec to declare what a stage outputs, enabling typed
    comparison dispatch and artifact persistence with schema validation.

    See docs/context/plans/e2e_ci_single_source_of_truth.md Section 7.2.
    """

    TOKEN_IDS = "token_ids"
    NORMALIZED_TEXT = "normalized_text"
    EMBEDDING = "embedding"
    CLASS_MAP = "class_map"
    SEMANTIC_TOKENS = "semantic_tokens"
    WAVEFORM = "waveform"
    IMAGE = "image"
    VIDEO_FRAMES = "video_frames"
    LOGITS = "logits"
    SCORE = "score"
    RANKING = "ranking"


class ComparisonMode(enum.Enum):
    """How to compare TRT output against reference output.

    Each mode implies specific metrics and gating logic.  A stage's
    artifact_type + comparison_mode together determine the comparator
    behavior.

    See docs/context/plans/e2e_ci_single_source_of_truth.md Section 8.
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
    INVARIANT_CHECK = "invariant_check"


class CILane(enum.Enum):
    """CI execution lane — determines when and how a test runs.

    See docs/context/plans/e2e_ci_single_source_of_truth.md Section 6.
    """

    ACCEPTANCE = "acceptance"
    NIGHTLY_PARITY = "nightly_parity"
    CROSSOVER_DEBUG = "crossover_debug"


class UserContract(enum.Enum):
    """The user-facing behavior that CI should verify for a model.

    Strong-behavior models have clear product contracts (chat response,
    transcript, etc.).  Weak-behavior models can only verify parity
    (continuation, representation).

    See docs/context/plans/e2e_ci_single_source_of_truth.md Section 3.3.
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
    EMBEDDING_VECTOR = "embedding_vector"
    RANKING_ORDER = "ranking_order"
    VL_ANSWER = "vl_answer"
    OCR_TEXT = "ocr_text"
    DIFFUSION_IMAGE = "diffusion_image"
    DIFFUSION_VIDEO = "diffusion_video"

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
              and "timeout_s".
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
        name: Stage identifier (e.g. "generate", "t5_encode", "vae_decode").
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
    """Complete definition of one E2E test case (one model).

    Loaded from a manifest v2 JSON file. Contains everything needed to
    build, run, compare, and report for a single model.

    Attributes:
        name: Unique case identifier (e.g. "qwen3-0.6b").
        hf_id: HuggingFace model ID or local path.
        family: Family plugin name (e.g. "qwen", "llama").
        runtime_strategy: C++ runtime backend selector (e.g. "decoder_kv_cache").
        task_strategy: Logical task type derived from runtime_strategy
            (e.g. "text_generation_causal"). See RUNTIME_TO_TASK_STRATEGY.
        reference_backend: Which reference implementation to compare against
            (e.g. "hf_transformers", "torch_reference", "golden_snapshot").
        oracle_level: Strength of the reference oracle. See OracleLevel.
        bundle: Bundle filename (e.g. "qwen3-0.6b.trtfb").
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
        """Derive task_strategy from runtime_strategy if not set."""
        if not self.task_strategy:
            self.task_strategy = RUNTIME_TO_TASK_STRATEGY.get(
                self.runtime_strategy, self.runtime_strategy
            )


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
        if str(self.case.runtime_strategy or "") not in {"speech_to_speech"}:
            return ""
        return self.runtime_python_path()


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

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
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

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
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
class ArtifactSink(Protocol):
    """Persists commands, stage outputs, comparison results, and final report.

    Implementations write to disk (the common case) or could aggregate
    results in memory for testing.
    """

    def log_command(
        self, command: List[str], rc: int, stdout: str, stderr: str
    ) -> None:
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

    def check(
        self, case: E2ECase, outputs: List[StageOutput]
    ) -> CompareResult:
        """Evaluate determinism across multiple stage outputs."""
        ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-model reference family classification.
# Keys are manifest "name" values; values are ReferenceFamily enum values.
# Derived from docs/context/plans/e2e_ci_single_source_of_truth.md Section 5.
MODEL_REFERENCE_FAMILY: Dict[str, str] = {
    # 5.1 CAUSAL_BASE_CONTINUATION
    "gpt2-125m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "gpt-neo-125m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "pythia-70m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "bloom-560m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "opt-125m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "granite-3.1-2b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "glm-4-9b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "deepseek-v2-lite": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "falcon-rw-1b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "stablelm2-1.6b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "xglm-564m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "mamba-130m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "rwkv-169m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "olmo-1b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "olmo2-1b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "minitron-4b-depth": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "minitron-4b-width": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "nemotron-hindi-4b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "falcon3-1b": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "deepseek-v2-tiny": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "mixtral-stories-15m": ReferenceFamily.CAUSAL_BASE_CONTINUATION.value,
    "nemotron-h-nano-9b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    # 5.2 CODE_BASE_COMPLETION
    "codegen-350m": ReferenceFamily.CODE_BASE_COMPLETION.value,
    "starcoder2-3b": ReferenceFamily.CODE_BASE_COMPLETION.value,
    # 5.3 CHAT_INSTRUCT_TEMPLATE
    "gemma-2-2b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "mistral-7b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "phi3-mini": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "phi-moe": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "gpt-oss-20b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "nemotron-mini-4b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "tinyllama-1.1b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "internlm2-1.8b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    "nemotron-nano-4b": ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value,
    # 5.4 CHAT_QWEN3_POSTTRAINED
    "qwen3-0.6b": ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value,
    "qwen3-0.6b-fp16": ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value,
    "qwen3-4b-instruct-2507": ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value,
    "qwen3-moe-30b-a3b": ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value,
    # 5.5 MULTIMODAL_CHAT_QWEN35
    "qwen35-9b": ReferenceFamily.MULTIMODAL_CHAT_QWEN35.value,
    # 5.6 TRANSLATION_CHAT_TEMPLATE
    "riva-translate-4b": ReferenceFamily.TRANSLATION_CHAT_TEMPLATE.value,
    # 5.7 SEQ2SEQ_TEXT2TEXT
    "t5-small": ReferenceFamily.SEQ2SEQ_TEXT2TEXT.value,
    # 5.8 SEQ2SEQ_TRANSLATION
    "marian-en-ru": ReferenceFamily.SEQ2SEQ_TRANSLATION.value,
    "nllb-200": ReferenceFamily.SEQ2SEQ_TRANSLATION.value,
    "nllb-200-distilled-600m": ReferenceFamily.SEQ2SEQ_TRANSLATION.value,
    # 5.9 SEQ2SEQ_BASE_WEAK
    "bart-base": ReferenceFamily.SEQ2SEQ_BASE_WEAK.value,
    # 5.10 ENCODER_BASE_FEATURES
    "albert-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "bert-base-uncased": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "roberta-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "roberta-large": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "xlm-roberta-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "camembert-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "deberta-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "distilbert-base-uncased": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "modernbert-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "convbert-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "fnet-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "xlnet-base": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    "electra-base-discriminator": ReferenceFamily.ENCODER_BASE_FEATURES.value,
    # 5.11 DPR_CONTEXT_EMBED
    "dpr-ctx-encoder": ReferenceFamily.DPR_CONTEXT_EMBED.value,
    # 5.12 SENTENCE_TRANSFORMER_EMBED
    "all-minilm-l6-v2": ReferenceFamily.SENTENCE_TRANSFORMER_EMBED.value,
    "all-mpnet-base-v2": ReferenceFamily.SENTENCE_TRANSFORMER_EMBED.value,
    "paraphrase-multilingual-minilm-l12-v2": ReferenceFamily.SENTENCE_TRANSFORMER_EMBED.value,
    # 5.13 BGE_RETRIEVAL_EMBED
    "bge-small-en-v1.5": ReferenceFamily.BGE_RETRIEVAL_EMBED.value,
    # 5.14 VL_EMBED_RETRIEVAL
    "eagle-embed-vl-1b-v2": ReferenceFamily.VL_EMBED_RETRIEVAL.value,
    "nemotron-embed-vl-1b-v2": ReferenceFamily.VL_EMBED_RETRIEVAL.value,
    # 5.15 VL_RERANK
    "eagle-rerank-vl-1b-v2": ReferenceFamily.VL_RERANK.value,
    # 5.16 VL_INSTRUCT_QA
    "qwen25vl-3b": ReferenceFamily.VL_INSTRUCT_QA.value,
    "qwen3-vl-2b": ReferenceFamily.VL_INSTRUCT_QA.value,
    "internvl3-8b": ReferenceFamily.VL_INSTRUCT_QA.value,
    "phi4-multimodal": ReferenceFamily.VL_INSTRUCT_QA.value,
    # 5.17 OCR_MARKDOWN
    "deepseek-ocr": ReferenceFamily.OCR_MARKDOWN.value,
    # 5.18 ASR_WHISPER
    "whisper-tiny-fp16": ReferenceFamily.ASR_WHISPER.value,
    "whisper-large-v3-turbo": ReferenceFamily.ASR_WHISPER.value,
    # 5.19 ASR_CANARY
    "canary-1b-v2": ReferenceFamily.ASR_CANARY.value,
    "nemotron-speech-streaming-en-0.6b": ReferenceFamily.ASR_CANARY.value,
    # 5.20 TTS_BARK
    "bark-small": ReferenceFamily.TTS_BARK.value,
    "bark-large": ReferenceFamily.TTS_BARK.value,
    # 5.21 TTS_MAGPIE
    "magpie-tts-357m": ReferenceFamily.TTS_MAGPIE.value,
    # 5.22 S2S_PERSONAPLEX
    "personaplex-7b": ReferenceFamily.S2S_PERSONAPLEX.value,
    # 5.23 SEGMENTATION_SEGFORMER
    "segformer-b0-ade": ReferenceFamily.SEGMENTATION_SEGFORMER.value,
    # 5.24 PROMPTED_SEGMENTATION_SAM
    "sam-vit-base": ReferenceFamily.PROMPTED_SEGMENTATION_SAM.value,
    # 5.25 DIFFUSERS_IMAGE_GEN
    "flux-schnell": ReferenceFamily.DIFFUSERS_IMAGE_GEN.value,
    "flux-2-dev": ReferenceFamily.DIFFUSERS_IMAGE_GEN.value,
    "pixart-sigma-1024": ReferenceFamily.DIFFUSERS_IMAGE_GEN.value,
    "z-image-turbo": ReferenceFamily.DIFFUSERS_IMAGE_GEN.value,
    # 5.26 DIFFUSERS_VIDEO_GEN
    "wan21-t2v-1.3b": ReferenceFamily.DIFFUSERS_VIDEO_GEN.value,
    # 5.27 TIME_SERIES_POINT_FORECAST
    "patchtst-granite-official": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
    "patchtsmixer-granite-official": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
    "timesfm-2.0-500m-official": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
    # 5.28 TIME_SERIES_QUANTILE_FORECAST
    "chronos-bolt-tiny-official": ReferenceFamily.TIME_SERIES_QUANTILE_FORECAST.value,
    # 5.29 TIME_SERIES_REGRESSION
    "patchtst-etth1-regression-distribution": ReferenceFamily.TIME_SERIES_REGRESSION.value,
}

# Reference family -> user contract mapping.
# Derived from docs/context/plans/e2e_ci_single_source_of_truth.md Sections 3.3 and 5.
REFERENCE_FAMILY_TO_USER_CONTRACT: Dict[str, str] = {
    ReferenceFamily.CAUSAL_BASE_CONTINUATION.value: UserContract.CONTINUATION_PARITY.value,
    ReferenceFamily.CODE_BASE_COMPLETION.value: UserContract.CODE_COMPLETION.value,
    ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value: UserContract.CHAT_RESPONSE.value,
    ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value: UserContract.CHAT_RESPONSE.value,
    ReferenceFamily.MULTIMODAL_CHAT_QWEN35.value: UserContract.CHAT_RESPONSE.value,
    ReferenceFamily.TRANSLATION_CHAT_TEMPLATE.value: UserContract.TRANSLATION.value,
    ReferenceFamily.SEQ2SEQ_TEXT2TEXT.value: UserContract.SEQ2SEQ_OUTPUT.value,
    ReferenceFamily.SEQ2SEQ_TRANSLATION.value: UserContract.TRANSLATION.value,
    ReferenceFamily.SEQ2SEQ_BASE_WEAK.value: UserContract.CONTINUATION_PARITY.value,
    ReferenceFamily.ENCODER_BASE_FEATURES.value: UserContract.REPRESENTATION_PARITY.value,
    ReferenceFamily.DPR_CONTEXT_EMBED.value: UserContract.EMBEDDING_VECTOR.value,
    ReferenceFamily.SENTENCE_TRANSFORMER_EMBED.value: UserContract.EMBEDDING_VECTOR.value,
    ReferenceFamily.BGE_RETRIEVAL_EMBED.value: UserContract.EMBEDDING_VECTOR.value,
    ReferenceFamily.VL_EMBED_RETRIEVAL.value: UserContract.EMBEDDING_VECTOR.value,
    ReferenceFamily.VL_RERANK.value: UserContract.RANKING_ORDER.value,
    ReferenceFamily.VL_INSTRUCT_QA.value: UserContract.VL_ANSWER.value,
    ReferenceFamily.OCR_MARKDOWN.value: UserContract.OCR_TEXT.value,
    ReferenceFamily.ASR_WHISPER.value: UserContract.EXACT_TRANSCRIPT.value,
    ReferenceFamily.ASR_CANARY.value: UserContract.EXACT_TRANSCRIPT.value,
    ReferenceFamily.TTS_BARK.value: UserContract.TTS_AUDIO.value,
    ReferenceFamily.TTS_MAGPIE.value: UserContract.TTS_AUDIO.value,
    ReferenceFamily.S2S_PERSONAPLEX.value: UserContract.SPEECH_RESPONSE.value,
    ReferenceFamily.SEGMENTATION_SEGFORMER.value: UserContract.SEGMENTATION_MASK.value,
    ReferenceFamily.PROMPTED_SEGMENTATION_SAM.value: UserContract.PROMPTED_MASK.value,
    ReferenceFamily.DIFFUSERS_IMAGE_GEN.value: UserContract.DIFFUSION_IMAGE.value,
    ReferenceFamily.DIFFUSERS_VIDEO_GEN.value: UserContract.DIFFUSION_VIDEO.value,
    ReferenceFamily.TIME_SERIES_POINT_FORECAST.value: UserContract.TIME_SERIES_POINT_FORECAST.value,
    ReferenceFamily.TIME_SERIES_QUANTILE_FORECAST.value: UserContract.TIME_SERIES_QUANTILE_FORECAST.value,
    ReferenceFamily.TIME_SERIES_CLASSIFICATION.value: UserContract.TIME_SERIES_CLASSIFICATION.value,
    ReferenceFamily.TIME_SERIES_REGRESSION.value: UserContract.TIME_SERIES_REGRESSION.value,
}

# Reference family -> default comparison mode mapping.
# Derived from docs/context/plans/e2e_ci_single_source_of_truth.md Section 8.
REFERENCE_FAMILY_TO_COMPARISON_MODE: Dict[str, str] = {
    ReferenceFamily.CAUSAL_BASE_CONTINUATION.value: ComparisonMode.PREFIX_TEXT.value,
    ReferenceFamily.CODE_BASE_COMPLETION.value: ComparisonMode.PREFIX_TEXT.value,
    ReferenceFamily.CHAT_INSTRUCT_TEMPLATE.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.CHAT_QWEN3_POSTTRAINED.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.MULTIMODAL_CHAT_QWEN35.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.TRANSLATION_CHAT_TEMPLATE.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.SEQ2SEQ_TEXT2TEXT.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.SEQ2SEQ_TRANSLATION.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.SEQ2SEQ_BASE_WEAK.value: ComparisonMode.NUMERIC_TENSOR.value,
    ReferenceFamily.ENCODER_BASE_FEATURES.value: ComparisonMode.NUMERIC_TENSOR.value,
    ReferenceFamily.DPR_CONTEXT_EMBED.value: ComparisonMode.STRUCTURED_RANKING.value,
    ReferenceFamily.SENTENCE_TRANSFORMER_EMBED.value: ComparisonMode.STRUCTURED_RANKING.value,
    ReferenceFamily.BGE_RETRIEVAL_EMBED.value: ComparisonMode.STRUCTURED_RANKING.value,
    ReferenceFamily.VL_EMBED_RETRIEVAL.value: ComparisonMode.STRUCTURED_RANKING.value,
    ReferenceFamily.VL_RERANK.value: ComparisonMode.STRUCTURED_RANKING.value,
    ReferenceFamily.VL_INSTRUCT_QA.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.OCR_MARKDOWN.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.ASR_WHISPER.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.ASR_CANARY.value: ComparisonMode.EXACT_TEXT.value,
    ReferenceFamily.TTS_BARK.value: ComparisonMode.SEMANTIC_JUDGMENT.value,
    ReferenceFamily.TTS_MAGPIE.value: ComparisonMode.SEMANTIC_JUDGMENT.value,
    ReferenceFamily.S2S_PERSONAPLEX.value: ComparisonMode.SEMANTIC_JUDGMENT.value,
    ReferenceFamily.SEGMENTATION_SEGFORMER.value: ComparisonMode.MASK_OVERLAP.value,
    ReferenceFamily.PROMPTED_SEGMENTATION_SAM.value: ComparisonMode.MASK_OVERLAP.value,
    ReferenceFamily.DIFFUSERS_IMAGE_GEN.value: ComparisonMode.MEDIA_SIMILARITY.value,
    ReferenceFamily.DIFFUSERS_VIDEO_GEN.value: ComparisonMode.MEDIA_SIMILARITY.value,
    ReferenceFamily.TIME_SERIES_POINT_FORECAST.value: ComparisonMode.NUMERIC_TENSOR.value,
    ReferenceFamily.TIME_SERIES_QUANTILE_FORECAST.value: ComparisonMode.NUMERIC_TENSOR.value,
    ReferenceFamily.TIME_SERIES_CLASSIFICATION.value: ComparisonMode.NUMERIC_TENSOR.value,
    ReferenceFamily.TIME_SERIES_REGRESSION.value: ComparisonMode.NUMERIC_TENSOR.value,
}

RUNTIME_TO_TASK_STRATEGY: Dict[str, str] = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_moe": "text_generation_causal",
    "ssm_recurrent": "text_generation_causal",
    "rwkv_recurrent": "text_generation_causal",
    "hybrid_mamba_attention": "text_generation_causal",
    "vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "speech_to_text_rnnt": "speech_to_text",
    "text_to_audio": "text_to_audio",              # legacy alias
    "text_to_audio_bark": "text_to_audio",
    "text_to_audio_magpie": "text_to_audio",
    "speech_to_speech": "speech_to_speech",
    "segmentation": "segmentation",
    "prompted_segmentation": "prompted_segmentation",
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "patchtst_torchtrt": "neural_operator",
    "patchtsmixer_torchtrt": "neural_operator",
    "timesfm_torchtrt": "neural_operator",
    "chronos_bolt_torchtrt": "neural_operator",
    "diffusion": "diffusion_media_generation",      # legacy alias
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_ltx": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_zimage": "diffusion_media_generation",
    "diffusion_pixart": "diffusion_media_generation",
    "torchtrt_diffusion": "torchtrt_diffusion",
    "diffusion_pixart_torchtrt": "diffusion_media_generation",
    "omni_multimodal": "omni_multimodal",
    "text_to_text": "text_generation_causal",
    "marian_translation": "text_generation_causal",
    "seq2seq_encoder_decoder": "text_generation_causal",
    "torchtrt_decoder": "text_generation_causal",
}

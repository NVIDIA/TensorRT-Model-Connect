from __future__ import annotations
import argparse
import html
import json
import math
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
from array import array
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence
import yaml
from apps.benchmark.performance.baselines.timing_contracts import timing_contract
from apps.benchmark.performance.baselines.hf_transformers import flatten_config
from trtmc_benchmark.catalog import ManifestCatalog, resolve_case, selected_task_for_case
from trtmc_benchmark.task_adapters import default_operation
from trtmc_benchmark.types import BenchmarkError
REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER_SOURCE = REPOSITORY / 'core/builder'
BENCHMARK_SOURCE = REPOSITORY / 'apps/benchmark'
MANIFEST_ROOT = REPOSITORY / 'families'
SUITE_SCHEMA = 'trtmc.perf-suite/v2'
ENVIRONMENT_SCHEMA = 'trtmc.perf-environment/v2'
RESULT_SCHEMA = 'trtmc.perf-matrix/v2'
REPORT_SCHEMA = 'trtmc.perf-report/v2'
PREPARATION_SCHEMA = 'trtmc.perf-bundle-preparation/v2'
TERMINAL_COMPARISONS = {'green', 'yellow', 'red'}
FINISHED_RESULTS = TERMINAL_COMPARISONS | {'contract-mismatch'}
HF_CACHE_ENVIRONMENT_NAMES = ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'HF_ASSETS_CACHE', 'HF_MODULES_CACHE', 'HF_XET_CACHE')
OUTPUT_CONTRACTS = {'audio-shape', 'classification-top-class', 'disparity-parity', 'embedding-shape', 'exact-text', 'exact-token-ids', 'forecast-shape', 'forecast-parity', 'head-scores-shape', 'regression-distribution', 'regression-values', 'generated-token-count', 'image-features-shape', 'localization', 'media-shape', 'metric-geometry-shape', 'molecular-structure-shape', 'normalized-text', 'ocr-text', 'offline-speech-shape', 'pose-refinement-shape', 'reranking-order', 'robot-action-shape', 'segmentation-shape', 'transcription-text'}
REFERENCE_INPUTS = {'pytorch-lerobot-act': (('source_root', 'lerobot_repo'),), 'upstream-elf': (('reference_repo', 'elf_repo'),), 'upstream-lance': (('reference_repo', 'lance_repo'),), 'upstream-sana-wm': (('reference_repo', 'sana_repo'), ('model_dir', 'sana_model')), 'pytorch-personaplex': (('official_repo', 'personaplex_repo'),), 'upstream-fast-foundation-stereo': (('model_dir', 'fast_foundation_stereo_model'),)}
REFERENCE_FIELDS = {'elf_repo', 'lance_repo', 'lerobot_repo', 'sana_repo', 'sana_model', 'personaplex_repo', 'fast_foundation_stereo_model'}
_TIMING_STABILITY_SAMPLE_COUNT = 10
_TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT = 5.0
_TIMING_STABILITY_MEDIAN_BAND_PERCENT = 5.0
_TIMING_STABILITY_MIN_IN_BAND = 8
class PerfMatrixError(RuntimeError):
    pass

@dataclass(frozen=True)
class Environment:
    name: str
    trtmc_bench: Path
    worker: Path
    hf_runner: Path
    task_runner: Path
    results_root: Path
    scratch_root: Path
    bundle_cache: Path
    bundle_roots: tuple[Path, ...]
    runtime_root: Path
    bundle_retention: str
    local_files_only: bool
    timeout_seconds: int
    references: Mapping[str, str]
    reference_python: Path = Path(sys.executable)
    storage_root: Path | None = None
    hf_cache_mode: str = 'shared'
    hf_cache_retention: str = 'retain'

@dataclass(frozen=True)
class ResolvedEntry:
    spec: Mapping[str, Any]
    model: Any
    case: Any
    manifest: Mapping[str, Any]
    reference_precision: str
    baseline_timing: Mapping[str, Any]


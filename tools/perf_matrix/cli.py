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
from .types import PerfMatrixError

from .core import *
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
'Run and report the TRTMC release performance matrix.'

for source in (REPOSITORY, BUILDER_SOURCE, BENCHMARK_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare', 'run'):
        command = commands.add_parser(name)
        command.add_argument('suite', type=Path)
        command.add_argument('--environment', required=True, type=Path)
        command.add_argument('--entry', action='append', default=[])
        command.add_argument('--model', action='append', default=[])
        command.add_argument('--model-selection', type=Path)
        command.add_argument('--verbose', action='store_true')
        command.add_argument('--allow-partial', action='store_true', help='skip release-catalog coverage checks for a selected QA suite')
        if name == 'prepare':
            command.add_argument('--output', required=True, type=Path)
        if name == 'run':
            command.add_argument('--no-build', action='store_true')
    resume = commands.add_parser('resume')
    resume.add_argument('run_directory', type=Path)
    resume.add_argument('--verbose', action='store_true')
    resume.add_argument('--no-build', action='store_true')
    report = commands.add_parser('report')
    report.add_argument('run_directory', type=Path)
    report.add_argument('--preparation-receipt', type=Path)
    return value

def main(argv: Sequence[str] | None=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command in {'check', 'prepare', 'run'}:
            suite_path, environment_path, suite_name, _selected, _excluded, environment, resolved = _common(arguments)
            if arguments.command == 'check':
                print(f'Ready: {len(resolved)} performance entrie(s)')
                return 0
            if arguments.command == 'prepare':
                return prepare_entries(resolved, environment, arguments.output, verbose=arguments.verbose)
            run_directory = _new_run_directory(environment.results_root)
            results = _initial_results(suite_path, environment_path, suite_name, environment, resolved, no_build=arguments.no_build)
            _write_json(run_directory / 'results.json', results)
            print(f'Run directory: {run_directory}')
            return _run_rows(run_directory, results, resolved, environment, no_build=arguments.no_build, verbose=arguments.verbose)
        if arguments.command == 'resume':
            run_directory = arguments.run_directory.resolve()
            results = _load_results(run_directory)
            suite_name, all_entries, excluded = load_suite(Path(results['suite_path']))
            environment = load_environment(Path(results['environment_path']))
            stored_ids = results.get('selected_entry_ids')
            if not isinstance(stored_ids, list) or not stored_ids or (not all((isinstance(value, str) and value for value in stored_ids))):
                raise PerfMatrixError('matrix results has no selected entry IDs')
            selected_ids = set(stored_ids)
            missing = selected_ids - {str(entry['id']) for entry in all_entries}
            if missing:
                raise PerfMatrixError('selected entries are missing from the suite: ' + ', '.join(sorted(missing)))
            _coverage(all_entries, excluded)
            selected = [entry for entry in all_entries if entry['id'] in selected_ids]
            resolved = preflight(selected, environment, require_runtime=True)
            results['suite'] = suite_name
            results['status'] = 'running'
            return _run_rows(run_directory, results, resolved, environment, no_build=arguments.no_build or bool(results.get('no_build')), verbose=arguments.verbose)
        if arguments.command == 'report':
            run_directory = arguments.run_directory.resolve()
            results = _load_results(run_directory, allow_legacy_report=True)
            preparation = _json_file(arguments.preparation_receipt.resolve(), 'preparation receipt') if arguments.preparation_receipt else None
            if preparation is not None and preparation.get('schema_version') != PREPARATION_SCHEMA:
                raise PerfMatrixError('preparation receipt has an unsupported schema')
            report = write_report(run_directory, results, preparation)
            print(f"{report['status']}: {report['summary']['comparable']}/{report['summary']['selected']} comparable")
            return 0
    except (OSError, PerfMatrixError, ValueError, yaml.YAMLError) as error:
        print(f'perf-matrix: {error}', file=sys.stderr)
        return 2
    return 2

if __name__ == '__main__':
    raise SystemExit(main())


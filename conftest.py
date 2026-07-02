# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def pytest_addoption(parser):
    opts = [
        ('--engine-dir', dict(default=None, help='Engine directory')),
        ('--trtmc-binary', dict(default=None, help='Path to trtmc binary')),
        ('--hf-python', dict(default=None, help='Python with HF tokenizers')),
        ('--model-plugin-dir', dict(default=None, help='Directory containing libtrtmc_model_*.so')),
        ('--rebuild-engines', dict(action='store_true', default=False, help='Rebuild bundles')),
        ('--e2e-task-strategy', dict(default=None, help='Filter by task strategy')),
        ('--e2e-model', dict(action='append', default=[],
                             help='Filter by E2E case name or family; repeat or comma-separate values')),
        ('--e2e-artifacts-dir', dict(default=None, help='Artifacts output dir')),
        ('--e2e-core-only', dict(action='store_true', default=False,
                                help='Only run core E2E models')),
        ('--e2e-exclude-ci-tier', dict(action='append', default=[],
                                      help='Exclude manifests with this ci_tier')),
        ('--e2e-models-file', dict(default=None,
                                  help='Only collect E2E models listed in this file')),
        ('--e2e-group-by-bundle', dict(action='store_true', default=False,
                                      help='Collect one E2E entry per selected bundle')),
        ('--multi-device-only', dict(action='store_true', default=False,
                                     help='Only run multi-device E2E models')),
        ('--e2e-platform', dict(default='',
                               help='Platform name used to select platform-prefixed waives')),
        ('--e2e-partition-id', dict(type=int, default=None,
                                   help='Agent partition ID for parallel E2E execution')),
        ('--e2e-partition-size', dict(type=int, default=None,
                                     help='Total number of E2E partitions')),
    ]
    for name, kw in opts:
        try:
            parser.addoption(name, **kw)
        except ValueError:
            pass


def pytest_collection_modifyitems(config, items):
    """Enforce exact E2E selection after model-owned parametrization."""
    from tests.e2e_harness.manifest_loader import load_all_manifests
    from tests.e2e_harness.model_selection import (
        case_matches_e2e_model,
        case_names_from_param,
        parse_e2e_model_filters,
        read_e2e_models_file,
    )

    models_file = config.getoption("--e2e-models-file", default=None)
    selected_names = read_e2e_models_file(models_file) if models_file else None
    model_filters = parse_e2e_model_filters(
        config.getoption("--e2e-model", default=[]) or []
    )
    if selected_names is None and not model_filters:
        return

    cases_by_name = (
        {case.name: case for case in load_all_manifests()}
        if selected_names is None
        else {}
    )

    kept = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or "case_name" not in callspec.params:
            kept.append(item)
            continue

        case_names = case_names_from_param(str(callspec.params["case_name"]))
        if selected_names is not None:
            matches = bool(case_names) and set(case_names).issubset(selected_names)
        else:
            cases = [cases_by_name.get(name) for name in case_names]
            matches = bool(cases) and all(
                case is not None and case_matches_e2e_model(case, model_filters)
                for case in cases
            )

        (kept if matches else deselected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

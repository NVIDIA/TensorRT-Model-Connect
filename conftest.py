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

def pytest_addoption(parser):
    opts = [
        ('--engine-dir', dict(default=None, help='Engine directory')),
        ('--trtmc-binary', dict(default=None, help='Path to trtmc binary')),
        ('--hf-python', dict(default=None, help='Python with HF tokenizers')),
        ('--rebuild-engines', dict(action='store_true', default=False, help='Rebuild bundles')),
        ('--e2e-task-strategy', dict(default=None, help='Filter by task strategy')),
        ('--e2e-artifacts-dir', dict(default=None, help='Artifacts output dir')),
        ('--e2e-core-only', dict(action='store_true', default=False,
                                help='Only run core E2E models')),
        ('--e2e-exclude-ci-tier', dict(action='append', default=[],
                                      help='Exclude manifests with this ci_tier')),
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

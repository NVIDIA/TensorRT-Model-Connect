# Bundle Helpers

No behavior-bearing helpers for the public C-linkage C++ subset remain in this
directory.

Current owners:

- `src/bundle/bundle_format.h/cpp`: bundle format read/write helpers.
- `src/bundle/bundle_view.h/cpp`: bundle section lookup and access.
- `src/runtime/registry/pipeline_factory.cpp`: bundle-kind selection and
  pipeline composition.
- `src/runtime/providers/optimized_runtime_host.cpp`: verification and loading
  of an optimized bundle's embedded implementation.
- `src/tokenizer/`: tokenizer implementations used by model-owned pipelines.

Keep new model-specific engine setup and tokenizer policy under the owning
`python/tensorrt_model_connect/models/<family>/runtime/` directory.

<!-- Collaborative review anchor: batch 2. -->

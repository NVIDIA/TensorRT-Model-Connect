# Bundle Helpers

No behavior-bearing C ABI bundle helpers remain in this directory.

Current owners:

- `src/bundle/bundle_format.h/cpp`: bundle format read/write helpers.
- `src/bundle/bundle_view.h/cpp`: bundle section lookup and access.
- `src/runtime/registry/pipeline_factory.cpp`: bundle materialization and
  model-plugin composition.
- `src/tokenizer/`: tokenizer implementations used by model-owned pipelines.

Keep new model-specific engine setup and tokenizer policy under the owning
`src/runtime/models/<family>/` directory.

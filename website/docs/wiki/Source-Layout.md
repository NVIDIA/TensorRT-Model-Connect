# Source Layout

The maintained, verified directory map is
[Source Layout](../reference/source-layout.md).

:::info Why this Wiki page is short

The former version duplicated hundreds of paths, model counts, strategy names,
and test filenames. That copy drifted as model ownership moved into
per-family descriptors. The reference page now documents stable directory
roles, while the three `MODEL.toml` trees remain the machine-readable source
of truth for native family/build/runtime/E2E ownership.

:::

Use these ownership roots:

```text
python/tensorrt_model_connect/families/<builder-family>/MODEL.toml
src/runtime/models/<runtime-owner>/MODEL.toml
tests/e2e/models/<e2e-family>/MODEL.toml
```

Each descriptor `id` matches its own directory. Names normally match across
the roots, but the exact `runtime_strategy` is authoritative when they differ:
current builder/E2E owners `magpie_tts` and `wan_t2v` map to runtime owners
`magpie` and `wan`.

An optimized implementation stays family-owned but uses a separate contract:

```text
python/tensorrt_model_connect/families/<builder-family>/<implementation>/IMPLEMENTATION.toml
python/tensorrt_model_connect/families/<builder-family>/<implementation>/profiles/*.toml
tests/e2e/models/<e2e-family>/<implementation>/QUALIFICATION.<target>.toml
```

`IMPLEMENTATION.toml` owns the isolated build adapter and embedded runtime
library identity. Profiles make exact model revision, target, operation,
precision, quantization, and shape support claims. `QUALIFICATION.*.toml` owns
the target-specific proof workflow. These files complement rather than replace
the three native `MODEL.toml` trees.

Do not rely on removed root-level `graph_ops.py`/`graph_blocks.py`, shared
encoder backends, or a central runtime plugin directory. Model-semantic code
belongs to the owning family.

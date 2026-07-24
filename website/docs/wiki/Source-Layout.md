# Source Layout

The maintained, verified directory map is
[Source Layout](../reference/source-layout.md).

:::info Why this Wiki page is short

The former version duplicated hundreds of paths, model counts, strategy names,
and test filenames. That copy drifted as model ownership moved into
per-family descriptors. The reference page now documents stable directory
roles, while the three `MODEL.toml` trees remain the machine-readable source
of truth.

:::

Use these ownership roots:

```text
python/tensorrt_model_connect/families/<family>/MODEL.toml
src/runtime/models/<family>/MODEL.toml
tests/e2e/models/<family>/MODEL.toml
```

Do not rely on removed root-level `graph_ops.py`/`graph_blocks.py`, shared
encoder backends, or a central runtime plugin directory. Model-semantic code
belongs to the owning family.

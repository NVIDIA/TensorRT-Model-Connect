---
title: Add a Model Family
---

Add a Python family plugin when the model can reuse an existing runtime strategy.

## 1. Create a plugin file

Create `python/tensorrt_model_connect/families/<family>.py`:

```python
from __future__ import annotations

from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..config import ModelConfig
from ..standard_decoder_builder import build_standard_decoder_engine


class YiPlugin:
    name = "yi"
    runtime_strategy = "decoder_kv_cache"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("yi")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
        )


plugin = YiPlugin()
```

Auto-discovery registers modules with a module-level `plugin` attribute. No central Python registration edit is required.

## 2. Build a smoke bundle

```bash
./build/trtmc build <hf-repo-or-local-dir> -o /tmp/family-smoke.trtfb --max-cache-length 256
```

## 3. Validate

```bash
./scripts/validate_family.sh <hf-repo-or-local-dir>
```

If the model requires remote tokenizer or modeling code, pass the relevant trust flag through the validation flow.

## 4. Add an E2E manifest

Create `tests/e2e/models/<family>/manifests/<model-name>.json` with:

- `name`
- `hf_id`
- `bundle`
- `family`
- `runtime_strategy`
- task input fields
- reference backend

Use nearby manifests with the same task contract as the template.
List the manifest in `tests/e2e/models/<family>/MODEL.toml` under `test_manifests`.

Each family directory also owns its E2E runner surface:

- `tests/e2e/models/<family>/runner.py`
- `tests/e2e/models/<family>/test_<family>_e2e.py`
- `tests/e2e/models/<family>/e2e_plugins/*.py`
- `tests/e2e/models/<family>/thresholds/<model-name>.json`
- optional `tests/e2e/models/<family>/waives.txt`

For a new family, copy the runner and test shim from the closest existing
family and update only the family name in the docstrings and filename. Put any
family-specific runner, reference, or comparator overrides in `e2e_plugins/`
with module-level `runner`, `reference`, or `comparator` objects. Validate with:

```bash
pytest tests/e2e/models/<family> --e2e-model <model-name> -v \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models
```

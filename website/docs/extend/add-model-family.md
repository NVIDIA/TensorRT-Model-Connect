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

Create `tests/e2e/models/<model-name>.json` with:

- `name`
- `hf_id`
- `bundle`
- `family`
- `runtime_strategy`
- task input fields
- thresholds or reference backend

Use nearby manifests with the same task contract as the template.

---
title: Quantization
---

Quantization is a family-owned build path. The shared `BuildRequest` carries a
plain `quantization` string, but the selected `families/<family>/model.py` owns
the accepted values, calibration, graph edits, weight formats, exclusions,
bundle sections, and validation.

## Build an explicitly supported path

```bash
python -m tensorrt_model_connect build MODEL \
  --output model-fp8.bundle \
  --precision fp16 \
  --quantization fp8
```

Do not infer support from the shared option. Check the exact family builder and
one matching manifest under `families/<family>/tests/manifests/` first. An
unsupported value must fail; core does not reroute the request.

Current examples include family-specific FP8 paths in Qwen, FLUX, and Wan2.2
TI2V. Their accepted precisions, topology, calibration inputs, and output
evidence differ.

## Ownership standard

Quantization-specific implementation belongs in the family that consumes it:

```text
families/<family>/
├── model.py
├── quantization.py          # only when that family needs it
├── requirements.txt         # optional calibration dependencies
└── tests/
```

A family may call ModelOpt or consume already quantized weights if that exact
path requires it. ModelOpt is not a core runtime dependency. Calibration does
not run when a native bundle is loaded.

The normal contribution boundary is `families/<family>/**`. Do not introduce a
shared quantization registry, base class, profile hierarchy, or digest layer.
Copying a small implementation is preferable to coupling unrelated families.

## Qwen FP8 example

The current Qwen family accepts `fp8` only for the combinations enforced in
`families/qwen/model.py`. It calibrates through
`families/qwen/quantization.py` and owns its ModelOpt dependency in
`families/qwen/requirements.txt`.

```bash
python -m pip install -r families/qwen/requirements.txt
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --output qwen3-fp8.bundle \
  --precision fp16 \
  --quantization fp8
```

Use the Qwen FP8 manifest and family E2E for the exact correctness contract. A
successful engine build alone does not prove parity.

## FLUX and Wan2.2

FLUX and Wan2.2 TI2V also interpret `quantization=fp8` inside their own
builders. They do not import Qwen calibration code and they do not establish a
generic FP8 contract for another diffusion family.

Follow the matching family manifest for checkpoint preparation, precision,
shape, task, and oracle. Keep target-specific benchmark evidence separate from
correctness evidence.

## Validation

At minimum, record:

1. the exact family manifest and checkpoint revision;
2. the build command and selected family dependencies;
3. the produced bundle's `family`, `task`, and `backend`;
4. the family-owned parity or quality result;
5. target hardware only when that target was actually run.

The runtime reads the already built engine and family sections. It does not
recalibrate, consult a central quantization policy, or verify a repository-added
content hash.

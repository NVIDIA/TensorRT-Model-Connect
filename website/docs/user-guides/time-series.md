---
title: Time-Series
description: Build and run forecasting and neural-operator task bundles.
---

Time-series interfaces consume little-endian float32 files rather than
comma-separated command-line values. The family owns shape, ordering,
precision, and result semantics.

## Forecasting

```bash
python -c 'from array import array; array("f", [100.1,100.15,100.18,100.22,100.21,100.27]).tofile(open("values.f32", "wb"))'

python -m tensorrt_model_connect build amazon/chronos-bolt-tiny \
  --precision fp32 \
  --output forecast.bundle

trtmc forecast forecast.bundle \
  --runtime-root /opt/trtmc/lib \
  --input values.f32 \
  --frequency 0
```

An optional `--mask` file must contain the same number of float32 values as
`--input`.

## Neural operators

`solve` takes separate branch and trunk float32 files:

```bash
trtmc solve operator.bundle \
  --runtime-root /opt/trtmc/lib \
  --branch branch.f32 \
  --trunk trunk.f32
```

Do not interchange `forecast` and `solve`. Confirm the exact Task interface and
input contract in the selected family's manifest under
`families/<family>/tests/manifests/`. Success is a zero exit status plus values
that pass the family-owned comparator; a plausible vector alone is not parity
evidence.

---
title: Time-Series
description: Build and run numeric forecasting and neural-operator task bundles.
---

Time-series forecasting bundles expose `forecast()`. Neural-operator bundles
expose `solve()`. The exact Task contract determines the required raw float32
input files.

```bash
trtmc forecast forecast.bundle \
  --input history.f32
```

Other model contracts can use:

```bash
trtmc solve operator.bundle \
  --branch branch.f32 \
  --trunk trunk.f32
```

Do not interchange these forms. Copy the input shape, ordering, precision, and
oracle from an exact manifest in the
[Time Series Forecasting recipes](/models-recipes/model-recipes/tasks/time-series-forecast).
Success is a zero exit status plus an output vector whose shape and values pass
the model-owned comparator; a plausible vector alone is not parity evidence.

{/* Collaborative review anchor: batch 2. */}

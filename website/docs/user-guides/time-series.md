---
title: Time-Series
description: Build and run numeric forecasting and neural-operator task bundles.
---

Time-series bundles expose `solve()` rather than text or media generation. The
exact model contract determines which numeric input form is valid.

```bash
trtmc solve forecast.bundle \
  --branch-input "100.1,100.15,100.18,100.22,100.21,100.27"
```

Other model contracts can use:

```bash
trtmc solve operator.bundle \
  --field-input "..."

trtmc solve operator.bundle \
  --branch-input "..." \
  --trunk-input "..."
```

Do not interchange these forms. Copy the input shape, ordering, precision, and
oracle from an exact manifest in the
[Time Series Forecasting recipes](/models-recipes/model-recipes/tasks/time-series-forecasting).
Success is a zero exit status plus an output vector whose shape and values pass
the model-owned comparator; a plausible vector alone is not parity evidence.

{/* Collaborative review anchor: batch 2. */}

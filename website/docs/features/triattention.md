---
title: TriAttention (Historical)
---

TriAttention was an experimental Qwen KV-cache compaction policy in the
pre-#1093 architecture. Its shared schema, build flags, bundle sections, and
runtime policy were removed during the family-isolated cutover. The current
CLI does not accept TriAttention options, and no family currently exposes it as
a supported Task behavior.

Do not use restored examples from the historical worklog as a current runbook
or support claim. Current Qwen-family dynamic KV behavior, when implemented,
is selected by the explicit `BuildRequest.dynamic_kv_cache` field and remains
entirely family-owned.

The original investigation, scoring/compaction design, experiments, and dated
evidence are retained in the
[TriAttention Native C++ Worklog](../context/triattention-native-cpp-worklog.md).
It remains useful for future design work, but reintroducing any policy must
follow the current boundaries:

- implementation and tests remain inside the owning family;
- shared code receives no family-specific schema or orchestration;
- unsupported requests fail explicitly;
- long-context quality, cache behavior, and performance require fresh evidence
  against the exact current family and checkpoint.

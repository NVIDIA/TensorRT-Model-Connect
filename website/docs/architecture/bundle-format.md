---
title: Bundle Format
---

import Diagram from '@site/src/components/Diagram';

A bundle contains eight magic bytes, an unsigned little-endian 64-bit JSON
header length, the UTF-8 JSON header, and concatenated named sections.

<Diagram
  src="/img/diagrams/architecture/bundle-layout.svg"
  alt="A minimal bundle contains a fixed prefix, bounded JSON header, and opaque family-owned sections"
  caption="Core validates byte ranges; the selected family owns every section name and meaning."
/>

```json
{
  "format": 1,
  "family": "gpt2",
  "task": "text_generation",
  "backend": "trt",
  "sections": {
    "runtime.json": {"offset": 0, "length": 42},
    "engine.plan": {"offset": 42, "length": 1234}
  }
}
```

`BundleReader` validates this exact shape and every section bound when it is
constructed. It owns the normalized bundle path and immutable section table,
then reads a requested section directly from the file. It has no write API and
does not eagerly load section contents.

The core does not interpret family sections or compute section hashes. Only
format 1 is supported.

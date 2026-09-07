---
title: Bundle Format
---

A bundle contains eight magic bytes, an unsigned little-endian 64-bit JSON
header length, the UTF-8 JSON header, and concatenated named sections.

```json
{
  "format": 1,
  "family": "gpt2",
  "task": "text_generation",
  "backend": "trt",
  "sections": {
    "provenance.json": {"offset": 0, "length": 391},
    "runtime.json": {"offset": 391, "length": 42},
    "engine.plan": {"offset": 433, "length": 1234}
  }
}
```

`BundleReader` validates this exact shape and every section bound when it is
constructed. It owns the normalized bundle path and immutable section table,
then reads a requested section directly from the file. It has no write API and
does not eagerly load section contents.

Every newly built bundle contains a `provenance.json` section. It records the
canonical checkpoint ID, the exact checkpoint commit, the exact TRTMC source
commit, and every build-affecting request option. `trtmc inspect BUNDLE`
returns this provenance alongside the fixed header and section table. A build
fails instead of publishing a bundle when its TRTMC source commit cannot be
resolved exactly.

Benchmark-managed bundles are reused only when all of this provenance matches
the selected manifest and current source revision. Bundles without provenance,
or with only a partial match, are rebuilt when builds are enabled and rejected
when `--no-build` is set.

The core does not interpret family sections or compute section hashes. Only
format 1 is supported.

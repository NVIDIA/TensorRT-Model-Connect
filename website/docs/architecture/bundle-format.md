---
title: Bundle Format
---

A bundle contains eight magic bytes, an unsigned little-endian 64-bit JSON
header length, the UTF-8 JSON header, concatenated family-owned named sections,
and an optional core-owned provenance trailer.

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

Every bundle produced through the CLI contains a provenance JSON
trailer followed by its unsigned little-endian 64-bit length and the eight-byte
`PROV\x01\x00\x00\x00` marker. The trailer is outside the section payload, so
it does not reserve a family section name or change the fixed v1 header. It
records the canonical checkpoint ID, its immutable revision, the exact TRTMC
source commit, and every build-affecting request option. A Hugging Face revision
is an exact commit; another source uses the provider's resolved version-object
ID, for example `ngc:version:1.0.1_onnx`. Mutable branches, channels, and aliases
are not valid provenance identities.

`trtmc inspect BUNDLE` returns the trailer alongside the fixed header and
section table. The CLI fails instead of publishing a bundle when checkpoint
identity is missing. Every build rejects a missing source identity, a dirty
locally inferred source checkout, or a graph transform without a stable
identity.

Benchmark-managed bundles are reused only when all of this provenance matches
the selected manifest and current source revision. Bundles without provenance,
or with only a partial match, are rebuilt when builds are enabled and rejected
when `--no-build` is set.

The core owns and validates the provenance envelope. It does not interpret
family sections or compute section hashes. Only format 1 is supported.

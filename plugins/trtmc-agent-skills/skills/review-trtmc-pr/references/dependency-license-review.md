# Dependency And License Review

Use this reference to answer three questions for the exact PR base/head:

1. Does the PR add or change third-party material?
2. Can an obligation from that material's license propagate into
   TensorRT-Model-Connect (TRTMC) source or artifacts, or into a program or
   artifact that also contains TensorRT?
3. What does carrying the dependency cost TRTMC?

This is engineering triage, not legal advice. Establish facts from
authoritative sources, apply the conservative defaults below, and route
anything the evidence cannot settle to `HUMAN DECISION` for the project's
legal or open-source review. Never clear a license from memory, popularity,
package name, or the publishing organization.

## 1. Inventory Third-Party Material

Material is new when the head adds a package, library, vendored or adapted
source, binary, external download, model/data/media asset, or a version change
whose license differs from the base. Also flag an existing dependency whose
integration mode changes, for example from a test-only import to a runtime
import or from a separate process to a linked library.

| Surface | Where it appears in TRTMC | Default reach |
| --- | --- | --- |
| Root Python | `pyproject.toml` dependencies and extras, `requirements/*.txt` | every user and family |
| Family Python | `families/<family>/requirements.txt` | that family's build/runtime |
| Test or app Python | `families/<family>/tests/requirements.txt`, `apps/<app>/requirements.txt` | tests or that application |
| Native libraries | `conanfile.py`, CMake `find_package`, `FetchContent`, `ExternalProject`, `target_link_libraries` | linked into a DSO or executable |
| Vendored or adapted code | `third_party/**`, copied files or snippets in family code | compiled or shipped as source |
| Undeclared use | new imports, includes, or `dlopen` targets absent from any declaration | hidden dependency |
| Fetched at build, CI, or run time | Dockerfiles, scripts, workflows, `trust_remote_code`, `torch.hub`, raw URLs | executes or redistributes third-party code |
| Committed binaries | `.so`, `.a`, `.whl`, `.onnx`, `.plan`, archives | redistributed as-is |
| Models, data, media | manifest `hf_id`/`hf_revision`, files a family copies into its bundle, fixtures, golden outputs, datasets | bundles, engine plans, repository |

Start from the diff, then read the surrounding code for each hit:

```bash
git diff --name-status <base>...<head>
git diff <base>...<head> -- pyproject.toml requirements conanfile.py \
  '*requirements*.txt' '*CMakeLists.txt' '*.cmake' '*Dockerfile*' \
  third_party NOTICE '*NOTICE' ASSET_LICENSES.md .github
git diff --numstat <base>...<head> | awk '$1 == "-" {print $3}'
git diff -U0 <base>...<head> | grep -E '^\+[^+]' | grep -E \
  '^\+\s*(import|from) |#include <|pip install|apt-get install|curl |wget |git clone|FetchContent|ExternalProject|find_package|target_link_libraries|dlopen|ctypes|trust_remote_code|torch\.hub|hf_hub_download|snapshot_download|Copyright|SPDX-License-Identifier|[Aa]dapted from|[Pp]orted from|[Dd]erived from'
```

The numstat line lists binary files. Separate standard-library and TRTMC
imports from third-party ones by hand. A new direct dependency can pull new
transitive ones. Resolve the delta relative to the pinned base environment
(the Community CI CPU image or the development container), without building or
installing anything:

```bash
python3 -c 'import tomllib; d = tomllib.load(open("pyproject.toml", "rb")); \
  print("\n".join(r for r in d["project"]["dependencies"] if "==" in r))' \
  > <scratch>/base-pins.txt
python3 -m venv --system-site-packages <scratch>/deps
<scratch>/deps/bin/pip install --dry-run --only-binary=:all: \
  -c <scratch>/base-pins.txt --report <scratch>/pip-report.json \
  -r <changed requirements file>
```

`--only-binary=:all:` prevents source builds from executing package code, and
the constraints stop the resolver from silently moving a base pin; a
resolution failure is itself a conflict finding. Every entry in the report's
`install` list is a new package or a version change against the base
environment, so treat each one as an inventory item. Unpinned family
requirements commonly pull a different torch build together with its own
CUDA runtime wheels.

## 2. Establish License Facts

For each item, record its name, exact version or revision, source, SPDX
expression, where the license evidence was read, and what form TRTMC would
ship (source, wheel, shared library, bundle, container layer).

- **Python packages:** read `METADATA` (`License-Expression`, `License`,
  `Classifier: License`) and the `licenses/` or `LICENSE*` files inside the
  exact wheel; fetch it with
  `pip download --no-deps --only-binary=:all: <name>==<version> -d <scratch>`.
  The pip report's metadata can be empty for newer metadata versions, so it
  is a pointer, not evidence.
- **Bundled native code in wheels:** list the wheel (`python3 -m zipfile -l`)
  and look for `*.libs/`, `_*_data/`, or other shared objects. Each bundled
  library carries its own license. For example, the `soundfile` 0.13.1 wheel
  declares BSD-3-Clause but bundles `libsndfile` under LGPL-2.1.
- **Native libraries:** read `LICENSE`/`COPYING` at the pinned tag, plus build
  options that change the license (for example FFmpeg GPL components or
  OpenCV non-free modules) and the license of the system package actually
  installed in the container.
- **Copied or adapted code:** keep the original header and confirm its source
  and revision. Code with no traceable source has unknown provenance.
- **Models, tokenizers, datasets, media:** read the card's `license` field and
  the license file at the pinned revision, including use restrictions,
  attribution or naming requirements, gating, and acceptable-use policies.
- **Missing, contradictory, or `NOASSERTION` licenses** are `UNKNOWN`.

## 3. Classify The License

| Class | Examples | Default disposition |
| --- | --- | --- |
| Permissive | MIT, MIT-0, BSD-2/3-Clause, Apache-2.0, ISC, Zlib, BSL-1.0, PSF-2.0, Unicode-3.0, CC0-1.0 | compatible; keep notices |
| Weak copyleft | LGPL-2.1/3.0, MPL-2.0, EPL-2.0, CDDL-1.0 | depends on integration mode (section 4) |
| Strong copyleft | GPL-2.0, GPL-3.0 | must stay outside every TRTMC or TensorRT program and artifact |
| Network copyleft | AGPL-3.0, SSPL-1.0 | never on a serving or runtime path |
| Restricted terms | CC-BY-NC*, research-only or non-commercial, BUSL, Commons Clause, model licenses with field-of-use or naming terms | `HUMAN DECISION` |
| Share-alike data | CC-BY-SA | `HUMAN DECISION` for committed derivatives |
| Unknown or third-party proprietary | missing, custom, contradictory | cannot be shipped until resolved |

Apache-2.0 is incompatible with GPL-2.0-only even without TensorRT.

## 4. Decide Whether Obligations Propagate

The license alone does not answer the question; the way the material meets
TRTMC and TensorRT does.

Boundary facts:

- TRTMC source is Apache-2.0 (root `LICENSE`). Third-party attributions live
  in the root `NOTICE`, family-owned `families/<family>/NOTICE`, and
  `ASSET_LICENSES.md`.
- TensorRT core libraries and CUDA are proprietary NVIDIA software that users
  install separately; only some TensorRT components, such as the open-source
  plugins and parsers, are open source. Every family DSO, TensorRT custom
  plugin, builder process, and runtime/server process loads TensorRT.
- A license whose obligations attach to the whole combined program (GPL,
  AGPL) cannot be satisfied by a program that also contains proprietary
  TensorRT. The result cannot be distributed, and the risk passes to every
  downstream user who ships TRTMC with TensorRT. This is the "propagation into
  TensorRT" case to rule out.
- Weights, tokenizer files, and data shape the artifacts built from them. A
  bundle or engine plan embeds checkpoint weights and any files the family
  copies into the bundle, so the checkpoint's license governs redistribution
  of those artifacts, fixtures, and golden outputs.

| Integration mode | Shares a program or artifact with TensorRT | GPL / AGPL | LGPL / MPL / EPL | Permissive |
| --- | --- | --- | --- | --- |
| Vendored or adapted into TRTMC source | yes once compiled | `BLOCK` | `BLOCK` (MPL only as separate unmodified files: `HUMAN DECISION`) | keep headers; record attribution |
| Statically linked into a DSO, executable, or plugin | yes | `BLOCK` | `BLOCK` (relinking obligation) | record attribution |
| TensorRT custom plugin code | yes, inside engine execution | `BLOCK` | `BLOCK` unless a separate, replaceable shared library | record attribution |
| Dynamically linked or `dlopen`ed by a family DSO | yes | `BLOCK` | unmodified, replaceable, user-installed: `OBLIGATIONS`; shipped by TRTMC: `HUMAN DECISION` | attribution when shipped |
| Imported in the builder, runtime, or server Python process | yes | `BLOCK` unless legal review clears it | `OBLIGATIONS` | — |
| Separate process (subprocess CLI, sidecar) | no | `HUMAN DECISION`: optional, not shipped, arm's-length data exchange | `OBLIGATIONS` | — |
| Build-time tool whose output is not derived from its licensed content | no | `HUMAN DECISION` (check output exceptions) | clear | clear |
| Test or CI only, never shipped | no | `HUMAN DECISION` (contributors and CI images still carry it) | clear | clear |
| Weights, tokenizer files, or data in bundles, plans, fixtures, golden outputs | bundle is loaded by TensorRT | the checkpoint or data license governs; restricted or share-alike: `HUMAN DECISION` | | |

Check every distribution channel the material reaches: the source
repository, the `tensorrt-model-connect` wheel, containers built from the
`Dockerfile*` files, bundles and engine plans, committed fixtures and golden
data, CI artifacts, and published reports. A tool used only locally still
distributes derived material when its output is committed.

Also check the reverse direction: TensorRT, CUDA, and cuDNN terms constrain
TRTMC. Flag a PR that copies their binaries or headers into the repository,
wheel, or bundle, or that adds a dependency shipping its own TensorRT, CUDA,
or `nvidia-*` runtime build that differs from the base pins.

## 5. Assess Impact On TRTMC

- **Placement and blast radius:** a root dependency (`pyproject.toml`,
  `requirements/`) reaches every user and family. A
  family-only dependency belongs in `families/<family>/requirements.txt`, with
  no project extra, shared lock, or include of another family's file. Neither
  `support.py`, discovery, core, nor a sibling family may import it.
- **Compatibility with base pins:** check `tensorrt`, `transformers`,
  `cuda-python`, `numpy`, and `torch` against the dry-run resolution. Flag any
  upgrade, downgrade, or second build of TensorRT, CUDA, or torch.
- **Platforms:** require x86_64 and aarch64 wheels or libraries (Grace-based
  GB300 and Thor are aarch64), CUDA 13 compatibility, and a compatible glibc
  and manylinux tag; note any required container changes.
- **Supply chain:** exact or bounded version pins, the official index rather
  than ad hoc `--extra-index-url` or Git URLs, no source builds that run code
  at install time, active maintenance, pinned revisions for
  `trust_remote_code` or `torch.hub` code, and provenance for every binary.
- **Optional dependency behavior:** when the dependency is missing, the owning
  family fails with a clear error instead of silently falling back.
- **Attribution:** `CONTRIBUTING.md` requires the PR to state each item's
  source, version, and license. Vendored files keep their original headers
  rather than an NVIDIA SPDX header. Community CI source-quality checks reject
  changes to the root `LICENSE` and `NOTICE`, so a contribution records
  redistributed material in the owning family's `NOTICE` or in
  `ASSET_LICENSES.md`, and asks a maintainer when the root `NOTICE` must
  change. An Apache-2.0 dependency's own `NOTICE` text travels with it.

## 6. Report The Result

Classify the overall result, then carry it into the review verdict:

| Result | Meaning | Effect on the review |
| --- | --- | --- |
| `NONE` | no new or changed third-party material | no finding; list the surfaces checked |
| `CLEAR` | permissive or properly isolated, correctly placed, attribution complete | no finding; summarize the inventory |
| `OBLIGATIONS` | acceptable license, but attribution, placement, pins, platform coverage, or PR disclosure is incomplete | `High` finding when redistributed material lacks attribution or source/version/license disclosure; otherwise `Medium` |
| `HUMAN DECISION` | restricted or share-alike terms, unresolved but plausible license, separate-process copyleft, or shipped weak copyleft | verdict `HUMAN REVIEW REQUIRED`; name the exact question for the project's legal or open-source review |
| `BLOCK` | copyleft that propagates per section 4, an unknown license on shipped material, or redistributed TensorRT/CUDA binaries | `Blocking` finding; verdict `BLOCK` |

When new material exists, report a table with `Dependency`,
`Version / source`, `License (SPDX)`, `Integration mode`, `Reaches`,
`Propagation`, and `Action`. When nothing is new, state that in one line
together with the surfaces inspected.

In contributor self-review, resolve `OBLIGATIONS` before marking the PR
ready, and describe any `HUMAN DECISION` item in the PR's third-party
provenance notes so a maintainer can route it before review.

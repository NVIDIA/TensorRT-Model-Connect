---
title: Family Runtime DSOs
---

The pre-#1093 runtime-plugin registry no longer exists. The stable load unit is
one family DSO named `libtrtmc_model_<family>.so`.

The bundle header names exactly one family and backend. Each DSO publishes the
model-agnostic `trtmc_plugin_descriptor_v1` interface with its product-build
identity, kind, and ID. The loader opens the exact backend and family paths
from one selected plugin root, validates both descriptors, resolves
`trtmc_create_family`, and receives an implementation of an abstract Task
interface.

A family owns its descriptor declaration, factory, pipeline, preprocessing,
postprocessing, dispatch, bindings, state, samplers, and any genuinely
model-specific TensorRT plugin. Source ownership remains family-local, while
the resulting DSO is released as part of one coordinated product build.

Descriptor v1 is an exact metadata layout, not ABI negotiation. The factory,
context, backend, module, and Task seams contain C++ interfaces, so plugins from
another product build are rejected even when they expose the same descriptor
version. Core and runtime also expose and compare their product-build identities
before the runtime calls a core C++ interface.

The native CLI also embeds that identity and compares it with the Runtime and
Core C symbols before calling bundle or loader C++ interfaces. This check is
application assembly, not family selection, so it remains outside the generic
Runtime Loader.

Plugin roots are trusted native-code inputs. ELF constructors run during
`dlopen`, before the descriptor can be called. The loader therefore never opens
discovery candidates speculatively, never falls back after a selected load
fails, and keeps opened DSOs resident so TensorRT registrars cannot outlive
their defining code.

There is no runtime-strategy map, central manifest, sibling fallback, or hot
plugin marketplace. To extend an existing family, edit only its
`families/<family>/runtime/` implementation and family-owned tests. To add a
new family, follow [Add a Model Family](../extend/add-model-family.md).

Every family factory declares
`TRTMC_DEFINE_FAMILY_PLUGIN_V1("<family>")`; every backend declares
`TRTMC_DEFINE_BACKEND_PLUGIN_V1("<backend>")`. Runtime extensions use the
generic descriptor macro with `PluginKind::kRuntimeExtension`. These macros
only publish the fixed descriptor symbol and build identity. They are required
because calling a factory first would cross an unvalidated C++ interface.
